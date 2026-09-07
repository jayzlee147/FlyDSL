# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device-side work compaction for grouped SonicMoE GEMMs.

The route sorters pad every non-empty expert segment to ``sorted_block_m``.
Launching one GEMM workgroup for every padded sub-tile wastes work for sparse
experts, while assigning one workgroup to an expert serializes long segments.
This module builds a compact list containing only the real ``block_m`` tiles:

``descriptor = global sorted-row offset // block_m``.

One device thread handles one expert.  It locates that expert's first sorted
metadata block, atomically reserves ``ceil(frequency / block_m)`` descriptor
slots, and writes the corresponding global M-block indices.  Descriptor order
is deliberately unspecified; grouped GEMMs must treat it as a work queue.
"""

from __future__ import annotations

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from kernels.common import buffer_ops
from kernels.common.mem_ops import atomic_add
from kernels.common.tensor_shim import _run_compiled


_BLOCK_THREADS = 256
_MAX_SIGNED_I32 = (1 << 31) - 1


def compact_m_tile_descriptor_upper_bound(
    routes: int,
    num_experts: int,
    block_m: int,
    *,
    max_expert_rows: int | None = None,
) -> int:
    """Return a host-known upper bound for ``sum_e ceil(freq[e] / block_m)``.

    ``max_expert_rows`` can tighten the bound when the routing contract limits
    each expert's count.  Fixed-K routing with unique experts per token may pass
    ``tokens``; ragged routing, which permits duplicate ``(token, expert)``
    edges, should leave it unset.
    """

    for name, value in (("routes", routes), ("num_experts", num_experts), ("block_m", block_m)):
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if routes < 0:
        raise ValueError(f"routes must be non-negative, got {routes}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    if routes > _MAX_SIGNED_I32:
        raise ValueError(f"routes exceeds signed int32 metadata capacity: {routes}")
    if max_expert_rows is not None:
        if not isinstance(max_expert_rows, int):
            raise TypeError(f"max_expert_rows must be an int or None, got {type(max_expert_rows).__name__}")
        if max_expert_rows < 0:
            raise ValueError(f"max_expert_rows must be non-negative, got {max_expert_rows}")
        if routes > num_experts * max_expert_rows:
            raise ValueError(
                f"routes={routes} cannot fit num_experts={num_experts} with "
                f"max_expert_rows={max_expert_rows}"
            )

    if routes == 0:
        return 0
    active_experts = min(num_experts, routes)
    # For A non-empty experts, sum ceil(c_e/BM) is at most
    # floor((sum(c_e) + A*(BM-1))/BM).
    bound = (routes + active_experts * (block_m - 1)) // block_m
    if max_expert_rows is not None:
        per_expert_tiles = (max_expert_rows + block_m - 1) // block_m
        bound = min(bound, active_experts * per_expert_tiles)
    if bound > _MAX_SIGNED_I32:
        raise ValueError(f"descriptor count exceeds signed int32: {bound}")
    return bound


def fixed_compact_m_tile_descriptor_upper_bound(
    tokens: int,
    num_experts: int,
    topk: int,
    block_m: int,
) -> int:
    """Upper bound for fixed-K routing with unique expert IDs per token."""

    for name, value in (("tokens", tokens), ("topk", topk)):
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    if topk > num_experts:
        raise ValueError(
            f"fixed-K unique routing requires topk <= num_experts, got topk={topk}, "
            f"num_experts={num_experts}"
        )
    return compact_m_tile_descriptor_upper_bound(
        tokens * topk,
        num_experts,
        block_m,
        max_expert_rows=tokens,
    )


def ragged_compact_m_tile_descriptor_upper_bound(
    routes: int,
    num_experts: int,
    block_m: int,
) -> int:
    """Upper bound for flat ragged routing, including duplicate edges."""

    return compact_m_tile_descriptor_upper_bound(routes, num_experts, block_m)


@functools.lru_cache(maxsize=128)
def compile_compact_m_tile_descriptor_builder(
    num_experts: int,
    block_m: int,
    sorted_block_m: int,
    device_index: int,
    emit_active_experts: bool = False,
):
    """Compile the counter-clear plus descriptor-build launch sequence.

    When ``emit_active_experts`` is true, the same expert scan also writes a
    shared ``[count, (expert, first_sorted_row) * capacity]`` queue.  This lets
    output-stationary grouped weight-gradient kernels reuse the W1/dX compact
    scheduler without paying for a second active-expert builder.
    """

    del device_index
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    if sorted_block_m <= 0:
        raise ValueError(f"sorted_block_m must be positive, got {sorted_block_m}")
    if sorted_block_m % block_m:
        raise ValueError(
            f"sorted_block_m ({sorted_block_m}) must be divisible by block_m ({block_m})"
        )

    subtiles_per_sorted_block = sorted_block_m // block_m
    max_metadata_blocks = _MAX_SIGNED_I32 // sorted_block_m
    lower_bound_steps = max(1, max_metadata_blocks.bit_length())

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def clear_counter_kernel(
        total_tiles: fx.Tensor,
        active_expert_storage: fx.Tensor,
    ):
        if gpu.thread_idx.x == fx.Int32(0):
            total_rsrc = buffer_ops.create_buffer_resource(total_tiles, max_size=True)
            buffer_ops.buffer_store(fx.Int32(0), total_rsrc, fx.Int32(0))
            if const_expr(emit_active_experts):
                active_rsrc = buffer_ops.create_buffer_resource(
                    active_expert_storage,
                    max_size=True,
                )
                buffer_ops.buffer_store(fx.Int32(0), active_rsrc, fx.Int32(0))

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def build_descriptor_kernel(
        expert_frequency: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        descriptors: fx.Tensor,
        total_tiles: fx.Tensor,
        i32_descriptor_capacity: fx.Int32,
        active_expert_storage: fx.Tensor,
        i32_active_expert_capacity: fx.Int32,
    ):
        expert = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        if expert < fx.Int32(num_experts):
            frequency_rsrc = buffer_ops.create_buffer_resource(expert_frequency, max_size=True)
            frequency = fx.Int32(
                buffer_ops.buffer_load(frequency_rsrc, expert, vec_width=1, dtype=T.i32)
            )
            if frequency > fx.Int32(0):
                valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
                padded_rows = fx.Int32(
                    buffer_ops.buffer_load(valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32)
                )
                metadata_blocks = padded_rows // fx.Int32(sorted_block_m)
                expert_ids_rsrc = buffer_ops.create_buffer_resource(sorted_expert_ids, max_size=True)

                lo = fx.Int32(0)
                hi = metadata_blocks
                for _ in range_constexpr(lower_bound_steps):
                    searching = lo < hi
                    mid = (lo + hi) // fx.Int32(2)
                    safe_mid = searching.select(mid, fx.Int32(0))
                    mid_expert = fx.Int32(
                        buffer_ops.buffer_load(expert_ids_rsrc, safe_mid, vec_width=1, dtype=T.i32)
                    )
                    move_right = searching & (mid_expert < expert)
                    lo = move_right.select(mid + fx.Int32(1), lo)
                    move_left = searching & (mid_expert >= expert)
                    hi = move_left.select(mid, hi)

                in_metadata = lo < metadata_blocks
                safe_lo = in_metadata.select(lo, fx.Int32(0))
                found_expert = fx.Int32(
                    buffer_ops.buffer_load(expert_ids_rsrc, safe_lo, vec_width=1, dtype=T.i32)
                )
                if in_metadata & (found_expert == expert):
                    if const_expr(emit_active_experts):
                        active_slot = atomic_add(
                            active_expert_storage,
                            fx.Int32(0),
                            fx.Int32(1),
                            dtype_bytes=4,
                        )
                        active_index = fx.Int32(active_slot)
                        if active_index < i32_active_expert_capacity:
                            active_rsrc = buffer_ops.create_buffer_resource(
                                active_expert_storage,
                                max_size=True,
                            )
                            active_offset = fx.Int32(1) + active_index * fx.Int32(2)
                            buffer_ops.buffer_store(expert, active_rsrc, active_offset)
                            buffer_ops.buffer_store(
                                lo * fx.Int32(sorted_block_m),
                                active_rsrc,
                                active_offset + fx.Int32(1),
                            )
                    tile_count = (frequency + fx.Int32(block_m - 1)) // fx.Int32(block_m)
                    reservation = atomic_add(
                        total_tiles,
                        fx.Int32(0),
                        tile_count,
                        dtype_bytes=4,
                    )
                    output_start = fx.Int32(reservation)
                    first_m_block = lo * fx.Int32(subtiles_per_sorted_block)
                    descriptors_rsrc = buffer_ops.create_buffer_resource(descriptors, max_size=True)
                    for local_tile in range(0, tile_count, 1):
                        output_index = output_start + fx.Int32(local_tile)
                        if output_index < i32_descriptor_capacity:
                            buffer_ops.buffer_store(
                                first_m_block + fx.Int32(local_tile),
                                descriptors_rsrc,
                                output_index,
                            )

    @flyc.jit
    def launch(
        expert_frequency: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        descriptors: fx.Tensor,
        total_tiles: fx.Tensor,
        i32_descriptor_capacity: fx.Int32,
        active_expert_storage: fx.Tensor,
        i32_active_expert_capacity: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        clear_counter_kernel(total_tiles, active_expert_storage).launch(
            grid=(1, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )
        build_descriptor_kernel(
            expert_frequency,
            sorted_expert_ids,
            num_valid_ids,
            descriptors,
            total_tiles,
            i32_descriptor_capacity,
            active_expert_storage,
            i32_active_expert_capacity,
        ).launch(
            grid=((num_experts + _BLOCK_THREADS - 1) // _BLOCK_THREADS, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


def build_compact_m_tile_descriptors(
    expert_frequency: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    descriptors: torch.Tensor,
    total_tiles: torch.Tensor,
    *,
    block_m: int,
    sorted_block_m: int,
    descriptor_capacity: int | None = None,
    active_expert_storage: torch.Tensor | None = None,
    active_expert_capacity: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clear counters and asynchronously build compact M descriptors.

    Supplying ``active_expert_storage`` additionally emits the counter-first
    active-expert queue used by grouped dW1/dW2, within the same clear/build
    kernel pair.  The optional storage ABI is
    ``[count, expert0, first_row0, expert1, first_row1, ...]``.
    """

    tensors = {
        "expert_frequency": expert_frequency,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "descriptors": descriptors,
        "total_tiles": total_tiles,
    }
    device = expert_frequency.device
    if device.type != "cuda":
        raise ValueError("compact M-tile descriptor construction requires a ROCm device")
    for name, tensor in tensors.items():
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if expert_frequency.ndim != 1 or expert_frequency.numel() <= 0:
        raise ValueError("expert_frequency must be a non-empty rank-1 tensor")
    if sorted_expert_ids.ndim != 1:
        raise ValueError("sorted_expert_ids must be rank 1")
    if num_valid_ids.ndim != 1 or num_valid_ids.numel() < 1:
        raise ValueError("num_valid_ids must contain at least one element")
    if descriptors.ndim != 1:
        raise ValueError("descriptors must be rank 1")
    if total_tiles.ndim != 1 or total_tiles.numel() < 1:
        raise ValueError("total_tiles must contain at least one element")
    if active_expert_storage is not None:
        if active_expert_storage.device != device:
            raise ValueError(
                f"active_expert_storage must be on {device}, got {active_expert_storage.device}"
            )
        if active_expert_storage.dtype != torch.int32:
            raise ValueError(
                "active_expert_storage must have dtype torch.int32, "
                f"got {active_expert_storage.dtype}"
            )
        if not active_expert_storage.is_contiguous():
            raise ValueError("active_expert_storage must be contiguous")
        if (
            active_expert_storage.ndim != 1
            or active_expert_storage.numel() < 1
            or (active_expert_storage.numel() - 1) % 2
        ):
            raise ValueError(
                "active_expert_storage must use the counter-first "
                "[count, (expert, first_row) * capacity] ABI"
            )
    elif active_expert_capacity is not None:
        raise ValueError("active_expert_capacity requires active_expert_storage")
    if block_m <= 0 or sorted_block_m <= 0 or sorted_block_m % block_m:
        raise ValueError(
            f"expected positive block_m dividing sorted_block_m, got block_m={block_m}, "
            f"sorted_block_m={sorted_block_m}"
        )

    capacity = descriptors.numel() if descriptor_capacity is None else descriptor_capacity
    if not isinstance(capacity, int):
        raise TypeError(f"descriptor_capacity must be an int or None, got {type(capacity).__name__}")
    if capacity < 0 or capacity > descriptors.numel() or capacity > _MAX_SIGNED_I32:
        raise ValueError(
            f"descriptor_capacity must be in [0, {min(descriptors.numel(), _MAX_SIGNED_I32)}], got {capacity}"
        )
    if active_expert_storage is None:
        active_capacity = 0
        active_storage_arg = total_tiles
    else:
        storage_capacity = (active_expert_storage.numel() - 1) // 2
        active_capacity = (
            storage_capacity
            if active_expert_capacity is None
            else active_expert_capacity
        )
        if not isinstance(active_capacity, int):
            raise TypeError(
                "active_expert_capacity must be an int or None, "
                f"got {type(active_capacity).__name__}"
            )
        if (
            active_capacity < 0
            or active_capacity > storage_capacity
            or active_capacity > _MAX_SIGNED_I32
        ):
            raise ValueError(
                "active_expert_capacity must be in "
                f"[0, {min(storage_capacity, _MAX_SIGNED_I32)}], got {active_capacity}"
            )
        active_storage_arg = active_expert_storage
    if stream is None:
        stream = torch.cuda.current_stream(device)
    device_index = device.index or 0
    launcher = compile_compact_m_tile_descriptor_builder(
        int(expert_frequency.numel()),
        block_m,
        sorted_block_m,
        device_index,
        active_expert_storage is not None,
    )
    _run_compiled(
        launcher,
        expert_frequency,
        sorted_expert_ids,
        num_valid_ids,
        descriptors,
        total_tiles,
        capacity,
        active_storage_arg,
        active_capacity,
        stream,
    )
    return descriptors, total_tiles


__all__ = [
    "build_compact_m_tile_descriptors",
    "compact_m_tile_descriptor_upper_bound",
    "compile_compact_m_tile_descriptor_builder",
    "fixed_compact_m_tile_descriptor_upper_bound",
    "ragged_compact_m_tile_descriptor_upper_bound",
]
