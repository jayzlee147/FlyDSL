# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device-driven gfx950 grouped TN contractions for SonicMoE backward.

The backward sorter lays every non-empty expert out as one or more contiguous
64-row blocks.  This kernel consumes that layout directly and computes

``output[expert, M, N] = lhs[expert, rows, M].T @ rhs[expert, rows, N]``.

A small device builder emits one ``(expert, first_sorted_row)`` descriptor per
active expert.  Its public queue ABI is ``[count, expert0, row0, ...]`` so dW2
and dW1 can consume the same queue without rebuilding it.  Decode can bypass
the builder and consume the sorter's one-block-per-expert metadata directly.
The GEMM grid is persistent and walks output tiles from either schedule, so no
output tile performs a lower-bound search and no expert metadata is copied to
the host.  Inputs and output are BF16; each tile accumulates all of an expert's
route rows in FP32 before one BF16 materialization.
"""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as _raw
from kernels.common import buffer_ops
from kernels.common.mem_ops import atomic_add
from kernels.common.tensor_shim import _run_compiled
from kernels.gemm.gemm_a16w16_gfx950 import async_load_to_lds
from kernels.gemm.gemm_a16w16_gfx950_utils import (
    GFX950_DMA_BYTES,
    GFX950_WAVE_SIZE,
    __barrier,
    get_wave_lds_offset,
    make_transposed_lds_layout,
)


_BLOCK_THREADS = 256
_SORTED_BLOCK_M = 64
_NUM_CU = 256
_PERSIST_THRESHOLD = _NUM_CU * 4
_MAX_SIGNED_I32 = (1 << 31) - 1
_MAX_BUFFER_BYTES = (1 << 32) - 1
_ZERO_VECTOR_ELEMENTS = GFX950_DMA_BYTES // 2
_ZERO_BLOCK_THREADS = 1024


def _global_bf16_ptr(address):
    pointer_type = fx.PointerType.get(
        fx.BFloat16.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=GFX950_DMA_BYTES,
    )
    return fx.inttoptr(pointer_type, fx.Int64(address))


def active_expert_descriptor_capacity(routes: int, num_experts: int) -> int:
    """Return a tight host-known upper bound on active expert descriptors."""

    for name, value in (("routes", routes), ("num_experts", num_experts)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if routes < 0:
        raise ValueError(f"routes must be non-negative, got {routes}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if routes > _MAX_SIGNED_I32:
        raise ValueError(f"routes exceeds signed int32 metadata capacity: {routes}")
    return min(routes, num_experts)


def active_expert_queue_elements(routes: int, num_experts: int) -> int:
    """Return the int32 storage size for ``[count, (expert, row) * capacity]``."""

    return 1 + 2 * active_expert_descriptor_capacity(routes, num_experts)


@functools.lru_cache(maxsize=128)
def compile_active_expert_queue(
    num_experts: int,
    max_metadata_blocks: int,
    device_index: int,
):
    """Compile the standalone producer for the shared active-expert queue."""

    del device_index
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if max_metadata_blocks <= 0:
        raise ValueError("max_metadata_blocks must be positive")
    if max_metadata_blocks > _MAX_SIGNED_I32 // _SORTED_BLOCK_M:
        raise ValueError("sorter metadata exceeds signed int32 row capacity")
    lower_bound_steps = max(1, max_metadata_blocks.bit_length())

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def clear_queue_count(queue_storage: fx.Tensor):
        if gpu.thread_idx.x == fx.Int32(0):
            storage_rsrc = buffer_ops.create_buffer_resource(queue_storage, max_size=True)
            buffer_ops.buffer_store(fx.Int32(0), storage_rsrc, fx.Int32(0))

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def build_queue(
        expert_frequency: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        queue_storage: fx.Tensor,
        i32_queue_capacity: fx.Int32,
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
                metadata_blocks = padded_rows // fx.Int32(_SORTED_BLOCK_M)
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
                    slot = atomic_add(
                        queue_storage,
                        fx.Int32(0),
                        fx.Int32(1),
                        dtype_bytes=4,
                    )
                    queue_index = fx.Int32(slot)
                    if queue_index < i32_queue_capacity:
                        storage_rsrc = buffer_ops.create_buffer_resource(queue_storage, max_size=True)
                        output_index = fx.Int32(1) + queue_index * fx.Int32(2)
                        buffer_ops.buffer_store(expert, storage_rsrc, output_index)
                        buffer_ops.buffer_store(
                            lo * fx.Int32(_SORTED_BLOCK_M),
                            storage_rsrc,
                            output_index + fx.Int32(1),
                        )

    @flyc.jit
    def launch(
        expert_frequency: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        queue_storage: fx.Tensor,
        i32_queue_capacity: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        clear_queue_count(queue_storage).launch(
            grid=(1, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )
        build_queue(
            expert_frequency,
            sorted_expert_ids,
            num_valid_ids,
            queue_storage,
            i32_queue_capacity,
        ).launch(
            grid=((num_experts + _BLOCK_THREADS - 1) // _BLOCK_THREADS, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


def grouped_tn_tuning(output_m: int, output_n: int) -> tuple[int, int, int, int, int, int]:
    """Choose ``(BM, BN, BK, K-pad, M-waves, N-waves)`` for grouped TN."""

    if output_m % 64 or output_n % 64:
        raise ValueError("grouped TN requires output dimensions divisible by 64")
    block_m = next(tile for tile in (256, 128, 64) if output_m % tile == 0)
    block_n = next(tile for tile in (256, 128, 64) if output_n % tile == 0)
    m_waves = 4 if block_m == 256 else 2
    n_waves = 4 if block_n == 256 else 2
    return block_m, block_n, 32, 0, m_waves, n_waves


def grouped_dw2_tuning(hidden_size: int, intermediate_size: int) -> tuple[int, int, int, int, int, int]:
    """Semantic alias for dW2 callers of the reusable grouped TN policy."""

    return grouped_tn_tuning(hidden_size, intermediate_size)


@functools.lru_cache(maxsize=64)
def compile_inactive_weight_grad_zero(
    dw1_expert_elements: int,
    dw2_expert_elements: int,
    num_experts: int,
    device_index: int,
):
    """Compile an expert-local zero fill for grouped BF16 weight gradients.

    The dense dW1 allocation for the production SonicMoE shape is larger than
    4 GiB.  Address each expert slab with an i64 base and give the buffer
    instruction an expert-local resource, keeping every vector offset within
    the hardware's 32-bit BRSRC window.
    """

    del device_index
    if min(dw1_expert_elements, dw2_expert_elements, num_experts) <= 0:
        raise ValueError("weight-gradient slab sizes and num_experts must be positive")
    if dw1_expert_elements % _ZERO_VECTOR_ELEMENTS:
        raise ValueError("dW1 expert slabs must have 128-bit size alignment")
    if dw2_expert_elements % _ZERO_VECTOR_ELEMENTS:
        raise ValueError("dW2 expert slabs must have 128-bit size alignment")
    dw1_expert_bytes = dw1_expert_elements * 2
    dw2_expert_bytes = dw2_expert_elements * 2
    if max(dw1_expert_bytes, dw2_expert_bytes) > _MAX_BUFFER_BYTES:
        raise ValueError("each expert-local weight-gradient slab must fit one BRSRC")
    dw1_vectors = dw1_expert_elements // _ZERO_VECTOR_ELEMENTS
    dw2_vectors = dw2_expert_elements // _ZERO_VECTOR_ELEMENTS

    @flyc.kernel(
        name=(
            f"sonic_zero_inactive_weight_grads_e{num_experts}"
            f"_v{dw1_vectors}x{dw2_vectors}"
        ),
        known_block_size=[_ZERO_BLOCK_THREADS, 1, 1],
    )
    def zero_inactive_weight_grads_kernel(
        expert_frequency: fx.Tensor,
        dw1_base: fx.Int64,
        dw2_base: fx.Int64,
    ):
        expert = fx.Int32(gpu.block_idx.x)
        tid = fx.Int32(gpu.thread_idx.x)
        frequency_rsrc = buffer_ops.create_buffer_resource(expert_frequency, max_size=True)
        frequency = rocdl.readfirstlane(
            T.i32,
            _raw(buffer_ops.buffer_load(frequency_rsrc, expert, vec_width=1, dtype=T.i32)),
        )
        if frequency == fx.Int32(0):
            # Do not form a descriptor over the full dense tensor: dW1 can be
            # larger than 4 GiB, while the AMD buffer offset is only 32 bits.
            # The i64 expert base plus exact expert-local resource keeps both
            # the address and the OOB bound correct for the last expert.
            dw1_addr = dw1_base + fx.Int64(expert) * fx.Int64(dw1_expert_bytes)
            dw2_addr = dw2_base + fx.Int64(expert) * fx.Int64(dw2_expert_bytes)
            dw1_rsrc = buffer_ops.create_buffer_resource_from_addr(
                _raw(dw1_addr),
                num_records_bytes=dw1_expert_bytes,
            )
            dw2_rsrc = buffer_ops.create_buffer_resource_from_addr(
                _raw(dw2_addr),
                num_records_bytes=dw2_expert_bytes,
            )
            # Store raw zero dwords so lowering selects one 128-bit buffer
            # store rather than a typed packed-BF16 sequence.
            zero = fx.Vector.filled(GFX950_DMA_BYTES // 4, 0, fx.Int32)
            for vector_index in range(
                tid,
                fx.Int32(dw1_vectors),
                fx.Int32(_ZERO_BLOCK_THREADS),
            ):
                buffer_ops.buffer_store(
                    zero,
                    dw1_rsrc,
                    vector_index * fx.Int32(GFX950_DMA_BYTES // 4),
                )
            for vector_index in range(
                tid,
                fx.Int32(dw2_vectors),
                fx.Int32(_ZERO_BLOCK_THREADS),
            ):
                buffer_ops.buffer_store(
                    zero,
                    dw2_rsrc,
                    vector_index * fx.Int32(GFX950_DMA_BYTES // 4),
                )

    @flyc.jit
    def launch(
        expert_frequency: fx.Tensor,
        dw1_base: fx.Int64,
        dw2_base: fx.Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        zero_inactive_weight_grads_kernel(
            expert_frequency,
            dw1_base,
            dw2_base,
        ).launch(
            grid=(num_experts, 1, 1),
            block=(_ZERO_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


def zero_inactive_weight_grads_flydsl(
    expert_frequency: torch.Tensor,
    dw1: torch.Tensor,
    dw2: torch.Tensor,
    *,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero only inactive expert slabs before grouped dW1/dW2 overwrite actives."""

    if expert_frequency.ndim != 1:
        raise ValueError("expert_frequency must have shape [E]")
    num_experts = int(expert_frequency.numel())
    if num_experts <= 0:
        raise ValueError("expert_frequency must contain at least one expert")
    if dw1.ndim != 3 or dw2.ndim != 3:
        raise ValueError("dw1 and dw2 must be dense three-dimensional expert weights")
    if int(dw1.shape[0]) != num_experts or int(dw2.shape[0]) != num_experts:
        raise ValueError("weight gradients and expert_frequency must have the same E")
    tensors = (expert_frequency, dw1, dw2)
    if any(tensor.device != expert_frequency.device for tensor in tensors):
        raise ValueError("inactive-gradient zero tensors must share one device")
    if expert_frequency.dtype != torch.int32:
        raise TypeError("expert_frequency must use int32")
    if dw1.dtype != torch.bfloat16 or dw2.dtype != torch.bfloat16:
        raise TypeError("inactive-gradient zero currently requires BF16 outputs")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("inactive-gradient zero tensors must be contiguous")
    dw1_expert_elements = int(dw1.numel()) // num_experts
    dw2_expert_elements = int(dw2.numel()) // num_experts
    if dw1_expert_elements % _ZERO_VECTOR_ELEMENTS:
        raise ValueError("dW1 expert slabs must have 128-bit size alignment")
    if dw2_expert_elements % _ZERO_VECTOR_ELEMENTS:
        raise ValueError("dW2 expert slabs must have 128-bit size alignment")
    if max(dw1_expert_elements, dw2_expert_elements) * 2 > _MAX_BUFFER_BYTES:
        raise ValueError("each expert-local weight-gradient slab must fit one BRSRC")
    if stream is None:
        stream = torch.cuda.current_stream(expert_frequency.device)
    launcher = compile_inactive_weight_grad_zero(
        dw1_expert_elements,
        dw2_expert_elements,
        num_experts,
        expert_frequency.device.index or 0,
    )
    _run_compiled(
        launcher,
        expert_frequency,
        dw1.data_ptr(),
        dw2.data_ptr(),
        stream,
    )
    expert_frequency.record_stream(stream)
    dw1.record_stream(stream)
    dw2.record_stream(stream)
    return dw1, dw2


@functools.lru_cache(maxsize=128)
def compile_grouped_tn(
    output_m: int,
    output_n: int,
    num_experts: int,
    block_m: int,
    block_n: int,
    block_k: int,
    k_padding: int,
    m_waves: int,
    n_waves: int,
    device_index: int,
    metadata_direct: bool = False,
):
    """Compile the persistent grouped TN consumer for a prebuilt queue.

    ``block_k`` and ``k_padding`` are separate constexprs so consumers can
    tune the MFMA reduction tile independently from the number of materialized
    sorter rows.  ``k_padding=0`` means the logical reduction bound is the
    actual expert frequency; physical loads still round to ``block_k`` and are
    safe because sorter padding is zero-filled.  ``metadata_direct`` treats
    the schedule tensor as sorter expert IDs and is valid when every active
    expert occupies one 64-row sorter block (the fixed-K T1 fast path).
    """

    del device_index
    stages = 2
    mma_m = 16
    mma_n = 16
    mma_k = 32
    in_data_bytes = 2
    async_load_bytes = GFX950_DMA_BYTES
    async_load_vec_size = async_load_bytes // in_data_bytes
    block_threads = m_waves * n_waves * GFX950_WAVE_SIZE
    ldg_x_threads = block_k // async_load_vec_size
    ldg_a_iters = (block_m * block_k) // (block_threads * async_load_vec_size)
    ldg_b_iters = (block_n * block_k) // (block_threads * async_load_vec_size)
    mma_m_iters = block_m // (m_waves * mma_m)
    mma_n_iters = block_n // (n_waves * mma_n)
    k_mma_iters = block_k // mma_k
    cshuffle_vec_size = async_load_vec_size
    num_m_tiles = output_m // block_m
    num_n_tiles = output_n // block_n
    output_tiles_per_expert = num_m_tiles * num_n_tiles
    if min(output_m, output_n, num_experts, block_m, block_n, block_k, m_waves, n_waves) <= 0:
        raise ValueError("grouped TN dimensions and tuning values must be positive")
    if block_k not in (32, 64):
        raise ValueError("grouped TN block_k must be 32 or 64")
    if k_padding not in (0, 32, 64):
        raise ValueError("grouped TN k_padding must be 0, 32, or 64")
    if k_padding and k_padding % block_k:
        raise ValueError("grouped TN k_padding must be divisible by block_k")
    if output_m % block_m or output_n % block_n:
        raise ValueError("grouped TN output tiles must divide output dimensions exactly")
    if block_m % (m_waves * mma_m) or block_n % (n_waves * mma_n):
        raise ValueError("grouped TN tile sizes must divide their wave tiling")
    if block_threads > 1024:
        raise ValueError("grouped TN cannot exceed 1024 workgroup threads")
    if block_k % async_load_vec_size:
        raise ValueError("grouped TN block_k must satisfy 128-bit DMA alignment")
    if block_threads % ldg_x_threads:
        raise ValueError("grouped TN thread layout must divide block_k DMA lanes")
    if ldg_a_iters * block_threads * async_load_vec_size != block_m * block_k:
        raise ValueError("grouped TN A tile must have exact whole-workgroup DMA coverage")
    if ldg_b_iters * block_threads * async_load_vec_size != block_n * block_k:
        raise ValueError("grouped TN B tile must have exact whole-workgroup DMA coverage")
    lds_ab_bytes = stages * (block_m + block_n) * block_k * in_data_bytes
    lds_c_bytes = block_m * block_n * in_data_bytes
    if max(lds_ab_bytes, lds_c_bytes) > 163840:
        raise ValueError("grouped TN tuning exceeds gfx950 LDS capacity")

    @fx.struct
    class SharedABStorage:
        a: fx.Array[fx.BFloat16, stages * block_m * block_k, 16]
        b: fx.Array[fx.BFloat16, stages * block_n * block_k, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[fx.BFloat16, block_m * block_n, 16]

    @flyc.kernel(
        name=(
            f"sonic_grouped_tn_bf16_m{output_m}_n{output_n}_e{num_experts}"
            f"_bm{block_m}_bn{block_n}_bk{block_k}_kp{k_padding}_w{m_waves}x{n_waves}"
            f"_md{int(metadata_direct)}"
        ),
        known_block_size=[block_threads, 1, 1],
    )
    def grouped_tn_kernel(
        lhs_rows: fx.Tensor,
        rhs_rows: fx.Tensor,
        expert_frequency: fx.Tensor,
        schedule_storage: fx.Tensor,
        num_valid_ids: fx.Tensor,
        output: fx.Tensor,
        tiled_mma: fx.TiledMma,
    ):
        tid = gpu.thread_idx.x
        bid = gpu.block_idx.x
        grid_size = gpu.grid_dim.x
        storage_rsrc = buffer_ops.create_buffer_resource(schedule_storage, max_size=True)
        if const_expr(metadata_direct):
            valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
            padded_rows = rocdl.readfirstlane(
                T.i32,
                _raw(buffer_ops.buffer_load(valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32)),
            )
            descriptor_count = padded_rows // fx.Int32(_SORTED_BLOCK_M)
        else:
            descriptor_count = rocdl.readfirstlane(
                T.i32,
                _raw(
                    buffer_ops.buffer_load(
                        storage_rsrc,
                        fx.Int32(0),
                        vec_width=1,
                        dtype=T.i32,
                    )
                ),
            )
        work_bound = descriptor_count * fx.Int32(output_tiles_per_expert)

        storage = fx.SharedAllocator().allocate(SharedStorage)
        smem_a = storage.ab.a.peek().ptr
        smem_b = storage.ab.b.peek().ptr
        smem_c = storage.c.peek().ptr
        lhs_base_addr = fx.Int64(fx.ptrtoint(fx.get_iter(lhs_rows)))
        rhs_base_addr = fx.Int64(fx.ptrtoint(fx.get_iter(rhs_rows)))
        output_base_addr = fx.Int64(fx.ptrtoint(fx.get_iter(output)))
        frequency_rsrc = buffer_ops.create_buffer_resource(expert_frequency, max_size=True)

        a_read_atom = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans16_64b(), fx.BFloat16)
        b_read_atom = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans16_64b(), fx.BFloat16)
        a_lds_layout = make_transposed_lds_layout(block_m, block_k)
        b_lds_layout = make_transposed_lds_layout(block_n, block_k)
        c_lds_layout = fx.make_layout((block_m, block_n), (block_n, 1))
        thr_mma = tiled_mma.thr_slice(tid)
        thr_copy_a = fx.make_tiled_copy_A(a_read_atom, tiled_mma).get_slice(tid)
        thr_copy_b = fx.make_tiled_copy_B(b_read_atom, tiled_mma).get_slice(tid)
        row_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (1, 0)))
        col_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1)))
        thr_mma_crow = thr_mma.partition_C(row_coords)
        thr_mma_ccol = thr_mma.partition_C(col_coords)
        wave_offset = get_wave_lds_offset(tid, async_load_bytes)

        def run_output_tile(work_index):
            descriptor_index = work_index // fx.Int32(output_tiles_per_expert)
            output_tile = work_index % fx.Int32(output_tiles_per_expert)
            if const_expr(metadata_direct):
                expert = rocdl.readfirstlane(
                    T.i32,
                    _raw(
                        buffer_ops.buffer_load(
                            storage_rsrc,
                            descriptor_index,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    ),
                )
                first_sorted_row = descriptor_index * fx.Int32(_SORTED_BLOCK_M)
            else:
                descriptor_offset = fx.Int32(1) + descriptor_index * fx.Int32(2)
                expert = rocdl.readfirstlane(
                    T.i32,
                    _raw(
                        buffer_ops.buffer_load(
                            storage_rsrc,
                            descriptor_offset,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    ),
                )
                first_sorted_row = rocdl.readfirstlane(
                    T.i32,
                    _raw(
                        buffer_ops.buffer_load(
                            storage_rsrc,
                            descriptor_offset + fx.Int32(1),
                            vec_width=1,
                            dtype=T.i32,
                        )
                    ),
                )
            frequency = rocdl.readfirstlane(
                T.i32,
                _raw(buffer_ops.buffer_load(frequency_rsrc, expert, vec_width=1, dtype=T.i32)),
            )
            padding_quantum = block_k if k_padding == 0 else k_padding
            padded_k = (
                (frequency + fx.Int32(padding_quantum - 1))
                // fx.Int32(padding_quantum)
            ) * fx.Int32(padding_quantum)
            k_tiles = padded_k // fx.Int32(block_k)
            resource_rows = frequency if k_padding == 0 else padded_k
            lhs_addr = lhs_base_addr + fx.Int64(first_sorted_row) * fx.Int64(output_m * in_data_bytes)
            rhs_addr = rhs_base_addr + fx.Int64(first_sorted_row) * fx.Int64(
                output_n * in_data_bytes
            )
            lhs_rsrc = buffer_ops.create_buffer_resource_from_addr(
                _raw(lhs_addr),
                num_records_bytes=_raw(
                    fx.Int64(resource_rows) * fx.Int64(output_m * in_data_bytes)
                ),
            )
            rhs_rsrc = buffer_ops.create_buffer_resource_from_addr(
                _raw(rhs_addr),
                num_records_bytes=_raw(
                    fx.Int64(resource_rows) * fx.Int64(output_n * in_data_bytes)
                ),
            )
            block_m_index = output_tile // fx.Int32(num_n_tiles)
            block_n_index = output_tile % fx.Int32(num_n_tiles)
            block_m_offset = block_m_index * fx.Int32(block_m)
            block_n_offset = block_n_index * fx.Int32(block_n)

            output_addr = output_base_addr + fx.Int64(expert) * fx.Int64(
                output_m * output_n * in_data_bytes
            )
            output_rsrc = buffer_ops.create_buffer_resource_from_addr(
                _raw(output_addr),
                num_records_bytes=output_m * output_n * in_data_bytes,
            )
            expert_out = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    _global_bf16_ptr(output_addr),
                    fx.make_layout((output_m, output_n), (output_n, 1)),
                ),
                max_size=False,
            )
            g_c = fx.flat_divide(expert_out, (block_m, block_n))[
                None,
                None,
                block_m_index,
                block_n_index,
            ]
            s_a = fx.make_view(smem_a, a_lds_layout)
            s_b = fx.make_view(smem_b, b_lds_layout)
            s_c = fx.make_view(smem_c, c_lds_layout)
            frag_a = thr_mma.make_fragment_A(s_a)
            frag_b = thr_mma.make_fragment_B(s_b)
            frag_c = thr_mma.make_fragment_C(g_c)
            frag_a_retile = thr_copy_a.retile(frag_a)
            frag_b_retile = thr_copy_b.retile(frag_b)
            frag_c.fill(0.0)

            async_context = (
                wave_offset,
                tid,
                block_threads,
                async_load_vec_size,
                ldg_x_threads,
                0,
                block_k,
                in_data_bytes,
                async_load_bytes,
            )

            def load_a(k_tile, stage):
                async_load_to_lds(
                    smem_a + stage * block_m * block_k,
                    lhs_rsrc,
                    a_lds_layout,
                    block_m,
                    output_m,
                    block_m_offset,
                    output_m,
                    ldg_a_iters,
                    True,
                    k_tile,
                    async_context,
                )

            def load_b(k_tile, stage):
                async_load_to_lds(
                    smem_b + stage * block_n * block_k,
                    rhs_rsrc,
                    b_lds_layout,
                    block_n,
                    output_n,
                    block_n_offset,
                    output_n,
                    ldg_b_iters,
                    True,
                    k_tile,
                    async_context,
                )

            def compute_stage(read_stage):
                stage_a = fx.make_view(
                    smem_a + read_stage * block_m * block_k,
                    a_lds_layout,
                )
                stage_b = fx.make_view(
                    smem_b + read_stage * block_n * block_k,
                    b_lds_layout,
                )
                copy_a = thr_copy_a.partition_S(stage_a)
                copy_b = thr_copy_b.partition_S(stage_b)
                for k_iter in range_constexpr(k_mma_iters):
                    fx.copy(
                        b_read_atom,
                        copy_b[None, None, k_iter],
                        frag_b_retile[None, None, k_iter],
                    )
                    fx.copy(
                        a_read_atom,
                        copy_a[None, None, k_iter],
                        frag_a_retile[None, None, k_iter],
                    )
                    fx.gemm(
                        tiled_mma,
                        frag_c,
                        frag_a[None, None, k_iter],
                        frag_b[None, None, k_iter],
                        frag_c,
                        traversal_order=fx.GemmTraversalOrder.KNM,
                    )

            load_b(fx.Int32(0), 0)
            load_a(fx.Int32(0), 0)
            rocdl.sched_barrier(0)
            main_loop_end = k_tiles - fx.Int32(1)
            for k_tile in range(fx.Int32(0), main_loop_end, fx.Int32(1)):
                current_stage = k_tile % fx.Int32(stages)
                write_stage = (current_stage + fx.Int32(stages - 1)) % fx.Int32(stages)
                __barrier(0)
                load_b(k_tile + fx.Int32(1), write_stage)
                load_a(k_tile + fx.Int32(1), write_stage)
                compute_stage(current_stage)
                rocdl.sched_vmem(ldg_a_iters + ldg_b_iters)
                for _ in range_constexpr(k_mma_iters):
                    rocdl.sched_dsrd(mma_n_iters)
                    rocdl.sched_dsrd(mma_m_iters)
                    for _ in range_constexpr(mma_m_iters):
                        rocdl.sched_mfma(mma_n_iters)
                rocdl.sched_barrier(0)

            __barrier(0)
            compute_stage(main_loop_end % fx.Int32(stages))

            frag_c_out = fx.make_fragment_like(frag_c, fx.BFloat16)
            frag_c_out.store(frag_c.load().to(fx.BFloat16))
            gpu.barrier()
            for value_index in range_constexpr(fx.size(frag_c_out.shape).unpack()):
                row = fx.get_scalar(thr_mma_crow[value_index])
                column = fx.get_scalar(thr_mma_ccol[value_index])
                s_c[row, column] = frag_c_out[value_index]
            gpu.barrier()

            vectors_per_row = block_n // cshuffle_vec_size
            tile_vectors = block_m * vectors_per_row
            store_iters = (tile_vectors + block_threads - 1) // block_threads
            for store_iter in range_constexpr(store_iters):
                vector_index = fx.Int32(block_threads * store_iter) + tid
                if vector_index < fx.Int32(tile_vectors):
                    local_row = vector_index // fx.Int32(vectors_per_row)
                    local_col = (vector_index % fx.Int32(vectors_per_row)) * fx.Int32(cshuffle_vec_size)
                    global_row = block_m_offset + local_row
                    global_col = block_n_offset + local_col
                    value = fx.ptr_load(
                        smem_c + local_row * fx.Int32(block_n) + local_col,
                        result_type=fx.Vector.make_type(cshuffle_vec_size, fx.BFloat16),
                    )
                    output_offset = global_row * fx.Int32(output_n) + global_col
                    buffer_ops.buffer_store(value, output_rsrc, output_offset)

        if bid < work_bound:
            run_output_tile(bid)
        for work_index in range(bid + grid_size, work_bound, grid_size):
            gpu.barrier()
            run_output_tile(fx.Int32(work_index))

    @flyc.jit
    def launch(
        lhs_rows: fx.Tensor,
        rhs_rows: fx.Tensor,
        expert_frequency: fx.Tensor,
        schedule_storage: fx.Tensor,
        num_valid_ids: fx.Tensor,
        output: fx.Tensor,
        i32_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(mma_m, mma_n, mma_k, fx.BFloat16))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((m_waves, n_waves, 1), (n_waves, 1, 0)),
            fx.make_tile(
                None,
                None,
                fx.make_layout((mma_k // 4, 4), (1, mma_k // 4)),
            ),
        )
        grouped_tn_kernel(
            lhs_rows,
            rhs_rows,
            expert_frequency,
            schedule_storage,
            num_valid_ids,
            output,
            tiled_mma,
        ).launch(
            grid=(i32_grid, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


def build_active_expert_queue_flydsl(
    expert_frequency: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    *,
    routes: int,
    queue_storage: torch.Tensor | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Build the shared ``[count, (expert, first_row) * capacity]`` queue."""

    if expert_frequency.ndim != 1:
        raise ValueError("expert_frequency must have shape [E]")
    num_experts = int(expert_frequency.numel())
    capacity = active_expert_descriptor_capacity(routes, num_experts)
    required_elements = 1 + 2 * capacity
    metadata = (expert_frequency, sorted_expert_ids, num_valid_ids)
    if any(tensor.dtype != torch.int32 for tensor in metadata):
        raise TypeError("active-expert queue metadata must use int32")
    if any(tensor.device != expert_frequency.device for tensor in metadata):
        raise ValueError("active-expert queue metadata must share one device")
    if not all(tensor.is_contiguous() for tensor in metadata):
        raise ValueError("active-expert queue metadata must be contiguous")
    if sorted_expert_ids.ndim != 1:
        raise ValueError("sorted_expert_ids must be one-dimensional")
    if num_valid_ids.ndim != 1 or num_valid_ids.numel() < 1:
        raise ValueError("num_valid_ids must contain the padded row count")
    if capacity and sorted_expert_ids.numel() < 1:
        raise ValueError("non-empty routes require sorter expert metadata")
    if queue_storage is None:
        queue_storage = torch.empty(
            required_elements,
            dtype=torch.int32,
            device=expert_frequency.device,
        )
    elif (
        queue_storage.dtype != torch.int32
        or queue_storage.device != expert_frequency.device
        or not queue_storage.is_contiguous()
        or queue_storage.ndim != 1
        or queue_storage.numel() < required_elements
    ):
        raise ValueError(
            "queue_storage must be contiguous int32 on the metadata device "
            f"with at least {required_elements} elements"
        )

    if stream is None:
        stream = torch.cuda.current_stream(expert_frequency.device)
    if capacity == 0:
        with torch.cuda.stream(stream):
            queue_storage[:1].zero_()
        queue_storage.record_stream(stream)
        return queue_storage

    launcher = compile_active_expert_queue(
        num_experts,
        int(sorted_expert_ids.numel()),
        expert_frequency.device.index or 0,
    )
    _run_compiled(
        launcher,
        expert_frequency,
        sorted_expert_ids,
        num_valid_ids,
        queue_storage,
        capacity,
        stream,
    )
    queue_storage.record_stream(stream)
    return queue_storage


def grouped_tn_from_queue_flydsl(
    lhs_rows: torch.Tensor,
    rhs_rows: torch.Tensor,
    expert_frequency: torch.Tensor,
    queue_storage: torch.Tensor,
    output: torch.Tensor,
    *,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    k_padding: int | None = None,
    m_waves: int | None = None,
    n_waves: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Consume a prebuilt active-expert queue for one grouped TN contraction.

    Both inputs use the same sorter-padded row dimension.  ``output`` must
    already be zero so empty experts retain exact-zero gradients.  Optional
    tuning arguments let dW1 and dW2 select independent output/K tiles.
    """

    if lhs_rows.ndim != 2 or rhs_rows.ndim != 2 or output.ndim != 3:
        raise ValueError("expected lhs_rows[P,M], rhs_rows[P,N], and output[E,M,N]")
    num_experts, output_m, output_n = (int(value) for value in output.shape)
    if tuple(lhs_rows.shape) != (int(rhs_rows.shape[0]), output_m):
        raise ValueError("lhs_rows and rhs_rows must share P and match output M")
    if int(rhs_rows.shape[1]) != output_n:
        raise ValueError("rhs_rows width must match output N")
    tensors = (lhs_rows, rhs_rows, expert_frequency, queue_storage, output)
    if any(tensor.device != lhs_rows.device for tensor in tensors):
        raise ValueError("grouped TN tensors must share one device")
    if (
        lhs_rows.dtype != torch.bfloat16
        or rhs_rows.dtype != torch.bfloat16
        or output.dtype != torch.bfloat16
    ):
        raise TypeError("grouped TN currently requires BF16 inputs and output")
    if expert_frequency.dtype != torch.int32 or queue_storage.dtype != torch.int32:
        raise TypeError("grouped TN metadata must use int32")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("grouped TN tensors must be contiguous")
    if tuple(expert_frequency.shape) != (num_experts,):
        raise ValueError("expert_frequency must have shape [E]")
    if queue_storage.ndim != 1 or queue_storage.numel() < 1 or (queue_storage.numel() - 1) % 2:
        raise ValueError("queue_storage must use [count, (expert, first_row) * capacity] ABI")

    capacity = (int(queue_storage.numel()) - 1) // 2
    if capacity == 0:
        return output

    default_bm, default_bn, default_bk, default_k_padding, default_m_waves, default_n_waves = grouped_tn_tuning(
        output_m,
        output_n,
    )
    block_m = default_bm if block_m is None else block_m
    block_n = default_bn if block_n is None else block_n
    block_k = default_bk if block_k is None else block_k
    k_padding = default_k_padding if k_padding is None else k_padding
    m_waves = default_m_waves if m_waves is None else m_waves
    n_waves = default_n_waves if n_waves is None else n_waves
    output_tiles = (output_m // block_m) * (output_n // block_n)
    max_work = capacity * output_tiles
    grid = min(max_work, _NUM_CU) if max_work > _PERSIST_THRESHOLD else max_work
    if stream is None:
        stream = torch.cuda.current_stream(lhs_rows.device)
    launcher = compile_grouped_tn(
        output_m,
        output_n,
        num_experts,
        block_m,
        block_n,
        block_k,
        k_padding,
        m_waves,
        n_waves,
        lhs_rows.device.index or 0,
        False,
    )
    _run_compiled(
        launcher,
        lhs_rows,
        rhs_rows,
        expert_frequency,
        queue_storage,
        queue_storage,
        output,
        grid,
        stream,
    )
    queue_storage.record_stream(stream)
    return output


def grouped_tn_from_metadata_flydsl(
    lhs_rows: torch.Tensor,
    rhs_rows: torch.Tensor,
    expert_frequency: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    k_padding: int | None = None,
    m_waves: int | None = None,
    n_waves: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Run grouped TN directly from one-block-per-expert sorter metadata.

    This is the builder-free fixed-K T1 path.  Callers must guarantee every
    non-empty expert has at most ``_SORTED_BLOCK_M`` rows; otherwise repeated
    expert IDs would race while writing the same output tile.
    """

    if lhs_rows.ndim != 2 or rhs_rows.ndim != 2 or output.ndim != 3:
        raise ValueError("expected lhs_rows[P,M], rhs_rows[P,N], and output[E,M,N]")
    num_experts, output_m, output_n = (int(value) for value in output.shape)
    if tuple(lhs_rows.shape) != (int(rhs_rows.shape[0]), output_m):
        raise ValueError("lhs_rows and rhs_rows must share P and match output M")
    if int(rhs_rows.shape[1]) != output_n:
        raise ValueError("rhs_rows width must match output N")
    tensors = (
        lhs_rows,
        rhs_rows,
        expert_frequency,
        sorted_expert_ids,
        num_valid_ids,
        output,
    )
    if any(tensor.device != lhs_rows.device for tensor in tensors):
        raise ValueError("grouped TN tensors must share one device")
    if (
        lhs_rows.dtype != torch.bfloat16
        or rhs_rows.dtype != torch.bfloat16
        or output.dtype != torch.bfloat16
    ):
        raise TypeError("grouped TN currently requires BF16 inputs and output")
    if any(
        tensor.dtype != torch.int32
        for tensor in (expert_frequency, sorted_expert_ids, num_valid_ids)
    ):
        raise TypeError("grouped TN metadata must use int32")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("grouped TN tensors must be contiguous")
    if tuple(expert_frequency.shape) != (num_experts,):
        raise ValueError("expert_frequency must have shape [E]")
    if sorted_expert_ids.ndim != 1:
        raise ValueError("sorted_expert_ids must be one-dimensional")
    if num_valid_ids.ndim != 1 or num_valid_ids.numel() < 1:
        raise ValueError("num_valid_ids must contain the padded row count")
    if sorted_expert_ids.numel() == 0:
        return output

    (
        default_bm,
        default_bn,
        default_bk,
        default_k_padding,
        default_m_waves,
        default_n_waves,
    ) = grouped_tn_tuning(output_m, output_n)
    block_m = default_bm if block_m is None else block_m
    block_n = default_bn if block_n is None else block_n
    block_k = default_bk if block_k is None else block_k
    k_padding = default_k_padding if k_padding is None else k_padding
    m_waves = default_m_waves if m_waves is None else m_waves
    n_waves = default_n_waves if n_waves is None else n_waves
    output_tiles = (output_m // block_m) * (output_n // block_n)
    max_work = int(sorted_expert_ids.numel()) * output_tiles
    grid = min(max_work, _NUM_CU) if max_work > _PERSIST_THRESHOLD else max_work
    if stream is None:
        stream = torch.cuda.current_stream(lhs_rows.device)
    launcher = compile_grouped_tn(
        output_m,
        output_n,
        num_experts,
        block_m,
        block_n,
        block_k,
        k_padding,
        m_waves,
        n_waves,
        lhs_rows.device.index or 0,
        True,
    )
    _run_compiled(
        launcher,
        lhs_rows,
        rhs_rows,
        expert_frequency,
        sorted_expert_ids,
        num_valid_ids,
        output,
        grid,
        stream,
    )
    sorted_expert_ids.record_stream(stream)
    num_valid_ids.record_stream(stream)
    return output


def grouped_tn_flydsl(
    lhs_rows: torch.Tensor,
    rhs_rows: torch.Tensor,
    expert_frequency: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    routes: int,
    descriptor_storage: torch.Tensor | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    k_padding: int | None = None,
    m_waves: int | None = None,
    n_waves: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Build a queue and compute one standalone grouped TN contraction."""

    if routes == 0:
        # Preserve the output identity and exact zeros without compiling or
        # launching metadata/GEMM kernels for an empty ragged batch.
        active_expert_descriptor_capacity(routes, int(output.shape[0]))
        return output
    queue_storage = build_active_expert_queue_flydsl(
        expert_frequency,
        sorted_expert_ids,
        num_valid_ids,
        routes=routes,
        queue_storage=descriptor_storage,
        stream=stream,
    )
    return grouped_tn_from_queue_flydsl(
        lhs_rows,
        rhs_rows,
        expert_frequency,
        queue_storage,
        output,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        k_padding=k_padding,
        m_waves=m_waves,
        n_waves=n_waves,
        stream=stream,
    )


def grouped_dw2_flydsl(
    dy: torch.Tensor,
    activation: torch.Tensor,
    expert_frequency: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    dw2: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Semantic dW2 wrapper around :func:`grouped_tn_flydsl`."""

    return grouped_tn_flydsl(
        dy,
        activation,
        expert_frequency,
        sorted_expert_ids,
        num_valid_ids,
        dw2,
        **kwargs,
    )


__all__ = [
    "active_expert_descriptor_capacity",
    "active_expert_queue_elements",
    "build_active_expert_queue_flydsl",
    "compile_active_expert_queue",
    "compile_grouped_tn",
    "compile_inactive_weight_grad_zero",
    "grouped_dw2_flydsl",
    "grouped_dw2_tuning",
    "grouped_tn_from_metadata_flydsl",
    "grouped_tn_from_queue_flydsl",
    "grouped_tn_flydsl",
    "grouped_tn_tuning",
    "zero_inactive_weight_grads_flydsl",
]
