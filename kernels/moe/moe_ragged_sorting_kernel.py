# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Flat-route MoE sorting for variable-K/ragged routing.

The dense sorter in :mod:`kernels.moe.moe_sorting_kernel` consumes a
``[tokens, top_k]`` rectangle.  Token-choice rounding instead supplies a flat
edge list ``(token, expert, weight)`` where each token may have a different
number of routes.  This module converts that list to the same metadata contract
consumed by the Sonic grouped GEMMs:

* ``sorted_token_ids`` and ``sorted_weights`` are grouped by expert;
* every expert segment is padded to ``unit_size`` with ``token == tokens``;
* ``sorted_expert_ids`` contains one expert id per padded GEMM tile; and
* ``num_valid_ids == [total_padded_routes, tokens]``.

Four kernels are launched on one stream: clear, expert histogram, padded
prefix, and route scatter.  Atomic cursors make duplicate ``(token, expert)``
edges well-defined: every input edge occupies its own output slot.
"""

from __future__ import annotations

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from kernels.common import buffer_ops
from kernels.common.mem_ops import atomic_add


BLOCK_SIZE = 256
UNIT_SIZE = 32


_ragged_cf_cache = {}


@functools.lru_cache(maxsize=128)
def _compile_moe_ragged_sorting(*, num_experts: int, unit_size: int = UNIT_SIZE):
    """Build the four-kernel flat-route counting sort."""

    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if unit_size <= 0:
        raise ValueError(f"unit_size must be positive, got {unit_size}")

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def clear_kernel(
        expert_frequency: fx.Tensor,
        expert_cursors: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        moe_buf_i32: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_max_padded: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
    ):
        c_num_experts = fx.Int32(num_experts)
        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        gid = gpu.block_idx.x * fx.Int32(BLOCK_SIZE) + gpu.thread_idx.x
        stride = gpu.grid_dim.x * fx.Int32(BLOCK_SIZE)
        freq_rsrc = buffer_ops.create_buffer_resource(expert_frequency, max_size=True)
        cursor_rsrc = buffer_ops.create_buffer_resource(expert_cursors, max_size=True)
        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        weights_rsrc = buffer_ops.create_buffer_resource(sorted_weights, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(moe_buf_i32, max_size=True)

        frequency_iters = (c_num_experts + stride - c_one) // stride
        for iteration in range(
            fx.Index(0),
            ArithValue(frequency_iters).index_cast(T.index),
            fx.Index(1),
        ):
            index = gid + fx.Int32(iteration) * stride
            if index < c_num_experts:
                buffer_ops.buffer_store(c_zero, freq_rsrc, index)
                buffer_ops.buffer_store(c_zero, cursor_rsrc, index)

        sorted_iters = (i32_max_padded + stride - c_one) // stride
        for iteration in range(
            fx.Index(0),
            ArithValue(sorted_iters).index_cast(T.index),
            fx.Index(1),
        ):
            index = gid + fx.Int32(iteration) * stride
            if index < i32_max_padded:
                # ``tokens`` is outside the valid [0, tokens) range and
                # therefore serves as the padding sentinel after GEMM's
                # low-24-bit decode.
                buffer_ops.buffer_store(i32_tokens, ids_rsrc, index)
                buffer_ops.buffer_store(c_zero, weights_rsrc, index)

        output_iters = (i32_moe_buf_elems + stride - c_one) // stride
        for iteration in range(
            fx.Index(0),
            ArithValue(output_iters).index_cast(T.index),
            fx.Index(1),
        ):
            index = gid + fx.Int32(iteration) * stride
            if index < i32_moe_buf_elems:
                buffer_ops.buffer_store(c_zero, out_rsrc, index)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def histogram_kernel(
        expert_indices: fx.Tensor,
        expert_frequency: fx.Tensor,
        i32_routes: fx.Int32,
    ):
        c_num_experts = fx.Int32(num_experts)
        c_one = fx.Int32(1)
        gid = gpu.block_idx.x * fx.Int32(BLOCK_SIZE) + gpu.thread_idx.x
        if gid < i32_routes:
            experts_rsrc = buffer_ops.create_buffer_resource(expert_indices, max_size=True)
            expert = buffer_ops.buffer_load(experts_rsrc, gid, vec_width=1, dtype=T.i32)
            if (expert >= fx.Int32(0)) & (expert < c_num_experts):
                atomic_add(expert_frequency, expert, c_one, dtype_bytes=4)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def padded_prefix_kernel(
        expert_frequency: fx.Tensor,
        expert_cursors: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
    ):
        c_unit = fx.Int32(unit_size)
        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        # E is at most O(1k) for the target models.  A single serial prefix is
        # cheaper than another hierarchy of global scans and keeps this phase
        # deterministic; the route-heavy work remains fully parallel.
        if gpu.thread_idx.x == c_zero:
            freq_rsrc = buffer_ops.create_buffer_resource(expert_frequency, max_size=True)
            cursor_rsrc = buffer_ops.create_buffer_resource(expert_cursors, max_size=True)
            sorted_e_rsrc = buffer_ops.create_buffer_resource(sorted_expert_ids, max_size=True)
            nvalid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)

            offset = c_zero
            for expert_id in range_constexpr(num_experts):
                expert = fx.Int32(expert_id)
                count = buffer_ops.buffer_load(freq_rsrc, expert, vec_width=1, dtype=T.i32)
                blocks = (count + c_unit - c_one) // c_unit
                padded = (count == c_zero).select(c_zero, blocks * c_unit)
                buffer_ops.buffer_store(offset, cursor_rsrc, expert)

                block_start = offset // c_unit
                for block in range(
                    fx.Index(0),
                    ArithValue(blocks).index_cast(T.index),
                    fx.Index(1),
                ):
                    buffer_ops.buffer_store(
                        expert,
                        sorted_e_rsrc,
                        block_start + fx.Int32(block),
                    )
                offset = offset + padded

            buffer_ops.buffer_store(offset, nvalid_rsrc, c_zero)
            buffer_ops.buffer_store(i32_tokens, nvalid_rsrc, c_one)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def scatter_kernel(
        token_indices: fx.Tensor,
        expert_indices: fx.Tensor,
        route_weights: fx.Tensor,
        expert_cursors: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        i32_routes: fx.Int32,
    ):
        c_num_experts = fx.Int32(num_experts)
        c_one = fx.Int32(1)
        gid = gpu.block_idx.x * fx.Int32(BLOCK_SIZE) + gpu.thread_idx.x
        if gid < i32_routes:
            token_rsrc = buffer_ops.create_buffer_resource(token_indices, max_size=True)
            expert_rsrc = buffer_ops.create_buffer_resource(expert_indices, max_size=True)
            weights_rsrc = buffer_ops.create_buffer_resource(route_weights, max_size=True)
            sorted_ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
            sorted_w_rsrc = buffer_ops.create_buffer_resource(sorted_weights, max_size=True)

            token = buffer_ops.buffer_load(token_rsrc, gid, vec_width=1, dtype=T.i32)
            expert = buffer_ops.buffer_load(expert_rsrc, gid, vec_width=1, dtype=T.i32)
            weight_bits = buffer_ops.buffer_load(weights_rsrc, gid, vec_width=1, dtype=T.i32)
            if (expert >= fx.Int32(0)) & (expert < c_num_experts):
                position = fx.Int32(
                    atomic_add(expert_cursors, expert, c_one, dtype_bytes=4)
                )
                buffer_ops.buffer_store(token, sorted_ids_rsrc, position)
                buffer_ops.buffer_store(weight_bits, sorted_w_rsrc, position)

    @flyc.jit
    def launch_ragged_sorting(
        token_indices: fx.Tensor,
        expert_indices: fx.Tensor,
        route_weights: fx.Tensor,
        expert_frequency: fx.Tensor,
        expert_cursors: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        moe_buf_i32: fx.Tensor,
        i32_routes: fx.Int32,
        i32_tokens: fx.Int32,
        i32_max_padded: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
        i32_clear_grid: fx.Int32,
        i32_route_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        clear = clear_kernel(
            expert_frequency,
            expert_cursors,
            sorted_token_ids,
            sorted_weights,
            moe_buf_i32,
            i32_tokens,
            i32_max_padded,
            i32_moe_buf_elems,
        )
        clear.launch(
            grid=(i32_clear_grid, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

        histogram = histogram_kernel(expert_indices, expert_frequency, i32_routes)
        histogram.launch(
            grid=(i32_route_grid, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

        prefix = padded_prefix_kernel(
            expert_frequency,
            expert_cursors,
            sorted_expert_ids,
            num_valid_ids,
            i32_tokens,
        )
        prefix.launch(grid=(1, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

        scatter = scatter_kernel(
            token_indices,
            expert_indices,
            route_weights,
            expert_cursors,
            sorted_token_ids,
            sorted_weights,
            i32_routes,
        )
        scatter.launch(
            grid=(i32_route_grid, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_ragged_sorting


def _launch_cached(cache, key, launch_fn, args, stream):
    compiled = cache.get(key)
    stream_arg = fx.Stream(stream)
    if compiled is not None:
        compiled(*args, stream_arg)
        return
    launch_fn(*args, stream=stream)
    cache[key] = flyc.compile(launch_fn, *args, stream_arg)


def moe_ragged_sorting_flydsl(
    token_indices: torch.Tensor,
    expert_indices: torch.Tensor,
    route_weights: torch.Tensor,
    expert_frequency: torch.Tensor,
    expert_cursors: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    moe_buf: torch.Tensor,
    num_experts: int,
    *,
    tokens: int,
    max_padded_routes: int,
    unit_size: int = UNIT_SIZE,
):
    """Group a flat route list into the metadata consumed by grouped GEMMs.

    Inputs are canonical device tensors: token/expert indices are contiguous
    int32 and route weights contiguous float32.  Bounds validation intentionally
    stays in the higher-level API so this launch path does not synchronize.
    ``expert_frequency`` receives the exact route occurrence count for each
    expert and is not overwritten by the prefix phase.
    """

    routes = int(route_weights.numel())
    if int(token_indices.numel()) != routes or int(expert_indices.numel()) != routes:
        raise ValueError("token_indices, expert_indices, and route_weights must have equal length")
    if token_indices.dtype != torch.int32 or expert_indices.dtype != torch.int32:
        raise TypeError("token_indices and expert_indices must be int32")
    if route_weights.dtype != torch.float32:
        raise TypeError("route_weights must be float32")

    device = token_indices.device
    stream = torch.cuda.current_stream(device)
    moe_buf_i32 = moe_buf.view(torch.int32)
    clear_elems = max(num_experts, max_padded_routes, int(moe_buf_i32.numel()))
    num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    clear_grid = min(
        max(1, (clear_elems + BLOCK_SIZE - 1) // BLOCK_SIZE),
        num_cu * 2,
    )
    route_grid = max(1, (routes + BLOCK_SIZE - 1) // BLOCK_SIZE)

    launch_fn = _compile_moe_ragged_sorting(
        num_experts=num_experts,
        unit_size=unit_size,
    )
    args = (
        token_indices,
        expert_indices,
        route_weights,
        expert_frequency,
        expert_cursors,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf_i32,
        routes,
        int(tokens),
        int(max_padded_routes),
        int(moe_buf_i32.numel()),
        clear_grid,
        route_grid,
    )
    cache_key = (num_experts, unit_size, device.index)
    _launch_cached(_ragged_cf_cache, cache_key, launch_fn, args, stream)

    return (
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        expert_frequency,
        moe_buf,
    )


__all__ = ["moe_ragged_sorting_flydsl"]
