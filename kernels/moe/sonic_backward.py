# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""First-stage training backward for the gfx950 SonicMoE operator.

This module implements the dense BF16/FP16, fixed-K training contract needed by
the SonicMoE ROCm adapter.  All seven forward activations and optional expert bias
are supported.  It does not reuse the inference workspace.  Re-sorting and
recomputing the two forward intermediates makes retained graphs and overlapping
forward calls safe.

The implementation is entirely FlyDSL on device.  The general A16W16 GEMM is
used for all expert matrix products; small FlyDSL kernels implement routing
metadata, gather/scatter, activation derivatives, and the top-K reduction.  The
bring-up path performs one host synchronization to read expert frequencies and
dispatches six GEMMs per active expert.  A future grouped-MFMA implementation
can replace that dispatch without changing the public API.
"""

import functools
import math
from typing import TYPE_CHECKING

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch
from kernels.common import buffer_ops
from kernels.common.kernels_common import get_warp_size
from kernels.common.mem_ops import atomic_add
from kernels.common.tensor_shim import _run_compiled
from kernels.gemm.gemm_a16w16_gfx950 import gemm_a16w16
from kernels.moe.moe_2stage_a16wmix.gemm1 import _gelu_tanh_f32, _relu_f32, _sigmoid_f32, _tanh_f32
from kernels.moe.moe_gemm_2stage.moe_reduce import compile_moe_reduction
from kernels.moe.moe_sorting_kernel import moe_sorting_flydsl, moe_sorting_get_workspace_size

if TYPE_CHECKING:
    from kernels.moe.sonic import SonicMoEConfig


_BLOCK_THREADS = 256
_SORT_UNIT = 64
_TOKEN_MASK = 0x00FFFFFF
_WARP_SIZE = get_warp_size()
_RED_SLOTS = max(1, (_BLOCK_THREADS + _WARP_SIZE - 1) // _WARP_SIZE)
_GLU_ACTIVATIONS = frozenset({"swiglu", "geglu", "reglu"})
_SUPPORTED_ACTIVATIONS = frozenset({"swiglu", "geglu", "reglu", "gelu_tanh_approx", "relu", "silu", "relu_sq"})

_GEMM_KWARGS = {
    "block_m": 64,
    "block_n": 64,
    "block_k": 64,
    "stages": 2,
    "split_k": 1,
    "m_waves": 2,
    "n_waves": 2,
    "k_waves": 1,
    "group_m": 0,
    "use_half_tile_interleaved": False,
}


@fx.struct
class _ScoreBackwardSharedStorage:
    reduction: fx.Array[fx.Float32, _RED_SLOTS, 16]


def _gelu_tanh_derivative_f32(x):
    """Derivative of the tanh-approximate GELU used by stage 1."""

    one = fx.Float32(1.0)
    half = fx.Float32(0.5)
    sqrt_2_over_pi = fx.Float32(0.7978845608028654)
    coeff = fx.Float32(0.044715)
    inner = sqrt_2_over_pi * (x + coeff * x * x * x)
    tanh_inner = _tanh_f32(inner)
    dinner = sqrt_2_over_pi * (one + fx.Float32(3.0) * coeff * x * x)
    return half * (one + tanh_inner) + half * x * (one - tanh_inner * tanh_inner) * dinner


def _relu_derivative_f32(x):
    zero = fx.Float32(0.0)
    one = fx.Float32(1.0)
    positive = arith.cmpf(arith.CmpFPredicate.OGT, _raw(x), _raw(zero))
    return fx.Float32(arith.select(positive, _raw(one), _raw(zero)))


def _activation_f32(gate, up, activation: str):
    """Apply a compile-time selected public SonicMoE activation."""

    if activation == "swiglu":
        return gate * _sigmoid_f32(gate) * up
    if activation == "geglu":
        return _gelu_tanh_f32(gate) * up
    if activation == "reglu":
        return _relu_f32(gate) * up
    if activation == "gelu_tanh_approx":
        return _gelu_tanh_f32(gate)
    if activation == "relu":
        return _relu_f32(gate)
    if activation == "silu":
        return gate * _sigmoid_f32(gate)
    if activation == "relu_sq":
        relu = _relu_f32(gate)
        return relu * relu
    raise AssertionError(f"unexpected activation {activation!r}")


def _activation_backward_f32(gate, up, da, activation: str):
    """Apply a compile-time selected activation Jacobian-vector product."""

    one = fx.Float32(1.0)
    if activation == "swiglu":
        sigmoid = _sigmoid_f32(gate)
        activated_gate = gate * sigmoid
        derivative = sigmoid * (one + gate * (one - sigmoid))
        return da * up * derivative, da * activated_gate
    if activation == "geglu":
        return (
            da * up * _gelu_tanh_derivative_f32(gate),
            da * _gelu_tanh_f32(gate),
        )
    if activation == "reglu":
        return da * up * _relu_derivative_f32(gate), da * _relu_f32(gate)
    if activation == "gelu_tanh_approx":
        return da * _gelu_tanh_derivative_f32(gate), fx.Float32(0.0)
    if activation == "relu":
        return da * _relu_derivative_f32(gate), fx.Float32(0.0)
    if activation == "silu":
        sigmoid = _sigmoid_f32(gate)
        return da * sigmoid * (one + gate * (one - sigmoid)), fx.Float32(0.0)
    if activation == "relu_sq":
        return da * fx.Float32(2.0) * _relu_f32(gate), fx.Float32(0.0)
    raise AssertionError(f"unexpected activation {activation!r}")


def _max_padded_routes(tokens: int, num_experts: int, topk: int) -> tuple[int, int]:
    """Return the dense sorter's safe ``(rows, blocks)`` allocation bound."""

    routes = tokens * topk
    active_experts = min(num_experts, routes)
    padding_bound = (routes + active_experts * (_SORT_UNIT - 1)) // _SORT_UNIT
    per_expert_bound = active_experts * ((tokens + _SORT_UNIT - 1) // _SORT_UNIT)
    blocks = min(padding_bound, per_expert_bound)
    return blocks * _SORT_UNIT, blocks


@functools.lru_cache(maxsize=128)
def _compile_expert_histogram(num_experts: int, device_index: int):
    """Compile fixed-K expert histogram kernels for one device specialization."""

    del device_index

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def clear_kernel(expert_frequency: fx.Tensor):
        index = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        if index < fx.Int32(num_experts):
            rsrc = buffer_ops.create_buffer_resource(expert_frequency, max_size=True)
            buffer_ops.buffer_store(fx.Int32(0), rsrc, index)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def histogram_kernel(
        topk_ids: fx.Tensor,
        expert_frequency: fx.Tensor,
        i32_routes: fx.Int32,
    ):
        index = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        if index < i32_routes:
            ids_rsrc = buffer_ops.create_buffer_resource(topk_ids, max_size=True)
            expert = fx.Int32(buffer_ops.buffer_load(ids_rsrc, index, vec_width=1, dtype=T.i32))
            # Public validation deliberately avoids a device synchronization;
            # keep this guard so malformed ids cannot write out of bounds.
            if (expert >= fx.Int32(0)) & (expert < fx.Int32(num_experts)):
                atomic_add(expert_frequency, expert, fx.Int32(1), dtype_bytes=4)

    @flyc.jit
    def launch(
        topk_ids: fx.Tensor,
        expert_frequency: fx.Tensor,
        i32_routes: fx.Int32,
        i32_route_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        clear_kernel(expert_frequency).launch(
            grid=((num_experts + _BLOCK_THREADS - 1) // _BLOCK_THREADS, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )
        histogram_kernel(topk_ids, expert_frequency, i32_routes).launch(
            grid=(i32_route_grid, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_gather(hidden_size: int, compute_dtype: str, device_index: int):
    """Compile sorted-row gathers for hidden states and output gradients."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16
    vector_width = 4
    vectors_per_row = hidden_size // vector_width

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def gather_kernel(
        hidden_states: fx.Tensor,
        grad_output: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        x_sorted: fx.Tensor,
        dout_sorted: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
    ):
        index = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        total = i32_padded_rows * fx.Int32(vectors_per_row)
        if index < total:
            row = index // fx.Int32(vectors_per_row)
            column = (index % fx.Int32(vectors_per_row)) * fx.Int32(vector_width)
            ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
            x_rsrc = buffer_ops.create_buffer_resource(hidden_states, max_size=True)
            dout_rsrc = buffer_ops.create_buffer_resource(grad_output, max_size=True)
            x_sorted_rsrc = buffer_ops.create_buffer_resource(x_sorted, max_size=True)
            dout_sorted_rsrc = buffer_ops.create_buffer_resource(dout_sorted, max_size=True)

            packed = fx.Int32(buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32))
            token = packed & fx.Int32(_TOKEN_MASK)
            valid = token < i32_tokens
            safe_token = valid.select(token, fx.Int32(0))
            source = safe_token * fx.Int32(hidden_size) + column
            destination = row * fx.Int32(hidden_size) + column
            x_value = buffer_ops.buffer_load(
                x_rsrc,
                source,
                vec_width=vector_width,
                dtype=elem_dtype,
            )
            dout_value = buffer_ops.buffer_load(
                dout_rsrc,
                source,
                vec_width=vector_width,
                dtype=elem_dtype,
            )
            zero = fx.Vector.filled(vector_width, 0.0, elem_dtype)
            buffer_ops.buffer_store(valid.select(fx.Vector(x_value), zero), x_sorted_rsrc, destination)
            buffer_ops.buffer_store(valid.select(fx.Vector(dout_value), zero), dout_sorted_rsrc, destination)

    @flyc.jit
    def launch(
        hidden_states: fx.Tensor,
        grad_output: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        x_sorted: fx.Tensor,
        dout_sorted: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        gather_kernel(
            hidden_states,
            grad_output,
            sorted_token_ids,
            x_sorted,
            dout_sorted,
            i32_tokens,
            i32_padded_rows,
        ).launch(
            grid=(i32_grid, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_activation_prepare(
    hidden_size: int,
    intermediate_size: int,
    activation_name: str,
    compute_dtype: str,
    device_index: int,
):
    """Compile activation recomputation and routed-dout scaling."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16
    is_glu = activation_name in _GLU_ACTIVATIONS
    projection_size = intermediate_size * (2 if is_glu else 1)
    up_column_offset = intermediate_size if is_glu else 0

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def activation_prepare_kernel(
        preactivation: fx.Tensor,
        activation: fx.Tensor,
        dout_sorted: fx.Tensor,
        dy: fx.Tensor,
        sorted_weights: fx.Tensor,
    ):
        row = gpu.block_idx.x
        tid = gpu.thread_idx.x
        weights_rsrc = buffer_ops.create_buffer_resource(sorted_weights, max_size=True)
        preact_rsrc = buffer_ops.create_buffer_resource(preactivation, max_size=True)
        activation_rsrc = buffer_ops.create_buffer_resource(activation, max_size=True)
        dout_rsrc = buffer_ops.create_buffer_resource(dout_sorted, max_size=True)
        dy_rsrc = buffer_ops.create_buffer_resource(dy, max_size=True)
        route_weight = fx.Float32(buffer_ops.buffer_load(weights_rsrc, row, vec_width=1, dtype=T.f32))

        for base in range_constexpr(0, intermediate_size, _BLOCK_THREADS):
            column = tid + fx.Int32(base)
            if column < fx.Int32(intermediate_size):
                gate_offset = row * fx.Int32(projection_size) + column
                act_offset = row * fx.Int32(intermediate_size) + column
                gate = buffer_ops.buffer_load(preact_rsrc, gate_offset, vec_width=1, dtype=elem_dtype).extf(T.f32)
                up_offset = gate_offset + fx.Int32(up_column_offset)
                up = buffer_ops.buffer_load(
                    preact_rsrc,
                    up_offset,
                    vec_width=1,
                    dtype=elem_dtype,
                ).extf(T.f32)
                activation_f32 = _activation_f32(gate, up, activation_name)
                buffer_ops.buffer_store(
                    fx.Float32(activation_f32).to(elem_dtype),
                    activation_rsrc,
                    act_offset,
                )

        # Materialize dy in A16 before its two GEMMs.  This preserves the
        # multiply-before-GEMM dependency while making the standalone FlyDSL
        # backward's A16 input boundary explicit.
        for base in range_constexpr(0, hidden_size, _BLOCK_THREADS):
            column = tid + fx.Int32(base)
            if column < fx.Int32(hidden_size):
                offset = row * fx.Int32(hidden_size) + column
                dout_value = buffer_ops.buffer_load(dout_rsrc, offset, vec_width=1, dtype=elem_dtype).extf(T.f32)
                buffer_ops.buffer_store(
                    fx.Float32(dout_value * route_weight).to(elem_dtype),
                    dy_rsrc,
                    offset,
                )

    @flyc.jit
    def launch(
        preactivation: fx.Tensor,
        activation: fx.Tensor,
        dout_sorted: fx.Tensor,
        dy: fx.Tensor,
        sorted_weights: fx.Tensor,
        i32_padded_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        activation_prepare_kernel(
            preactivation,
            activation,
            dout_sorted,
            dy,
            sorted_weights,
        ).launch(
            grid=(i32_padded_rows, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_activation_derivative(
    intermediate_size: int,
    activation_name: str,
    compute_dtype: str,
    device_index: int,
):
    """Compile the selected activation's Jacobian-vector product."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16
    is_glu = activation_name in _GLU_ACTIVATIONS
    projection_size = intermediate_size * (2 if is_glu else 1)
    up_column_offset = intermediate_size if is_glu else 0

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def activation_derivative_kernel(
        preactivation: fx.Tensor,
        da: fx.Tensor,
        dz: fx.Tensor,
    ):
        row = gpu.block_idx.x
        tid = gpu.thread_idx.x
        preact_rsrc = buffer_ops.create_buffer_resource(preactivation, max_size=True)
        da_rsrc = buffer_ops.create_buffer_resource(da, max_size=True)
        dz_rsrc = buffer_ops.create_buffer_resource(dz, max_size=True)

        for base in range_constexpr(0, intermediate_size, _BLOCK_THREADS):
            column = tid + fx.Int32(base)
            if column < fx.Int32(intermediate_size):
                gate_offset = row * fx.Int32(projection_size) + column
                up_offset = gate_offset + fx.Int32(up_column_offset)
                act_offset = row * fx.Int32(intermediate_size) + column
                gate = buffer_ops.buffer_load(preact_rsrc, gate_offset, vec_width=1, dtype=elem_dtype).extf(T.f32)
                up = buffer_ops.buffer_load(
                    preact_rsrc,
                    up_offset,
                    vec_width=1,
                    dtype=elem_dtype,
                ).extf(T.f32)
                da_value = buffer_ops.buffer_load(da_rsrc, act_offset, vec_width=1, dtype=elem_dtype).extf(T.f32)
                dz_gate, dz_up = _activation_backward_f32(
                    gate,
                    up,
                    da_value,
                    activation_name,
                )
                buffer_ops.buffer_store(fx.Float32(dz_gate).to(elem_dtype), dz_rsrc, gate_offset)
                if const_expr(is_glu):
                    buffer_ops.buffer_store(fx.Float32(dz_up).to(elem_dtype), dz_rsrc, up_offset)

    @flyc.jit
    def launch(
        preactivation: fx.Tensor,
        da: fx.Tensor,
        dz: fx.Tensor,
        i32_padded_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        activation_derivative_kernel(preactivation, da, dz).launch(
            grid=(i32_padded_rows, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_bias_gradient_clear(
    projection_size: int,
    hidden_size: int,
    num_experts: int,
    compute_dtype: str,
    device_index: int,
):
    """Compile the exact-zero initialization for all expert bias gradients."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16
    db1_elements = num_experts * projection_size
    db2_elements = num_experts * hidden_size
    grid_size = (max(db1_elements, db2_elements) + _BLOCK_THREADS - 1) // _BLOCK_THREADS

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def clear_kernel(db1: fx.Tensor, db2: fx.Tensor):
        index = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        db1_rsrc = buffer_ops.create_buffer_resource(db1, max_size=True)
        db2_rsrc = buffer_ops.create_buffer_resource(db2, max_size=True)
        if index < fx.Int32(db1_elements):
            buffer_ops.buffer_store(elem_dtype(0.0), db1_rsrc, index)
        if index < fx.Int32(db2_elements):
            buffer_ops.buffer_store(elem_dtype(0.0), db2_rsrc, index)

    @flyc.jit
    def launch(
        db1: fx.Tensor,
        db2: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        clear_kernel(db1, db2).launch(
            grid=(grid_size, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_bias_gradient_reduction(
    projection_size: int,
    hidden_size: int,
    compute_dtype: str,
    device_index: int,
):
    """Compile one expert segment's FP32-accumulating bias reductions."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16
    grid_size = (max(projection_size, hidden_size) + _BLOCK_THREADS - 1) // _BLOCK_THREADS

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def reduction_kernel(
        dz: fx.Tensor,
        dy: fx.Tensor,
        db1: fx.Tensor,
        db2: fx.Tensor,
        i32_rows: fx.Int32,
    ):
        column = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        dz_rsrc = buffer_ops.create_buffer_resource(dz, max_size=True)
        dy_rsrc = buffer_ops.create_buffer_resource(dy, max_size=True)
        db1_rsrc = buffer_ops.create_buffer_resource(db1, max_size=True)
        db2_rsrc = buffer_ops.create_buffer_resource(db2, max_size=True)

        if column < fx.Int32(projection_size):
            db1_acc = fx.Float32(0.0)
            for row in range(fx.Int32(0), i32_rows, fx.Int32(1)):
                offset = row * fx.Int32(projection_size) + column
                value = buffer_ops.buffer_load(dz_rsrc, offset, vec_width=1, dtype=elem_dtype).extf(T.f32)
                db1_acc = db1_acc + value
            buffer_ops.buffer_store(db1_acc.to(elem_dtype), db1_rsrc, column)

        if column < fx.Int32(hidden_size):
            db2_acc = fx.Float32(0.0)
            for row in range(fx.Int32(0), i32_rows, fx.Int32(1)):
                offset = row * fx.Int32(hidden_size) + column
                value = buffer_ops.buffer_load(dy_rsrc, offset, vec_width=1, dtype=elem_dtype).extf(T.f32)
                db2_acc = db2_acc + value
            buffer_ops.buffer_store(db2_acc.to(elem_dtype), db2_rsrc, column)

    @flyc.jit
    def launch(
        dz: fx.Tensor,
        dy: fx.Tensor,
        db1: fx.Tensor,
        db2: fx.Tensor,
        i32_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        reduction_kernel(dz, dy, db1, db2, i32_rows).launch(
            grid=(grid_size, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_score_backward(hidden_size: int, topk: int, compute_dtype: str, device_index: int):
    """Compile ``ds = dot(dout, materialized_down_projection)``."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def score_backward_kernel(
        dout_sorted: fx.Tensor,
        projection: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        dtopk_weights: fx.Tensor,
        i32_tokens: fx.Int32,
    ):
        row = gpu.block_idx.x
        tid = gpu.thread_idx.x
        zero_f32 = fx.Float32(0.0)
        fm_fast = arith.FastMathFlags.fast

        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        dout_rsrc = buffer_ops.create_buffer_resource(dout_sorted, max_size=True)
        projection_rsrc = buffer_ops.create_buffer_resource(projection, max_size=True)
        ds_rsrc = buffer_ops.create_buffer_resource(dtopk_weights, max_size=True)
        packed = fx.Int32(buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32))
        token = packed & fx.Int32(_TOKEN_MASK)
        slot = packed >> fx.Int32(24)
        valid_route = token < i32_tokens

        thread_dot = zero_f32
        for base in range_constexpr(0, hidden_size, _BLOCK_THREADS):
            column = tid + fx.Int32(base)
            if column < fx.Int32(hidden_size):
                offset = row * fx.Int32(hidden_size) + column
                dout_value = buffer_ops.buffer_load(
                    dout_rsrc,
                    offset,
                    vec_width=1,
                    dtype=elem_dtype,
                ).extf(T.f32)
                projected = buffer_ops.buffer_load(
                    projection_rsrc,
                    offset,
                    vec_width=1,
                    dtype=elem_dtype,
                ).extf(T.f32)
                thread_dot = thread_dot + dout_value * projected

        lds = fx.SharedAllocator().allocate(_ScoreBackwardSharedStorage).peek()
        reduction = lds.reduction.view(fx.make_layout(_RED_SLOTS, 1))

        def wave_reduce_add(value):
            result = value
            with fx.fastmath(fm_fast):
                for shift_index in range_constexpr(int(math.log2(_WARP_SIZE))):
                    offset = _WARP_SIZE // (2 << shift_index)
                    result = result + gpu.shuffle_xor(result, offset, _WARP_SIZE)
            return result

        reduced = wave_reduce_add(thread_dot)
        if const_expr(_RED_SLOTS > 1):
            lane = tid % fx.Int32(_WARP_SIZE)
            wave = tid // fx.Int32(_WARP_SIZE)
            if lane == fx.Int32(0):
                fx.memref_store(reduced, reduction, wave)
            gpu.barrier()
            if wave == fx.Int32(0):
                in_range = lane < fx.Int32(_RED_SLOTS)
                safe_lane = in_range.select(lane, fx.Int32(0))
                partial = fx.memref_load(reduction, safe_lane)
                reduced = wave_reduce_add(in_range.select(partial, zero_f32))
                if lane == fx.Int32(0):
                    fx.memref_store(reduced, reduction, fx.Int32(0))
            gpu.barrier()
            reduced = fx.memref_load(reduction, fx.Int32(0))

        if tid == fx.Int32(0):
            if valid_route:
                destination = token * fx.Int32(topk) + slot
                buffer_ops.buffer_store(reduced, ds_rsrc, destination)

    @flyc.jit
    def launch(
        dout_sorted: fx.Tensor,
        projection: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        dtopk_weights: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        score_backward_kernel(
            dout_sorted,
            projection,
            sorted_token_ids,
            dtopk_weights,
            i32_tokens,
        ).launch(
            grid=(i32_padded_rows, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_unsort(hidden_size: int, topk: int, compute_dtype: str, device_index: int):
    """Compile sorted expert-row to dense ``[tokens, topk, H]`` scatter."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16
    vector_width = 4
    vectors_per_row = hidden_size // vector_width

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def unsort_kernel(
        dx_sorted: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        dx_routes: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
    ):
        index = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        total = i32_padded_rows * fx.Int32(vectors_per_row)
        if index < total:
            row = index // fx.Int32(vectors_per_row)
            column = (index % fx.Int32(vectors_per_row)) * fx.Int32(vector_width)
            ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
            source_rsrc = buffer_ops.create_buffer_resource(dx_sorted, max_size=True)
            destination_rsrc = buffer_ops.create_buffer_resource(dx_routes, max_size=True)
            packed = fx.Int32(buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32))
            token = packed & fx.Int32(_TOKEN_MASK)
            slot = packed >> fx.Int32(24)
            valid = token < i32_tokens
            if valid:
                source = row * fx.Int32(hidden_size) + column
                destination = ((token * fx.Int32(topk) + slot) * fx.Int32(hidden_size)) + column
                value = buffer_ops.buffer_load(
                    source_rsrc,
                    source,
                    vec_width=vector_width,
                    dtype=elem_dtype,
                )
                buffer_ops.buffer_store(value, destination_rsrc, destination)

    @flyc.jit
    def launch(
        dx_sorted: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        dx_routes: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        unsort_kernel(
            dx_sorted,
            sorted_token_ids,
            dx_routes,
            i32_tokens,
            i32_padded_rows,
        ).launch(
            grid=(i32_grid, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


def _validate_backward_inputs(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    grad_output: torch.Tensor,
    config: "SonicMoEConfig",
    b1: torch.Tensor | None,
    b2: torch.Tensor | None,
) -> tuple[int, int, int, int]:
    dtype_by_name = {"bf16": torch.bfloat16, "fp16": torch.float16}
    if config.compute_dtype not in dtype_by_name:
        raise ValueError(
            "sonic_moe_backward supports compute_dtype='bf16' or 'fp16', "
            f"got {config.compute_dtype!r}"
        )
    expected_dtype = dtype_by_name[config.compute_dtype]
    if config.activation not in _SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"sonic_moe_backward does not support activation={config.activation!r}; "
            f"expected one of {sorted(_SUPPORTED_ACTIVATIONS)}"
        )

    tokens = int(hidden_states.shape[0]) if hidden_states.ndim == 2 else -1
    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    num_experts = int(config.num_experts)
    topk = int(config.top_k)
    projection_size = intermediate_size * (2 if config.activation in _GLU_ACTIVATIONS else 1)
    if (b1 is None) != (b2 is None):
        raise ValueError("b1 and b2 must both be None or both be tensors")
    expected = {
        "hidden_states": (tokens, hidden_size),
        "w1": (num_experts, projection_size, hidden_size),
        "w2": (num_experts, hidden_size, intermediate_size),
        "topk_ids": (tokens, topk),
        "topk_weights": (tokens, topk),
        "grad_output": (tokens, hidden_size),
    }
    tensors = {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "grad_output": grad_output,
    }
    if b1 is not None:
        expected["b1"] = (num_experts, projection_size)
        expected["b2"] = (num_experts, hidden_size)
        tensors["b1"] = b1
        tensors["b2"] = b2
    if tokens <= 0:
        raise ValueError(f"hidden_states must be non-empty 2D, got shape {tuple(hidden_states.shape)}")
    if tokens > _TOKEN_MASK:
        raise ValueError(f"token count must fit the sorter's 24-bit token field, got {tokens}")
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected[name]:
            raise ValueError(f"{name} must have shape {expected[name]}, got {tuple(tensor.shape)}")
        if not tensor.is_cuda or tensor.device != hidden_states.device:
            raise ValueError(f"{name} must be on the same ROCm device as hidden_states")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    floating_names = ["hidden_states", "w1", "w2", "grad_output"]
    if b1 is not None:
        floating_names.extend(("b1", "b2"))
    for name in floating_names:
        if tensors[name].dtype != expected_dtype:
            raise TypeError(f"{name} must be {expected_dtype}, got {tensors[name].dtype}")
    if topk_ids.dtype != torch.int32:
        raise TypeError(f"topk_ids must be int32, got {topk_ids.dtype}")
    if topk_weights.dtype != torch.float32:
        raise TypeError(f"topk_weights must be float32, got {topk_weights.dtype}")
    if hidden_size % 64 != 0 or intermediate_size % 64 != 0:
        raise ValueError("hidden_size and intermediate_size must be multiples of 64")
    return tokens, hidden_size, intermediate_size, num_experts


def _ptr(tensor: torch.Tensor):
    return flyc.from_c_void_p(fx.Uint8, tensor.data_ptr())


def sonic_moe_backward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    grad_output: torch.Tensor,
    config: "SonicMoEConfig",
    *,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Differentiate dense BF16/FP16 fixed-K SonicMoE, optionally with bias.

    Parameters use logical, expert-major weights: ``w1[E, 2I, H]`` for GLU
    activations, ``w1[E, I, H]`` for pointwise activations, and
    ``w2[E, H, I]``.  Routing tensors are ``topk_ids[int32, T, K]`` and
    ``topk_weights[float32, T, K]``.  Without bias, the returned tuple remains
    ``(dx, dw1, dw2, dtopk_weights)``.  When ``b1`` and ``b2`` are supplied,
    ``(db1, db2)`` are appended. Tensor and bias gradients preserve the A16
    input dtype; routing-score gradients use FP32.

    Expert ids must be in range and distinct within each token.  As in the
    inference fixed-K path, value validation is an unchecked hot-path
    precondition so the only synchronization is the bring-up implementation's
    expert-frequency copy used for per-expert GEMM dispatch.
    """

    tokens, hidden_size, intermediate_size, num_experts = _validate_backward_inputs(
        hidden_states,
        w1,
        w2,
        topk_ids,
        topk_weights,
        grad_output,
        config,
        b1,
        b2,
    )
    topk = int(config.top_k)
    compute_dtype = str(config.compute_dtype)
    activation_name = str(config.activation)
    projection_size = intermediate_size * (2 if activation_name in _GLU_ACTIVATIONS else 1)
    has_bias = b1 is not None
    device = hidden_states.device
    device_index = device.index or 0
    with torch.cuda.device(device):
        arch = get_rocm_arch()
    if not str(arch).startswith("gfx95"):
        raise RuntimeError(f"SonicMoE backward requires gfx95*, got {arch!r}")
    max_padded, max_blocks = _max_padded_routes(tokens, num_experts, topk)
    workspace_elements = moe_sorting_get_workspace_size(
        tokens,
        num_experts,
        topk,
        unit_size=_SORT_UNIT,
    )

    # Backward owns every buffer: no forward LRU scratch is retained or read.
    sorted_token_ids = torch.empty(max_padded, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(max_padded, dtype=torch.float32, device=device)
    sorted_expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    sorting_workspace = (
        torch.empty(workspace_elements, dtype=torch.int32, device=device) if workspace_elements else None
    )
    sorter_dummy = torch.empty(4, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(num_experts, dtype=torch.int32, device=device)

    x_sorted = torch.empty((max_padded, hidden_size), dtype=hidden_states.dtype, device=device)
    dout_sorted = torch.empty_like(x_sorted)
    dy = torch.empty_like(x_sorted)
    projection = torch.empty_like(x_sorted)
    preactivation = torch.empty((max_padded, projection_size), dtype=hidden_states.dtype, device=device)
    activation = torch.empty((max_padded, intermediate_size), dtype=hidden_states.dtype, device=device)
    da = torch.empty_like(activation)
    dz = torch.empty_like(preactivation)
    dx_sorted = torch.empty_like(x_sorted)
    dx_routes = torch.empty((tokens, topk, hidden_size), dtype=hidden_states.dtype, device=device)

    dx = torch.empty_like(hidden_states, memory_format=torch.contiguous_format)
    dw1 = torch.zeros_like(w1, memory_format=torch.contiguous_format)
    dw2 = torch.zeros_like(w2, memory_format=torch.contiguous_format)
    dtopk_weights = torch.empty_like(topk_weights, memory_format=torch.contiguous_format)
    db1 = torch.empty_like(b1, memory_format=torch.contiguous_format) if b1 is not None else None
    db2 = torch.empty_like(b2, memory_format=torch.contiguous_format) if b2 is not None else None

    # Custom kernels consume raw storage only; detach keeps DLPack conversion
    # valid when this function is called from a torch.autograd.Function.
    x_arg = hidden_states.detach()
    w1_arg = w1.detach()
    w2_arg = w2.detach()
    ids_arg = topk_ids.detach()
    weights_arg = topk_weights.detach()
    dout_arg = grad_output.detach()
    b1_arg = b1.detach() if b1 is not None else None
    b2_arg = b2.detach() if b2 is not None else None

    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        routes = tokens * topk
        route_grid = max(1, (routes + _BLOCK_THREADS - 1) // _BLOCK_THREADS)
        histogram = _compile_expert_histogram(num_experts, device_index)
        _run_compiled(histogram, ids_arg, expert_frequency, routes, route_grid, stream)

        if has_bias:
            clear_bias_gradients = _compile_bias_gradient_clear(
                projection_size,
                hidden_size,
                num_experts,
                compute_dtype,
                device_index,
            )
            _run_compiled(clear_bias_gradients, db1, db2, stream)

        moe_sorting_flydsl(
            ids_arg,
            weights_arg,
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            sorter_dummy,
            num_experts,
            unit_size=_SORT_UNIT,
            workspace=sorting_workspace,
        )

        # One explicit synchronization is accepted in this bring-up path.  It
        # determines active expert slices; every tensor operation remains on
        # device and is implemented by FlyDSL.
        frequencies = expert_frequency.cpu().tolist()
        segments: list[tuple[int, int, int]] = []
        offset = 0
        for expert, count in enumerate(frequencies):
            if count:
                padded = ((int(count) + _SORT_UNIT - 1) // _SORT_UNIT) * _SORT_UNIT
                segments.append((expert, offset, padded))
                offset += padded
        padded_rows = offset

        gather = _compile_gather(hidden_size, compute_dtype, device_index)
        gather_work = padded_rows * (hidden_size // 4)
        gather_grid = max(1, (gather_work + _BLOCK_THREADS - 1) // _BLOCK_THREADS)
        _run_compiled(
            gather,
            x_arg,
            dout_arg,
            sorted_token_ids,
            x_sorted,
            dout_sorted,
            tokens,
            padded_rows,
            gather_grid,
            stream,
        )

        # Recompute the materialized A16 preactivation.
        for expert, start, rows in segments:
            end = start + rows
            gemm_a16w16(
                x_sorted[start:end],
                w1_arg[expert].transpose(0, 1),
                out=preactivation[start:end],
                bias=None if b1_arg is None else b1_arg[expert],
                user_kwargs=_GEMM_KWARGS,
                stream=stream,
                layout="nt",
            )

        activation_prepare = _compile_activation_prepare(
            hidden_size,
            intermediate_size,
            activation_name,
            compute_dtype,
            device_index,
        )
        _run_compiled(
            activation_prepare,
            preactivation,
            activation,
            dout_sorted,
            dy,
            sorted_weights,
            padded_rows,
            stream,
        )

        # Recompute the down projection for ds, and use the materialized A16
        # dy for both da and dW2.  The A16 materialization is this backward
        # implementation's explicit numerical contract.
        for expert, start, rows in segments:
            end = start + rows
            gemm_a16w16(
                activation[start:end],
                w2_arg[expert].transpose(0, 1),
                out=projection[start:end],
                bias=None if b2_arg is None else b2_arg[expert],
                user_kwargs=_GEMM_KWARGS,
                stream=stream,
                layout="nt",
            )
            gemm_a16w16(
                dy[start:end],
                w2_arg[expert],
                out=da[start:end],
                user_kwargs=_GEMM_KWARGS,
                stream=stream,
                layout="nn",
            )
            gemm_a16w16(
                dy[start:end].transpose(0, 1),
                activation[start:end],
                out=dw2[expert],
                user_kwargs=_GEMM_KWARGS,
                stream=stream,
                layout="tn",
            )

        activation_derivative = _compile_activation_derivative(
            intermediate_size,
            activation_name,
            compute_dtype,
            device_index,
        )
        _run_compiled(
            activation_derivative,
            preactivation,
            da,
            dz,
            padded_rows,
            stream,
        )

        # Each expert owns a disjoint output slice, so no atomics or
        # cross-expert reductions are needed for dW1 or routed dX.
        bias_gradient_reduction = (
            _compile_bias_gradient_reduction(
                projection_size,
                hidden_size,
                compute_dtype,
                device_index,
            )
            if has_bias
            else None
        )
        for expert, start, rows in segments:
            end = start + rows
            gemm_a16w16(
                dz[start:end].transpose(0, 1),
                x_sorted[start:end],
                out=dw1[expert],
                user_kwargs=_GEMM_KWARGS,
                stream=stream,
                layout="tn",
            )
            gemm_a16w16(
                dz[start:end],
                w1_arg[expert],
                out=dx_sorted[start:end],
                user_kwargs=_GEMM_KWARGS,
                stream=stream,
                layout="nn",
            )
            if bias_gradient_reduction is not None:
                _run_compiled(
                    bias_gradient_reduction,
                    dz[start:end],
                    dy[start:end],
                    db1[expert],
                    db2[expert],
                    rows,
                    stream,
                )

        score_backward = _compile_score_backward(hidden_size, topk, compute_dtype, device_index)
        _run_compiled(
            score_backward,
            dout_sorted,
            projection,
            sorted_token_ids,
            dtopk_weights,
            tokens,
            padded_rows,
            stream,
        )

        unsort = _compile_unsort(hidden_size, topk, compute_dtype, device_index)
        unsort_work = padded_rows * (hidden_size // 4)
        unsort_grid = max(1, (unsort_work + _BLOCK_THREADS - 1) // _BLOCK_THREADS)
        _run_compiled(
            unsort,
            dx_sorted,
            sorted_token_ids,
            dx_routes,
            tokens,
            padded_rows,
            unsort_grid,
            stream,
        )

        reduction_dtype = "f16" if compute_dtype == "fp16" else "bf16"
        reduce = compile_moe_reduction(topk=topk, model_dim=hidden_size, dtype_str=reduction_dtype)
        _run_compiled(
            reduce,
            _ptr(dx_routes),
            _ptr(dx),
            _ptr(expert_frequency),
            _ptr(ids_arg),
            tokens,
            stream,
        )

    result = (dx, dw1, dw2, dtopk_weights)
    if has_bias:
        return (*result, db1, db2)
    return result


__all__ = ["sonic_moe_backward"]
