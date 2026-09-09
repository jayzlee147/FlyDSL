# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""First-stage training backward for the gfx950 SonicMoE operator.

This module implements the dense BF16/FP16 fixed-K and flat ragged-route
training contracts needed by the SonicMoE ROCm adapter. All seven forward
activations and optional expert bias are supported. It does not reuse the
inference workspace. Backward-owned sorting/workspace and invocation-owned
forward state keep retained graphs and overlapping forward calls safe.

The implementation is entirely FlyDSL on device.  Standalone backward
recomputes both forward intermediates; fixed-K BF16 SwiGLU backward can instead
consume an invocation-owned forward state and skip W1 recomputation.  Eligible
no-bias state paths also compute an A16 ``q = dout @ W2`` once and fuse route
score, activation-derivative, and dW2-input work without materializing the W2
projection.  The fused exact-row schedule covers the fully grouped short-token
BM16 path and the tuned T4096/H4096/I2048/E64/K8 BM64 path.  Other grouped W1,
W2, dA, dW1, dW2, and dX contractions use gfx950 MFMA kernels where their
individual policies allow; remaining matrix products use the general A16W16
GEMM.  Small FlyDSL kernels implement routing metadata, gather/scatter,
activation derivatives, and the top-K reduction.  Legacy dtype, activation,
bias, ragged-route, and untuned long-token contracts retain the conservative
host-dispatched fallback.
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
from kernels.moe.grouped_da_gfx950 import compile_grouped_da_gfx950
from kernels.moe.moe_2stage_a16wmix.gemm1 import (
    _gelu_tanh_f32,
    _relu_f32,
    _sigmoid_f32,
    _tanh_f32,
    compile_gemm1_a16w4_port,
)
from kernels.moe.moe_2stage_a16wmix.gemm2 import compile_gemm2_a16w4_port
from kernels.moe.moe_gemm_2stage.moe_reduce import compile_moe_reduction
from kernels.moe.moe_ragged_sorting_kernel import moe_ragged_sorting_flydsl
from kernels.moe.moe_sorting_kernel import moe_sorting_flydsl, moe_sorting_get_workspace_size
from kernels.moe.sonic_grouped_a16w16 import compile_sonic_grouped_a16w16_nn
from kernels.moe.sonic_grouped_scheduler import (
    build_compact_m_tile_descriptors,
    fixed_compact_m_tile_descriptor_upper_bound,
    ragged_compact_m_tile_descriptor_upper_bound,
)
from kernels.moe.sonic_grouped_tn import (
    active_expert_descriptor_capacity,
    active_expert_queue_elements,
    build_active_expert_queue_flydsl,
    grouped_tn_from_metadata_flydsl,
    grouped_tn_from_queue_flydsl,
    zero_inactive_weight_grads_flydsl,
    zero_weight_grads_adaptive_flydsl,
)

if TYPE_CHECKING:
    from kernels.moe.sonic import SonicMoEConfig


_BLOCK_THREADS = 256
# Backward materializes expert segments and contracts across their padded row
# dimension with the generic A16 GEMM below.  That GEMM uses BLOCK_K=64 and has
# no K-tail path, so every non-empty segment must remain a multiple of 64.
# Forward's independently tuned route tile is not a legal substitute here.
_BACKWARD_SORT_UNIT = 64
_TOKEN_MASK = 0x00FFFFFF
_MAX_SIGNED_I32 = (1 << 31) - 1
_MAX_BUFFER_BYTE_OFFSET = (1 << 32) - 1
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

# The first device-driven backward GEMM specialization targets the dominant
# BF16 SwiGLU training shape.  A 16-row compute tile preserves the sorter's
# 64-row expert metadata granularity while four K-waves cover gfx950's short-M
# regime.  These are compile-time policy choices, not public ABI knobs.
_GROUPED_W1_BM = 16
_GROUPED_W1_BN = 64
_GROUPED_W1_BK = 64
_GROUPED_W1_K_WAVE = 4
# At the fixed-K policy boundary, a compact real-M-tile queue exposes long
# expert segments as independent CTAs.  BN128 halves the queue grid, while two
# K waves retain enough N parallelism for the production H3584/I512 shape.
_COMPACT_W1_MIN_TOKENS = 64
_COMPACT_W1_BM = 16
_COMPACT_W1_BN = 128
_COMPACT_W1_BK = 64
_COMPACT_W1_K_WAVE = 2
# This BM16/k-wave4 specialization is tuned for short expert segments.  Fixed-K
# routing guarantees at most one edge per (token, expert), so ``tokens`` is a
# distribution-independent upper bound.  Ragged routing permits duplicates and
# therefore uses the total route count as its conservative bound.  Above this
# limit the existing BM64 GEMM has enough M work to amortize host dispatch and
# is substantially faster (for example, balanced T4096/E64 has 512 rows/expert).
_GROUPED_W1_MAX_EXPERT_ROWS = 128

# W2 recompute has the opposite aspect ratio (K=intermediate, N=hidden).  Keep
# its policy independent so gfx950 tuning can evolve without coupling the two
# kernels. BM32/BN256/BK64 is the measured balanced-routing winner for the
# production I=512 contraction on MI350, while remaining within 1 us of the
# best decode profile.
_GROUPED_W2_BM = 32
_GROUPED_W2_BN = 256
_GROUPED_W2_BK = 64
_GROUPED_W2_MAX_EXPERT_ROWS = 128

# dA consumes public row-major W2 directly: [M, H] @ [H, I] -> [M, I].  The
# NN-specialized kernel transposes B in LDS and uses CDNA4 LDSReadTrans16_64b
# before MFMA.  Three measured gfx950 profiles cover decode, sparse short-M,
# and long/hot expert segments.  Legacy fallbacks reuse their existing host
# frequency read; the short hostless path dispatches from the device queue.
_GROUPED_DA_BN = 64
_GROUPED_DA_MAX_EXPERT_ROWS = 4096
_GROUPED_DA_STAGES = 2
_GROUPED_DA_E896_DENSE_STAGES = 3
_GROUPED_DA_E896_DENSE_GRID_CAP = 256

# Keep the first token-major grouped-dW1 rollout on the production E896
# retained-state contract until its cache behavior has been measured broadly.
_DIRECT_GROUPED_DW1_RHS_SHAPE = (4096, 3584, 512, 896, 16)

# Weight-gradient TN contractions are output-stationary: each CTA owns one
# output tile and reduces all rows for one expert before a single BF16 store.
# BK32 is the measured short/ragged-row winner on gfx950.  Exact resource
# bounds make the unused portion of the final K tile read as zero without
# loading the sorter's full 64-row padding.
_GROUPED_DW2_BK = 32
_GROUPED_DW2_PIPELINE_THRESHOLD = 64
_GROUPED_DW2_SPARSE_EXPERTS = 32


def _grouped_dw2_tuning(
    max_expert_rows: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    active_experts: int | None = None,
) -> tuple[int, int, int, int, int, int]:
    """Return ``(BM, BN, BK, K-pad, M-waves, N-waves)`` for dW2."""

    block_m = next(tile for tile in (256, 128, 64) if hidden_size % tile == 0)
    block_n = next(tile for tile in (256, 128, 64) if intermediate_size % tile == 0)
    # Resource-aware persistent grids let smaller output tiles expose useful
    # second/third/fourth resident CTAs on gfx950.  Decode and sparse short-K
    # routes prefer 128x128; a dense 2-4 row/expert batch benefits from twice
    # as many M tiles while retaining BN256's contiguous output stores.
    supports_128 = hidden_size % 128 == 0 and intermediate_size % 128 == 0
    sparse_short = (
        active_experts is not None and active_experts <= _GROUPED_DW2_SPARSE_EXPERTS and max_expert_rows <= 128
    )
    if supports_128 and (max_expert_rows <= 1 or sparse_short):
        block_m = 128
        block_n = 128
    elif max_expert_rows <= 4 and hidden_size % 128 == 0 and intermediate_size % 256 == 0:
        block_m = 128
        block_n = 256
    m_waves = 4 if block_m == 256 else 2
    n_waves = 4 if block_n == 256 else 2
    return block_m, block_n, _GROUPED_DW2_BK, 0, m_waves, n_waves


def _grouped_dw2_stages(max_expert_rows: int) -> int:
    """Use triple buffering only once an expert spans multiple K tiles."""

    return 3 if max_expert_rows >= _GROUPED_DW2_PIPELINE_THRESHOLD else 2


def _launch_grouped_dw2(
    dy: torch.Tensor,
    activation: torch.Tensor,
    expert_frequency: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    dw2: torch.Tensor,
    active_expert_storage: torch.Tensor | None,
    *,
    use_hostless_grouped: bool,
    use_tn_metadata_direct: bool,
    max_expert_rows: int,
    hidden_size: int,
    intermediate_size: int,
    active_experts: int,
    stream: torch.cuda.Stream,
) -> None:
    """Launch the tuned grouped dW2 profiles after dy becomes available."""

    if use_hostless_grouped and not use_tn_metadata_direct:
        # The queue count is already produced by compact W1.  Launch disjoint
        # sparse/dense profiles without a host-side active-count readback.
        sparse_dw2 = _grouped_dw2_tuning(
            max_expert_rows,
            hidden_size,
            intermediate_size,
            active_experts=_GROUPED_DW2_SPARSE_EXPERTS,
        )
        balanced_dw2 = _grouped_dw2_tuning(
            min(max_expert_rows, 4),
            hidden_size,
            intermediate_size,
        )
        grouped_dw2_profiles = (
            (*sparse_dw2, _grouped_dw2_stages(max_expert_rows), 0, _GROUPED_DW2_SPARSE_EXPERTS),
            (
                *balanced_dw2,
                _grouped_dw2_stages(min(max_expert_rows, 4)),
                _GROUPED_DW2_SPARSE_EXPERTS + 1,
                None,
            ),
        )
    else:
        grouped_dw2_profiles = (
            (
                *_grouped_dw2_tuning(
                    max_expert_rows,
                    hidden_size,
                    intermediate_size,
                    active_experts=active_experts,
                ),
                _grouped_dw2_stages(max_expert_rows),
                0,
                None,
            ),
        )

    for (
        dw2_bm,
        dw2_bn,
        dw2_bk,
        dw2_k_padding,
        dw2_mw,
        dw2_nw,
        dw2_stages,
        min_active_experts,
        max_active_experts,
    ) in grouped_dw2_profiles:
        grouped_dw2_kwargs = {
            "block_m": dw2_bm,
            "block_n": dw2_bn,
            "block_k": dw2_bk,
            "k_padding": dw2_k_padding,
            "m_waves": dw2_mw,
            "n_waves": dw2_nw,
            "stages": dw2_stages,
            "stream": stream,
        }
        if use_tn_metadata_direct:
            grouped_tn_from_metadata_flydsl(
                dy,
                activation,
                expert_frequency,
                sorted_expert_ids,
                num_valid_ids,
                dw2,
                **grouped_dw2_kwargs,
            )
        else:
            assert active_expert_storage is not None
            grouped_tn_from_queue_flydsl(
                dy,
                activation,
                expert_frequency,
                active_expert_storage,
                dw2,
                min_active_experts=min_active_experts,
                max_active_experts=max_active_experts,
                **grouped_dw2_kwargs,
            )


def _grouped_da_tuning(max_expert_rows: int, hidden_size: int) -> tuple[int, int, int, int, int]:
    """Return ``(BM, BN, BK, m_waves, n_waves)`` for grouped dA."""

    if max_expert_rows <= 1 and hidden_size % 128 == 0:
        return (16, _GROUPED_DA_BN, 128, 1, 4)
    if max_expert_rows <= 16:
        return (32, _GROUPED_DA_BN, 64, 2, 2)
    return (64, _GROUPED_DA_BN, 64, 2, 2)


# dX has the row-major NN shape ``[M, 2I] @ [2I, H]``.  BM16 avoids doing the
# sorter's full 64-row padding for short experts.  The production T4096 shape
# instead benefits from one BM64 tile per sorter block: it cuts repeated W1
# slab loads while retaining enough M work per CTA.  That shape gets its own
# descriptor queue so the latency-sensitive BM16 W1/short-dX queue remains
# unchanged.  A persistent four-workgroup-per-CU launch bound (1024 on
# MI350/MI355X) keeps large descriptor grids resident without a long tail.
_GROUPED_DX_BM = 16
_GROUPED_DX_BK = 64
_GROUPED_DX_STAGES = 2
_GROUPED_DX_GRID_CAP = 1024
_GROUPED_DX_DENSE_EXPERTS = 256
_LARGE_GROUPED_DX_BM = 64
_LARGE_GROUPED_DX_BN = 256
_LARGE_GROUPED_DX_N_WAVES = 4
_LARGE_GROUPED_DX_SHAPES = frozenset(
    {
        # Dense prefill/training bucket used by the standalone FlyDSL tests.
        (4096, 4096, 2048, 64, 8),
        # Default SonicMoE ROCm adapter bucket.  Without the device queue this
        # shape falls back to one dX GEMM launch per active expert (up to 896).
        (4096, 3584, 512, 896, 16),
    }
)
# The first long-token hostless rollout stays narrower than the large-dX
# policy.  In particular, the E64/H4096 bucket has not yet been audited for
# distribution-independent dA/dW scheduling without the legacy frequency
# readback.  The default ROCm adapter bucket below has a retained-state dataflow
# in which every contraction and row transform already consumes a device queue
# or device-resident extent.
_HOSTLESS_LARGE_STATE_SHAPES = frozenset({(4096, 3584, 512, 896, 16)})


def _grouped_da_hostless_profiles(
    *,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    max_expert_rows: int,
):
    """Return sparse/dense device-guarded dA profiles for hostless backward."""

    shape = (tokens, hidden_size, intermediate_size, num_experts, topk)
    if shape in _HOSTLESS_LARGE_STATE_SHAPES:
        # The E896 production bucket is dominated by repeated W2 fetches.  A
        # wider M tile amortizes each slab across more routes while preserving
        # the per-wave BM64 profile's two M repeats.  Dense routing uses eight
        # waves (MW4/NW2); sparse hot routing uses BM256 with MW8/NW1 so the B
        # tile remains exactly covered by the 512-thread async-load layout.
        # Triple buffering raises the dense BM128 profile from 48 to 72 KiB
        # LDS.  LDS alone fits two CTAs/CU, but its combined VGPR/AGPR usage
        # quantizes residency to one; a 256-CTA persistent grid fills gfx950.
        # Keep sparse BM256 double-buffered: its 120 KiB stage-3 footprint
        # lowers occupancy and regresses instruction-wait/VMEM-latency PMCs.
        sparse_tuning = (256, _GROUPED_DA_BN, 64, 8, 1)
        dense_tuning = (128, _GROUPED_DA_BN, 64, 4, 2)
        sparse_stages = _GROUPED_DA_STAGES
        dense_stages = _GROUPED_DA_E896_DENSE_STAGES
        dense_grid_cap = _GROUPED_DA_E896_DENSE_GRID_CAP
    else:
        sparse_tuning = _grouped_da_tuning(max_expert_rows, hidden_size)
        dense_tuning = _grouped_da_tuning(min(max_expert_rows, 2), hidden_size)
        sparse_stages = dense_stages = _GROUPED_DA_STAGES
        dense_grid_cap = None
    return (
        (*sparse_tuning, True, 0, _GROUPED_DW2_SPARSE_EXPERTS, sparse_stages, None),
        (
            *dense_tuning,
            False,
            _GROUPED_DW2_SPARSE_EXPERTS + 1,
            None,
            dense_stages,
            dense_grid_cap,
        ),
    )


# Weight gradients are output-stationary TN contractions.  BM/BN128 with BK32
# is the measured throughput winner once an expert can own multiple rows;
# decode prefers BM/BN64 because its output tile count exposes more parallelism
# without increasing per-CTA work.  Exact-frequency buffer resources suppress
# sorter-padding traffic in both profiles.
_GROUPED_DW1_BLOCK_K = 32
_GROUPED_DW1_MAX_EXPERT_ROWS = 4096
# The custom inactive-only fill wins once at least one eighth of the production
# expert set is live.  Below that point nearly the whole 9.9-GiB pair is still
# written and torch's dense memset is faster.  Legacy paths reuse their host
# frequency readback; hostless paths select from the device queue count.
_INACTIVE_WEIGHT_GRAD_ZERO_ACTIVE_RATIO = 8

# Device-sized row kernels use a host-known allocation bound only to size the
# launch.  Each workgroup then walks the sorter-produced ``num_valid_ids[0]``
# extent in a grid-stride loop.  Four workgroups per gfx950 CU is enough to
# cover latency without launching tens of thousands of idle CTAs for sparse
# hot-expert distributions.
_HOSTLESS_ROW_GRID_CAP = 1024


def _grouped_dw1_tuning(
    max_expert_rows: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    direct_rhs: bool = False,
) -> tuple[int, int, int, int, int, int]:
    """Return ``(BM, BN, BK, K-pad, M-waves, N-waves)`` for grouped dW1."""

    preferred_tile = 64 if max_expert_rows <= 1 else 128
    output_m = 2 * intermediate_size
    block_m = preferred_tile if output_m % preferred_tile == 0 else 64
    if direct_rhs and max_expert_rows > 1 and hidden_size % 256 == 0:
        block_n = 256
        n_waves = 4
    else:
        block_n = preferred_tile if hidden_size % preferred_tile == 0 else 64
        n_waves = 2
    return (block_m, block_n, _GROUPED_DW1_BLOCK_K, 0, 2, n_waves)


def _use_grouped_dw1(
    *,
    compute_dtype: str,
    activation: str,
    hidden_size: int,
    intermediate_size: int,
    tokens: int,
    routes: int,
    flat_routes: bool,
) -> bool:
    """Select gfx950's output-stationary grouped dW1 contraction."""

    max_expert_rows = routes if flat_routes else tokens
    return (
        compute_dtype == "bf16"
        and activation == "swiglu"
        and hidden_size % 64 == 0
        and (2 * intermediate_size) % 64 == 0
        and max_expert_rows <= _GROUPED_DW1_MAX_EXPERT_ROWS
    )


def _grouped_dx_tuning(active_experts: int, hidden_size: int) -> tuple[int, int]:
    """Return ``(BN, n_waves)`` for the observed expert distribution."""

    if active_experts >= _GROUPED_DX_DENSE_EXPERTS and hidden_size % 256 == 0:
        return (256, 4)
    if hidden_size % 128 == 0:
        return (128, 2)
    return (64, 2)


def _use_large_grouped_dx_descriptor_queue(
    *,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    flat_routes: bool,
) -> bool:
    """Select the shape-static BM64 dX queue tuned on gfx950.

    Keep this deliberately exact until other long-token shapes have end-to-end
    measurements.  Every input is host-known, so enabling the queue never
    introduces a routing-statistics readback.
    """

    shape = (tokens, hidden_size, intermediate_size, num_experts, topk)
    return not flat_routes and shape in _LARGE_GROUPED_DX_SHAPES


def _use_compact_w1_descriptor_queue(
    *,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
) -> bool:
    """Select the no-readback real-M-tile queue when its tuning is legal."""

    return (
        tokens >= _COMPACT_W1_MIN_TOKENS
        and hidden_size % (_COMPACT_W1_K_WAVE * _COMPACT_W1_BK) == 0
        and intermediate_size % _COMPACT_W1_BN == 0
    )


def _grouped_w1_tuning(
    *,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[int, int, int, int, bool]:
    """Return ``(BM, BN, BK, k_wave, compact_grid)`` for grouped W1."""

    compact_grid = _use_compact_w1_descriptor_queue(
        tokens=tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    if compact_grid:
        return (
            _COMPACT_W1_BM,
            _COMPACT_W1_BN,
            _COMPACT_W1_BK,
            _COMPACT_W1_K_WAVE,
            True,
        )
    return (
        _GROUPED_W1_BM,
        _GROUPED_W1_BN,
        _GROUPED_W1_BK,
        _GROUPED_W1_K_WAVE,
        False,
    )


def _use_grouped_w1_recompute(
    *,
    compute_dtype: str,
    activation: str,
    hidden_size: int,
    intermediate_size: int,
    tokens: int,
    routes: int,
    flat_routes: bool,
) -> bool:
    """Return whether the short-M grouped W1 specialization is applicable."""

    max_expert_rows = routes if flat_routes else tokens
    bm, bn, bk, k_wave, _ = _grouped_w1_tuning(
        tokens=tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    return (
        compute_dtype == "bf16"
        and activation == "swiglu"
        and hidden_size % (k_wave * bk) == 0
        and intermediate_size % bn == 0
        and _BACKWARD_SORT_UNIT % bm == 0
        and max_expert_rows <= _GROUPED_W1_MAX_EXPERT_ROWS
    )


def _use_grouped_dx(
    *,
    compute_dtype: str,
    activation: str,
    hidden_size: int,
    intermediate_size: int,
    tokens: int,
    num_experts: int,
    topk: int,
    flat_routes: bool,
    compact_w1: bool,
) -> bool:
    """Select the device-scheduled BF16 SwiGLU dX contraction.

    Short compact mode deliberately reuses the already-built W1 descriptor
    queue.  Fixed-K T1 is queue-free because every active expert owns one real
    row.  The tuned T4096 production shape uses an independent BM64 queue.
    """

    return (
        compute_dtype == "bf16"
        and activation == "swiglu"
        and (2 * intermediate_size) % _GROUPED_DX_BK == 0
        and hidden_size % 64 == 0
        and (
            compact_w1
            or (tokens == 1 and not flat_routes)
            or _use_large_grouped_dx_descriptor_queue(
                tokens=tokens,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                topk=topk,
                flat_routes=flat_routes,
            )
        )
    )


def _use_direct_grouped_dx_routes(*, use_grouped_dx: bool, flat_routes: bool) -> bool:
    """Fold fixed-K grouped dX's sorted-row permutation into its epilogue."""

    return use_grouped_dx and not flat_routes


def _use_hostless_grouped_backward(
    *,
    flat_routes: bool,
    has_bias: bool,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    reuse_forward_preactivation: bool,
    use_large_grouped_dx: bool,
    use_grouped_w1: bool,
    use_grouped_w2: bool,
    use_grouped_dw2: bool,
    use_grouped_da: bool,
    use_grouped_dw1: bool,
    use_grouped_dx: bool,
) -> bool:
    """Select a fully device-dispatched fixed-K backward.

    Every matrix contraction must already have a grouped implementation.  The
    remaining row kernels can then consume the sorter extent directly, so no
    host-side expert-frequency reconstruction is necessary.  The short path
    recomputes W1/W2 and therefore requires both grouped projections.  The
    strict long-shape path instead requires retained W1 state and the fused
    dA/dscore dataflow, so neither projection is part of its backward graph.
    Keep that rollout limited to the audited default E896 adapter bucket.
    """

    short_grouped = tokens <= 128 and use_grouped_w1 and use_grouped_w2
    shape = (tokens, hidden_size, intermediate_size, num_experts, topk)
    retained_large_grouped = reuse_forward_preactivation and use_large_grouped_dx and shape in _HOSTLESS_LARGE_STATE_SHAPES
    return (
        not flat_routes
        and not has_bias
        and use_grouped_dw2
        and use_grouped_da
        and use_grouped_dw1
        and use_grouped_dx
        and (short_grouped or retained_large_grouped)
    )


def _use_fused_da_dscore(
    *,
    reuse_forward_preactivation: bool,
    use_hostless_grouped: bool,
    use_compact_w1: bool,
    use_large_grouped_dx: bool,
    flat_routes: bool,
    has_bias: bool,
    compute_dtype: str,
    activation: str,
) -> bool:
    """Select the first gfx950 unscaled-dA/route-score fusion rollout.

    The fused dataflow uses retained forward state plus an exact-row descriptor
    schedule.  Short shapes reuse the BM16 compact W1/dX queue; the production
    T4096 shape reuses its independently tuned BM64 dX queue.  ``dy`` initially
    carries the exact gathered A16 output gradient, grouped dA computes
    ``q = dout @ W2``, and one row kernel subsequently forms dA's score
    scaling, dscore, and the scaled dW2 input.  Bias, ragged routes, legacy
    host dispatch, and non-SwiGLU/dtype contracts retain the projection-
    recompute implementation.
    """

    return (
        reuse_forward_preactivation
        and ((use_hostless_grouped and use_compact_w1) or use_large_grouped_dx)
        and not flat_routes
        and not has_bias
        and compute_dtype == "bf16"
        and activation == "swiglu"
    )


def _use_direct_grouped_dw1_rhs(
    *,
    reuse_forward_preactivation: bool,
    use_fused_forward_state_prepare: bool,
    use_grouped_dw1: bool,
    flat_routes: bool,
    has_bias: bool,
    compute_dtype: str,
    activation: str,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
) -> bool:
    """Select token-major RHS gather for retained-state grouped dW1.

    Keep the first rollout on the exact fixed-K BF16 SwiGLU path whose fused
    state preparation has no other consumer for the sorter-order hidden-state
    copy.  Standalone, ragged, bias, and legacy contracts continue to
    materialize ``x_sorted``.
    """

    return (
        reuse_forward_preactivation
        and use_fused_forward_state_prepare
        and use_grouped_dw1
        and not flat_routes
        and not has_bias
        and compute_dtype == "bf16"
        and activation == "swiglu"
        and (tokens, hidden_size, intermediate_size, num_experts, topk)
        == _DIRECT_GROUPED_DW1_RHS_SHAPE
    )


@functools.lru_cache(maxsize=64)
def _compile_grouped_dx(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    block_m: int,
    block_n: int,
    n_waves: int,
    compact_grid: bool,
    device_index: int,
    min_active_experts: int = 0,
    max_active_experts: int | None = None,
    store_route_slots: bool = False,
    top_k: int = 1,
):
    """Build the gfx950 grouped ``dZ @ W1`` specialization."""

    return compile_sonic_grouped_a16w16_nn(
        contraction_size=2 * intermediate_size,
        output_size=hidden_size,
        num_experts=num_experts,
        block_m=block_m,
        block_n=block_n,
        block_k=_GROUPED_DX_BK,
        stages=_GROUPED_DX_STAGES,
        n_waves=n_waves,
        sorted_block_m=_BACKWARD_SORT_UNIT,
        compact_grid=compact_grid,
        device_index=device_index,
        min_active_experts=min_active_experts,
        max_active_experts=max_active_experts,
        store_route_slots=store_route_slots,
        top_k=top_k,
    )


@functools.lru_cache(maxsize=64)
def _compile_grouped_w1_recompute(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    has_bias: bool,
    interleaved_w1: bool,
    compact_grid: bool,
    device_index: int,
):
    """Build grouped raw-W1 preactivation recompute for BF16 SwiGLU.

    ``device_index`` intentionally participates in the cache key because a
    loaded compiled function is tied to its ROCm device.
    """

    del device_index
    if compact_grid:
        bm, bn, bk, k_wave = (
            _COMPACT_W1_BM,
            _COMPACT_W1_BN,
            _COMPACT_W1_BK,
            _COMPACT_W1_K_WAVE,
        )
    else:
        bm, bn, bk, k_wave = (
            _GROUPED_W1_BM,
            _GROUPED_W1_BN,
            _GROUPED_W1_BK,
            _GROUPED_W1_K_WAVE,
        )
    return compile_gemm1_a16w4_port(
        BM=bm,
        SORTED_BM=_BACKWARD_SORT_UNIT,
        D_HIDDEN=hidden_size,
        D_INTER=intermediate_size,
        NE=num_experts,
        TOPK=topk,
        TILE_N=bn,
        TILE_K=bk,
        act="swiglu",
        b_cache_mod=0,
        w_dtype="bf16",
        a_dtype="bf16",
        w_layout="interleaved" if interleaved_w1 else "standard",
        k_wave=k_wave,
        round_preact_bf16=False,
        has_bias=has_bias,
        logical_dense_weight=True,
        store_preactivation=True,
        expert_grid=not compact_grid,
        compact_grid=compact_grid,
    )


def _use_grouped_w2_recompute(
    *,
    compute_dtype: str,
    activation: str,
    hidden_size: int,
    intermediate_size: int,
    tokens: int,
    routes: int,
    flat_routes: bool,
) -> bool:
    """Return whether the short-M grouped W2 projection is applicable."""

    max_expert_rows = routes if flat_routes else tokens
    return (
        compute_dtype == "bf16"
        and activation == "swiglu"
        and hidden_size % _GROUPED_W2_BN == 0
        and intermediate_size % _GROUPED_W2_BK == 0
        and max_expert_rows <= _GROUPED_W2_MAX_EXPERT_ROWS
    )


def _use_grouped_dw2(
    *,
    compute_dtype: str,
    hidden_size: int,
    intermediate_size: int,
) -> bool:
    """Return whether the gfx950 grouped BF16 dW2 contraction is legal."""

    return compute_dtype == "bf16" and hidden_size % 64 == 0 and intermediate_size % 64 == 0


def _use_grouped_da(
    *,
    compute_dtype: str,
    activation: str,
    hidden_size: int,
    intermediate_size: int,
    tokens: int,
    routes: int,
    flat_routes: bool,
) -> bool:
    """Return whether the short-M grouped dA specialization is applicable."""

    max_expert_rows = routes if flat_routes else tokens
    return (
        compute_dtype == "bf16"
        and activation == "swiglu"
        and hidden_size % 64 == 0
        and intermediate_size % _GROUPED_DA_BN == 0
        and max_expert_rows <= _GROUPED_DA_MAX_EXPERT_ROWS
    )


@functools.lru_cache(maxsize=64)
def _compile_grouped_da(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    block_m: int,
    block_n: int,
    block_k: int,
    m_waves: int,
    n_waves: int,
    device_index: int,
    queue_direct: bool = False,
    min_active_experts: int = 0,
    max_active_experts: int | None = None,
    stages: int = _GROUPED_DA_STAGES,
    persistent: bool = False,
):
    """Build the grouped raw-W2 dA contraction for BF16 SwiGLU."""

    del device_index
    return compile_grouped_da_gfx950(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        sorted_block_size=_BACKWARD_SORT_UNIT,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        m_waves=m_waves,
        n_waves=n_waves,
        queue_direct=queue_direct,
        persistent=persistent,
        min_active_experts=min_active_experts,
        max_active_experts=max_active_experts,
    )


@functools.lru_cache(maxsize=64)
def _compile_grouped_w2_recompute(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    has_bias: bool,
    device_index: int,
):
    """Build grouped raw-W2 sorted projection recompute for BF16 SwiGLU."""

    del device_index
    return compile_gemm2_a16w4_port(
        BM=_GROUPED_W2_BM,
        SORTED_BM=_BACKWARD_SORT_UNIT,
        NE=num_experts,
        N_OUT=hidden_size,
        D_INTER=intermediate_size,
        TILE_N=_GROUPED_W2_BN,
        TILE_K=_GROUPED_W2_BK,
        xcd_swizzle=0,
        b_cache_mod=0,
        w_dtype="bf16",
        a_dtype="bf16",
        persist=False,
        has_bias=has_bias,
        round_projection_bf16=False,
        output_mode="atomic",
        TOPK=topk,
        logical_dense_weight=True,
        store_sorted_projection=True,
        expert_grid=True,
    )


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


def _max_padded_routes(
    tokens: int,
    num_experts: int,
    topk: int,
    sort_unit: int,
) -> tuple[int, int]:
    """Return the dense sorter's safe ``(rows, blocks)`` allocation bound."""

    routes = tokens * topk
    active_experts = min(num_experts, routes)
    padding_bound = (routes + active_experts * (sort_unit - 1)) // sort_unit
    per_expert_bound = active_experts * ((tokens + sort_unit - 1) // sort_unit)
    blocks = min(padding_bound, per_expert_bound)
    return blocks * sort_unit, blocks


def _max_padded_flat_routes(
    routes: int,
    num_experts: int,
    sort_unit: int,
) -> tuple[int, int]:
    """Return the ragged sorter's safe ``(rows, blocks)`` allocation bound."""

    active_experts = min(num_experts, routes)
    blocks = (routes + active_experts * (sort_unit - 1)) // sort_unit
    return blocks * sort_unit, blocks


def _materialize_expert_segments(
    expert_frequency: torch.Tensor,
    sort_unit: int,
) -> tuple[list[int], list[tuple[int, int, int]], int, int]:
    """Synchronously reconstruct padded expert slices for legacy fallbacks."""

    frequencies = expert_frequency.cpu().tolist()
    segments: list[tuple[int, int, int]] = []
    offset = 0
    for expert, count in enumerate(frequencies):
        if count:
            padded = ((int(count) + sort_unit - 1) // sort_unit) * sort_unit
            segments.append((expert, offset, padded))
            offset += padded
    return frequencies, segments, offset, max(int(count) for count in frequencies)


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
def _compile_gather(
    hidden_size: int,
    compute_dtype: str,
    device_index: int,
    device_padded_rows: bool = False,
):
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
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
    ):
        index = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        stride = gpu.grid_dim.x * fx.Int32(_BLOCK_THREADS)
        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        x_rsrc = buffer_ops.create_buffer_resource(hidden_states, max_size=True)
        dout_rsrc = buffer_ops.create_buffer_resource(grad_output, max_size=True)
        x_sorted_rsrc = buffer_ops.create_buffer_resource(x_sorted, max_size=True)
        dout_sorted_rsrc = buffer_ops.create_buffer_resource(dout_sorted, max_size=True)

        def gather_vector(index):
            row = index // fx.Int32(vectors_per_row)
            column = (index % fx.Int32(vectors_per_row)) * fx.Int32(vector_width)
            packed = fx.Int32(buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32))
            token = packed & fx.Int32(_TOKEN_MASK)
            valid = (token >= fx.Int32(0)) & (token < i32_tokens)
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

        padded_rows = i32_padded_rows
        if const_expr(device_padded_rows):
            valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
            padded_rows = fx.Int32(buffer_ops.buffer_load(valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32))
        total = padded_rows * fx.Int32(vectors_per_row)
        if const_expr(device_padded_rows):
            for vector_index in range(index, total, stride):
                gather_vector(fx.Int32(vector_index))
        else:
            if index < total:
                gather_vector(index)

    @flyc.jit
    def launch(
        hidden_states: fx.Tensor,
        grad_output: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        x_sorted: fx.Tensor,
        dout_sorted: fx.Tensor,
        num_valid_ids: fx.Tensor,
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
            num_valid_ids,
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
    interleaved_w1: bool,
    device_index: int,
    device_padded_rows: bool = False,
):
    """Compile activation recomputation and routed-dout scaling."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16
    is_glu = activation_name in _GLU_ACTIVATIONS
    projection_size = intermediate_size * (2 if is_glu else 1)
    projection_column_stride = 2 if is_glu and interleaved_w1 else 1
    up_column_offset = 1 if is_glu and interleaved_w1 else intermediate_size if is_glu else 0

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def activation_prepare_kernel(
        preactivation: fx.Tensor,
        activation: fx.Tensor,
        dout_sorted: fx.Tensor,
        dy: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        weights_rsrc = buffer_ops.create_buffer_resource(sorted_weights, max_size=True)
        preact_rsrc = buffer_ops.create_buffer_resource(preactivation, max_size=True)
        activation_rsrc = buffer_ops.create_buffer_resource(activation, max_size=True)
        dout_rsrc = buffer_ops.create_buffer_resource(dout_sorted, max_size=True)
        dy_rsrc = buffer_ops.create_buffer_resource(dy, max_size=True)
        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)

        def prepare_row(row):
            route_weight = fx.Float32(buffer_ops.buffer_load(weights_rsrc, row, vec_width=1, dtype=T.f32))

            for base in range_constexpr(0, intermediate_size, _BLOCK_THREADS):
                column = tid + fx.Int32(base)
                if column < fx.Int32(intermediate_size):
                    gate_offset = row * fx.Int32(projection_size) + column * fx.Int32(projection_column_stride)
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
            # multiply-before-GEMM dependency while making the standalone
            # FlyDSL backward's A16 input boundary explicit.
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

        padded_rows = i32_padded_rows
        if const_expr(device_padded_rows):
            valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
            padded_rows = fx.Int32(buffer_ops.buffer_load(valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32))

        if const_expr(device_padded_rows):
            for row_value in range(gpu.block_idx.x, padded_rows, gpu.grid_dim.x):
                row = fx.Int32(row_value)
                packed = fx.Int32(buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32))
                token = packed & fx.Int32(_TOKEN_MASK)
                if token < i32_tokens:
                    prepare_row(row)
        else:
            prepare_row(gpu.block_idx.x)

    @flyc.jit
    def launch(
        preactivation: fx.Tensor,
        activation: fx.Tensor,
        dout_sorted: fx.Tensor,
        dy: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        activation_prepare_kernel(
            preactivation,
            activation,
            dout_sorted,
            dy,
            sorted_weights,
            sorted_token_ids,
            num_valid_ids,
            i32_tokens,
            i32_padded_rows,
        ).launch(
            grid=(i32_padded_rows, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_activation_prepare_from_forward_state(
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    interleaved_w1: bool,
    device_index: int,
    device_padded_rows: bool = False,
):
    """Gather route-order BF16 preactivation and prepare backward rows.

    The state is compact ``[T, K, 2I]`` route-order storage, while every
    downstream backward contraction consumes sorter order.  Folding that
    permutation into activation preparation avoids both W1 recomputation and
    a standalone gather launch.
    """

    del device_index
    elem_dtype = fx.BFloat16
    projection_size = 2 * intermediate_size
    projection_column_stride = 2 if interleaved_w1 else 1
    up_column_offset = 1 if interleaved_w1 else intermediate_size

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def activation_prepare_from_state_kernel(
        route_preactivation: fx.Tensor,
        preactivation: fx.Tensor,
        activation: fx.Tensor,
        dout_sorted: fx.Tensor,
        dy: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        route_preact_rsrc = buffer_ops.create_buffer_resource(route_preactivation, max_size=True)
        preact_rsrc = buffer_ops.create_buffer_resource(preactivation, max_size=True)
        activation_rsrc = buffer_ops.create_buffer_resource(activation, max_size=True)
        dout_rsrc = buffer_ops.create_buffer_resource(dout_sorted, max_size=True)
        dy_rsrc = buffer_ops.create_buffer_resource(dy, max_size=True)
        weights_rsrc = buffer_ops.create_buffer_resource(sorted_weights, max_size=True)
        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        zero = fx.Float32(0.0).to(elem_dtype)

        def prepare_real_row(row, token, slot):
            route_row = token * fx.Int32(topk) + slot
            route_base = route_row * fx.Int32(projection_size)
            sorted_base = row * fx.Int32(projection_size)
            for base in range_constexpr(0, intermediate_size, _BLOCK_THREADS):
                column = tid + fx.Int32(base)
                if column < fx.Int32(intermediate_size):
                    relative_gate = column * fx.Int32(projection_column_stride)
                    route_gate_offset = route_base + relative_gate
                    sorted_gate_offset = sorted_base + relative_gate
                    gate = buffer_ops.buffer_load(
                        route_preact_rsrc,
                        route_gate_offset,
                        vec_width=1,
                        dtype=elem_dtype,
                    )
                    up = buffer_ops.buffer_load(
                        route_preact_rsrc,
                        route_gate_offset + fx.Int32(up_column_offset),
                        vec_width=1,
                        dtype=elem_dtype,
                    )
                    buffer_ops.buffer_store(gate, preact_rsrc, sorted_gate_offset)
                    buffer_ops.buffer_store(
                        up,
                        preact_rsrc,
                        sorted_gate_offset + fx.Int32(up_column_offset),
                    )
                    activation_f32 = _activation_f32(gate.extf(T.f32), up.extf(T.f32), "swiglu")
                    buffer_ops.buffer_store(
                        fx.Float32(activation_f32).to(elem_dtype),
                        activation_rsrc,
                        row * fx.Int32(intermediate_size) + column,
                    )

            route_weight = fx.Float32(buffer_ops.buffer_load(weights_rsrc, row, vec_width=1, dtype=T.f32))
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

        def clear_padding_row(row):
            for base in range_constexpr(0, intermediate_size, _BLOCK_THREADS):
                column = tid + fx.Int32(base)
                if column < fx.Int32(intermediate_size):
                    relative_gate = column * fx.Int32(projection_column_stride)
                    sorted_gate_offset = row * fx.Int32(projection_size) + relative_gate
                    buffer_ops.buffer_store(zero, preact_rsrc, sorted_gate_offset)
                    buffer_ops.buffer_store(
                        zero,
                        preact_rsrc,
                        sorted_gate_offset + fx.Int32(up_column_offset),
                    )
                    buffer_ops.buffer_store(
                        zero,
                        activation_rsrc,
                        row * fx.Int32(intermediate_size) + column,
                    )
            for base in range_constexpr(0, hidden_size, _BLOCK_THREADS):
                column = tid + fx.Int32(base)
                if column < fx.Int32(hidden_size):
                    buffer_ops.buffer_store(
                        zero,
                        dy_rsrc,
                        row * fx.Int32(hidden_size) + column,
                    )

        def decode_route(row):
            packed = fx.Int32(buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32))
            token = packed & fx.Int32(_TOKEN_MASK)
            slot = (packed >> fx.Int32(24)) & fx.Int32(0xFF)
            return token, slot

        padded_rows = i32_padded_rows
        if const_expr(device_padded_rows):
            valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
            padded_rows = fx.Int32(buffer_ops.buffer_load(valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32))

        if const_expr(device_padded_rows):
            for row_value in range(gpu.block_idx.x, padded_rows, gpu.grid_dim.x):
                row = fx.Int32(row_value)
                token, slot = decode_route(row)
                if (token < i32_tokens) & (slot < fx.Int32(topk)):
                    prepare_real_row(row, token, slot)
        else:
            # Legacy contractions consume complete SORTED_BM-padded segments,
            # so their sentinel rows must explicitly materialize zero tails.
            # The hostless branch above skips sentinels throughout and gets
            # dX's short tile tail from its independently zeroed dZ workspace.
            row = gpu.block_idx.x
            token, slot = decode_route(row)
            if (token < i32_tokens) & (slot < fx.Int32(topk)):
                prepare_real_row(row, token, slot)
            else:
                clear_padding_row(row)

    @flyc.jit
    def launch(
        route_preactivation: fx.Tensor,
        preactivation: fx.Tensor,
        activation: fx.Tensor,
        dout_sorted: fx.Tensor,
        dy: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        activation_prepare_from_state_kernel(
            route_preactivation,
            preactivation,
            activation,
            dout_sorted,
            dy,
            sorted_weights,
            sorted_token_ids,
            num_valid_ids,
            i32_tokens,
            i32_padded_rows,
        ).launch(
            grid=(i32_padded_rows, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_fused_forward_state_prepare(
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    interleaved_w1: bool,
    device_index: int,
    schedule_block_m: int = _COMPACT_W1_BM,
    defer_dy_scaling: bool = False,
    store_x_sorted: bool = True,
):
    """Gather the exact live rows and prepare retained-state backward inputs.

    This specialization is restricted to the fully grouped BF16 SwiGLU
    fixed-K path.  It avoids walking every 64-row sorter pad, never
    materializes a sorted copy of ``grad_output``, and combines the remaining
    hidden-state gather with activation and routed-gradient preparation.

    It consumes an existing counter-first BM16 or BM64 dX descriptor queue.
    The schedule touches real routes plus at most one tile tail per expert;
    the independently zeroed ``dZ`` buffer continues to provide that tail's
    dX contract.
    """

    del device_index
    elem_dtype = fx.BFloat16
    projection_size = 2 * intermediate_size
    projection_column_stride = 2 if interleaved_w1 else 1
    up_column_offset = 1 if interleaved_w1 else intermediate_size
    if schedule_block_m not in (_COMPACT_W1_BM, _LARGE_GROUPED_DX_BM):
        raise ValueError("state-prepare schedule_block_m must be 16 or 64")

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def fused_prepare_kernel(
        hidden_states: fx.Tensor,
        grad_output: fx.Tensor,
        route_preactivation: fx.Tensor,
        x_sorted: fx.Tensor,
        activation: fx.Tensor,
        dy: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        schedule: fx.Tensor,
        i32_tokens: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        dout_rsrc = buffer_ops.create_buffer_resource(grad_output, max_size=True)
        x_rsrc = dout_rsrc
        if const_expr(store_x_sorted):
            x_rsrc = buffer_ops.create_buffer_resource(hidden_states, max_size=True)
        route_preact_rsrc = buffer_ops.create_buffer_resource(route_preactivation, max_size=True)
        activation_rsrc = buffer_ops.create_buffer_resource(activation, max_size=True)
        dy_rsrc = buffer_ops.create_buffer_resource(dy, max_size=True)
        x_sorted_rsrc = dy_rsrc
        if const_expr(store_x_sorted):
            x_sorted_rsrc = buffer_ops.create_buffer_resource(x_sorted, max_size=True)
        weights_rsrc = buffer_ops.create_buffer_resource(sorted_weights, max_size=True)
        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)

        def prepare_real_row(row):
            packed = fx.Int32(buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32))
            token = packed & fx.Int32(_TOKEN_MASK)
            slot = (packed >> fx.Int32(24)) & fx.Int32(0xFF)
            if (token < i32_tokens) & (slot < fx.Int32(topk)):
                route_weight = fx.Float32(1.0)
                if const_expr(not defer_dy_scaling):
                    route_weight = fx.Float32(
                        buffer_ops.buffer_load(weights_rsrc, row, vec_width=1, dtype=T.f32)
                    )
                source_base = token * fx.Int32(hidden_size)
                destination_base = row * fx.Int32(hidden_size)
                for base in range_constexpr(0, hidden_size, _BLOCK_THREADS):
                    column = tid + fx.Int32(base)
                    if column < fx.Int32(hidden_size):
                        source = source_base + column
                        destination = destination_base + column
                        dout_value = buffer_ops.buffer_load(
                            dout_rsrc,
                            source,
                            vec_width=1,
                            dtype=elem_dtype,
                        ).extf(T.f32)
                        if const_expr(store_x_sorted):
                            x_value = buffer_ops.buffer_load(
                                x_rsrc,
                                source,
                                vec_width=1,
                                dtype=elem_dtype,
                            )
                            buffer_ops.buffer_store(x_value, x_sorted_rsrc, destination)
                        buffer_ops.buffer_store(
                            fx.Float32(dout_value * route_weight).to(elem_dtype),
                            dy_rsrc,
                            destination,
                        )

                route_row = token * fx.Int32(topk) + slot
                route_base = route_row * fx.Int32(projection_size)
                activation_base = row * fx.Int32(intermediate_size)
                for base in range_constexpr(0, intermediate_size, _BLOCK_THREADS):
                    column = tid + fx.Int32(base)
                    if column < fx.Int32(intermediate_size):
                        relative_gate = column * fx.Int32(projection_column_stride)
                        gate = buffer_ops.buffer_load(
                            route_preact_rsrc,
                            route_base + relative_gate,
                            vec_width=1,
                            dtype=elem_dtype,
                        ).extf(T.f32)
                        up = buffer_ops.buffer_load(
                            route_preact_rsrc,
                            route_base + relative_gate + fx.Int32(up_column_offset),
                            vec_width=1,
                            dtype=elem_dtype,
                        ).extf(T.f32)
                        activation_f32 = _activation_f32(gate, up, "swiglu")
                        buffer_ops.buffer_store(
                            fx.Float32(activation_f32).to(elem_dtype),
                            activation_rsrc,
                            activation_base + column,
                        )

        schedule_rsrc = buffer_ops.create_buffer_resource(schedule, max_size=True)
        total_tiles = fx.Int32(
            buffer_ops.buffer_load(schedule_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32)
        )
        total_rows = total_tiles * fx.Int32(schedule_block_m)
        for task_value in range(gpu.block_idx.x, total_rows, gpu.grid_dim.x):
            task = fx.Int32(task_value)
            descriptor_index = task // fx.Int32(schedule_block_m)
            local_row = task % fx.Int32(schedule_block_m)
            descriptor = fx.Int32(
                buffer_ops.buffer_load(
                    schedule_rsrc,
                    descriptor_index + fx.Int32(1),
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            prepare_real_row(descriptor * fx.Int32(schedule_block_m) + local_row)

    @flyc.jit
    def launch(
        hidden_states: fx.Tensor,
        grad_output: fx.Tensor,
        route_preactivation: fx.Tensor,
        x_sorted: fx.Tensor,
        activation: fx.Tensor,
        dy: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        schedule: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        fused_prepare_kernel(
            hidden_states,
            grad_output,
            route_preactivation,
            x_sorted,
            activation,
            dy,
            sorted_weights,
            sorted_token_ids,
            schedule,
            i32_tokens,
        ).launch(
            grid=(i32_grid, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_activation_derivative(
    intermediate_size: int,
    activation_name: str,
    compute_dtype: str,
    interleaved_w1: bool,
    device_index: int,
    device_padded_rows: bool = False,
):
    """Compile the selected activation's Jacobian-vector product."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16
    is_glu = activation_name in _GLU_ACTIVATIONS
    projection_size = intermediate_size * (2 if is_glu else 1)
    projection_column_stride = 2 if is_glu and interleaved_w1 else 1
    up_column_offset = 1 if is_glu and interleaved_w1 else intermediate_size if is_glu else 0

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def activation_derivative_kernel(
        preactivation: fx.Tensor,
        da: fx.Tensor,
        dz: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        preact_rsrc = buffer_ops.create_buffer_resource(preactivation, max_size=True)
        da_rsrc = buffer_ops.create_buffer_resource(da, max_size=True)
        dz_rsrc = buffer_ops.create_buffer_resource(dz, max_size=True)
        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)

        def differentiate_row(row):
            for base in range_constexpr(0, intermediate_size, _BLOCK_THREADS):
                column = tid + fx.Int32(base)
                if column < fx.Int32(intermediate_size):
                    gate_offset = row * fx.Int32(projection_size) + column * fx.Int32(projection_column_stride)
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

        padded_rows = i32_padded_rows
        if const_expr(device_padded_rows):
            valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
            padded_rows = fx.Int32(buffer_ops.buffer_load(valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32))

        if const_expr(device_padded_rows):
            for row_value in range(gpu.block_idx.x, padded_rows, gpu.grid_dim.x):
                row = fx.Int32(row_value)
                packed = fx.Int32(buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32))
                token = packed & fx.Int32(_TOKEN_MASK)
                if token < i32_tokens:
                    differentiate_row(row)
        else:
            differentiate_row(gpu.block_idx.x)

    @flyc.jit
    def launch(
        preactivation: fx.Tensor,
        da: fx.Tensor,
        dz: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        activation_derivative_kernel(
            preactivation,
            da,
            dz,
            sorted_token_ids,
            num_valid_ids,
            i32_tokens,
            i32_padded_rows,
        ).launch(
            grid=(i32_padded_rows, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=64)
def _compile_fused_activation_derivative_dscore_scale_dy(
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    interleaved_w1: bool,
    device_index: int,
    schedule_block_m: int = _COMPACT_W1_BM,
):
    """Fuse the post-dA row work for the BF16 SwiGLU state fast path.

    On entry ``da`` contains the unscaled, A16-materialized contraction
    ``q = dout @ W2`` and ``dy`` contains the gathered, unscaled ``dout``.
    Each workgroup owns a sorted route and performs three operations while the
    same row is resident:

    * multiply q by the FP32 route weight before applying the SwiGLU Jacobian;
    * reduce ``dot(q, activation)`` directly into route-order FP32 dscore;
    * scale dy in place to restore the established A16 dW2 input contract.

    A route is owned by exactly one workgroup, so dscore needs neither an
    initialization launch nor atomics.  This is intentionally separate from
    the general derivative kernel until bias and ragged-route epilogues are
    implemented.
    """

    del device_index
    elem_dtype = fx.BFloat16
    projection_size = 2 * intermediate_size
    projection_column_stride = 2 if interleaved_w1 else 1
    up_column_offset = 1 if interleaved_w1 else intermediate_size
    if schedule_block_m not in (_COMPACT_W1_BM, _LARGE_GROUPED_DX_BM):
        raise ValueError("fused derivative schedule_block_m must be 16 or 64")

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def fused_kernel(
        route_preactivation: fx.Tensor,
        activation: fx.Tensor,
        da: fx.Tensor,
        dy: fx.Tensor,
        dz: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        dtopk_weights: fx.Tensor,
        schedule: fx.Tensor,
        i32_tokens: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        zero_f32 = fx.Float32(0.0)
        fm_fast = arith.FastMathFlags.fast
        route_preact_rsrc = buffer_ops.create_buffer_resource(route_preactivation, max_size=True)
        activation_rsrc = buffer_ops.create_buffer_resource(activation, max_size=True)
        da_rsrc = buffer_ops.create_buffer_resource(da, max_size=True)
        dy_rsrc = buffer_ops.create_buffer_resource(dy, max_size=True)
        dz_rsrc = buffer_ops.create_buffer_resource(dz, max_size=True)
        weights_rsrc = buffer_ops.create_buffer_resource(sorted_weights, max_size=True)
        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        ds_rsrc = buffer_ops.create_buffer_resource(dtopk_weights, max_size=True)
        schedule_rsrc = buffer_ops.create_buffer_resource(schedule, max_size=True)
        lds = fx.SharedAllocator().allocate(_ScoreBackwardSharedStorage).peek()
        reduction = lds.reduction.view(fx.make_layout(_RED_SLOTS, 1))

        def wave_reduce_add(value):
            result = value
            with fx.fastmath(fm_fast):
                for shift_index in range_constexpr(int(math.log2(_WARP_SIZE))):
                    offset = _WARP_SIZE // (2 << shift_index)
                    result = result + gpu.shuffle_xor(result, offset, _WARP_SIZE)
            return result

        def process_row(row, token, slot):
            route_weight = fx.Float32(buffer_ops.buffer_load(weights_rsrc, row, vec_width=1, dtype=T.f32))
            thread_dot = zero_f32
            route_row = token * fx.Int32(topk) + slot
            route_base = route_row * fx.Int32(projection_size)
            sorted_base = row * fx.Int32(projection_size)
            for base in range_constexpr(0, intermediate_size, _BLOCK_THREADS):
                column = tid + fx.Int32(base)
                if column < fx.Int32(intermediate_size):
                    relative_gate = column * fx.Int32(projection_column_stride)
                    route_gate_offset = route_base + relative_gate
                    sorted_gate_offset = sorted_base + relative_gate
                    act_offset = row * fx.Int32(intermediate_size) + column
                    gate = buffer_ops.buffer_load(
                        route_preact_rsrc,
                        route_gate_offset,
                        vec_width=1,
                        dtype=elem_dtype,
                    ).extf(T.f32)
                    up = buffer_ops.buffer_load(
                        route_preact_rsrc,
                        route_gate_offset + fx.Int32(up_column_offset),
                        vec_width=1,
                        dtype=elem_dtype,
                    ).extf(T.f32)
                    activation_value = buffer_ops.buffer_load(
                        activation_rsrc,
                        act_offset,
                        vec_width=1,
                        dtype=elem_dtype,
                    ).extf(T.f32)
                    q = buffer_ops.buffer_load(da_rsrc, act_offset, vec_width=1, dtype=elem_dtype).extf(T.f32)
                    thread_dot = thread_dot + q * activation_value
                    dz_gate, dz_up = _activation_backward_f32(
                        gate,
                        up,
                        q * route_weight,
                        "swiglu",
                    )
                    buffer_ops.buffer_store(fx.Float32(dz_gate).to(elem_dtype), dz_rsrc, sorted_gate_offset)
                    buffer_ops.buffer_store(
                        fx.Float32(dz_up).to(elem_dtype),
                        dz_rsrc,
                        sorted_gate_offset + fx.Int32(up_column_offset),
                    )

            # dW2 retains the existing multiply-before-GEMM A16 boundary.
            for base in range_constexpr(0, hidden_size, _BLOCK_THREADS):
                column = tid + fx.Int32(base)
                if column < fx.Int32(hidden_size):
                    offset = row * fx.Int32(hidden_size) + column
                    dout_value = buffer_ops.buffer_load(dy_rsrc, offset, vec_width=1, dtype=elem_dtype).extf(T.f32)
                    buffer_ops.buffer_store(
                        fx.Float32(dout_value * route_weight).to(elem_dtype),
                        dy_rsrc,
                        offset,
                    )

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
                destination = token * fx.Int32(topk) + slot
                buffer_ops.buffer_store(reduced, ds_rsrc, destination)

        total_tiles = fx.Int32(
            buffer_ops.buffer_load(schedule_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32)
        )
        total_rows = total_tiles * fx.Int32(schedule_block_m)
        for task_value in range(gpu.block_idx.x, total_rows, gpu.grid_dim.x):
            task = fx.Int32(task_value)
            descriptor_index = task // fx.Int32(schedule_block_m)
            local_row = task % fx.Int32(schedule_block_m)
            descriptor = fx.Int32(
                buffer_ops.buffer_load(
                    schedule_rsrc,
                    descriptor_index + fx.Int32(1),
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            row = descriptor * fx.Int32(schedule_block_m) + local_row
            packed = fx.Int32(buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32))
            token = packed & fx.Int32(_TOKEN_MASK)
            slot = (packed >> fx.Int32(24)) & fx.Int32(0xFF)
            if (token < i32_tokens) & (slot < fx.Int32(topk)):
                process_row(row, token, slot)

    @flyc.jit
    def launch(
        route_preactivation: fx.Tensor,
        activation: fx.Tensor,
        da: fx.Tensor,
        dy: fx.Tensor,
        dz: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        dtopk_weights: fx.Tensor,
        schedule: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        fused_kernel(
            route_preactivation,
            activation,
            da,
            dy,
            dz,
            sorted_weights,
            sorted_token_ids,
            dtopk_weights,
            schedule,
            i32_tokens,
        ).launch(
            grid=(i32_grid, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_activation_derivative_from_forward_state(
    intermediate_size: int,
    topk: int,
    interleaved_w1: bool,
    device_index: int,
):
    """Differentiate SwiGLU directly from compact route-order forward state."""

    del device_index
    elem_dtype = fx.BFloat16
    projection_size = 2 * intermediate_size
    projection_column_stride = 2 if interleaved_w1 else 1
    up_column_offset = 1 if interleaved_w1 else intermediate_size
    compact_block_m = _COMPACT_W1_BM

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def derivative_from_state_kernel(
        route_preactivation: fx.Tensor,
        da: fx.Tensor,
        dz: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        schedule: fx.Tensor,
        i32_tokens: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        route_preact_rsrc = buffer_ops.create_buffer_resource(route_preactivation, max_size=True)
        da_rsrc = buffer_ops.create_buffer_resource(da, max_size=True)
        dz_rsrc = buffer_ops.create_buffer_resource(dz, max_size=True)
        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)

        def differentiate_real_row(row):
            packed = fx.Int32(buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32))
            token = packed & fx.Int32(_TOKEN_MASK)
            slot = (packed >> fx.Int32(24)) & fx.Int32(0xFF)
            if (token < i32_tokens) & (slot < fx.Int32(topk)):
                route_row = token * fx.Int32(topk) + slot
                route_base = route_row * fx.Int32(projection_size)
                sorted_base = row * fx.Int32(projection_size)
                activation_base = row * fx.Int32(intermediate_size)
                for base in range_constexpr(0, intermediate_size, _BLOCK_THREADS):
                    column = tid + fx.Int32(base)
                    if column < fx.Int32(intermediate_size):
                        relative_gate = column * fx.Int32(projection_column_stride)
                        gate = buffer_ops.buffer_load(
                            route_preact_rsrc,
                            route_base + relative_gate,
                            vec_width=1,
                            dtype=elem_dtype,
                        ).extf(T.f32)
                        up = buffer_ops.buffer_load(
                            route_preact_rsrc,
                            route_base + relative_gate + fx.Int32(up_column_offset),
                            vec_width=1,
                            dtype=elem_dtype,
                        ).extf(T.f32)
                        da_value = buffer_ops.buffer_load(
                            da_rsrc,
                            activation_base + column,
                            vec_width=1,
                            dtype=elem_dtype,
                        ).extf(T.f32)
                        dz_gate, dz_up = _activation_backward_f32(
                            gate,
                            up,
                            da_value,
                            "swiglu",
                        )
                        buffer_ops.buffer_store(
                            fx.Float32(dz_gate).to(elem_dtype),
                            dz_rsrc,
                            sorted_base + relative_gate,
                        )
                        buffer_ops.buffer_store(
                            fx.Float32(dz_up).to(elem_dtype),
                            dz_rsrc,
                            sorted_base + relative_gate + fx.Int32(up_column_offset),
                        )

        schedule_rsrc = buffer_ops.create_buffer_resource(schedule, max_size=True)
        total_tiles = fx.Int32(
            buffer_ops.buffer_load(schedule_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32)
        )
        total_rows = total_tiles * fx.Int32(compact_block_m)
        for task_value in range(gpu.block_idx.x, total_rows, gpu.grid_dim.x):
            task = fx.Int32(task_value)
            descriptor_index = task // fx.Int32(compact_block_m)
            local_row = task % fx.Int32(compact_block_m)
            descriptor = fx.Int32(
                buffer_ops.buffer_load(
                    schedule_rsrc,
                    descriptor_index + fx.Int32(1),
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            differentiate_real_row(descriptor * fx.Int32(compact_block_m) + local_row)

    @flyc.jit
    def launch(
        route_preactivation: fx.Tensor,
        da: fx.Tensor,
        dz: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        schedule: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        derivative_from_state_kernel(
            route_preactivation,
            da,
            dz,
            sorted_token_ids,
            schedule,
            i32_tokens,
        ).launch(
            grid=(i32_grid, 1, 1),
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
def _compile_score_backward(
    hidden_size: int,
    topk: int,
    compute_dtype: str,
    device_index: int,
    device_padded_rows: bool = False,
    token_major_dout: bool = False,
):
    """Compile ``ds = dot(dout, materialized_down_projection)``."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def score_backward_kernel(
        dout_sorted: fx.Tensor,
        projection: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        dtopk_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        zero_f32 = fx.Float32(0.0)
        fm_fast = arith.FastMathFlags.fast

        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        dout_rsrc = buffer_ops.create_buffer_resource(dout_sorted, max_size=True)
        projection_rsrc = buffer_ops.create_buffer_resource(projection, max_size=True)
        ds_rsrc = buffer_ops.create_buffer_resource(dtopk_weights, max_size=True)
        lds = fx.SharedAllocator().allocate(_ScoreBackwardSharedStorage).peek()
        reduction = lds.reduction.view(fx.make_layout(_RED_SLOTS, 1))

        def wave_reduce_add(value):
            result = value
            with fx.fastmath(fm_fast):
                for shift_index in range_constexpr(int(math.log2(_WARP_SIZE))):
                    offset = _WARP_SIZE // (2 << shift_index)
                    result = result + gpu.shuffle_xor(result, offset, _WARP_SIZE)
            return result

        def reduce_row(row, token, slot):
            thread_dot = zero_f32
            for base in range_constexpr(0, hidden_size, _BLOCK_THREADS):
                column = tid + fx.Int32(base)
                if column < fx.Int32(hidden_size):
                    projection_offset = row * fx.Int32(hidden_size) + column
                    dout_row = token if const_expr(token_major_dout) else row
                    dout_offset = dout_row * fx.Int32(hidden_size) + column
                    dout_value = buffer_ops.buffer_load(
                        dout_rsrc,
                        dout_offset,
                        vec_width=1,
                        dtype=elem_dtype,
                    ).extf(T.f32)
                    projected = buffer_ops.buffer_load(
                        projection_rsrc,
                        projection_offset,
                        vec_width=1,
                        dtype=elem_dtype,
                    ).extf(T.f32)
                    thread_dot = thread_dot + dout_value * projected

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
                destination = token * fx.Int32(topk) + slot
                buffer_ops.buffer_store(reduced, ds_rsrc, destination)

        padded_rows = i32_padded_rows
        if const_expr(device_padded_rows):
            valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
            padded_rows = fx.Int32(buffer_ops.buffer_load(valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32))

        if const_expr(device_padded_rows):
            for row_value in range(gpu.block_idx.x, padded_rows, gpu.grid_dim.x):
                row = fx.Int32(row_value)
                packed = fx.Int32(buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32))
                token = packed & fx.Int32(_TOKEN_MASK)
                slot = packed >> fx.Int32(24)
                valid_route = token < i32_tokens
                if valid_route:
                    reduce_row(row, token, slot)
        else:
            row = gpu.block_idx.x
            packed = fx.Int32(buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32))
            token = packed & fx.Int32(_TOKEN_MASK)
            slot = packed >> fx.Int32(24)
            valid_route = token < i32_tokens
            # Legacy row grids intentionally retain their original
            # execute-then-suppress behavior for padded routes.
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
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        score_backward_kernel(
            dout_sorted,
            projection,
            sorted_token_ids,
            dtopk_weights,
            num_valid_ids,
            i32_tokens,
            i32_padded_rows,
        ).launch(
            grid=(i32_padded_rows, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_route_score_backward(hidden_size: int, compute_dtype: str, device_index: int):
    """Compile flat-route ``ds = dot(dout, down_projection)`` scatter."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def score_backward_kernel(
        dout_sorted: fx.Tensor,
        projection: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_route_ids: fx.Tensor,
        droute_weights: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_routes: fx.Int32,
    ):
        row = gpu.block_idx.x
        tid = gpu.thread_idx.x
        zero_f32 = fx.Float32(0.0)
        fm_fast = arith.FastMathFlags.fast

        token_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        route_rsrc = buffer_ops.create_buffer_resource(sorted_route_ids, max_size=True)
        dout_rsrc = buffer_ops.create_buffer_resource(dout_sorted, max_size=True)
        projection_rsrc = buffer_ops.create_buffer_resource(projection, max_size=True)
        ds_rsrc = buffer_ops.create_buffer_resource(droute_weights, max_size=True)
        token = fx.Int32(buffer_ops.buffer_load(token_rsrc, row, vec_width=1, dtype=T.i32))
        route = fx.Int32(buffer_ops.buffer_load(route_rsrc, row, vec_width=1, dtype=T.i32))
        valid_route = (token >= fx.Int32(0)) & (token < i32_tokens) & (route >= fx.Int32(0)) & (route < i32_routes)

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
                buffer_ops.buffer_store(reduced, ds_rsrc, route)

    @flyc.jit
    def launch(
        dout_sorted: fx.Tensor,
        projection: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_route_ids: fx.Tensor,
        droute_weights: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_routes: fx.Int32,
        i32_padded_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        score_backward_kernel(
            dout_sorted,
            projection,
            sorted_token_ids,
            sorted_route_ids,
            droute_weights,
            i32_tokens,
            i32_routes,
        ).launch(
            grid=(i32_padded_rows, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_ragged_dx_reduction(hidden_size: int, compute_dtype: str, device_index: int):
    """Compile FP32 atomic accumulation of variable-count route gradients."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def clear_kernel(dx_accum: fx.Tensor, i32_elements: fx.Int32):
        index = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        if index < i32_elements:
            accum_rsrc = buffer_ops.create_buffer_resource(dx_accum, max_size=True)
            buffer_ops.buffer_store(fx.Float32(0.0), accum_rsrc, index)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def scatter_kernel(
        dx_sorted: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        dx_accum: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
    ):
        index = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        total = i32_padded_rows * fx.Int32(hidden_size)
        if index < total:
            row = index // fx.Int32(hidden_size)
            column = index % fx.Int32(hidden_size)
            token_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
            dx_rsrc = buffer_ops.create_buffer_resource(dx_sorted, max_size=True)
            token = fx.Int32(buffer_ops.buffer_load(token_rsrc, row, vec_width=1, dtype=T.i32))
            if (token >= fx.Int32(0)) & (token < i32_tokens):
                value = buffer_ops.buffer_load(dx_rsrc, index, vec_width=1, dtype=elem_dtype).extf(T.f32)
                destination = token * fx.Int32(hidden_size) + column
                atomic_add(dx_accum, destination, value, dtype_bytes=4)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def finalize_kernel(dx_accum: fx.Tensor, dx: fx.Tensor, i32_elements: fx.Int32):
        index = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        if index < i32_elements:
            accum_rsrc = buffer_ops.create_buffer_resource(dx_accum, max_size=True)
            dx_rsrc = buffer_ops.create_buffer_resource(dx, max_size=True)
            value = buffer_ops.buffer_load(accum_rsrc, index, vec_width=1, dtype=T.f32)
            buffer_ops.buffer_store(fx.Float32(value).to(elem_dtype), dx_rsrc, index)

    @flyc.jit
    def launch(
        dx_sorted: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        dx_accum: fx.Tensor,
        dx: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
        i32_clear_grid: fx.Int32,
        i32_scatter_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        elements = i32_tokens * fx.Int32(hidden_size)
        clear_kernel(dx_accum, elements).launch(
            grid=(i32_clear_grid, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )
        scatter_kernel(
            dx_sorted,
            sorted_token_ids,
            dx_accum,
            i32_tokens,
            i32_padded_rows,
        ).launch(
            grid=(i32_scatter_grid, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )
        finalize_kernel(dx_accum, dx, elements).launch(
            grid=(i32_clear_grid, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_unsort(
    hidden_size: int,
    topk: int,
    compute_dtype: str,
    device_index: int,
    device_padded_rows: bool = False,
):
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
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
    ):
        index = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        stride = gpu.grid_dim.x * fx.Int32(_BLOCK_THREADS)
        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        source_rsrc = buffer_ops.create_buffer_resource(dx_sorted, max_size=True)
        destination_rsrc = buffer_ops.create_buffer_resource(dx_routes, max_size=True)

        def unsort_vector(index):
            row = index // fx.Int32(vectors_per_row)
            column = (index % fx.Int32(vectors_per_row)) * fx.Int32(vector_width)
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

        padded_rows = i32_padded_rows
        if const_expr(device_padded_rows):
            valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
            padded_rows = fx.Int32(buffer_ops.buffer_load(valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32))
        total = padded_rows * fx.Int32(vectors_per_row)
        if const_expr(device_padded_rows):
            for vector_index in range(index, total, stride):
                unsort_vector(fx.Int32(vector_index))
        else:
            if index < total:
                unsort_vector(index)

    @flyc.jit
    def launch(
        dx_sorted: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        dx_routes: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        unsort_kernel(
            dx_sorted,
            sorted_token_ids,
            dx_routes,
            num_valid_ids,
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
    interleaved_w1: bool,
) -> tuple[int, int, int, int]:
    dtype_by_name = {"bf16": torch.bfloat16, "fp16": torch.float16}
    if config.compute_dtype not in dtype_by_name:
        raise ValueError("sonic_moe_backward supports compute_dtype='bf16' or 'fp16', " f"got {config.compute_dtype!r}")
    expected_dtype = dtype_by_name[config.compute_dtype]
    if config.activation not in _SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"sonic_moe_backward does not support activation={config.activation!r}; "
            f"expected one of {sorted(_SUPPORTED_ACTIVATIONS)}"
        )
    if not isinstance(interleaved_w1, bool):
        raise TypeError(f"interleaved_w1 must be bool, got {type(interleaved_w1).__name__}")
    if interleaved_w1 and config.activation not in _GLU_ACTIVATIONS:
        raise ValueError(
            "interleaved_w1 is valid only for GLU activations "
            f"{sorted(_GLU_ACTIVATIONS)}, got activation={config.activation!r}"
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
    max_padded, _ = _max_padded_routes(tokens, num_experts, topk, _BACKWARD_SORT_UNIT)
    if max_padded > _MAX_SIGNED_I32:
        raise ValueError(f"padded route count exceeds the signed 32-bit limit, got {max_padded}")
    if max_padded * max(hidden_size, projection_size) * 2 > _MAX_BUFFER_BYTE_OFFSET:
        raise ValueError("fixed-K backward workspace exceeds the 32-bit buffer offset limit")
    mesh_stride = ((tokens + _BACKWARD_SORT_UNIT - 1) // _BACKWARD_SORT_UNIT) * _BACKWARD_SORT_UNIT
    if num_experts * mesh_stride > _MAX_SIGNED_I32:
        raise ValueError("fixed-K backward sorting mesh exceeds the signed 32-bit index limit")
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


def _validate_forward_state(
    forward_state: object,
    hidden_states: torch.Tensor,
    config: "SonicMoEConfig",
    interleaved_w1: bool,
    has_bias: bool,
) -> tuple[torch.Tensor, int, torch.cuda.Event]:
    """Validate the fixed-K training-forward state consumed by backward.

    This deliberately uses a structural contract instead of importing the
    forward state's concrete class.  Forward and backward can therefore
    evolve independently while structurally malformed or incompatible state
    is rejected instead of silently falling back to W1 recomputation.
    """

    field_names = (
        "preactivation",
        "tokens",
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "top_k",
        "activation",
        "compute_dtype",
        "interleaved_w1",
        "has_bias",
        "producer_stream",
        "ready_event",
    )
    values: dict[str, object] = {}
    missing: list[str] = []
    for name in field_names:
        try:
            values[name] = getattr(forward_state, name)
        except AttributeError:
            missing.append(name)
    if missing:
        raise ValueError("forward_state is missing required field(s): " + ", ".join(missing))

    tokens = int(hidden_states.shape[0])
    expected_ints = {
        "tokens": tokens,
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.intermediate_size),
        "num_experts": int(config.num_experts),
        "top_k": int(config.top_k),
    }
    for name, expected in expected_ints.items():
        value = values[name]
        if type(value) is not int:
            raise TypeError(f"forward_state.{name} must be int, got {type(value).__name__}")
        if value != expected:
            raise ValueError(f"forward_state.{name} must equal {expected}, got {value}")

    expected_strings = {
        "activation": str(config.activation),
        "compute_dtype": str(config.compute_dtype),
    }
    for name, expected in expected_strings.items():
        value = values[name]
        if type(value) is not str:
            raise TypeError(f"forward_state.{name} must be str, got {type(value).__name__}")
        if value != expected:
            raise ValueError(f"forward_state.{name} must equal {expected!r}, got {value!r}")
    if config.activation != "swiglu" or config.compute_dtype != "bf16":
        raise ValueError("forward_state reuse currently supports only dense BF16 SwiGLU fixed-K backward")

    state_interleaved = values["interleaved_w1"]
    if type(state_interleaved) is not bool:
        raise TypeError("forward_state.interleaved_w1 must be bool, got " f"{type(state_interleaved).__name__}")
    if state_interleaved != interleaved_w1:
        raise ValueError(
            "forward_state.interleaved_w1 must match the backward call, got "
            f"{state_interleaved} and {interleaved_w1}"
        )
    state_has_bias = values["has_bias"]
    if type(state_has_bias) is not bool:
        raise TypeError("forward_state.has_bias must be bool, got " f"{type(state_has_bias).__name__}")
    if state_has_bias != has_bias:
        raise ValueError("forward_state.has_bias must match the backward call, got " f"{state_has_bias} and {has_bias}")

    preactivation = values["preactivation"]
    if not isinstance(preactivation, torch.Tensor):
        raise TypeError("forward_state.preactivation must be a torch.Tensor, got " f"{type(preactivation).__name__}")
    expected_shape = (
        tokens,
        int(config.top_k),
        2 * int(config.intermediate_size),
    )
    if tuple(preactivation.shape) != expected_shape:
        raise ValueError(
            f"forward_state.preactivation must have shape {expected_shape}, " f"got {tuple(preactivation.shape)}"
        )
    if not preactivation.is_cuda or preactivation.device != hidden_states.device:
        raise ValueError("forward_state.preactivation must be on the same ROCm device as hidden_states")
    if preactivation.dtype != torch.bfloat16:
        raise TypeError("forward_state.preactivation must be torch.bfloat16, got " f"{preactivation.dtype}")
    preactivation_bytes = preactivation.numel() * preactivation.element_size()
    if preactivation_bytes > _MAX_SIGNED_I32:
        raise ValueError(
            "forward_state.preactivation byte span exceeds the signed 32-bit "
            f"buffer limit, got {preactivation_bytes}"
        )
    if not preactivation.is_contiguous():
        raise ValueError("forward_state.preactivation must be contiguous")

    producer_stream = values["producer_stream"]
    if type(producer_stream) is not int:
        raise TypeError("forward_state.producer_stream must be int, got " f"{type(producer_stream).__name__}")
    if producer_stream < 0:
        raise ValueError(f"forward_state.producer_stream must be non-negative, got {producer_stream}")

    ready_event = values["ready_event"]
    if not isinstance(ready_event, torch.cuda.Event):
        raise TypeError("forward_state.ready_event must be torch.cuda.Event, got " f"{type(ready_event).__name__}")
    event_device = ready_event.device
    if event_device is None:
        raise ValueError("forward_state.ready_event must already be recorded")
    if torch.device(event_device) != hidden_states.device:
        raise ValueError("forward_state.ready_event must be recorded on the same ROCm device " "as hidden_states")

    return preactivation, producer_stream, ready_event


def _validate_backward_route_inputs(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    token_indices: torch.Tensor,
    expert_indices: torch.Tensor,
    route_weights: torch.Tensor,
    grad_output: torch.Tensor,
    config: "SonicMoEConfig",
    b1: torch.Tensor | None,
    b2: torch.Tensor | None,
    interleaved_w1: bool,
) -> tuple[int, int, int, int, int]:
    dtype_by_name = {"bf16": torch.bfloat16, "fp16": torch.float16}
    if config.compute_dtype not in dtype_by_name:
        raise ValueError(
            "sonic_moe_backward_routes supports compute_dtype='bf16' or 'fp16', " f"got {config.compute_dtype!r}"
        )
    expected_dtype = dtype_by_name[config.compute_dtype]
    if config.activation not in _SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"sonic_moe_backward_routes does not support activation={config.activation!r}; "
            f"expected one of {sorted(_SUPPORTED_ACTIVATIONS)}"
        )
    if not isinstance(interleaved_w1, bool):
        raise TypeError(f"interleaved_w1 must be bool, got {type(interleaved_w1).__name__}")
    if interleaved_w1 and config.activation not in _GLU_ACTIVATIONS:
        raise ValueError(
            "interleaved_w1 is valid only for GLU activations "
            f"{sorted(_GLU_ACTIVATIONS)}, got activation={config.activation!r}"
        )
    if token_indices.ndim != 1 or expert_indices.ndim != 1 or route_weights.ndim != 1:
        raise ValueError("token_indices, expert_indices, and route_weights must be one-dimensional")

    tokens = int(hidden_states.shape[0]) if hidden_states.ndim == 2 else -1
    routes = int(route_weights.numel())
    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    num_experts = int(config.num_experts)
    projection_size = intermediate_size * (2 if config.activation in _GLU_ACTIVATIONS else 1)
    if (b1 is None) != (b2 is None):
        raise ValueError("b1 and b2 must both be None or both be tensors")
    expected = {
        "hidden_states": (tokens, hidden_size),
        "w1": (num_experts, projection_size, hidden_size),
        "w2": (num_experts, hidden_size, intermediate_size),
        "token_indices": (routes,),
        "expert_indices": (routes,),
        "route_weights": (routes,),
        "grad_output": (tokens, hidden_size),
    }
    tensors = {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "token_indices": token_indices,
        "expert_indices": expert_indices,
        "route_weights": route_weights,
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
    if routes > _MAX_SIGNED_I32:
        raise ValueError(f"route count exceeds the signed 32-bit limit, got {routes}")
    max_padded, _ = _max_padded_flat_routes(routes, num_experts, _BACKWARD_SORT_UNIT)
    if max_padded > _MAX_SIGNED_I32:
        raise ValueError(f"padded route count exceeds the signed 32-bit limit, got {max_padded}")
    if max_padded * max(hidden_size, projection_size) * 2 > _MAX_BUFFER_BYTE_OFFSET:
        raise ValueError("ragged backward workspace exceeds the 32-bit buffer offset limit")
    if tokens * hidden_size > _MAX_SIGNED_I32 or tokens * hidden_size * 4 > _MAX_BUFFER_BYTE_OFFSET:
        raise ValueError("ragged backward FP32 input-gradient workspace exceeds the 32-bit buffer offset limit")
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
    if token_indices.dtype != torch.int32 or expert_indices.dtype != torch.int32:
        raise TypeError(
            "token_indices and expert_indices must be int32, got " f"{token_indices.dtype}/{expert_indices.dtype}"
        )
    if route_weights.dtype != torch.float32:
        raise TypeError(f"route_weights must be float32, got {route_weights.dtype}")
    if hidden_size % 64 != 0 or intermediate_size % 64 != 0:
        raise ValueError("hidden_size and intermediate_size must be multiples of 64")
    return tokens, hidden_size, intermediate_size, num_experts, routes


def _ptr(tensor: torch.Tensor):
    return flyc.from_c_void_p(fx.Uint8, tensor.data_ptr())


def _sonic_moe_backward_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
    grad_output: torch.Tensor,
    config: "SonicMoEConfig",
    *,
    token_indices: torch.Tensor | None,
    dimensions: tuple[int, int, int, int],
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
    interleaved_w1: bool = False,
    forward_state_data: tuple[torch.Tensor, int, torch.cuda.Event] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Shared sorted-expert implementation for fixed-K and flat routes."""

    tokens, hidden_size, intermediate_size, num_experts = dimensions
    flat_routes = token_indices is not None
    routes = int(route_weights.numel())
    topk = int(config.top_k)
    compute_dtype = str(config.compute_dtype)
    activation_name = str(config.activation)
    sort_unit = _BACKWARD_SORT_UNIT
    projection_size = intermediate_size * (2 if activation_name in _GLU_ACTIVATIONS else 1)
    has_bias = b1 is not None
    reuse_forward_preactivation = forward_state_data is not None
    use_grouped_w1 = _use_grouped_w1_recompute(
        compute_dtype=compute_dtype,
        activation=activation_name,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        tokens=tokens,
        routes=routes,
        flat_routes=flat_routes,
    )
    use_grouped_w2 = _use_grouped_w2_recompute(
        compute_dtype=compute_dtype,
        activation=activation_name,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        tokens=tokens,
        routes=routes,
        flat_routes=flat_routes,
    )
    use_grouped_dw2 = _use_grouped_dw2(
        compute_dtype=compute_dtype,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    use_grouped_da = _use_grouped_da(
        compute_dtype=compute_dtype,
        activation=activation_name,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        tokens=tokens,
        routes=routes,
        flat_routes=flat_routes,
    )
    use_grouped_dw1 = _use_grouped_dw1(
        compute_dtype=compute_dtype,
        activation=activation_name,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        tokens=tokens,
        routes=routes,
        flat_routes=flat_routes,
    )
    grouped_w1_bm, grouped_w1_bn, _, _, compact_w1_grid = _grouped_w1_tuning(
        tokens=tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    use_compact_w1 = use_grouped_w1 and compact_w1_grid
    use_grouped_dx = _use_grouped_dx(
        compute_dtype=compute_dtype,
        activation=activation_name,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        tokens=tokens,
        num_experts=num_experts,
        topk=topk,
        flat_routes=flat_routes,
        compact_w1=use_compact_w1,
    )
    direct_grouped_dx_routes = _use_direct_grouped_dx_routes(
        use_grouped_dx=use_grouped_dx,
        flat_routes=flat_routes,
    )
    use_large_grouped_dx = use_grouped_dx and _use_large_grouped_dx_descriptor_queue(
        tokens=tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
        flat_routes=flat_routes,
    )
    use_hostless_grouped = _use_hostless_grouped_backward(
        flat_routes=flat_routes,
        has_bias=has_bias,
        tokens=tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
        reuse_forward_preactivation=reuse_forward_preactivation,
        use_large_grouped_dx=use_large_grouped_dx,
        use_grouped_w1=use_grouped_w1,
        use_grouped_w2=use_grouped_w2,
        use_grouped_dw2=use_grouped_dw2,
        use_grouped_da=use_grouped_da,
        use_grouped_dw1=use_grouped_dw1,
        use_grouped_dx=use_grouped_dx,
    )
    # Exact-row state preparation reuses either the short BM16 queue or the
    # production T4096 BM64 dX queue.  Decode has only ~18 us of gather/prepare
    # work and is faster on the original one-row kernels.
    use_fused_forward_state_prepare = (
        reuse_forward_preactivation
        and not has_bias
        and ((use_hostless_grouped and use_compact_w1) or use_large_grouped_dx)
    )
    use_direct_grouped_dw1_rhs = _use_direct_grouped_dw1_rhs(
        reuse_forward_preactivation=reuse_forward_preactivation,
        use_fused_forward_state_prepare=use_fused_forward_state_prepare,
        use_grouped_dw1=use_grouped_dw1,
        flat_routes=flat_routes,
        has_bias=has_bias,
        compute_dtype=compute_dtype,
        activation=activation_name,
        tokens=tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
    )
    use_fused_da_dscore = _use_fused_da_dscore(
        reuse_forward_preactivation=reuse_forward_preactivation,
        use_hostless_grouped=use_hostless_grouped,
        use_compact_w1=use_compact_w1,
        use_large_grouped_dx=use_large_grouped_dx,
        flat_routes=flat_routes,
        has_bias=has_bias,
        compute_dtype=compute_dtype,
        activation=activation_name,
    )
    # If even the maximum possible active set falls below the measured
    # selective-clear crossover, a normal dense memset is unconditionally the
    # best choice.  This route-count test is host-known and distribution
    # independent; all ambiguous hostless cases select on device later.
    hostless_dense_weight_zero = use_hostless_grouped and routes * _INACTIVE_WEIGHT_GRAD_ZERO_ACTIVE_RATIO < num_experts
    device = hidden_states.device
    device_index = device.index or 0
    with torch.cuda.device(device):
        arch = get_rocm_arch()
    if not str(arch).startswith("gfx95"):
        raise RuntimeError(f"SonicMoE backward requires gfx95*, got {arch!r}")
    if flat_routes and routes == 0:
        result = (
            torch.zeros_like(hidden_states, memory_format=torch.contiguous_format),
            torch.zeros_like(w1, memory_format=torch.contiguous_format),
            torch.zeros_like(w2, memory_format=torch.contiguous_format),
            torch.empty_like(route_weights, memory_format=torch.contiguous_format),
        )
        if has_bias:
            return (
                *result,
                torch.zeros_like(b1, memory_format=torch.contiguous_format),
                torch.zeros_like(b2, memory_format=torch.contiguous_format),
            )
        return result
    if flat_routes:
        max_padded, max_blocks = _max_padded_flat_routes(routes, num_experts, sort_unit)
        workspace_elements = num_experts
    else:
        max_padded, max_blocks = _max_padded_routes(tokens, num_experts, topk, sort_unit)
        workspace_elements = moe_sorting_get_workspace_size(
            tokens,
            num_experts,
            topk,
            unit_size=sort_unit,
        )

    # Backward owns every buffer: no forward LRU scratch is retained or read.
    sorted_token_ids = torch.empty(max_padded, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(max_padded, dtype=torch.float32, device=device)
    sorted_route_ids = torch.empty(max_padded, dtype=torch.int32, device=device) if flat_routes else None
    sorted_expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    sorting_workspace = (
        torch.empty(workspace_elements, dtype=torch.int32, device=device) if workspace_elements else None
    )
    sorter_dummy = torch.empty(4, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(num_experts, dtype=torch.int32, device=device)
    active_expert_capacity = active_expert_descriptor_capacity(routes, num_experts)
    # A compact descriptor builder can emit this queue in its existing two
    # launches.  Other routing regimes select metadata-direct or standalone
    # construction after the already-required frequency readback below.
    active_expert_storage = (
        torch.empty(
            active_expert_queue_elements(routes, num_experts),
            dtype=torch.int32,
            device=device,
        )
        if (use_grouped_dw1 or use_grouped_dw2) and (use_compact_w1 or use_large_grouped_dx)
        else None
    )
    if use_compact_w1:
        if flat_routes:
            compact_w1_bound = ragged_compact_m_tile_descriptor_upper_bound(
                routes,
                num_experts,
                grouped_w1_bm,
            )
        else:
            compact_w1_bound = fixed_compact_m_tile_descriptor_upper_bound(
                tokens,
                num_experts,
                topk,
                grouped_w1_bm,
            )
        # Entry zero stores the device-produced live count.  Keeping the count
        # at a fixed offset lets one compiled GEMM specialization serve every
        # host-known descriptor-capacity bound.
        compact_w1_storage = torch.empty(compact_w1_bound + 1, dtype=torch.int32, device=device)
        compact_w1_total = compact_w1_storage[:1]
        compact_w1_descriptors = compact_w1_storage[1:]
    else:
        compact_w1_bound = 0
        compact_w1_storage = None
        compact_w1_descriptors = None
        compact_w1_total = None
    if use_large_grouped_dx:
        large_dx_bound = fixed_compact_m_tile_descriptor_upper_bound(
            tokens,
            num_experts,
            topk,
            _LARGE_GROUPED_DX_BM,
        )
        large_dx_storage = torch.empty(large_dx_bound + 1, dtype=torch.int32, device=device)
        large_dx_total = large_dx_storage[:1]
        large_dx_descriptors = large_dx_storage[1:]
    else:
        large_dx_bound = 0
        large_dx_storage = None
        large_dx_total = None
        large_dx_descriptors = None
    if use_fused_forward_state_prepare:
        if use_large_grouped_dx:
            assert large_dx_storage is not None
            state_row_schedule = large_dx_storage
            state_schedule_block_m = _LARGE_GROUPED_DX_BM
            state_schedule_bound = large_dx_bound
        else:
            assert compact_w1_storage is not None
            state_row_schedule = compact_w1_storage
            state_schedule_block_m = _COMPACT_W1_BM
            state_schedule_bound = compact_w1_bound
    else:
        state_row_schedule = None
        state_schedule_block_m = 0
        state_schedule_bound = 0
    sorted_hidden_shape = (max_padded, hidden_size)
    x_sorted = (
        None
        if use_direct_grouped_dw1_rhs
        else torch.empty(sorted_hidden_shape, dtype=hidden_states.dtype, device=device)
    )
    # The retained-state hostless path reads token-major grad_output directly
    # in both dy preparation and dscore, avoiding a large padded sorted copy.
    dout_sorted = (
        None
        if use_fused_forward_state_prepare
        else torch.empty(sorted_hidden_shape, dtype=hidden_states.dtype, device=device)
    )
    dy = torch.empty(sorted_hidden_shape, dtype=hidden_states.dtype, device=device)
    # The fused state path derives route-score gradients from q=dout@W2 and
    # therefore does not materialize the forward down projection.  Fallbacks
    # retain their original padding initialization contract.
    projection = (
        None
        if use_fused_da_dscore
        else (torch.empty_like(dy) if use_hostless_grouped or not use_grouped_w2 else torch.zeros_like(dy))
    )
    # Grouped W1 writes ceil(real_rows/BM)*BM rows instead of every
    # SORTED_BM-padded row.  Zero-initialize the untouched suffix: gather makes
    # padded x/dout zero, so its dy/da/dz and therefore dW/db contributions
    # remain exactly zero while all activation inputs stay finite.
    # The same fast path differentiates directly from compact route-order
    # state, so it does not allocate or write a padded sorted preactivation.
    preactivation = (
        None
        if use_fused_forward_state_prepare
        else torch.empty(
            (max_padded, projection_size),
            dtype=hidden_states.dtype,
            device=device,
        )
    )
    if (
        preactivation is not None
        and not reuse_forward_preactivation
        and use_grouped_w1
        and not use_hostless_grouped
    ):
        preactivation.zero_()
    activation = torch.empty((max_padded, intermediate_size), dtype=hidden_states.dtype, device=device)
    # Grouped dA writes real expert rows only.  Legacy derivatives consume full
    # sorter-padded segments and therefore need zero tails; exact-row fused
    # derivatives skip those rows and can leave dA uninitialized there.
    da = (
        torch.empty_like(activation)
        if use_fused_da_dscore or use_hostless_grouped or not use_grouped_da
        else torch.zeros_like(activation)
    )
    # Exact-row derivatives skip sentinel rows, so zeroing dZ supplies the
    # at-most-(BM-1) tails consumed by compact BM16 or production BM64 dX.
    dz_factory = torch.zeros if use_hostless_grouped or use_fused_da_dscore else torch.empty
    dz = dz_factory(
        (max_padded, projection_size),
        dtype=hidden_states.dtype,
        device=device,
    )
    dx_sorted = None if direct_grouped_dx_routes else torch.empty_like(dy)
    dx_routes = (
        None if flat_routes else torch.empty((tokens, topk, hidden_size), dtype=hidden_states.dtype, device=device)
    )
    dx_accum = torch.empty((tokens, hidden_size), dtype=torch.float32, device=device) if flat_routes else None

    dx = torch.empty_like(hidden_states, memory_format=torch.contiguous_format)
    # The grouped pair is initialized from routing metadata below, before its
    # first contraction; every other path retains eager zero initialization.
    grouped_weight_grads = use_grouped_dw1 and use_grouped_dw2
    weight_grad_factory = (
        torch.zeros_like if not grouped_weight_grads or hostless_dense_weight_zero else torch.empty_like
    )
    dw1 = weight_grad_factory(w1, memory_format=torch.contiguous_format)
    dw2 = weight_grad_factory(w2, memory_format=torch.contiguous_format)
    droute_weights = torch.empty_like(route_weights, memory_format=torch.contiguous_format)
    db1 = torch.empty_like(b1, memory_format=torch.contiguous_format) if b1 is not None else None
    db2 = torch.empty_like(b2, memory_format=torch.contiguous_format) if b2 is not None else None

    # Custom kernels consume raw storage only; detach keeps DLPack conversion
    # valid when this function is called from a torch.autograd.Function.
    x_arg = hidden_states.detach()
    w1_arg = w1.detach()
    w2_arg = w2.detach()
    ids_arg = expert_ids.detach()
    weights_arg = route_weights.detach()
    token_arg = token_indices.detach() if token_indices is not None else None
    dout_arg = grad_output.detach()
    b1_arg = b1.detach() if b1 is not None else None
    b2_arg = b2.detach() if b2 is not None else None

    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        if has_bias:
            clear_bias_gradients = _compile_bias_gradient_clear(
                projection_size,
                hidden_size,
                num_experts,
                compute_dtype,
                device_index,
            )
            _run_compiled(clear_bias_gradients, db1, db2, stream)

        if flat_routes:
            assert token_arg is not None
            assert sorted_route_ids is not None
            assert sorting_workspace is not None
            moe_ragged_sorting_flydsl(
                token_arg,
                ids_arg,
                weights_arg,
                expert_frequency,
                sorting_workspace,
                sorted_token_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                sorter_dummy,
                num_experts,
                tokens=tokens,
                max_padded_routes=max_padded,
                unit_size=sort_unit,
                sorted_route_ids=sorted_route_ids,
            )
        else:
            route_grid = max(1, (routes + _BLOCK_THREADS - 1) // _BLOCK_THREADS)
            histogram = _compile_expert_histogram(num_experts, device_index)
            _run_compiled(histogram, ids_arg, expert_frequency, routes, route_grid, stream)
            moe_sorting_flydsl(
                ids_arg,
                weights_arg,
                sorted_token_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                sorter_dummy,
                num_experts,
                unit_size=sort_unit,
                workspace=sorting_workspace,
            )

        if use_large_grouped_dx:
            assert large_dx_descriptors is not None
            assert large_dx_total is not None
            build_compact_m_tile_descriptors(
                expert_frequency,
                sorted_expert_ids,
                num_valid_ids,
                large_dx_descriptors,
                large_dx_total,
                block_m=_LARGE_GROUPED_DX_BM,
                sorted_block_m=sort_unit,
                descriptor_capacity=large_dx_bound,
                active_expert_storage=active_expert_storage,
                active_expert_capacity=(active_expert_capacity if active_expert_storage is not None else None),
                stream=stream,
            )

        # The grouped W1 kernel derives its live CTA bound and expert mapping
        # exclusively from device-produced sorter metadata.  Its launch grid is
        # only a safe allocation bound, so no expert-frequency readback is
        # required to recompute preactivation.  Keep the generic fallback for
        # non-SwiGLU/FP16 shapes until their epilogues are enabled here.
        if use_grouped_w1:
            if use_compact_w1:
                assert compact_w1_descriptors is not None
                assert compact_w1_total is not None
                build_compact_m_tile_descriptors(
                    expert_frequency,
                    sorted_expert_ids,
                    num_valid_ids,
                    compact_w1_descriptors,
                    compact_w1_total,
                    block_m=grouped_w1_bm,
                    sorted_block_m=sort_unit,
                    descriptor_capacity=compact_w1_bound,
                    active_expert_storage=active_expert_storage,
                    active_expert_capacity=(active_expert_capacity if active_expert_storage is not None else None),
                    stream=stream,
                )
            if not reuse_forward_preactivation:
                assert preactivation is not None
                grouped_w1 = _compile_grouped_w1_recompute(
                    hidden_size,
                    intermediate_size,
                    num_experts,
                    topk,
                    has_bias,
                    interleaved_w1,
                    use_compact_w1,
                    device_index,
                )
                grouped_w1_grid = (compact_w1_bound if use_compact_w1 else num_experts) * (
                    intermediate_size // grouped_w1_bn
                )
                dummy_ptr = w1_arg.data_ptr()
                _run_compiled(
                    grouped_w1,
                    x_arg.data_ptr(),
                    w1_arg.data_ptr(),
                    (compact_w1_storage.data_ptr() if compact_w1_storage is not None else expert_frequency.data_ptr()),
                    dummy_ptr if b1_arg is None else b1_arg.data_ptr(),
                    sorted_expert_ids.data_ptr(),
                    num_valid_ids.data_ptr(),
                    sorted_token_ids.data_ptr(),
                    tokens,
                    int(grouped_w1_grid),
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    float("inf"),
                    preactivation.data_ptr(),
                    stream,
                )

        # Fully grouped BF16/SwiGLU paths never reconstruct expert segments on
        # the host.  Fixed-K routing bounds each expert by the token count,
        # while row kernels read a device queue or the exact padded extent from
        # ``num_valid_ids[0]``.  All fallback contracts retain the original
        # frequency readback and segment loop.
        if use_hostless_grouped:
            frequencies = None
            segments: list[tuple[int, int, int]] = []
            padded_rows = 0
            max_expert_rows = tokens
        else:
            frequencies, segments, padded_rows, max_expert_rows = _materialize_expert_segments(
                expert_frequency, sort_unit
            )

        if grouped_weight_grads:
            if use_hostless_grouped:
                if not hostless_dense_weight_zero:
                    active_count_storage = active_expert_storage if active_expert_storage is not None else num_valid_ids
                    active_count_divisor = 1 if active_expert_storage is not None else sort_unit
                    zero_weight_grads_adaptive_flydsl(
                        expert_frequency,
                        active_count_storage,
                        dw1,
                        dw2,
                        active_count_divisor=active_count_divisor,
                        dense_active_ratio=_INACTIVE_WEIGHT_GRAD_ZERO_ACTIVE_RATIO,
                        stream=stream,
                    )
            else:
                active_experts = len(segments)
                selective_weight_grad_zero = active_experts * _INACTIVE_WEIGHT_GRAD_ZERO_ACTIVE_RATIO >= num_experts
                if selective_weight_grad_zero and active_experts < num_experts:
                    zero_inactive_weight_grads_flydsl(
                        expert_frequency,
                        dw1,
                        dw2,
                        stream=stream,
                    )
                elif not selective_weight_grad_zero:
                    dw1.zero_()
                    dw2.zero_()

        # One sorter block per active expert is itself a valid schedule and is
        # the lowest-latency path.  Compact W1 regimes already produced the
        # shared queue above; long/non-compact regimes build it once here for
        # both weight-gradient TN contractions.
        use_tn_metadata_direct = (
            (use_grouped_dw1 or use_grouped_dw2)
            and active_expert_storage is None
            and (routes <= sort_unit if use_hostless_grouped else max_expert_rows <= sort_unit)
        )
        if (use_grouped_dw1 or use_grouped_dw2) and not use_tn_metadata_direct and active_expert_storage is None:
            active_expert_storage = torch.empty(
                active_expert_queue_elements(routes, num_experts),
                dtype=torch.int32,
                device=device,
            )
            build_active_expert_queue_flydsl(
                expert_frequency,
                sorted_expert_ids,
                num_valid_ids,
                routes=routes,
                queue_storage=active_expert_storage,
                stream=stream,
            )

        if not use_fused_forward_state_prepare:
            assert x_sorted is not None
            assert dout_sorted is not None
            gather = _compile_gather(
                hidden_size,
                compute_dtype,
                device_index,
                use_hostless_grouped,
            )
            gather_work = (max_padded if use_hostless_grouped else padded_rows) * (hidden_size // 4)
            gather_grid = max(1, (gather_work + _BLOCK_THREADS - 1) // _BLOCK_THREADS)
            if use_hostless_grouped:
                gather_grid = min(_HOSTLESS_ROW_GRID_CAP, gather_grid)
            _run_compiled(
                gather,
                x_arg,
                dout_arg,
                sorted_token_ids,
                x_sorted,
                dout_sorted,
                num_valid_ids,
                tokens,
                padded_rows,
                gather_grid,
                stream,
            )

        # Other activation/dtype combinations retain the original per-expert
        # preactivation path.
        if not use_grouped_w1 and not reuse_forward_preactivation:
            assert preactivation is not None
            assert x_sorted is not None
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

        activation_grid = min(_HOSTLESS_ROW_GRID_CAP, max_padded) if use_hostless_grouped else padded_rows
        if forward_state_data is None:
            assert preactivation is not None
            assert dout_sorted is not None
            activation_prepare = _compile_activation_prepare(
                hidden_size,
                intermediate_size,
                activation_name,
                compute_dtype,
                interleaved_w1,
                device_index,
                use_hostless_grouped,
            )
            _run_compiled(
                activation_prepare,
                preactivation,
                activation,
                dout_sorted,
                dy,
                sorted_weights,
                sorted_token_ids,
                num_valid_ids,
                tokens,
                activation_grid,
                stream,
            )
        else:
            route_preactivation, producer_stream, ready_event = forward_state_data
            route_preactivation.record_stream(stream)
            if int(stream.cuda_stream) != producer_stream:
                stream.wait_event(ready_event)
            if use_fused_forward_state_prepare:
                fused_prepare = _compile_fused_forward_state_prepare(
                    hidden_size,
                    intermediate_size,
                    topk,
                    interleaved_w1,
                    device_index,
                    state_schedule_block_m,
                    use_fused_da_dscore,
                    store_x_sorted=not use_direct_grouped_dw1_rhs,
                )
                assert state_row_schedule is not None
                state_prepare_grid = min(
                    _HOSTLESS_ROW_GRID_CAP,
                    max(1, state_schedule_bound * state_schedule_block_m),
                )
                _run_compiled(
                    fused_prepare,
                    x_arg,
                    dout_arg,
                    route_preactivation,
                    x_arg if x_sorted is None else x_sorted,
                    activation,
                    dy,
                    sorted_weights,
                    sorted_token_ids,
                    state_row_schedule,
                    tokens,
                    state_prepare_grid,
                    stream,
                )
            else:
                assert preactivation is not None
                assert dout_sorted is not None
                activation_prepare = _compile_activation_prepare_from_forward_state(
                    hidden_size,
                    intermediate_size,
                    topk,
                    interleaved_w1,
                    device_index,
                    use_hostless_grouped,
                )
                _run_compiled(
                    activation_prepare,
                    route_preactivation,
                    preactivation,
                    activation,
                    dout_sorted,
                    dy,
                    sorted_weights,
                    sorted_token_ids,
                    num_valid_ids,
                    tokens,
                    activation_grid,
                    stream,
                )

        if use_grouped_dw2 and not use_fused_da_dscore:
            _launch_grouped_dw2(
                dy,
                activation,
                expert_frequency,
                sorted_expert_ids,
                num_valid_ids,
                dw2,
                active_expert_storage,
                use_hostless_grouped=use_hostless_grouped,
                use_tn_metadata_direct=use_tn_metadata_direct,
                max_expert_rows=max_expert_rows,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                active_experts=len(segments),
                stream=stream,
            )

        if use_grouped_da:
            if use_hostless_grouped and active_expert_storage is not None:
                grouped_da_profiles = _grouped_da_hostless_profiles(
                    tokens=tokens,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts=num_experts,
                    topk=topk,
                    max_expert_rows=max_expert_rows,
                )
                active_count_ptr = active_expert_storage.data_ptr()
            else:
                grouped_da_profiles = (
                    (
                        *_grouped_da_tuning(max_expert_rows, hidden_size),
                        False,
                        0,
                        None,
                        _GROUPED_DA_STAGES,
                        None,
                    ),
                )
                # Unguarded specializations do not dereference this argument.
                active_count_ptr = expert_frequency.data_ptr()

            for (
                grouped_da_bm,
                grouped_da_bn,
                grouped_da_bk,
                grouped_da_mw,
                grouped_da_nw,
                grouped_da_queue_direct,
                min_active_experts,
                max_active_experts,
                grouped_da_stages,
                grouped_da_grid_cap,
            ) in grouped_da_profiles:
                grouped_da = _compile_grouped_da(
                    hidden_size,
                    intermediate_size,
                    num_experts,
                    grouped_da_bm,
                    grouped_da_bn,
                    grouped_da_bk,
                    grouped_da_mw,
                    grouped_da_nw,
                    device_index,
                    grouped_da_queue_direct,
                    min_active_experts,
                    max_active_experts,
                    stages=grouped_da_stages,
                    persistent=grouped_da_grid_cap is not None,
                )
                if grouped_da_queue_direct:
                    guarded_capacity = active_expert_capacity
                    if max_active_experts is not None:
                        guarded_capacity = min(guarded_capacity, max_active_experts)
                    grouped_da_grid = guarded_capacity * (intermediate_size // grouped_da_bn)
                else:
                    grouped_da_grid = num_experts * (intermediate_size // grouped_da_bn)
                    if grouped_da_grid_cap is not None:
                        grouped_da_grid = min(grouped_da_grid, grouped_da_grid_cap)
                _run_compiled(
                    grouped_da,
                    dy.data_ptr(),
                    w2_arg.data_ptr(),
                    expert_frequency.data_ptr(),
                    sorted_expert_ids.data_ptr(),
                    num_valid_ids.data_ptr(),
                    active_count_ptr,
                    da.data_ptr(),
                    int(grouped_da_grid),
                    stream,
                )

        if use_grouped_w2 and not use_fused_da_dscore:
            assert projection is not None
            grouped_w2 = _compile_grouped_w2_recompute(
                hidden_size,
                intermediate_size,
                num_experts,
                topk,
                has_bias,
                device_index,
            )
            grouped_w2_grid = num_experts * (hidden_size // _GROUPED_W2_BN)
            dummy_ptr = w2_arg.data_ptr()
            _run_compiled(
                grouped_w2,
                activation.data_ptr(),
                w2_arg.data_ptr(),
                expert_frequency.data_ptr(),
                dummy_ptr if b2_arg is None else b2_arg.data_ptr(),
                sorted_expert_ids.data_ptr(),
                num_valid_ids.data_ptr(),
                sorted_token_ids.data_ptr(),
                sorted_weights.data_ptr(),
                tokens,
                max_blocks,
                int(grouped_w2_grid),
                projection.data_ptr(),
                stream,
            )

        # Fallbacks recompute the down projection for ds and use materialized
        # A16 dy for both da and dW2.  The fused path instead gets dscore from
        # q=dout@W2, so it deliberately skips these per-expert projections.
        for expert, start, rows in segments:
            end = start + rows
            if not use_grouped_w2 and not use_fused_da_dscore:
                assert projection is not None
                gemm_a16w16(
                    activation[start:end],
                    w2_arg[expert].transpose(0, 1),
                    out=projection[start:end],
                    bias=None if b2_arg is None else b2_arg[expert],
                    user_kwargs=_GEMM_KWARGS,
                    stream=stream,
                    layout="nt",
                )
            if not use_grouped_da:
                gemm_a16w16(
                    dy[start:end],
                    w2_arg[expert],
                    out=da[start:end],
                    user_kwargs=_GEMM_KWARGS,
                    stream=stream,
                    layout="nn",
                )
            if not use_grouped_dw2:
                gemm_a16w16(
                    dy[start:end].transpose(0, 1),
                    activation[start:end],
                    out=dw2[expert],
                    user_kwargs=_GEMM_KWARGS,
                    stream=stream,
                    layout="tn",
                )

        derivative_grid = min(_HOSTLESS_ROW_GRID_CAP, max_padded) if use_hostless_grouped else padded_rows
        if use_fused_da_dscore:
            assert forward_state_data is not None
            route_preactivation = forward_state_data[0]
            fused_derivative = _compile_fused_activation_derivative_dscore_scale_dy(
                hidden_size,
                intermediate_size,
                topk,
                interleaved_w1,
                device_index,
                state_schedule_block_m,
            )
            assert state_row_schedule is not None
            derivative_grid = min(
                _HOSTLESS_ROW_GRID_CAP,
                max(1, state_schedule_bound * state_schedule_block_m),
            )
            _run_compiled(
                fused_derivative,
                route_preactivation,
                activation,
                da,
                dy,
                dz,
                sorted_weights,
                sorted_token_ids,
                droute_weights,
                state_row_schedule,
                tokens,
                derivative_grid,
                stream,
            )
            # dy was intentionally left unscaled until the fused row kernel so
            # grouped dA could compute q=dout@W2.  dW2 can start only now.
            _launch_grouped_dw2(
                dy,
                activation,
                expert_frequency,
                sorted_expert_ids,
                num_valid_ids,
                dw2,
                active_expert_storage,
                use_hostless_grouped=use_hostless_grouped,
                use_tn_metadata_direct=use_tn_metadata_direct,
                max_expert_rows=max_expert_rows,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                active_experts=len(segments),
                stream=stream,
            )
        elif use_fused_forward_state_prepare:
            assert forward_state_data is not None
            route_preactivation = forward_state_data[0]
            derivative_from_state = _compile_activation_derivative_from_forward_state(
                intermediate_size,
                topk,
                interleaved_w1,
                device_index,
            )
            assert compact_w1_storage is not None
            derivative_grid = min(
                _HOSTLESS_ROW_GRID_CAP,
                max(1, compact_w1_bound * _COMPACT_W1_BM),
            )
            _run_compiled(
                derivative_from_state,
                route_preactivation,
                da,
                dz,
                sorted_token_ids,
                compact_w1_storage,
                tokens,
                derivative_grid,
                stream,
            )
        else:
            assert preactivation is not None
            activation_derivative = _compile_activation_derivative(
                intermediate_size,
                activation_name,
                compute_dtype,
                interleaved_w1,
                device_index,
                use_hostless_grouped,
            )
            _run_compiled(
                activation_derivative,
                preactivation,
                da,
                dz,
                sorted_token_ids,
                num_valid_ids,
                tokens,
                derivative_grid,
                stream,
            )

        if use_grouped_dw1:
            grouped_dw1_rhs = x_arg if use_direct_grouped_dw1_rhs else x_sorted
            assert grouped_dw1_rhs is not None
            (
                grouped_dw1_bm,
                grouped_dw1_bn,
                grouped_dw1_bk,
                grouped_dw1_k_padding,
                grouped_dw1_m_waves,
                grouped_dw1_n_waves,
            ) = _grouped_dw1_tuning(
                max_expert_rows,
                hidden_size,
                intermediate_size,
                direct_rhs=use_direct_grouped_dw1_rhs,
            )
            grouped_dw1_kwargs = {
                "block_m": grouped_dw1_bm,
                "block_n": grouped_dw1_bn,
                "block_k": grouped_dw1_bk,
                "k_padding": grouped_dw1_k_padding,
                "m_waves": grouped_dw1_m_waves,
                "n_waves": grouped_dw1_n_waves,
                "stream": stream,
            }
            if use_tn_metadata_direct:
                grouped_tn_from_metadata_flydsl(
                    dz,
                    grouped_dw1_rhs,
                    expert_frequency,
                    sorted_expert_ids,
                    num_valid_ids,
                    dw1,
                    sorted_token_ids=(sorted_token_ids if use_direct_grouped_dw1_rhs else None),
                    **grouped_dw1_kwargs,
                )
            else:
                assert active_expert_storage is not None
                grouped_tn_from_queue_flydsl(
                    dz,
                    grouped_dw1_rhs,
                    expert_frequency,
                    active_expert_storage,
                    dw1,
                    sorted_token_ids=(sorted_token_ids if use_direct_grouped_dw1_rhs else None),
                    **grouped_dw1_kwargs,
                )

        if use_grouped_dx:
            # Compact queues round real expert rows to their selected BM.
            # Every scheduled tail stays inside the sorter's 64-row padding,
            # and the fixed-K route epilogue ignores non-sentinel rows.  Ragged
            # routing retains the separate sorted-output reduction below.
            if use_large_grouped_dx:
                grouped_dx_bm = _LARGE_GROUPED_DX_BM
                grouped_dx_profiles = ((_LARGE_GROUPED_DX_BN, _LARGE_GROUPED_DX_N_WAVES, 0, None),)
                grouped_dx_m_tiles = large_dx_bound
                grouped_dx_compact = True
                grouped_dx_schedule = large_dx_storage
            elif use_hostless_grouped and active_expert_storage is not None:
                grouped_dx_bm = _GROUPED_DX_BM
                sparse_dx = _grouped_dx_tuning(0, hidden_size)
                dense_dx = _grouped_dx_tuning(_GROUPED_DX_DENSE_EXPERTS, hidden_size)
                if sparse_dx == dense_dx:
                    grouped_dx_profiles = ((*sparse_dx, 0, None),)
                else:
                    grouped_dx_profiles = (
                        (*sparse_dx, 0, _GROUPED_DX_DENSE_EXPERTS - 1),
                        (*dense_dx, _GROUPED_DX_DENSE_EXPERTS, None),
                    )
                grouped_dx_m_tiles = compact_w1_bound
                grouped_dx_compact = use_compact_w1
                grouped_dx_schedule = compact_w1_storage
            else:
                grouped_dx_bm = _GROUPED_DX_BM
                active_experts = min(routes, num_experts) if use_hostless_grouped else len(segments)
                grouped_dx_profiles = ((*_grouped_dx_tuning(active_experts, hidden_size), 0, None),)
                grouped_dx_m_tiles = (
                    sum((int(count) + _GROUPED_DX_BM - 1) // _GROUPED_DX_BM for count in frequencies)
                    if use_compact_w1
                    else active_experts
                )
                grouped_dx_compact = use_compact_w1
                grouped_dx_schedule = compact_w1_storage

            for (
                grouped_dx_bn,
                grouped_dx_n_waves,
                min_active_experts,
                max_active_experts,
            ) in grouped_dx_profiles:
                grouped_dx = _compile_grouped_dx(
                    hidden_size,
                    intermediate_size,
                    num_experts,
                    grouped_dx_bm,
                    grouped_dx_bn,
                    grouped_dx_n_waves,
                    grouped_dx_compact,
                    device_index,
                    min_active_experts,
                    max_active_experts,
                    store_route_slots=direct_grouped_dx_routes,
                    top_k=topk,
                )
                grouped_dx_grid = max(
                    1,
                    min(
                        _GROUPED_DX_GRID_CAP,
                        grouped_dx_m_tiles * (hidden_size // grouped_dx_bn),
                    ),
                )
                _run_compiled(
                    grouped_dx,
                    dz.data_ptr(),
                    w1_arg.data_ptr(),
                    (
                        grouped_dx_schedule.data_ptr()
                        if grouped_dx_schedule is not None
                        else expert_frequency.data_ptr()
                    ),
                    sorted_expert_ids.data_ptr(),
                    (
                        active_expert_storage.data_ptr()
                        if min_active_experts > 0 or max_active_experts is not None
                        else num_valid_ids.data_ptr()
                    ),
                    (dx_routes if direct_grouped_dx_routes else dx_sorted).data_ptr(),
                    *(
                        (sorted_token_ids.data_ptr(), tokens)
                        if direct_grouped_dx_routes
                        else ()
                    ),
                    grouped_dx_grid,
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
            if not use_grouped_dw1:
                assert x_sorted is not None
                gemm_a16w16(
                    dz[start:end].transpose(0, 1),
                    x_sorted[start:end],
                    out=dw1[expert],
                    user_kwargs=_GEMM_KWARGS,
                    stream=stream,
                    layout="tn",
                )
            if not use_grouped_dx:
                assert dx_sorted is not None
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

        if flat_routes:
            assert sorted_route_ids is not None
            assert dx_accum is not None
            assert dx_sorted is not None
            assert dout_sorted is not None
            assert projection is not None
            route_score_backward = _compile_route_score_backward(
                hidden_size,
                compute_dtype,
                device_index,
            )
            _run_compiled(
                route_score_backward,
                dout_sorted,
                projection,
                sorted_token_ids,
                sorted_route_ids,
                droute_weights,
                tokens,
                routes,
                padded_rows,
                stream,
            )

            reduce_routes = _compile_ragged_dx_reduction(
                hidden_size,
                compute_dtype,
                device_index,
            )
            output_elements = tokens * hidden_size
            scatter_elements = padded_rows * hidden_size
            clear_grid = max(1, (output_elements + _BLOCK_THREADS - 1) // _BLOCK_THREADS)
            scatter_grid = max(1, (scatter_elements + _BLOCK_THREADS - 1) // _BLOCK_THREADS)
            _run_compiled(
                reduce_routes,
                dx_sorted,
                sorted_token_ids,
                dx_accum,
                dx,
                tokens,
                padded_rows,
                clear_grid,
                scatter_grid,
                stream,
            )
        else:
            if not use_fused_da_dscore:
                assert projection is not None
                score_backward = _compile_score_backward(
                    hidden_size,
                    topk,
                    compute_dtype,
                    device_index,
                    use_hostless_grouped,
                    use_fused_forward_state_prepare,
                )
                _run_compiled(
                    score_backward,
                    dout_arg if use_fused_forward_state_prepare else dout_sorted,
                    projection,
                    sorted_token_ids,
                    droute_weights,
                    num_valid_ids,
                    tokens,
                    (min(_HOSTLESS_ROW_GRID_CAP, max_padded) if use_hostless_grouped else padded_rows),
                    stream,
                )

            assert dx_routes is not None
            if not direct_grouped_dx_routes:
                assert dx_sorted is not None
                unsort = _compile_unsort(
                    hidden_size,
                    topk,
                    compute_dtype,
                    device_index,
                    use_hostless_grouped,
                )
                unsort_work = (max_padded if use_hostless_grouped else padded_rows) * (hidden_size // 4)
                unsort_grid = max(1, (unsort_work + _BLOCK_THREADS - 1) // _BLOCK_THREADS)
                if use_hostless_grouped:
                    unsort_grid = min(_HOSTLESS_ROW_GRID_CAP, unsort_grid)
                _run_compiled(
                    unsort,
                    dx_sorted,
                    sorted_token_ids,
                    dx_routes,
                    num_valid_ids,
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

    result = (dx, dw1, dw2, droute_weights)
    if has_bias:
        return (*result, db1, db2)
    return result


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
    interleaved_w1: bool = False,
    forward_state: object | None = None,
) -> tuple[torch.Tensor, ...]:
    """Differentiate dense BF16/FP16 fixed-K SonicMoE, optionally with bias.

    Parameters use logical, expert-major weights: ``w1[E, 2I, H]`` for GLU
    activations, ``w1[E, I, H]`` for pointwise activations, and
    ``w2[E, H, I]``. By default GLU W1 rows are ``[g0..gI, u0..uI]``. Setting
    ``interleaved_w1=True`` instead consumes W1 (and optional B1) in native
    ``[g0, u0, g1, u1, ...]`` row order and returns ``dw1``/``db1`` in that
    same order. Routing tensors are ``topk_ids[int32, T, K]`` and
    ``topk_weights[float32, T, K]``. Without bias, the returned tuple is
    ``(dx, dw1, dw2, dtopk_weights)``. When ``b1`` and ``b2`` are supplied,
    ``(db1, db2)`` are appended. Tensor and bias gradients preserve the A16
    input dtype; routing-score gradients use FP32.

    ``forward_state`` may be the invocation-owned state returned by SonicMoE's
    BF16 SwiGLU fixed-K training forward.  Its route-order preactivation is
    gathered directly into backward sorter order, skipping W1 recomputation.
    Passing ``None`` retains the standalone recompute path.

    State validation is structural.  The state must come from the official
    training forward for this exact ``hidden_states``, ``w1``, ``topk_ids``,
    and optional ``b1`` invocation.  As with the hot-path expert-id value
    contract, proving that semantic identity would require synchronization or
    retaining/hash-reading large inputs and is therefore an unchecked caller
    precondition.

    Expert ids must be in range and distinct within each token. As in the
    inference fixed-K path, value validation is an unchecked hot-path
    precondition so the only synchronization is the bring-up implementation's
    expert-frequency copy used for per-expert GEMM dispatch.
    """

    dimensions = _validate_backward_inputs(
        hidden_states,
        w1,
        w2,
        topk_ids,
        topk_weights,
        grad_output,
        config,
        b1,
        b2,
        interleaved_w1,
    )
    forward_state_data = (
        None
        if forward_state is None
        else _validate_forward_state(
            forward_state,
            hidden_states,
            config,
            interleaved_w1,
            b1 is not None,
        )
    )
    return _sonic_moe_backward_impl(
        hidden_states,
        w1,
        w2,
        topk_ids,
        topk_weights,
        grad_output,
        config,
        token_indices=None,
        dimensions=dimensions,
        b1=b1,
        b2=b2,
        interleaved_w1=interleaved_w1,
        forward_state_data=forward_state_data,
    )


def sonic_moe_backward_routes(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    token_indices: torch.Tensor,
    expert_indices: torch.Tensor,
    route_weights: torch.Tensor,
    grad_output: torch.Tensor,
    config: "SonicMoEConfig",
    *,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
    interleaved_w1: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Differentiate SonicMoE over a flat variable-count route list.

    ``token_indices``, ``expert_indices``, and ``route_weights`` are contiguous
    ``[R]`` tensors with int32, int32, and float32 dtype. Every route is tracked
    by its original position, so duplicate ``(token, expert)`` pairs receive
    independent score gradients. Tokens may have zero routes and ``R`` may be
    zero. The result contract matches :func:`sonic_moe_backward`, with
    ``droute_weights`` replacing the fixed-K score gradient.
    ``interleaved_w1`` has the same GLU W1/B1 input and gradient-layout
    contract as :func:`sonic_moe_backward`.

    Token and expert ids must be in range. Value validation remains an unchecked
    hot-path precondition; the compatibility adapter validates it before launch.
    """

    validated = _validate_backward_route_inputs(
        hidden_states,
        w1,
        w2,
        token_indices,
        expert_indices,
        route_weights,
        grad_output,
        config,
        b1,
        b2,
        interleaved_w1,
    )
    return _sonic_moe_backward_impl(
        hidden_states,
        w1,
        w2,
        expert_indices,
        route_weights,
        grad_output,
        config,
        token_indices=token_indices,
        dimensions=validated[:4],
        b1=b1,
        b2=b2,
        interleaved_w1=interleaved_w1,
    )


__all__ = ["sonic_moe_backward", "sonic_moe_backward_routes"]
