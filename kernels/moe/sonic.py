# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""SonicMoE-style A16 inference forward for CDNA4 (gfx950).

The hot path is three logical stages:

1. router softmax + top-k + expert sort/token rounding (and output zeroing),
2. grouped stage-1 GEMM with indexed row-gather and fused activation, and
3. grouped down-projection with routing-weighted atomic scatter.

No gathered activation tensor is materialized.  Expert rows are represented by
``sorted_token_ids`` and each expert's row count is rounded to ``tile_m`` by the
sorting kernel.  Stage 1 gathers the original activation rows while loading A;
stage 2 consumes the sorted 16-bit intermediate and scatters directly to tokens.

Weights may be dense BF16/FP16 (A16W16) or per-1x32 E8M0-scaled MXFP4
(A16W4). MXFP4 currently uses BF16 activations; dense weights use the compute
dtype selected by :class:`SonicMoEConfig`.
The reusable :class:`SonicMoE` object remains an inference-forward API.  This
module also exports standalone ``sonic_moe_backward`` and
``sonic_moe_backward_routes`` entry points for dense BF16/FP16 fixed-K and flat
ragged-route training across all supported activations, including optional
expert bias gradients.
"""

from __future__ import annotations

import functools
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.common.tensor_shim import _run_compiled
from kernels.moe.moe_2stage_a16wmix.gemm1 import (
    compile_gemm1_a16w4_port,
    gemm1_a16w4_grid,
)
from kernels.moe.moe_2stage_a16wmix.gemm2 import (
    compile_gemm2_a16w4_port,
    gemm2_a16w4_grid,
)
from kernels.moe.moe_ragged_sorting_kernel import moe_ragged_sorting_flydsl
from kernels.moe.moe_sorting_kernel import (
    moe_softmax_sort_flydsl,
    moe_sorting_flydsl,
    moe_sorting_get_workspace_size,
)
from kernels.moe.sonic_backward import (
    sonic_moe_backward as sonic_moe_backward,
    sonic_moe_backward_routes as sonic_moe_backward_routes,
)

_GFX950_LDS_BYTES = 160 * 1024
_MAX_BUFFER_BYTE_OFFSET = 0xFFFFFFFF
_MAX_SIGNED_I32 = 0x7FFFFFFF
_DEFAULT_MAX_CACHED_WORKSPACES = 8
_SUPPORTED_ROUTER_DTYPES = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}
_COMPUTE_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
_SUPPORTED_ACTIVATIONS = frozenset({"swiglu", "geglu", "reglu", "gelu_tanh_approx", "relu", "silu", "relu_sq"})
_GLU_ACTIVATIONS = frozenset({"swiglu", "geglu", "reglu"})
_GEMM1_ACTIVATIONS = {
    # gemm1's historical ``silu`` spelling means the fused SwiGLU epilogue.
    "swiglu": "silu",
    "geglu": "geglu",
    "reglu": "reglu",
    "gelu_tanh_approx": "gelu_tanh_approx",
    "relu": "relu",
    "silu": "silu_pointwise",
    "relu_sq": "relu_sq",
}


@dataclass(frozen=True)
class SonicMoEConfig:
    """Static shape and tile configuration for :class:`SonicMoE`.

    ``tile_n``/``tile_k`` configure the stage-1 GEMM.  The down-projection
    defaults to the same values and can be tuned independently with
    ``down_tile_n``/``down_tile_k``.  All tiles are compile-time constants.
    """

    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    tile_m: int = 32
    tile_n: int = 128
    tile_k: int = 128
    down_tile_n: int | None = None
    down_tile_k: int | None = None
    renormalize: bool = True
    stage1_b_cache_mod: int | None = None
    stage2_b_cache_mod: int | None = None
    stage1_xcd_swizzle: int = 0
    stage2_xcd_swizzle: int = 1
    waves_per_eu: int | None = None
    persistent_stage2: bool = False
    activation: str = "swiglu"
    compute_dtype: str = "bf16"

    def __post_init__(self) -> None:
        if self.compute_dtype not in _COMPUTE_DTYPES:
            raise ValueError(
                f"unsupported compute_dtype {self.compute_dtype!r}; expected one of " f"{sorted(_COMPUTE_DTYPES)}"
            )
        if not isinstance(self.activation, str):
            raise TypeError(f"activation must be a string, got {type(self.activation).__name__}")
        if self.activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"unsupported activation {self.activation!r}; expected one of " f"{sorted(_SUPPORTED_ACTIVATIONS)}"
            )
        positive = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "down_tile_n": self.stage2_tile_n,
            "down_tile_k": self.stage2_tile_k,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) cannot exceed num_experts ({self.num_experts})")
        if self.top_k > 16:
            raise ValueError(f"top_k must be <= 16 for the gfx950 router, got {self.top_k}")
        if self.tile_m % 16 != 0:
            raise ValueError(f"tile_m must be a multiple of 16, got {self.tile_m}")
        if self.tile_n % 64 != 0 or self.stage2_tile_n % 64 != 0:
            raise ValueError(
                "tile_n and down_tile_n must be multiples of 64, got " f"{self.tile_n}/{self.stage2_tile_n}"
            )
        if self.tile_k % 32 != 0 or self.stage2_tile_k % 32 != 0:
            raise ValueError(
                "tile_k and down_tile_k must be multiples of MFMA-K=32, got " f"{self.tile_k}/{self.stage2_tile_k}"
            )
        if self.tile_k & (self.tile_k - 1) or self.stage2_tile_k & (self.stage2_tile_k - 1):
            raise ValueError(
                "tile_k and down_tile_k must be powers of two for the LDS swizzle, got "
                f"{self.tile_k}/{self.stage2_tile_k}"
            )
        # A direct-to-LDS copy uses 256 threads x 16 bytes. Each workgroup must
        # cover an integral number of those 4096-byte transfer rounds.
        if (self.tile_m * self.tile_k) % 2048 != 0:
            raise ValueError(
                "tile_m * tile_k must be a multiple of 2048 A16 elements for " "the stage1 direct-to-LDS copy"
            )
        if (self.tile_m * self.stage2_tile_k) % 2048 != 0:
            raise ValueError(
                "tile_m * down_tile_k must be a multiple of 2048 A16 elements for " "the stage2 direct-to-LDS copy"
            )
        if self.hidden_size % 32 != 0 or self.intermediate_size % 32 != 0:
            raise ValueError("hidden_size and intermediate_size must be multiples of 32 for the " "16-bit preshuffle")
        if self.hidden_size % self.tile_k != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by tile_k ({self.tile_k})")
        if self.intermediate_size % self.tile_n != 0:
            raise ValueError(
                "intermediate_size " f"({self.intermediate_size}) must be divisible by tile_n ({self.tile_n})"
            )
        if self.intermediate_size % self.stage2_tile_k != 0:
            raise ValueError(
                "intermediate_size "
                f"({self.intermediate_size}) must be divisible by down_tile_k ({self.stage2_tile_k})"
            )
        if self.hidden_size % self.stage2_tile_n != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by " f"down_tile_n ({self.stage2_tile_n})"
            )
        if self.stage1_b_cache_mod not in (None, 0, 2):
            raise ValueError("stage1_b_cache_mod must be None, 0 (cached), or 2 (non-temporal)")
        if self.stage2_b_cache_mod not in (None, 0, 2):
            raise ValueError("stage2_b_cache_mod must be None, 0 (cached), or 2 (non-temporal)")
        if self.stage1_xcd_swizzle < 0 or self.stage2_xcd_swizzle < 0:
            raise ValueError("XCD swizzle values must be non-negative")

        # Fail before JIT compilation if a tile cannot fit the gfx950 160 KiB LDS.
        stage1_stages = 2 if self.hidden_size // self.tile_k > 1 else 1
        stage1_lds = stage1_stages * self.tile_m * self.tile_k * 2
        stage2_lds = self.tile_m * self.stage2_tile_k * 2 + self.tile_m * self.stage2_tile_n * 4
        if stage1_lds > _GFX950_LDS_BYTES:
            raise ValueError(
                f"stage1 tile needs {stage1_lds} LDS bytes, exceeding gfx950's " f"{_GFX950_LDS_BYTES} bytes"
            )
        if stage2_lds > _GFX950_LDS_BYTES:
            raise ValueError(
                f"stage2 tile needs {stage2_lds} LDS bytes, exceeding gfx950's " f"{_GFX950_LDS_BYTES} bytes"
            )

    @property
    def stage2_tile_n(self) -> int:
        return self.tile_n if self.down_tile_n is None else self.down_tile_n

    @property
    def stage2_tile_k(self) -> int:
        return self.tile_k if self.down_tile_k is None else self.down_tile_k

    @property
    def supports_flydsl_router(self) -> bool:
        """Whether the logits-to-top-k FlyDSL layout supports this expert count."""

        return self.num_experts <= 1024 and not (self.num_experts & (self.num_experts - 1))

    @property
    def is_glu(self) -> bool:
        return self.activation in _GLU_ACTIVATIONS

    @property
    def stage1_projection_size(self) -> int:
        return self.intermediate_size * (2 if self.is_glu else 1)


@dataclass(frozen=True)
class SonicMoEWeights:
    """Prepared weights consumed by the gfx950 grouped MFMA kernels.

    ``weight_dtype`` is ``"bf16"``/``"fp16"`` for dense A16W16 or ``"mxfp4"``
    for per-1x32 E8M0-scaled FP4 weights. MXFP4 currently requires a BF16
    activation/intermediate ABI.
    The logical weight shapes are captured by ``config``; tile-only config
    changes do not require another preshuffle.
    """

    gate_up: torch.Tensor
    down: torch.Tensor
    dummy_scale: torch.Tensor
    config: SonicMoEConfig
    gate_up_scale: torch.Tensor | None = None
    down_scale: torch.Tensor | None = None
    weight_dtype: str = "bf16"
    stage1_bias: torch.Tensor | None = None
    stage2_bias: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.gate_up.device

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        tensors = [self.gate_up, self.down, self.dummy_scale]
        if self.stage1_bias is not None:
            tensors.append(self.stage1_bias)
        if self.stage2_bias is not None:
            tensors.append(self.stage2_bias)
        if self.gate_up_scale is not None:
            tensors.append(self.gate_up_scale)
        if self.down_scale is not None:
            tensors.append(self.down_scale)
        return tuple(tensors)

    @property
    def has_bias(self) -> bool:
        return self.stage1_bias is not None

    @property
    def compute_dtype(self) -> torch.dtype:
        return _COMPUTE_DTYPES[self.config.compute_dtype]


@dataclass
class SonicMoEWorkspace:
    """Reusable routing, intermediate, and output buffers for one route shape."""

    tokens: int
    routes: int | None
    max_padded_tokens: int
    max_m_blocks: int
    sorted_token_ids: torch.Tensor
    sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor
    sorting_workspace: torch.Tensor | None
    expert_frequency: torch.Tensor
    router_topk_weights: torch.Tensor
    router_topk_ids: torch.Tensor
    router_topk_expert_indices: torch.Tensor
    intermediate: torch.Tensor
    output: torch.Tensor
    _launch_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    @functools.cached_property
    def storage_ptrs(self) -> frozenset[int]:
        """Storage bases owned by this workspace, cached off the hot path."""

        tensors = [
            self.sorted_token_ids,
            self.sorted_weights,
            self.sorted_expert_ids,
            self.num_valid_ids,
            self.expert_frequency,
            self.router_topk_weights,
            self.router_topk_ids,
            self.router_topk_expert_indices,
            self.intermediate,
            self.output,
        ]
        if self.sorting_workspace is not None:
            tensors.append(self.sorting_workspace)
        return frozenset(tensor.untyped_storage().data_ptr() for tensor in tensors)

    @classmethod
    def allocate(
        cls,
        config: SonicMoEConfig,
        tokens: int,
        device: torch.device,
        *,
        routes: int | None = None,
    ) -> "SonicMoEWorkspace":
        if tokens <= 0:
            raise ValueError(f"tokens must be positive, got {tokens}")
        if routes is not None and routes < 0:
            raise ValueError(f"routes must be non-negative, got {routes}")

        # If A experts are active, Q padded tiles need at least
        # Q*tile_m - A*(tile_m-1) real routes.  Dense top-k routing additionally
        # has at most one edge per (token, expert); flat routing deliberately
        # supports duplicates, so it cannot use that tighter bound.
        route_count = tokens * config.top_k if routes is None else int(routes)
        active_experts = min(config.num_experts, route_count)
        padding_bound = (route_count + active_experts * (config.tile_m - 1)) // config.tile_m
        if routes is None:
            per_expert_bound = active_experts * ((tokens + config.tile_m - 1) // config.tile_m)
            max_blocks = min(padding_bound, per_expert_bound)
        else:
            max_blocks = padding_bound
        max_padded = max_blocks * config.tile_m
        if max_padded > _MAX_SIGNED_I32:
            raise ValueError(
                "padded route count exceeds the sorting kernel's signed 32-bit " f"index limit: {max_padded}"
            )
        if max_padded * 4 > _MAX_BUFFER_BYTE_OFFSET:
            raise ValueError(
                "sorted route metadata exceeds the kernel's 32-bit byte-offset " f"limit: max_padded={max_padded}"
            )
        if max_padded * config.intermediate_size * 2 > _MAX_BUFFER_BYTE_OFFSET:
            raise ValueError(
                "sorted 16-bit intermediate exceeds the kernel's 32-bit byte-offset limit: "
                f"max_padded={max_padded}, intermediate_size={config.intermediate_size}"
            )
        if routes is None:
            sorting_workspace_size = moe_sorting_get_workspace_size(
                tokens,
                config.num_experts,
                config.top_k,
                unit_size=config.tile_m,
            )
            mesh_stride = ((tokens + config.tile_m - 1) // config.tile_m) * config.tile_m
            if config.num_experts * mesh_stride > _MAX_SIGNED_I32:
                raise ValueError(
                    "sorting mesh exceeds the kernel's signed 32-bit byte-index limit: "
                    f"experts={config.num_experts}, mesh_stride={mesh_stride}"
                )
        else:
            # Flat sorting only needs one atomic cursor per expert.  The exact
            # frequency is kept separately because the prefix phase must not
            # destroy the public frequency result.
            sorting_workspace_size = config.num_experts
        if sorting_workspace_size * 4 > _MAX_BUFFER_BYTE_OFFSET:
            raise ValueError(
                "sorting workspace exceeds the kernel's 32-bit byte-offset limit: "
                f"{sorting_workspace_size * 4} bytes"
            )
        return cls(
            tokens=tokens,
            routes=routes,
            max_padded_tokens=max_padded,
            max_m_blocks=max_blocks,
            # Keep a one-element backing allocation for the all-empty ragged
            # case so raw buffer descriptors never receive a null data pointer.
            sorted_token_ids=torch.empty(max(1, max_padded), dtype=torch.int32, device=device),
            sorted_weights=torch.empty(max(1, max_padded), dtype=torch.float32, device=device),
            sorted_expert_ids=torch.empty(max(1, max_blocks), dtype=torch.int32, device=device),
            num_valid_ids=torch.empty(2, dtype=torch.int32, device=device),
            sorting_workspace=(
                torch.empty(sorting_workspace_size, dtype=torch.int32, device=device)
                if sorting_workspace_size
                else None
            ),
            expert_frequency=torch.empty(config.num_experts, dtype=torch.int32, device=device),
            router_topk_weights=torch.empty((tokens, config.top_k), dtype=torch.float32, device=device),
            router_topk_ids=torch.empty((tokens, config.top_k), dtype=torch.int32, device=device),
            router_topk_expert_indices=torch.empty((tokens, config.top_k), dtype=torch.int32, device=device),
            intermediate=torch.empty(
                (max(1, max_padded), config.intermediate_size),
                dtype=_COMPUTE_DTYPES[config.compute_dtype],
                device=device,
            ),
            output=torch.empty(
                (tokens, config.hidden_size),
                dtype=_COMPUTE_DTYPES[config.compute_dtype],
                device=device,
            ),
        )


def _preshuffle_16bit_weight(
    weight: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert ``[..., N, K]`` FP16/BF16 rows to the 16x16 N-major layout."""

    n, k = weight.shape[-2:]
    if n % 16 != 0 or k % 32 != 0:
        raise ValueError(f"weight N/K must be divisible by 16/32, got {n}/{k}")
    if dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"dense SonicMoE weight dtype must be FP16 or BF16, got {dtype}")
    x = weight.detach().to(dtype=dtype).contiguous()
    leading = x.numel() // (n * k)
    # BK=32, KPack=8 16-bit values (16 bytes), BN=16.
    return x.view(leading, n // 16, 16, k // 32, 4, 8).permute(0, 1, 3, 4, 2, 5).contiguous().view_as(x)


def _validate_weight_inputs(
    w1: torch.Tensor,
    w2: torch.Tensor,
    config: SonicMoEConfig,
) -> None:
    expected_w1 = (config.num_experts, config.stage1_projection_size, config.hidden_size)
    expected_w2 = (config.num_experts, config.hidden_size, config.intermediate_size)
    if tuple(w1.shape) != expected_w1:
        raise ValueError(f"w1 must have shape {expected_w1}, got {tuple(w1.shape)}")
    if tuple(w2.shape) != expected_w2:
        raise ValueError(f"w2 must have shape {expected_w2}, got {tuple(w2.shape)}")
    if not w1.is_cuda or not w2.is_cuda:
        raise ValueError("SonicMoE weights must be on a ROCm device")
    if w1.device != w2.device:
        raise ValueError(f"w1 and w2 must share a device, got {w1.device}/{w2.device}")
    if not (w1.dtype.is_floating_point and w2.dtype.is_floating_point):
        raise TypeError(f"w1/w2 must be floating point, got {w1.dtype}/{w2.dtype}")


def _prepare_biases(
    b1: torch.Tensor | None,
    b2: torch.Tensor | None,
    w1: torch.Tensor,
    config: SonicMoEConfig,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Validate optional logical expert biases and materialize the A16 ABI."""

    if (b1 is None) != (b2 is None):
        raise ValueError("b1 and b2 must either both be provided or both be None")
    if b1 is None:
        return None, None
    assert b2 is not None

    expected_b1 = (config.num_experts, config.stage1_projection_size)
    expected_b2 = (config.num_experts, config.hidden_size)
    if tuple(b1.shape) != expected_b1:
        raise ValueError(f"b1 must have shape {expected_b1}, got {tuple(b1.shape)}")
    if tuple(b2.shape) != expected_b2:
        raise ValueError(f"b2 must have shape {expected_b2}, got {tuple(b2.shape)}")
    if not b1.is_cuda or not b2.is_cuda:
        raise ValueError("SonicMoE biases must be on a ROCm device")
    if b1.device != w1.device or b2.device != w1.device:
        raise ValueError("b1/b2 and expert weights must share a device")
    if not (b1.dtype.is_floating_point and b2.dtype.is_floating_point):
        raise TypeError(f"b1/b2 must be floating point, got {b1.dtype}/{b2.dtype}")
    compute_dtype = _COMPUTE_DTYPES[config.compute_dtype]
    return (
        b1.detach().to(dtype=compute_dtype, copy=True).contiguous(),
        b2.detach().to(dtype=compute_dtype, copy=True).contiguous(),
    )


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + multiple - 1) // multiple) * multiple


def _mxfp4_scale_storage_numel(experts: int, rows: int, k: int) -> int:
    """Physical uint8 elements in the padded gfx950 E8M0 scale layout."""

    return _round_up(experts * rows, 256) * _round_up(k // 32, 8)


def _validate_dense_resource_limits(config: SonicMoEConfig) -> None:
    # Each expert gets a 64-bit resource base, but offsets within it remain u32.
    gate_up_bytes_per_expert = config.stage1_projection_size * config.hidden_size * 2
    if gate_up_bytes_per_expert > _MAX_BUFFER_BYTE_OFFSET:
        raise ValueError(
            "dense 16-bit gate/up weights for one expert exceed the 32-bit "
            "byte-offset limit: "
            f"{gate_up_bytes_per_expert} bytes"
        )


def _validate_mxfp4_resource_limits(config: SonicMoEConfig) -> None:
    """Guard per-expert packed weights and whole-tensor E8M0 scale spans."""

    packed_gate_up_bytes_per_expert = config.stage1_projection_size * config.hidden_size // 2
    if packed_gate_up_bytes_per_expert > _MAX_BUFFER_BYTE_OFFSET:
        raise ValueError(
            "MXFP4 gate/up weights for one expert exceed the 32-bit byte-offset limit: "
            f"{packed_gate_up_bytes_per_expert} bytes"
        )

    gate_scale_cols = _round_up(config.hidden_size // 32, 8)
    down_scale_cols = _round_up(config.intermediate_size // 32, 8)
    spans = {
        "gate/up E8M0 scales": config.num_experts * config.stage1_projection_size * gate_scale_cols,
        "down E8M0 scales": config.num_experts * config.hidden_size * down_scale_cols,
    }
    for name, span in spans.items():
        if span > _MAX_BUFFER_BYTE_OFFSET:
            raise ValueError(
                f"{name} require a {span}-byte whole-tensor resource, exceeding "
                "the current 32-bit buffer-offset limit"
            )


def _validate_prepared_weight_storage(
    weights: SonicMoEWeights,
    config: SonicMoEConfig,
) -> None:
    """Validate the exact physical ABI before passing raw pointers to kernels."""

    if not weights.gate_up.is_cuda or not weights.down.is_cuda:
        raise ValueError("prepared SonicMoE weights must be on a ROCm device")
    if weights.gate_up.device != weights.down.device:
        raise ValueError("prepared gate/up and down weights must share a device")
    if any(tensor.device != weights.device for tensor in weights.tensors):
        raise ValueError("all prepared weights and scales must share a device")
    if any(not tensor.is_contiguous() for tensor in weights.tensors):
        raise ValueError("all prepared weights and scales must be contiguous")
    if any(tensor.requires_grad for tensor in weights.tensors):
        raise ValueError("prepared inference weights and scales must not require gradients")
    if weights.dummy_scale.dtype != torch.uint8 or weights.dummy_scale.numel() < 1:
        raise TypeError("dummy_scale must be a non-empty contiguous uint8 tensor")
    if weights.gate_up.data_ptr() % 16 or weights.down.data_ptr() % 16:
        raise ValueError("prepared gate/up and down weights must be 16-byte aligned")
    if (weights.stage1_bias is None) != (weights.stage2_bias is None):
        raise ValueError("prepared stage1_bias and stage2_bias must either both be present or absent")
    if weights.stage1_bias is not None:
        expected_b1 = (config.num_experts, config.stage1_projection_size)
        expected_b2 = (config.num_experts, config.hidden_size)
        if tuple(weights.stage1_bias.shape) != expected_b1:
            raise ValueError(
                f"prepared stage1_bias must have shape {expected_b1}, " f"got {tuple(weights.stage1_bias.shape)}"
            )
        if tuple(weights.stage2_bias.shape) != expected_b2:
            raise ValueError(
                f"prepared stage2_bias must have shape {expected_b2}, " f"got {tuple(weights.stage2_bias.shape)}"
            )
        expected_dtype = _COMPUTE_DTYPES[config.compute_dtype]
        if weights.stage1_bias.dtype != expected_dtype or weights.stage2_bias.dtype != expected_dtype:
            raise TypeError(f"prepared SonicMoE biases must use {expected_dtype} storage")
        if weights.stage1_bias.data_ptr() % 2 or weights.stage2_bias.data_ptr() % 2:
            raise ValueError("prepared SonicMoE biases must be 2-byte aligned")

    if weights.weight_dtype in ("bf16", "fp16"):
        expected_gate_up = (
            config.num_experts,
            config.stage1_projection_size,
            config.hidden_size,
        )
        expected_down = (
            config.num_experts,
            config.hidden_size,
            config.intermediate_size,
        )
        expected_dtype = _COMPUTE_DTYPES[config.compute_dtype]
        expected_format = config.compute_dtype
        if weights.weight_dtype != expected_format:
            raise ValueError(
                "dense prepared weight format must match compute_dtype, got "
                f"{weights.weight_dtype!r}/{config.compute_dtype!r}"
            )
        if weights.gate_up.dtype != expected_dtype or weights.down.dtype != expected_dtype:
            raise TypeError(f"{expected_format.upper()} prepared weights must use " f"{expected_dtype} storage")
        if weights.gate_up_scale is not None or weights.down_scale is not None:
            raise ValueError("dense prepared weights must not carry MXFP4 scale buffers")
    else:
        if config.compute_dtype != "bf16":
            raise ValueError("MXFP4 prepared weights currently require compute_dtype='bf16'")
        expected_gate_up = (
            config.num_experts,
            config.stage1_projection_size,
            config.hidden_size // 2,
        )
        expected_down = (
            config.num_experts,
            config.hidden_size,
            config.intermediate_size // 2,
        )
        if weights.gate_up.dtype != torch.uint8 or weights.down.dtype != torch.uint8:
            raise TypeError("MXFP4 prepared weights must use packed uint8 storage")
        if weights.gate_up_scale is None or weights.down_scale is None:
            raise ValueError("MXFP4 prepared weights require gate/up and down E8M0 scales")
        if weights.gate_up_scale.dtype != torch.uint8 or weights.down_scale.dtype != torch.uint8:
            raise TypeError("MXFP4 E8M0 scales must use uint8 storage")
        expected_gate_scale = _mxfp4_scale_storage_numel(
            config.num_experts,
            config.stage1_projection_size,
            config.hidden_size,
        )
        expected_down_scale = _mxfp4_scale_storage_numel(
            config.num_experts,
            config.hidden_size,
            config.intermediate_size,
        )
        if weights.gate_up_scale.ndim != 1 or weights.gate_up_scale.numel() != expected_gate_scale:
            raise ValueError(
                "MXFP4 gate/up scale storage has the wrong padded size: "
                f"expected {expected_gate_scale}, got {weights.gate_up_scale.numel()}"
            )
        if weights.down_scale.ndim != 1 or weights.down_scale.numel() != expected_down_scale:
            raise ValueError(
                "MXFP4 down scale storage has the wrong padded size: "
                f"expected {expected_down_scale}, got {weights.down_scale.numel()}"
            )
        if weights.gate_up_scale.data_ptr() % 4 or weights.down_scale.data_ptr() % 4:
            raise ValueError("MXFP4 E8M0 scale buffers must be 4-byte aligned")

    if tuple(weights.gate_up.shape) != expected_gate_up:
        raise ValueError(
            f"prepared gate/up storage must have shape {expected_gate_up}, " f"got {tuple(weights.gate_up.shape)}"
        )
    if tuple(weights.down.shape) != expected_down:
        raise ValueError(f"prepared down storage must have shape {expected_down}, " f"got {tuple(weights.down.shape)}")


def _f32_to_e8m0(values: torch.Tensor) -> torch.Tensor:
    """Encode positive FP32 scales as E8M0 exponent bytes."""

    values = values.to(torch.float32).contiguous()
    bits = values.view(torch.int32)
    exponent = ((bits >> 23) & 0xFF).to(torch.uint8)
    is_nan_or_inf = exponent == 0xFF
    round_up = ((bits & 0x400000) > 0) & (((bits & 0x200000) > 0) | ((bits & 0x1FFFFF) > 0) | (exponent > 0))
    rounded = (exponent.to(torch.int16) + round_up.to(torch.int16)).clamp_max(0xFE)
    return torch.where(
        is_nan_or_inf,
        torch.full_like(exponent, 0xFF),
        rounded.to(torch.uint8),
    )


def _e8m0_to_f32(scale: torch.Tensor) -> torch.Tensor:
    scale_u8 = scale.view(torch.uint8)
    bits = scale_u8.to(torch.int32) << 23
    bits = torch.where(scale_u8 == 0, torch.full_like(bits, 0x00400000), bits)
    bits = torch.where(scale_u8 == 0xFF, torch.full_like(bits, 0x7F800001), bits)
    return bits.view(torch.float32)


def _quantize_mxfp4_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-1x32 E8M0 quantization to packed E2M1 codes.

    Quantization is row-chunked so preparing a large expert tensor does not
    materialize a second full-size FP32 weight tensor.
    """

    if weight.ndim != 3 or weight.shape[-1] % 32 != 0:
        raise ValueError("MXFP4 weights must be rank-3 with K divisible by 32")
    experts, rows, k = (int(v) for v in weight.shape)
    packed = torch.empty((experts, rows, k // 2), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((experts, rows, k // 32), dtype=torch.uint8, device=weight.device)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=torch.float32,
        device=weight.device,
    )
    target_chunk_bytes = 64 * 1024 * 1024
    rows_per_chunk = max(1, target_chunk_bytes // (k * 4))

    packed_rows = packed.view(experts * rows, k // 2)
    scale_rows = scales.view(experts * rows, k // 32)

    def quantize_rows(chunk: torch.Tensor, output_row: int) -> None:
        chunk = chunk.to(torch.float32).contiguous()
        blocks = chunk.view(-1, 32)
        amax = blocks.abs().amax(dim=1)
        if not torch.isfinite(amax).all().item():
            raise ValueError("MXFP4 weights must contain only finite values")
        scale_e8m0 = _f32_to_e8m0(amax / 4.0)
        scale_f32 = _e8m0_to_f32(scale_e8m0)
        normalized = blocks / scale_f32[:, None]
        normalized_abs = normalized.abs()
        magnitude = torch.bucketize(normalized_abs, boundaries).to(torch.uint8)
        # torch.bucketize(right=False) selects the lower code at every exact
        # midpoint. MXFP4 uses round-to-nearest-even, so the three midpoints
        # whose lower code is odd must select the upper (even) code instead.
        rne_upper_tie = (normalized_abs == 0.75) | (normalized_abs == 1.75) | (normalized_abs == 3.5)
        magnitude = magnitude + rne_upper_tie.to(torch.uint8)
        codes = magnitude | (torch.signbit(normalized).to(torch.uint8) << 3)
        packed_chunk = (codes[:, 1::2] << 4) | codes[:, ::2]
        chunk_rows = int(chunk.shape[0])
        packed_rows[output_row : output_row + chunk_rows].copy_(packed_chunk.view(chunk_rows, k // 2))
        scale_rows[output_row : output_row + chunk_rows].copy_(scale_e8m0.view(chunk_rows, k // 32))

    if weight.is_contiguous():
        source_rows = weight.detach().view(experts * rows, k)
        for row_start in range(0, experts * rows, rows_per_chunk):
            row_end = min(row_start + rows_per_chunk, experts * rows)
            quantize_rows(source_rows[row_start:row_end], row_start)
    else:
        # Preserve the bounded-memory behavior for unusual strided inputs rather
        # than allowing reshape() to materialize the entire expert tensor.
        for expert in range(experts):
            for row_start in range(0, rows, rows_per_chunk):
                row_end = min(row_start + rows_per_chunk, rows)
                quantize_rows(
                    weight[expert, row_start:row_end].detach(),
                    expert * rows + row_start,
                )
    return packed, scales


def _dequantize_mxfp4_weight(
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Decode an unshuffled MXFP4 tensor; used by correctness oracles."""

    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise TypeError("packed MXFP4 values and E8M0 scales must be uint8")
    codes = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        dtype=torch.uint8,
        device=packed.device,
    )
    codes[..., ::2] = packed & 0xF
    codes[..., 1::2] = packed >> 4
    values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed.device,
    )
    decoded = values[codes.long()]
    scale_f32 = _e8m0_to_f32(scales).repeat_interleave(32, dim=-1)
    if decoded.shape != scale_f32.shape:
        raise ValueError(f"packed/scales shapes are inconsistent: {tuple(decoded.shape)}/{tuple(scales.shape)}")
    return decoded * scale_f32


def _preshuffle_mxfp4_weight(weight: torch.Tensor) -> torch.Tensor:
    """Preshuffle packed FP4 ``[..., N, K/2]`` into 16x16 byte tiles."""

    n, packed_k = (int(v) for v in weight.shape[-2:])
    if n % 16 != 0 or packed_k % 32 != 0:
        raise ValueError(f"packed MXFP4 weight N/(K/2) must be divisible by 16/32, got {n}/{packed_k}")
    leading = weight.numel() // (n * packed_k)
    return (
        weight.view(leading, n // 16, 16, packed_k // 32, 2, 16).permute(0, 1, 3, 4, 2, 5).contiguous().view_as(weight)
    )


def _preshuffle_e8m0_scale(scale: torch.Tensor) -> torch.Tensor:
    """Apply the gfx950 per-1x32 scale layout, including 256x8 padding."""

    if scale.ndim != 3:
        raise ValueError(f"scale must have shape [E, N, K/32], got {tuple(scale.shape)}")
    rows = int(scale.shape[0] * scale.shape[1])
    cols = int(scale.shape[2])
    padded_rows = ((rows + 255) // 256) * 256
    padded_cols = ((cols + 7) // 8) * 8
    padded = torch.full(
        (padded_rows, padded_cols),
        127,
        dtype=torch.uint8,
        device=scale.device,
    )
    padded[:rows, :cols] = scale.reshape(rows, cols)
    return padded.view(padded_rows // 32, 2, 16, padded_cols // 8, 2, 4).permute(0, 3, 5, 2, 4, 1).contiguous().view(-1)


def prepare_sonic_bf16_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    config: SonicMoEConfig,
    *,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
) -> SonicMoEWeights:
    """Validate and preshuffle stage-1 and down-projection weights.

    Parameters
    ----------
    w1:
        ``[num_experts, 2 * intermediate_size, hidden_size]`` in ``[gate | up]``
        order for GLU activations, otherwise
        ``[num_experts, intermediate_size, hidden_size]``. Floating input is
        converted to BF16 once during preparation.
    w2:
        ``[num_experts, hidden_size, intermediate_size]``.
    b1, b2:
        Optional expert-major biases with shapes
        ``[num_experts, stage1_projection_size]`` and
        ``[num_experts, hidden_size]``. They must be provided together and are
        copied to contiguous BF16 storage during preparation.
    """

    _validate_weight_inputs(w1, w2, config)
    if config.compute_dtype != "bf16":
        raise ValueError(
            "prepare_sonic_bf16_weights requires config.compute_dtype='bf16', got " f"{config.compute_dtype!r}"
        )
    _validate_dense_resource_limits(config)
    stage1_bias, stage2_bias = _prepare_biases(b1, b2, w1, config)

    return SonicMoEWeights(
        gate_up=_preshuffle_16bit_weight(w1, torch.bfloat16),
        down=_preshuffle_16bit_weight(w2, torch.bfloat16),
        dummy_scale=torch.zeros(1, dtype=torch.uint8, device=w1.device),
        config=config,
        stage1_bias=stage1_bias,
        stage2_bias=stage2_bias,
    )


def prepare_sonic_fp16_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    config: SonicMoEConfig,
    *,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
) -> SonicMoEWeights:
    """Validate and preshuffle dense FP16 stage-1/down-projection weights.

    The logical layouts match :func:`prepare_sonic_bf16_weights`. Floating
    inputs and optional expert biases are copied to contiguous FP16 storage.
    """

    if config.compute_dtype != "fp16":
        raise ValueError(
            "prepare_sonic_fp16_weights requires config.compute_dtype='fp16', got " f"{config.compute_dtype!r}"
        )
    _validate_weight_inputs(w1, w2, config)
    _validate_dense_resource_limits(config)
    stage1_bias, stage2_bias = _prepare_biases(b1, b2, w1, config)
    return SonicMoEWeights(
        gate_up=_preshuffle_16bit_weight(w1, torch.float16),
        down=_preshuffle_16bit_weight(w2, torch.float16),
        dummy_scale=torch.zeros(1, dtype=torch.uint8, device=w1.device),
        config=config,
        stage1_bias=stage1_bias,
        stage2_bias=stage2_bias,
        weight_dtype="fp16",
    )


def prepare_sonic_mxfp4_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    config: SonicMoEConfig,
    *,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
) -> SonicMoEWeights:
    """Quantize and preshuffle weight-only MXFP4 gate/up and down weights.

    Activations and the sorted stage-1 intermediate remain BF16 (A16W4). Each
    32-value weight block receives one E8M0 scale. This is the numerically
    validated low-memory path; it does not quantize activations to MXFP8.
    Optional ``b1``/``b2`` follow the same expert-major BF16 contract as
    :func:`prepare_sonic_bf16_weights`.
    """

    if config.compute_dtype != "bf16":
        raise ValueError("MXFP4 weights currently require config.compute_dtype='bf16'")
    _validate_weight_inputs(w1, w2, config)
    stage1_bias, stage2_bias = _prepare_biases(b1, b2, w1, config)
    if config.hidden_size % 64 != 0 or config.intermediate_size % 64 != 0:
        raise ValueError("MXFP4 weight preshuffle requires hidden/intermediate sizes divisible by 64")
    _validate_mxfp4_resource_limits(config)
    w1_quant, w1_scale = _quantize_mxfp4_weight(w1)
    w2_quant, w2_scale = _quantize_mxfp4_weight(w2)
    return SonicMoEWeights(
        gate_up=_preshuffle_mxfp4_weight(w1_quant),
        down=_preshuffle_mxfp4_weight(w2_quant),
        dummy_scale=torch.zeros(1, dtype=torch.uint8, device=w1.device),
        config=config,
        stage1_bias=stage1_bias,
        stage2_bias=stage2_bias,
        gate_up_scale=_preshuffle_e8m0_scale(w1_scale),
        down_scale=_preshuffle_e8m0_scale(w2_scale),
        weight_dtype="mxfp4",
    )


def _stage1_cache_mod(config: SonicMoEConfig, tokens: int) -> int:
    if config.stage1_b_cache_mod is not None:
        return config.stage1_b_cache_mod
    return 2 if 16 <= tokens <= 1024 else 0


def _stage2_cache_mod(config: SonicMoEConfig, tokens: int) -> int:
    if config.stage2_b_cache_mod is not None:
        return config.stage2_b_cache_mod
    return 0 if tokens <= 16 or tokens >= 2048 else 2


@functools.lru_cache(maxsize=256)
def _get_stage1_launcher(
    config: SonicMoEConfig,
    b_cache_mod: int,
    weight_dtype: str,
    has_bias: bool,
    device_index: int,
):
    # ``device_index`` is intentionally part of the LRU key.  Compiled
    # launchers cache a device-loaded function after first use and cannot be
    # shared across ROCm devices in one process.
    del device_index
    return compile_gemm1_a16w4_port(
        BM=config.tile_m,
        D_HIDDEN=config.hidden_size,
        D_INTER=config.intermediate_size,
        NE=config.num_experts,
        TOPK=config.top_k,
        TILE_N=config.tile_n,
        TILE_K=config.tile_k,
        act=_GEMM1_ACTIVATIONS[config.activation],
        b_cache_mod=b_cache_mod,
        xcd_swizzle=config.stage1_xcd_swizzle,
        waves_per_eu=config.waves_per_eu,
        w_dtype=weight_dtype,
        a_dtype=config.compute_dtype,
        w_layout="standard",
        k_wave=1,
        round_preact_bf16=True,
        has_bias=has_bias,
    )


@functools.lru_cache(maxsize=256)
def _get_stage2_launcher(
    config: SonicMoEConfig,
    b_cache_mod: int,
    weight_dtype: str,
    has_bias: bool,
    device_index: int,
):
    # See _get_stage1_launcher: keep a distinct loaded function per device.
    del device_index
    return compile_gemm2_a16w4_port(
        BM=config.tile_m,
        NE=config.num_experts,
        N_OUT=config.hidden_size,
        D_INTER=config.intermediate_size,
        TILE_N=config.stage2_tile_n,
        TILE_K=config.stage2_tile_k,
        xcd_swizzle=config.stage2_xcd_swizzle,
        b_cache_mod=b_cache_mod,
        waves_per_eu=config.waves_per_eu,
        w_dtype=weight_dtype,
        a_dtype=config.compute_dtype,
        persist=config.persistent_stage2,
        has_bias=has_bias,
        round_projection_bf16=True,
    )


class SonicMoE:
    """Reusable gfx950 SonicMoE A16 forward operator.

    Up to ``max_cached_workspaces`` workspaces are retained in LRU order, keyed by
    ``(device, stream, token_count, route_count)``; dense top-k calls use a
    dedicated route-count sentinel. The returned default output aliases that
    workspace and is overwritten by the next call with the same key; pass
    ``out=`` when the caller owns output storage. Independent streams receive
    independent workspaces, while calls sharing a cache key serialize their
    complete kernel enqueue sequence. Eviction only drops the operator's
    reference: an in-progress call retains its workspace locally, and its
    buffers are allocated and used on the keyed stream so PyTorch's stream-aware
    allocator defers unsafe reuse. Call :meth:`clear_workspace` to release all
    cached entries eagerly.
    """

    def __init__(
        self,
        config: SonicMoEConfig,
        weights: SonicMoEWeights,
        *,
        max_cached_workspaces: int = _DEFAULT_MAX_CACHED_WORKSPACES,
    ):
        prepared_shape = (
            weights.config.hidden_size,
            weights.config.intermediate_size,
            weights.config.num_experts,
            weights.config.activation,
            weights.config.compute_dtype,
        )
        requested_shape = (
            config.hidden_size,
            config.intermediate_size,
            config.num_experts,
            config.activation,
            config.compute_dtype,
        )
        if prepared_shape != requested_shape:
            raise ValueError(
                "prepared weights were created for different H/I/E/activation/dtype: "
                f"{prepared_shape} != {requested_shape}"
            )
        if weights.weight_dtype not in ("bf16", "fp16", "mxfp4"):
            raise ValueError(f"unsupported prepared weight dtype {weights.weight_dtype!r}")
        if weights.weight_dtype == "mxfp4":
            if config.tile_k < 128 or config.stage2_tile_k < 128:
                raise ValueError("MXFP4 requires tile_k and down_tile_k >= 128 for packed FP4 loads")
            _validate_mxfp4_resource_limits(config)
        else:
            _validate_dense_resource_limits(config)
        _validate_prepared_weight_storage(weights, config)
        if isinstance(max_cached_workspaces, bool) or not isinstance(max_cached_workspaces, int):
            raise TypeError("max_cached_workspaces must be an integer")
        if max_cached_workspaces <= 0:
            raise ValueError("max_cached_workspaces must be positive")
        self.config = config
        self.weights = weights
        self._max_cached_workspaces = max_cached_workspaces
        self._workspaces: OrderedDict[tuple[int, int, int, int], SonicMoEWorkspace] = OrderedDict()
        self._workspace_lock = threading.RLock()
        self.workspace: SonicMoEWorkspace | None = None

    def clear_workspace(self) -> None:
        with self._workspace_lock:
            self._workspaces.clear()
            self.workspace = None

    def reserve(self, tokens: int, *, routes: int | None = None) -> SonicMoEWorkspace:
        if routes is not None and routes < 0:
            raise ValueError(f"routes must be non-negative, got {routes}")
        stream_id = int(torch.cuda.current_stream(self.weights.device).cuda_stream)
        route_key = -1 if routes is None else int(routes)
        key = (self.weights.device.index or 0, stream_id, int(tokens), route_key)
        with self._workspace_lock:
            workspace = self._workspaces.get(key)
            if workspace is None:
                workspace = SonicMoEWorkspace.allocate(
                    self.config,
                    int(tokens),
                    self.weights.device,
                    routes=routes,
                )
                self._workspaces[key] = workspace
                while len(self._workspaces) > self._max_cached_workspaces:
                    self._workspaces.popitem(last=False)
            else:
                self._workspaces.move_to_end(key)
            self.workspace = workspace
            return workspace

    def _validate_hidden(self, hidden_states: torch.Tensor) -> int:
        if not hidden_states.is_cuda:
            raise ValueError("hidden_states must be on a ROCm device")
        if hidden_states.device != self.weights.device:
            raise ValueError(
                f"hidden_states and weights must share a device, got " f"{hidden_states.device}/{self.weights.device}"
            )
        expected_dtype = self.weights.compute_dtype
        if hidden_states.dtype != expected_dtype:
            raise TypeError(f"hidden_states must use {expected_dtype}, got {hidden_states.dtype}")
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.config.hidden_size:
            raise ValueError(
                f"hidden_states must have shape [tokens, {self.config.hidden_size}], "
                f"got {tuple(hidden_states.shape)}"
            )
        if not hidden_states.is_contiguous():
            raise ValueError("hidden_states must be contiguous")
        if hidden_states.data_ptr() % 16:
            raise ValueError("hidden_states must be 16-byte aligned for direct-to-LDS loads")
        if hidden_states.requires_grad:
            raise ValueError("SonicMoE is inference-only; hidden_states must not require gradients")
        tokens = int(hidden_states.shape[0])
        if tokens <= 0 or tokens > 0xFFFFFF:
            raise ValueError(f"tokens must be in [1, 2^24-1], got {tokens}")
        if tokens * self.config.hidden_size * 2 > _MAX_BUFFER_BYTE_OFFSET:
            raise ValueError(
                "16-bit atomic output addressing exceeds the 32-bit byte-offset limit: "
                f"tokens={tokens}, hidden_size={self.config.hidden_size}"
            )
        arch = get_rocm_arch()
        if not str(arch).startswith("gfx95"):
            raise RuntimeError(f"SonicMoE CDNA4 forward requires gfx95*, got {arch!r}")
        return tokens

    def _validate_out(
        self,
        out: torch.Tensor | None,
        workspace: SonicMoEWorkspace,
        *read_tensors: torch.Tensor,
    ) -> torch.Tensor:
        if out is None:
            out = workspace.output
        expected = (workspace.tokens, self.config.hidden_size)
        if tuple(out.shape) != expected:
            raise ValueError(f"out must have shape {expected}, got {tuple(out.shape)}")
        expected_dtype = self.weights.compute_dtype
        if out.device != self.weights.device or out.dtype != expected_dtype or not out.is_contiguous():
            raise ValueError(f"out must be contiguous {expected_dtype} on the same ROCm device as the weights")
        if out.data_ptr() % 4:
            raise ValueError("out must be 4-byte aligned for packed 16-bit atomic scatter")
        if out.requires_grad:
            raise ValueError("SonicMoE is inference-only; out must not require gradients")

        workspace_storages = workspace.storage_ptrs
        if any(tensor.untyped_storage().data_ptr() in workspace_storages for tensor in read_tensors):
            raise ValueError("inputs and prepared weights must not alias internal workspace storage")

        out_storage = out.untyped_storage().data_ptr()
        if any(out_storage == tensor.untyped_storage().data_ptr() for tensor in read_tensors):
            raise ValueError("out must not alias an input or prepared-weight storage")
        workspace_output_storage = workspace.output.untyped_storage().data_ptr()
        if out_storage in workspace_storages and out_storage != workspace_output_storage:
            raise ValueError("out must not alias internal workspace scratch storage")
        if out_storage == workspace_output_storage and out.data_ptr() != workspace.output.data_ptr():
            raise ValueError("out must start at the internal workspace output base address")
        return out

    def _run_grouped_gemms(
        self,
        hidden_states: torch.Tensor,
        workspace: SonicMoEWorkspace,
        out: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        tokens = workspace.tokens
        stream = torch.cuda.current_stream(hidden_states.device)

        stage1 = _get_stage1_launcher(
            cfg,
            _stage1_cache_mod(cfg, tokens),
            self.weights.weight_dtype,
            self.weights.has_bias,
            hidden_states.device.index or 0,
        )
        grid1 = gemm1_a16w4_grid(
            cfg.tile_m,
            INTER=cfg.intermediate_size,
            TILE_N=cfg.tile_n,
            max_m_blocks=workspace.max_m_blocks,
        )
        _run_compiled(
            stage1,
            hidden_states.data_ptr(),
            self.weights.gate_up.data_ptr(),
            (self.weights.dummy_scale if self.weights.gate_up_scale is None else self.weights.gate_up_scale).data_ptr(),
            (self.weights.dummy_scale if self.weights.stage1_bias is None else self.weights.stage1_bias).data_ptr(),
            workspace.sorted_expert_ids.data_ptr(),
            workspace.num_valid_ids.data_ptr(),
            workspace.sorted_token_ids.data_ptr(),
            tokens,
            int(grid1),
            1.0,
            1.0,
            1.0,
            1.0,
            float("inf"),
            workspace.intermediate.data_ptr(),
            stream,
        )

        stage2 = _get_stage2_launcher(
            cfg,
            _stage2_cache_mod(cfg, tokens),
            self.weights.weight_dtype,
            self.weights.has_bias,
            hidden_states.device.index or 0,
        )
        grid2 = gemm2_a16w4_grid(
            cfg.tile_m,
            N_OUT=cfg.hidden_size,
            TILE_N=cfg.stage2_tile_n,
            max_m_blocks=workspace.max_m_blocks,
            persist=cfg.persistent_stage2,
        )
        _run_compiled(
            stage2,
            workspace.intermediate.data_ptr(),
            self.weights.down.data_ptr(),
            (self.weights.dummy_scale if self.weights.down_scale is None else self.weights.down_scale).data_ptr(),
            (self.weights.dummy_scale if self.weights.stage2_bias is None else self.weights.stage2_bias).data_ptr(),
            workspace.sorted_expert_ids.data_ptr(),
            workspace.num_valid_ids.data_ptr(),
            workspace.sorted_token_ids.data_ptr(),
            workspace.sorted_weights.data_ptr(),
            tokens,
            workspace.max_m_blocks,
            int(grid2),
            out.data_ptr(),
            stream,
        )
        return out

    def __call__(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run router + grouped expert MLP from router logits.

        ``router_logits`` is ``[tokens, num_experts]`` in FP32/FP16/BF16.
        """

        if not hidden_states.is_cuda:
            raise ValueError("hidden_states must be on a ROCm device")
        with torch.cuda.device(hidden_states.device):
            return self._forward_from_logits(hidden_states, router_logits, out)

    def _forward_from_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        tokens = self._validate_hidden(hidden_states)
        if not router_logits.is_cuda or router_logits.device != hidden_states.device:
            raise ValueError("router_logits must be on the same ROCm device as hidden_states")
        if tuple(router_logits.shape) != (tokens, self.config.num_experts):
            raise ValueError(
                f"router_logits must have shape {(tokens, self.config.num_experts)}, "
                f"got {tuple(router_logits.shape)}"
            )
        dtype_str = _SUPPORTED_ROUTER_DTYPES.get(router_logits.dtype)
        if dtype_str is None:
            raise TypeError(f"router_logits must be FP32/FP16/BF16, got {router_logits.dtype}")
        if not router_logits.is_contiguous():
            raise ValueError("router_logits must be contiguous")
        if router_logits.requires_grad:
            raise ValueError("SonicMoE is inference-only; router_logits must not require gradients")
        if (
            self.config.supports_flydsl_router
            and router_logits.numel() * router_logits.element_size() > _MAX_BUFFER_BYTE_OFFSET
        ):
            raise ValueError("router_logits exceed the FlyDSL router's 32-bit byte-offset limit")

        # The current FlyDSL top-k gating layout maps a power-of-two expert row
        # (up to 1024 experts) to fixed lane groups. Models such as Kimi K2.5 use
        # E=896, so retain full operator coverage with a PyTorch router fallback;
        # the grouped GEMMs and expert sort remain FlyDSL kernels.
        if not self.config.supports_flydsl_router:
            probs = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(probs, self.config.top_k, dim=-1)
            if self.config.renormalize:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            return self.forward_topk(
                hidden_states,
                topk_ids.to(torch.int32),
                topk_weights.contiguous(),
                out=out,
            )

        workspace = self.reserve(tokens)
        output = self._validate_out(
            out,
            workspace,
            hidden_states,
            router_logits,
            *self.weights.tensors,
        )
        with workspace._launch_lock:
            moe_softmax_sort_flydsl(
                router_logits,
                workspace.sorted_token_ids,
                workspace.sorted_weights,
                workspace.sorted_expert_ids,
                workspace.num_valid_ids,
                output,
                self.config.num_experts,
                self.config.top_k,
                dtype_str,
                unit_size=self.config.tile_m,
                renormalize=self.config.renormalize,
                workspace=workspace.sorting_workspace,
                topk_scratch=(
                    workspace.router_topk_weights,
                    workspace.router_topk_ids,
                    workspace.router_topk_expert_indices,
                ),
            )
            return self._run_grouped_gemms(hidden_states, workspace, output)

    def forward_routes(
        self,
        hidden_states: torch.Tensor,
        token_indices: torch.Tensor,
        expert_indices: torch.Tensor,
        route_weights: torch.Tensor,
        out: torch.Tensor | None = None,
        expert_frequency_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the grouped MLP from a flat variable-K route list.

        ``token_indices`` and ``expert_indices`` must be contiguous int32 and
        ``route_weights`` contiguous float32, all with shape ``[routes]``.
        Every edge is consumed independently, including duplicate
        ``(token, expert)`` pairs; weights are used verbatim.  IDs must be in
        range.  This hot path intentionally leaves value validation to the
        caller to avoid a device synchronization.

        When supplied, ``expert_frequency_out`` must be contiguous int32 with
        shape ``[num_experts]`` and receives the number of route occurrences
        for each expert.
        """

        if not hidden_states.is_cuda:
            raise ValueError("hidden_states must be on a ROCm device")
        with torch.cuda.device(hidden_states.device):
            return self._forward_routes_on_current_device(
                hidden_states,
                token_indices,
                expert_indices,
                route_weights,
                out,
                expert_frequency_out,
            )

    def _forward_routes_on_current_device(
        self,
        hidden_states: torch.Tensor,
        token_indices: torch.Tensor,
        expert_indices: torch.Tensor,
        route_weights: torch.Tensor,
        out: torch.Tensor | None,
        expert_frequency_out: torch.Tensor | None,
    ) -> torch.Tensor:
        tokens = self._validate_hidden(hidden_states)
        if token_indices.ndim != 1 or expert_indices.ndim != 1 or route_weights.ndim != 1:
            raise ValueError("token_indices, expert_indices, and route_weights must be one-dimensional")
        routes = int(route_weights.numel())
        if int(token_indices.numel()) != routes or int(expert_indices.numel()) != routes:
            raise ValueError("token_indices, expert_indices, and route_weights must have equal length")
        if routes > _MAX_SIGNED_I32:
            raise ValueError(f"route count exceeds the sorting kernel's signed 32-bit limit: {routes}")
        if (
            not token_indices.is_cuda
            or not expert_indices.is_cuda
            or not route_weights.is_cuda
            or token_indices.device != hidden_states.device
            or expert_indices.device != hidden_states.device
            or route_weights.device != hidden_states.device
        ):
            raise ValueError("route tensors must be on the same ROCm device as hidden_states")
        if token_indices.dtype != torch.int32 or expert_indices.dtype != torch.int32:
            raise TypeError(
                "token_indices/expert_indices must be int32, got " f"{token_indices.dtype}/{expert_indices.dtype}"
            )
        if route_weights.dtype != torch.float32:
            raise TypeError(f"route_weights must be float32, got {route_weights.dtype}")
        if not token_indices.is_contiguous() or not expert_indices.is_contiguous() or not route_weights.is_contiguous():
            raise ValueError("route tensors must be contiguous")
        if route_weights.requires_grad:
            raise ValueError("SonicMoE is inference-only; route_weights must not require gradients")

        workspace = self.reserve(tokens, routes=routes)
        output = self._validate_out(
            out,
            workspace,
            hidden_states,
            token_indices,
            expert_indices,
            route_weights,
            *self.weights.tensors,
        )

        frequency = workspace.expert_frequency
        if expert_frequency_out is not None:
            frequency = expert_frequency_out
            expected_frequency = (self.config.num_experts,)
            if tuple(frequency.shape) != expected_frequency:
                raise ValueError(
                    f"expert_frequency_out must have shape {expected_frequency}, " f"got {tuple(frequency.shape)}"
                )
            if (
                not frequency.is_cuda
                or frequency.device != hidden_states.device
                or frequency.dtype != torch.int32
                or not frequency.is_contiguous()
            ):
                raise ValueError("expert_frequency_out must be contiguous int32 on the same ROCm device")
            frequency_storage = frequency.untyped_storage().data_ptr()
            read_tensors = (
                hidden_states,
                token_indices,
                expert_indices,
                route_weights,
                *self.weights.tensors,
            )
            if frequency_storage == output.untyped_storage().data_ptr() or any(
                frequency_storage == tensor.untyped_storage().data_ptr() for tensor in read_tensors
            ):
                raise ValueError("expert_frequency_out must not alias an input or output")
            if frequency_storage in workspace.storage_ptrs:
                raise ValueError("expert_frequency_out must not alias internal workspace storage")

        assert workspace.sorting_workspace is not None
        with workspace._launch_lock:
            moe_ragged_sorting_flydsl(
                token_indices,
                expert_indices,
                route_weights,
                frequency,
                workspace.sorting_workspace,
                workspace.sorted_token_ids,
                workspace.sorted_weights,
                workspace.sorted_expert_ids,
                workspace.num_valid_ids,
                output,
                self.config.num_experts,
                tokens=tokens,
                max_padded_routes=workspace.max_padded_tokens,
                unit_size=self.config.tile_m,
            )
            if routes == 0:
                return output
            return self._run_grouped_gemms(hidden_states, workspace, output)

    def forward_topk(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the grouped MLP from precomputed route ids and weights.

        ``topk_ids`` must be contiguous int32 and ``topk_weights`` contiguous
        float32, both with shape ``[tokens, top_k]``.  Every token's expert IDs
        must be distinct and in ``[0, num_experts)``; this hot path intentionally
        avoids a device synchronization to validate their values.  Weights are
        consumed as-is; normalize them before this call when desired.
        """

        if not hidden_states.is_cuda:
            raise ValueError("hidden_states must be on a ROCm device")
        with torch.cuda.device(hidden_states.device):
            return self._forward_topk_on_current_device(
                hidden_states,
                topk_ids,
                topk_weights,
                out,
            )

    def _forward_topk_on_current_device(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        tokens = self._validate_hidden(hidden_states)
        expected = (tokens, self.config.top_k)
        if tuple(topk_ids.shape) != expected or tuple(topk_weights.shape) != expected:
            raise ValueError(
                f"topk_ids/topk_weights must both have shape {expected}, got "
                f"{tuple(topk_ids.shape)}/{tuple(topk_weights.shape)}"
            )
        if (
            not topk_ids.is_cuda
            or not topk_weights.is_cuda
            or topk_ids.device != hidden_states.device
            or topk_weights.device != hidden_states.device
        ):
            raise ValueError("topk ids/weights must be on the same ROCm device as hidden_states")
        if topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.float32:
            raise TypeError(
                f"topk_ids/topk_weights must be int32/float32, got " f"{topk_ids.dtype}/{topk_weights.dtype}"
            )
        if not topk_ids.is_contiguous() or not topk_weights.is_contiguous():
            raise ValueError("topk ids/weights must be contiguous")
        if topk_weights.requires_grad:
            raise ValueError("SonicMoE is inference-only; topk_weights must not require gradients")

        workspace = self.reserve(tokens)
        output = self._validate_out(
            out,
            workspace,
            hidden_states,
            topk_ids,
            topk_weights,
            *self.weights.tensors,
        )
        with workspace._launch_lock:
            moe_sorting_flydsl(
                topk_ids,
                topk_weights,
                workspace.sorted_token_ids,
                workspace.sorted_weights,
                workspace.sorted_expert_ids,
                workspace.num_valid_ids,
                output,
                self.config.num_experts,
                unit_size=self.config.tile_m,
                workspace=workspace.sorting_workspace,
            )
            return self._run_grouped_gemms(hidden_states, workspace, output)


@torch.no_grad()
def sonic_moe_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    router_logits: torch.Tensor,
    config: SonicMoEConfig,
    *,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Approximate oracle with FP32 GEMMs and A16 rounding boundaries.

    The production kernel uses unordered 16-bit atomic additions in stage 2, while
    this reference accumulates routes in FP32 and casts once at the end.  Small
    last-bit differences are therefore expected.
    """

    if tuple(hidden_states.shape) != (router_logits.shape[0], config.hidden_size):
        raise ValueError("hidden_states shape does not match config/router_logits")
    if tuple(w1.shape) != (
        config.num_experts,
        config.stage1_projection_size,
        config.hidden_size,
    ):
        raise ValueError("w1 shape does not match config")
    if tuple(w2.shape) != (
        config.num_experts,
        config.hidden_size,
        config.intermediate_size,
    ):
        raise ValueError("w2 shape does not match config")
    compute_dtype = _COMPUTE_DTYPES[config.compute_dtype]
    if hidden_states.dtype != compute_dtype:
        raise TypeError(f"hidden_states must use {compute_dtype} for this config, got " f"{hidden_states.dtype}")
    prepared_b1, prepared_b2 = _prepare_biases(b1, b2, w1, config)

    probs = torch.softmax(router_logits.float(), dim=-1)
    route_weights, route_ids = torch.topk(probs, config.top_k, dim=-1)
    if config.renormalize:
        route_weights = route_weights / route_weights.sum(dim=-1, keepdim=True)

    x = hidden_states.float()
    # Match the prepare API: arbitrary floating-point source weights are first
    # materialized in the configured A16 storage dtype before the kernel reads
    # them.  Keeping that boundary in the oracle matters for FP32 source weights.
    w1f = w1.to(compute_dtype).float()
    w2f = w2.to(compute_dtype).float()
    b1f = None if prepared_b1 is None else prepared_b1.float()
    b2f = None if prepared_b2 is None else prepared_b2.float()
    result = torch.zeros(
        (hidden_states.shape[0], config.hidden_size),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    for slot in range(config.top_k):
        expert = route_ids[:, slot]
        stage1 = torch.bmm(w1f[expert], x.unsqueeze(-1)).squeeze(-1)
        if b1f is not None:
            stage1 = stage1 + b1f[expert]
        # The legacy grouped GEMM materializes A16 preactivation before the
        # activation kernel reloads it in FP32. The fused kernel preserves this
        # observable rounding boundary.
        stage1 = stage1.to(compute_dtype).float()
        # Stage 1 stores an A16 sorted intermediate before stage 2 reloads it.
        if config.activation == "swiglu":
            gate, up = stage1.split(config.intermediate_size, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
        elif config.activation == "geglu":
            gate, up = stage1.split(config.intermediate_size, dim=-1)
            activated = torch.nn.functional.gelu(gate, approximate="tanh") * up
        elif config.activation == "reglu":
            gate, up = stage1.split(config.intermediate_size, dim=-1)
            activated = torch.nn.functional.relu(gate) * up
        elif config.activation == "gelu_tanh_approx":
            activated = torch.nn.functional.gelu(stage1, approximate="tanh")
        elif config.activation == "relu":
            activated = torch.nn.functional.relu(stage1)
        elif config.activation == "silu":
            activated = torch.nn.functional.silu(stage1)
        elif config.activation == "relu_sq":
            activated = torch.nn.functional.relu(stage1).square()
        else:  # guarded by SonicMoEConfig validation
            raise AssertionError(f"unexpected activation {config.activation!r}")
        activated = activated.to(compute_dtype).float()
        projected = torch.bmm(w2f[expert], activated.unsqueeze(-1)).squeeze(-1)
        if b2f is not None:
            projected = projected + b2f[expert]
        projected = projected.to(compute_dtype).float()
        result.add_(projected * route_weights[:, slot, None])
    return result.to(compute_dtype)


@torch.no_grad()
def sonic_moe_mxfp4_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    router_logits: torch.Tensor,
    config: SonicMoEConfig,
    *,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference the A16W4 path after canonical per-1x32 weight quantization.

    This intentionally quantizes the logical weights before calling
    :func:`sonic_moe_reference`, separating kernel error from the model-level
    error introduced by MXFP4 weights.
    """

    _validate_weight_inputs(w1, w2, config)
    w1_quant, w1_scale = _quantize_mxfp4_weight(w1)
    w2_quant, w2_scale = _quantize_mxfp4_weight(w2)
    w1_dequant = _dequantize_mxfp4_weight(w1_quant, w1_scale)
    w2_dequant = _dequantize_mxfp4_weight(w2_quant, w2_scale)
    return sonic_moe_reference(
        hidden_states,
        w1_dequant,
        w2_dequant,
        router_logits,
        config,
        b1=b1,
        b2=b2,
    )


__all__ = [
    "SonicMoE",
    "SonicMoEConfig",
    "SonicMoEWeights",
    "SonicMoEWorkspace",
    "prepare_sonic_bf16_weights",
    "prepare_sonic_fp16_weights",
    "prepare_sonic_mxfp4_weights",
    "sonic_moe_backward",
    "sonic_moe_backward_routes",
    "sonic_moe_mxfp4_reference",
    "sonic_moe_reference",
]
