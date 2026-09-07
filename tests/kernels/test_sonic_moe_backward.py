# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Correctness and contract tests for the first gfx950 SonicMoE backward."""

import math
import time
from types import SimpleNamespace

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.moe import sonic_backward as sonic_backward_module
from kernels.moe.moe_sorting_kernel import _multiphase_cf_cache, _oneshot_cf_cache
from kernels.moe.sonic import (
    SonicMoE,
    SonicMoEConfig,
    SonicMoEForwardState,
    prepare_sonic_bf16_weights,
    sonic_moe_backward,
    sonic_moe_backward_routes,
)
from kernels.moe.sonic_backward import (
    _grouped_da_tuning,
    _grouped_dw1_tuning,
    _grouped_dw2_stages,
    _grouped_dw2_tuning,
    _grouped_dx_tuning,
    _grouped_w1_tuning,
    _use_fused_da_dscore,
    _use_grouped_da,
    _use_grouped_dw1,
    _use_grouped_dw2,
    _use_grouped_dx,
    _use_grouped_w1_recompute,
    _use_grouped_w2_recompute,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_GLU_ACTIVATIONS = frozenset({"swiglu", "geglu", "reglu"})
_ACTIVATIONS = (
    "swiglu",
    "geglu",
    "reglu",
    "gelu_tanh_approx",
    "relu",
    "silu",
    "relu_sq",
)
_DTYPES = ((torch.bfloat16, "bf16"), (torch.float16, "fp16"))


@pytest.mark.parametrize(
    ("overrides", "expected"),
    (
        ({}, True),
        ({"reuse_forward_preactivation": False}, False),
        ({"use_hostless_grouped": False}, False),
        ({"use_compact_w1": False}, False),
        (
            {
                "use_hostless_grouped": False,
                "use_compact_w1": False,
                "use_large_grouped_dx": True,
            },
            True,
        ),
        ({"flat_routes": True}, False),
        ({"has_bias": True}, False),
        ({"compute_dtype": "fp16"}, False),
        ({"activation": "geglu"}, False),
    ),
)
def test_fused_da_dscore_policy_is_narrow(overrides, expected):
    kwargs = {
        "reuse_forward_preactivation": True,
        "use_hostless_grouped": True,
        "use_compact_w1": True,
        "use_large_grouped_dx": False,
        "flat_routes": False,
        "has_bias": False,
        "compute_dtype": "bf16",
        "activation": "swiglu",
    }
    kwargs.update(overrides)
    assert _use_fused_da_dscore(**kwargs) is expected


@pytest.mark.parametrize(
    ("max_expert_rows", "hidden_size", "intermediate_size", "expected"),
    (
        (1, 3584, 512, (64, 64, 32, 0, 2, 2)),
        (2, 3584, 512, (128, 128, 32, 0, 2, 2)),
        (4096, 4096, 2048, (128, 128, 32, 0, 2, 2)),
        (128, 192, 64, (128, 64, 32, 0, 2, 2)),
    ),
)
def test_grouped_dw1_tuning(max_expert_rows, hidden_size, intermediate_size, expected):
    assert _grouped_dw1_tuning(max_expert_rows, hidden_size, intermediate_size) == expected


@pytest.mark.parametrize(
    (
        "compute_dtype",
        "activation",
        "tokens",
        "routes",
        "flat_routes",
        "expected",
    ),
    (
        ("bf16", "swiglu", 1, 16, False, True),
        ("bf16", "swiglu", 4096, 32768, False, True),
        ("bf16", "swiglu", 4097, 32776, False, False),
        ("bf16", "swiglu", 128, 4096, True, True),
        ("bf16", "swiglu", 128, 4097, True, False),
        ("fp16", "swiglu", 128, 2048, False, False),
        ("bf16", "geglu", 128, 2048, False, False),
    ),
)
def test_grouped_dw1_policy(
    compute_dtype,
    activation,
    tokens,
    routes,
    flat_routes,
    expected,
):
    assert (
        _use_grouped_dw1(
            compute_dtype=compute_dtype,
            activation=activation,
            hidden_size=3584,
            intermediate_size=512,
            tokens=tokens,
            routes=routes,
            flat_routes=flat_routes,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("tokens", "routes", "flat_routes", "expected"),
    (
        (1, 16, False, True),
        (128, 2048, False, True),
        (129, 2064, False, False),
        (4096, 32768, False, False),
        (16, 128, True, True),
        (16, 129, True, False),
    ),
)
def test_grouped_w1_policy_bounds_worst_case_expert_rows(tokens, routes, flat_routes, expected):
    assert (
        _use_grouped_w1_recompute(
            compute_dtype="bf16",
            activation="swiglu",
            hidden_size=3584,
            intermediate_size=512,
            tokens=tokens,
            routes=routes,
            flat_routes=flat_routes,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("tokens", "routes", "flat_routes", "expected"),
    (
        (1, 16, False, True),
        (128, 2048, False, True),
        (129, 2064, False, False),
        (4096, 32768, False, False),
        (16, 128, True, True),
        (16, 129, True, False),
    ),
)
def test_grouped_w2_policy_bounds_worst_case_expert_rows(tokens, routes, flat_routes, expected):
    assert (
        _use_grouped_w2_recompute(
            compute_dtype="bf16",
            activation="swiglu",
            hidden_size=3584,
            intermediate_size=512,
            tokens=tokens,
            routes=routes,
            flat_routes=flat_routes,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("compute_dtype", "activation", "hidden_size", "intermediate_size"),
    (
        ("fp16", "swiglu", 3584, 512),
        ("bf16", "geglu", 3584, 512),
        ("bf16", "swiglu", 3552, 512),
        ("bf16", "swiglu", 3584, 480),
    ),
)
def test_grouped_w2_policy_keeps_unsupported_contracts_on_legacy(
    compute_dtype,
    activation,
    hidden_size,
    intermediate_size,
):
    assert not _use_grouped_w2_recompute(
        compute_dtype=compute_dtype,
        activation=activation,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        tokens=128,
        routes=2048,
        flat_routes=False,
    )


@pytest.mark.parametrize(
    ("compute_dtype", "hidden_size", "intermediate_size", "expected"),
    (
        ("bf16", 3584, 512, True),
        ("bf16", 128, 64, True),
        ("fp16", 3584, 512, False),
        ("bf16", 3552, 512, False),
        ("bf16", 3584, 480, False),
    ),
)
def test_grouped_dw2_policy(compute_dtype, hidden_size, intermediate_size, expected):
    assert (
        _use_grouped_dw2(
            compute_dtype=compute_dtype,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        is expected
    )


@pytest.mark.parametrize(
    (
        "max_expert_rows",
        "active_experts",
        "hidden_size",
        "intermediate_size",
        "expected",
    ),
    (
        (1, 16, 3584, 512, (128, 128, 32, 0, 2, 2)),
        (3, 896, 3584, 512, (128, 256, 32, 0, 2, 4)),
        (128, 16, 3584, 512, (128, 128, 32, 0, 2, 2)),
        (74, 896, 3584, 512, (256, 256, 32, 0, 4, 4)),
        (128, 4, 128, 64, (128, 64, 32, 0, 2, 2)),
    ),
)
def test_grouped_dw2_tuning(
    max_expert_rows,
    active_experts,
    hidden_size,
    intermediate_size,
    expected,
):
    assert (
        _grouped_dw2_tuning(
            max_expert_rows,
            hidden_size,
            intermediate_size,
            active_experts=active_experts,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("max_expert_rows", "expected"),
    ((1, 2), (63, 2), (64, 3), (4096, 3)),
)
def test_grouped_dw2_pipeline_depth(max_expert_rows, expected):
    assert _grouped_dw2_stages(max_expert_rows) == expected


@pytest.mark.parametrize(
    ("tokens", "routes", "flat_routes", "expected"),
    (
        (1, 16, False, True),
        (128, 2048, False, True),
        (129, 2064, False, True),
        (4096, 32768, False, True),
        (4097, 32776, False, False),
        (16, 128, True, True),
        (16, 4096, True, True),
        (16, 4097, True, False),
    ),
)
def test_grouped_da_policy_bounds_worst_case_expert_rows(tokens, routes, flat_routes, expected):
    assert (
        _use_grouped_da(
            compute_dtype="bf16",
            activation="swiglu",
            hidden_size=3584,
            intermediate_size=512,
            tokens=tokens,
            routes=routes,
            flat_routes=flat_routes,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("max_expert_rows", "hidden_size", "expected"),
    (
        (1, 3584, (16, 64, 128, 1, 4)),
        (1, 64, (32, 64, 64, 2, 2)),
        (2, 3584, (32, 64, 64, 2, 2)),
        (16, 3584, (32, 64, 64, 2, 2)),
        (17, 3584, (64, 64, 64, 2, 2)),
        (512, 4096, (64, 64, 64, 2, 2)),
    ),
)
def test_grouped_da_tuning_tracks_actual_expert_rows(max_expert_rows, hidden_size, expected):
    assert _grouped_da_tuning(max_expert_rows, hidden_size) == expected


@pytest.mark.parametrize(
    ("compute_dtype", "activation", "hidden_size", "intermediate_size"),
    (
        ("fp16", "swiglu", 3584, 512),
        ("bf16", "geglu", 3584, 512),
        ("bf16", "swiglu", 3552, 512),
        ("bf16", "swiglu", 3584, 480),
    ),
)
def test_grouped_da_policy_keeps_unsupported_contracts_on_legacy(
    compute_dtype,
    activation,
    hidden_size,
    intermediate_size,
):
    assert not _use_grouped_da(
        compute_dtype=compute_dtype,
        activation=activation,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        tokens=128,
        routes=2048,
        flat_routes=False,
    )


@pytest.mark.parametrize(
    ("tokens", "expected_compact"),
    ((1, False), (63, False), (64, True), (128, True)),
)
def test_grouped_w1_compact_queue_policy(tokens, expected_compact):
    bm, bn, bk, k_wave, compact = _grouped_w1_tuning(
        tokens=tokens,
        hidden_size=3584,
        intermediate_size=512,
    )
    assert compact is expected_compact
    assert (bm, bn, bk, k_wave) == ((16, 128, 64, 2) if expected_compact else (16, 64, 64, 4))


@pytest.mark.parametrize(
    (
        "compute_dtype",
        "activation",
        "hidden_size",
        "intermediate_size",
        "tokens",
        "num_experts",
        "topk",
        "flat_routes",
        "compact_w1",
        "expected",
    ),
    (
        ("bf16", "swiglu", 3584, 512, 1, 64, 8, False, False, True),
        ("bf16", "swiglu", 3584, 512, 7, 64, 8, False, False, False),
        ("bf16", "swiglu", 3584, 512, 64, 64, 8, False, True, True),
        ("bf16", "swiglu", 3584, 512, 128, 64, 8, True, True, True),
        ("bf16", "swiglu", 3584, 512, 1, 64, 8, True, False, False),
        ("bf16", "swiglu", 3584, 512, 4096, 64, 8, False, False, False),
        ("bf16", "swiglu", 4096, 2048, 4096, 64, 8, False, False, True),
        ("bf16", "swiglu", 4096, 2048, 4096, 65, 8, False, False, False),
        ("bf16", "swiglu", 4096, 2048, 4096, 64, 8, True, False, False),
        ("fp16", "swiglu", 3584, 512, 64, 64, 8, False, True, False),
        ("bf16", "geglu", 3584, 512, 64, 64, 8, False, True, False),
        ("bf16", "swiglu", 3552, 512, 64, 64, 8, False, True, False),
        ("bf16", "swiglu", 3584, 500, 64, 64, 8, False, True, False),
    ),
)
def test_grouped_dx_policy(
    compute_dtype,
    activation,
    hidden_size,
    intermediate_size,
    tokens,
    num_experts,
    topk,
    flat_routes,
    compact_w1,
    expected,
):
    assert (
        _use_grouped_dx(
            compute_dtype=compute_dtype,
            activation=activation,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tokens=tokens,
            num_experts=num_experts,
            topk=topk,
            flat_routes=flat_routes,
            compact_w1=compact_w1,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("active_experts", "hidden_size", "expected"),
    (
        (16, 3584, (128, 2)),
        (255, 3584, (128, 2)),
        (256, 3584, (256, 4)),
        (896, 3584, (256, 4)),
        (896, 128, (128, 2)),
        (896, 192, (64, 2)),
    ),
)
def test_grouped_dx_tuning(active_experts, hidden_size, expected):
    assert _grouped_dx_tuning(active_experts, hidden_size) == expected


def _gfx950_device():
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU is required")
    arch = str(get_rocm_arch())
    if "gfx950" not in arch:
        pytest.skip(f"SonicMoE backward test requires gfx950, found {arch}")
    return torch.device("cuda")


def _config(hidden_size, intermediate_size, num_experts, topk, **overrides):
    values = {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_experts": num_experts,
        "top_k": topk,
        "tile_m": 32,
        "tile_n": 64 if intermediate_size == 64 else 128,
        "tile_k": 128,
        "down_tile_n": 128,
        "down_tile_k": 64 if intermediate_size == 64 else 128,
    }
    values.update(overrides)
    return SonicMoEConfig(**values)


def _make_case(
    tokens,
    hidden_size,
    intermediate_size,
    num_experts,
    topk,
    seed,
    *,
    activation="swiglu",
    dtype=torch.bfloat16,
):
    device = _gfx950_device()
    generator = torch.Generator(device=device).manual_seed(seed)
    projection_size = intermediate_size * (2 if activation in _GLU_ACTIVATIONS else 1)
    x = torch.randn(
        (tokens, hidden_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(dtype)
    w1 = (
        torch.randn(
            (num_experts, projection_size, hidden_size),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        / math.sqrt(hidden_size)
    ).to(dtype)
    w2 = (
        torch.randn(
            (num_experts, hidden_size, intermediate_size),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        / math.sqrt(intermediate_size)
    ).to(dtype)
    # Leave the final expert empty while preserving distinct ids per token.
    active_experts = num_experts - 1
    ids_host = [[(token + slot) % active_experts for slot in range(topk)] for token in range(tokens)]
    topk_ids = torch.tensor(ids_host, dtype=torch.int32, device=device)
    topk_weights = torch.rand(
        (tokens, topk),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    topk_weights[0, 0] = 0.0
    grad_output = torch.randn(
        (tokens, hidden_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(dtype)
    return x, w1, w2, topk_ids, topk_weights, grad_output


def _make_biases(w1, w2, seed):
    generator = torch.Generator(device=w1.device).manual_seed(seed)
    b1 = (
        torch.randn(
            w1.shape[:2],
            dtype=torch.float32,
            device=w1.device,
            generator=generator,
        )
        / math.sqrt(w1.shape[-1])
    ).to(w1.dtype)
    b2 = (
        torch.randn(
            (w2.shape[0], w2.shape[1]),
            dtype=torch.float32,
            device=w2.device,
            generator=generator,
        )
        / math.sqrt(w2.shape[-1])
    ).to(w1.dtype)
    return b1, b2


@torch.no_grad()
def _make_forward_state(
    x,
    w1,
    topk_ids,
    config,
    *,
    b1=None,
    interleaved_w1=False,
):
    """Build the route-order state contract without depending on forward API."""

    tokens, topk = topk_ids.shape
    projection_size = 2 * config.intermediate_size
    preactivation = torch.empty(
        (tokens, topk, projection_size),
        dtype=torch.bfloat16,
        device=x.device,
    )
    for expert in range(config.num_experts):
        pairs = (topk_ids == expert).nonzero(as_tuple=False)
        if pairs.numel() == 0:
            continue
        token_indices, slots = pairs[:, 0], pairs[:, 1]
        values = x[token_indices].float() @ w1[expert].float().transpose(0, 1)
        if b1 is not None:
            values = values + b1[expert].float()
        preactivation[token_indices, slots] = values.to(torch.bfloat16)

    stream = torch.cuda.current_stream(x.device)
    ready_event = torch.cuda.Event()
    ready_event.record(stream)
    return SimpleNamespace(
        preactivation=preactivation,
        tokens=tokens,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_experts=config.num_experts,
        top_k=config.top_k,
        activation=config.activation,
        compute_dtype=config.compute_dtype,
        interleaved_w1=interleaved_w1,
        has_bias=b1 is not None,
        producer_stream=int(stream.cuda_stream),
        ready_event=ready_event,
    )


def _tanh_reference(value):
    exp_value = torch.exp2(value.abs() * (-2.0 * math.log2(math.e)))
    tanh_abs = (1.0 - exp_value) / (1.0 + exp_value)
    return torch.where(value > 0, tanh_abs, -tanh_abs)


def _gelu_tanh_reference(value):
    inner = math.sqrt(2.0 / math.pi) * (value + 0.044715 * value.square() * value)
    tanh_inner = _tanh_reference(inner)
    activated = 0.5 * value * (1.0 + tanh_inner)
    derivative = 0.5 * (1.0 + tanh_inner) + (
        0.5 * value * (1.0 - tanh_inner.square()) * math.sqrt(2.0 / math.pi) * (1.0 + 3.0 * 0.044715 * value.square())
    )
    return activated, derivative


def _interleave_glu_rows(tensor):
    """Convert expert-major ``[gate..., up...]`` rows to ``[g0,u0,...]``."""

    gate, up = tensor.chunk(2, dim=1)
    return torch.stack((gate, up), dim=2).flatten(1, 2).contiguous()


def _activation_reference(
    preactivation,
    da,
    intermediate_size,
    activation_name,
    *,
    interleaved_w1=False,
):
    if activation_name in _GLU_ACTIVATIONS:
        if interleaved_w1:
            gate, up = preactivation.float().reshape(-1, intermediate_size, 2).unbind(dim=2)
        else:
            gate, up = preactivation.float().split(intermediate_size, dim=1)
        if activation_name == "swiglu":
            sigmoid = torch.sigmoid(gate)
            activated_gate = gate * sigmoid
            derivative = sigmoid * (1.0 + gate * (1.0 - sigmoid))
        elif activation_name == "geglu":
            activated_gate, derivative = _gelu_tanh_reference(gate)
        else:
            activated_gate = torch.relu(gate)
            derivative = (gate > 0).float()
        activated = activated_gate * up
        dz_gate = da * up * derivative
        dz_up = da * activated_gate
        dz = (
            torch.stack((dz_gate, dz_up), dim=2).flatten(1, 2) if interleaved_w1 else torch.cat((dz_gate, dz_up), dim=1)
        )
    else:
        value = preactivation.float()
        if activation_name == "gelu_tanh_approx":
            activated, derivative = _gelu_tanh_reference(value)
        elif activation_name == "relu":
            activated = torch.relu(value)
            derivative = (value > 0).float()
        elif activation_name == "silu":
            sigmoid = torch.sigmoid(value)
            activated = value * sigmoid
            derivative = sigmoid * (1.0 + value * (1.0 - sigmoid))
        elif activation_name == "relu_sq":
            relu = torch.relu(value)
            activated = relu.square()
            derivative = 2.0 * relu
        else:
            raise AssertionError(f"unexpected activation {activation_name!r}")
        dz = da * derivative
    return activated.to(preactivation.dtype), dz.to(preactivation.dtype)


@torch.no_grad()
def _backward_reference(
    x,
    w1,
    w2,
    topk_ids,
    topk_weights,
    grad_output,
    *,
    activation_name="swiglu",
    b1=None,
    b2=None,
    interleaved_w1=False,
    reassociate_da_dscore=False,
):
    """Match the standalone backward's explicit A16 materialization contract."""

    tokens, hidden_size = x.shape
    num_experts, projection_size, _ = w1.shape
    topk = topk_ids.shape[1]
    intermediate_size = w2.shape[-1]
    expected_projection_size = intermediate_size * (2 if activation_name in _GLU_ACTIVATIONS else 1)
    assert projection_size == expected_projection_size
    dx_routes = torch.empty(
        (tokens, topk, hidden_size),
        dtype=x.dtype,
        device=x.device,
    )
    dw1 = torch.zeros_like(w1)
    dw2 = torch.zeros_like(w2)
    dtopk_weights = torch.empty_like(topk_weights)
    has_bias = b1 is not None
    assert has_bias == (b2 is not None)
    db1 = torch.zeros_like(b1) if b1 is not None else None
    db2 = torch.zeros_like(b2) if b2 is not None else None

    for expert in range(num_experts):
        pairs = (topk_ids == expert).nonzero(as_tuple=False)
        if pairs.numel() == 0:
            continue
        token_indices, slots = pairs[:, 0], pairs[:, 1]
        x_e = x[token_indices]
        dout_e = grad_output[token_indices]
        scores_e = topk_weights[token_indices, slots]

        preactivation = x_e.float() @ w1[expert].float().transpose(0, 1)
        if b1 is not None:
            preactivation = preactivation + b1[expert].float()
        preactivation = preactivation.to(x.dtype)
        # da is computed before dz, while activation is needed by projection
        # and dW2.  Passing a zero placeholder here avoids duplicating the
        # activation formulas; dz is recomputed after da is available.
        activation, _ = _activation_reference(
            preactivation,
            torch.zeros(
                (preactivation.shape[0], intermediate_size),
                dtype=torch.float32,
                device=preactivation.device,
            ),
            intermediate_size,
            activation_name,
            interleaved_w1=interleaved_w1,
        )
        dy = (dout_e.float() * scores_e[:, None]).to(x.dtype)
        if reassociate_da_dscore:
            assert b2 is None
            # The fused gfx950 state path computes the mathematically
            # equivalent q contraction once, then reuses its A16 boundary for
            # both gradients instead of materializing activation @ W2.T.
            q = (dout_e.float() @ w2[expert].float()).to(x.dtype).float()
            dtopk_weights[token_indices, slots] = (q * activation.float()).sum(dim=1)
            da = q * scores_e[:, None]
        else:
            projection = activation.float() @ w2[expert].float().transpose(0, 1)
            if b2 is not None:
                projection = projection + b2[expert].float()
            projection = projection.to(x.dtype)
            dtopk_weights[token_indices, slots] = (dout_e.float() * projection.float()).sum(dim=1)
            da = (dy.float() @ w2[expert].float()).to(x.dtype).float()
        _, dz = _activation_reference(
            preactivation,
            da,
            intermediate_size,
            activation_name,
            interleaved_w1=interleaved_w1,
        )

        dw2[expert] = (dy.float().transpose(0, 1) @ activation.float()).to(x.dtype)
        dw1[expert] = (dz.float().transpose(0, 1) @ x_e.float()).to(x.dtype)
        dx_routes[token_indices, slots] = (dz.float() @ w1[expert].float()).to(x.dtype)
        if db1 is not None:
            db1[expert] = dz.float().sum(dim=0).to(x.dtype)
            db2[expert] = dy.float().sum(dim=0).to(x.dtype)

    dx = dx_routes.float().sum(dim=1).to(x.dtype)
    result = (dx, dw1, dw2, dtopk_weights)
    return (*result, db1, db2) if has_bias else result


@torch.no_grad()
def _backward_routes_reference(
    x,
    w1,
    w2,
    token_indices,
    expert_indices,
    route_weights,
    grad_output,
    *,
    activation_name="swiglu",
    b1=None,
    b2=None,
    interleaved_w1=False,
):
    """Match the flat-route backward's explicit A16 boundaries."""

    tokens, hidden_size = x.shape
    num_experts, projection_size, _ = w1.shape
    routes = int(route_weights.numel())
    intermediate_size = w2.shape[-1]
    expected_projection_size = intermediate_size * (2 if activation_name in _GLU_ACTIVATIONS else 1)
    assert projection_size == expected_projection_size
    dx_routes = torch.empty((routes, hidden_size), dtype=x.dtype, device=x.device)
    dw1 = torch.zeros_like(w1)
    dw2 = torch.zeros_like(w2)
    droute_weights = torch.empty_like(route_weights)
    has_bias = b1 is not None
    assert has_bias == (b2 is not None)
    db1 = torch.zeros_like(b1) if b1 is not None else None
    db2 = torch.zeros_like(b2) if b2 is not None else None

    for expert in range(num_experts):
        route_ids = (expert_indices == expert).nonzero(as_tuple=False).flatten()
        if route_ids.numel() == 0:
            continue
        tokens_e = token_indices[route_ids].long()
        x_e = x[tokens_e]
        dout_e = grad_output[tokens_e]
        scores_e = route_weights[route_ids]

        preactivation = x_e.float() @ w1[expert].float().transpose(0, 1)
        if b1 is not None:
            preactivation = preactivation + b1[expert].float()
        preactivation = preactivation.to(x.dtype)
        activation, _ = _activation_reference(
            preactivation,
            torch.zeros(
                (preactivation.shape[0], intermediate_size),
                dtype=torch.float32,
                device=x.device,
            ),
            intermediate_size,
            activation_name,
            interleaved_w1=interleaved_w1,
        )
        projection = activation.float() @ w2[expert].float().transpose(0, 1)
        if b2 is not None:
            projection = projection + b2[expert].float()
        projection = projection.to(x.dtype)
        droute_weights[route_ids] = (dout_e.float() * projection.float()).sum(dim=1)
        dy = (dout_e.float() * scores_e[:, None]).to(x.dtype)
        da = (dy.float() @ w2[expert].float()).to(x.dtype).float()
        _, dz = _activation_reference(
            preactivation,
            da,
            intermediate_size,
            activation_name,
            interleaved_w1=interleaved_w1,
        )

        dw2[expert] = (dy.float().transpose(0, 1) @ activation.float()).to(x.dtype)
        dw1[expert] = (dz.float().transpose(0, 1) @ x_e.float()).to(x.dtype)
        dx_routes[route_ids] = (dz.float() @ w1[expert].float()).to(x.dtype)
        if db1 is not None:
            db1[expert] = dz.float().sum(dim=0).to(x.dtype)
            db2[expert] = dy.float().sum(dim=0).to(x.dtype)

    dx_fp32 = torch.zeros((tokens, hidden_size), dtype=torch.float32, device=x.device)
    if routes:
        dx_fp32.index_add_(0, token_indices.long(), dx_routes.float())
    result = (dx_fp32.to(x.dtype), dw1, dw2, droute_weights)
    return (*result, db1, db2) if has_bias else result


def test_sonic_moe_backward_t1_keeps_expert_grid_without_descriptor_builder(monkeypatch):
    """The latency-critical T1 grouped W1 path must not launch queue builders."""

    tokens, hidden_size, intermediate_size, num_experts, topk = 1, 256, 128, 4, 2
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        compute_dtype="bf16",
        down_tile_m=128,
    )
    args = _make_case(
        tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        seed=197,
        dtype=torch.bfloat16,
    )

    def _unexpected_builder(*_args, **_kwargs):
        raise AssertionError("T1 must not build compact W1 descriptors")

    def _unexpected_active_builder(*_args, **_kwargs):
        raise AssertionError("T1 dW2 must consume sorter metadata directly")

    original_metadata_tn = sonic_backward_module.grouped_tn_from_metadata_flydsl
    metadata_tn_calls = 0

    def _tracked_metadata_tn(*tn_args, **tn_kwargs):
        nonlocal metadata_tn_calls
        metadata_tn_calls += 1
        return original_metadata_tn(*tn_args, **tn_kwargs)

    monkeypatch.setattr(
        sonic_backward_module,
        "build_compact_m_tile_descriptors",
        _unexpected_builder,
    )
    monkeypatch.setattr(
        sonic_backward_module,
        "build_active_expert_queue_flydsl",
        _unexpected_active_builder,
    )
    monkeypatch.setattr(
        sonic_backward_module,
        "grouped_tn_from_metadata_flydsl",
        _tracked_metadata_tn,
    )
    actual = sonic_moe_backward(*args, config)
    expected = _backward_reference(*args)
    torch.cuda.synchronize()
    for actual_gradient, expected_gradient in zip(actual[:3], expected[:3]):
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=3e-2,
            atol=5e-2,
        )
    assert metadata_tn_calls == 2
    torch.testing.assert_close(actual[3], expected[3], rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("tokens", (1, 64, 128))
def test_sonic_moe_backward_grouped_short_path_never_materializes_host_segments(
    monkeypatch,
    tokens,
):
    """Fully grouped fixed-K backward keeps sorter extents device-resident."""

    hidden_size, intermediate_size, num_experts, topk = 256, 128, 8, 4
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        compute_dtype="bf16",
        down_tile_m=128,
    )
    args = _make_case(
        tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        seed=503 + tokens,
        dtype=torch.bfloat16,
    )

    def _unexpected_host_segments(*_args, **_kwargs):
        raise AssertionError("fully grouped short backward must not copy frequencies to host")

    monkeypatch.setattr(
        sonic_backward_module,
        "_materialize_expert_segments",
        _unexpected_host_segments,
    )
    actual = sonic_moe_backward(*args, config)
    expected = _backward_reference(*args)
    torch.cuda.synchronize()

    for actual_gradient, expected_gradient in zip(actual[:3], expected[:3]):
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=3e-2,
            atol=5e-2,
        )
    # The score dot follows the kernel's wave-reduction order rather than
    # PyTorch's GEMM reduction order; this test's purpose is the no-readback
    # dispatch contract, while the dedicated numerical tests use tighter
    # shape-specific bounds.
    torch.testing.assert_close(actual[3], expected[3], rtol=2e-3, atol=6e-3)


@pytest.mark.parametrize(
    "tokens,hidden_size,intermediate_size,num_experts,topk",
    (
        (7, 128, 64, 4, 2),
        (9, 256, 128, 5, 1),
        (65, 128, 64, 3, 2),
        (11, 512, 256, 8, 4),
    ),
)
@pytest.mark.parametrize("dtype,compute_dtype", _DTYPES, ids=("bf16", "fp16"))
def test_sonic_moe_backward_matches_a16_reference(
    tokens,
    hidden_size,
    intermediate_size,
    num_experts,
    topk,
    dtype,
    compute_dtype,
):
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        compute_dtype=compute_dtype,
        down_tile_m=128,
    )
    args = _make_case(
        tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        seed=211 + topk,
        dtype=dtype,
    )

    actual = sonic_moe_backward(*args, config)
    expected = _backward_reference(*args)
    torch.cuda.synchronize()

    assert len(actual) == 4
    for actual_gradient, expected_gradient in zip(actual[:3], expected[:3]):
        assert actual_gradient.shape == expected_gradient.shape
        assert actual_gradient.dtype == expected_gradient.dtype
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=3e-2,
            atol=5e-2,
        )
    torch.testing.assert_close(actual[3], expected[3], rtol=5e-4, atol=5e-4)

    # The intentionally unused expert must receive exact zero weight grads.
    assert torch.count_nonzero(actual[1][-1]) == 0
    assert torch.count_nonzero(actual[2][-1]) == 0


@pytest.mark.parametrize("activation_name", tuple(sorted(_GLU_ACTIVATIONS)))
@pytest.mark.parametrize("dtype,compute_dtype", _DTYPES, ids=("bf16", "fp16"))
def test_sonic_moe_backward_native_interleaved_w1_matches_reference(
    activation_name,
    dtype,
    compute_dtype,
):
    """Fixed-K preserves native GLU row order through W1, dW1, B1, and dB1."""

    tokens, hidden_size, intermediate_size, num_experts, topk = 1, 256, 128, 4, 2
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        activation=activation_name,
        compute_dtype=compute_dtype,
        down_tile_m=128,
    )
    args = list(
        _make_case(
            tokens,
            hidden_size,
            intermediate_size,
            num_experts,
            topk,
            seed=443,
            activation=activation_name,
            dtype=dtype,
        )
    )
    b1, b2 = _make_biases(args[1], args[2], seed=449)
    args[1] = _interleave_glu_rows(args[1])
    b1 = _interleave_glu_rows(b1)
    args = tuple(args)

    actual = sonic_moe_backward(
        *args,
        config,
        b1=b1,
        b2=b2,
        interleaved_w1=True,
    )
    expected = _backward_reference(
        *args,
        activation_name=activation_name,
        b1=b1,
        b2=b2,
        interleaved_w1=True,
    )
    torch.cuda.synchronize()

    for actual_gradient, expected_gradient in zip(actual, expected):
        if actual_gradient.dtype == torch.float32:
            rtol, atol = 5e-4, 5e-4
        else:
            rtol, atol = 3e-2, 5e-2
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=rtol,
            atol=atol,
        )


def test_sonic_moe_backward_grouped_w1_spans_sort_blocks_with_bias(monkeypatch):
    """The device-driven W1 path handles a real M tail past one sort block."""

    tokens, hidden_size, intermediate_size, num_experts, topk = 65, 512, 256, 4, 2
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        compute_dtype="bf16",
        down_tile_m=128,
    )
    args = list(
        _make_case(
            tokens,
            hidden_size,
            intermediate_size,
            num_experts,
            topk,
            seed=367,
            dtype=torch.bfloat16,
        )
    )
    # Sixty-five routes per active expert cross a 64-row sorter boundary and
    # require five BM16 compute tiles, with one live row in the final tile. The
    # second route remains a distinct expert because the dense sorter contract
    # assumes unique top-k expert ids within a token.
    args[3][:, 0].fill_(2)
    args[3][:, 1].fill_(0)
    args = tuple(args)
    b1, b2 = _make_biases(args[1], args[2], seed=373)

    original_builder = sonic_backward_module.build_compact_m_tile_descriptors
    builder_calls = 0
    emitted_active_queue = False

    def _tracked_builder(*builder_args, **builder_kwargs):
        nonlocal builder_calls, emitted_active_queue
        builder_calls += 1
        emitted_active_queue = builder_kwargs.get("active_expert_storage") is not None
        return original_builder(*builder_args, **builder_kwargs)

    def _unexpected_active_builder(*_args, **_kwargs):
        raise AssertionError("compact W1 and dW2 must share one active-expert queue")

    monkeypatch.setattr(
        sonic_backward_module,
        "build_compact_m_tile_descriptors",
        _tracked_builder,
    )
    monkeypatch.setattr(
        sonic_backward_module,
        "build_active_expert_queue_flydsl",
        _unexpected_active_builder,
    )
    actual = sonic_moe_backward(*args, config, b1=b1, b2=b2)
    expected = _backward_reference(*args, b1=b1, b2=b2)
    torch.cuda.synchronize()

    # W1 recompute and dX share one BM16 queue; the same builder also emits
    # the active-expert queue consumed by grouped dW1 and dW2.
    assert builder_calls == 1
    assert emitted_active_queue

    for actual_gradient, expected_gradient in zip(actual, expected):
        if actual_gradient.dtype == torch.float32:
            rtol, atol = 5e-4, 5e-4
        else:
            rtol, atol = 3e-2, 5e-2
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=rtol,
            atol=atol,
        )


def test_sonic_moe_backward_grouped_w1_handles_e896_active_and_empty_experts():
    """The expert-grid specialization skips empty experts at production E."""

    tokens, hidden_size, intermediate_size, num_experts, topk = 7, 512, 256, 896, 2
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        compute_dtype="bf16",
        down_tile_m=128,
    )
    args = list(
        _make_case(
            tokens,
            hidden_size,
            intermediate_size,
            num_experts,
            topk,
            seed=379,
            dtype=torch.bfloat16,
        )
    )
    args[3][:, 0].fill_(0)
    args[3][:, 1].fill_(num_experts - 1)
    args = tuple(args)

    actual = sonic_moe_backward(*args, config)
    reduced_ids = torch.stack(
        (
            torch.zeros(tokens, dtype=torch.int32, device=args[0].device),
            torch.ones(tokens, dtype=torch.int32, device=args[0].device),
        ),
        dim=1,
    )
    expected = _backward_reference(
        args[0],
        args[1][[0, num_experts - 1]].contiguous(),
        args[2][[0, num_experts - 1]].contiguous(),
        reduced_ids,
        args[4],
        args[5],
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual[0].float(), expected[0].float(), rtol=3e-2, atol=5e-2)
    torch.testing.assert_close(actual[3], expected[3], rtol=5e-4, atol=5e-4)
    for actual_gradient, expected_gradient in (
        (actual[1][0], expected[1][0]),
        (actual[1][-1], expected[1][1]),
        (actual[2][0], expected[2][0]),
        (actual[2][-1], expected[2][1]),
    ):
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=3e-2,
            atol=5e-2,
        )
    assert torch.count_nonzero(actual[1][1:-1]) == 0
    assert torch.count_nonzero(actual[2][1:-1]) == 0


def test_sonic_moe_backward_grouped_dx_t1_handles_e896_metadata_grid():
    """T1 dX maps sparse first/last experts through sorter metadata only."""

    tokens, hidden_size, intermediate_size, num_experts, topk = 1, 128, 64, 896, 2
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        compute_dtype="bf16",
        down_tile_m=128,
    )
    args = list(
        _make_case(
            tokens,
            hidden_size,
            intermediate_size,
            num_experts,
            topk,
            seed=431,
            dtype=torch.bfloat16,
        )
    )
    args[3][0, 0] = 0
    args[3][0, 1] = num_experts - 1
    args = tuple(args)

    actual = sonic_moe_backward(*args, config)
    expected = _backward_reference(*args)
    torch.cuda.synchronize()

    for actual_gradient, expected_gradient in zip(actual[:3], expected[:3]):
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=3e-2,
            atol=5e-2,
        )
    torch.testing.assert_close(actual[3], expected[3], rtol=5e-4, atol=5e-4)
    assert torch.count_nonzero(actual[1][1:-1]) == 0
    assert torch.count_nonzero(actual[2][1:-1]) == 0


def test_sonic_moe_backward_large_grouped_dx_uses_independent_bm64_queue(
    monkeypatch,
):
    """The long-token path builds BM64 dX work without a compact W1 queue."""

    tokens, hidden_size, intermediate_size, num_experts, topk = 129, 256, 128, 4, 2
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        compute_dtype="bf16",
        down_tile_m=128,
    )
    args = list(
        _make_case(
            tokens,
            hidden_size,
            intermediate_size,
            num_experts,
            topk,
            seed=439,
            dtype=torch.bfloat16,
        )
    )
    # Deterministically span three sorter blocks per live expert and leave a
    # partial BM64 tail.  Fixed-K still requires unique experts per token.
    args[3][:, 0] = 0
    args[3][:, 1] = num_experts - 1
    args = tuple(args)

    monkeypatch.setattr(
        sonic_backward_module,
        "_use_large_grouped_dx_descriptor_queue",
        lambda **_kwargs: True,
    )
    original_builder = sonic_backward_module.build_compact_m_tile_descriptors
    original_compile = sonic_backward_module._compile_grouped_dx
    builder_kwargs = []
    compiled_block_m = []

    def _tracked_builder(*builder_args, **kwargs):
        builder_kwargs.append(kwargs)
        return original_builder(*builder_args, **kwargs)

    def _tracked_compile(*compile_args, **kwargs):
        compiled_block_m.append(compile_args[3])
        return original_compile(*compile_args, **kwargs)

    monkeypatch.setattr(
        sonic_backward_module,
        "build_compact_m_tile_descriptors",
        _tracked_builder,
    )
    monkeypatch.setattr(
        sonic_backward_module,
        "_compile_grouped_dx",
        _tracked_compile,
    )

    torch.cuda.synchronize()
    stream = torch.cuda.Stream(device=args[0].device)
    with torch.cuda.stream(stream):
        actual = sonic_moe_backward(*args, config)
    stream.synchronize()
    expected = _backward_reference(*args)
    torch.cuda.synchronize()

    assert len(builder_kwargs) == 1
    assert builder_kwargs[0]["block_m"] == 64
    assert builder_kwargs[0]["sorted_block_m"] == 64
    assert builder_kwargs[0]["active_expert_storage"] is not None
    assert compiled_block_m == [64]
    for actual_gradient, expected_gradient in zip(actual[:3], expected[:3]):
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=3e-2,
            atol=5e-2,
        )
    torch.testing.assert_close(actual[3], expected[3], rtol=5e-3, atol=5e-2)


@pytest.mark.parametrize("tokens", (65, 129), ids=("oneshot", "multiphase"))
def test_sonic_moe_forward_then_backward_separates_sorter_tensor_ranks(tokens):
    """Forward and backward use rank-2/rank-1 sorter scratch buffers."""

    hidden_size, intermediate_size, num_experts, topk = 128, 64, 4, 2
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        tile_m=64,
    )
    args = _make_case(
        tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        seed=271 + tokens,
    )
    x, w1, w2, topk_ids, topk_weights, _ = args

    _oneshot_cf_cache.clear()
    _multiphase_cf_cache.clear()
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    op.forward_topk(x, topk_ids, topk_weights)

    actual = sonic_moe_backward(*args, config)
    expected = _backward_reference(*args)
    torch.cuda.synchronize()

    for actual_gradient, expected_gradient in zip(actual[:3], expected[:3]):
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=3e-2,
            atol=5e-2,
        )
    torch.testing.assert_close(actual[3], expected[3], rtol=2e-3, atol=2e-2)


@pytest.mark.parametrize("activation_name", _ACTIVATIONS)
@pytest.mark.parametrize("dtype,compute_dtype", _DTYPES, ids=("bf16", "fp16"))
def test_sonic_moe_backward_bias_gradients_match_a16_reference(
    activation_name,
    dtype,
    compute_dtype,
):
    hidden_size, intermediate_size, num_experts, topk = 128, 64, 4, 2
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        activation=activation_name,
        compute_dtype=compute_dtype,
    )
    args = _make_case(
        7,
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        seed=307,
        activation=activation_name,
        dtype=dtype,
    )
    b1, b2 = _make_biases(args[1], args[2], seed=311)

    actual = sonic_moe_backward(*args, config, b1=b1, b2=b2)
    expected = _backward_reference(
        *args,
        activation_name=activation_name,
        b1=b1,
        b2=b2,
    )
    torch.cuda.synchronize()

    assert len(actual) == 6
    for actual_gradient, expected_gradient in zip(actual[:3], expected[:3]):
        assert actual_gradient.shape == expected_gradient.shape
        assert actual_gradient.dtype == expected_gradient.dtype
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=3e-2,
            atol=5e-2,
        )
    torch.testing.assert_close(actual[3], expected[3], rtol=5e-4, atol=5e-4)
    for actual_bias_gradient, expected_bias_gradient in zip(actual[4:], expected[4:]):
        assert actual_bias_gradient.shape == expected_bias_gradient.shape
        assert actual_bias_gradient.dtype == dtype
        torch.testing.assert_close(
            actual_bias_gradient.float(),
            expected_bias_gradient.float(),
            rtol=3e-2,
            atol=5e-2,
        )

    # The final expert is intentionally unused and must remain exactly zero.
    assert torch.count_nonzero(actual[1][-1]) == 0
    assert torch.count_nonzero(actual[2][-1]) == 0
    assert torch.count_nonzero(actual[4][-1]) == 0
    assert torch.count_nonzero(actual[5][-1]) == 0


@pytest.mark.parametrize("activation_name", ("swiglu", "relu_sq"))
@pytest.mark.parametrize("dtype,compute_dtype", _DTYPES, ids=("bf16", "fp16"))
def test_sonic_moe_backward_bias_reduction_spans_route_tiles(
    activation_name,
    dtype,
    compute_dtype,
):
    tokens, hidden_size, intermediate_size, num_experts, topk = 65, 128, 64, 3, 2
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        activation=activation_name,
        compute_dtype=compute_dtype,
    )
    args = _make_case(
        tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        seed=331,
        activation=activation_name,
        dtype=dtype,
    )
    b1, b2 = _make_biases(args[1], args[2], seed=337)

    actual = sonic_moe_backward(*args, config, b1=b1, b2=b2)
    expected = _backward_reference(
        *args,
        activation_name=activation_name,
        b1=b1,
        b2=b2,
    )
    torch.cuda.synchronize()

    for actual_gradient, expected_gradient in zip(actual, expected):
        if actual_gradient.dtype == torch.float32:
            rtol, atol = 5e-4, 5e-4
        else:
            rtol, atol = 3e-2, 5e-2
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=rtol,
            atol=atol,
        )
    assert torch.count_nonzero(actual[4][-1]) == 0
    assert torch.count_nonzero(actual[5][-1]) == 0


@pytest.mark.parametrize("activation_name", _ACTIVATIONS)
@pytest.mark.parametrize("dtype,compute_dtype", _DTYPES, ids=("bf16", "fp16"))
def test_sonic_moe_backward_activation_variants_match_a16_reference(
    activation_name,
    dtype,
    compute_dtype,
):
    hidden_size, intermediate_size, num_experts, topk = 128, 64, 4, 2
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        activation=activation_name,
        compute_dtype=compute_dtype,
    )
    args = _make_case(
        7,
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        seed=271,
        activation=activation_name,
        dtype=dtype,
    )

    actual = sonic_moe_backward(*args, config)
    expected = _backward_reference(*args, activation_name=activation_name)
    torch.cuda.synchronize()

    for actual_gradient, expected_gradient in zip(actual[:3], expected[:3]):
        assert actual_gradient.shape == expected_gradient.shape
        assert actual_gradient.dtype == expected_gradient.dtype
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=3e-2,
            atol=5e-2,
        )
    torch.testing.assert_close(actual[3], expected[3], rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("activation_name", _ACTIVATIONS)
@pytest.mark.parametrize("dtype,compute_dtype", _DTYPES, ids=("bf16", "fp16"))
def test_sonic_moe_backward_routes_matches_a16_reference(
    activation_name,
    dtype,
    compute_dtype,
):
    tokens, hidden_size, intermediate_size, num_experts = 7, 128, 64, 5
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        1,
        activation=activation_name,
        compute_dtype=compute_dtype,
        down_tile_m=128,
    )
    fixed_case = _make_case(
        tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        1,
        seed=347,
        activation=activation_name,
        dtype=dtype,
    )
    x, w1, w2, _, _, grad_output = fixed_case
    # Tokens 1, 4, and 5 have no routes. The first two routes deliberately
    # duplicate (token=0, expert=1), and expert 4 remains unused.
    token_indices = torch.tensor([0, 0, 0, 2, 3, 3, 6], dtype=torch.int32, device=x.device)
    expert_indices = torch.tensor([1, 1, 3, 0, 2, 0, 2], dtype=torch.int32, device=x.device)
    route_weights = torch.tensor(
        [0.75, 0.25, -0.1, 0.0, 1.25, 0.4, 0.8],
        dtype=torch.float32,
        device=x.device,
    )
    b1, b2 = _make_biases(w1, w2, seed=349)

    actual = sonic_moe_backward_routes(
        x,
        w1,
        w2,
        token_indices,
        expert_indices,
        route_weights,
        grad_output,
        config,
        b1=b1,
        b2=b2,
    )
    expected = _backward_routes_reference(
        x,
        w1,
        w2,
        token_indices,
        expert_indices,
        route_weights,
        grad_output,
        activation_name=activation_name,
        b1=b1,
        b2=b2,
    )
    torch.cuda.synchronize()

    assert len(actual) == 6
    for actual_gradient, expected_gradient in zip(actual, expected):
        if actual_gradient.dtype == torch.float32:
            rtol, atol = 5e-4, 5e-4
        else:
            rtol, atol = 3e-2, 5e-2
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=rtol,
            atol=atol,
        )
    assert torch.count_nonzero(actual[0][torch.tensor([1, 4, 5], device=x.device)]) == 0
    assert torch.count_nonzero(actual[1][-1]) == 0
    assert torch.count_nonzero(actual[2][-1]) == 0
    assert torch.count_nonzero(actual[4][-1]) == 0
    assert torch.count_nonzero(actual[5][-1]) == 0


@pytest.mark.parametrize("dtype,compute_dtype", _DTYPES, ids=("bf16-grouped", "fp16-generic"))
def test_sonic_moe_backward_routes_native_interleaved_w1_matches_reference(
    dtype,
    compute_dtype,
):
    """Ragged duplicate routes preserve native interleaved W1/B1 gradients."""

    tokens, hidden_size, intermediate_size, num_experts = 64, 256, 128, 5
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        1,
        compute_dtype=compute_dtype,
        down_tile_m=128,
    )
    x, w1, w2, _, _, grad_output = _make_case(
        tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        1,
        seed=457,
        dtype=dtype,
    )
    b1, b2 = _make_biases(w1, w2, seed=461)
    w1 = _interleave_glu_rows(w1)
    b1 = _interleave_glu_rows(b1)
    token_indices = torch.tensor(
        [0, 0, 3, 7, 7, 31, 63],
        dtype=torch.int32,
        device=x.device,
    )
    expert_indices = torch.tensor(
        [1, 1, 3, 0, 2, 0, 2],
        dtype=torch.int32,
        device=x.device,
    )
    route_weights = torch.tensor(
        [0.75, 0.25, -0.1, 0.0, 1.25, 0.4, 0.8],
        dtype=torch.float32,
        device=x.device,
    )

    actual = sonic_moe_backward_routes(
        x,
        w1,
        w2,
        token_indices,
        expert_indices,
        route_weights,
        grad_output,
        config,
        b1=b1,
        b2=b2,
        interleaved_w1=True,
    )
    expected = _backward_routes_reference(
        x,
        w1,
        w2,
        token_indices,
        expert_indices,
        route_weights,
        grad_output,
        b1=b1,
        b2=b2,
        interleaved_w1=True,
    )
    torch.cuda.synchronize()

    for actual_gradient, expected_gradient in zip(actual, expected):
        if actual_gradient.dtype == torch.float32:
            rtol, atol = 2e-3, 4e-3
        else:
            rtol, atol = 3e-2, 5e-2
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=rtol,
            atol=atol,
        )
    assert torch.count_nonzero(actual[1][4]) == 0
    assert torch.count_nonzero(actual[4][4]) == 0


def test_sonic_moe_backward_routes_grouped_dx_reuses_compact_queue(monkeypatch):
    """Duplicate ragged routes use one BM16 queue for W1 recompute and dX."""

    tokens, hidden_size, intermediate_size, num_experts = 64, 256, 128, 4
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        1,
        compute_dtype="bf16",
        down_tile_m=128,
    )
    x, w1, w2, _, _, grad_output = _make_case(
        tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        1,
        seed=419,
        dtype=torch.bfloat16,
    )
    # Every token has the same (token, expert) edge twice.  This exercises the
    # ragged atomic scatter and makes one expert span two sorter blocks.
    token_indices = torch.arange(tokens, dtype=torch.int32, device=x.device).repeat_interleave(2)
    expert_indices = torch.zeros(tokens * 2, dtype=torch.int32, device=x.device)
    route_weights = torch.linspace(-0.5, 1.0, tokens * 2, dtype=torch.float32, device=x.device)
    b1, b2 = _make_biases(w1, w2, seed=421)

    original_builder = sonic_backward_module.build_compact_m_tile_descriptors
    builder_calls = 0
    active_queue_shared = False

    def _tracked_builder(*builder_args, **builder_kwargs):
        nonlocal active_queue_shared, builder_calls
        builder_calls += 1
        active_queue_shared = builder_kwargs.get("active_expert_storage") is not None
        return original_builder(*builder_args, **builder_kwargs)

    monkeypatch.setattr(
        sonic_backward_module,
        "build_compact_m_tile_descriptors",
        _tracked_builder,
    )
    actual = sonic_moe_backward_routes(
        x,
        w1,
        w2,
        token_indices,
        expert_indices,
        route_weights,
        grad_output,
        config,
        b1=b1,
        b2=b2,
    )
    expected = _backward_routes_reference(
        x,
        w1,
        w2,
        token_indices,
        expert_indices,
        route_weights,
        grad_output,
        b1=b1,
        b2=b2,
    )
    torch.cuda.synchronize()

    assert builder_calls == 1
    assert active_queue_shared
    for actual_gradient, expected_gradient in zip(actual, expected):
        if actual_gradient.dtype == torch.float32:
            rtol, atol = 2e-3, 4e-3
        else:
            rtol, atol = 3e-2, 5e-2
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=rtol,
            atol=atol,
        )
    assert torch.count_nonzero(actual[1][1:]) == 0
    assert torch.count_nonzero(actual[2][1:]) == 0
    assert torch.count_nonzero(actual[4][1:]) == 0
    assert torch.count_nonzero(actual[5][1:]) == 0


@pytest.mark.parametrize("with_bias", (False, True), ids=("no-bias", "bias"))
@pytest.mark.parametrize("dtype,compute_dtype", _DTYPES, ids=("bf16", "fp16"))
def test_sonic_moe_backward_routes_supports_globally_empty_routes(
    with_bias,
    dtype,
    compute_dtype,
):
    config = _config(128, 64, 4, 1, compute_dtype=compute_dtype)
    x, w1, w2, _, _, grad_output = _make_case(7, 128, 64, 4, 1, seed=353, dtype=dtype)
    empty_i32 = torch.empty(0, dtype=torch.int32, device=x.device)
    empty_f32 = torch.empty(0, dtype=torch.float32, device=x.device)
    b1, b2 = _make_biases(w1, w2, seed=359) if with_bias else (None, None)

    actual = sonic_moe_backward_routes(
        x,
        w1,
        w2,
        empty_i32,
        empty_i32,
        empty_f32,
        grad_output,
        config,
        b1=b1,
        b2=b2,
    )
    torch.cuda.synchronize()

    assert len(actual) == (6 if with_bias else 4)
    assert all(torch.count_nonzero(gradient) == 0 for gradient in actual)
    assert actual[0].dtype == actual[1].dtype == actual[2].dtype == dtype
    assert actual[3].dtype == torch.float32


@pytest.mark.parametrize("dtype,compute_dtype", _DTYPES, ids=("bf16", "fp16"))
def test_sonic_moe_backward_routes_spans_multiple_expert_tiles(dtype, compute_dtype):
    tokens, hidden_size, intermediate_size, num_experts = 70, 128, 64, 4
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        1,
        compute_dtype=compute_dtype,
    )
    x, w1, w2, _, _, grad_output = _make_case(
        tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        1,
        seed=367,
        dtype=dtype,
    )
    token_indices = torch.arange(65, dtype=torch.int32, device=x.device).repeat_interleave(2)
    expert_indices = torch.tensor([0, 1], dtype=torch.int32, device=x.device).repeat(65)
    route_weights = torch.linspace(-0.4, 1.2, 130, dtype=torch.float32, device=x.device)
    b1, b2 = _make_biases(w1, w2, seed=373)

    actual = sonic_moe_backward_routes(
        x,
        w1,
        w2,
        token_indices,
        expert_indices,
        route_weights,
        grad_output,
        config,
        b1=b1,
        b2=b2,
    )
    expected = _backward_routes_reference(
        x,
        w1,
        w2,
        token_indices,
        expert_indices,
        route_weights,
        grad_output,
        b1=b1,
        b2=b2,
    )
    torch.cuda.synchronize()

    for actual_gradient, expected_gradient in zip(actual, expected):
        if actual_gradient.dtype == torch.float32:
            rtol, atol = 2e-3, 4e-3
        else:
            rtol, atol = 3e-2, 5e-2
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=rtol,
            atol=atol,
        )
    assert torch.count_nonzero(actual[0][65:]) == 0
    assert torch.count_nonzero(actual[1][2:]) == 0
    assert torch.count_nonzero(actual[2][2:]) == 0
    assert torch.count_nonzero(actual[4][2:]) == 0
    assert torch.count_nonzero(actual[5][2:]) == 0


@pytest.mark.parametrize(
    (
        "tokens",
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "topk",
        "hot_experts",
        "interleaved_w1",
        "has_bias",
    ),
    (
        (1, 128, 64, 4, 2, None, True, True),
        (64, 256, 128, 16, 3, None, True, False),
        (128, 256, 128, 64, 4, None, False, False),
        (128, 256, 128, 64, 4, 16, False, False),
        (4096, 128, 64, 4, 2, None, False, False),
    ),
    ids=(
        "t1-interleaved-bias",
        "t64-interleaved-fused",
        "t128-balanced",
        "t128-hot16",
        "t4096-legacy",
    ),
)
def test_sonic_moe_backward_reuses_route_order_forward_preactivation(
    tokens,
    hidden_size,
    intermediate_size,
    num_experts,
    topk,
    hot_experts,
    interleaved_w1,
    has_bias,
):
    """Forward state matches standalone numerics across tuned routing regimes."""

    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        compute_dtype="bf16",
        down_tile_m=128,
    )
    args = list(
        _make_case(
            tokens,
            hidden_size,
            intermediate_size,
            num_experts,
            topk,
            seed=607 + tokens + (hot_experts or 0),
            dtype=torch.bfloat16,
        )
    )
    if hot_experts is not None:
        ids_host = [[(token + slot) % hot_experts for slot in range(topk)] for token in range(tokens)]
        args[3] = torch.tensor(ids_host, dtype=torch.int32, device=args[0].device)
    if interleaved_w1:
        args[1] = _interleave_glu_rows(args[1])
    args = tuple(args)
    if has_bias:
        b1, b2 = _make_biases(args[1], args[2], seed=659)
        if interleaved_w1:
            b1 = _interleave_glu_rows(b1)
    else:
        b1 = b2 = None

    state = _make_forward_state(
        args[0],
        args[1],
        args[3],
        config,
        b1=b1,
        interleaved_w1=interleaved_w1,
    )
    state_snapshot = state.preactivation.clone()
    actual = sonic_moe_backward(
        *args,
        config,
        b1=b1,
        b2=b2,
        interleaved_w1=interleaved_w1,
        forward_state=state,
    )
    expected = _backward_reference(
        *args,
        b1=b1,
        b2=b2,
        interleaved_w1=interleaved_w1,
        reassociate_da_dscore=tokens in (64, 128) and not has_bias,
    )
    torch.cuda.synchronize()

    assert torch.equal(state.preactivation, state_snapshot)
    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=3e-2,
            atol=5e-2,
        )


@pytest.mark.parametrize(
    ("interleaved_w1", "has_bias"),
    ((False, False), (True, True)),
    ids=("separate-no-bias", "interleaved-bias"),
)
def test_sonic_moe_training_forward_state_drives_backward_reference(
    interleaved_w1,
    has_bias,
):
    """The official training-forward state is directly consumable by backward."""

    tokens, hidden_size, intermediate_size, num_experts, topk = 7, 128, 64, 4, 2
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        compute_dtype="bf16",
    )
    x, separate_w1, w2, topk_ids, topk_weights, grad_output = _make_case(
        tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        seed=673 + int(interleaved_w1),
        dtype=torch.bfloat16,
    )
    if has_bias:
        separate_b1, b2 = _make_biases(separate_w1, w2, seed=675)
    else:
        separate_b1 = b2 = None

    op = SonicMoE(
        config,
        prepare_sonic_bf16_weights(
            separate_w1,
            w2,
            config,
            b1=separate_b1,
            b2=b2,
        ),
    )
    forward_output, state = op.forward_topk_training(
        x,
        topk_ids,
        topk_weights,
        interleaved_w1=interleaved_w1,
    )
    state_snapshot = state.preactivation.clone()

    backward_w1 = _interleave_glu_rows(separate_w1) if interleaved_w1 else separate_w1
    backward_b1 = _interleave_glu_rows(separate_b1) if interleaved_w1 and separate_b1 is not None else separate_b1
    actual = sonic_moe_backward(
        x,
        backward_w1,
        w2,
        topk_ids,
        topk_weights,
        grad_output,
        config,
        b1=backward_b1,
        b2=b2,
        interleaved_w1=interleaved_w1,
        forward_state=state,
    )
    expected = _backward_reference(
        x,
        backward_w1,
        w2,
        topk_ids,
        topk_weights,
        grad_output,
        b1=backward_b1,
        b2=b2,
        interleaved_w1=interleaved_w1,
    )
    torch.cuda.synchronize()

    assert forward_output.shape == x.shape
    assert isinstance(state, SonicMoEForwardState)
    assert state.interleaved_w1 is interleaved_w1
    assert state.has_bias is has_bias
    assert torch.equal(state.preactivation, state_snapshot)
    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=3e-2,
            atol=5e-2,
        )


def test_sonic_moe_backward_forward_state_is_reusable_and_none_is_fallback():
    config = _config(128, 64, 4, 2, compute_dtype="bf16")
    args = _make_case(7, 128, 64, 4, 2, seed=677, dtype=torch.bfloat16)

    implicit_fallback = tuple(t.clone() for t in sonic_moe_backward(*args, config))
    explicit_fallback = sonic_moe_backward(*args, config, forward_state=None)
    state = _make_forward_state(args[0], args[1], args[3], config)
    state_snapshot = state.preactivation.clone()
    first = tuple(t.clone() for t in sonic_moe_backward(*args, config, forward_state=state))
    second = sonic_moe_backward(*args, config, forward_state=state)
    torch.cuda.synchronize()

    for implicit, explicit in zip(implicit_fallback, explicit_fallback):
        assert torch.equal(implicit, explicit)
    for first_gradient, second_gradient in zip(first, second):
        assert torch.equal(first_gradient, second_gradient)
    assert torch.equal(state.preactivation, state_snapshot)


def test_sonic_moe_backward_forward_state_skips_grouped_w1_but_keeps_compact_queue(
    monkeypatch,
):
    tokens, hidden_size, intermediate_size, num_experts, topk = 128, 256, 128, 64, 4
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        compute_dtype="bf16",
        down_tile_m=128,
    )
    args = _make_case(
        tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        seed=683,
        dtype=torch.bfloat16,
    )
    state = _make_forward_state(args[0], args[1], args[3], config)
    original_builder = sonic_backward_module.build_compact_m_tile_descriptors
    builder_blocks = []

    def _unexpected_grouped_w1(*_args, **_kwargs):
        raise AssertionError("forward state must skip grouped W1 recomputation")

    def _tracked_builder(*builder_args, **builder_kwargs):
        builder_blocks.append(builder_kwargs["block_m"])
        return original_builder(*builder_args, **builder_kwargs)

    monkeypatch.setattr(
        sonic_backward_module,
        "_compile_grouped_w1_recompute",
        _unexpected_grouped_w1,
    )
    monkeypatch.setattr(
        sonic_backward_module,
        "build_compact_m_tile_descriptors",
        _tracked_builder,
    )
    sonic_moe_backward(*args, config, forward_state=state)
    torch.cuda.synchronize()

    assert builder_blocks == [16]


def test_sonic_moe_backward_compact_state_fuses_live_row_prepare(monkeypatch):
    """Compact retained-state backward must bypass padded row materialization."""

    tokens, hidden_size, intermediate_size, num_experts, topk = 128, 256, 128, 64, 4
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        compute_dtype="bf16",
        down_tile_m=128,
    )
    args = _make_case(
        tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        seed=687,
        dtype=torch.bfloat16,
    )
    state = _make_forward_state(args[0], args[1], args[3], config)

    def _unexpected_legacy_kernel(*_args, **_kwargs):
        raise AssertionError("compact retained-state path must use exact-row kernels")

    monkeypatch.setattr(sonic_backward_module, "_compile_gather", _unexpected_legacy_kernel)
    monkeypatch.setattr(
        sonic_backward_module,
        "_compile_activation_prepare_from_forward_state",
        _unexpected_legacy_kernel,
    )
    monkeypatch.setattr(
        sonic_backward_module,
        "_compile_activation_derivative",
        _unexpected_legacy_kernel,
    )
    monkeypatch.setattr(
        sonic_backward_module,
        "_compile_activation_derivative_from_forward_state",
        _unexpected_legacy_kernel,
    )
    monkeypatch.setattr(
        sonic_backward_module,
        "_compile_grouped_w2_recompute",
        _unexpected_legacy_kernel,
    )
    monkeypatch.setattr(
        sonic_backward_module,
        "_compile_score_backward",
        _unexpected_legacy_kernel,
    )
    actual = sonic_moe_backward(*args, config, forward_state=state)
    expected = _backward_reference(*args, reassociate_da_dscore=True)
    torch.cuda.synchronize()

    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=3e-2,
            atol=5e-2,
        )


def test_sonic_moe_backward_decode_state_keeps_low_latency_row_kernels(monkeypatch):
    """T1 avoids the compact fused kernel because its queue is not available."""

    tokens, hidden_size, intermediate_size, num_experts, topk = 1, 256, 128, 4, 2
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        compute_dtype="bf16",
        down_tile_m=128,
    )
    args = _make_case(
        tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        seed=689,
        dtype=torch.bfloat16,
    )
    state = _make_forward_state(args[0], args[1], args[3], config)

    def _unexpected_fused_kernel(*_args, **_kwargs):
        raise AssertionError("decode must retain its measured low-latency row path")

    monkeypatch.setattr(
        sonic_backward_module,
        "_compile_fused_forward_state_prepare",
        _unexpected_fused_kernel,
    )
    actual = sonic_moe_backward(*args, config, forward_state=state)
    expected = _backward_reference(*args)
    torch.cuda.synchronize()

    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=3e-2,
            atol=5e-2,
        )


def test_sonic_moe_backward_forward_state_skips_generic_w1_and_keeps_large_dx_queue(
    monkeypatch,
):
    tokens, hidden_size, intermediate_size, num_experts, topk = 4096, 256, 128, 4, 2
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        compute_dtype="bf16",
        down_tile_m=128,
    )
    args = _make_case(
        tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        seed=691,
        dtype=torch.bfloat16,
    )
    state = _make_forward_state(args[0], args[1], args[3], config)
    original_builder = sonic_backward_module.build_compact_m_tile_descriptors
    original_fused_prepare = sonic_backward_module._compile_fused_forward_state_prepare
    original_fused_derivative = sonic_backward_module._compile_fused_activation_derivative_dscore_scale_dy
    original_gemm = sonic_backward_module.gemm_a16w16
    builder_blocks = []
    prepare_blocks = []
    derivative_blocks = []

    def _schedule_block_m(call_args, call_kwargs):
        return call_kwargs.get("schedule_block_m", call_args[5] if len(call_args) > 5 else 16)

    def _tracked_builder(*builder_args, **builder_kwargs):
        builder_blocks.append(builder_kwargs["block_m"])
        return original_builder(*builder_args, **builder_kwargs)

    def _tracked_fused_prepare(*compile_args, **compile_kwargs):
        prepare_blocks.append(_schedule_block_m(compile_args, compile_kwargs))
        return original_fused_prepare(*compile_args, **compile_kwargs)

    def _tracked_fused_derivative(*compile_args, **compile_kwargs):
        derivative_blocks.append(_schedule_block_m(compile_args, compile_kwargs))
        return original_fused_derivative(*compile_args, **compile_kwargs)

    def _guarded_gemm(a, b, *gemm_args, **gemm_kwargs):
        if gemm_kwargs.get("layout") == "nt":
            raise AssertionError("fused state path must skip generic W1/W2 projection recomputation")
        return original_gemm(a, b, *gemm_args, **gemm_kwargs)

    def _unexpected_legacy_kernel(*_args, **_kwargs):
        raise AssertionError("large retained-state path must use exact-row fused kernels")

    monkeypatch.setattr(
        sonic_backward_module,
        "_use_large_grouped_dx_descriptor_queue",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        sonic_backward_module,
        "build_compact_m_tile_descriptors",
        _tracked_builder,
    )
    monkeypatch.setattr(
        sonic_backward_module,
        "_compile_fused_forward_state_prepare",
        _tracked_fused_prepare,
    )
    monkeypatch.setattr(
        sonic_backward_module,
        "_compile_fused_activation_derivative_dscore_scale_dy",
        _tracked_fused_derivative,
    )
    monkeypatch.setattr(sonic_backward_module, "_compile_gather", _unexpected_legacy_kernel)
    monkeypatch.setattr(
        sonic_backward_module,
        "_compile_activation_prepare_from_forward_state",
        _unexpected_legacy_kernel,
    )
    monkeypatch.setattr(
        sonic_backward_module,
        "_compile_activation_derivative",
        _unexpected_legacy_kernel,
    )
    monkeypatch.setattr(
        sonic_backward_module,
        "_compile_activation_derivative_from_forward_state",
        _unexpected_legacy_kernel,
    )
    monkeypatch.setattr(
        sonic_backward_module,
        "_compile_grouped_w2_recompute",
        _unexpected_legacy_kernel,
    )
    monkeypatch.setattr(
        sonic_backward_module,
        "_compile_score_backward",
        _unexpected_legacy_kernel,
    )
    monkeypatch.setattr(sonic_backward_module, "gemm_a16w16", _guarded_gemm)
    actual = sonic_moe_backward(*args, config, forward_state=state)
    expected = _backward_reference(*args, reassociate_da_dscore=True)
    torch.cuda.synchronize()

    assert builder_blocks == [64]
    assert prepare_blocks == [64]
    assert derivative_blocks == [64]
    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=3e-2,
            atol=5e-2,
        )


@pytest.mark.large_shape
def test_sonic_moe_backward_t4096_fused_da_dscore_matches_q_reference():
    """The production BM64 state path satisfies its reassociated A16 contract."""

    tokens, hidden_size, intermediate_size, num_experts, topk = 4096, 4096, 2048, 64, 8
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        compute_dtype="bf16",
        down_tile_m=128,
    )
    args = _make_case(
        tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        seed=20260907,
        dtype=torch.bfloat16,
    )
    state = _make_forward_state(args[0], args[1], args[3], config)
    actual = sonic_moe_backward(*args, config, forward_state=state)
    expected = _backward_reference(*args, reassociate_da_dscore=True)
    torch.cuda.synchronize()

    # A billion-element dW1 has a handful of near-zero elements outside the
    # small-shape elementwise tolerance even though its global error is tiny.
    # Keep this opt-in production gate sensitive to meaningful drift without
    # weakening the default suite's elementwise assertions.
    relative_l2_limits = (7.5e-4, 3.0e-4, 1.0e-4, 1.0e-4)
    max_abs_limits = (0.0625, 0.5, 0.5, 0.25)
    chunk_elements = 32 * 1024 * 1024
    for actual_gradient, expected_gradient, relative_l2_limit, max_abs_limit in zip(
        actual,
        expected,
        relative_l2_limits,
        max_abs_limits,
    ):
        actual_flat = actual_gradient.reshape(-1)
        expected_flat = expected_gradient.reshape(-1)
        actual_sq = 0.0
        reference_sq = 0.0
        difference_sq = 0.0
        max_abs = 0.0
        for start in range(0, actual_flat.numel(), chunk_elements):
            end = min(start + chunk_elements, actual_flat.numel())
            actual_chunk = actual_flat[start:end].float()
            expected_chunk = expected_flat[start:end].float()
            difference = actual_chunk - expected_chunk
            assert torch.isfinite(actual_chunk).all()
            actual_sq += float(torch.sum(actual_chunk * actual_chunk))
            reference_sq += float(torch.sum(expected_chunk * expected_chunk))
            difference_sq += float(torch.sum(difference * difference))
            max_abs = max(max_abs, float(difference.abs().max()))

        actual_norm = math.sqrt(actual_sq)
        reference_norm = math.sqrt(reference_sq)
        relative_l2 = math.sqrt(difference_sq / reference_sq)
        assert abs(actual_norm / reference_norm - 1.0) <= 1.0e-5
        assert relative_l2 <= relative_l2_limit
        assert max_abs <= max_abs_limit


@pytest.mark.parametrize(
    ("tokens", "hidden_size", "intermediate_size", "num_experts", "topk", "interleaved_w1", "fused"),
    (
        (7, 128, 64, 4, 2, False, False),
        (64, 256, 128, 16, 3, True, True),
    ),
    ids=("legacy-row-path", "fused-interleaved"),
)
def test_sonic_moe_backward_forward_state_waits_cross_stream_and_tracks_lifetime(
    tokens,
    hidden_size,
    intermediate_size,
    num_experts,
    topk,
    interleaved_w1,
    fused,
):
    config = _config(hidden_size, intermediate_size, num_experts, topk, compute_dtype="bf16")
    args = list(
        _make_case(
            tokens,
            hidden_size,
            intermediate_size,
            num_experts,
            topk,
            seed=701 + tokens,
            dtype=torch.bfloat16,
        )
    )
    if interleaved_w1:
        args[1] = _interleave_glu_rows(args[1])
    args = tuple(args)
    source_state = _make_forward_state(
        args[0],
        args[1],
        args[3],
        config,
        interleaved_w1=interleaved_w1,
    )
    source = source_state.preactivation.clone()
    # Compile every backward launcher before constructing the delayed producer;
    # otherwise JIT time can outlast the device delay and turn the wait check
    # into another false positive.
    sonic_moe_backward(
        *args,
        config,
        interleaved_w1=interleaved_w1,
        forward_state=source_state,
    )
    torch.cuda.current_stream(args[0].device).synchronize()

    producer = torch.cuda.Stream(device=args[0].device)
    consumer = torch.cuda.Stream(device=args[0].device)
    releaser = torch.cuda.Stream(device=args[0].device)
    release_event = torch.cuda.Event()
    ready_event = torch.cuda.Event()
    route_preactivation = torch.empty_like(source)
    with torch.cuda.stream(releaser):
        # Record a real, incomplete dependency.  Waiting on an event before it
        # has ever been recorded is a no-op in CUDA/HIP and would make this
        # test pass without exercising backward's cross-stream wait.
        torch.cuda._sleep(300_000_000)
        release_event.record(releaser)
    with torch.cuda.stream(producer):
        producer.wait_event(release_event)
        route_preactivation.copy_(source)
        ready_event.record(producer)
    assert not ready_event.query()
    state = SimpleNamespace(
        **{
            **vars(source_state),
            "preactivation": route_preactivation,
            "producer_stream": int(producer.cuda_stream),
            "ready_event": ready_event,
        }
    )

    with torch.cuda.stream(consumer):
        actual = sonic_moe_backward(
            *args,
            config,
            interleaved_w1=interleaved_w1,
            forward_state=state,
        )
    del state, route_preactivation
    with torch.cuda.stream(releaser):
        allocator_pressure = torch.full_like(source, -17.0)
    wait_start = time.perf_counter()
    consumer.synchronize()
    wait_seconds = time.perf_counter() - wait_start
    expected = _backward_reference(
        *args,
        interleaved_w1=interleaved_w1,
        reassociate_da_dscore=fused,
    )
    torch.cuda.synchronize()

    assert wait_seconds >= 0.02
    assert torch.count_nonzero(allocator_pressure) == allocator_pressure.numel()
    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=3e-2,
            atol=5e-2,
        )


def test_sonic_moe_backward_rejects_bad_forward_state():
    config = _config(128, 64, 4, 2, compute_dtype="bf16")
    args = _make_case(7, 128, 64, 4, 2, seed=709, dtype=torch.bfloat16)
    state = _make_forward_state(args[0], args[1], args[3], config)
    fields = vars(state).copy()

    missing = fields.copy()
    missing.pop("top_k")
    with pytest.raises(ValueError, match="missing required field.*top_k"):
        sonic_moe_backward(*args, config, forward_state=SimpleNamespace(**missing))
    with pytest.raises(TypeError, match="tokens must be int"):
        sonic_moe_backward(
            *args,
            config,
            forward_state=SimpleNamespace(**{**fields, "tokens": True}),
        )
    with pytest.raises(ValueError, match="tokens must equal"):
        sonic_moe_backward(
            *args,
            config,
            forward_state=SimpleNamespace(**{**fields, "tokens": 8}),
        )
    with pytest.raises(TypeError, match="preactivation must be torch.bfloat16"):
        sonic_moe_backward(
            *args,
            config,
            forward_state=SimpleNamespace(**{**fields, "preactivation": state.preactivation.half()}),
        )
    with pytest.raises(ValueError, match="interleaved_w1 must match"):
        sonic_moe_backward(
            *args,
            config,
            forward_state=SimpleNamespace(**{**fields, "interleaved_w1": True}),
        )
    with pytest.raises(TypeError, match="has_bias must be bool"):
        sonic_moe_backward(
            *args,
            config,
            forward_state=SimpleNamespace(**{**fields, "has_bias": 0}),
        )
    with pytest.raises(ValueError, match="has_bias must match"):
        sonic_moe_backward(
            *args,
            config,
            b1=torch.zeros_like(args[1][:, :, 0]),
            b2=torch.zeros_like(args[2][:, :, 0]),
            forward_state=state,
        )
    with pytest.raises(TypeError, match="producer_stream must be int"):
        sonic_moe_backward(
            *args,
            config,
            forward_state=SimpleNamespace(**{**fields, "producer_stream": False}),
        )
    with pytest.raises(ValueError, match="ready_event must already be recorded"):
        sonic_moe_backward(
            *args,
            config,
            forward_state=SimpleNamespace(**{**fields, "ready_event": torch.cuda.Event()}),
        )
    with pytest.raises(ValueError, match="only dense BF16 SwiGLU"):
        sonic_moe_backward(
            *args,
            _config(128, 64, 4, 2, compute_dtype="bf16", activation="geglu"),
            forward_state=SimpleNamespace(**{**fields, "activation": "geglu"}),
        )

    oversized_intermediate = 40_000_000
    oversized_config = _config(
        128,
        oversized_intermediate,
        4,
        2,
        compute_dtype="bf16",
    )
    oversized_preactivation = torch.empty(1, dtype=torch.bfloat16, device=args[0].device).as_strided(
        (7, 2, 2 * oversized_intermediate),
        (0, 0, 0),
    )
    oversized_fields = {
        **fields,
        "preactivation": oversized_preactivation,
        "intermediate_size": oversized_intermediate,
    }
    with pytest.raises(ValueError, match="byte span exceeds the signed 32-bit"):
        sonic_backward_module._validate_forward_state(
            SimpleNamespace(**oversized_fields),
            args[0],
            oversized_config,
            False,
            False,
        )


@pytest.mark.parametrize("dtype,compute_dtype", _DTYPES, ids=("bf16", "fp16"))
def test_sonic_moe_backward_repeated_calls_do_not_alias_workspace(dtype, compute_dtype):
    config = _config(128, 64, 4, 2, compute_dtype=compute_dtype)
    first = _make_case(7, 128, 64, 4, 2, seed=251, dtype=dtype)
    second = _make_case(7, 128, 64, 4, 2, seed=257, dtype=dtype)

    first_actual = tuple(t.clone() for t in sonic_moe_backward(*first, config))
    sonic_moe_backward(*second, config)
    first_again = sonic_moe_backward(*first, config)
    torch.cuda.synchronize()

    for saved, repeated in zip(first_actual, first_again):
        assert torch.equal(saved, repeated)


def test_sonic_moe_backward_rejects_unsupported_contracts():
    args = _make_case(7, 128, 64, 4, 2, seed=263)
    b1, b2 = _make_biases(args[1], args[2], seed=317)

    with pytest.raises(TypeError, match="must be torch.float16"):
        sonic_moe_backward(*args, _config(128, 64, 4, 2, compute_dtype="fp16"))
    with pytest.raises(ValueError, match="unsupported compute_dtype"):
        sonic_moe_backward(*args, _config(128, 64, 4, 2, compute_dtype="fp32"))
    with pytest.raises(ValueError, match="w1 must have shape"):
        sonic_moe_backward(*args, _config(128, 64, 4, 2, activation="relu"))
    with pytest.raises(TypeError, match="interleaved_w1 must be bool"):
        sonic_moe_backward(*args, _config(128, 64, 4, 2), interleaved_w1=1)
    with pytest.raises(ValueError, match="valid only for GLU activations"):
        sonic_moe_backward(
            *args,
            _config(128, 64, 4, 2, activation="relu"),
            interleaved_w1=True,
        )
    with pytest.raises(ValueError, match="both be None or both be tensors"):
        sonic_moe_backward(*args, _config(128, 64, 4, 2), b1=b1)
    with pytest.raises(ValueError, match="b1 must have shape"):
        sonic_moe_backward(
            *args,
            _config(128, 64, 4, 2),
            b1=b1[:, :-1],
            b2=b2,
        )
    with pytest.raises(TypeError, match="b2 must be torch.bfloat16"):
        sonic_moe_backward(
            *args,
            _config(128, 64, 4, 2),
            b1=b1,
            b2=b2.float(),
        )

    noncontiguous_dout = args[-1].transpose(0, 1).contiguous().transpose(0, 1)
    assert not noncontiguous_dout.is_contiguous()
    with pytest.raises(ValueError, match="grad_output must be contiguous"):
        sonic_moe_backward(*args[:-1], noncontiguous_dout, _config(128, 64, 4, 2))

    token_indices = torch.tensor([0, 2, 4], dtype=torch.int32, device=args[0].device)
    expert_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device=args[0].device)
    route_weights = torch.ones(3, dtype=torch.float32, device=args[0].device)
    route_config = _config(128, 64, 4, 1)
    with pytest.raises(ValueError, match="one-dimensional"):
        sonic_moe_backward_routes(
            args[0],
            args[1],
            args[2],
            token_indices[:, None],
            expert_indices,
            route_weights,
            args[-1],
            route_config,
        )
    with pytest.raises(TypeError, match="token_indices and expert_indices must be int32"):
        sonic_moe_backward_routes(
            args[0],
            args[1],
            args[2],
            token_indices.long(),
            expert_indices,
            route_weights,
            args[-1],
            route_config,
        )
    with pytest.raises(TypeError, match="route_weights must be float32"):
        sonic_moe_backward_routes(
            args[0],
            args[1],
            args[2],
            token_indices,
            expert_indices,
            route_weights.half(),
            args[-1],
            route_config,
        )
    with pytest.raises(TypeError, match="interleaved_w1 must be bool"):
        sonic_moe_backward_routes(
            args[0],
            args[1],
            args[2],
            token_indices,
            expert_indices,
            route_weights,
            args[-1],
            route_config,
            interleaved_w1="yes",
        )
    with pytest.raises(ValueError, match="valid only for GLU activations"):
        sonic_moe_backward_routes(
            args[0],
            args[1],
            args[2],
            token_indices,
            expert_indices,
            route_weights,
            args[-1],
            _config(128, 64, 4, 1, activation="relu"),
            interleaved_w1=True,
        )
