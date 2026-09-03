# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Correctness and API-contract tests for gfx950 SonicMoE A16W16/A16W4."""

import math
import threading
import weakref
from dataclasses import replace

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.moe.sonic import (
    SonicMoE,
    SonicMoEConfig,
    SonicMoEWeights,
    SonicMoEWorkspace,
    _get_stage1_launcher,
    _get_stage2_launcher,
    _quantize_mxfp4_weight,
    prepare_sonic_bf16_weights,
    prepare_sonic_mxfp4_weights,
    sonic_moe_mxfp4_reference,
    sonic_moe_reference,
)
from kernels.moe.sonic_autotune import SonicMoEAutotuner

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

TOKENS = 7
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 128
NUM_EXPERTS = 4
TOP_K = 2


def _config(**overrides):
    values = {
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "tile_m": 16,
        "tile_n": 128,
        "tile_k": 128,
    }
    values.update(overrides)
    return SonicMoEConfig(**values)


def _gfx950_device():
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU is required")
    arch = str(get_rocm_arch())
    if "gfx950" not in arch:
        pytest.skip(f"SonicMoE BF16 test requires gfx950, found {arch}")
    return torch.device("cuda")


def _make_case(tokens=TOKENS, seed=17, activation="swiglu"):
    device = _gfx950_device()
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        (tokens, HIDDEN_SIZE), device=device, dtype=torch.float32, generator=generator
    ).to(torch.bfloat16)
    stage1_size = INTERMEDIATE_SIZE * (
        2 if activation in ("swiglu", "geglu", "reglu") else 1
    )
    w1 = (
        torch.randn(
            (NUM_EXPERTS, stage1_size, HIDDEN_SIZE),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        / math.sqrt(HIDDEN_SIZE)
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(
            (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        / math.sqrt(INTERMEDIATE_SIZE)
    ).to(torch.bfloat16)
    router_logits = torch.randn(
        (tokens, NUM_EXPERTS), device=device, dtype=torch.float32, generator=generator
    ).to(torch.bfloat16)
    return x, w1, w2, router_logits


def _topk_from_logits(router_logits, config):
    scores = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(scores, config.top_k, dim=-1)
    if config.renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_ids.to(torch.int32), topk_weights.contiguous()


def _assert_close(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert actual.device == expected.device
    torch.testing.assert_close(actual.float(), expected.float(), rtol=3e-2, atol=5e-2)


def test_sonic_moe_bf16_forward_matches_reference():
    config = _config()
    x, w1, w2, router_logits = _make_case()
    prepared = prepare_sonic_bf16_weights(w1, w2, config)
    op = SonicMoE(config, prepared)

    expected = sonic_moe_reference(x, w1, w2, router_logits, config)
    actual = op(x, router_logits)
    torch.cuda.synchronize()
    _assert_close(actual, expected)

    topk_ids, topk_weights = _topk_from_logits(router_logits, config)
    actual_topk = op.forward_topk(x, topk_ids, topk_weights)
    torch.cuda.synchronize()
    _assert_close(actual_topk, expected)

    out = torch.empty(expected.shape, device=expected.device, dtype=torch.bfloat16)
    returned = op(x, router_logits, out=out)
    torch.cuda.synchronize()
    assert returned is out
    _assert_close(out, expected)


@pytest.mark.parametrize(
    "activation",
    ("swiglu", "geglu", "reglu", "gelu_tanh_approx", "relu", "silu", "relu_sq"),
)
def test_sonic_moe_bf16_activation_variants_fixed_and_flat_routes(activation):
    config = _config(activation=activation)
    x, w1, w2, router_logits = _make_case(seed=83, activation=activation)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    expected = sonic_moe_reference(x, w1, w2, router_logits, config)
    topk_ids, topk_weights = _topk_from_logits(router_logits, config)

    actual_fixed = op.forward_topk(x, topk_ids, topk_weights)
    token_indices = torch.arange(TOKENS, dtype=torch.int32, device=x.device).repeat_interleave(
        config.top_k
    )
    actual_flat = op.forward_routes(
        x,
        token_indices,
        topk_ids.reshape(-1).contiguous(),
        topk_weights.reshape(-1).contiguous(),
    )
    torch.cuda.synchronize()

    _assert_close(actual_fixed, expected)
    _assert_close(actual_flat, expected)


@pytest.mark.parametrize("activation", ("geglu", "relu_sq"))
def test_sonic_moe_mxfp4_activation_variants(activation):
    config = _config(activation=activation)
    x, w1, w2, router_logits = _make_case(seed=89, activation=activation)
    prepared = prepare_sonic_mxfp4_weights(w1, w2, config)
    expected = sonic_moe_mxfp4_reference(x, w1, w2, router_logits, config)
    actual = SonicMoE(config, prepared)(x, router_logits)
    torch.cuda.synchronize()
    _assert_close(actual, expected)


@pytest.mark.parametrize("activation", ("swiglu", "relu"))
def test_sonic_moe_bf16_bias_matches_reference_fixed_and_flat_routes(activation):
    config = _config(activation=activation)
    x, w1, w2, router_logits = _make_case(seed=97, activation=activation)
    generator = torch.Generator(device=x.device).manual_seed(101)
    b1 = (
        torch.randn(
            (NUM_EXPERTS, config.stage1_projection_size),
            device=x.device,
            dtype=torch.float32,
            generator=generator,
        )
        / 8
    ).to(torch.bfloat16)
    b2 = (
        torch.randn(
            (NUM_EXPERTS, HIDDEN_SIZE),
            device=x.device,
            dtype=torch.float32,
            generator=generator,
        )
        / 8
    ).to(torch.bfloat16)
    prepared = prepare_sonic_bf16_weights(w1, w2, config, b1=b1, b2=b2)
    assert prepared.has_bias
    expected = sonic_moe_reference(
        x, w1, w2, router_logits, config, b1=b1, b2=b2
    )
    op = SonicMoE(config, prepared)

    actual_fixed = op(x, router_logits)
    topk_ids, topk_weights = _topk_from_logits(router_logits, config)
    token_indices = torch.arange(
        TOKENS, dtype=torch.int32, device=x.device
    ).repeat_interleave(config.top_k)
    actual_flat = op.forward_routes(
        x,
        token_indices,
        topk_ids.reshape(-1).contiguous(),
        topk_weights.reshape(-1).contiguous(),
    )
    torch.cuda.synchronize()

    _assert_close(actual_fixed, expected)
    _assert_close(actual_flat, expected)


def test_sonic_moe_mxfp4_bias_matches_reference():
    config = _config(activation="geglu")
    x, w1, w2, router_logits = _make_case(seed=103, activation="geglu")
    b1 = torch.linspace(
        -0.25,
        0.25,
        NUM_EXPERTS * config.stage1_projection_size,
        device=x.device,
        dtype=torch.bfloat16,
    ).view(NUM_EXPERTS, config.stage1_projection_size)
    b2 = torch.linspace(
        0.125,
        -0.125,
        NUM_EXPERTS * HIDDEN_SIZE,
        device=x.device,
        dtype=torch.bfloat16,
    ).view(NUM_EXPERTS, HIDDEN_SIZE)
    prepared = prepare_sonic_mxfp4_weights(w1, w2, config, b1=b1, b2=b2)
    expected = sonic_moe_mxfp4_reference(
        x, w1, w2, router_logits, config, b1=b1, b2=b2
    )
    actual = SonicMoE(config, prepared)(x, router_logits)
    torch.cuda.synchronize()
    _assert_close(actual, expected)


def test_sonic_moe_bias_ragged_duplicate_zero_score_and_empty_routes():
    device = _gfx950_device()
    config = _config(activation="relu")
    x = torch.zeros((3, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
    w1 = torch.zeros(
        (NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
    )
    w2 = torch.zeros(
        (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        dtype=torch.bfloat16,
        device=device,
    )
    b1 = torch.ones(
        (NUM_EXPERTS, INTERMEDIATE_SIZE), dtype=torch.bfloat16, device=device
    )
    b2 = torch.stack(
        [
            torch.full((HIDDEN_SIZE,), expert + 1, dtype=torch.bfloat16, device=device)
            for expert in range(NUM_EXPERTS)
        ]
    )
    op = SonicMoE(
        config,
        prepare_sonic_bf16_weights(w1, w2, config, b1=b1, b2=b2),
    )
    token_indices = torch.tensor([0, 0, 2, 2], dtype=torch.int32, device=device)
    expert_indices = torch.tensor([1, 1, 0, 2], dtype=torch.int32, device=device)
    route_weights = torch.tensor([0.25, -0.5, 0.0, 1.5], device=device)

    actual = op.forward_routes(x, token_indices, expert_indices, route_weights)
    torch.cuda.synchronize()
    expected = torch.zeros_like(x)
    expected[0].fill_(-0.5)
    expected[2].fill_(4.5)
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(actual[1]) == 0

    empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
    empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
    empty = op.forward_routes(x, empty_i32, empty_i32, empty_f32)
    torch.cuda.synchronize()
    assert torch.count_nonzero(empty) == 0


def test_sonic_moe_activation_preserves_legacy_bf16_preactivation_rounding():
    device = _gfx950_device()
    config = _config(num_experts=1, top_k=1, activation="relu_sq")
    value = 1.5078125
    x = torch.zeros((1, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
    x[0, 0] = value
    w1 = torch.zeros(
        (1, INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
    )
    w1[0, :, 0] = value
    w2 = torch.zeros(
        (1, HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=torch.bfloat16, device=device
    )
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))

    op.forward_topk(
        x,
        torch.zeros((1, 1), dtype=torch.int32, device=device),
        torch.ones((1, 1), dtype=torch.float32, device=device),
    )
    torch.cuda.synchronize()

    assert op.workspace is not None
    preactivation = (torch.tensor(value) * torch.tensor(value)).to(torch.bfloat16).float()
    expected = preactivation.square().to(torch.bfloat16)
    without_preactivation_rounding = torch.tensor(value * value).square().to(torch.bfloat16)
    assert expected.item() != without_preactivation_rounding.item()
    assert torch.equal(
        op.workspace.intermediate[0],
        torch.full(
            (INTERMEDIATE_SIZE,), expected.item(), dtype=torch.bfloat16, device=device
        ),
    )


def test_sonic_moe_ragged_routes_match_reference_and_frequency():
    """Flat routes allow missing tokens, duplicate edges, and arbitrary weights."""

    config = _config()
    x, w1, w2, _ = _make_case(seed=19)
    prepared = prepare_sonic_bf16_weights(w1, w2, config)
    op = SonicMoE(config, prepared)
    token_indices = torch.tensor(
        [0, 0, 1, 3, 3, 3, 5, 6, 6, 6, 6],
        dtype=torch.int32,
        device=x.device,
    )
    expert_indices = torch.tensor(
        [1, 3, 0, 2, 2, 1, 3, 0, 1, 2, 3],
        dtype=torch.int32,
        device=x.device,
    )
    route_weights = torch.tensor(
        [0.7, 0.3, 1.2, -0.4, 0.6, 0.0, 0.9, 0.1, 0.2, 0.3, 0.4],
        dtype=torch.float32,
        device=x.device,
    )
    frequency = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=x.device)

    expected = torch.zeros_like(x, dtype=torch.float32)
    x_f32, w1_f32, w2_f32 = x.float(), w1.float(), w2.float()
    for route in range(route_weights.numel()):
        token = int(token_indices[route])
        expert = int(expert_indices[route])
        gate_up = (w1_f32[expert] @ x_f32[token]).to(torch.bfloat16).float()
        gate, up = gate_up.split(INTERMEDIATE_SIZE)
        activated = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).float()
        projected = w2_f32[expert] @ activated
        expected[token].add_(projected * route_weights[route])
    expected = expected.to(torch.bfloat16)

    actual = op.forward_routes(
        x,
        token_indices,
        expert_indices,
        route_weights,
        expert_frequency_out=frequency,
    )
    torch.cuda.synchronize()

    expected_frequency = torch.bincount(
        expert_indices.to(torch.int64), minlength=NUM_EXPERTS
    ).to(torch.int32)
    assert torch.equal(frequency, expected_frequency)
    assert op.workspace is not None
    assert op.workspace.routes == route_weights.numel()
    expected_blocks = sum(
        (int(count) + config.tile_m - 1) // config.tile_m
        for count in expected_frequency
        if int(count) > 0
    )
    assert int(op.workspace.num_valid_ids[0]) == expected_blocks * config.tile_m
    assert int(op.workspace.num_valid_ids[1]) == TOKENS
    _assert_close(actual, expected)

    # Kernel/JIT cache specialization must include tile_m.  Running 16 then 32
    # in one process catches accidental reuse of captured DSL constants.
    config_tile32 = replace(config, tile_m=32)
    op_tile32 = SonicMoE(config_tile32, prepared)
    frequency_tile32 = torch.empty_like(frequency)
    actual_tile32 = op_tile32.forward_routes(
        x,
        token_indices,
        expert_indices,
        route_weights,
        expert_frequency_out=frequency_tile32,
    )
    torch.cuda.synchronize()
    assert op_tile32.workspace is not None
    assert int(op_tile32.workspace.num_valid_ids[0]) == expected_blocks * 32
    assert torch.equal(frequency_tile32, expected_frequency)
    _assert_close(actual_tile32, expected)

    actual_tile16_again = op.forward_routes(
        x,
        token_indices,
        expert_indices,
        route_weights,
    )
    torch.cuda.synchronize()
    assert op.workspace is not None
    assert int(op.workspace.num_valid_ids[0]) == expected_blocks * config.tile_m
    assert torch.equal(op.workspace.expert_frequency, expected_frequency)
    _assert_close(actual_tile16_again, expected)

    empty_indices = torch.empty(0, dtype=torch.int32, device=x.device)
    empty_weights = torch.empty(0, dtype=torch.float32, device=x.device)
    empty_frequency = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=x.device)
    empty_actual = op.forward_routes(
        x,
        empty_indices,
        empty_indices,
        empty_weights,
        expert_frequency_out=empty_frequency,
    )
    torch.cuda.synchronize()
    assert torch.count_nonzero(empty_actual) == 0
    assert torch.count_nonzero(empty_frequency) == 0
    assert op.workspace is not None and op.workspace.max_m_blocks == 0


def test_sonic_moe_ragged_high_fan_in_matches_fp32_reference():
    """An up-rounded token may receive many experts and BF16 atomic contributions."""

    device = _gfx950_device()
    tokens, experts = 3, 64
    config = _config(num_experts=experts, top_k=1)
    generator = torch.Generator(device=device).manual_seed(59)
    x = torch.randn(
        (tokens, HIDDEN_SIZE), dtype=torch.float32, device=device, generator=generator
    ).to(torch.bfloat16)
    w1 = (
        torch.randn(
            (experts, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        / math.sqrt(HIDDEN_SIZE)
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(
            (experts, HIDDEN_SIZE, INTERMEDIATE_SIZE),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        / math.sqrt(INTERMEDIATE_SIZE)
    ).to(torch.bfloat16)
    token_indices = torch.zeros(experts, dtype=torch.int32, device=device)
    expert_indices = torch.arange(experts, dtype=torch.int32, device=device)
    route_weights = torch.softmax(
        torch.randn(experts, dtype=torch.float32, device=device, generator=generator),
        dim=0,
    )
    frequency = torch.empty(experts, dtype=torch.int32, device=device)

    expected_row = torch.zeros(HIDDEN_SIZE, dtype=torch.float32, device=device)
    for expert in range(experts):
        gate_up = (w1[expert].float() @ x[0].float()).to(torch.bfloat16).float()
        gate, up = gate_up.split(INTERMEDIATE_SIZE)
        activated = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).float()
        expected_row.add_((w2[expert].float() @ activated) * route_weights[expert])
    expected = torch.zeros_like(x)
    expected[0] = expected_row.to(torch.bfloat16)

    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    actual = op.forward_routes(
        x,
        token_indices,
        expert_indices,
        route_weights,
        expert_frequency_out=frequency,
    )
    torch.cuda.synchronize()

    assert torch.equal(frequency, torch.ones_like(frequency))
    assert torch.count_nonzero(actual[1:]) == 0
    _assert_close(actual, expected)


def test_sonic_moe_mxfp4_weight_only_forward():
    """A16W4 keeps BF16 activations while consuming per-1x32 MXFP4 weights."""

    device = _gfx950_device()
    config = SonicMoEConfig(
        hidden_size=256,
        intermediate_size=256,
        num_experts=4,
        top_k=2,
        tile_m=16,
        tile_n=256,
        tile_k=256,
    )
    generator = torch.Generator(device=device).manual_seed(43)
    x = torch.randn((7, 256), device=device, dtype=torch.bfloat16, generator=generator)
    w1 = (
        torch.randn((4, 512, 256), device=device, dtype=torch.float32, generator=generator)
        / math.sqrt(256)
    ).to(torch.bfloat16)
    w2 = (
        torch.randn((4, 256, 256), device=device, dtype=torch.float32, generator=generator)
        / math.sqrt(256)
    ).to(torch.bfloat16)
    logits = torch.randn((7, 4), device=device, dtype=torch.bfloat16, generator=generator)

    prepared = prepare_sonic_mxfp4_weights(w1, w2, config)
    assert prepared.weight_dtype == "mxfp4"
    assert prepared.gate_up.dtype == torch.uint8
    assert prepared.down.dtype == torch.uint8
    assert prepared.gate_up_scale is not None
    assert prepared.down_scale is not None
    assert prepared.gate_up.numel() * 2 == w1.numel()
    assert prepared.down.numel() * 2 == w2.numel()

    expected = sonic_moe_mxfp4_reference(x, w1, w2, logits, config)
    actual = SonicMoE(config, prepared)(x, logits)
    torch.cuda.synchronize()
    _assert_close(actual, expected)
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    ).item()
    assert cosine >= 0.999


def test_sonic_moe_mxfp4_quantizer_uses_rne_midpoints():
    """Golden codes prevent a circular quantize/dequantize oracle."""

    from tests.kernels.utils import gemm_common_utils

    device = _gfx950_device()
    weight = torch.zeros((1, 3, 32), dtype=torch.float32, device=device)
    weight[0, 0, :4] = torch.tensor([0.75, 1.75, 3.5, 4.0], device=device)
    weight[0, 1, :4] = torch.tensor([-0.75, -1.75, -3.5, -4.0], device=device)
    weight[0, 2, 0] = -0.0
    packed, scale = _quantize_mxfp4_weight(weight)

    assert scale[0, 0, 0].item() == 127
    assert scale[0, 1, 0].item() == 127
    assert scale[0, 2, 0].item() == 0
    assert packed[0, 0, 0].item() == 0x42
    assert packed[0, 0, 1].item() == 0x66
    assert packed[0, 1, 0].item() == 0xCA
    assert packed[0, 1, 1].item() == 0xEE
    assert packed[0, 2, 0].item() == 0x08
    assert not packed[0, 2, 1:].any().item()

    blocks = weight.view(-1, 32)
    reference_scale = gemm_common_utils.f32_to_e8m0(
        blocks.abs().amax(dim=1) / 4.0
    ).view(torch.uint8)
    reference_values = blocks / gemm_common_utils.e8m0_to_f32(reference_scale)[:, None]
    reference_packed = (
        gemm_common_utils.f32_to_mxfp4(reference_values)
        .view(torch.uint8)
        .view_as(packed)
    )
    assert torch.equal(scale.view(-1), reference_scale)
    assert torch.equal(packed, reference_packed)


def test_sonic_moe_reuses_workspace_by_device_stream_and_token_count():
    config = _config()
    x, w1, w2, router_logits = _make_case()
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))

    first = op(x, router_logits)
    torch.cuda.synchronize()
    workspace_t7 = op.workspace
    assert workspace_t7 is not None

    out = torch.empty_like(first)
    returned = op(x, router_logits, out=out)
    torch.cuda.synchronize()
    assert returned is out
    assert op.workspace is workspace_t7
    torch.testing.assert_close(out.float(), first.float(), rtol=3e-2, atol=5e-2)

    x_t3, _, _, logits_t3 = _make_case(tokens=3, seed=23)
    result_t3 = op(x_t3, logits_t3)
    torch.cuda.synchronize()
    workspace_t3 = op.workspace
    assert result_t3.shape == (3, HIDDEN_SIZE)
    assert workspace_t3 is not None
    assert workspace_t3 is not workspace_t7

    op(x, router_logits)
    torch.cuda.synchronize()
    assert op.workspace is workspace_t7


def test_sonic_moe_workspace_cache_is_bounded_lru():
    config = _config()
    _, w1, w2, _ = _make_case()
    op = SonicMoE(
        config,
        prepare_sonic_bf16_weights(w1, w2, config),
        max_cached_workspaces=2,
    )

    workspace_t7 = op.reserve(7)
    workspace_t3 = op.reserve(3)
    cached = tuple(op._workspaces.values())
    assert cached[0] is workspace_t7
    assert cached[1] is workspace_t3

    assert op.reserve(7) is workspace_t7
    cached = tuple(op._workspaces.values())
    assert cached[0] is workspace_t3
    assert cached[1] is workspace_t7

    workspace_t5 = op.reserve(5)
    cached = tuple(op._workspaces.values())
    assert cached[0] is workspace_t7
    assert cached[1] is workspace_t5

    replacement_t3 = op.reserve(3)
    cached = tuple(op._workspaces.values())
    assert replacement_t3 is not workspace_t3
    assert cached[0] is workspace_t5
    assert cached[1] is replacement_t3
    assert op.workspace is replacement_t3

    op.clear_workspace()
    assert not op._workspaces
    assert op.workspace is None


@pytest.mark.parametrize(
    ("capacity", "exception"),
    [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_sonic_moe_rejects_invalid_workspace_cache_capacity(capacity, exception):
    config = _config()
    _, w1, w2, _ = _make_case()
    weights = prepare_sonic_bf16_weights(w1, w2, config)

    with pytest.raises(exception, match="max_cached_workspaces"):
        SonicMoE(config, weights, max_cached_workspaces=capacity)


@pytest.mark.parametrize("entrypoint", ["router", "topk"])
def test_sonic_moe_serializes_same_workspace_enqueues(monkeypatch, entrypoint):
    config = _config()
    x, w1, w2, router_logits = _make_case()
    topk_ids, topk_weights = _topk_from_logits(router_logits, config)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    stream = torch.cuda.Stream(device=x.device)
    with torch.cuda.device(x.device), torch.cuda.stream(stream):
        workspace = op.reserve(TOKENS)

    first_sort_entered = threading.Event()
    second_lock_attempted = threading.Event()
    second_sort_entered = threading.Event()
    release_first_sort = threading.Event()
    first_gemms_entered = threading.Event()
    release_first_gemms = threading.Event()
    order = []
    errors = []
    result_lock = threading.Lock()

    class TrackingLock:
        def __init__(self):
            self._lock = threading.Lock()

        def __enter__(self):
            if threading.current_thread().name == "sonic-second":
                second_lock_attempted.set()
            self._lock.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self._lock.release()

    workspace._launch_lock = TrackingLock()

    def fake_sort(*args, **kwargs):
        del args, kwargs
        name = threading.current_thread().name
        with result_lock:
            order.append((name, "sort"))
        if name == "sonic-first":
            first_sort_entered.set()
            if not release_first_sort.wait(timeout=5):
                raise TimeoutError("first sort was not released")
        else:
            second_sort_entered.set()

    def fake_grouped_gemms(hidden_states, active_workspace, out):
        assert hidden_states is x
        assert active_workspace is workspace
        name = threading.current_thread().name
        with result_lock:
            order.append((name, "gemms"))
        if name == "sonic-first":
            first_gemms_entered.set()
            if not release_first_gemms.wait(timeout=5):
                raise TimeoutError("first GEMMs were not released")
        return out

    monkeypatch.setattr("kernels.moe.sonic.moe_softmax_sort_flydsl", fake_sort)
    monkeypatch.setattr("kernels.moe.sonic.moe_sorting_flydsl", fake_sort)
    monkeypatch.setattr(op, "_run_grouped_gemms", fake_grouped_gemms)

    outputs = (torch.empty_like(x), torch.empty_like(x))

    def run(output):
        try:
            with torch.cuda.device(x.device), torch.cuda.stream(stream):
                if entrypoint == "router":
                    result = op(x, router_logits, out=output)
                else:
                    result = op.forward_topk(
                        x,
                        topk_ids,
                        topk_weights,
                        out=output,
                    )
            assert result is output
        except Exception as error:
            with result_lock:
                errors.append(error)

    first = threading.Thread(target=run, args=(outputs[0],), name="sonic-first")
    second = threading.Thread(target=run, args=(outputs[1],), name="sonic-second")
    first.start()
    try:
        assert first_sort_entered.wait(timeout=5)
        second.start()
        assert second_lock_attempted.wait(timeout=5)
        assert not second_sort_entered.is_set()
        release_first_sort.set()
        assert first_gemms_entered.wait(timeout=5)
        assert not second_sort_entered.is_set()
        release_first_gemms.set()
    finally:
        release_first_sort.set()
        release_first_gemms.set()
        first.join(timeout=5)
        if second.ident is not None:
            second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not errors
    assert order == [
        ("sonic-first", "sort"),
        ("sonic-first", "gemms"),
        ("sonic-second", "sort"),
        ("sonic-second", "gemms"),
    ]


def test_sonic_moe_workspace_bound_scales_with_active_experts():
    """Decode must not allocate or launch one padded tile for every expert."""

    device = _gfx950_device()
    dense_routes = SonicMoEWorkspace.allocate(
        _config(num_experts=2, top_k=2, tile_m=16),
        tokens=16,
        device=device,
    )
    assert dense_routes.max_m_blocks == 2
    assert dense_routes.max_padded_tokens == 32

    config = _config(num_experts=896, top_k=2, tile_m=32)
    generator = torch.Generator(device=device).manual_seed(53)
    x = torch.randn((1, HIDDEN_SIZE), dtype=torch.bfloat16, device=device, generator=generator)
    w1 = (
        torch.randn(
            (896, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / math.sqrt(HIDDEN_SIZE)
    )
    w2 = (
        torch.randn(
            (896, HIDDEN_SIZE, INTERMEDIATE_SIZE),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / math.sqrt(INTERMEDIATE_SIZE)
    )
    logits = torch.randn((1, 896), dtype=torch.bfloat16, device=device, generator=generator)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    expected = sonic_moe_reference(x, w1, w2, logits, config)
    actual = op(x, logits)
    torch.cuda.synchronize()

    workspace = op.workspace
    assert workspace is not None
    assert workspace.max_padded_tokens == 64
    assert workspace.max_m_blocks == 2
    assert workspace.intermediate.shape == (64, INTERMEDIATE_SIZE)
    _assert_close(actual, expected)

    topk_ids, topk_weights = _topk_from_logits(logits, config)
    ragged_frequency = torch.empty(896, dtype=torch.int32, device=device)
    ragged_actual = op.forward_routes(
        x,
        torch.zeros(config.top_k, dtype=torch.int32, device=device),
        topk_ids.reshape(-1),
        topk_weights.reshape(-1),
        expert_frequency_out=ragged_frequency,
    )
    torch.cuda.synchronize()
    assert op.workspace is not None
    assert op.workspace.routes == config.top_k
    assert op.workspace.max_padded_tokens == 64
    assert int(ragged_frequency.sum()) == config.top_k
    _assert_close(ragged_actual, expected)

    mxfp4_weights = prepare_sonic_mxfp4_weights(w1, w2, config)
    mxfp4_expected = sonic_moe_mxfp4_reference(x, w1, w2, logits, config)
    mxfp4_op = SonicMoE(config, mxfp4_weights)
    mxfp4_actual = mxfp4_op(x, logits)
    torch.cuda.synchronize()
    assert mxfp4_op.workspace is not None
    assert mxfp4_op.workspace.max_padded_tokens == 64
    _assert_close(mxfp4_actual, mxfp4_expected)


def test_sonic_moe_autotuner_search_and_disk_cache(tmp_path):
    config = _config()
    x, w1, w2, router_logits = _make_case(seed=47)
    weights = prepare_sonic_bf16_weights(w1, w2, config)
    candidates = (config, replace(config, tile_m=32))
    with pytest.raises(ValueError, match="at least one"):
        SonicMoEAutotuner(config, weights, candidates=[], cache_dir=tmp_path)
    tuner = SonicMoEAutotuner(
        config,
        weights,
        candidates=candidates,
        warmup=0,
        rep=1,
        cache_dir=tmp_path,
    )
    b1 = torch.zeros(
        (NUM_EXPERTS, config.stage1_projection_size),
        dtype=torch.bfloat16,
        device=x.device,
    )
    b2 = torch.zeros(
        (NUM_EXPERTS, HIDDEN_SIZE), dtype=torch.bfloat16, device=x.device
    )
    biased_tuner = SonicMoEAutotuner(
        config,
        prepare_sonic_bf16_weights(w1, w2, config, b1=b1, b2=b2),
        candidates=candidates,
        warmup=0,
        rep=1,
        cache_dir=tmp_path / "biased",
    )
    assert tuner._cache_key(x, router_logits) != biased_tuner._cache_key(x, router_logits)

    expected = sonic_moe_reference(x, w1, w2, router_logits, config)
    actual = tuner(x, router_logits)
    torch.cuda.synchronize()
    _assert_close(actual, expected)
    assert tuner.best_config in candidates
    assert tuner.search_count == 1
    assert tuner.cache_file.is_file()

    tuner(x, router_logits)
    torch.cuda.synchronize()
    assert tuner.search_count == 1

    reloaded = SonicMoEAutotuner(
        config,
        weights,
        candidates=candidates,
        warmup=0,
        rep=1,
        cache_dir=tmp_path,
    )
    cached = reloaded(x, router_logits)
    torch.cuda.synchronize()
    _assert_close(cached, expected)
    assert reloaded.search_count == 0
    assert reloaded.best_config == tuner.best_config
    assert len(reloaded._ops) == 1

    scaled = torch.randn_like(actual.float())
    assert not SonicMoEAutotuner._candidate_matches(scaled, scaled * 10)

    profiled = SonicMoEAutotuner(
        config,
        weights,
        candidates=candidates,
        warmup=0,
        rep=1,
        cache_dir=tmp_path,
        profile_key="decode-skew",
    )
    profiled(x, router_logits)
    torch.cuda.synchronize()
    assert profiled.search_count == 1

    unvalidated = SonicMoEAutotuner(
        config,
        weights,
        candidates=(config,),
        warmup=0,
        rep=1,
        cache_dir=tmp_path / "unvalidated",
        validate_candidates=False,
    )
    unvalidated(x, router_logits)
    torch.cuda.synchronize()
    assert unvalidated.search_count == 1
    assert not unvalidated.cache_file.exists()

    non_object_cache_dir = tmp_path / "non_object_json"
    non_object_cache_dir.mkdir()
    non_object_cache_file = non_object_cache_dir / "sonic_moe.json"
    non_object_cache_file.write_text("[]", encoding="utf-8")
    recovered = SonicMoEAutotuner(
        config,
        weights,
        candidates=(config,),
        warmup=0,
        rep=1,
        cache_dir=non_object_cache_dir,
    )
    recovered(x, router_logits)
    torch.cuda.synchronize()
    assert recovered.search_count == 1
    assert '"version": 4' in non_object_cache_file.read_text(encoding="utf-8")


def test_sonic_moe_router_and_multiphase_sort_fallback():
    """T > 128 exercises supplied top-k scratch and sorting workspace."""

    config = _config()
    x, w1, w2, router_logits = _make_case(tokens=129, seed=31)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))

    expected = sonic_moe_reference(x, w1, w2, router_logits, config)
    actual = op(x, router_logits)
    torch.cuda.synchronize()
    assert op.workspace is not None
    assert op.workspace.sorting_workspace is not None
    _assert_close(actual, expected)


def test_sonic_moe_uses_independent_workspaces_across_streams():
    config = _config()
    x_a, w1, w2, logits_a = _make_case(seed=37)
    x_b, _, _, logits_b = _make_case(seed=41)
    op = SonicMoE(
        config,
        prepare_sonic_bf16_weights(w1, w2, config),
        max_cached_workspaces=1,
    )
    expected_a = sonic_moe_reference(x_a, w1, w2, logits_a, config)
    expected_b = sonic_moe_reference(x_b, w1, w2, logits_b, config)
    out_a = torch.empty_like(expected_a)
    out_b = torch.empty_like(expected_b)
    out_a_again = torch.empty_like(expected_a)

    default_stream = torch.cuda.current_stream(x_a.device)
    stream_a = torch.cuda.Stream(device=x_a.device)
    stream_b = torch.cuda.Stream(device=x_a.device)
    stream_a.wait_stream(default_stream)
    stream_b.wait_stream(default_stream)

    with torch.cuda.stream(stream_a):
        actual_a = op(x_a, logits_a, out=out_a)
        workspace_a_ref = weakref.ref(op.workspace)
    with torch.cuda.stream(stream_b):
        actual_b = op(x_b, logits_b, out=out_b)
        workspace_b_ref = weakref.ref(op.workspace)
    assert workspace_a_ref() is None
    with torch.cuda.stream(stream_a):
        actual_a_again = op(x_a, logits_a, out=out_a_again)
        workspace_a_again = op.workspace
    assert workspace_b_ref() is None

    torch.cuda.synchronize()
    assert actual_a is out_a
    assert actual_b is out_b
    assert actual_a_again is out_a_again
    assert workspace_a_again is not None
    assert len(op._workspaces) == 1
    assert next(iter(op._workspaces.values())) is workspace_a_again
    _assert_close(actual_a, expected_a)
    _assert_close(actual_b, expected_b)
    _assert_close(actual_a_again, expected_a)


@pytest.mark.multi_gpu
def test_sonic_moe_selects_input_device_and_restores_current_device():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least two ROCm GPUs")

    for index in (0, 1):
        device = torch.device("cuda", index)
        arch = str(getattr(torch.cuda.get_device_properties(device), "gcnArchName", ""))
        if not arch.startswith("gfx950"):
            pytest.skip(f"SonicMoE test requires two gfx950 devices, found {device}={arch!r}")

    target_device = torch.device("cuda", 1)
    with torch.cuda.device(target_device):
        config = _config()
        x, w1, w2, router_logits = _make_case(seed=71)
        assert x.device == target_device
        topk_ids, topk_weights = _topk_from_logits(router_logits, config)
        expected = sonic_moe_reference(x, w1, w2, router_logits, config)
        op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
        router_out = torch.empty_like(expected)
        topk_out = torch.empty_like(expected)

    with torch.cuda.device(0):
        assert torch.cuda.current_device() == 0
        actual_router = op(x, router_logits, out=router_out)
        assert torch.cuda.current_device() == 0
        actual_topk = op.forward_topk(x, topk_ids, topk_weights, out=topk_out)
        assert torch.cuda.current_device() == 0

    torch.cuda.synchronize(target_device)
    assert actual_router is router_out
    assert actual_topk is topk_out
    _assert_close(actual_router, expected)
    _assert_close(actual_topk, expected)


@pytest.mark.multi_gpu
def test_sonic_moe_runs_sequentially_on_two_devices():
    """A launcher materialized on cuda:0 must not be reused on cuda:1."""

    from flydsl.compiler.jit_function import _current_device_cache_signature
    from flydsl.runtime.device_runtime import get_device_runtime

    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least two ROCm GPUs")

    devices = (torch.device("cuda", 0), torch.device("cuda", 1))
    for device in devices:
        properties = torch.cuda.get_device_properties(device)
        arch = str(getattr(properties, "gcnArchName", ""))
        if not arch.startswith("gfx950"):
            pytest.skip(f"SonicMoE test requires two gfx950 devices, found {device}={arch!r}")

    config = _config()
    results = []
    _get_stage1_launcher.cache_clear()
    _get_stage2_launcher.cache_clear()
    try:
        for device, seed in zip(devices, (61, 67)):
            with torch.cuda.device(device):
                assert get_device_runtime().current_device_id() == device.index
                assert _current_device_cache_signature() == ("rocm", device.index)
                x, w1, w2, router_logits = _make_case(seed=seed)
                assert x.device == device

                expected = sonic_moe_reference(x, w1, w2, router_logits, config)
                actual = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))(
                    x, router_logits
                )
                torch.cuda.synchronize(device)
                results.append((actual, expected))

        assert _get_stage1_launcher.cache_info().currsize == 2
        assert _get_stage2_launcher.cache_info().currsize == 2
    finally:
        _get_stage1_launcher.cache_clear()
        _get_stage2_launcher.cache_clear()

    for actual, expected in results:
        _assert_close(actual, expected)


def test_sonic_moe_non_power_of_two_router_fallback():
    """Arbitrary expert counts use torch top-k but keep FlyDSL sort/GEMMs."""

    device = _gfx950_device()
    config = _config(num_experts=6)
    assert not config.supports_flydsl_router
    generator = torch.Generator(device=device).manual_seed(29)
    x = torch.randn((3, HIDDEN_SIZE), device=device, dtype=torch.bfloat16, generator=generator)
    w1 = torch.randn(
        (6, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ) / math.sqrt(HIDDEN_SIZE)
    w2 = torch.randn(
        (6, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ) / math.sqrt(INTERMEDIATE_SIZE)
    logits = torch.randn((3, 6), device=device, dtype=torch.bfloat16, generator=generator)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))

    expected = sonic_moe_reference(x, w1, w2, logits, config)
    actual = op(x, logits)
    torch.cuda.synchronize()
    _assert_close(actual, expected)


def test_sonic_moe_config_validation():
    device = _gfx950_device()
    config = _config()
    assert config.down_tile_n is None
    assert config.down_tile_k is None
    assert config.stage2_tile_n == config.tile_n
    assert config.stage2_tile_k == config.tile_k
    assert config.renormalize is True

    invalid_configs = [
        {"hidden_size": 0},
        {"intermediate_size": 0},
        {"num_experts": 0},
        {"top_k": 0},
        {"top_k": NUM_EXPERTS + 1},
        {"tile_m": 0},
        {"tile_n": 0},
        {"tile_k": 0},
        {"hidden_size": HIDDEN_SIZE - 1},
        {"intermediate_size": INTERMEDIATE_SIZE - 1},
    ]
    for override in invalid_configs:
        with pytest.raises((TypeError, ValueError)):
            _config(**override)
    with pytest.raises(ValueError, match="unsupported activation"):
        _config(activation="not-an-activation")
    with pytest.raises(TypeError, match="activation must be a string"):
        _config(activation=None)

    large_mesh_config = _config(num_experts=300, top_k=1)
    with pytest.raises(ValueError, match="signed 32-bit byte-index"):
        SonicMoEWorkspace.allocate(large_mesh_config, 8_000_000, device)


def test_sonic_moe_tensor_shape_and_dtype_validation():
    config = _config()
    x, w1, w2, router_logits = _make_case()

    with pytest.raises((TypeError, ValueError)):
        prepare_sonic_bf16_weights(w1[:, :-1, :], w2, config)
    with pytest.raises((TypeError, ValueError)):
        prepare_sonic_bf16_weights(w1, w2[:, :, :-1], config)
    prepared_fp32 = prepare_sonic_bf16_weights(w1.float(), w2.float(), config)
    assert prepared_fp32.gate_up.dtype == torch.bfloat16
    assert prepared_fp32.down.dtype == torch.bfloat16
    with pytest.raises((TypeError, ValueError)):
        prepare_sonic_bf16_weights(w1.to(torch.int16), w2, config)

    b1 = torch.randn(
        (NUM_EXPERTS, config.stage1_projection_size),
        dtype=torch.bfloat16,
        device=x.device,
    )
    b2 = torch.randn(
        (NUM_EXPERTS, HIDDEN_SIZE), dtype=torch.bfloat16, device=x.device
    )
    with pytest.raises(ValueError, match="both be provided"):
        prepare_sonic_bf16_weights(w1, w2, config, b1=b1)
    with pytest.raises(ValueError, match="b1 must have shape"):
        prepare_sonic_bf16_weights(w1, w2, config, b1=b1[:, :-1], b2=b2)
    with pytest.raises(ValueError, match="b2 must have shape"):
        prepare_sonic_bf16_weights(w1, w2, config, b1=b1, b2=b2[:, :-1])
    with pytest.raises(TypeError, match="b1/b2 must be floating point"):
        prepare_sonic_bf16_weights(w1, w2, config, b1=b1.to(torch.int16), b2=b2)

    prepared_bias = prepare_sonic_bf16_weights(w1, w2, config, b1=b1, b2=b2)
    prepared_b1 = prepared_bias.stage1_bias.clone()
    prepared_b2 = prepared_bias.stage2_bias.clone()
    b1.zero_()
    b2.zero_()
    assert torch.equal(prepared_bias.stage1_bias, prepared_b1)
    assert torch.equal(prepared_bias.stage2_bias, prepared_b2)

    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    with pytest.raises((TypeError, ValueError)):
        op(x[:, :-1], router_logits)
    with pytest.raises((TypeError, ValueError)):
        op(x.float(), router_logits)
    misaligned_x_storage = torch.empty(
        TOKENS * HIDDEN_SIZE + 1, device=x.device, dtype=torch.bfloat16
    )
    misaligned_x = misaligned_x_storage[1:].view(TOKENS, HIDDEN_SIZE)
    assert misaligned_x.is_contiguous() and misaligned_x.data_ptr() % 16
    with pytest.raises(ValueError, match="16-byte aligned"):
        op(misaligned_x, router_logits)
    with pytest.raises((TypeError, ValueError)):
        op(x, router_logits[:, :-1])
    with pytest.raises((TypeError, ValueError)):
        op(x, router_logits[:-1])
    with pytest.raises((TypeError, ValueError)):
        op(
            x,
            router_logits,
            out=torch.empty((TOKENS, HIDDEN_SIZE - 1), device=x.device, dtype=x.dtype),
        )
    with pytest.raises((TypeError, ValueError)):
        op(x, router_logits, out=x)
    misaligned_out_storage = torch.empty(
        TOKENS * HIDDEN_SIZE + 1, device=x.device, dtype=torch.bfloat16
    )
    misaligned_out = misaligned_out_storage[1:].view(TOKENS, HIDDEN_SIZE)
    assert misaligned_out.is_contiguous() and misaligned_out.data_ptr() % 4 == 2
    with pytest.raises(ValueError, match="4-byte aligned"):
        op(x, router_logits, out=misaligned_out)
    with pytest.raises((TypeError, ValueError)):
        op(
            x,
            router_logits,
            out=torch.empty_like(x, requires_grad=True),
        )
    workspace = op.reserve(TOKENS)
    internal_out_alias = workspace.intermediate.flatten()[: TOKENS * HIDDEN_SIZE].view(
        TOKENS, HIDDEN_SIZE
    )
    with pytest.raises((TypeError, ValueError)):
        op(x, router_logits, out=internal_out_alias)
    with pytest.raises((TypeError, ValueError)):
        op(x.detach().requires_grad_(True), router_logits)
    with pytest.raises((TypeError, ValueError)):
        op(x, router_logits.detach().requires_grad_(True))

    topk_ids, topk_weights = _topk_from_logits(router_logits, config)
    with pytest.raises((TypeError, ValueError)):
        op.forward_topk(x, topk_ids[:, :1], topk_weights)
    with pytest.raises((TypeError, ValueError)):
        op.forward_topk(x, topk_ids, topk_weights[:, :1])


def test_sonic_moe_rejects_malformed_prepared_storage_and_mxfp4_tiles():
    config = _config()
    _, w1, w2, _ = _make_case()
    prepared = prepare_sonic_bf16_weights(w1, w2, config)
    with pytest.raises(ValueError, match="prepared gate/up storage"):
        SonicMoE(config, replace(prepared, gate_up=prepared.gate_up.flatten()[:16]))
    with pytest.raises(TypeError, match="dummy_scale"):
        SonicMoE(
            config,
            replace(prepared, dummy_scale=prepared.dummy_scale.to(torch.int32)),
        )

    mxfp4 = prepare_sonic_mxfp4_weights(w1, w2, config)
    assert mxfp4.gate_up_scale is not None
    with pytest.raises(ValueError, match="wrong padded size"):
        SonicMoE(
            config,
            replace(mxfp4, gate_up_scale=mxfp4.gate_up_scale[:-4]),
        )
    nonfinite_w1 = w1.clone()
    nonfinite_w1[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        prepare_sonic_mxfp4_weights(nonfinite_w1, w2, config)

    tile64 = _config(tile_m=32, tile_k=64, down_tile_k=64)
    tile64_weights = prepare_sonic_mxfp4_weights(w1, w2, tile64)
    with pytest.raises(ValueError, match=">= 128"):
        SonicMoE(tile64, tile64_weights)

    format_specific_limit = SonicMoEConfig(
        hidden_size=32768,
        intermediate_size=32768,
        num_experts=1,
        top_k=1,
        tile_m=16,
        tile_n=128,
        tile_k=128,
    )
    tiny_bf16 = torch.empty(16, dtype=torch.bfloat16, device=w1.device)
    unsafe_bf16 = SonicMoEWeights(
        gate_up=tiny_bf16,
        down=tiny_bf16,
        dummy_scale=torch.zeros(1, dtype=torch.uint8, device=w1.device),
        config=format_specific_limit,
    )
    with pytest.raises(ValueError, match="BF16 gate/up weights"):
        SonicMoE(format_specific_limit, unsafe_bf16)

    huge_expert_config = _config(num_experts=2_097_153)
    tiny = torch.empty(16, dtype=torch.uint8, device=w1.device)
    unsafe = SonicMoEWeights(
        gate_up=tiny,
        down=tiny,
        dummy_scale=tiny[:1],
        config=huge_expert_config,
        gate_up_scale=tiny,
        down_scale=tiny,
        weight_dtype="mxfp4",
    )
    with pytest.raises(ValueError, match="32-bit buffer-offset limit"):
        SonicMoE(huge_expert_config, unsafe)
