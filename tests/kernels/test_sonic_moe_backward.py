# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Correctness and contract tests for the first gfx950 SonicMoE backward."""

import math

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.moe.sonic import SonicMoEConfig, sonic_moe_backward

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


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


def _make_case(tokens, hidden_size, intermediate_size, num_experts, topk, seed):
    device = _gfx950_device()
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        (tokens, hidden_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    w1 = (
        torch.randn(
            (num_experts, 2 * intermediate_size, hidden_size),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        / math.sqrt(hidden_size)
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(
            (num_experts, hidden_size, intermediate_size),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        / math.sqrt(intermediate_size)
    ).to(torch.bfloat16)
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
    ).to(torch.bfloat16)
    return x, w1, w2, topk_ids, topk_weights, grad_output


@torch.no_grad()
def _backward_reference(x, w1, w2, topk_ids, topk_weights, grad_output):
    """Match the standalone backward's explicit A16 materialization contract."""

    tokens, hidden_size = x.shape
    num_experts, projection_size, _ = w1.shape
    topk = topk_ids.shape[1]
    intermediate_size = projection_size // 2
    dx_routes = torch.empty(
        (tokens, topk, hidden_size),
        dtype=torch.bfloat16,
        device=x.device,
    )
    dw1 = torch.zeros_like(w1)
    dw2 = torch.zeros_like(w2)
    dtopk_weights = torch.empty_like(topk_weights)

    for expert in range(num_experts):
        pairs = (topk_ids == expert).nonzero(as_tuple=False)
        if pairs.numel() == 0:
            continue
        token_indices, slots = pairs[:, 0], pairs[:, 1]
        x_e = x[token_indices]
        dout_e = grad_output[token_indices]
        scores_e = topk_weights[token_indices, slots]

        preactivation = (x_e.float() @ w1[expert].float().transpose(0, 1)).to(torch.bfloat16)
        gate, up = preactivation.float().split(intermediate_size, dim=1)
        activation = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        projection = (activation.float() @ w2[expert].float().transpose(0, 1)).to(torch.bfloat16)
        dtopk_weights[token_indices, slots] = (dout_e.float() * projection.float()).sum(dim=1)
        dy = (dout_e.float() * scores_e[:, None]).to(torch.bfloat16)
        da = (dy.float() @ w2[expert].float()).to(torch.bfloat16).float()
        sigmoid = torch.sigmoid(gate)
        dz = torch.cat(
            (
                da * up * sigmoid * (1.0 + gate * (1.0 - sigmoid)),
                da * gate * sigmoid,
            ),
            dim=1,
        ).to(torch.bfloat16)

        dw2[expert] = (dy.float().transpose(0, 1) @ activation.float()).to(torch.bfloat16)
        dw1[expert] = (dz.float().transpose(0, 1) @ x_e.float()).to(torch.bfloat16)
        dx_routes[token_indices, slots] = (dz.float() @ w1[expert].float()).to(torch.bfloat16)

    dx = dx_routes.float().sum(dim=1).to(torch.bfloat16)
    return dx, dw1, dw2, dtopk_weights


@pytest.mark.parametrize(
    "tokens,hidden_size,intermediate_size,num_experts,topk",
    (
        (7, 128, 64, 4, 2),
        (9, 256, 128, 5, 1),
        (65, 128, 64, 3, 2),
        (11, 512, 256, 8, 4),
    ),
)
def test_sonic_moe_backward_matches_a16_reference(
    tokens,
    hidden_size,
    intermediate_size,
    num_experts,
    topk,
):
    config = _config(hidden_size, intermediate_size, num_experts, topk)
    args = _make_case(tokens, hidden_size, intermediate_size, num_experts, topk, seed=211 + topk)

    actual = sonic_moe_backward(*args, config)
    expected = _backward_reference(*args)
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

    # The intentionally unused expert must receive exact zero weight grads.
    assert torch.count_nonzero(actual[1][-1]) == 0
    assert torch.count_nonzero(actual[2][-1]) == 0


def test_sonic_moe_backward_repeated_calls_do_not_alias_workspace():
    config = _config(128, 64, 4, 2)
    first = _make_case(7, 128, 64, 4, 2, seed=251)
    second = _make_case(7, 128, 64, 4, 2, seed=257)

    first_actual = tuple(t.clone() for t in sonic_moe_backward(*first, config))
    sonic_moe_backward(*second, config)
    first_again = sonic_moe_backward(*first, config)
    torch.cuda.synchronize()

    for saved, repeated in zip(first_actual, first_again):
        assert torch.equal(saved, repeated)


def test_sonic_moe_backward_rejects_unsupported_contracts():
    args = _make_case(7, 128, 64, 4, 2, seed=263)

    with pytest.raises(ValueError, match="compute_dtype='bf16'"):
        sonic_moe_backward(*args, _config(128, 64, 4, 2, compute_dtype="fp16"))
    with pytest.raises(ValueError, match="activation='swiglu'"):
        sonic_moe_backward(*args, _config(128, 64, 4, 2, activation="relu"))

    noncontiguous_dout = args[-1].transpose(0, 1).contiguous().transpose(0, 1)
    assert not noncontiguous_dout.is_contiguous()
    with pytest.raises(ValueError, match="grad_output must be contiguous"):
        sonic_moe_backward(*args[:-1], noncontiguous_dout, _config(128, 64, 4, 2))
