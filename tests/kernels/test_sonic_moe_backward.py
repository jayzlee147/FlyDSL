# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Correctness and contract tests for the first gfx950 SonicMoE backward."""

import math

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.moe.sonic import SonicMoEConfig, sonic_moe_backward

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
):
    device = _gfx950_device()
    generator = torch.Generator(device=device).manual_seed(seed)
    projection_size = intermediate_size * (2 if activation in _GLU_ACTIVATIONS else 1)
    x = torch.randn(
        (tokens, hidden_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    w1 = (
        torch.randn(
            (num_experts, projection_size, hidden_size),
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
    ).to(torch.bfloat16)
    b2 = (
        torch.randn(
            (w2.shape[0], w2.shape[1]),
            dtype=torch.float32,
            device=w2.device,
            generator=generator,
        )
        / math.sqrt(w2.shape[-1])
    ).to(torch.bfloat16)
    return b1, b2


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


def _activation_reference(preactivation, da, intermediate_size, activation_name):
    if activation_name in _GLU_ACTIVATIONS:
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
        dz = torch.cat((da * up * derivative, da * activated_gate), dim=1)
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
    return activated.to(torch.bfloat16), dz.to(torch.bfloat16)


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
        dtype=torch.bfloat16,
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
        preactivation = preactivation.to(torch.bfloat16)
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
        )
        projection = activation.float() @ w2[expert].float().transpose(0, 1)
        if b2 is not None:
            projection = projection + b2[expert].float()
        projection = projection.to(torch.bfloat16)
        dtopk_weights[token_indices, slots] = (dout_e.float() * projection.float()).sum(dim=1)
        dy = (dout_e.float() * scores_e[:, None]).to(torch.bfloat16)
        da = (dy.float() @ w2[expert].float()).to(torch.bfloat16).float()
        _, dz = _activation_reference(
            preactivation,
            da,
            intermediate_size,
            activation_name,
        )

        dw2[expert] = (dy.float().transpose(0, 1) @ activation.float()).to(torch.bfloat16)
        dw1[expert] = (dz.float().transpose(0, 1) @ x_e.float()).to(torch.bfloat16)
        dx_routes[token_indices, slots] = (dz.float() @ w1[expert].float()).to(torch.bfloat16)
        if db1 is not None:
            db1[expert] = dz.float().sum(dim=0).to(torch.bfloat16)
            db2[expert] = dy.float().sum(dim=0).to(torch.bfloat16)

    dx = dx_routes.float().sum(dim=1).to(torch.bfloat16)
    result = (dx, dw1, dw2, dtopk_weights)
    return (*result, db1, db2) if has_bias else result


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


@pytest.mark.parametrize("activation_name", _ACTIVATIONS)
def test_sonic_moe_backward_bias_gradients_match_a16_reference(activation_name):
    hidden_size, intermediate_size, num_experts, topk = 128, 64, 4, 2
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        activation=activation_name,
    )
    args = _make_case(
        7,
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        seed=307,
        activation=activation_name,
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
        assert actual_bias_gradient.dtype == torch.bfloat16
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
def test_sonic_moe_backward_bias_reduction_spans_route_tiles(activation_name):
    tokens, hidden_size, intermediate_size, num_experts, topk = 65, 128, 64, 3, 2
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        activation=activation_name,
    )
    args = _make_case(
        tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        seed=331,
        activation=activation_name,
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
def test_sonic_moe_backward_activation_variants_match_a16_reference(activation_name):
    hidden_size, intermediate_size, num_experts, topk = 128, 64, 4, 2
    config = _config(
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        activation=activation_name,
    )
    args = _make_case(
        7,
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        seed=271,
        activation=activation_name,
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
    b1, b2 = _make_biases(args[1], args[2], seed=317)

    with pytest.raises(ValueError, match="compute_dtype='bf16'"):
        sonic_moe_backward(*args, _config(128, 64, 4, 2, compute_dtype="fp16"))
    with pytest.raises(ValueError, match="w1 must have shape"):
        sonic_moe_backward(*args, _config(128, 64, 4, 2, activation="relu"))
    with pytest.raises(ValueError, match="both be None or both be tensors"):
        sonic_moe_backward(*args, _config(128, 64, 4, 2), b1=b1)
    with pytest.raises(ValueError, match="b1 must have shape"):
        sonic_moe_backward(
            *args,
            _config(128, 64, 4, 2),
            b1=b1[:, :-1],
            b2=b2,
        )
    with pytest.raises(TypeError, match="b2 must be bfloat16"):
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
