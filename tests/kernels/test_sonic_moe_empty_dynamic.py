# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""End-to-end empty-rank coverage for dynamic E16 expert-major training."""

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.moe.sonic import (
    SonicMoE,
    SonicMoEConfig,
    prepare_sonic_bf16_weights,
    sonic_moe_backward_routes,
)
from kernels.moe.sonic_backward import _validate_routes_forward_state


pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


def _gfx950_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU is required")
    arch = str(get_rocm_arch())
    if "gfx950" not in arch:
        pytest.skip(f"dynamic E16 empty-route test requires gfx950, found {arch}")
    return torch.device("cuda")


def _qwen3_e16_config() -> SonicMoEConfig:
    return SonicMoEConfig(
        hidden_size=2048,
        intermediate_size=768,
        num_experts=16,
        top_k=1,
        tile_m=128,
        tile_n=192,
        tile_k=64,
        down_tile_m=64,
        down_tile_n=256,
        down_tile_k=64,
        stage1_xcd_swizzle=8,
        stage2_xcd_swizzle=0,
        stage2_pipeline_stages=2,
        stage1_write_padded_rows=True,
        stage1_lds_swizzle=True,
        renormalize=False,
    )


def test_e16_expert_major_empty_training_state_round_trips_backward():
    """A rank with T=R=0 keeps the normal retained-state API contract."""

    device = _gfx950_device()
    config = _qwen3_e16_config()
    hidden = torch.empty(
        (0, config.hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    w1 = torch.empty(
        (
            config.num_experts,
            2 * config.intermediate_size,
            config.hidden_size,
        ),
        dtype=torch.bfloat16,
        device=device,
    )
    w2 = torch.empty(
        (
            config.num_experts,
            config.hidden_size,
            config.intermediate_size,
        ),
        dtype=torch.bfloat16,
        device=device,
    )
    empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
    empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
    expert_offsets = torch.zeros(
        config.num_experts + 1,
        dtype=torch.int32,
        device=device,
    )
    frequency = torch.full(
        (config.num_experts,),
        -1,
        dtype=torch.int32,
        device=device,
    )
    operator = SonicMoE(
        config,
        prepare_sonic_bf16_weights(w1, w2, config),
    )

    output, state = operator.forward_routes_training(
        hidden,
        empty_i32,
        empty_i32,
        empty_f32,
        expert_frequency_out=frequency,
        expert_offsets=expert_offsets,
        token_indices_identity=True,
        route_policy_size=65536,
    )
    validated = _validate_routes_forward_state(
        state,
        hidden,
        0,
        config,
        False,
        False,
    )
    gradients = sonic_moe_backward_routes(
        hidden,
        w1,
        w2,
        empty_i32,
        empty_i32,
        empty_f32,
        torch.empty_like(output),
        config,
        forward_state=state,
        token_indices_sorted=True,
    )
    torch.cuda.synchronize(device)

    assert tuple(output.shape) == (0, config.hidden_size)
    assert output.numel() == 0
    assert state.tokens == state.routes == 0
    assert tuple(state.preactivation.shape) == (0, 2 * config.intermediate_size)
    assert state.expert_major
    assert state.token_indices_identity
    assert state.route_policy_size == 65536
    assert validated[0] is state.preactivation
    assert validated[1] == state.producer_stream
    assert validated[2] is state.ready_event
    assert validated[3] is not None
    assert validated[4]
    assert validated[5] == 65536
    assert tuple(state.sorted_token_ids.shape) == (0,)
    assert tuple(state.sorted_route_ids.shape) == (0,)
    assert tuple(state.sorted_weights.shape) == (0,)
    assert tuple(state.sorted_expert_ids.shape) == (0,)
    assert torch.count_nonzero(frequency) == 0

    dx, dw1, dw2, droute_weights = gradients
    assert tuple(dx.shape) == (0, config.hidden_size)
    assert tuple(dw1.shape) == tuple(w1.shape)
    assert tuple(dw2.shape) == tuple(w2.shape)
    assert tuple(droute_weights.shape) == (0,)
    assert dx.dtype == dw1.dtype == dw2.dtype == torch.bfloat16
    assert droute_weights.dtype == torch.float32
    assert dx.numel() == 0
    assert droute_weights.numel() == 0
    assert torch.count_nonzero(dw1) == 0
    assert torch.count_nonzero(dw2) == 0
