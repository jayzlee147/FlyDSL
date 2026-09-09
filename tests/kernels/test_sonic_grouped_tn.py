# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Correctness tests for the gfx950 device-driven grouped TN kernel."""

import inspect
import math

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.moe.sonic_grouped_tn import (
    active_expert_descriptor_capacity,
    active_expert_queue_elements,
    build_active_expert_queue_flydsl,
    compile_grouped_tn,
    grouped_dw2_flydsl,
    grouped_dw2_tuning,
    grouped_tn_from_metadata_flydsl,
    grouped_tn_from_queue_flydsl,
    grouped_tn_grid_cap,
    grouped_tn_launch_grid,
    zero_inactive_weight_grads_flydsl,
    zero_weight_grads_adaptive_flydsl,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]
_SORTED_BLOCK_M = 64


def _gfx950_device():
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU is required")
    arch = str(get_rocm_arch())
    if "gfx950" not in arch:
        pytest.skip(f"grouped dW2 test requires gfx950, found {arch}")
    return torch.device("cuda")


def _make_sorted_inputs(frequencies, hidden_size, intermediate_size, seed):
    device = _gfx950_device()
    generator = torch.Generator(device=device).manual_seed(seed)
    padded_rows = sum(math.ceil(count / _SORTED_BLOCK_M) * _SORTED_BLOCK_M for count in frequencies)
    dy = torch.zeros((padded_rows, hidden_size), dtype=torch.bfloat16, device=device)
    activation = torch.zeros(
        (padded_rows, intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    )
    sorted_experts = []
    segments = []
    start = 0
    for expert, count in enumerate(frequencies):
        if not count:
            continue
        padded = math.ceil(count / _SORTED_BLOCK_M) * _SORTED_BLOCK_M
        dy[start : start + count] = torch.randn(
            (count, hidden_size),
            dtype=torch.float32,
            device=device,
            generator=generator,
        ).to(torch.bfloat16)
        activation[start : start + count] = torch.randn(
            (count, intermediate_size),
            dtype=torch.float32,
            device=device,
            generator=generator,
        ).to(torch.bfloat16)
        sorted_experts.extend([expert] * (padded // _SORTED_BLOCK_M))
        segments.append((expert, start, padded))
        start += padded
    return (
        dy,
        activation,
        torch.tensor(frequencies, dtype=torch.int32, device=device),
        torch.tensor(sorted_experts, dtype=torch.int32, device=device),
        torch.tensor([padded_rows, sum(frequencies)], dtype=torch.int32, device=device),
        segments,
    )


@pytest.mark.parametrize(
    ("frequencies", "hidden_size", "intermediate_size"),
    (
        ([0, 1, 63, 64, 65, 2, 0, 17], 128, 64),
        ([65, 0, 129, 7], 256, 256),
        ([33, 0, 65, 129], 128, 64),
    ),
    ids=("sparse-cross-block", "production-tile", "bk32-boundaries"),
)
def test_grouped_dw2_matches_a16_reference_and_keeps_empty_experts_zero(
    frequencies,
    hidden_size,
    intermediate_size,
):
    dy, activation, frequency, sorted_experts, num_valid, segments = _make_sorted_inputs(
        frequencies,
        hidden_size,
        intermediate_size,
        seed=419,
    )
    output = torch.zeros(
        (len(frequencies), hidden_size, intermediate_size),
        dtype=torch.bfloat16,
        device=dy.device,
    )
    expected = torch.zeros_like(output)
    for expert, start, rows in segments:
        expected[expert] = (
            dy[start : start + rows].float().transpose(0, 1)
            @ activation[start : start + rows].float()
        ).to(torch.bfloat16)

    returned = grouped_dw2_flydsl(
        dy,
        activation,
        frequency,
        sorted_experts,
        num_valid,
        output,
        routes=sum(frequencies),
    )
    torch.cuda.synchronize()

    assert returned is output
    torch.testing.assert_close(output.float(), expected.float(), rtol=3e-2, atol=5e-2)
    for expert, count in enumerate(frequencies):
        if count == 0:
            assert torch.count_nonzero(output[expert]) == 0


def test_grouped_tn_reuses_prebuilt_active_expert_queue():
    frequencies = [0, 33, 0, 65, 129]
    dy, activation, frequency, sorted_experts, num_valid, segments = _make_sorted_inputs(
        frequencies,
        128,
        64,
        seed=421,
    )
    queue = torch.empty(
        active_expert_queue_elements(sum(frequencies), len(frequencies)),
        dtype=torch.int32,
        device=dy.device,
    )
    returned_queue = build_active_expert_queue_flydsl(
        frequency,
        sorted_experts,
        num_valid,
        routes=sum(frequencies),
        queue_storage=queue,
    )
    first = torch.zeros(
        (len(frequencies), 128, 64),
        dtype=torch.bfloat16,
        device=dy.device,
    )
    second = torch.zeros_like(first)
    grouped_tn_from_queue_flydsl(dy, activation, frequency, queue, first)
    grouped_tn_from_queue_flydsl(dy, activation, frequency, queue, second)
    torch.cuda.synchronize()

    assert returned_queue is queue
    live_count = int(queue[0].item())
    descriptors = queue[1 : 1 + 2 * live_count].view(-1, 2).cpu().tolist()
    assert sorted(descriptors) == sorted([expert, start] for expert, start, _ in segments)
    torch.testing.assert_close(first.float(), second.float(), rtol=0, atol=0)


@pytest.mark.parametrize("active_experts", (32, 33))
def test_grouped_tn_active_count_guards_are_mutually_exclusive(active_experts):
    num_experts = 40
    frequencies = [1] * active_experts + [0] * (num_experts - active_experts)
    lhs, rhs, frequency, sorted_experts, num_valid, segments = _make_sorted_inputs(
        frequencies,
        128,
        64,
        seed=521 + active_experts,
    )
    queue = build_active_expert_queue_flydsl(
        frequency,
        sorted_experts,
        num_valid,
        routes=active_experts,
    )
    low_output = torch.zeros(
        (num_experts, 128, 64),
        dtype=torch.bfloat16,
        device=lhs.device,
    )
    high_output = torch.zeros_like(low_output)

    grouped_tn_from_queue_flydsl(
        lhs,
        rhs,
        frequency,
        queue,
        low_output,
        block_m=128,
        block_n=64,
        block_k=32,
        k_padding=0,
        m_waves=2,
        n_waves=2,
        max_active_experts=32,
    )
    grouped_tn_from_queue_flydsl(
        lhs,
        rhs,
        frequency,
        queue,
        high_output,
        block_m=64,
        block_n=64,
        block_k=32,
        k_padding=0,
        m_waves=2,
        n_waves=2,
        min_active_experts=33,
    )
    torch.cuda.synchronize()

    expected = torch.zeros_like(low_output)
    for expert, start, rows in segments:
        expected[expert] = (
            lhs[start : start + rows].float().transpose(0, 1)
            @ rhs[start : start + rows].float()
        ).to(torch.bfloat16)
    selected, rejected = (
        (low_output, high_output)
        if active_experts == 32
        else (high_output, low_output)
    )
    torch.testing.assert_close(selected.float(), expected.float(), rtol=3e-2, atol=5e-2)
    assert torch.count_nonzero(rejected) == 0


def test_grouped_tn_consumes_single_block_metadata_without_builder():
    frequencies = [0, 1, 0, 7, 63]
    dy, activation, frequency, sorted_experts, num_valid, segments = _make_sorted_inputs(
        frequencies,
        128,
        64,
        seed=423,
    )
    output = torch.zeros(
        (len(frequencies), 128, 64),
        dtype=torch.bfloat16,
        device=dy.device,
    )
    expected = torch.zeros_like(output)
    for expert, start, rows in segments:
        expected[expert] = (
            dy[start : start + rows].float().transpose(0, 1)
            @ activation[start : start + rows].float()
        ).to(torch.bfloat16)

    returned = grouped_tn_from_metadata_flydsl(
        dy,
        activation,
        frequency,
        sorted_experts,
        num_valid,
        output,
        block_m=128,
        block_n=64,
        block_k=32,
        k_padding=0,
        m_waves=2,
        n_waves=2,
    )
    torch.cuda.synchronize()

    assert returned is output
    torch.testing.assert_close(output.float(), expected.float(), rtol=3e-2, atol=5e-2)
    for expert, count in enumerate(frequencies):
        if count == 0:
            assert torch.count_nonzero(output[expert]) == 0


def test_grouped_tn_metadata_direct_gathers_t1_rhs_and_zeroes_sentinel_tail():
    device = _gfx950_device()
    generator = torch.Generator(device=device).manual_seed(1423)
    tokens, output_m, output_n = 1, 64, 64
    frequencies = [1, 0, 1]
    padded_rows = 2 * _SORTED_BLOCK_M
    lhs = torch.zeros((padded_rows, output_m), dtype=torch.bfloat16, device=device)
    lhs[0].normal_(generator=generator)
    lhs[_SORTED_BLOCK_M].normal_(generator=generator)
    token_rhs = torch.randn(
        (tokens, output_n),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    sorted_rhs = torch.zeros((padded_rows, output_n), dtype=torch.bfloat16, device=device)
    sorted_rhs[0] = token_rhs[0]
    sorted_rhs[_SORTED_BLOCK_M] = token_rhs[0]
    sorted_token_ids = torch.full(
        (padded_rows,),
        tokens,
        dtype=torch.int32,
        device=device,
    )
    sorted_token_ids[0] = 0
    sorted_token_ids[_SORTED_BLOCK_M] = 1 << 24
    frequency = torch.tensor(frequencies, dtype=torch.int32, device=device)
    sorted_experts = torch.tensor([0, 2], dtype=torch.int32, device=device)
    num_valid = torch.tensor([padded_rows, sum(frequencies)], dtype=torch.int32, device=device)
    materialized = torch.zeros(
        (len(frequencies), output_m, output_n),
        dtype=torch.bfloat16,
        device=device,
    )
    gathered = torch.zeros_like(materialized)
    tuning = {
        "block_m": 64,
        "block_n": 64,
        "block_k": 32,
        "k_padding": 0,
        "m_waves": 2,
        "n_waves": 2,
    }

    grouped_tn_from_metadata_flydsl(
        lhs,
        sorted_rhs,
        frequency,
        sorted_experts,
        num_valid,
        materialized,
        **tuning,
    )
    grouped_tn_from_metadata_flydsl(
        lhs,
        token_rhs,
        frequency,
        sorted_experts,
        num_valid,
        gathered,
        sorted_token_ids=sorted_token_ids,
        **tuning,
    )
    torch.cuda.synchronize()

    assert torch.equal(gathered, materialized)


@pytest.mark.parametrize(
    ("tokens", "frequencies", "block_k", "stages"),
    (
        (128, [33, 0, 65, 30], 32, 2),
        (256, [33, 0, 129, 94], 64, 3),
        (512, [257, 0, 129, 126], 64, 3),
        (512, [257, 0, 129, 126], 64, 4),
    ),
    ids=(
        "bk32-stage2",
        "bk64-stage3-short",
        "bk64-stage3-wrapped",
        "bk64-stage4-wrapped",
    ),
)
def test_grouped_tn_queue_gathers_nonmonotonic_rhs_across_bk_tails(
    tokens,
    frequencies,
    block_k,
    stages,
):
    device = _gfx950_device()
    generator = torch.Generator(device=device).manual_seed(1425)
    output_m, output_n = 128, 128
    padded_counts = [math.ceil(count / _SORTED_BLOCK_M) * _SORTED_BLOCK_M for count in frequencies]
    padded_rows = sum(padded_counts)
    lhs = torch.zeros((padded_rows, output_m), dtype=torch.bfloat16, device=device)
    token_rhs = torch.randn(
        (tokens, output_n),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    sorted_rhs = torch.zeros((padded_rows, output_n), dtype=torch.bfloat16, device=device)
    sorted_token_ids = torch.full(
        (padded_rows,),
        tokens,
        dtype=torch.int32,
        device=device,
    )
    token_order = torch.tensor(
        [value for pair in zip(range(tokens - 1, tokens // 2 - 1, -1), range(tokens // 2)) for value in pair],
        dtype=torch.int32,
        device=device,
    )
    sorted_experts = []
    route_offset = 0
    sorted_offset = 0
    for expert, (count, padded) in enumerate(zip(frequencies, padded_counts)):
        if count == 0:
            continue
        expert_tokens = token_order[route_offset : route_offset + count]
        lhs[sorted_offset : sorted_offset + count] = torch.randn(
            (count, output_m),
            dtype=torch.float32,
            device=device,
            generator=generator,
        ).to(torch.bfloat16)
        sorted_rhs[sorted_offset : sorted_offset + count] = token_rhs[expert_tokens.long()]
        slots = torch.arange(count, dtype=torch.int32, device=device) % 4
        sorted_token_ids[sorted_offset : sorted_offset + count] = expert_tokens | (slots << 24)
        sorted_experts.extend([expert] * (padded // _SORTED_BLOCK_M))
        route_offset += count
        sorted_offset += padded

    frequency = torch.tensor(frequencies, dtype=torch.int32, device=device)
    sorted_experts = torch.tensor(sorted_experts, dtype=torch.int32, device=device)
    num_valid = torch.tensor([padded_rows, tokens], dtype=torch.int32, device=device)
    queue = build_active_expert_queue_flydsl(
        frequency,
        sorted_experts,
        num_valid,
        routes=tokens,
    )
    materialized = torch.zeros(
        (len(frequencies), output_m, output_n),
        dtype=torch.bfloat16,
        device=device,
    )
    gathered = torch.zeros_like(materialized)
    tuning = {
        "block_m": 128,
        "block_n": 128,
        "block_k": block_k,
        "k_padding": 0,
        "m_waves": 2,
        "n_waves": 2,
        "stages": stages,
    }

    grouped_tn_from_queue_flydsl(
        lhs,
        sorted_rhs,
        frequency,
        queue,
        materialized,
        **tuning,
    )
    grouped_tn_from_queue_flydsl(
        lhs,
        token_rhs,
        frequency,
        queue,
        gathered,
        sorted_token_ids=sorted_token_ids,
        **tuning,
    )
    torch.cuda.synchronize()

    assert torch.equal(gathered, materialized)


def test_grouped_tn_gather_rhs_validates_packed_ids():
    device = _gfx950_device()
    lhs = torch.zeros((64, 64), dtype=torch.bfloat16, device=device)
    rhs = torch.zeros((1, 64), dtype=torch.bfloat16, device=device)
    frequency = torch.tensor([1], dtype=torch.int32, device=device)
    sorted_experts = torch.tensor([0], dtype=torch.int32, device=device)
    num_valid = torch.tensor([64, 1], dtype=torch.int32, device=device)
    output = torch.zeros((1, 64, 64), dtype=torch.bfloat16, device=device)

    with pytest.raises(TypeError, match="sorted_token_ids must use int32"):
        grouped_tn_from_metadata_flydsl(
            lhs,
            rhs,
            frequency,
            sorted_experts,
            num_valid,
            output,
            sorted_token_ids=torch.zeros(64, dtype=torch.int64, device=device),
        )
    with pytest.raises(ValueError, match="1D and cover"):
        grouped_tn_from_metadata_flydsl(
            lhs,
            rhs,
            frequency,
            sorted_experts,
            num_valid,
            output,
            sorted_token_ids=torch.zeros((8, 8), dtype=torch.int32, device=device),
        )
    with pytest.raises(ValueError, match="1D and cover"):
        grouped_tn_from_metadata_flydsl(
            lhs,
            rhs,
            frequency,
            sorted_experts,
            num_valid,
            output,
            sorted_token_ids=torch.zeros(63, dtype=torch.int32, device=device),
        )

    queue = torch.zeros(3, dtype=torch.int32, device=device)
    with pytest.raises(TypeError, match="sorted_token_ids must use int32"):
        grouped_tn_from_queue_flydsl(
            lhs,
            rhs,
            frequency,
            queue,
            output,
            sorted_token_ids=torch.zeros(64, dtype=torch.int64, device=device),
        )
    with pytest.raises(ValueError, match="1D and cover"):
        grouped_tn_from_queue_flydsl(
            lhs,
            rhs,
            frequency,
            queue,
            output,
            sorted_token_ids=torch.zeros((8, 8), dtype=torch.int32, device=device),
        )
    with pytest.raises(ValueError, match="1D and cover"):
        grouped_tn_from_queue_flydsl(
            lhs,
            rhs,
            frequency,
            queue,
            output,
            sorted_token_ids=torch.zeros(63, dtype=torch.int32, device=device),
        )


def test_compile_grouped_tn_preserves_sorted_rhs_launcher_abi():
    compile_args = (64, 64, 1, 64, 64, 32, 0, 2, 2, 0)
    sorted_rhs_launcher = compile_grouped_tn(*compile_args)
    gathered_rhs_launcher = compile_grouped_tn(*compile_args, gather_rhs=True)

    assert tuple(inspect.signature(sorted_rhs_launcher.func).parameters) == (
        "lhs_rows",
        "rhs_rows",
        "expert_frequency",
        "schedule_storage",
        "num_valid_ids",
        "output",
        "i32_grid",
        "stream",
    )
    assert tuple(inspect.signature(gathered_rhs_launcher.func).parameters) == (
        "lhs_rows",
        "rhs_rows",
        "sorted_token_ids",
        "expert_frequency",
        "schedule_storage",
        "num_valid_ids",
        "output",
        "i32_grid",
        "stream",
    )


def test_grouped_tn_uses_64_bit_output_base_for_last_production_expert():
    device = _gfx950_device()
    num_experts = 896
    output_m = 1024
    output_n = 3584
    generator = torch.Generator(device=device).manual_seed(425)
    lhs = torch.zeros((64, output_m), dtype=torch.bfloat16, device=device)
    rhs = torch.zeros((64, output_n), dtype=torch.bfloat16, device=device)
    lhs[0].normal_(std=0.1, generator=generator)
    rhs[0].normal_(std=0.1, generator=generator)
    frequency = torch.zeros(num_experts, dtype=torch.int32, device=device)
    frequency[-1] = 1
    sorted_experts = torch.tensor([num_experts - 1], dtype=torch.int32, device=device)
    num_valid = torch.tensor([64, 1], dtype=torch.int32, device=device)
    output = torch.zeros(
        (num_experts, output_m, output_n),
        dtype=torch.bfloat16,
        device=device,
    )

    grouped_tn_from_metadata_flydsl(
        lhs,
        rhs,
        frequency,
        sorted_experts,
        num_valid,
        output,
        block_m=64,
        block_n=64,
        block_k=32,
        k_padding=0,
        m_waves=2,
        n_waves=2,
    )
    torch.cuda.synchronize()

    expected = lhs[:1].float().transpose(0, 1) @ rhs[:1].float()
    torch.testing.assert_close(output[-1].float(), expected, rtol=3e-2, atol=5e-2)
    assert torch.count_nonzero(output[0]) == 0


def test_inactive_weight_grad_zero_preserves_active_expert_slabs():
    device = _gfx950_device()
    frequency = torch.tensor([0, 3, 0, 1], dtype=torch.int32, device=device)
    dw1 = torch.full((4, 128, 64), 7.0, dtype=torch.bfloat16, device=device)
    dw2 = torch.full((4, 64, 64), 9.0, dtype=torch.bfloat16, device=device)

    returned_dw1, returned_dw2 = zero_inactive_weight_grads_flydsl(
        frequency,
        dw1,
        dw2,
    )
    torch.cuda.synchronize()

    assert returned_dw1 is dw1
    assert returned_dw2 is dw2
    for expert in (0, 2):
        assert torch.count_nonzero(dw1[expert]) == 0
        assert torch.count_nonzero(dw2[expert]) == 0
    for expert in (1, 3):
        assert torch.all(dw1[expert] == 7)
        assert torch.all(dw2[expert] == 9)


@pytest.mark.parametrize("active_experts", (1, 2), ids=("dense-all", "inactive-only"))
def test_adaptive_weight_grad_zero_selects_device_count_branch(active_experts):
    device = _gfx950_device()
    frequency = torch.tensor([1, 1, 0, 0], dtype=torch.int32, device=device)
    active_count = torch.tensor([active_experts], dtype=torch.int32, device=device)
    dw1 = torch.full((4, 128, 64), 7.0, dtype=torch.bfloat16, device=device)
    dw2 = torch.full((4, 64, 64), 9.0, dtype=torch.bfloat16, device=device)

    zero_weight_grads_adaptive_flydsl(
        frequency,
        active_count,
        dw1,
        dw2,
        dense_active_ratio=2,
    )
    torch.cuda.synchronize()

    if active_experts == 1:
        assert torch.count_nonzero(dw1) == 0
        assert torch.count_nonzero(dw2) == 0
    else:
        for expert in (0, 1):
            assert torch.all(dw1[expert] == 7)
            assert torch.all(dw2[expert] == 9)
        for expert in (2, 3):
            assert torch.count_nonzero(dw1[expert]) == 0
            assert torch.count_nonzero(dw2[expert]) == 0


def test_inactive_weight_grad_zero_uses_64_bit_last_expert_base():
    device = _gfx950_device()
    num_experts = 896
    # 896 * 1024 * 3584 BF16 elements is about 6.1 GiB.  Only the final
    # expert is inactive, so this also keeps the device-side clear itself
    # small while exercising an expert base well beyond the 32-bit window.
    dw1 = torch.empty(
        (num_experts, 1024, 3584),
        dtype=torch.bfloat16,
        device=device,
    )
    dw2 = torch.empty(
        (num_experts, 64, 64),
        dtype=torch.bfloat16,
        device=device,
    )
    dw1[0].fill_(3.0)
    dw1[-1].fill_(5.0)
    dw2[0].fill_(7.0)
    dw2[-1].fill_(9.0)
    frequency = torch.ones(num_experts, dtype=torch.int32, device=device)
    frequency[-1] = 0

    zero_inactive_weight_grads_flydsl(frequency, dw1, dw2)
    torch.cuda.synchronize()

    assert torch.all(dw1[0] == 3)
    assert torch.count_nonzero(dw1[-1]) == 0
    assert torch.all(dw2[0] == 7)
    assert torch.count_nonzero(dw2[-1]) == 0


def test_grouped_tn_empty_routes_are_an_exact_noop():
    device = _gfx950_device()
    lhs = torch.empty((0, 64), dtype=torch.bfloat16, device=device)
    rhs = torch.empty((0, 64), dtype=torch.bfloat16, device=device)
    frequency = torch.zeros(4, dtype=torch.int32, device=device)
    sorted_experts = torch.empty(0, dtype=torch.int32, device=device)
    num_valid = torch.zeros(2, dtype=torch.int32, device=device)
    output = torch.zeros((4, 64, 64), dtype=torch.bfloat16, device=device)

    returned = grouped_dw2_flydsl(
        lhs,
        rhs,
        frequency,
        sorted_experts,
        num_valid,
        output,
        routes=0,
    )

    assert returned is output
    assert torch.count_nonzero(output) == 0


def test_grouped_dw2_policy_helpers():
    assert active_expert_descriptor_capacity(0, 896) == 0
    assert active_expert_descriptor_capacity(16, 896) == 16
    assert active_expert_descriptor_capacity(65536, 896) == 896
    assert active_expert_queue_elements(16, 896) == 33
    assert grouped_dw2_tuning(3584, 512) == (256, 256, 32, 0, 4, 4)
    assert grouped_dw2_tuning(128, 64) == (128, 64, 32, 0, 2, 2)
    assert grouped_tn_grid_cap(256, 256, 32, 2, 4, 4) == 256
    assert grouped_tn_grid_cap(256, 128, 32, 2, 4, 2) == 512
    assert grouped_tn_grid_cap(128, 256, 32, 2, 2, 4) == 512
    assert grouped_tn_grid_cap(128, 128, 32, 2, 2, 2) == 1024
    assert grouped_tn_grid_cap(128, 128, 32, 3, 2, 2) == 768
    assert grouped_tn_grid_cap(64, 64, 32, 2, 1, 1) == 2560
    assert grouped_tn_grid_cap(64, 64, 32, 2, 1, 1, True) == 2304
    assert grouped_tn_launch_grid(3, 256, 256, 128, 128, 32, 2, 2, 2) == 12
    with pytest.raises(ValueError):
        active_expert_descriptor_capacity(-1, 8)
    with pytest.raises(TypeError):
        active_expert_descriptor_capacity(1.0, 8)
    with pytest.raises(ValueError):
        grouped_dw2_tuning(96, 64)


@pytest.mark.parametrize("stages", (3, 4))
def test_grouped_tn_deep_pipeline_handles_short_and_long_expert_segments(stages):
    frequencies = [1, 33, 65]
    lhs, rhs, frequency, sorted_experts, num_valid, segments = _make_sorted_inputs(
        frequencies,
        128,
        64,
        seed=427 + stages,
    )
    queue = build_active_expert_queue_flydsl(
        frequency,
        sorted_experts,
        num_valid,
        routes=sum(frequencies),
    )
    output = torch.zeros(
        (len(frequencies), 128, 64),
        dtype=torch.bfloat16,
        device=lhs.device,
    )
    expected = torch.zeros_like(output)
    for expert, start, rows in segments:
        expected[expert] = (
            lhs[start : start + rows].float().transpose(0, 1)
            @ rhs[start : start + rows].float()
        ).to(torch.bfloat16)

    grouped_tn_from_queue_flydsl(
        lhs,
        rhs,
        frequency,
        queue,
        output,
        block_m=128,
        block_n=64,
        block_k=32,
        k_padding=0,
        m_waves=2,
        n_waves=2,
        stages=stages,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output.float(), expected.float(), rtol=3e-2, atol=5e-2)
