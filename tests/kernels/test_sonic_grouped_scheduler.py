# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for SonicMoE's device-side compact real-M-tile work builder."""

import math

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.moe.sonic_grouped_scheduler import (
    build_compact_m_tile_descriptors,
    compact_m_tile_descriptor_upper_bound,
    fixed_compact_m_tile_descriptor_upper_bound,
    ragged_compact_m_tile_descriptor_upper_bound,
)


pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_SORTED_BLOCK_M = 64
_CANARY = 0x5A5A5A5A


def _gfx950_device():
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU is required")
    arch = str(get_rocm_arch())
    if "gfx950" not in arch:
        pytest.skip(f"compact SonicMoE scheduler test requires gfx950, found {arch}")
    return torch.device("cuda")


def _metadata_from_frequencies(frequencies, device):
    sorted_experts = []
    padded_rows = 0
    for expert, count in enumerate(frequencies):
        blocks = (count + _SORTED_BLOCK_M - 1) // _SORTED_BLOCK_M
        sorted_experts.extend([expert] * blocks)
        padded_rows += blocks * _SORTED_BLOCK_M
    # A one-element allocation also lets the all-empty case exercise the
    # launcher without relying on a non-null pointer for an empty tensor.
    if not sorted_experts:
        sorted_experts = [-1]
    return (
        torch.tensor(frequencies, dtype=torch.int32, device=device),
        torch.tensor(sorted_experts, dtype=torch.int32, device=device),
        torch.tensor([padded_rows, sum(frequencies)], dtype=torch.int32, device=device),
    )


def _expected_descriptors(frequencies, block_m):
    result = []
    first_row = 0
    for count in frequencies:
        result.extend(range(first_row // block_m, first_row // block_m + math.ceil(count / block_m)))
        first_row += math.ceil(count / _SORTED_BLOCK_M) * _SORTED_BLOCK_M
    return result


def _run_and_check(frequencies, block_m, capacity, *, repeats=3):
    device = _gfx950_device()
    frequency, sorted_experts, num_valid = _metadata_from_frequencies(frequencies, device)
    expected = _expected_descriptors(frequencies, block_m)
    guard = 16
    descriptors = torch.full((max(1, capacity) + guard,), _CANARY, dtype=torch.int32, device=device)
    total = torch.full((1,), -1, dtype=torch.int32, device=device)

    for _ in range(repeats):
        descriptors.fill_(_CANARY)
        total.fill_(-1)
        build_compact_m_tile_descriptors(
            frequency,
            sorted_experts,
            num_valid,
            descriptors,
            total,
            block_m=block_m,
            sorted_block_m=_SORTED_BLOCK_M,
            descriptor_capacity=capacity,
        )
        torch.cuda.synchronize()
        assert total.item() == len(expected)
        written = min(capacity, len(expected))
        if capacity >= len(expected):
            assert sorted(descriptors[:written].cpu().tolist()) == expected
        else:
            # Reservation order is intentionally nondeterministic, so a short
            # output can contain any subset, but every written value must be a
            # valid descriptor and the capacity guard must remain untouched.
            assert set(descriptors[:written].cpu().tolist()).issubset(set(expected))
        assert torch.all(descriptors[capacity:] == _CANARY)
        if expected:
            assert min(expected) >= 0
            assert max(expected) * block_m < num_valid[0].item()


@pytest.mark.parametrize("block_m", (16, 32, 64))
def test_compact_descriptors_cover_frequency_boundaries(block_m):
    frequencies = [0, 1, 15, 16, 17, 63, 64, 65]
    routes = sum(frequencies)
    bound = ragged_compact_m_tile_descriptor_upper_bound(routes, len(frequencies), block_m)
    assert bound >= len(_expected_descriptors(frequencies, block_m))
    _run_and_check(frequencies, block_m, bound)


@pytest.mark.parametrize(
    ("name", "frequencies", "bound_kind", "tokens", "topk"),
    (
        ("fixed_t1_sparse", [1] * 16 + [0] * 880, "fixed", 1, 16),
        ("fixed_t64_hot", [64] * 16 + [0] * 880, "fixed", 64, 16),
        ("fixed_t128_hot", [128] * 16 + [0] * 880, "fixed", 128, 16),
        ("fixed_t128_balanced", [3] * 256 + [2] * 640, "fixed", 128, 16),
        ("ragged_sparse_duplicates", [129, 65, 17, 1] + [0] * 892, "ragged", None, None),
    ),
)
@pytest.mark.parametrize("block_m", (16, 32, 64))
def test_compact_descriptors_cover_fixed_and_ragged_distributions(
    name,
    frequencies,
    bound_kind,
    tokens,
    topk,
    block_m,
):
    del name
    routes = sum(frequencies)
    if bound_kind == "fixed":
        assert routes == tokens * topk
        bound = fixed_compact_m_tile_descriptor_upper_bound(tokens, len(frequencies), topk, block_m)
    else:
        bound = ragged_compact_m_tile_descriptor_upper_bound(routes, len(frequencies), block_m)
    expected = _expected_descriptors(frequencies, block_m)
    assert len(expected) <= bound
    _run_and_check(frequencies, block_m, bound, repeats=2)


def test_compact_descriptor_builder_handles_all_empty_input():
    _run_and_check([0] * 896, 16, 0)


def test_compact_descriptor_builder_accepts_empty_output_allocation():
    device = _gfx950_device()
    frequency = torch.zeros(896, dtype=torch.int32, device=device)
    sorted_experts = torch.empty(0, dtype=torch.int32, device=device)
    num_valid = torch.zeros(2, dtype=torch.int32, device=device)
    descriptors = torch.empty(0, dtype=torch.int32, device=device)
    total = torch.full((1,), -1, dtype=torch.int32, device=device)
    build_compact_m_tile_descriptors(
        frequency,
        sorted_experts,
        num_valid,
        descriptors,
        total,
        block_m=16,
        sorted_block_m=_SORTED_BLOCK_M,
    )
    torch.cuda.synchronize()
    assert total.item() == 0


def test_compact_descriptor_builder_drops_stores_past_capacity():
    frequencies = [0, 1, 15, 16, 17, 63, 64, 65]
    expected = _expected_descriptors(frequencies, 16)
    _run_and_check(frequencies, 16, len(expected) - 1)


@pytest.mark.parametrize(
    ("routes", "experts", "block_m", "max_expert_rows", "expected"),
    (
        (0, 896, 16, None, 0),
        (16, 896, 16, 1, 16),
        (1024, 896, 64, 64, 896),
        (2048, 896, 16, 128, 968),
        (2048, 896, 64, 128, 914),
        (212, 896, 16, None, 212),
    ),
)
def test_compact_descriptor_upper_bound(routes, experts, block_m, max_expert_rows, expected):
    assert (
        compact_m_tile_descriptor_upper_bound(
            routes,
            experts,
            block_m,
            max_expert_rows=max_expert_rows,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        ({"routes": -1, "num_experts": 8, "block_m": 16}, ValueError),
        ({"routes": 1, "num_experts": 0, "block_m": 16}, ValueError),
        ({"routes": 1, "num_experts": 8, "block_m": 0}, ValueError),
        ({"routes": 1, "num_experts": 8, "block_m": 16, "max_expert_rows": -1}, ValueError),
        ({"routes": 9, "num_experts": 8, "block_m": 16, "max_expert_rows": 1}, ValueError),
        ({"routes": 1 << 31, "num_experts": 8, "block_m": 16}, ValueError),
        ({"routes": 1.0, "num_experts": 8, "block_m": 16}, TypeError),
    ),
)
def test_compact_descriptor_upper_bound_rejects_invalid_inputs(kwargs, error):
    with pytest.raises(error):
        compact_m_tile_descriptor_upper_bound(**kwargs)
