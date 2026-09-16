# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Correctness tests for SonicMoE's exact device-side M-tile queue."""

import math

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.common.tensor_shim import _run_compiled
from kernels.moe.sonic_grouped_a16w16 import compile_sonic_grouped_a16w16_nn
from kernels.moe.sonic_grouped_scheduler import (
    build_exact_m_tile_queue,
    exact_m_tile_queue_elements,
    exact_m_tile_queue_upper_bound,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_SORTED_BLOCK_M = 64
_CANARY = 0x5A5A5A5A


def _gfx950_device():
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU is required")
    arch = str(get_rocm_arch())
    if "gfx950" not in arch:
        pytest.skip(f"exact SonicMoE tile queue test requires gfx950, found {arch}")
    return torch.device("cuda")


def _metadata_from_frequencies(frequencies, device):
    sorted_experts = []
    padded_rows = 0
    for expert, count in enumerate(frequencies):
        blocks = math.ceil(count / _SORTED_BLOCK_M)
        sorted_experts.extend([expert] * blocks)
        padded_rows += blocks * _SORTED_BLOCK_M
    return (
        torch.tensor(frequencies, dtype=torch.int32, device=device),
        torch.tensor(sorted_experts or [-1], dtype=torch.int32, device=device),
        torch.tensor([padded_rows, sum(frequencies)], dtype=torch.int32, device=device),
    )


def _expected_records(frequencies, block_m):
    records = []
    first_row = 0
    for expert, count in enumerate(frequencies):
        for local_row in range(0, count, block_m):
            records.append((expert, first_row + local_row, min(block_m, count - local_row)))
        first_row += math.ceil(count / _SORTED_BLOCK_M) * _SORTED_BLOCK_M
    return records


@pytest.mark.parametrize(
    "frequencies",
    (
        [0, 1, 65, 129, 257, 0],
        [512] * 16,
        [2049, 7, 0, 64, 191, 1],
    ),
)
def test_exact_queue_supports_bm128_over_sorted_bm64(frequencies):
    device = _gfx950_device()
    block_m = 128
    frequency, sorted_experts, num_valid = _metadata_from_frequencies(frequencies, device)
    expected = _expected_records(frequencies, block_m)
    capacity = exact_m_tile_queue_upper_bound(
        sum(frequencies),
        len(frequencies),
        block_m,
    )
    assert capacity >= len(expected)
    queue = torch.full(
        (1 + 3 * capacity,),
        _CANARY,
        dtype=torch.int32,
        device=device,
    )

    build_exact_m_tile_queue(
        frequency,
        sorted_experts,
        num_valid,
        queue,
        block_m=block_m,
        sorted_block_m=_SORTED_BLOCK_M,
    )
    torch.cuda.synchronize(device)

    assert queue[0].item() == len(expected)
    actual = queue[1 : 1 + 3 * len(expected)].reshape(-1, 3).cpu().tolist()
    assert sorted(map(tuple, actual)) == sorted(expected)


@pytest.mark.parametrize("capacity", (0, 3))
def test_exact_queue_respects_record_capacity(capacity):
    device = _gfx950_device()
    frequencies = [1, 65, 129, 257]
    frequency, sorted_experts, num_valid = _metadata_from_frequencies(frequencies, device)
    expected = _expected_records(frequencies, 128)
    storage_capacity = len(expected) + 4
    queue = torch.full(
        (1 + 3 * storage_capacity,),
        _CANARY,
        dtype=torch.int32,
        device=device,
    )

    build_exact_m_tile_queue(
        frequency,
        sorted_experts,
        num_valid,
        queue,
        block_m=128,
        sorted_block_m=_SORTED_BLOCK_M,
        queue_capacity=capacity,
    )
    torch.cuda.synchronize(device)

    assert queue[0].item() == capacity
    written = queue[1 : 1 + 3 * capacity].reshape(-1, 3).cpu().tolist()
    assert set(map(tuple, written)).issubset(set(expected))
    assert torch.all(queue[1 + 3 * capacity :] == _CANARY)


def test_grouped_nn_exact_queue_consumes_only_stored_records():
    device = _gfx950_device()
    contraction_size, output_size = 128, 256
    frequencies = [129, 129]
    frequency, sorted_experts, num_valid = _metadata_from_frequencies(
        frequencies,
        device,
    )
    padded_rows = int(num_valid[0].item())
    storage_capacity = exact_m_tile_queue_upper_bound(
        sum(frequencies),
        len(frequencies),
        128,
    )
    queue_capacity = 1
    queue = torch.full(
        (1 + 3 * storage_capacity,),
        _CANARY,
        dtype=torch.int32,
        device=device,
    )
    build_exact_m_tile_queue(
        frequency,
        sorted_experts,
        num_valid,
        queue,
        block_m=128,
        sorted_block_m=_SORTED_BLOCK_M,
        queue_capacity=queue_capacity,
    )
    torch.cuda.synchronize(device)
    assert queue[0].item() == queue_capacity
    expert, first_row, valid_rows = queue[1:4].cpu().tolist()

    generator = torch.Generator(device=device).manual_seed(20260917)
    a = torch.full(
        (padded_rows, contraction_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    first_expert_row = 0
    for count in frequencies:
        a[first_expert_row : first_expert_row + count] = torch.randn(
            (count, contraction_size),
            generator=generator,
            dtype=torch.float32,
            device=device,
        ).to(torch.bfloat16)
        first_expert_row += math.ceil(count / _SORTED_BLOCK_M) * _SORTED_BLOCK_M
    weights = torch.randn(
        (len(frequencies), contraction_size, output_size),
        generator=generator,
        dtype=torch.float32,
        device=device,
    ).to(torch.bfloat16)
    output = torch.full(
        (padded_rows, output_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    kernel = compile_sonic_grouped_a16w16_nn(
        contraction_size=contraction_size,
        output_size=output_size,
        num_experts=len(frequencies),
        block_m=128,
        block_n=256,
        block_k=64,
        stages=2,
        n_waves=4,
        sorted_block_m=_SORTED_BLOCK_M,
        compact_grid=True,
        exact_tile_queue=True,
        device_index=device.index or 0,
    )
    _run_compiled(
        kernel,
        a.data_ptr(),
        weights.data_ptr(),
        queue.data_ptr(),
        sorted_experts.data_ptr(),
        num_valid.data_ptr(),
        output.data_ptr(),
        storage_capacity,
        torch.cuda.current_stream(device),
    )
    torch.cuda.synchronize(device)

    expected = a[first_row : first_row + valid_rows].float() @ weights[expert].float()
    torch.testing.assert_close(
        output[first_row : first_row + valid_rows].float(),
        expected,
        rtol=3e-2,
        atol=5e-2,
    )
    written_mask = torch.zeros(padded_rows, dtype=torch.bool, device=device)
    written_mask[first_row : first_row + valid_rows] = True
    assert torch.isnan(output[~written_mask]).all()


def test_exact_queue_capacity_and_storage_units_are_explicit():
    capacity = exact_m_tile_queue_upper_bound(8192, 16, 128)
    assert capacity == 79
    assert exact_m_tile_queue_elements(8192, 16, 128) == 238


def test_exact_queue_builder_also_emits_active_expert_queue():
    device = _gfx950_device()
    frequencies = [0, 1, 65, 0, 129, 257]
    frequency, sorted_experts, num_valid = _metadata_from_frequencies(frequencies, device)
    capacity = exact_m_tile_queue_upper_bound(sum(frequencies), len(frequencies), 128)
    queue = torch.empty((1 + 3 * capacity,), dtype=torch.int32, device=device)
    active_capacity = min(sum(frequencies), len(frequencies))
    active_queue = torch.full(
        (1 + 2 * active_capacity,),
        -1,
        dtype=torch.int32,
        device=device,
    )

    build_exact_m_tile_queue(
        frequency,
        sorted_experts,
        num_valid,
        queue,
        block_m=128,
        sorted_block_m=_SORTED_BLOCK_M,
        active_expert_storage=active_queue,
        active_expert_capacity=active_capacity,
    )
    torch.cuda.synchronize(device)

    expected_records = _expected_records(frequencies, 128)
    assert queue[0].item() == len(expected_records)
    actual_records = queue[1 : 1 + 3 * len(expected_records)].reshape(-1, 3).cpu().tolist()
    assert sorted(map(tuple, actual_records)) == sorted(expected_records)

    expected_active = []
    first_row = 0
    for expert, count in enumerate(frequencies):
        if count:
            expected_active.append((expert, first_row))
        first_row += math.ceil(count / _SORTED_BLOCK_M) * _SORTED_BLOCK_M
    assert active_queue[0].item() == len(expected_active)
    actual_active = active_queue[1 : 1 + 2 * len(expected_active)].reshape(-1, 2)
    assert sorted(map(tuple, actual_active.cpu().tolist())) == expected_active


def test_exact_queue_clamps_short_active_expert_capacity():
    device = _gfx950_device()
    frequencies = [1, 65, 0, 129, 257]
    frequency, sorted_experts, num_valid = _metadata_from_frequencies(
        frequencies,
        device,
    )
    queue_capacity = exact_m_tile_queue_upper_bound(
        sum(frequencies),
        len(frequencies),
        128,
    )
    queue = torch.empty((1 + 3 * queue_capacity,), dtype=torch.int32, device=device)
    active_capacity = 2
    active_storage_capacity = 4
    active_queue = torch.full(
        (1 + 2 * active_storage_capacity,),
        _CANARY,
        dtype=torch.int32,
        device=device,
    )

    build_exact_m_tile_queue(
        frequency,
        sorted_experts,
        num_valid,
        queue,
        block_m=128,
        sorted_block_m=_SORTED_BLOCK_M,
        active_expert_storage=active_queue,
        active_expert_capacity=active_capacity,
    )
    torch.cuda.synchronize(device)

    assert active_queue[0].item() == active_capacity
    expected_experts = {expert for expert, count in enumerate(frequencies) if count}
    written = active_queue[1 : 1 + 2 * active_capacity].reshape(-1, 2).cpu().tolist()
    assert {expert for expert, _ in written}.issubset(expected_experts)
    assert torch.all(active_queue[1 + 2 * active_capacity :] == _CANARY)


def test_exact_queue_handles_empty_multi_block_expert_scan():
    device = _gfx950_device()
    frequency, sorted_experts, num_valid = _metadata_from_frequencies([0] * 896, device)
    queue = torch.full((1,), -1, dtype=torch.int32, device=device)

    build_exact_m_tile_queue(
        frequency,
        sorted_experts,
        num_valid,
        queue,
        block_m=128,
        sorted_block_m=_SORTED_BLOCK_M,
    )
    torch.cuda.synchronize(device)

    assert queue.item() == 0


def test_exact_queue_clamps_capacity_after_multi_block_expert_scan():
    device = _gfx950_device()
    frequencies = [1] * 257
    frequency, sorted_experts, num_valid = _metadata_from_frequencies(
        frequencies,
        device,
    )
    capacity = 3
    storage_capacity = 8
    queue = torch.full(
        (1 + 3 * storage_capacity,),
        _CANARY,
        dtype=torch.int32,
        device=device,
    )

    build_exact_m_tile_queue(
        frequency,
        sorted_experts,
        num_valid,
        queue,
        block_m=128,
        sorted_block_m=_SORTED_BLOCK_M,
        queue_capacity=capacity,
    )
    torch.cuda.synchronize(device)

    assert queue[0].item() == capacity
    written = queue[1 : 1 + 3 * capacity].reshape(-1, 3).cpu().tolist()
    assert all(expert in range(len(frequencies)) for expert, _, _ in written)
    assert all(valid_rows == 1 for _, _, valid_rows in written)
    assert torch.all(queue[1 + 3 * capacity :] == _CANARY)


def test_grouped_nn_exact_queue_is_correct_for_ragged_bm128():
    device = _gfx950_device()
    contraction_size, output_size = 128, 256
    frequencies = [1, 65, 0, 129]
    num_experts = len(frequencies)
    block_m = 128
    frequency, sorted_experts, num_valid = _metadata_from_frequencies(frequencies, device)
    padded_rows = int(num_valid[0].item())
    capacity = exact_m_tile_queue_upper_bound(
        sum(frequencies),
        num_experts,
        block_m,
    )
    queue = torch.empty((1 + 3 * capacity,), dtype=torch.int32, device=device)
    build_exact_m_tile_queue(
        frequency,
        sorted_experts,
        num_valid,
        queue,
        block_m=block_m,
        sorted_block_m=_SORTED_BLOCK_M,
    )

    generator = torch.Generator(device=device).manual_seed(20260915)
    a = torch.full(
        (padded_rows, contraction_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    weights = torch.randn(
        (num_experts, contraction_size, output_size),
        generator=generator,
        dtype=torch.float32,
        device=device,
    ).to(torch.bfloat16)
    output = torch.full(
        (padded_rows, output_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )

    expected_slices = []
    first_row = 0
    for expert, count in enumerate(frequencies):
        if count:
            values = torch.randn(
                (count, contraction_size),
                generator=generator,
                dtype=torch.float32,
                device=device,
            ).to(torch.bfloat16)
            a[first_row : first_row + count] = values
            expected_slices.append(
                (
                    first_row,
                    count,
                    values.float() @ weights[expert].float(),
                )
            )
        first_row += math.ceil(count / _SORTED_BLOCK_M) * _SORTED_BLOCK_M

    kernel = compile_sonic_grouped_a16w16_nn(
        contraction_size=contraction_size,
        output_size=output_size,
        num_experts=num_experts,
        block_m=block_m,
        block_n=256,
        block_k=64,
        stages=2,
        n_waves=4,
        sorted_block_m=_SORTED_BLOCK_M,
        compact_grid=True,
        exact_tile_queue=True,
        device_index=device.index or 0,
    )
    _run_compiled(
        kernel,
        a.data_ptr(),
        weights.data_ptr(),
        queue.data_ptr(),
        sorted_experts.data_ptr(),
        num_valid.data_ptr(),
        output.data_ptr(),
        capacity,
        torch.cuda.current_stream(device),
    )
    torch.cuda.synchronize(device)

    real_row_mask = torch.zeros(padded_rows, dtype=torch.bool, device=device)
    for first_row, count, expected in expected_slices:
        torch.testing.assert_close(
            output[first_row : first_row + count].float(),
            expected,
            rtol=3e-2,
            atol=5e-2,
        )
        real_row_mask[first_row : first_row + count] = True
    assert torch.isnan(output[~real_row_mask]).all()


def test_grouped_nn_exact_queue_route_slots_safe_for_129_rows_padded_to_192():
    device = _gfx950_device()
    tokens = 129
    contraction_size, output_size = 128, 256
    frequency, sorted_experts, num_valid = _metadata_from_frequencies(
        [tokens],
        device,
    )
    padded_rows = int(num_valid[0].item())
    assert padded_rows == 192
    capacity = exact_m_tile_queue_upper_bound(tokens, 1, 128)
    queue = torch.empty((1 + 3 * capacity,), dtype=torch.int32, device=device)
    build_exact_m_tile_queue(
        frequency,
        sorted_experts,
        num_valid,
        queue,
        block_m=128,
        sorted_block_m=_SORTED_BLOCK_M,
    )

    generator = torch.Generator(device=device).manual_seed(20260916)
    a = torch.full(
        (padded_rows, contraction_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    a[:tokens] = torch.randn(
        (tokens, contraction_size),
        generator=generator,
        dtype=torch.float32,
        device=device,
    ).to(torch.bfloat16)
    weights = torch.randn(
        (1, contraction_size, output_size),
        generator=generator,
        dtype=torch.float32,
        device=device,
    ).to(torch.bfloat16)
    sorted_token_ids = torch.full(
        (padded_rows,),
        tokens,
        dtype=torch.int32,
        device=device,
    )
    sorted_token_ids[:tokens] = torch.arange(
        tokens,
        dtype=torch.int32,
        device=device,
    )
    output = torch.full(
        (tokens, output_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )

    kernel = compile_sonic_grouped_a16w16_nn(
        contraction_size=contraction_size,
        output_size=output_size,
        num_experts=1,
        block_m=128,
        block_n=256,
        block_k=64,
        stages=2,
        n_waves=4,
        sorted_block_m=_SORTED_BLOCK_M,
        compact_grid=True,
        store_route_slots=True,
        top_k=1,
        exact_tile_queue=True,
        device_index=device.index or 0,
    )
    _run_compiled(
        kernel,
        a.data_ptr(),
        weights.data_ptr(),
        queue.data_ptr(),
        sorted_experts.data_ptr(),
        num_valid.data_ptr(),
        output.data_ptr(),
        sorted_token_ids.data_ptr(),
        tokens,
        capacity,
        torch.cuda.current_stream(device),
    )
    torch.cuda.synchronize(device)

    expected = a[:tokens].float() @ weights[0].float()
    torch.testing.assert_close(
        output.float(),
        expected,
        rtol=3e-2,
        atol=5e-2,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"exact_tile_queue": 1}, "exact_tile_queue"),
        (
            {"exact_tile_queue": True, "compact_grid": False},
            "requires compact_grid",
        ),
    ),
)
def test_exact_queue_mode_validates(overrides, message):
    kwargs = {
        "contraction_size": 128,
        "output_size": 256,
        "num_experts": 4,
        "block_m": 128,
        "block_n": 256,
        "block_k": 64,
        "n_waves": 4,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        compile_sonic_grouped_a16w16_nn(**kwargs)
