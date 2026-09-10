# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Correctness tests for SonicMoE's gfx950 grouped A16 NN kernel."""

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.common.tensor_shim import _run_compiled
from kernels.moe.grouped_da_gfx950 import compile_grouped_da_gfx950
from kernels.moe.sonic_grouped_a16w16 import compile_sonic_grouped_a16w16_nn

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


def _gfx950_device():
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU is required")
    arch = str(get_rocm_arch())
    if "gfx950" not in arch:
        pytest.skip(f"grouped A16 test requires gfx950, found {arch}")
    return torch.device("cuda")


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"store_route_slots": 1}, "store_route_slots"),
        ({"expert_m_reuse": 1}, "expert_m_reuse"),
        ({"expert_m_reuse_threshold": True}, "expert_m_reuse_threshold"),
        ({"expert_m_reuse_threshold": 0}, "expert_m_reuse_threshold"),
        ({"expert_m_reuse": True}, "requires store_route_slots"),
        ({"expert_m_reuse_threshold": 3}, "requires store_route_slots"),
        (
            {
                "expert_m_reuse": True,
                "expert_m_reuse_threshold": 3,
                "store_route_slots": True,
            },
            "mutually exclusive",
        ),
        (
            {
                "expert_m_reuse": True,
                "store_route_slots": True,
                "compact_grid": False,
            },
            "requires compact_grid",
        ),
        ({"store_route_slots": True, "top_k": 0}, "top_k"),
        ({"store_route_slots": True, "top_k": 257}, "top_k"),
        (
            {
                "store_route_slots": True,
                "output_size": 96,
                "block_n": 96,
                "n_waves": 2,
            },
            "route-slot output vectors",
        ),
    ),
)
def test_grouped_dx_route_slot_options_validate(overrides, message):
    kwargs = {
        "contraction_size": 128,
        "output_size": 128,
        "num_experts": 3,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        compile_sonic_grouped_a16w16_nn(**kwargs)


@pytest.mark.parametrize(
    ("block_n", "n_waves"),
    ((64, 2), (128, 2), (256, 4)),
)
def test_compact_final_tile_writes_zero_dz_padding(block_n, n_waves):
    """A partial real-M tile is safe because backward produces zero padded dZ."""

    device = _gfx950_device()
    contraction_size, output_size, num_experts = 128, 256, 3
    rows = 128
    real_rows = 65
    generator = torch.Generator(device=device).manual_seed(443)

    # Expert 2 has 65 real rows, rounded to five BM16 descriptors.  Backward's
    # gather/activation chain makes the other 15 rows in the final tile zero.
    dz = torch.zeros((rows, contraction_size), dtype=torch.bfloat16, device=device)
    dz[:real_rows] = torch.randn(
        (real_rows, contraction_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    w1 = torch.randn(
        (num_experts, contraction_size, output_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    dx_sorted = torch.full(
        (rows, output_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    schedule = torch.tensor([5, 0, 1, 2, 3, 4], dtype=torch.int32, device=device)
    sorted_expert_ids = torch.tensor([2, 2], dtype=torch.int32, device=device)
    num_valid_ids = torch.tensor([rows, real_rows], dtype=torch.int32, device=device)

    grouped_dx = compile_sonic_grouped_a16w16_nn(
        contraction_size=contraction_size,
        output_size=output_size,
        num_experts=num_experts,
        block_m=16,
        block_n=block_n,
        block_k=64,
        stages=2,
        n_waves=n_waves,
        compact_grid=True,
        device_index=device.index or 0,
    )
    _run_compiled(
        grouped_dx,
        dz.data_ptr(),
        w1.data_ptr(),
        schedule.data_ptr(),
        sorted_expert_ids.data_ptr(),
        num_valid_ids.data_ptr(),
        dx_sorted.data_ptr(),
        3,  # Force multiple persistent grid-stride iterations.
        torch.cuda.current_stream(device),
    )
    torch.cuda.synchronize(device)

    expected = dz[:real_rows].float() @ w1[2].float()
    torch.testing.assert_close(
        dx_sorted[:real_rows].float(),
        expected,
        rtol=3e-2,
        atol=5e-2,
    )
    assert torch.count_nonzero(dx_sorted[real_rows:80]) == 0
    assert torch.isnan(dx_sorted[80:]).all()


@pytest.mark.parametrize(
    "active_count",
    (2, 3),
)
def test_grouped_dx_single_launch_device_dispatch_is_bitwise(active_count):
    """One kernel selects descriptor-major or M-reuse from device metadata."""

    device = _gfx950_device()
    contraction_size, output_size, num_experts = 128, 512, 3
    frequencies = (129, 0, 65) if active_count == 2 else (129, 1, 65)
    first_rows = (0, 192, 192 if active_count == 2 else 256)
    padded_rows = 320 if active_count == 2 else 384
    tokens = sum(frequencies)
    generator = torch.Generator(device=device).manual_seed(20260910 + active_count)

    dz = torch.zeros((padded_rows, contraction_size), dtype=torch.bfloat16, device=device)
    sorted_token_ids = torch.full((padded_rows,), tokens, dtype=torch.int32, device=device)
    descriptor_values = []
    sorted_expert_values = []
    active_values = [active_count]
    route_row = 0
    for expert, (first_row, frequency) in enumerate(zip(first_rows, frequencies, strict=True)):
        if frequency == 0:
            continue
        dz[first_row : first_row + frequency] = torch.randn(
            (frequency, contraction_size),
            dtype=torch.float32,
            device=device,
            generator=generator,
        ).to(torch.bfloat16)
        sorted_token_ids[first_row : first_row + frequency] = torch.arange(
            route_row,
            route_row + frequency,
            dtype=torch.int32,
            device=device,
        )
        first_m_block = first_row // 64
        num_m_blocks = (frequency + 63) // 64
        descriptor_values.extend(first_m_block + tile for tile in range(num_m_blocks))
        sorted_expert_values.extend([expert] * num_m_blocks)
        active_values.extend((expert, first_row))
        route_row += frequency

    w1 = torch.randn(
        (num_experts, contraction_size, output_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    frequency = torch.tensor(frequencies, dtype=torch.int32, device=device)
    descriptor_schedule = torch.tensor(
        [len(descriptor_values), *descriptor_values],
        dtype=torch.int32,
        device=device,
    )
    sorted_expert_ids = torch.tensor(sorted_expert_values, dtype=torch.int32, device=device)
    active_queue = torch.tensor(active_values, dtype=torch.int32, device=device)
    reference_output = torch.full(
        (tokens + 1, output_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    dispatched_output = torch.full_like(reference_output, float("nan"))

    common = {
        "contraction_size": contraction_size,
        "output_size": output_size,
        "num_experts": num_experts,
        "block_m": 64,
        "block_n": 256,
        "block_k": 64,
        "stages": 2,
        "n_waves": 4,
        "sorted_block_m": 64,
        "compact_grid": True,
        "device_index": device.index or 0,
        "store_route_slots": True,
        "top_k": 1,
    }
    reference_kernel = compile_sonic_grouped_a16w16_nn(**common)
    dispatched_kernel = compile_sonic_grouped_a16w16_nn(
        **common,
        expert_m_reuse_threshold=3,
    )
    stream = torch.cuda.current_stream(device)
    _run_compiled(
        reference_kernel,
        dz.data_ptr(),
        w1.data_ptr(),
        descriptor_schedule.data_ptr(),
        sorted_expert_ids.data_ptr(),
        active_queue.data_ptr(),
        reference_output.data_ptr(),
        sorted_token_ids.data_ptr(),
        tokens,
        5,
        stream,
    )
    _run_compiled(
        dispatched_kernel,
        dz.data_ptr(),
        w1.data_ptr(),
        descriptor_schedule.data_ptr(),
        sorted_expert_ids.data_ptr(),
        active_queue.data_ptr(),
        frequency.data_ptr(),
        dispatched_output.data_ptr(),
        sorted_token_ids.data_ptr(),
        tokens,
        5,
        stream,
    )
    torch.cuda.synchronize(device)

    expected_chunks = []
    for expert, (first_row, expert_rows) in enumerate(zip(first_rows, frequencies, strict=True)):
        if expert_rows:
            expected_chunks.append(
                dz[first_row : first_row + expert_rows].float() @ w1[expert].float()
            )
    expected = torch.cat(expected_chunks)
    assert torch.equal(dispatched_output[:tokens], reference_output[:tokens])
    assert torch.isnan(reference_output[tokens]).all()
    assert torch.isnan(dispatched_output[tokens]).all()
    torch.testing.assert_close(
        dispatched_output[:tokens].float(),
        expected,
        rtol=3e-2,
        atol=5e-2,
    )


@pytest.mark.parametrize("active_experts", (32, 33))
def test_grouped_dx_active_count_guards_are_mutually_exclusive(active_experts):
    device = _gfx950_device()
    contraction_size, output_size, num_experts = 128, 256, 40
    generator = torch.Generator(device=device).manual_seed(541 + active_experts)
    dz = torch.randn(
        (16, contraction_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    w1 = torch.randn(
        (num_experts, contraction_size, output_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    schedule = torch.tensor([1, 0], dtype=torch.int32, device=device)
    sorted_expert_ids = torch.tensor([0], dtype=torch.int32, device=device)
    active_count = torch.tensor([active_experts], dtype=torch.int32, device=device)
    low_output = torch.full(
        (16, output_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    high_output = torch.full_like(low_output, float("nan"))

    profiles = (
        (low_output, 128, 2, 0, 32),
        (high_output, 256, 4, 33, None),
    )
    for output, block_n, n_waves, min_active, max_active in profiles:
        grouped_dx = compile_sonic_grouped_a16w16_nn(
            contraction_size=contraction_size,
            output_size=output_size,
            num_experts=num_experts,
            block_m=16,
            block_n=block_n,
            block_k=64,
            stages=2,
            n_waves=n_waves,
            compact_grid=True,
            device_index=device.index or 0,
            min_active_experts=min_active,
            max_active_experts=max_active,
        )
        _run_compiled(
            grouped_dx,
            dz.data_ptr(),
            w1.data_ptr(),
            schedule.data_ptr(),
            sorted_expert_ids.data_ptr(),
            active_count.data_ptr(),
            output.data_ptr(),
            output_size // block_n,
            torch.cuda.current_stream(device),
        )
    torch.cuda.synchronize(device)

    expected = dz.float() @ w1[0].float()
    selected, rejected = (
        (low_output, high_output)
        if active_experts == 32
        else (high_output, low_output)
    )
    torch.testing.assert_close(selected.float(), expected, rtol=3e-2, atol=5e-2)
    assert torch.isnan(rejected).all()


@pytest.mark.parametrize("compact_grid", (False, True), ids=("metadata", "compact"))
def test_grouped_dx_route_slot_epilogue_matches_sorted_output_bitwise(compact_grid):
    """The fused fixed-K permutation preserves the grouped GEMM's BF16 result."""

    device = _gfx950_device()
    contraction_size, output_size, num_experts, top_k = 128, 128, 3, 2
    generator = torch.Generator(device=device).manual_seed(601 + compact_grid)

    if compact_grid:
        tokens = 5
        rows = 16
        real_rows = tuple(range(tokens * top_k))
        route_rows = (7, 0, 9, 2, 5, 1, 8, 4, 6, 3)
        schedule = torch.tensor([1, 0], dtype=torch.int32, device=device)
        sorted_expert_ids = torch.tensor([1], dtype=torch.int32, device=device)
        # Guarded profiles receive the active-count queue here, not
        # num_valid_ids.  The direct epilogue must use its explicit token bound.
        cumsum = torch.tensor([1], dtype=torch.int32, device=device)
        profile = {"max_active_experts": 32}
    else:
        tokens = 1
        rows = 128
        real_rows = (0, 64)
        route_rows = (1, 0)
        schedule = torch.zeros(1, dtype=torch.int32, device=device)
        sorted_expert_ids = torch.tensor([0, 2], dtype=torch.int32, device=device)
        cumsum = torch.tensor([rows, tokens], dtype=torch.int32, device=device)
        profile = {}

    dz = torch.zeros((rows, contraction_size), dtype=torch.bfloat16, device=device)
    dz[list(real_rows)] = torch.randn(
        (len(real_rows), contraction_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    w1 = torch.randn(
        (num_experts, contraction_size, output_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    sorted_token_ids = torch.full((rows,), tokens, dtype=torch.int32, device=device)
    for sorted_row, route_row in zip(real_rows, route_rows):
        token, slot = divmod(route_row, top_k)
        sorted_token_ids[sorted_row] = token | (slot << 24)

    sorted_output = torch.full(
        (rows, output_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    # One extra route row catches an accidentally unmasked padding write.
    route_storage = torch.full(
        (tokens * top_k + 1, output_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    common = {
        "contraction_size": contraction_size,
        "output_size": output_size,
        "num_experts": num_experts,
        "block_m": 16,
        "block_n": 64,
        "block_k": 64,
        "stages": 2,
        "n_waves": 2,
        "compact_grid": compact_grid,
        "device_index": device.index or 0,
        **profile,
    }
    sorted_kernel = compile_sonic_grouped_a16w16_nn(**common)
    route_kernel = compile_sonic_grouped_a16w16_nn(
        **common,
        store_route_slots=True,
        top_k=top_k,
    )
    grid = 2
    stream = torch.cuda.current_stream(device)
    common_args = (
        dz.data_ptr(),
        w1.data_ptr(),
        schedule.data_ptr(),
        sorted_expert_ids.data_ptr(),
        cumsum.data_ptr(),
    )
    _run_compiled(
        sorted_kernel,
        *common_args,
        sorted_output.data_ptr(),
        grid,
        stream,
    )
    _run_compiled(
        route_kernel,
        *common_args,
        route_storage.data_ptr(),
        sorted_token_ids.data_ptr(),
        tokens,
        grid,
        stream,
    )
    torch.cuda.synchronize(device)

    expected_routes = torch.empty_like(route_storage[:-1])
    for sorted_row, route_row in zip(real_rows, route_rows):
        expected_routes[route_row] = sorted_output[sorted_row]
    assert torch.equal(route_storage[:-1], expected_routes)
    assert torch.isnan(route_storage[-1]).all()


@pytest.mark.parametrize("active_experts", (32, 33))
def test_grouped_da_active_count_guards_are_mutually_exclusive(active_experts):
    device = _gfx950_device()
    hidden_size, intermediate_size, num_experts = 128, 64, 40
    generator = torch.Generator(device=device).manual_seed(557 + active_experts)
    padded_rows = active_experts * 64
    dy = torch.zeros((padded_rows, hidden_size), dtype=torch.bfloat16, device=device)
    dy[::64] = torch.randn(
        (active_experts, hidden_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    w2 = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    frequency = torch.zeros(num_experts, dtype=torch.int32, device=device)
    frequency[:active_experts] = 1
    sorted_expert_ids = torch.arange(active_experts, dtype=torch.int32, device=device)
    num_valid_ids = torch.tensor(
        [padded_rows, active_experts],
        dtype=torch.int32,
        device=device,
    )
    queue_entries = [active_experts]
    for expert in range(active_experts):
        queue_entries.extend((expert, expert * 64))
    active_queue = torch.tensor(queue_entries, dtype=torch.int32, device=device)
    low_output = torch.full(
        (padded_rows, intermediate_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    high_output = torch.full_like(low_output, float("nan"))
    default_output = torch.full_like(low_output, float("nan"))

    profiles = (
        (low_output, 64, True, 0, 32, 2, False, active_experts),
        (high_output, 32, False, 33, None, 3, True, 1),
        (default_output, 32, False, 0, None, 2, False, num_experts),
    )
    for output, block_m, queue_direct, min_active, max_active, stages, persistent, grid in profiles:
        grouped_da = compile_grouped_da_gfx950(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            sorted_block_size=64,
            block_m=block_m,
            block_n=64,
            block_k=64,
            stages=stages,
            m_waves=2,
            n_waves=2,
            queue_direct=queue_direct,
            persistent=persistent,
            min_active_experts=min_active,
            max_active_experts=max_active,
        )
        _run_compiled(
            grouped_da,
            dy.data_ptr(),
            w2.data_ptr(),
            frequency.data_ptr(),
            sorted_expert_ids.data_ptr(),
            num_valid_ids.data_ptr(),
            active_queue.data_ptr(),
            output.data_ptr(),
            grid,
            torch.cuda.current_stream(device),
        )
    torch.cuda.synchronize(device)

    expected = torch.stack(
        [dy[expert * 64].float() @ w2[expert].float() for expert in range(active_experts)]
    )
    selected, rejected = (
        (low_output, high_output)
        if active_experts == 32
        else (high_output, low_output)
    )
    torch.testing.assert_close(selected[::64].float(), expected, rtol=3e-2, atol=5e-2)
    torch.testing.assert_close(default_output[::64].float(), expected, rtol=3e-2, atol=5e-2)
    assert torch.isnan(rejected).all()
    live_rows = torch.zeros(padded_rows, dtype=torch.bool, device=device)
    live_rows[::64] = True
    assert torch.isnan(selected[~live_rows]).all()
    assert torch.isnan(default_output[~live_rows]).all()


@pytest.mark.parametrize(
    ("block_m", "m_waves", "n_waves"),
    ((128, 4, 2), (256, 8, 1)),
)
@pytest.mark.parametrize("queue_direct", (False, True))
@pytest.mark.parametrize("stages", (2, 3))
def test_grouped_da_large_m_tile_handles_sort_blocks_and_frequency_tails(
    block_m,
    m_waves,
    n_waves,
    queue_direct,
    stages,
):
    """A dA tile may span sort blocks while masking each expert's real tail."""

    device = _gfx950_device()
    hidden_size, intermediate_size, num_experts = 128, 64, 3
    frequencies = (1, 65, 127)
    first_rows = (0, 64, 192)
    padded_rows = 320
    generator = torch.Generator(device=device).manual_seed(
        611 + block_m + int(queue_direct)
    )

    dy = torch.full(
        (padded_rows, hidden_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    live_rows = torch.zeros(padded_rows, dtype=torch.bool, device=device)
    for first_row, frequency in zip(first_rows, frequencies, strict=True):
        dy[first_row : first_row + frequency] = torch.randn(
            (frequency, hidden_size),
            dtype=torch.float32,
            device=device,
            generator=generator,
        ).to(torch.bfloat16)
        live_rows[first_row : first_row + frequency] = True

    w2 = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    frequency = torch.tensor(frequencies, dtype=torch.int32, device=device)
    # One metadata entry per 64-row sorter block.
    sorted_expert_ids = torch.tensor([0, 1, 1, 2, 2], dtype=torch.int32, device=device)
    num_valid_ids = torch.tensor(
        [padded_rows, sum(frequencies)],
        dtype=torch.int32,
        device=device,
    )
    queue_entries = [num_experts]
    for expert, first_row in enumerate(first_rows):
        queue_entries.extend((expert, first_row))
    active_queue = torch.tensor(queue_entries, dtype=torch.int32, device=device)
    output = torch.full(
        (padded_rows, intermediate_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )

    grouped_da = compile_grouped_da_gfx950(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        sorted_block_size=64,
        block_m=block_m,
        block_n=64,
        block_k=64,
        stages=stages,
        m_waves=m_waves,
        n_waves=n_waves,
        queue_direct=queue_direct,
        persistent=not queue_direct,
    )
    _run_compiled(
        grouped_da,
        dy.data_ptr(),
        w2.data_ptr(),
        frequency.data_ptr(),
        sorted_expert_ids.data_ptr(),
        num_valid_ids.data_ptr(),
        active_queue.data_ptr(),
        output.data_ptr(),
        1,
        torch.cuda.current_stream(device),
    )
    torch.cuda.synchronize(device)

    for expert, (first_row, expert_rows) in enumerate(
        zip(first_rows, frequencies, strict=True)
    ):
        expected = dy[first_row : first_row + expert_rows].float() @ w2[expert].float()
        torch.testing.assert_close(
            output[first_row : first_row + expert_rows].float(),
            expected,
            rtol=3e-2,
            atol=5e-2,
        )
    assert torch.isnan(output[~live_rows]).all()


def test_grouped_da_stage3_h192_executes_steady_state_pipeline():
    """Three BK64 tiles exercise one stage-3 main-loop iteration before drain."""

    device = _gfx950_device()
    hidden_size, intermediate_size, num_experts = 192, 64, 1
    frequency_value = 65
    padded_rows = 128
    generator = torch.Generator(device=device).manual_seed(719)

    dy = torch.full(
        (padded_rows, hidden_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    dy[:frequency_value] = torch.randn(
        (frequency_value, hidden_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    w2 = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    frequency = torch.tensor([frequency_value], dtype=torch.int32, device=device)
    sorted_expert_ids = torch.zeros(2, dtype=torch.int32, device=device)
    num_valid_ids = torch.tensor(
        [padded_rows, frequency_value],
        dtype=torch.int32,
        device=device,
    )
    active_queue = torch.tensor([1, 0, 0], dtype=torch.int32, device=device)
    output = torch.full(
        (padded_rows, intermediate_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )

    grouped_da = compile_grouped_da_gfx950(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        sorted_block_size=64,
        block_m=128,
        block_n=64,
        block_k=64,
        stages=3,
        m_waves=4,
        n_waves=2,
        persistent=True,
    )
    _run_compiled(
        grouped_da,
        dy.data_ptr(),
        w2.data_ptr(),
        frequency.data_ptr(),
        sorted_expert_ids.data_ptr(),
        num_valid_ids.data_ptr(),
        active_queue.data_ptr(),
        output.data_ptr(),
        1,
        torch.cuda.current_stream(device),
    )
    torch.cuda.synchronize(device)

    expected = dy[:frequency_value].float() @ w2[0].float()
    torch.testing.assert_close(
        output[:frequency_value].float(),
        expected,
        rtol=3e-2,
        atol=5e-2,
    )
    assert torch.isnan(output[frequency_value:]).all()
