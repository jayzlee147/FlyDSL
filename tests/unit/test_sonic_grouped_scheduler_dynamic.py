# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CPU-only checks for dynamic SonicMoE scheduler launch parameters."""

from __future__ import annotations

import inspect

import pytest
import torch

from kernels.moe import sonic_backward as backward
from kernels.moe import sonic_grouped_scheduler as scheduler
from kernels.moe import sonic_grouped_tn as grouped_tn
from kernels.moe.sonic_dynamic_policy import (
    E16RoutePolicy,
    select_e16_route_policy,
)


_MAX_LEGAL_E16_RETAINED_ROUTES = ((1 << 31) - 1) // (2 * 768 * 2)
_DYNAMIC_ROUTE_COUNTS = (
    0,
    1,
    63,
    64,
    4095,
    4096,
    4097,
    16383,
    16384,
    16385,
    32767,
    32768,
    32769,
    65536,
    *range(131072, 131089),
    134000,
    139279,
    139280,
    _MAX_LEGAL_E16_RETAINED_ROUTES,
)
_ROUTING_DISTRIBUTIONS = ("balanced", "hot1", "hot4", "long_tail")


def _expert_frequencies(routes: int, distribution: str) -> tuple[int, ...]:
    if distribution == "balanced":
        quotient, remainder = divmod(routes, 16)
        frequencies = tuple(
            quotient + int(expert < remainder) for expert in range(16)
        )
    elif distribution == "hot1":
        frequencies = (routes, *([0] * 15))
    elif distribution == "hot4":
        quotient, remainder = divmod(routes, 4)
        frequencies = (
            *(quotient + int(expert < remainder) for expert in range(4)),
            *([0] * 12),
        )
    elif distribution == "long_tail":
        head = (routes + 1) // 2
        quotient, remainder = divmod(routes - head, 15)
        frequencies = (
            head,
            *(quotient + int(expert < remainder) for expert in range(15)),
        )
    else:  # pragma: no cover - helper is only called by fixed parametrization.
        raise AssertionError(f"unknown distribution {distribution}")

    assert len(frequencies) == 16
    assert sum(frequencies) == routes
    return frequencies


def _assert_partition_covers_frequency(frequency: int, rows_per_part: int) -> int:
    """Check the CPU reference partition has no gaps or overlapping rows."""

    cursor = 0
    records = 0
    while cursor < frequency:
        start = records * rows_per_part
        assert start == cursor
        cursor = min(start + rows_per_part, frequency)
        records += 1
    assert cursor == frequency
    return records


class _FakeI32Tensor:
    def __init__(self, elements: int, *, device: torch.device):
        self.ndim = 1
        self.dtype = torch.int32
        self.device = device
        self._elements = elements

    def numel(self) -> int:
        return self._elements

    def is_contiguous(self) -> bool:
        return True


def _runtime_queue_tensors():
    device = torch.device("cuda", 0)
    return {
        "expert_frequency": _FakeI32Tensor(16, device=device),
        "sorted_expert_ids": _FakeI32Tensor(64, device=device),
        "num_valid_ids": _FakeI32Tensor(2, device=device),
        "queue": _FakeI32Tensor(1 + 3 * 128, device=device),
        "active_expert_storage": _FakeI32Tensor(1 + 2 * 16, device=device),
        "hot_split_storage": _FakeI32Tensor(1 + 3 * 32, device=device),
        "hot_expert_storage": _FakeI32Tensor(1 + 3 * 16, device=device),
    }


@pytest.mark.parametrize("routes", _DYNAMIC_ROUTE_COUNTS)
@pytest.mark.parametrize("distribution", _ROUTING_DISTRIBUTIONS)
def test_exact_queue_capacity_covers_every_dynamic_distribution(
    routes,
    distribution,
):
    """The exact queue bound covers every row once at all supported sizes."""

    block_m = 128
    frequencies = _expert_frequencies(routes, distribution)
    emitted_records = sum(
        _assert_partition_covers_frequency(frequency, block_m)
        for frequency in frequencies
    )
    capacity = scheduler.exact_m_tile_queue_upper_bound(
        routes,
        16,
        block_m,
    )

    assert emitted_records <= capacity
    assert capacity == scheduler.ragged_compact_m_tile_descriptor_upper_bound(
        routes,
        16,
        block_m,
    )


@pytest.mark.parametrize("routes", _DYNAMIC_ROUTE_COUNTS)
@pytest.mark.parametrize("distribution", _ROUTING_DISTRIBUTIONS)
def test_hot_split_schedule_is_bounded_and_covers_dynamic_routes(
    routes,
    distribution,
):
    """Hot-split and regular paths partition, rather than drop, expert rows."""

    frequencies = _expert_frequencies(routes, distribution)
    # None covers the natural family; the two shared floors exercise both
    # aligned-rank scheduling and automatic promotion of unexpected outliers.
    for route_policy_size in (None, 16384, 65536):
        policy = select_e16_route_policy(routes, route_policy_size)
        if policy not in (E16RoutePolicy.LARGE, E16RoutePolicy.XLARGE):
            # SMALL/MEDIUM intentionally use only the regular grouped path.
            assert sum(frequencies) == routes
            continue

        capacity, split_rows, min_hot_rows = backward._e16_hot_split_schedule(
            routes,
            16,
            policy,
        )
        expected_min_hot_rows = split_rows + 1
        if policy == E16RoutePolicy.XLARGE:
            expected_min_hot_rows = max(
                expected_min_hot_rows,
                backward._e16_dw2_hot_profile_min_rows(routes),
            )

        policy_capacity = (
            backward._E16_DW1_LARGE_SPLIT_CAPACITY
            if policy == E16RoutePolicy.LARGE
            else backward._E16_DW1_XLARGE_SPLIT_CAPACITY
        )
        assert split_rows >= backward._E16_DW1_SPLIT_ROWS
        assert split_rows % backward._E16_DW1_SPLIT_ROW_QUANTUM == 0
        assert min_hot_rows == expected_min_hot_rows

        emitted_records = 0
        hot_rows = 0
        regular_rows = 0
        for frequency in frequencies:
            if frequency >= min_hot_rows:
                emitted_records += _assert_partition_covers_frequency(
                    frequency,
                    split_rows,
                )
                hot_rows += frequency
            else:
                regular_rows += frequency

        descriptor_bound = grouped_tn.hot_split_descriptor_capacity(
            routes,
            16,
            split_rows,
            min_hot_rows,
        )
        assert capacity == max(1, descriptor_bound)
        assert emitted_records <= descriptor_bound <= policy_capacity
        assert hot_rows + regular_rows == routes
        if distribution == "balanced":
            assert emitted_records == 0


def test_exact_queue_split_thresholds_are_not_compile_parameters():
    parameters = inspect.signature(
        scheduler.compile_exact_m_tile_queue_builder
    ).parameters

    assert "routes" not in parameters
    assert "split_rows" not in parameters
    assert "min_hot_rows" not in parameters


def test_exact_queue_split_thresholds_share_compile_family(monkeypatch):
    compile_calls = []
    runtime_calls = []

    def fake_compile(*args, **kwargs):
        compile_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(
        scheduler,
        "compile_exact_m_tile_queue_builder",
        fake_compile,
    )
    monkeypatch.setattr(
        scheduler,
        "_run_compiled",
        lambda *args: runtime_calls.append(args),
    )

    tensors = _runtime_queue_tensors()
    thresholds = ((8192, 8193), (12288, 12289), (65536, 65537))
    stream = object()
    for split_rows, min_hot_rows in thresholds:
        scheduler.build_exact_m_tile_queue(
            tensors["expert_frequency"],
            tensors["sorted_expert_ids"],
            tensors["num_valid_ids"],
            tensors["queue"],
            block_m=128,
            sorted_block_m=128,
            active_expert_storage=tensors["active_expert_storage"],
            active_expert_capacity=16,
            hot_split_storage=tensors["hot_split_storage"],
            hot_expert_storage=tensors["hot_expert_storage"],
            split_rows=split_rows,
            min_hot_rows=min_hot_rows,
            stream=stream,
        )

    assert compile_calls == [compile_calls[0]] * len(thresholds)
    assert compile_calls[0][0] == (16, 128, 128, 0, True, True)
    assert tuple(call[-3:-1] for call in runtime_calls) == thresholds
    assert all(call[9] == 32 and call[11] == 16 for call in runtime_calls)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"split_rows": 0, "min_hot_rows": 1}, "0 < split_rows"),
        ({"split_rows": 128, "min_hot_rows": 127}, "0 < split_rows"),
        (
            {"hot_split_storage": None},
            "hot split and hot expert storage must be supplied together",
        ),
    ),
)
def test_exact_queue_runtime_split_threshold_validation(
    monkeypatch,
    overrides,
    message,
):
    monkeypatch.setattr(
        scheduler,
        "compile_exact_m_tile_queue_builder",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(scheduler, "_run_compiled", lambda *_args: None)
    tensors = _runtime_queue_tensors()
    kwargs = {
        "block_m": 128,
        "sorted_block_m": 128,
        "active_expert_storage": tensors["active_expert_storage"],
        "active_expert_capacity": 16,
        "hot_split_storage": tensors["hot_split_storage"],
        "hot_expert_storage": tensors["hot_expert_storage"],
        "split_rows": 8192,
        "min_hot_rows": 8193,
        "stream": object(),
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        scheduler.build_exact_m_tile_queue(
            tensors["expert_frequency"],
            tensors["sorted_expert_ids"],
            tensors["num_valid_ids"],
            tensors["queue"],
            **kwargs,
        )


def test_exact_queue_runtime_split_storage_capacities_are_validated(monkeypatch):
    monkeypatch.setattr(
        scheduler,
        "compile_exact_m_tile_queue_builder",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(scheduler, "_run_compiled", lambda *_args: None)
    tensors = _runtime_queue_tensors()
    tensors["hot_split_storage"] = _FakeI32Tensor(
        1 + 3 * 32 + 1,
        device=tensors["expert_frequency"].device,
    )

    with pytest.raises(ValueError, match="hot_split_storage has an invalid"):
        scheduler.build_exact_m_tile_queue(
            tensors["expert_frequency"],
            tensors["sorted_expert_ids"],
            tensors["num_valid_ids"],
            tensors["queue"],
            block_m=128,
            sorted_block_m=128,
            active_expert_storage=tensors["active_expert_storage"],
            active_expert_capacity=16,
            hot_split_storage=tensors["hot_split_storage"],
            hot_expert_storage=tensors["hot_expert_storage"],
            split_rows=8192,
            min_hot_rows=8193,
            stream=object(),
        )
