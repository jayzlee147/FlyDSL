# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CPU-only policy tests for sorter-native E896 backward metadata."""

import pytest

from kernels.moe.moe_sorting_kernel import _supports_e896_backward_metadata
from kernels.moe.sonic_backward import _use_sorter_native_backward_metadata


@pytest.mark.parametrize(
    ("overrides", "expected"),
    (
        ({}, True),
        ({"tokens": 2048}, False),
        ({"num_experts": 895}, False),
        ({"topk": 8}, False),
        ({"unit_size": 32}, False),
        ({"has_mask": True}, False),
    ),
)
def test_sorter_metadata_contract_is_exact(overrides, expected):
    kwargs = {
        "tokens": 4096,
        "num_experts": 896,
        "topk": 16,
        "unit_size": 64,
        "has_mask": False,
    }
    kwargs.update(overrides)
    assert _supports_e896_backward_metadata(**kwargs) is expected


@pytest.mark.parametrize(
    "disabled",
    (
        "flat_routes",
        "has_bias",
        "reuse_forward_preactivation",
        "use_hostless_grouped",
        "use_large_grouped_dx",
        "use_compact_w1",
        "sort_unit",
        "tokens",
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "topk",
    ),
)
def test_backward_selects_sorter_metadata_only_for_production_contract(disabled):
    kwargs = {
        "flat_routes": False,
        "has_bias": False,
        "tokens": 4096,
        "hidden_size": 3584,
        "intermediate_size": 512,
        "num_experts": 896,
        "topk": 16,
        "reuse_forward_preactivation": True,
        "use_hostless_grouped": True,
        "use_large_grouped_dx": True,
        "use_compact_w1": False,
        "sort_unit": 64,
    }
    assert _use_sorter_native_backward_metadata(**kwargs)

    if disabled in ("flat_routes", "has_bias", "use_compact_w1"):
        kwargs[disabled] = True
    elif disabled in (
        "reuse_forward_preactivation",
        "use_hostless_grouped",
        "use_large_grouped_dx",
    ):
        kwargs[disabled] = False
    elif disabled == "sort_unit":
        kwargs[disabled] = 32
    else:
        kwargs[disabled] -= 1
    assert not _use_sorter_native_backward_metadata(**kwargs)
