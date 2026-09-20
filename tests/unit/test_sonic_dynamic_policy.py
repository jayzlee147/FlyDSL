# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import pytest

from kernels.moe.sonic_dynamic_policy import (
    E16_ROUTE_POLICY_SIZES,
    E16RoutePolicy,
    canonical_e16_route_policy_size,
    e16_route_policy_representative,
    select_e16_route_policy,
)


# Retained Qwen3 E16 preactivation stores ``R x (2 * I)`` BF16 values and
# uses signed-i32 byte offsets in the kernels.  Keep the actual supported
# high-water mark in the dynamic-policy regression matrix instead of testing
# an arbitrary large integer only.
_MAX_LEGAL_E16_RETAINED_ROUTES = ((1 << 31) - 1) // (2 * 768 * 2)


def test_e16_policy_sizes_are_the_complete_finite_family():
    assert E16_ROUTE_POLICY_SIZES == (4096, 16384, 32768, 65536)


@pytest.mark.parametrize(
    ("shared_max_routes", "expected"),
    (
        (0, 4096),
        (1, 4096),
        (4096, 4096),
        (4097, 16384),
        (16384, 16384),
        (16385, 32768),
        (32768, 32768),
        (32769, 65536),
        (134000, 65536),
        (_MAX_LEGAL_E16_RETAINED_ROUTES, 65536),
    ),
)
def test_canonical_e16_route_policy_size(shared_max_routes, expected):
    assert canonical_e16_route_policy_size(shared_max_routes) == expected


@pytest.mark.parametrize("invalid", (-1, True, 1.5, "8192"))
def test_canonical_e16_route_policy_size_rejects_invalid_input(invalid):
    with pytest.raises((TypeError, ValueError)):
        canonical_e16_route_policy_size(invalid)


@pytest.mark.parametrize(
    ("routes", "expected"),
    (
        (0, E16RoutePolicy.SMALL),
        (1, E16RoutePolicy.SMALL),
        (63, E16RoutePolicy.SMALL),
        (64, E16RoutePolicy.SMALL),
        (2047, E16RoutePolicy.SMALL),
        (2048, E16RoutePolicy.SMALL),
        (2049, E16RoutePolicy.SMALL),
        (4095, E16RoutePolicy.SMALL),
        (4096, E16RoutePolicy.SMALL),
        (4097, E16RoutePolicy.MEDIUM),
        (8191, E16RoutePolicy.MEDIUM),
        (8192, E16RoutePolicy.MEDIUM),
        (8193, E16RoutePolicy.MEDIUM),
        (16383, E16RoutePolicy.MEDIUM),
        (16384, E16RoutePolicy.MEDIUM),
        (16385, E16RoutePolicy.LARGE),
        (32767, E16RoutePolicy.LARGE),
        (32768, E16RoutePolicy.LARGE),
        (32769, E16RoutePolicy.XLARGE),
        (65535, E16RoutePolicy.XLARGE),
        (65536, E16RoutePolicy.XLARGE),
        (65537, E16RoutePolicy.XLARGE),
        (133999, E16RoutePolicy.XLARGE),
        (134000, E16RoutePolicy.XLARGE),
        (134001, E16RoutePolicy.XLARGE),
        *((routes, E16RoutePolicy.XLARGE) for routes in range(131072, 131089)),
        (139279, E16RoutePolicy.XLARGE),
        (139280, E16RoutePolicy.XLARGE),
        (_MAX_LEGAL_E16_RETAINED_ROUTES, E16RoutePolicy.XLARGE),
        (1 << 30, E16RoutePolicy.XLARGE),
    ),
)
def test_e16_route_policy_boundary_matrix(routes, expected):
    assert select_e16_route_policy(routes) == expected


@pytest.mark.parametrize(
    ("route_policy_size", "expected"),
    (
        (2048, E16RoutePolicy.SMALL),
        (8192, E16RoutePolicy.MEDIUM),
        (32768, E16RoutePolicy.LARGE),
        (65536, E16RoutePolicy.XLARGE),
    ),
)
def test_rank_shared_policy_is_independent_of_local_routes(
    route_policy_size,
    expected,
):
    """One EP-wide hint gives every imbalanced rank one scheduling family."""

    # A shared hint aligns all ordinary rank-local shards which do not exceed
    # it.  Larger unexpected shards are covered by the promotion test below.
    local_routes_by_rank = (0, 1, route_policy_size // 2, route_policy_size)
    policies = {
        select_e16_route_policy(
            local_routes,
            route_policy_size=route_policy_size,
        )
        for local_routes in local_routes_by_rank
    }
    assert policies == {expected}


def test_route_policy_hint_is_a_floor_not_an_allocation_bound():
    """Actual R above the hint promotes safely; values below reuse its family."""

    assert (
        select_e16_route_policy(134000, route_policy_size=2048)
        == E16RoutePolicy.XLARGE
    )
    assert (
        select_e16_route_policy(1, route_policy_size=65536)
        == E16RoutePolicy.XLARGE
    )


def test_shared_capacity_aligns_balanced_and_skewed_rank_loads():
    """A conservative shared floor keeps every covered rank on one family."""

    balanced_rank_routes = (8192,) * 8
    hot_rank_routes = (134000, 16000, 12000, 9000, 8192, 4096, 2048, 1)
    route_policy_size = 134000
    balanced = tuple(
        (
            select_e16_route_policy(routes, route_policy_size),
            e16_route_policy_representative(
                select_e16_route_policy(routes, route_policy_size)
            ),
        )
        for routes in balanced_rank_routes
    )
    hot = tuple(
        (
            select_e16_route_policy(routes, route_policy_size),
            e16_route_policy_representative(
                select_e16_route_policy(routes, route_policy_size)
            ),
        )
        for routes in hot_rank_routes
    )

    assert len(set(balanced)) == 1
    assert len(set(hot)) == 1
    assert balanced[0] == hot[0] == (E16RoutePolicy.XLARGE, 65536)


def test_outlier_promotion_uses_only_existing_finite_families():
    """An underestimated shared floor cannot strand a large shard on SMALL."""

    policies = tuple(
        select_e16_route_policy(routes, route_policy_size=8192)
        for routes in (1, 8192, 8193, 16384, 16385, 32768, 32769, 134000)
    )
    assert policies == (
        E16RoutePolicy.MEDIUM,
        E16RoutePolicy.MEDIUM,
        E16RoutePolicy.MEDIUM,
        E16RoutePolicy.MEDIUM,
        E16RoutePolicy.LARGE,
        E16RoutePolicy.LARGE,
        E16RoutePolicy.XLARGE,
        E16RoutePolicy.XLARGE,
    )


def test_dynamic_route_sequence_has_only_four_policy_jit_families():
    """An open-ended dynamic sequence cannot grow policy variants with R."""

    routes = (
        0,
        1,
        2047,
        2048,
        2049,
        4095,
        4096,
        4097,
        8191,
        8192,
        8193,
        16383,
        16384,
        16385,
        32767,
        32768,
        32769,
        65535,
        65536,
        65537,
        133999,
        134000,
        134001,
        *range(131072, 131089),
        139279,
        139280,
        _MAX_LEGAL_E16_RETAINED_ROUTES,
        1 << 20,
        1 << 30,
    )
    policy_families = {
        (
            policy := select_e16_route_policy(value),
            e16_route_policy_representative(policy),
        )
        for value in routes
    }

    assert policy_families == {
        (E16RoutePolicy.SMALL, 4096),
        (E16RoutePolicy.MEDIUM, 16384),
        (E16RoutePolicy.LARGE, 32768),
        (E16RoutePolicy.XLARGE, 65536),
    }


def test_e16_retained_state_high_water_matches_i32_byte_extent():
    assert _MAX_LEGAL_E16_RETAINED_ROUTES == 699050
    assert _MAX_LEGAL_E16_RETAINED_ROUTES * 2 * 768 * 2 <= (1 << 31) - 1
    assert (
        (_MAX_LEGAL_E16_RETAINED_ROUTES + 1) * 2 * 768 * 2
        > (1 << 31) - 1
    )


@pytest.mark.parametrize(
    ("policy", "representative"),
    (
        (E16RoutePolicy.SMALL, 4096),
        (E16RoutePolicy.MEDIUM, 16384),
        (E16RoutePolicy.LARGE, 32768),
        (E16RoutePolicy.XLARGE, 65536),
    ),
)
def test_e16_route_policy_representatives_are_finite(policy, representative):
    assert e16_route_policy_representative(policy) == representative


@pytest.mark.parametrize("routes", (-1, True, 1.5))
def test_e16_route_policy_rejects_invalid_routes(routes):
    with pytest.raises((TypeError, ValueError)):
        select_e16_route_policy(routes)


@pytest.mark.parametrize("hint", (0, -1, True, 1.5))
def test_e16_route_policy_rejects_invalid_hint(hint):
    with pytest.raises((TypeError, ValueError)):
        select_e16_route_policy(8192, hint)
