# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CPU-only contract tests for the E896 backward acceptance policy."""

import pytest

from tools.accept_sonic_e896_backward import (
    _evaluate_launch_topology,
    _evaluate_policy_probe,
    _parse_args,
)


def _hostless_launches(**overrides):
    launches = {
        "generic_gemm_launches": 0,
        "projection_launches": 0,
        "legacy_dx_gemm_launches": 0,
        "grouped_dx_launches": 1,
        "total_dx_launches": 1,
        "host_segment_materializations": 0,
    }
    launches.update(overrides)
    return launches


def _legacy_launches():
    return {
        "generic_gemm_launches": 3,
        "projection_launches": 2,
        "legacy_dx_gemm_launches": 1,
        "grouped_dx_launches": 0,
        "total_dx_launches": 1,
        "host_segment_materializations": 1,
    }


def test_parse_args_preserves_legacy_mode_by_default():
    args = _parse_args(["--baseline", "/tmp/baseline.py", "--correctness-only"])
    assert args.comparison_mode == "legacy-vs-hostless"


def test_parse_args_accepts_incremental_mode():
    args = _parse_args(
        [
            "--comparison-mode",
            "incremental",
            "--baseline",
            "/tmp/baseline.py",
            "--correctness-only",
        ]
    )
    assert args.comparison_mode == "incremental"


def test_legacy_policy_retains_original_split():
    probe = _evaluate_policy_probe(
        "legacy-vs-hostless",
        baseline_large_grouped_dx=False,
        candidate_large_grouped_dx=True,
        baseline_hostless_retained_backward=None,
        candidate_hostless_retained_backward=True,
    )
    assert probe["passed"]
    assert set(probe["required_checks"]) == {
        "baseline_uses_legacy_dx",
        "candidate_uses_large_grouped_dx",
        "candidate_uses_hostless_retained_backward",
    }


def test_incremental_policy_accepts_two_hostless_large_grouped_dx_sources():
    probe = _evaluate_policy_probe(
        "incremental",
        baseline_large_grouped_dx=True,
        candidate_large_grouped_dx=True,
        baseline_hostless_retained_backward=True,
        candidate_hostless_retained_backward=True,
    )
    assert probe["passed"]


@pytest.mark.parametrize(
    "overrides",
    (
        {"baseline_large_grouped_dx": False},
        {"candidate_large_grouped_dx": False},
        {"baseline_hostless_retained_backward": False},
        {"candidate_hostless_retained_backward": False},
    ),
)
def test_incremental_policy_requires_both_large_grouped_dx_and_hostless(overrides):
    observations = {
        "baseline_large_grouped_dx": True,
        "candidate_large_grouped_dx": True,
        "baseline_hostless_retained_backward": True,
        "candidate_hostless_retained_backward": True,
    }
    observations.update(overrides)
    assert not _evaluate_policy_probe("incremental", **observations)["passed"]


def test_incremental_launch_gate_does_not_require_legacy_baseline_work():
    gate = _evaluate_launch_topology("incremental", _hostless_launches(), _hostless_launches())
    assert gate["passed"]
    assert not gate["baseline_exercises_legacy_dx"]
    assert not gate["baseline_exercises_projection"]
    assert not gate["baseline_exercises_host_segment_materialization"]
    assert "baseline_exercises_legacy_dx" not in gate["required_checks"]
    assert "baseline_exercises_projection" not in gate["required_checks"]
    assert "baseline_exercises_host_segment_materialization" not in gate["required_checks"]


@pytest.mark.parametrize(
    ("baseline_overrides", "candidate_overrides"),
    (
        ({"grouped_dx_launches": 0}, {}),
        ({}, {"grouped_dx_launches": 0, "total_dx_launches": 0}),
        ({}, {"generic_gemm_launches": 1}),
        ({}, {"host_segment_materializations": 1}),
    ),
)
def test_incremental_launch_gate_rejects_required_topology_violations(
    baseline_overrides,
    candidate_overrides,
):
    baseline = _hostless_launches(**baseline_overrides)
    candidate = _hostless_launches(**candidate_overrides)
    assert not _evaluate_launch_topology("incremental", baseline, candidate)["passed"]


def test_legacy_launch_gate_still_requires_legacy_baseline_work():
    assert _evaluate_launch_topology("legacy-vs-hostless", _legacy_launches(), _hostless_launches())["passed"]
    assert not _evaluate_launch_topology(
        "legacy-vs-hostless",
        _hostless_launches(),
        _hostless_launches(),
    )["passed"]
