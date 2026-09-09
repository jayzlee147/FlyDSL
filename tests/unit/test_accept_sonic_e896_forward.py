# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CPU-only contracts for the E896 forward acceptance comparison graph."""

from __future__ import annotations

from tools.accept_sonic_e896_forward import (
    _comparison_record,
    _parse_args,
    _reference_profile_name,
)


def test_forward_acceptance_defaults_to_global_baseline() -> None:
    args = _parse_args(["--list-profiles", "--profiles", "m80-pipeline2"])
    assert args.compare_to == "baseline"
    assert _reference_profile_name("m80-pipeline2", args.compare_to) == "baseline"


def test_forward_acceptance_parent_mode_resolves_each_declared_parent() -> None:
    args = _parse_args(
        [
            "--list-profiles",
            "--compare-to",
            "parent",
            "--profiles",
            "m80-pipeline2",
            "m80-stage1-persistent",
            "m80-stage2-persistent",
        ]
    )
    assert args.compare_to == "parent"
    assert {
        name: _reference_profile_name(name, args.compare_to)
        for name in args.profiles
    } == {
        "m80-pipeline2": "m80-equal",
        "m80-stage1-persistent": "m80-equal",
        "m80-stage2-persistent": "m80-equal",
    }


def test_parent_comparison_records_resolved_reference_identity_and_single_step_diff() -> None:
    comparison = _comparison_record("m80-pipeline2", "parent")
    assert comparison["reference_profile"] == "m80-equal"
    assert comparison["candidate_profile"] == "m80-pipeline2"
    assert comparison["reference_config_sha256"] != comparison["candidate_config_sha256"]
    assert comparison["reference_config"]["tile_m"] == 80
    assert comparison["candidate_config"]["tile_m"] == 80
    assert comparison["changes_from_reference"] == {
        "stage2_pipeline_stages": {"from": None, "to": 2}
    }
    assert comparison["prepared_weight_compatibility"] == {
        "fields": [
            "hidden_size",
            "intermediate_size",
            "num_experts",
            "activation",
            "compute_dtype",
        ],
        "reference_signature": {
            "hidden_size": 3584,
            "intermediate_size": 512,
            "num_experts": 896,
            "activation": "swiglu",
            "compute_dtype": "bf16",
        },
        "candidate_signature": {
            "hidden_size": 3584,
            "intermediate_size": 512,
            "num_experts": 896,
            "activation": "swiglu",
            "compute_dtype": "bf16",
        },
        "reference_matches_baseline": True,
        "candidate_matches_baseline": True,
        "candidate_matches_reference": True,
    }


def test_global_baseline_comparison_keeps_historical_reference() -> None:
    comparison = _comparison_record("m80-pipeline2", "baseline")
    assert comparison["reference_profile"] == "baseline"
    assert comparison["changes_from_reference"] == {
        "down_tile_m": {"from": 64, "to": 80},
        "stage2_pipeline_stages": {"from": None, "to": 2},
        "tile_m": {"from": 64, "to": 80},
    }


def test_acceptance_only_profiles_have_exact_parent_relative_diffs() -> None:
    expected = {
        "bn256-bk64-stage1-xcd8": (
            "bn256-bk64",
            {"stage1_xcd_swizzle": {"from": 0, "to": 8}},
        ),
        "bn256-bk64-stage2-xcd8": (
            "bn256-bk64",
            {"stage2_xcd_swizzle": {"from": 1, "to": 8}},
        ),
        "xcd8-cached-pipeline2": (
            "xcd8-cached",
            {"stage2_pipeline_stages": {"from": None, "to": 2}},
        ),
    }
    for profile_name, (parent, changes) in expected.items():
        comparison = _comparison_record(profile_name, "parent")
        assert comparison["reference_profile"] == parent
        assert comparison["changes_from_reference"] == changes
        assert comparison["prepared_weight_compatibility"]["candidate_matches_reference"]

    for profile_name in ("bn256-bk64-stage1-xcd8", "bn256-bk64-stage2-xcd8"):
        candidate = _comparison_record(profile_name, "parent")["candidate_config"]
        assert candidate["stage1_b_cache_mod"] is None
        assert candidate["stage2_b_cache_mod"] is None
