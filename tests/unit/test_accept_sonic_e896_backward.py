# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CPU-only contract tests for the E896 backward acceptance policy."""

import importlib
import sys
import types
from pathlib import Path

import pytest

import tools.accept_sonic_e896_backward as acceptance
from tools.accept_sonic_e896_backward import (
    BASELINE_RUNTIME_OVERRIDE_CHOICES,
    BASELINE_RUNTIME_OVERRIDE_SYMBOLS,
    _acceptance_report_fields,
    _collect_code_identity,
    _evaluate_launch_topology,
    _evaluate_policy_probe,
    _file_identity,
    _git_value,
    _load_baseline_runtime_overrides,
    _load_isolated_baseline,
    _load_private_runtime_module,
    _parse_args,
    _patch_baseline_runtime_symbols,
    _performance_report_fields,
    _timing_status,
)

TEST_SHARED_SOURCE = "kernels/common/buffer_ops.py"


@pytest.fixture
def identity_repos(tmp_path, monkeypatch):
    candidate_repo = tmp_path / "candidate"
    baseline_repo = tmp_path / "baseline"
    shared_sources = (*BASELINE_RUNTIME_OVERRIDE_CHOICES, TEST_SHARED_SOURCE)
    for repo in (candidate_repo, baseline_repo):
        (repo / "tools").mkdir(parents=True)
        (repo / "tools/accept_sonic_e896_backward.py").write_text("acceptance tool\n")
        backward = repo / "kernels/moe/sonic_backward.py"
        backward.parent.mkdir(parents=True)
        backward.write_text("same backward\n")
        for relative in shared_sources:
            source = repo / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"same {relative}\n")

    candidate_backward = candidate_repo / "kernels/moe/sonic_backward.py"
    monkeypatch.setattr(acceptance, "SHARED_RUNTIME_SOURCES", shared_sources)
    monkeypatch.setattr(acceptance, "__file__", str(candidate_repo / "tools/accept_sonic_e896_backward.py"))
    monkeypatch.setattr(acceptance.candidate_module, "__file__", str(candidate_backward))
    return {
        "candidate_repo": candidate_repo,
        "baseline_repo": baseline_repo,
        "candidate_backward": candidate_backward,
        "baseline_backward": baseline_repo / "kernels/moe/sonic_backward.py",
    }


def _write_private_runtime(tmp_path: Path, relative: str, *, omit: str | None = None):
    symbols = [symbol for symbol in BASELINE_RUNTIME_OVERRIDE_SYMBOLS[relative] if symbol != omit]
    source = "\n\n".join(f"def {symbol}(*args, **kwargs):\n    return {symbol!r}" for symbol in symbols)
    path = tmp_path / Path(relative).name
    path.write_text(source + "\n")
    private_name = f"kernels.moe._e896_test_{Path(relative).stem}_{tmp_path.name}"
    return acceptance._load_source_module(private_name, path)


def _baseline_bindings(relative: str):
    canonical = importlib.import_module(acceptance._runtime_module_name(relative))
    baseline = types.ModuleType("kernels.moe._e896_test_backward")
    for symbol in BASELINE_RUNTIME_OVERRIDE_SYMBOLS[relative]:
        setattr(baseline, symbol, getattr(canonical, symbol))
    return baseline


def _hostless_launches(**overrides):
    launches = {
        "generic_gemm_launches": 0,
        "projection_launches": 0,
        "legacy_dx_gemm_launches": 0,
        "grouped_dx_launches": 1,
        "total_dx_launches": 1,
        "host_segment_materializations": 0,
        "expert_histogram_sequences": 0,
        "compact_descriptor_builder_sequences": 0,
        "sorter_calls": 1,
        "sorter_backward_metadata_calls": 1,
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
        "expert_histogram_sequences": 1,
        "compact_descriptor_builder_sequences": 1,
        "sorter_calls": 1,
        "sorter_backward_metadata_calls": 0,
    }


def test_git_value_uses_command_scoped_safe_directory(tmp_path, monkeypatch):
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return types.SimpleNamespace(stdout="test-value\n")

    monkeypatch.setattr(acceptance.subprocess, "run", fake_run)
    assert _git_value(tmp_path, "rev-parse", "HEAD") == "test-value"
    resolved = tmp_path.resolve()
    assert observed["command"] == (
        "git",
        "-C",
        str(resolved),
        "-c",
        f"safe.directory={resolved}",
        "rev-parse",
        "HEAD",
    )
    assert observed["kwargs"] == {
        "check": True,
        "stdout": acceptance.subprocess.PIPE,
        "stderr": acceptance.subprocess.DEVNULL,
        "text": True,
    }


def test_parse_args_preserves_legacy_mode_by_default():
    args = _parse_args(["--baseline", "/tmp/baseline.py", "--correctness-only"])
    assert args.comparison_mode == "legacy-vs-hostless"
    assert args.baseline_runtime_override == []


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


def test_parse_args_accepts_and_normalizes_repeated_runtime_overrides():
    args = _parse_args(
        [
            "--baseline",
            "/tmp/baseline.py",
            "--correctness-only",
            "--baseline-runtime-override",
            "kernels/moe/sonic_grouped_tn.py",
            "--baseline-runtime-override",
            "kernels/moe/grouped_da_gfx950.py",
        ]
    )
    assert args.baseline_runtime_override == [
        "kernels/moe/grouped_da_gfx950.py",
        "kernels/moe/sonic_grouped_tn.py",
    ]


@pytest.mark.parametrize(
    "value",
    (
        "sonic_grouped_tn.py",
        "./kernels/moe/sonic_grouped_tn.py",
        "/tmp/kernels/moe/sonic_grouped_tn.py",
        "kernels/moe/../moe/sonic_grouped_tn.py",
    ),
)
def test_parse_args_rejects_non_exact_runtime_override(value):
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--baseline",
                "/tmp/baseline.py",
                "--correctness-only",
                "--baseline-runtime-override",
                value,
            ]
        )


def test_parse_args_rejects_duplicate_runtime_override():
    relative = "kernels/moe/sonic_grouped_tn.py"
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--baseline",
                "/tmp/baseline.py",
                "--correctness-only",
                "--baseline-runtime-override",
                relative,
                "--baseline-runtime-override",
                relative,
            ]
        )


def test_parse_args_accepts_explicit_shared_gpu_diagnostic_mode():
    args = _parse_args(["--baseline", "/tmp/baseline.py", "--diagnostic-shared-gpu"])
    assert args.diagnostic_shared_gpu is True
    assert args.exclusive_gpu is False
    assert args.correctness_only is False


def test_parse_args_preserves_formal_exclusive_gpu_mode():
    args = _parse_args(["--baseline", "/tmp/baseline.py", "--exclusive-gpu"])
    assert args.exclusive_gpu is True
    assert args.diagnostic_shared_gpu is False


def test_parse_args_still_rejects_implicit_timing_mode():
    with pytest.raises(SystemExit):
        _parse_args(["--baseline", "/tmp/baseline.py"])


@pytest.mark.parametrize("conflicting_flag", ("--correctness-only", "--exclusive-gpu"))
def test_parse_args_rejects_shared_gpu_diagnostic_mode_conflicts(conflicting_flag):
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--baseline",
                "/tmp/baseline.py",
                "--diagnostic-shared-gpu",
                conflicting_flag,
            ]
        )


def test_shared_gpu_diagnostic_report_can_never_claim_acceptance():
    fields = _acceptance_report_fields(observed_passed=True, diagnostic_shared_gpu=True)
    assert fields == {"acceptance": False, "observed_passed": True, "passed": False}
    assert _timing_status(correctness_only=False, diagnostic_shared_gpu=True) == "diagnostic-shared-gpu"
    performance_fields = _performance_report_fields(
        {"backward": {"passed": True}},
        observed_passed=True,
        diagnostic_shared_gpu=True,
    )
    assert "performance_gates" not in performance_fields
    assert performance_fields["observed_performance_gates"]["backward"]["passed"] is True
    assert performance_fields["acceptance"] is False
    assert performance_fields["passed"] is False


def test_formal_report_fields_and_timing_status_remain_unchanged():
    assert _acceptance_report_fields(observed_passed=True, diagnostic_shared_gpu=False) == {"passed": True}
    assert _timing_status(correctness_only=False, diagnostic_shared_gpu=False) == "measured"
    assert _timing_status(correctness_only=True, diagnostic_shared_gpu=False) == "skipped"
    performance_fields = _performance_report_fields(
        {"backward": {"passed": True}},
        observed_passed=True,
        diagnostic_shared_gpu=False,
    )
    assert performance_fields == {
        "performance_gates": {"backward": {"passed": True}},
        "passed": True,
    }


def test_default_identity_rejects_identical_backward(identity_repos):
    with pytest.raises(RuntimeError, match="byte-identical"):
        _collect_code_identity(identity_repos["baseline_backward"])


def test_default_identity_rejects_shared_runtime_mismatch(identity_repos):
    identity_repos["baseline_backward"].write_text("different backward\n")
    (identity_repos["baseline_repo"] / "kernels/moe/sonic_grouped_tn.py").write_text("different TN\n")
    with pytest.raises(RuntimeError, match="byte-identical shared runtime sources"):
        _collect_code_identity(identity_repos["baseline_backward"])


def test_default_identity_report_has_no_override_fields(identity_repos):
    identity_repos["baseline_backward"].write_text("different backward\n")
    identity = _collect_code_identity(identity_repos["baseline_backward"])
    assert identity["shared_runtime_sources_identical"] is True
    assert "baseline_runtime_overrides" not in identity


def test_runtime_override_permits_identical_backward_and_only_selected_mismatch(identity_repos):
    relative = "kernels/moe/sonic_grouped_tn.py"
    (identity_repos["baseline_repo"] / relative).write_text("zero-NT baseline TN\n")
    identity = _collect_code_identity(identity_repos["baseline_backward"], [relative])
    overrides = identity["baseline_runtime_overrides"]
    assert overrides["requested"] == [relative]
    assert overrides["sonic_backward_identical"] is True
    assert overrides["sources"][relative]["identical"] is False
    assert identity["shared_runtime_sources_identical"] is False


def test_runtime_override_rejects_unlisted_mismatch(identity_repos):
    selected = "kernels/moe/sonic_grouped_tn.py"
    (identity_repos["baseline_repo"] / selected).write_text("different selected runtime\n")
    (identity_repos["baseline_repo"] / TEST_SHARED_SOURCE).write_text("different unlisted runtime\n")
    with pytest.raises(RuntimeError, match="unlisted shared runtime mismatches"):
        _collect_code_identity(identity_repos["baseline_backward"], [selected])


def test_runtime_override_rejects_selected_identical_source(identity_repos):
    relative = "kernels/moe/sonic_grouped_tn.py"
    with pytest.raises(RuntimeError, match="selected runtime overrides are byte-identical"):
        _collect_code_identity(identity_repos["baseline_backward"], [relative])


def test_runtime_override_rejects_missing_selected_source(identity_repos):
    relative = "kernels/moe/sonic_grouped_tn.py"
    (identity_repos["baseline_repo"] / relative).unlink()
    with pytest.raises(FileNotFoundError, match="baseline shared runtime source does not exist"):
        _collect_code_identity(identity_repos["baseline_backward"], [relative])


def test_runtime_override_rejects_same_backward_path(identity_repos):
    relative = "kernels/moe/sonic_grouped_tn.py"
    with pytest.raises(RuntimeError, match="resolve to the same file"):
        _collect_code_identity(identity_repos["candidate_backward"], [relative])


def test_runtime_override_identity_order_is_stable(identity_repos):
    requested = (
        "kernels/moe/sonic_grouped_tn.py",
        "kernels/moe/grouped_da_gfx950.py",
    )
    for relative in requested:
        (identity_repos["baseline_repo"] / relative).write_text(f"different {relative}\n")
    identity = _collect_code_identity(identity_repos["baseline_backward"], list(requested))
    assert identity["baseline_runtime_overrides"]["requested"] == [
        "kernels/moe/grouped_da_gfx950.py",
        "kernels/moe/sonic_grouped_tn.py",
    ]


def test_runtime_override_rejects_mutation_between_identity_and_load(identity_repos):
    relative = "kernels/moe/sonic_grouped_tn.py"
    baseline_source = identity_repos["baseline_repo"] / relative
    baseline_source.write_text("VALUE = 1\n")
    identity = _collect_code_identity(identity_repos["baseline_backward"], [relative])
    baseline_source.write_text("VALUE = 2\n")

    with pytest.raises(RuntimeError, match="changed after code identity collection"):
        _load_baseline_runtime_overrides(
            identity_repos["baseline_repo"],
            (relative,),
            expected_sources=identity["baseline_runtime_overrides"]["sources"],
        )


def test_baseline_backward_loader_rejects_mutation_after_identity(tmp_path):
    baseline_path = tmp_path / "sonic_backward.py"
    baseline_path.write_text("def sonic_moe_backward():\n    return 1\n")
    expected = _file_identity(baseline_path)
    baseline_path.write_text("def sonic_moe_backward():\n    return 2\n")
    with pytest.raises(RuntimeError, match="baseline sonic_backward.py changed after code identity collection"):
        _load_isolated_baseline(baseline_path, expected_identity=expected)


def test_private_runtime_loader_supports_relative_import_without_replacing_canonical(tmp_path):
    relative = "kernels/moe/grouped_da_gfx950.py"
    canonical_name = acceptance._runtime_module_name(relative)
    canonical = importlib.import_module(canonical_name)
    baseline_path = tmp_path / "grouped_da_gfx950.py"
    baseline_path.write_bytes((Path(acceptance.__file__).parents[1] / relative).read_bytes())

    private = _load_private_runtime_module(relative, baseline_path)
    try:
        assert private.__name__.startswith("kernels.moe._e896_baseline_runtime_grouped_da_gfx950_")
        assert private.compile_grouped_da_gfx950 is not canonical.compile_grouped_da_gfx950
        assert sys.modules[canonical_name] is canonical
        assert sys.modules[private.__name__] is private
    finally:
        sys.modules.pop(private.__name__, None)


@pytest.mark.parametrize("relative", BASELINE_RUNTIME_OVERRIDE_CHOICES)
def test_symbol_patching_uses_exact_declared_symbol_set(tmp_path, relative):
    baseline = _baseline_bindings(relative)
    unrelated = object()
    baseline.unrelated = unrelated
    private = _write_private_runtime(tmp_path, relative)

    report = _patch_baseline_runtime_symbols(baseline, {relative: private})

    assert baseline.unrelated is unrelated
    assert [entry["symbol"] for entry in report[relative]["patched_symbols"]] == list(
        BASELINE_RUNTIME_OVERRIDE_SYMBOLS[relative]
    )
    for symbol in BASELINE_RUNTIME_OVERRIDE_SYMBOLS[relative]:
        assert getattr(baseline, symbol) is getattr(private, symbol)


def test_grouped_tn_override_patches_all_seven_zero_nt_runtime_symbols(tmp_path):
    relative = "kernels/moe/sonic_grouped_tn.py"
    baseline = _baseline_bindings(relative)
    private = _write_private_runtime(tmp_path, relative)

    report = _patch_baseline_runtime_symbols(baseline, {relative: private})[relative]

    assert len(report["patched_symbols"]) == 7
    assert report["canonical_module"] == "kernels.moe.sonic_grouped_tn"
    assert report["private_module"] == private.__name__
    for entry in report["patched_symbols"]:
        assert entry["original"]["source"].endswith("kernels/moe/sonic_grouped_tn.py")
        assert entry["replacement"]["source"] == str(Path(private.__file__).resolve())


def test_symbol_patching_rejects_missing_private_symbol_without_partial_mutation(tmp_path):
    relative = "kernels/moe/sonic_grouped_tn.py"
    missing = BASELINE_RUNTIME_OVERRIDE_SYMBOLS[relative][-1]
    baseline = _baseline_bindings(relative)
    originals = {symbol: getattr(baseline, symbol) for symbol in BASELINE_RUNTIME_OVERRIDE_SYMBOLS[relative]}
    private = _write_private_runtime(tmp_path, relative, omit=missing)

    with pytest.raises(RuntimeError, match=f"missing symbol {missing}"):
        _patch_baseline_runtime_symbols(baseline, {relative: private})

    assert all(getattr(baseline, symbol) is original for symbol, original in originals.items())


def test_symbol_patching_rejects_missing_baseline_binding(tmp_path):
    relative = "kernels/moe/grouped_da_gfx950.py"
    symbol = BASELINE_RUNTIME_OVERRIDE_SYMBOLS[relative][0]
    baseline = _baseline_bindings(relative)
    delattr(baseline, symbol)
    private = _write_private_runtime(tmp_path, relative)
    with pytest.raises(RuntimeError, match=f"missing imported runtime symbol {symbol}"):
        _patch_baseline_runtime_symbols(baseline, {relative: private})


def test_symbol_patching_rejects_wrong_origin_global(tmp_path):
    relative = "kernels/moe/grouped_da_gfx950.py"
    symbol = BASELINE_RUNTIME_OVERRIDE_SYMBOLS[relative][0]
    baseline = _baseline_bindings(relative)
    baseline.compile_grouped_da_gfx950 = lambda: None
    private = _write_private_runtime(tmp_path, relative)
    with pytest.raises(RuntimeError, match="was not imported from the expected canonical module"):
        _patch_baseline_runtime_symbols(baseline, {relative: private})
    assert getattr(baseline, symbol) is not getattr(private, symbol)


def test_symbol_patching_rejects_wrong_origin_replacement(tmp_path):
    relative = "kernels/moe/grouped_da_gfx950.py"
    symbol = BASELINE_RUNTIME_OVERRIDE_SYMBOLS[relative][0]
    baseline = _baseline_bindings(relative)
    original = getattr(baseline, symbol)
    private = _write_private_runtime(tmp_path, relative)
    private.compile_grouped_da_gfx950 = original
    with pytest.raises(RuntimeError, match="private baseline runtime symbol .* has the wrong origin"):
        _patch_baseline_runtime_symbols(baseline, {relative: private})
    assert getattr(baseline, symbol) is original


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
    baseline = _hostless_launches(
        expert_histogram_sequences=1,
        compact_descriptor_builder_sequences=1,
        sorter_backward_metadata_calls=0,
    )
    gate = _evaluate_launch_topology("incremental", baseline, _hostless_launches())
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
        ({"expert_histogram_sequences": 0}, {}),
        ({"compact_descriptor_builder_sequences": 0}, {}),
        ({"sorter_calls": 0}, {}),
        ({}, {"expert_histogram_sequences": 1}),
        ({}, {"compact_descriptor_builder_sequences": 1}),
        ({}, {"sorter_calls": 0}),
        ({}, {"sorter_backward_metadata_calls": 0}),
    ),
)
def test_incremental_launch_gate_rejects_required_topology_violations(
    baseline_overrides,
    candidate_overrides,
):
    baseline_values = {
        "expert_histogram_sequences": 1,
        "compact_descriptor_builder_sequences": 1,
        "sorter_backward_metadata_calls": 0,
    }
    baseline_values.update(baseline_overrides)
    baseline = _hostless_launches(**baseline_values)
    candidate = _hostless_launches(**candidate_overrides)
    assert not _evaluate_launch_topology("incremental", baseline, candidate)["passed"]


def test_legacy_launch_gate_still_requires_legacy_baseline_work():
    assert _evaluate_launch_topology("legacy-vs-hostless", _legacy_launches(), _hostless_launches())["passed"]
    assert not _evaluate_launch_topology(
        "legacy-vs-hostless",
        _hostless_launches(),
        _hostless_launches(),
    )["passed"]
