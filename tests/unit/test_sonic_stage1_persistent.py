# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CPU policy tests for the opt-in gfx950 SonicMoE Stage-1 scheduler."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from kernels.moe.moe_2stage_a16wmix.gemm1 import (
    NUM_CU,
    compile_gemm1_a16w4_port,
    gemm1_a16w4_grid,
)
from kernels.moe.sonic import (
    SonicMoEConfig,
    _get_stage1_launcher,
    _get_stage1_training_launcher,
    _validate_stage1_persistent_runtime,
)
from kernels.moe.sonic_autotune import _config_tuning_dict
from tools.accept_sonic_e896_forward import (
    PROFILE_BY_NAME,
    _resolved_profile_config,
    _static_topology,
)

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


def _e896_config(tile_m: int = 64, *, persistent: bool = True, **overrides) -> SonicMoEConfig:
    values = {
        "hidden_size": 3584,
        "intermediate_size": 512,
        "num_experts": 896,
        "top_k": 16,
        "tile_m": tile_m,
        "tile_n": 128,
        "tile_k": 128,
        "down_tile_m": tile_m,
        "down_tile_n": 128,
        "down_tile_k": 128,
        "renormalize": False,
        "persistent_stage1": persistent,
    }
    values.update(overrides)
    return SonicMoEConfig(**values)


def _persistent_logical_visits(bound: int, grid: int) -> list[int]:
    visits = []
    for block in range(grid):
        if block < bound:
            visits.append(block)
        visits.extend(range(block + grid, bound, grid))
    return visits


def _xcd_reference(pid: int, bound: int, swizzle: int, num_n_blocks: int) -> int:
    """Integer mirror of gemm1's runtime XCD/M-group bijection."""

    if swizzle <= 0:
        return pid
    quotient, remainder = divmod(bound, 8)
    xcd = pid % 8
    workgroup = xcd * quotient + min(xcd, remainder) + pid // 8
    group_span = swizzle * num_n_blocks
    group_id = workgroup // group_span
    first_m = group_id * swizzle
    total_m = bound // num_n_blocks
    group_m = min(total_m - first_m, swizzle)
    within_group = workgroup % group_span
    m_block = first_m + within_group % group_m
    n_block = within_group // group_m
    return m_block * num_n_blocks + n_block


def test_stage1_persistent_grid_cap_and_threshold() -> None:
    kwargs = {"INTER": 512, "TILE_N": 128}
    assert gemm1_a16w4_grid(64, max_m_blocks=17, persist=True, **kwargs) == 68
    assert gemm1_a16w4_grid(64, max_m_blocks=NUM_CU, persist=True, **kwargs) == NUM_CU * 4
    assert gemm1_a16w4_grid(64, max_m_blocks=NUM_CU + 1, persist=True, **kwargs) == NUM_CU
    assert gemm1_a16w4_grid(64, max_m_blocks=NUM_CU + 1, persist=False, **kwargs) == (NUM_CU + 1) * 4


@pytest.mark.parametrize("tile_m", (64, 80, 96, 112))
def test_stage1_persistent_config_accepts_only_audited_bm_profiles(tile_m: int) -> None:
    config = _e896_config(tile_m)
    assert config.persistent_stage1
    assert fields(SonicMoEConfig)[-1].name == "persistent_stage1"


def test_stage1_persistent_config_rejects_unvalidated_static_shapes() -> None:
    with pytest.raises(TypeError, match="persistent_stage1 must be bool"):
        _e896_config(persistent=1)
    with pytest.raises(ValueError, match="H3584/I512/E896/K16"):
        _e896_config(num_experts=64)
    with pytest.raises(ValueError, match="validated Stage-1 BM"):
        _e896_config(tile_m=128)
    with pytest.raises(ValueError, match="BF16 SwiGLU"):
        _e896_config(activation="relu")
    with pytest.raises(ValueError, match="BF16 SwiGLU"):
        _e896_config(compute_dtype="fp16")


def test_stage1_persistent_runtime_gate_is_exact() -> None:
    config = _e896_config()
    _validate_stage1_persistent_runtime(config, 4096, "gfx950:sramecc+:xnack-")
    with pytest.raises(ValueError, match="restricted to T4096"):
        _validate_stage1_persistent_runtime(config, 4095, "gfx950")
    with pytest.raises(RuntimeError, match="requires gfx950"):
        _validate_stage1_persistent_runtime(config, 4096, "gfx942")

    # The default path retains the established shape/architecture policy.
    _validate_stage1_persistent_runtime(replace(config, persistent_stage1=False), 7, "gfx942")


def test_stage1_persistent_schedule_and_xcd_map_cover_every_tile_once() -> None:
    grid = NUM_CU
    num_n_blocks = 4
    # Exercise actual E896-like work, a non-integral final persistent round,
    # and a real-work bound smaller than the CU-capped launch.
    for bound in (3584, 2764, 68):
        logical = _persistent_logical_visits(bound, grid)
        assert len(logical) == bound
        assert sorted(logical) == list(range(bound))
        for swizzle in (0, 1, 8):
            mapped = [_xcd_reference(pid, bound, swizzle, num_n_blocks) for pid in logical]
            assert sorted(mapped) == list(range(bound))


def test_stage1_persistent_compile_schedule_is_mutually_exclusive() -> None:
    common = {
        "BM": 64,
        "D_HIDDEN": 128,
        "D_INTER": 128,
        "NE": 1,
        "TOPK": 1,
        "TILE_N": 128,
        "TILE_K": 128,
        "w_dtype": "bf16",
        "a_dtype": "bf16",
        "logical_dense_weight": True,
        "persist": True,
    }
    with pytest.raises(AssertionError, match="mutually exclusive"):
        compile_gemm1_a16w4_port(expert_grid=True, **common)
    with pytest.raises(AssertionError, match="mutually exclusive"):
        compile_gemm1_a16w4_port(compact_grid=True, **common)


def test_stage1_persistent_reaches_inference_and_training_launchers(monkeypatch) -> None:
    import kernels.moe.sonic as sonic

    calls = []

    def fake_compile(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(sonic, "compile_gemm1_a16w4_port", fake_compile)
    _get_stage1_launcher.cache_clear()
    _get_stage1_training_launcher.cache_clear()
    try:
        ordinary = _e896_config(persistent=False)
        candidate = replace(ordinary, persistent_stage1=True)
        ordinary_launcher = _get_stage1_launcher(ordinary, 0, "bf16", False, 0)
        persistent_launcher = _get_stage1_launcher(candidate, 0, "bf16", False, 0)
        training_launcher = _get_stage1_training_launcher(
            candidate,
            0,
            False,
            True,
            0,
        )

        assert ordinary_launcher is not persistent_launcher
        assert training_launcher is not persistent_launcher
        assert [call["persist"] for call in calls] == [False, True, True]
        assert calls[-1]["store_route_preactivation"] is True
        assert _get_stage1_launcher.cache_info().currsize == 2
        assert _get_stage1_training_launcher.cache_info().currsize == 1
    finally:
        _get_stage1_launcher.cache_clear()
        _get_stage1_training_launcher.cache_clear()


def test_stage1_persistent_enters_autotune_fingerprint() -> None:
    ordinary = _e896_config(persistent=False)
    candidate = replace(ordinary, persistent_stage1=True)
    assert _config_tuning_dict(ordinary)["persistent_stage1"] is False
    assert _config_tuning_dict(candidate)["persistent_stage1"] is True
    assert _config_tuning_dict(ordinary) != _config_tuning_dict(candidate)


def test_e896_acceptance_profiles_report_actual_stage1_launch_grid() -> None:
    assert {"stage1-persistent", "m80-stage1-persistent"} <= PROFILE_BY_NAME.keys()
    for name in ("stage1-persistent", "m80-stage1-persistent"):
        config = _resolved_profile_config(name)
        assert config["persistent_stage1"] is True
        assert config["persistent_stage2"] is False
        for case in ("balanced", "hot16"):
            stage1 = _static_topology(config, case)["stage1"]
            assert stage1["persistent"] is True
            assert stage1["full_launch_grid"] > NUM_CU * 4
            assert stage1["launch_grid"] == NUM_CU
