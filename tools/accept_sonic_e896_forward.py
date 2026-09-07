#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Reproducible gfx950 acceptance sweep for the E896 SonicMoE forward.

The contract is deliberately fixed to the default adapter shape:
T4096/H3584/I512/E896/K16, dense BF16 SwiGLU, no bias, and fixed top-k
routing.  Both ``forward_topk`` and retained-state ``forward_topk_training``
can be checked with balanced and hot-16 routing.

The profile catalog is curated in stages instead of being a Cartesian product:
route-M choices, BN/BK, Stage-2 pipeline depth, XCD/cache policy, persistence,
and deterministic reduce output are isolated before a combined candidate is
offered.  The baseline and candidates use the exact same runtime source; every
report records source and canonical-config SHA256 fingerprints.

No timing is allowed unless ``--exclusive-gpu`` is supplied.  On a shared
machine, inspect the complete allocation-free plan with ``--list-profiles``.
``--correctness-only`` runs production-size numerical checks (and, unless
disabled, peak-memory samples) but no event timing.

Examples
--------
Allocation-free static inspection::

    python tools/accept_sonic_e896_forward.py --list-profiles --suite extended

Correctness on a gfx950, without timing::

    PYTHONPATH=. python tools/accept_sonic_e896_forward.py \
        --correctness-only --profiles pipeline2 xcd8-cached

Formal paired acceptance on an independently reserved gfx950::

    PYTHONPATH=. python tools/accept_sonic_e896_forward.py \
        --exclusive-gpu --profiles full-candidate --pairs 11 \
        --output /tmp/sonic-e896-forward.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

TOKENS = 4096
HIDDEN = 3584
INTERMEDIATE = 512
EXPERTS = 896
TOPK = 16
ROUTING_CASES = ("balanced", "hot16")
APIS = ("inference", "training")
GFX950_PERSISTENT_GRID_CAP = 256
GFX950_LDS_BYTES = 160 * 1024
CHUNK_ELEMENTS = 8 * 1024 * 1024
WORKSPACE_TENSOR_FIELDS = (
    "sorted_token_ids",
    "sorted_weights",
    "sorted_expert_ids",
    "num_valid_ids",
    "sorting_workspace",
    "expert_frequency",
    "router_topk_weights",
    "router_topk_ids",
    "router_topk_expert_indices",
    "intermediate",
    "route_output",
    "output",
)
RESOURCE_METADATA_FIELDS = (
    "agpr_count",
    "group_segment_fixed_size",
    "private_segment_fixed_size",
    "sgpr_count",
    "sgpr_spill_count",
    "vgpr_count",
    "vgpr_spill_count",
    "wavefront_size",
)
RESOURCE_GATE_FIELDS = (
    "group_segment_fixed_size",
    "private_segment_fixed_size",
    "sgpr_spill_count",
    "vgpr_spill_count",
)
CODEGEN_ENVIRONMENT_VARIABLES = (
    "FLYDSL_GPU_ARCH",
    "HSA_OVERRIDE_GFX_VERSION",
    "FLYDSL_A16WMIX_FORCE_K16",
    "FLYDSL_DUMP_IR",
    "FLYDSL_DUMP_DIR",
    "FLYDSL_COMPILE_OPT_LEVEL",
    "FLYDSL_COMPILE_BACKEND",
    "FLYDSL_COMPILE_LLVM_DIR",
    "FLYDSL_DEBUG_ENABLE_DEBUG_INFO",
    "FLYDSL_EXTRA_SOURCE_DIRS",
)
STATIC_TO_TORCH_DTYPE = {
    "bf16": "torch.bfloat16",
    "fp16": "torch.float16",
    "float32": "torch.float32",
    "int32": "torch.int32",
}

OUTPUT_LIMITS = {
    "relative_l2": 3.0e-2,
    "max_abs": 5.0e-2,
    "cosine_min": 0.999,
}
STATE_LIMITS = {
    "relative_l2": 3.0e-2,
    "max_abs": 5.0e-2,
    "cosine_min": 0.999,
}
WEIGHT_COMPATIBILITY_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "num_experts",
    "activation",
    "compute_dtype",
)

BASE_CONFIG: dict[str, Any] = {
    "hidden_size": HIDDEN,
    "intermediate_size": INTERMEDIATE,
    "num_experts": EXPERTS,
    "top_k": TOPK,
    "tile_m": 64,
    "tile_n": 128,
    "tile_k": 128,
    "down_tile_m": 64,
    "down_tile_n": 128,
    "down_tile_k": 128,
    "renormalize": False,
    "stage1_b_cache_mod": None,
    "stage2_b_cache_mod": None,
    "stage1_xcd_swizzle": 0,
    "stage1_k_wave": 1,
    "stage2_xcd_swizzle": 1,
    "waves_per_eu": None,
    "persistent_stage1": False,
    "persistent_stage2": False,
    "stage2_pipeline_stages": None,
    "stage2_output_mode": "atomic",
    "activation": "swiglu",
    "compute_dtype": "bf16",
}

RUNTIME_SOURCE_FILES = (
    "kernels/common/buffer_ops.py",
    "kernels/common/kernels_common.py",
    "kernels/common/layout_utils.py",
    "kernels/common/tensor_shim.py",
    "kernels/moe/moe_2stage_a16wmix/gemm1.py",
    "kernels/moe/moe_2stage_a16wmix/gemm2.py",
    "kernels/moe/moe_2stage_a16wmix/utils.py",
    "kernels/moe/moe_gemm_2stage/moe_reduce.py",
    "kernels/moe/moe_sorting_kernel.py",
    "kernels/moe/sonic.py",
    "kernels/moe/topk_gating_softmax_kernel.py",
)


@dataclass(frozen=True)
class Profile:
    name: str
    phase: str
    parent: str | None
    description: str
    overrides: dict[str, Any]


PROFILES = (
    Profile(
        "baseline",
        "0-baseline",
        None,
        "Adapter-equivalent conservative M64/BN128/BK128 serial atomic profile.",
        {},
    ),
    Profile(
        "m16-equal",
        "1-route-m",
        "baseline",
        "Minimize balanced E896 padding with equal BM16 and four-way Stage-1 split-K.",
        {"tile_m": 16, "down_tile_m": 16, "stage1_k_wave": 4},
    ),
    Profile(
        "m32-equal",
        "1-route-m",
        "baseline",
        "Equal BM32 with two-way Stage-1 split-K.",
        {"tile_m": 32, "down_tile_m": 32, "stage1_k_wave": 2},
    ),
    Profile(
        "m32-down128",
        "1-route-m",
        "baseline",
        "BM32 Stage 1 and BM128 Stage 2 sharing a 128-row route tile.",
        {"tile_m": 32, "down_tile_m": 128, "stage1_k_wave": 2},
    ),
    Profile(
        "m64-down128",
        "1-route-m",
        "baseline",
        "Keep baseline Stage 1 and widen Stage 2 to BM128.",
        {"down_tile_m": 128},
    ),
    Profile(
        "m80-equal",
        "1-route-m",
        "baseline",
        "Distribution-aware BM80: one balanced expert tile with only 1.09375x padding.",
        {"tile_m": 80, "down_tile_m": 80},
    ),
    Profile(
        "m96-equal",
        "1-route-m",
        "baseline",
        "Distribution-aware BM96 alternative for balanced and hot-16 routing.",
        {"tile_m": 96, "down_tile_m": 96},
    ),
    Profile(
        "m112-equal",
        "1-route-m",
        "baseline",
        "Distribution-aware BM112 alternative between BM96 and BM128.",
        {"tile_m": 112, "down_tile_m": 112},
    ),
    Profile(
        "m128-equal",
        "1-route-m",
        "baseline",
        "Throughput-oriented equal BM128 profile.",
        {"tile_m": 128, "down_tile_m": 128},
    ),
    Profile(
        "bn256-bk64",
        "2-bn-bk",
        "m64-down128",
        "Dense-A16 BN256/BK64 Stage 1 paired with BN128/BK64 Stage 2.",
        {
            "tile_n": 256,
            "tile_k": 64,
            "down_tile_n": 128,
            "down_tile_k": 64,
        },
    ),
    Profile(
        "pipeline2",
        "3-pipeline",
        "baseline",
        "Isolate the two-stage Stage-2 A-LDS pipeline on the baseline tiles.",
        {"stage2_pipeline_stages": 2},
    ),
    Profile(
        "bn256-bk64-pipeline2",
        "3-pipeline",
        "bn256-bk64",
        "Apply the two-stage Stage-2 pipeline after the BN/BK change.",
        {"stage2_pipeline_stages": 2},
    ),
    Profile(
        "xcd8-cached",
        "4-locality",
        "bn256-bk64",
        "Use XCD=8 for both stages and explicitly cache both expert weights.",
        {
            "stage1_xcd_swizzle": 8,
            "stage2_xcd_swizzle": 8,
            "stage1_b_cache_mod": 0,
            "stage2_b_cache_mod": 0,
        },
    ),
    Profile(
        "non-temporal",
        "4-locality",
        "bn256-bk64",
        "Isolate non-temporal expert-weight loads for both stages.",
        {"stage1_b_cache_mod": 2, "stage2_b_cache_mod": 2},
    ),
    Profile(
        "stage1-persistent",
        "5-persistent",
        "baseline",
        "Cap only the Stage-1 launch and consume the sorter-produced real-work bound on device.",
        {"persistent_stage1": True},
    ),
    Profile(
        "m80-stage1-persistent",
        "5-persistent",
        "m80-equal",
        "Combine distribution-aware BM80 with the Stage-1 persistent route grid.",
        {"persistent_stage1": True},
    ),
    Profile(
        "persistent",
        "5-persistent",
        "xcd8-cached",
        "Cap Stage-2 launch to the gfx950 persistent grid over actual work tiles.",
        {"persistent_stage2": True},
    ),
    Profile(
        "reduce-output",
        "6-output",
        "xcd8-cached",
        "Replace weighted BF16 atomics with fixed-slot projection plus FP32 reduction.",
        {"stage2_output_mode": "reduce"},
    ),
    Profile(
        "full-candidate",
        "7-combined",
        "persistent",
        "Combine BN256/BK64, XCD/cache, persistence, and the two-stage pipeline.",
        {"stage2_pipeline_stages": 2},
    ),
)

PROFILE_BY_NAME = {profile.name: profile for profile in PROFILES}
CANDIDATE_NAMES = tuple(profile.name for profile in PROFILES if profile.name != "baseline")
SUITES = {
    "smoke": ("pipeline2",),
    "core": (
        "m16-equal",
        "m32-down128",
        "m64-down128",
        "m80-equal",
        "m96-equal",
        "m112-equal",
        "m128-equal",
        "bn256-bk64",
        "pipeline2",
    ),
    "locality": (
        "xcd8-cached",
        "non-temporal",
        "stage1-persistent",
        "persistent",
    ),
    "persistent": (
        "stage1-persistent",
        "m80-stage1-persistent",
        "persistent",
    ),
    "output": ("reduce-output",),
    "extended": CANDIDATE_NAMES,
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _dict_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_command(repo: Path, *args: str) -> tuple[dict[str, Any], bytes]:
    """Run one auditable Git query with an explicit safe-directory override."""

    resolved_repo = repo.resolve()
    command = (
        "git",
        "-c",
        f"safe.directory={resolved_repo}",
        "-C",
        str(resolved_repo),
        *args,
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        return (
            {
                "command": list(command),
                "returncode": completed.returncode,
                "stdout": stdout.decode("utf-8", errors="backslashreplace"),
                "stderr": stderr.decode("utf-8", errors="backslashreplace"),
            },
            stdout,
        )
    except OSError as error:
        return (
            {
                "command": list(command),
                "returncode": None,
                "stdout": "",
                "stderr": f"{type(error).__name__}: {error}",
            },
            b"",
        )


def _codegen_environment() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in CODEGEN_ENVIRONMENT_VARIABLES}


def _resolved_profile_config(name: str, stack: tuple[str, ...] = ()) -> dict[str, Any]:
    if name not in PROFILE_BY_NAME:
        raise KeyError(f"unknown profile {name!r}")
    if name in stack:
        raise RuntimeError(f"profile parent cycle: {' -> '.join((*stack, name))}")
    profile = PROFILE_BY_NAME[name]
    if profile.parent is None:
        resolved = dict(BASE_CONFIG)
    else:
        resolved = _resolved_profile_config(profile.parent, (*stack, name))
    resolved.update(profile.overrides)
    return resolved


def _effective_cache_mod(config: dict[str, Any], stage: int) -> int:
    explicit = config[f"stage{stage}_b_cache_mod"]
    if explicit is not None:
        return int(explicit)
    if stage == 1:
        return 2 if 16 <= TOKENS <= 1024 else 0
    return 0 if TOKENS <= 16 or TOKENS >= 2048 else 2


def _effective_stage2_stages(config: dict[str, Any]) -> int:
    requested = config["stage2_pipeline_stages"]
    if requested is not None and type(requested) is not int:
        raise TypeError(f"stage2_pipeline_stages must be None or an integer, got {type(requested).__name__}")
    if requested not in (None, 1, 2):
        raise ValueError(f"stage2_pipeline_stages must be None, 1, or 2, got {requested!r}")
    if requested is None:
        auto_eligible = (
            HIDDEN == 4096
            and INTERMEDIATE == 2048
            and EXPERTS == 64
            and TOPK == 8
            and config["down_tile_m"] == 128
            and config["down_tile_n"] == 128
            and config["down_tile_k"] == 64
            and math.lcm(config["tile_m"], config["down_tile_m"]) == 128
            and config["stage2_xcd_swizzle"] == 8
            and config["stage2_b_cache_mod"] in (None, 0)
            and config["waves_per_eu"] is None
            and not config["persistent_stage2"]
            and config["stage2_output_mode"] == "atomic"
            and config["compute_dtype"] == "bf16"
        )
        requested = 2 if auto_eligible else 1
    if config["stage2_output_mode"] != "atomic" or config["compute_dtype"] != "bf16":
        return 1
    k_tiles = INTERMEDIATE // int(config["down_tile_k"])
    force_k16 = os.environ.get("FLYDSL_A16WMIX_FORCE_K16", "0") not in ("0", "", "false", "False")
    return 2 if requested == 2 and k_tiles > 1 and not force_k16 else 1


def _static_lds_usage(config: dict[str, Any]) -> dict[str, int]:
    bm1 = int(config["tile_m"])
    bn1 = int(config["tile_n"])
    bk1 = int(config["tile_k"])
    bm2 = int(config["down_tile_m"])
    bn2 = int(config["down_tile_n"])
    bk2 = int(config["down_tile_k"])
    k_wave = int(config["stage1_k_wave"])
    k_tiles_per_wave = HIDDEN // (k_wave * bk1)
    stage1_stages = 2 if k_tiles_per_wave > 1 else 1
    stage1_a_lds = k_wave * stage1_stages * bm1 * bk1 * 2
    stage1_reduce_lds = 0
    if k_wave > 1:
        n_waves = 4 // k_wave
        acc_n = (bn1 // n_waves) // 16
        m_repeat = bm1 // 16
        stage1_reduce_lds = 4 * (acc_n * m_repeat) * 64 * 4 * 4
    stage1_lds = max(stage1_a_lds, stage1_reduce_lds)
    stage2_stages = _effective_stage2_stages(config)
    stage2_a_lds = stage2_stages * bm2 * bk2 * 2
    stage2_epilogue_lds = bm2 * bn2 * 4
    return {
        "stage1_pipeline_stages": stage1_stages,
        "stage1_a_bytes": stage1_a_lds,
        "stage1_reduction_bytes": stage1_reduce_lds,
        "stage1_total_bytes": stage1_lds,
        "stage2_pipeline_stages": stage2_stages,
        "stage2_a_bytes": stage2_a_lds,
        "stage2_epilogue_bytes": stage2_epilogue_lds,
        "stage2_total_bytes": max(stage2_a_lds, stage2_epilogue_lds),
        "gfx950_limit_bytes": GFX950_LDS_BYTES,
    }


def _validate_static_config(name: str, config: dict[str, Any]) -> None:
    bm1 = int(config["tile_m"])
    bn1 = int(config["tile_n"])
    bk1 = int(config["tile_k"])
    bm2 = int(config["down_tile_m"])
    bn2 = int(config["down_tile_n"])
    bk2 = int(config["down_tile_k"])
    k_wave = int(config["stage1_k_wave"])
    route_m = math.lcm(bm1, bm2)
    errors = []
    if bm1 % 16 or bm2 % 16:
        errors.append("BM must be a multiple of 16")
    if bn1 % 64 or bn2 % 64:
        errors.append("BN must be a multiple of 64")
    if bk1 % 32 or bk2 % 32 or (bk1 & (bk1 - 1)) or (bk2 & (bk2 - 1)):
        errors.append("BK must be a power-of-two multiple of 32")
    if HIDDEN % (k_wave * bk1):
        errors.append("H must be divisible by stage1_k_wave * BK1")
    if INTERMEDIATE % bn1 or INTERMEDIATE % bk2 or HIDDEN % bn2:
        errors.append("N/K dimensions must divide H/I")
    if (bm1 * bk1) % 2048 or (bm2 * bk2) % 2048:
        errors.append("BM * BK must cover integral 4096-byte direct-to-LDS rounds")
    if route_m % bm1 or route_m % bm2:
        errors.append("route M must be divisible by both GEMM M tiles")
    if not isinstance(config["persistent_stage1"], bool):
        errors.append("persistent_stage1 must be bool")
    if config["persistent_stage1"] and bm1 not in (64, 80, 96, 112):
        errors.append("persistent_stage1 BM must be one of 64/80/96/112")

    lds = _static_lds_usage(config)
    if lds["stage1_total_bytes"] > GFX950_LDS_BYTES:
        errors.append(f"Stage 1 LDS {lds['stage1_total_bytes']} exceeds {GFX950_LDS_BYTES}")
    if lds["stage2_total_bytes"] > GFX950_LDS_BYTES:
        errors.append(f"Stage 2 LDS {lds['stage2_total_bytes']} exceeds {GFX950_LDS_BYTES}")
    if errors:
        raise ValueError(f"invalid static profile {name}: " + "; ".join(errors))


def _routing_frequencies(case: str) -> list[int]:
    if case == "balanced":
        quotient, remainder = divmod(TOKENS * TOPK, EXPERTS)
        return [quotient + int(expert < remainder) for expert in range(EXPERTS)]
    if case == "hot16":
        return [TOKENS if expert < TOPK else 0 for expert in range(EXPERTS)]
    raise ValueError(f"unknown routing case {case!r}")


def _static_topology(config: dict[str, Any], case: str) -> dict[str, Any]:
    bm1 = int(config["tile_m"])
    bn1 = int(config["tile_n"])
    bm2 = int(config["down_tile_m"])
    bn2 = int(config["down_tile_n"])
    route_m = math.lcm(bm1, bm2)
    frequencies = _routing_frequencies(case)
    active_experts = sum(frequency > 0 for frequency in frequencies)
    padded_rows = sum(((frequency + route_m - 1) // route_m) * route_m for frequency in frequencies)
    routes = TOKENS * TOPK

    # SonicMoEWorkspace.allocate must reserve for any legal dense fixed-K
    # distribution, not just the particular balanced/hot16 input in this run.
    capacity_active_experts = min(EXPERTS, routes)
    padding_bound = (routes + capacity_active_experts * (route_m - 1)) // route_m
    per_expert_bound = capacity_active_experts * ((TOKENS + route_m - 1) // route_m)
    capacity_route_blocks = min(padding_bound, per_expert_bound)
    capacity_padded_rows = capacity_route_blocks * route_m

    active_stage1_m_blocks = padded_rows // bm1
    active_stage2_m_blocks = padded_rows // bm2
    capacity_stage1_m_blocks = capacity_padded_rows // bm1
    capacity_stage2_m_blocks = capacity_padded_rows // bm2
    active_stage1_workgroups = active_stage1_m_blocks * (INTERMEDIATE // bn1)
    active_stage2_workgroups = active_stage2_m_blocks * (HIDDEN // bn2)
    stage1_full_grid = capacity_stage1_m_blocks * (INTERMEDIATE // bn1)
    if config["persistent_stage1"] and stage1_full_grid > GFX950_PERSISTENT_GRID_CAP * 4:
        stage1_launch_grid = min(stage1_full_grid, GFX950_PERSISTENT_GRID_CAP)
    else:
        stage1_launch_grid = stage1_full_grid
    stage2_full_grid = capacity_stage2_m_blocks * (HIDDEN // bn2)
    if config["persistent_stage2"] and stage2_full_grid > GFX950_PERSISTENT_GRID_CAP * 4:
        stage2_launch_grid = min(stage2_full_grid, GFX950_PERSISTENT_GRID_CAP)
    else:
        stage2_launch_grid = stage2_full_grid

    return {
        "routes": routes,
        "active_experts": active_experts,
        "active_frequency_min": min(frequency for frequency in frequencies if frequency),
        "active_frequency_max": max(frequencies),
        "route_tile_m": route_m,
        "actual_padded_rows": padded_rows,
        "workspace_capacity_padded_rows": capacity_padded_rows,
        "padding_over_routes": padded_rows / routes,
        "route_metadata_blocks": padded_rows // route_m,
        "stage1": {
            "active_m_blocks": active_stage1_m_blocks,
            "active_logical_workgroups": active_stage1_workgroups,
            "capacity_m_blocks": capacity_stage1_m_blocks,
            "full_launch_grid": stage1_full_grid,
            "launch_grid": stage1_launch_grid,
            "persistent": bool(config["persistent_stage1"]),
        },
        "stage2": {
            "active_m_blocks": active_stage2_m_blocks,
            "active_logical_workgroups": active_stage2_workgroups,
            "capacity_m_blocks": capacity_stage2_m_blocks,
            "full_launch_grid": stage2_full_grid,
            "launch_grid": stage2_launch_grid,
            "persistent": bool(config["persistent_stage2"]),
        },
    }


def _static_tensor_record(shape: tuple[int, ...] | None, dtype: str, element_size: int) -> dict[str, Any]:
    if shape is None:
        return {
            "allocated": False,
            "shape": None,
            "dtype": dtype,
            "element_size": element_size,
            "numel": 0,
            "bytes": 0,
        }
    numel = math.prod(shape)
    return {
        "allocated": True,
        "shape": list(shape),
        "dtype": dtype,
        "element_size": element_size,
        "numel": numel,
        "bytes": numel * element_size,
    }


def _gfx950_sorting_workspace_i32(route_tile_m: int) -> int:
    """Mirror the multiphase sorter allocation for this fixed T4096/E896 shape."""

    mesh_stride = ((TOKENS + route_tile_m - 1) // route_tile_m) * route_tile_m
    workspace_mesh_bytes = EXPERTS * mesh_stride
    return (workspace_mesh_bytes + 3) // 4 + (EXPERTS + 1)


def _static_workspace_footprint(config: dict[str, Any]) -> dict[str, Any]:
    """Exact tensors allocated by SonicMoEWorkspace.allocate for fixed top-k."""

    topology = _static_topology(config, "balanced")
    capacity_rows = topology["workspace_capacity_padded_rows"]
    capacity_route_blocks = capacity_rows // topology["route_tile_m"]
    sorting_workspace_i32 = _gfx950_sorting_workspace_i32(topology["route_tile_m"])
    route_output_shape = (TOKENS, TOPK, HIDDEN) if config["stage2_output_mode"] == "reduce" else None
    tensors = {
        "sorted_token_ids": _static_tensor_record((max(1, capacity_rows),), "int32", 4),
        "sorted_weights": _static_tensor_record((max(1, capacity_rows),), "float32", 4),
        "sorted_expert_ids": _static_tensor_record((max(1, capacity_route_blocks),), "int32", 4),
        "num_valid_ids": _static_tensor_record((2,), "int32", 4),
        "sorting_workspace": _static_tensor_record((sorting_workspace_i32,), "int32", 4),
        "expert_frequency": _static_tensor_record((EXPERTS,), "int32", 4),
        "router_topk_weights": _static_tensor_record((TOKENS, TOPK), "float32", 4),
        "router_topk_ids": _static_tensor_record((TOKENS, TOPK), "int32", 4),
        "router_topk_expert_indices": _static_tensor_record((TOKENS, TOPK), "int32", 4),
        "intermediate": _static_tensor_record(
            (max(1, capacity_rows), INTERMEDIATE),
            config["compute_dtype"],
            2,
        ),
        "route_output": _static_tensor_record(route_output_shape, config["compute_dtype"], 2),
        "output": _static_tensor_record((TOKENS, HIDDEN), config["compute_dtype"], 2),
    }
    if tuple(tensors) != WORKSPACE_TENSOR_FIELDS:
        raise AssertionError("static workspace inventory drifted from SonicMoEWorkspace")
    return {
        "storage_accounting": "sum of one owning allocation per listed tensor; absent optional tensors are zero",
        "sorter": {
            "path": "multiphase",
            "mesh_stride_bytes": ((TOKENS + topology["route_tile_m"] - 1) // topology["route_tile_m"])
            * topology["route_tile_m"],
            "workspace_i32_elements": sorting_workspace_i32,
        },
        "tensor_count": sum(entry["allocated"] for entry in tensors.values()),
        "total_owned_bytes": sum(entry["bytes"] for entry in tensors.values()),
        "tensors": tensors,
    }


def _static_footprint(config: dict[str, Any]) -> dict[str, Any]:
    dense_weights = EXPERTS * (2 * INTERMEDIATE * HIDDEN + HIDDEN * INTERMEDIATE) * 2
    input_bytes = TOKENS * HIDDEN * 2
    output_bytes = input_bytes
    retained_state = TOKENS * TOPK * 2 * INTERMEDIATE * 2
    route_output = TOKENS * TOPK * HIDDEN * 2 if config["stage2_output_mode"] == "reduce" else 0
    return {
        "prepared_dense_weight_bytes": dense_weights,
        "hidden_input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "training_retained_state_bytes_per_call": retained_state,
        "reduce_route_output_workspace_bytes": route_output,
        "sonic_moe_workspace": _static_workspace_footprint(config),
    }


def _weight_compatibility_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Fields SonicMoE requires to reuse one prepared dense-weight object."""

    return {name: config[name] for name in WEIGHT_COMPATIBILITY_FIELDS}


def _profile_record(profile: Profile) -> dict[str, Any]:
    config = _resolved_profile_config(profile.name)
    _validate_static_config(profile.name, config)
    parent_config = None if profile.parent is None else _resolved_profile_config(profile.parent)
    changes = {
        key: {"from": parent_config.get(key), "to": value}
        for key, value in config.items()
        if parent_config is not None and parent_config.get(key) != value
    }
    return {
        "name": profile.name,
        "phase": profile.phase,
        "parent": profile.parent,
        "description": profile.description,
        "changes_from_parent": changes,
        "config": config,
        "config_sha256": _dict_sha256(config),
        "prepared_weight_compatibility": {
            "signature": _weight_compatibility_signature(config),
            "matches_baseline": _weight_compatibility_signature(config) == _weight_compatibility_signature(BASE_CONFIG),
        },
        "effective": {
            "route_tile_m": math.lcm(config["tile_m"], config["down_tile_m"]),
            "stage1_b_cache_mod": _effective_cache_mod(config, 1),
            "stage2_b_cache_mod": _effective_cache_mod(config, 2),
            "stage2_pipeline_stages": _effective_stage2_stages(config),
            "static_lds": _static_lds_usage(config),
        },
        "topology": {case: _static_topology(config, case) for case in ROUTING_CASES},
        "static_footprint": _static_footprint(config),
    }


def _collect_source_identity(
    repo: Path,
    *,
    allow_dirty: bool,
    enforce_provenance: bool = True,
) -> dict[str, Any]:
    """Fingerprint runtime sources and retain complete Git command provenance."""

    files: dict[str, dict[str, Any]] = {}
    aggregate = hashlib.sha256()
    source_errors = []
    for relative in (*RUNTIME_SOURCE_FILES, "tools/accept_sonic_e896_forward.py"):
        path = repo / relative
        try:
            digest = _sha256(path)
        except OSError as error:
            files[relative] = {
                "path": str(path.resolve()),
                "sha256": None,
                "error": f"{type(error).__name__}: {error}",
            }
            source_errors.append(relative)
        else:
            files[relative] = {"path": str(path.resolve()), "sha256": digest}
            aggregate.update(relative.encode())
            aggregate.update(bytes.fromhex(digest))

    head_command, head_stdout = _git_command(repo, "rev-parse", "HEAD")
    branch_command, branch_stdout = _git_command(repo, "branch", "--show-current")
    status_command, status_stdout = _git_command(
        repo,
        "status",
        "--short",
        "--untracked-files=all",
    )
    unstaged_command, unstaged_diff = _git_command(repo, "diff", "--binary", "--no-ext-diff", "--")
    staged_command, staged_diff = _git_command(
        repo,
        "diff",
        "--cached",
        "--binary",
        "--no-ext-diff",
        "--",
    )
    untracked_command, untracked_stdout = _git_command(
        repo,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    head = head_stdout.decode("utf-8", errors="replace").strip()
    branch = branch_stdout.decode("utf-8", errors="replace").strip()
    status = status_stdout.decode("utf-8", errors="replace").strip()
    dirty = bool(status)

    dirty_hasher = hashlib.sha256()
    dirty_hasher.update(b"unstaged\0")
    dirty_hasher.update(unstaged_diff)
    dirty_hasher.update(b"staged\0")
    dirty_hasher.update(staged_diff)
    untracked_files = []
    untracked_content_errors = []
    for encoded_relative in sorted(filter(None, untracked_stdout.split(b"\0"))):
        relative = encoded_relative.decode("utf-8", errors="surrogateescape")
        path = repo / relative
        content_hasher = hashlib.sha256()
        content_bytes = 0
        dirty_hasher.update(b"untracked\0")
        dirty_hasher.update(encoded_relative)
        dirty_hasher.update(b"\0")
        try:
            if path.is_symlink():
                content = os.readlink(path).encode("utf-8", errors="surrogateescape")
                kind = "symlink"
                content_hasher.update(content)
                dirty_hasher.update(content)
                content_bytes = len(content)
            else:
                kind = "file"
                with path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        content_hasher.update(chunk)
                        dirty_hasher.update(chunk)
                        content_bytes += len(chunk)
        except OSError as error:
            kind = "unreadable"
            untracked_content_errors.append(relative)
            untracked_files.append(
                {
                    "path": relative,
                    "kind": kind,
                    "bytes": None,
                    "sha256": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        untracked_files.append(
            {
                "path": relative,
                "kind": kind,
                "bytes": content_bytes,
                "sha256": content_hasher.hexdigest(),
            }
        )

    git_commands = {
        "head": head_command,
        "branch": branch_command,
        "status": status_command,
        "unstaged_diff": unstaged_command,
        "staged_diff": staged_command,
        "untracked_files": untracked_command,
    }
    query_success = all(command["returncode"] == 0 for command in git_commands.values())
    checks = {
        "git_queries_succeeded": query_success,
        "head_resolved": head_command["returncode"] == 0 and bool(re.fullmatch(r"[0-9a-fA-F]{40,64}", head)),
        # An empty branch is a valid detached-HEAD provenance state.
        "branch_query_succeeded": branch_command["returncode"] == 0,
        "runtime_sources_readable": not source_errors,
        "untracked_content_hashed": not untracked_content_errors,
        "worktree_clean_or_explicitly_allowed": not dirty or allow_dirty,
    }
    return {
        "repo": str(repo.resolve()),
        "head": head or None,
        "branch": branch,
        "detached_head": branch_command["returncode"] == 0 and not branch,
        "status": status,
        "dirty": dirty,
        "allow_dirty": allow_dirty,
        "enforced": enforce_provenance,
        "git": {
            "safe_directory": str(repo.resolve()),
            "commands": git_commands,
            "working_tree_content_sha256": dirty_hasher.hexdigest(),
            "unstaged_diff_sha256": hashlib.sha256(unstaged_diff).hexdigest(),
            "staged_diff_sha256": hashlib.sha256(staged_diff).hexdigest(),
            "untracked_files": untracked_files,
            "untracked_content_errors": untracked_content_errors,
        },
        "codegen_environment": _codegen_environment(),
        "runtime_sha256": aggregate.hexdigest(),
        "files": files,
        "comparison_model": "config-only; baseline and candidates execute these identical runtime sources",
        "checks": checks,
        "passed": all(checks.values()) if enforce_provenance else True,
    }


def _selected_profiles(args: argparse.Namespace) -> tuple[str, ...]:
    names = tuple(args.profiles) if args.profiles is not None else SUITES[args.suite]
    unique = tuple(dict.fromkeys(names))
    if "baseline" in unique:
        raise ValueError("baseline is implicit and must not be selected as a candidate")
    return unique


def _emit(payload: dict[str, Any], output: Path | None) -> None:
    encoded = json.dumps(payload, indent=2)
    print(encoded)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")


def _plan_payload(args: argparse.Namespace, repo: Path) -> dict[str, Any]:
    selected = _selected_profiles(args)
    records = [_profile_record(profile) for profile in PROFILES]
    weight_reuse_passed = all(record["prepared_weight_compatibility"]["matches_baseline"] for record in records)
    return {
        "schema": "flydsl.sonic_e896_forward_plan.v1",
        "allocation_free": True,
        "contract": {
            "shape": {"T": TOKENS, "H": HIDDEN, "I": INTERMEDIATE, "E": EXPERTS, "K": TOPK},
            "dtype": "bf16",
            "activation": "swiglu",
            "bias": False,
            "training_state_interleaved_w1": True,
        },
        "suite": args.suite if args.profiles is None else None,
        "selected_candidates": list(selected),
        "selected_cases": list(dict.fromkeys(args.cases)),
        "selected_apis": list(dict.fromkeys(args.apis)),
        "suites": {name: list(profiles) for name, profiles in SUITES.items()},
        "profiles": records,
        "prepared_weight_reuse_gate": {
            "fields": list(WEIGHT_COMPATIBILITY_FIELDS),
            "all_profiles_match_baseline": weight_reuse_passed,
        },
        "source_identity": _collect_source_identity(
            repo,
            allow_dirty=True,
            enforce_provenance=False,
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--suite", choices=tuple(SUITES), default="smoke")
    selection.add_argument("--profiles", nargs="+", choices=CANDIDATE_NAMES)
    parser.add_argument("--cases", nargs="+", choices=ROUTING_CASES, default=list(ROUTING_CASES))
    parser.add_argument("--apis", nargs="+", choices=APIS, default=list(APIS))
    parser.add_argument(
        "--list-profiles",
        "--manifest-only",
        "--dry-run",
        dest="list_profiles",
        action="store_true",
        help="emit the complete static plan and exit without importing torch or allocating GPU tensors",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run pure-CPU profile, workspace-accounting, and resource-gate invariants",
    )
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="run numerical/repeatability checks but skip event timing",
    )
    parser.add_argument(
        "--exclusive-gpu",
        action="store_true",
        help="assert the selected gfx950 has been independently verified idle and reserved",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--correctness-repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--pairs",
        type=int,
        default=7,
        help="paired timing rounds; every round executes both ABBA and BAAB",
    )
    parser.add_argument("--peak-samples", type=int, default=2)
    parser.add_argument(
        "--skip-peak-memory",
        action="store_true",
        help="skip peak-memory sampling (normally retained even with --correctness-only)",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.01,
        help="minimum median and paired speedup (default: 1.01, a 1%% net gain)",
    )
    parser.add_argument("--min-paired-win-rate", type=float, default=0.75)
    parser.add_argument(
        "--require-performance",
        action="store_true",
        help="deprecated compatibility flag; timing already requires the performance gate",
    )
    parser.add_argument(
        "--skip-performance-gate",
        action="store_true",
        help="record timing thresholds without making them part of acceptance",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit a dirty worktree while retaining diffs and untracked-content hashes",
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        help="enable FlyDSL IR/ISA dumps and attach matching ISA resource summaries",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.device < 0:
        parser.error("--device must be non-negative")
    if args.correctness_repeats < 2:
        parser.error("--correctness-repeats must be at least 2")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    if args.peak_samples < 1:
        parser.error("--peak-samples must be positive")
    if args.min_speedup <= 0:
        parser.error("--min-speedup must be positive")
    if not 0.0 <= args.min_paired_win_rate <= 1.0:
        parser.error("--min-paired-win-rate must be in [0, 1]")
    if args.list_profiles and args.self_test:
        parser.error("--list-profiles and --self-test are mutually exclusive")
    if args.require_performance and args.correctness_only:
        parser.error("--require-performance cannot be combined with --correctness-only")
    if args.require_performance and args.skip_performance_gate:
        parser.error("--require-performance and --skip-performance-gate are mutually exclusive")
    if not args.list_profiles and not args.self_test and not args.correctness_only and not args.exclusive_gpu:
        parser.error("timing requires --exclusive-gpu after independently reserving the selected GPU")
    return args


def _make_routing(torch, case: str, generator) -> tuple[Any, Any, dict[str, Any]]:
    if case == "balanced":
        ids = (
            torch.arange(TOKENS * TOPK, device="cuda", dtype=torch.int64)
            .remainder(EXPERTS)
            .to(torch.int32)
            .reshape(TOKENS, TOPK)
        )
    elif case == "hot16":
        ids = torch.arange(TOPK, device="cuda", dtype=torch.int32).expand(TOKENS, TOPK).contiguous()
    else:
        raise ValueError(f"unknown routing case {case!r}")
    scores = torch.rand((TOKENS, TOPK), device="cuda", dtype=torch.float32, generator=generator)
    scores /= scores.sum(dim=1, keepdim=True)
    topology = _static_topology(BASE_CONFIG, case)
    return (
        ids,
        scores,
        {
            "routes": topology["routes"],
            "active_experts": topology["active_experts"],
            "active_frequency_min": topology["active_frequency_min"],
            "active_frequency_max": topology["active_frequency_max"],
        },
    )


def _chunked_torch_oracle(
    torch,
    x,
    w1,
    w2,
    ids,
    scores,
    *,
    retain_interleaved_state: bool,
    route_chunk: int = 256,
) -> tuple[Any, Any | None, dict[str, Any]]:
    """Independent expert-major PyTorch oracle without materializing route weights."""

    if route_chunk < 1:
        raise ValueError("route_chunk must be positive")
    flat_ids = ids.reshape(-1)
    flat_scores = scores.reshape(-1)
    output = torch.zeros((TOKENS, HIDDEN), device=x.device, dtype=torch.float32)
    state = (
        torch.empty((TOKENS * TOPK, 2 * INTERMEDIATE), device=x.device, dtype=torch.bfloat16)
        if retain_interleaved_state
        else None
    )
    active_experts = 0
    route_chunks = 0
    with torch.no_grad():
        x_float = x.float()
        expert_ids = [int(expert) for expert in torch.unique(flat_ids).tolist()]
        for expert in expert_ids:
            flat_positions = torch.nonzero(flat_ids == expert, as_tuple=False).flatten()
            if not flat_positions.numel():
                continue
            active_experts += 1
            expert_w1 = w1[expert].float()
            expert_w2 = w2[expert].float()
            for start in range(0, flat_positions.numel(), route_chunk):
                route_chunks += 1
                positions = flat_positions[start : start + route_chunk]
                token_indices = torch.div(positions, TOPK, rounding_mode="floor")
                preactivation = (x_float[token_indices] @ expert_w1.T).to(torch.bfloat16)
                gate, up = preactivation.float().chunk(2, dim=-1)
                intermediate = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
                projection = (intermediate.float() @ expert_w2.T).to(torch.bfloat16)
                output.index_add_(
                    0,
                    token_indices,
                    projection.float() * flat_scores[positions, None],
                )
                if state is not None:
                    state[positions] = torch.stack(
                        preactivation.chunk(2, dim=-1),
                        dim=-1,
                    ).reshape(positions.numel(), 2 * INTERMEDIATE)
            del expert_w1, expert_w2, flat_positions
    return (
        output.to(torch.bfloat16),
        None if state is None else state.view(TOKENS, TOPK, 2 * INTERMEDIATE),
        {
            "implementation": "independent expert-major chunked PyTorch matmul",
            "route_chunk": route_chunk,
            "active_experts": active_experts,
            "route_chunks": route_chunks,
            "accumulation_dtype": "torch.float32",
            "stage_boundaries": "BF16 after W1, activation, and W2; BF16 after FP32 weighted sum",
            "training_state_layout": "[g0,u0,g1,u1,...]" if retain_interleaved_state else None,
        },
    )


def _tensor_metrics(torch, actual, expected, limits: dict[str, float]) -> dict[str, Any]:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError(
            f"tensor contract mismatch: actual={tuple(actual.shape)}/{actual.dtype}, "
            f"expected={tuple(expected.shape)}/{expected.dtype}"
        )
    actual_flat = actual.detach().reshape(-1)
    expected_flat = expected.detach().reshape(-1)
    actual_sq = 0.0
    expected_sq = 0.0
    difference_sq = 0.0
    dot = 0.0
    max_abs = 0.0
    finite = True
    for start in range(0, actual_flat.numel(), CHUNK_ELEMENTS):
        end = min(start + CHUNK_ELEMENTS, actual_flat.numel())
        actual_chunk = actual_flat[start:end].float()
        expected_chunk = expected_flat[start:end].float()
        difference = actual_chunk - expected_chunk
        finite &= bool(torch.isfinite(actual_chunk).all())
        finite &= bool(torch.isfinite(expected_chunk).all())
        actual_sq += float(torch.sum(actual_chunk * actual_chunk))
        expected_sq += float(torch.sum(expected_chunk * expected_chunk))
        difference_sq += float(torch.sum(difference * difference))
        dot += float(torch.sum(actual_chunk * expected_chunk))
        if difference.numel():
            max_abs = max(max_abs, float(difference.abs().max()))
    relative_l2 = (difference_sq / expected_sq) ** 0.5 if expected_sq else difference_sq**0.5
    if actual_sq and expected_sq:
        cosine = dot / math.sqrt(actual_sq * expected_sq)
    else:
        cosine = 1.0 if actual_sq == expected_sq else 0.0
    norm_ratio = math.sqrt(actual_sq / expected_sq) if expected_sq else (1.0 if actual_sq == 0.0 else math.inf)
    passed = bool(
        finite
        and relative_l2 <= limits["relative_l2"]
        and max_abs <= limits["max_abs"]
        and cosine >= limits["cosine_min"]
    )
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "finite": finite,
        "max_abs": max_abs,
        "max_abs_limit": limits["max_abs"],
        "relative_l2": relative_l2,
        "relative_l2_limit": limits["relative_l2"],
        "cosine": cosine,
        "cosine_min": limits["cosine_min"],
        "norm_ratio": norm_ratio,
        "passed": passed,
    }


def _run_api(op, api: str, x, ids, scores, out):
    if api == "inference":
        return op.forward_topk(x, ids, scores, out=out), None
    if api == "training":
        return op.forward_topk_training(
            x,
            ids,
            scores,
            out=out,
            interleaved_w1=True,
        )
    raise ValueError(f"unknown API {api!r}")


def _workspace_owned_memory(workspace, config) -> dict[str, Any]:
    """Count workspace-owned allocations once by untyped-storage base pointer."""

    static = _static_workspace_footprint(_config_dict(config))
    tensors: dict[str, Any] = {}
    storages_by_ptr: dict[int, dict[str, Any]] = {}
    observed_ptrs = set()
    for name in WORKSPACE_TENSOR_FIELDS:
        tensor = getattr(workspace, name)
        expected = static["tensors"][name]
        if tensor is None:
            tensors[name] = {
                "allocated": False,
                "shape": None,
                "dtype": None,
                "logical_bytes": 0,
                "storage_id": None,
                "matches_static": not expected["allocated"],
            }
            continue
        storage = tensor.untyped_storage()
        pointer = int(storage.data_ptr())
        observed_ptrs.add(pointer)
        storage_entry = storages_by_ptr.get(pointer)
        if storage_entry is None:
            storage_entry = {
                "storage_id": f"storage-{len(storages_by_ptr)}",
                "bytes": int(storage.nbytes()),
                "tensor_fields": [],
            }
            storages_by_ptr[pointer] = storage_entry
        storage_entry["tensor_fields"].append(name)
        logical_bytes = int(tensor.numel() * tensor.element_size())
        tensors[name] = {
            "allocated": True,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "logical_bytes": logical_bytes,
            "storage_bytes": storage_entry["bytes"],
            "storage_id": storage_entry["storage_id"],
            "matches_static": bool(
                expected["allocated"]
                and list(tensor.shape) == expected["shape"]
                and str(tensor.dtype) == STATIC_TO_TORCH_DTYPE[expected["dtype"]]
                and logical_bytes == expected["bytes"]
            ),
        }
    declared_ptrs = set(workspace.storage_ptrs)
    storage_records = list(storages_by_ptr.values())
    total_owned = sum(entry["bytes"] for entry in storage_records)
    gates = {
        "all_fields_accounted": tuple(tensors) == WORKSPACE_TENSOR_FIELDS,
        "storage_ptr_inventory_matches_workspace": observed_ptrs == declared_ptrs,
        "tensor_shapes_and_logical_bytes_match_static": all(entry["matches_static"] for entry in tensors.values()),
        "deduplicated_total_matches_static": total_owned == static["total_owned_bytes"],
    }
    return {
        "accounting": "untyped storage bytes, deduplicated by storage base pointer",
        "total_owned_storage_bytes": total_owned,
        "static_expected_owned_bytes": static["total_owned_bytes"],
        "delta_from_static_bytes": total_owned - static["total_owned_bytes"],
        "unique_storage_count": len(storage_records),
        "storages": storage_records,
        "tensors": tensors,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _runtime_topology(sonic, config, workspace, case: str, api: str) -> dict[str, Any]:
    if workspace is None:
        raise RuntimeError("SonicMoE did not publish the workspace used by the completed forward")
    actual_padded = int(workspace.num_valid_ids[0].item())
    expected = _static_topology(_config_dict(config), case)
    training_tile_n = config.tile_n
    if api == "training":
        training_tile_n, _ = sonic._training_stage1_tuning(config, TOKENS, False)
    active_stage1 = actual_padded // config.tile_m * (INTERMEDIATE // training_tile_n)
    active_stage2 = actual_padded // config.stage2_tile_m * (HIDDEN // config.stage2_tile_n)
    stage1_full_grid = workspace.stage1_max_m_blocks * (INTERMEDIATE // training_tile_n)
    stage1_grid = stage1_full_grid
    if config.persistent_stage1 and stage1_full_grid > GFX950_PERSISTENT_GRID_CAP * 4:
        stage1_grid = min(stage1_full_grid, GFX950_PERSISTENT_GRID_CAP)
    stage2_full_grid = workspace.stage2_max_m_blocks * (HIDDEN // config.stage2_tile_n)
    stage2_grid = stage2_full_grid
    if config.persistent_stage2 and stage2_full_grid > GFX950_PERSISTENT_GRID_CAP * 4:
        stage2_grid = min(stage2_full_grid, GFX950_PERSISTENT_GRID_CAP)
    stages = sonic._stage2_stages(config, TOKENS)
    if config.stage2_output_mode != "atomic":
        stages = 1
    workspace_gates = {
        "tokens": workspace.tokens == TOKENS,
        "dense_fixed_topk": workspace.routes is None,
        "route_tile_m": workspace.route_tile_m == config.route_tile_m,
        "capacity_padded_rows": workspace.max_padded_tokens == expected["workspace_capacity_padded_rows"],
        "stage1_capacity_m_blocks": workspace.stage1_max_m_blocks == expected["stage1"]["capacity_m_blocks"],
        "stage2_capacity_m_blocks": workspace.stage2_max_m_blocks == expected["stage2"]["capacity_m_blocks"],
        "route_output_contract": (workspace.route_output is not None) == (config.stage2_output_mode == "reduce"),
    }
    owned_memory = _workspace_owned_memory(workspace, config)
    workspace_gates["owned_storage_accounting"] = owned_memory["passed"]
    return {
        "actual_padded_rows": actual_padded,
        "predicted_padded_rows": expected["actual_padded_rows"],
        "padded_rows_match_prediction": actual_padded == expected["actual_padded_rows"],
        "workspace_gates": workspace_gates,
        "workspace_gates_passed": all(workspace_gates.values()),
        "owned_workspace_memory": owned_memory,
        "workspace_capacity_padded_rows": workspace.max_padded_tokens,
        "route_metadata_blocks": actual_padded // config.route_tile_m,
        "stage1": {
            "effective_tile_n": training_tile_n,
            "active_logical_workgroups": active_stage1,
            "capacity_m_blocks": workspace.stage1_max_m_blocks,
            "full_launch_grid": stage1_full_grid,
            "launch_grid": stage1_grid,
            "persistent": config.persistent_stage1,
        },
        "stage2": {
            "active_logical_workgroups": active_stage2,
            "capacity_m_blocks": workspace.stage2_max_m_blocks,
            "full_launch_grid": stage2_full_grid,
            "launch_grid": stage2_grid,
            "pipeline_stages": stages,
            "persistent": config.persistent_stage2,
        },
    }


def _config_dict(config) -> dict[str, Any]:
    return {
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_experts": config.num_experts,
        "top_k": config.top_k,
        "tile_m": config.tile_m,
        "tile_n": config.tile_n,
        "tile_k": config.tile_k,
        "down_tile_m": config.stage2_tile_m,
        "down_tile_n": config.stage2_tile_n,
        "down_tile_k": config.stage2_tile_k,
        "renormalize": config.renormalize,
        "stage1_b_cache_mod": config.stage1_b_cache_mod,
        "stage2_b_cache_mod": config.stage2_b_cache_mod,
        "stage1_xcd_swizzle": config.stage1_xcd_swizzle,
        "stage1_k_wave": config.stage1_k_wave,
        "stage2_xcd_swizzle": config.stage2_xcd_swizzle,
        "waves_per_eu": config.waves_per_eu,
        "persistent_stage1": config.persistent_stage1,
        "persistent_stage2": config.persistent_stage2,
        "stage2_pipeline_stages": config.stage2_pipeline_stages,
        "stage2_output_mode": config.stage2_output_mode,
        "activation": config.activation,
        "compute_dtype": config.compute_dtype,
    }


def _artifact_resources(launcher) -> dict[str, Any]:
    """Return every kernel metadata record from every compiled launcher artifact."""

    artifacts = list(getattr(launcher, "_mem_cache", {}).items())
    records = []
    artifact_errors = []
    for artifact_index, (cache_key, artifact) in enumerate(artifacts):
        ir_text = getattr(artifact, "ir", None)
        cache_key_text = repr(cache_key)
        artifact_label = cache_key_text[:512]
        cache_key_sha256 = hashlib.sha256(cache_key_text.encode()).hexdigest()
        if not isinstance(ir_text, str):
            artifact_errors.append(
                {
                    "artifact_index": artifact_index,
                    "cache_key_repr_prefix": artifact_label,
                    "cache_key_sha256": cache_key_sha256,
                    "reason": "compiled launcher artifact has no textual IR",
                }
            )
            continue
        matches = list(re.finditer(r'gpu\.kernel_metadata<"([^"]+)"', ir_text))
        if not matches:
            artifact_errors.append(
                {
                    "artifact_index": artifact_index,
                    "cache_key_repr_prefix": artifact_label,
                    "cache_key_sha256": cache_key_sha256,
                    "reason": "compiled launcher artifact has no gpu.kernel_metadata record",
                }
            )
            continue
        for metadata_index, match in enumerate(matches):
            start = match.start()
            end = ir_text.find("}>", start)
            if end < 0:
                artifact_errors.append(
                    {
                        "artifact_index": artifact_index,
                        "cache_key_repr_prefix": artifact_label,
                        "cache_key_sha256": cache_key_sha256,
                        "metadata_index": metadata_index,
                        "kernel": match.group(1),
                        "reason": "gpu.kernel_metadata record is unterminated",
                    }
                )
                continue
            raw_metadata = ir_text[start : end + 2]
            fields = {}
            for key in RESOURCE_METADATA_FIELDS:
                field_match = re.search(rf"{key} = (\d+) : i64", raw_metadata)
                if field_match:
                    fields[key] = int(field_match.group(1))
            records.append(
                {
                    "metadata_found": True,
                    "missing_reason": None,
                    "artifact_index": artifact_index,
                    "cache_key_repr_prefix": artifact_label,
                    "cache_key_sha256": cache_key_sha256,
                    "metadata_index": metadata_index,
                    "kernel": match.group(1),
                    "raw_metadata": raw_metadata,
                    "raw_metadata_sha256": hashlib.sha256(raw_metadata.encode()).hexdigest(),
                    "fields": fields,
                }
            )
    if not artifacts:
        artifact_errors.append(
            {
                "artifact_index": None,
                "cache_key_repr_prefix": None,
                "cache_key_sha256": None,
                "reason": "compiled launcher has no in-memory artifacts",
            }
        )
    return {
        "artifacts_found": bool(artifacts),
        "artifact_count": len(artifacts),
        "metadata_count": len(records),
        "records": records,
        "artifact_errors": artifact_errors,
    }


def _resource_gate(metadata: dict[str, Any]) -> dict[str, Any]:
    fields = metadata.get("fields", {})
    missing_fields = [name for name in RESOURCE_GATE_FIELDS if name not in fields]
    checks = {
        "metadata_found": metadata.get("metadata_found") is True,
        "required_fields_present": not missing_fields,
        "private_segment_zero": fields.get("private_segment_fixed_size") == 0,
        "sgpr_spills_zero": fields.get("sgpr_spill_count") == 0,
        "vgpr_spills_zero": fields.get("vgpr_spill_count") == 0,
        "lds_within_gfx950_limit": (
            fields.get("group_segment_fixed_size") is not None
            and fields["group_segment_fixed_size"] <= GFX950_LDS_BYTES
        ),
    }
    return {
        "required_fields": list(RESOURCE_GATE_FIELDS),
        "missing_fields": missing_fields,
        "gfx950_lds_limit_bytes": GFX950_LDS_BYTES,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _isa_resources(dump_dir: Path | None, kernel_name: str | None) -> dict[str, Any]:
    if dump_dir is None or kernel_name is None:
        return {}
    direct_files = sorted((dump_dir / kernel_name).glob("*_final_isa.s"))
    files = direct_files or sorted(dump_dir.rglob("*_final_isa.s"))
    selected = None
    text = None
    for path in reversed(files):
        candidate = path.read_text(encoding="utf-8")
        if re.search(rf"(?m)^\s*\.amdhsa_kernel\s+{re.escape(kernel_name)}\s*$", candidate):
            selected = path
            text = candidate
            break
    if selected is None or text is None:
        return {}
    metadata_match = re.search(
        rf"(?ms)^\s*\.amdhsa_kernel\s+{re.escape(kernel_name)}\s*$.*?^\s*\.end_amdhsa_kernel\s*$",
        text,
    )
    metadata_text = metadata_match.group(0) if metadata_match else text
    function_match = re.search(
        rf"(?ms)^{re.escape(kernel_name)}:\s*(?:#.*)?$.*?^\s*\.size\s+{re.escape(kernel_name)},",
        text,
    )
    function_text = function_match.group(0) if function_match else text
    result: dict[str, Any] = {
        "isa_path": str(selected),
        "isa_kernel": kernel_name,
        "isa_kernel_metadata_found": metadata_match is not None,
        "isa_function_body_found": function_match is not None,
    }
    for field in (
        "group_segment_fixed_size",
        "private_segment_fixed_size",
        "next_free_vgpr",
        "next_free_sgpr",
        "accum_offset",
    ):
        match = re.search(rf"\.amdhsa_{field}\s+(\d+)", metadata_text)
        if match:
            result[f"isa_{field}"] = int(match.group(1))
    result["isa_uses_flat_scratch"] = bool(re.search(r"\.uses_flat_scratch,\s+1", metadata_text))
    result["isa_scratch_instructions"] = len(
        re.findall(r"\b(?:buffer_|flat_)?scratch_(?:load|store)\w*", function_text)
    )
    result["isa_mfma_instructions"] = len(re.findall(r"\bv_mfma_", function_text))
    result["isa_buffer_load_lds_instructions"] = len(re.findall(r"\bbuffer_load_\w+.*\blds\b", function_text))
    return result


def _kernel_name_matches(kernel_name: str, expected: str) -> bool:
    return kernel_name == expected or kernel_name.startswith(f"{expected}_")


def _launcher_resource(
    launcher,
    dump_dir: Path | None,
    *,
    expected_kernels: tuple[str, ...] = (),
) -> dict[str, Any]:
    metadata = _artifact_resources(launcher)
    kernels = []
    for record in metadata["records"]:
        kernels.append(
            {
                "metadata": record,
                "isa": _isa_resources(dump_dir, record["kernel"]),
                "gate": _resource_gate(record),
            }
        )
    actual_names = [entry["metadata"]["kernel"] for entry in kernels]
    missing_expected = [
        expected
        for expected in expected_kernels
        if not any(_kernel_name_matches(actual, expected) for actual in actual_names)
    ]
    coverage_checks = {
        "artifacts_present": metadata["artifacts_found"],
        "all_artifacts_have_complete_metadata": not metadata["artifact_errors"],
        "at_least_one_kernel_metadata_record": bool(kernels),
        "all_expected_kernels_present": not missing_expected,
        "all_discovered_kernels_pass_resource_gate": bool(kernels)
        and all(entry["gate"]["passed"] for entry in kernels),
    }
    return {
        "artifact_metadata": metadata,
        "expected_kernels": list(expected_kernels),
        "actual_kernels": actual_names,
        "missing_expected_kernels": missing_expected,
        "kernels": kernels,
        "coverage_checks": coverage_checks,
        "passed": all(coverage_checks.values()),
    }


def _launcher_resources(sonic, config, api: str, dump_dir: Path | None, device_index: int) -> dict[str, Any]:
    import kernels.moe.moe_sorting_kernel as sorting

    _, _, sorter = sorting.compile_moe_sorting(
        num_experts=config.num_experts,
        topk=config.top_k,
        unit_size=config.route_tile_m,
        has_mask=False,
    )
    if api == "training":
        tile_n, waves_per_eu = sonic._training_stage1_tuning(config, TOKENS, False)
        stage1 = sonic._get_stage1_training_launcher(
            config,
            sonic._stage1_cache_mod(config, TOKENS),
            False,
            True,
            device_index,
            tile_n,
            waves_per_eu,
        )
    else:
        stage1 = sonic._get_stage1_launcher(
            config,
            sonic._stage1_cache_mod(config, TOKENS),
            "bf16",
            False,
            device_index,
        )
    stages = sonic._stage2_stages(config, TOKENS)
    if config.stage2_output_mode != "atomic":
        stages = 1
    stage2 = sonic._get_stage2_launcher(
        config,
        sonic._stage2_cache_mod(config, TOKENS),
        "bf16",
        False,
        config.stage2_output_mode,
        stages,
        device_index,
    )
    launchers = [
        (
            "sorter",
            sorter,
            ("clear_workspace_kernel", "p0_scatter_kernel", "p1_count_kernel", "p23_kernel"),
        ),
        ("stage1", stage1, ()),
        ("stage2", stage2, ()),
    ]
    if config.stage2_output_mode == "reduce":
        reduction_dtype = "f16" if config.compute_dtype == "fp16" else "bf16"
        reduction = sonic.compile_moe_reduction(
            topk=config.top_k,
            model_dim=config.hidden_size,
            dtype_str=reduction_dtype,
        )
        launchers.append(("reduce", reduction, ("moe_reduction_kernel",)))
    resources = {
        name: _launcher_resource(
            launcher,
            dump_dir,
            expected_kernels=expected,
        )
        for name, launcher, expected in launchers
    }
    return {
        "kernels": resources,
        "passed": all(entry["passed"] for entry in resources.values()),
    }


def _run_correctness(
    torch,
    sonic,
    x,
    baseline_op,
    candidate_op,
    baseline_out,
    candidate_out,
    api: str,
    case: str,
    ids,
    scores,
    oracle_output,
    oracle_state,
    repeats: int,
    dump_dir: Path | None,
) -> dict[str, Any]:
    baseline_output, baseline_state = _run_api(baseline_op, api, x, ids, scores, baseline_out)
    candidate_output, candidate_state = _run_api(candidate_op, api, x, ids, scores, candidate_out)
    torch.cuda.synchronize()
    candidate_baseline_output = _tensor_metrics(torch, candidate_output, baseline_output, OUTPUT_LIMITS)
    baseline_oracle_output = _tensor_metrics(torch, baseline_output, oracle_output, OUTPUT_LIMITS)
    candidate_oracle_output = _tensor_metrics(torch, candidate_output, oracle_output, OUTPUT_LIMITS)
    candidate_baseline_state = None
    baseline_oracle_state = None
    candidate_oracle_state = None
    if api == "training":
        assert baseline_state is not None and candidate_state is not None and oracle_state is not None
        candidate_baseline_state = _tensor_metrics(
            torch,
            candidate_state.preactivation,
            baseline_state.preactivation,
            STATE_LIMITS,
        )
        baseline_oracle_state = _tensor_metrics(
            torch,
            baseline_state.preactivation,
            oracle_state,
            STATE_LIMITS,
        )
        candidate_oracle_state = _tensor_metrics(
            torch,
            candidate_state.preactivation,
            oracle_state,
            STATE_LIMITS,
        )

    first_output = candidate_output.clone()
    first_state = None if candidate_state is None else candidate_state.preactivation.clone()
    repeatability = []
    for repeat in range(1, repeats):
        repeated_output, repeated_state = _run_api(candidate_op, api, x, ids, scores, candidate_out)
        torch.cuda.synchronize()
        entry = {
            "repeat": repeat,
            "output_bitwise_equal": bool(torch.equal(repeated_output, first_output)),
            "output_vs_first": _tensor_metrics(torch, repeated_output, first_output, OUTPUT_LIMITS),
            "output_vs_oracle": _tensor_metrics(torch, repeated_output, oracle_output, OUTPUT_LIMITS),
        }
        if api == "training":
            assert repeated_state is not None and first_state is not None and oracle_state is not None
            entry["state_bitwise_equal"] = bool(torch.equal(repeated_state.preactivation, first_state))
            entry["state_vs_oracle"] = _tensor_metrics(
                torch,
                repeated_state.preactivation,
                oracle_state,
                STATE_LIMITS,
            )
        repeatability.append(entry)
        del repeated_output, repeated_state

    output_bitwise_required = candidate_op.config.stage2_output_mode == "reduce"
    output_bitwise_passed = all(entry["output_bitwise_equal"] for entry in repeatability)
    output_tolerance_passed = all(
        entry["output_vs_first"]["passed"] and entry["output_vs_oracle"]["passed"] for entry in repeatability
    )
    state_bitwise_required = api == "training"
    state_bitwise_passed = all(entry.get("state_bitwise_equal", True) for entry in repeatability)
    state_tolerance_passed = all(entry.get("state_vs_oracle", {"passed": True})["passed"] for entry in repeatability)
    repeatability_passed = (
        output_tolerance_passed
        and (not output_bitwise_required or output_bitwise_passed)
        and state_tolerance_passed
        and (not state_bitwise_required or state_bitwise_passed)
    )
    runtime_topology = {
        "baseline": _runtime_topology(sonic, baseline_op.config, baseline_op.workspace, case, api),
        "candidate": _runtime_topology(sonic, candidate_op.config, candidate_op.workspace, case, api),
    }
    baseline_workspace_bytes = runtime_topology["baseline"]["owned_workspace_memory"]["total_owned_storage_bytes"]
    candidate_workspace_bytes = runtime_topology["candidate"]["owned_workspace_memory"]["total_owned_storage_bytes"]
    owned_workspace_comparison = {
        "accounting": "per-operator owned storage; independent of simultaneous process residency",
        "baseline_bytes": baseline_workspace_bytes,
        "candidate_bytes": candidate_workspace_bytes,
        "candidate_minus_baseline_bytes": candidate_workspace_bytes - baseline_workspace_bytes,
    }
    topology_passed = bool(
        all(
            topology["padded_rows_match_prediction"] and topology["workspace_gates_passed"]
            for topology in runtime_topology.values()
        )
    )
    resources = {
        "baseline": _launcher_resources(
            sonic,
            baseline_op.config,
            api,
            dump_dir,
            x.device.index or 0,
        ),
        "candidate": _launcher_resources(
            sonic,
            candidate_op.config,
            api,
            dump_dir,
            x.device.index or 0,
        ),
    }
    resource_gate_passed = all(resource["passed"] for resource in resources.values())
    functional_passed = bool(
        candidate_baseline_output["passed"]
        and baseline_oracle_output["passed"]
        and candidate_oracle_output["passed"]
        and (candidate_baseline_state is None or candidate_baseline_state["passed"])
        and (baseline_oracle_state is None or baseline_oracle_state["passed"])
        and (candidate_oracle_state is None or candidate_oracle_state["passed"])
        and repeatability_passed
        and topology_passed
    )
    passed = functional_passed and resource_gate_passed
    result = {
        "candidate_vs_baseline": {
            "output": candidate_baseline_output,
            "training_preactivation": candidate_baseline_state,
        },
        "baseline_vs_oracle": {
            "output": baseline_oracle_output,
            "training_preactivation": baseline_oracle_state,
        },
        "candidate_vs_oracle": {
            "output": candidate_oracle_output,
            "training_preactivation": candidate_oracle_state,
        },
        "candidate_repeatability": {
            "samples": repeatability,
            "output_bitwise_required": output_bitwise_required,
            "output_bitwise_passed": output_bitwise_passed,
            "output_tolerance_required": True,
            "output_tolerance_passed": output_tolerance_passed,
            "training_state_bitwise_required": state_bitwise_required,
            "training_state_bitwise_passed": state_bitwise_passed,
            "training_state_tolerance_required": state_bitwise_required,
            "training_state_tolerance_passed": state_tolerance_passed,
            "passed": repeatability_passed,
        },
        "runtime_topology": runtime_topology,
        "owned_workspace_comparison": owned_workspace_comparison,
        "resources": resources,
        "functional_passed": functional_passed,
        "resource_gate_passed": resource_gate_passed,
        "passed": passed,
    }
    del first_output, first_state, baseline_state, candidate_state
    gc.collect()
    return result


def _event_time(torch, call: Callable[[], Any]) -> dict[str, float]:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    host_begin = time.perf_counter()
    begin.record()
    value = call()
    end.record()
    end.synchronize()
    host_ms = (time.perf_counter() - host_begin) * 1.0e3
    device_ms = begin.elapsed_time(end)
    del value
    return {"device_ms": device_ms, "host_ms": host_ms}


def _stats(values: Iterable[float]) -> dict[str, Any]:
    materialized = list(values)
    return {
        "samples_ms": materialized,
        "median_ms": statistics.median(materialized),
        "mean_ms": statistics.mean(materialized),
        "min_ms": min(materialized),
        "max_ms": max(materialized),
    }


def _measure_abba(torch, calls: dict[str, Callable[[], Any]], pairs: int) -> dict[str, Any]:
    raw_samples = []
    sequence_blocks = []
    pair_blocks = []
    by_mode = {
        "baseline": {"device": [], "host": []},
        "candidate": {"device": [], "host": []},
    }
    orders = (
        ("ABBA", ("baseline", "candidate", "candidate", "baseline")),
        ("BAAB", ("candidate", "baseline", "baseline", "candidate")),
    )
    for pair in range(pairs):
        pair_samples = {"baseline": [], "candidate": []}
        sequence_order = orders if pair % 2 == 0 else tuple(reversed(orders))
        for sequence_index, (sequence_name, order) in enumerate(sequence_order):
            sequence_samples = {"baseline": [], "candidate": []}
            for position, mode in enumerate(order):
                sample = _event_time(torch, calls[mode])
                by_mode[mode]["device"].append(sample["device_ms"])
                by_mode[mode]["host"].append(sample["host_ms"])
                sequence_samples[mode].append(sample["device_ms"])
                pair_samples[mode].append(sample["device_ms"])
                raw_samples.append(
                    {
                        "pair": pair,
                        "sequence": sequence_name,
                        "sequence_index": sequence_index,
                        "position": position,
                        "mode": mode,
                        **sample,
                    }
                )
            sequence_baseline = statistics.mean(sequence_samples["baseline"])
            sequence_candidate = statistics.mean(sequence_samples["candidate"])
            sequence_blocks.append(
                {
                    "pair": pair,
                    "sequence": sequence_name,
                    "sequence_index": sequence_index,
                    "baseline_mean_device_ms": sequence_baseline,
                    "candidate_mean_device_ms": sequence_candidate,
                    "speedup": sequence_baseline / sequence_candidate,
                    "candidate_faster": sequence_candidate < sequence_baseline,
                }
            )
        baseline_mean = statistics.mean(pair_samples["baseline"])
        candidate_mean = statistics.mean(pair_samples["candidate"])
        pair_blocks.append(
            {
                "pair": pair,
                "sequence_order": [name for name, _ in sequence_order],
                "samples_per_mode": len(pair_samples["baseline"]),
                "baseline_mean_device_ms": baseline_mean,
                "candidate_mean_device_ms": candidate_mean,
                "speedup": baseline_mean / candidate_mean,
                "candidate_faster": candidate_mean < baseline_mean,
            }
        )
    baseline_device = _stats(by_mode["baseline"]["device"])
    candidate_device = _stats(by_mode["candidate"]["device"])
    paired_speedups = [block["speedup"] for block in pair_blocks]
    return {
        "design": "every pair executes both ABBA and BAAB; sequence order alternates by pair",
        "raw_samples": raw_samples,
        "sequence_blocks": sequence_blocks,
        "pair_blocks": pair_blocks,
        "summary": {
            "baseline_device": baseline_device,
            "candidate_device": candidate_device,
            "baseline_host": _stats(by_mode["baseline"]["host"]),
            "candidate_host": _stats(by_mode["candidate"]["host"]),
            "median_speedup": baseline_device["median_ms"] / candidate_device["median_ms"],
            "paired_speedup_median": statistics.median(paired_speedups),
            "paired_win_rate": sum(block["candidate_faster"] for block in pair_blocks) / len(pair_blocks),
        },
    }


def _clear_operator_workspaces(operators: dict[str, Any]) -> dict[str, bool]:
    for op in operators.values():
        op.clear_workspace()
    return {name: op.workspace is None for name, op in operators.items()}


def _peak_sample(
    torch,
    mode: str,
    calls: dict[str, Callable[[], Any]],
    operators: dict[str, Any],
) -> dict[str, Any]:
    workspaces_absent = _clear_operator_workspaces(operators)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    resident_allocated = torch.cuda.memory_allocated()
    resident_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    value = calls[mode]()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    selected_workspace = operators[mode].workspace
    if selected_workspace is None:
        raise RuntimeError(f"{mode} peak sample completed without publishing a workspace")
    selected_memory = _workspace_owned_memory(selected_workspace, operators[mode].config)
    other = "candidate" if mode == "baseline" else "baseline"
    isolation_gates = {
        "both_workspaces_absent_before_sample": all(workspaces_absent.values()),
        "selected_workspace_allocated": selected_workspace is not None,
        "other_workspace_absent_after_sample": operators[other].workspace is None,
        "selected_workspace_accounting_passed": selected_memory["passed"],
    }
    del value
    operators[mode].clear_workspace()
    gc.collect()
    torch.cuda.synchronize()
    return {
        "mode": mode,
        "workspace_absent_before_sample": workspaces_absent,
        "selected_owned_workspace": selected_memory,
        "isolation_gates": isolation_gates,
        "isolation_passed": all(isolation_gates.values()),
        "resident_allocated_bytes": resident_allocated,
        "resident_reserved_bytes": resident_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_delta_bytes": peak_allocated - resident_allocated,
        "peak_reserved_delta_bytes": peak_reserved - resident_reserved,
    }


def _measure_peaks(
    torch,
    calls: dict[str, Callable[[], Any]],
    operators: dict[str, Any],
    samples: int,
) -> dict[str, Any]:
    raw = {"baseline": [], "candidate": []}
    for sample in range(samples):
        order = ("baseline", "candidate") if sample % 2 == 0 else ("candidate", "baseline")
        for mode in order:
            raw[mode].append(
                {
                    "sample": sample,
                    "order": "AB" if sample % 2 == 0 else "BA",
                    **_peak_sample(torch, mode, calls, operators),
                }
            )
    summary = {}
    for mode, values in raw.items():
        summary[mode] = {
            "peak_allocated_bytes_median": statistics.median(value["peak_allocated_bytes"] for value in values),
            "peak_allocated_delta_bytes_median": statistics.median(
                value["peak_allocated_delta_bytes"] for value in values
            ),
            "peak_reserved_bytes_median": statistics.median(value["peak_reserved_bytes"] for value in values),
            "peak_reserved_delta_bytes_median": statistics.median(
                value["peak_reserved_delta_bytes"] for value in values
            ),
        }
    summary["candidate_minus_baseline_peak_allocated_bytes"] = (
        summary["candidate"]["peak_allocated_bytes_median"] - summary["baseline"]["peak_allocated_bytes_median"]
    )
    owned_baseline = statistics.median(
        value["selected_owned_workspace"]["total_owned_storage_bytes"] for value in raw["baseline"]
    )
    owned_candidate = statistics.median(
        value["selected_owned_workspace"]["total_owned_storage_bytes"] for value in raw["candidate"]
    )
    isolation_passed = all(value["isolation_passed"] for values in raw.values() for value in values)
    return {
        "method": "in-process isolated: clear both operator workspaces before every single-mode sample",
        "raw_samples": raw,
        "summary": summary,
        "owned_workspace_comparison": {
            "baseline_bytes_median": owned_baseline,
            "candidate_bytes_median": owned_candidate,
            "candidate_minus_baseline_bytes": owned_candidate - owned_baseline,
        },
        "isolation_passed": isolation_passed,
    }


def _warmup(torch, calls: dict[str, Callable[[], Any]], iterations: int) -> None:
    for _ in range(iterations):
        for mode in ("baseline", "candidate"):
            value = calls[mode]()
            del value
    torch.cuda.synchronize()


def _device_identity(torch, device_index: int, arch: str) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "index": device_index,
        "name": properties.name,
        "arch": arch,
        "compute_units": int(properties.multi_processor_count),
        "total_memory_bytes": int(properties.total_memory),
        "warp_size": int(properties.warp_size),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "python": platform.python_version(),
        "hostname": platform.node(),
        "visibility": {
            name: os.environ.get(name)
            for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
        },
    }


class _MockStorage:
    def __init__(self, pointer: int, size: int):
        self._pointer = pointer
        self._size = size

    def data_ptr(self) -> int:
        return self._pointer

    def nbytes(self) -> int:
        return self._size


class _MockTensor:
    def __init__(self, shape: list[int], dtype: str, element_size: int, pointer: int):
        self.shape = tuple(shape)
        self.dtype = STATIC_TO_TORCH_DTYPE[dtype]
        self._element_size = element_size
        self._storage = _MockStorage(pointer, math.prod(shape) * element_size)

    def numel(self) -> int:
        return math.prod(self.shape)

    def element_size(self) -> int:
        return self._element_size

    def untyped_storage(self) -> _MockStorage:
        return self._storage


class _MockOperator:
    def __init__(self):
        self.workspace = object()
        self.clear_calls = 0

    def clear_workspace(self) -> None:
        self.clear_calls += 1
        self.workspace = None


def _mock_config(config: dict[str, Any]) -> SimpleNamespace:
    result = SimpleNamespace(**config)
    result.stage2_tile_m = config["down_tile_m"]
    result.stage2_tile_n = config["down_tile_n"]
    result.stage2_tile_k = config["down_tile_k"]
    result.route_tile_m = math.lcm(config["tile_m"], config["down_tile_m"])
    return result


def _mock_workspace(config: dict[str, Any]) -> SimpleNamespace:
    static = _static_workspace_footprint(config)
    fields = {}
    pointers = set()
    pointer = 0x1000
    for name, record in static["tensors"].items():
        if not record["allocated"]:
            fields[name] = None
            continue
        tensor = _MockTensor(record["shape"], record["dtype"], record["element_size"], pointer)
        fields[name] = tensor
        pointers.add(pointer)
        pointer += max(0x1000, record["bytes"] + 0x1000)
    fields["storage_ptrs"] = frozenset(pointers)
    return SimpleNamespace(**fields)


def _self_test_payload(repo: Path) -> dict[str, Any]:
    """Pure-CPU invariants for profile legality, storage accounting, and gates."""

    records = {profile.name: _profile_record(profile) for profile in PROFILES}
    baseline_config = _resolved_profile_config("baseline")
    baseline_workspace = _static_workspace_footprint(baseline_config)
    mock_accounting = _workspace_owned_memory(
        _mock_workspace(baseline_config),
        _mock_config(baseline_config),
    )
    alias_workspace = _mock_workspace(baseline_config)
    alias_workspace.sorted_weights._storage = alias_workspace.sorted_token_ids._storage
    alias_workspace.storage_ptrs = frozenset(
        tensor.untyped_storage().data_ptr()
        for name in WORKSPACE_TENSOR_FIELDS
        if (tensor := getattr(alias_workspace, name)) is not None
    )
    alias_accounting = _workspace_owned_memory(
        alias_workspace,
        _mock_config(baseline_config),
    )
    reduce_workspace = records["reduce-output"]["static_footprint"]["sonic_moe_workspace"]
    reduce_parent_workspace = records["xcd8-cached"]["static_footprint"]["sonic_moe_workspace"]
    reduce_route_bytes = records["reduce-output"]["static_footprint"]["reduce_route_output_workspace_bytes"]
    source_identity = _collect_source_identity(
        repo,
        allow_dirty=True,
        enforce_provenance=False,
    )

    raw_metadata = (
        'gpu.kernel_metadata<"mock_a", !llvm.func<void ()>, metadata = {'
        "agpr_count = 0 : i64, group_segment_fixed_size = 65536 : i64, "
        "private_segment_fixed_size = 0 : i64, sgpr_count = 32 : i64, "
        "sgpr_spill_count = 0 : i64, vgpr_count = 128 : i64, "
        "vgpr_spill_count = 0 : i64, wavefront_size = 64 : i64}>"
    )
    second_raw_metadata = raw_metadata.replace('"mock_a"', '"mock_b"').replace(
        "vgpr_count = 128",
        "vgpr_count = 96",
    )
    launcher = SimpleNamespace(_mem_cache={"mock": SimpleNamespace(ir=f"{raw_metadata}\n{second_raw_metadata}")})
    parsed_artifacts = _artifact_resources(launcher)
    parsed_metadata = parsed_artifacts["records"][0]
    passing_resource_gate = _launcher_resource(
        launcher,
        None,
        expected_kernels=("mock_a", "mock_b"),
    )
    absent_resource_gate = _launcher_resource(SimpleNamespace(_mem_cache={}), None)
    missing_expected_resource_gate = _launcher_resource(
        launcher,
        None,
        expected_kernels=("mock_a", "mock_b", "mock_c"),
    )
    partial_artifact_resource_gate = _launcher_resource(
        SimpleNamespace(
            _mem_cache={
                "good": SimpleNamespace(ir=raw_metadata),
                "missing": SimpleNamespace(ir="module {}"),
            }
        ),
        None,
        expected_kernels=("mock_a",),
    )
    missing_resource_gate = _resource_gate(
        {
            "metadata_found": True,
            "fields": {"group_segment_fixed_size": 0},
        }
    )
    spilling_fields = dict(parsed_metadata["fields"])
    spilling_fields["vgpr_spill_count"] = 1
    spilling_resource_gate = _resource_gate(
        {
            "metadata_found": True,
            "fields": spilling_fields,
        }
    )
    private_fields = dict(parsed_metadata["fields"])
    private_fields["private_segment_fixed_size"] = 16
    private_resource_gate = _resource_gate(
        {
            "metadata_found": True,
            "fields": private_fields,
        }
    )
    oversized_lds_fields = dict(parsed_metadata["fields"])
    oversized_lds_fields["group_segment_fixed_size"] = GFX950_LDS_BYTES + 1
    oversized_lds_resource_gate = _resource_gate(
        {
            "metadata_found": True,
            "fields": oversized_lds_fields,
        }
    )
    mock_operators = {"baseline": _MockOperator(), "candidate": _MockOperator()}
    mock_clear_result = _clear_operator_workspaces(mock_operators)

    checks = {
        "all_profiles_static_valid": len(records) == len(PROFILES),
        "all_profiles_reuse_prepared_weights": all(
            record["prepared_weight_compatibility"]["matches_baseline"] for record in records.values()
        ),
        "workspace_inventory_complete": tuple(baseline_workspace["tensors"]) == WORKSPACE_TENSOR_FIELDS,
        "mock_workspace_deduplicated_total_matches_static": mock_accounting["passed"],
        "mock_alias_is_counted_once": (
            alias_accounting["total_owned_storage_bytes"]
            == baseline_workspace["total_owned_bytes"] - baseline_workspace["tensors"]["sorted_weights"]["bytes"]
            and alias_accounting["unique_storage_count"] == mock_accounting["unique_storage_count"] - 1
        ),
        "reduce_workspace_delta_is_route_output": (
            reduce_workspace["total_owned_bytes"] - reduce_parent_workspace["total_owned_bytes"] == reduce_route_bytes
        ),
        "reduce_workspace_delta_is_exactly_448_mib": reduce_route_bytes == 469_762_048,
        "peak_isolation_clear_removes_both_workspaces": (
            all(mock_clear_result.values()) and all(operator.clear_calls == 1 for operator in mock_operators.values())
        ),
        "git_queries_use_explicit_safe_directory": all(
            command["command"][:3] == ["git", "-c", f"safe.directory={repo.resolve()}"]
            for command in source_identity["git"]["commands"].values()
        ),
        "git_provenance_queries_succeed": source_identity["checks"]["git_queries_succeeded"],
        "dirty_tree_content_is_hashed": bool(source_identity["git"]["working_tree_content_sha256"]),
        "m80_balanced_padding": (records["m80-equal"]["topology"]["balanced"]["actual_padded_rows"] == 71680),
        "m80_hot16_padding": (records["m80-equal"]["topology"]["hot16"]["actual_padded_rows"] == 66560),
        "distribution_aware_m_tiles_direct_to_lds": all(
            (records[name]["config"]["tile_m"] * records[name]["config"]["tile_k"]) % 2048 == 0
            and (records[name]["config"]["down_tile_m"] * records[name]["config"]["down_tile_k"]) % 2048 == 0
            for name in ("m80-equal", "m96-equal", "m112-equal")
        ),
        "distribution_aware_m_tiles_fit_lds": all(
            records[name]["effective"]["static_lds"][stage] <= GFX950_LDS_BYTES
            for name in ("m80-equal", "m96-equal", "m112-equal")
            for stage in ("stage1_total_bytes", "stage2_total_bytes")
        ),
        "all_multiphase_metadata_records_are_enumerated": parsed_artifacts["metadata_count"] == 2,
        "complete_zero_spill_metadata_passes": passing_resource_gate["passed"],
        "absent_metadata_fails": not absent_resource_gate["passed"],
        "missing_expected_kernel_fails": not missing_expected_resource_gate["passed"],
        "artifact_without_metadata_fails": not partial_artifact_resource_gate["passed"],
        "missing_metadata_field_fails": not missing_resource_gate["passed"],
        "nonzero_spill_fails": not spilling_resource_gate["passed"],
        "nonzero_private_segment_fails": not private_resource_gate["passed"],
        "oversized_lds_fails": not oversized_lds_resource_gate["passed"],
        "raw_metadata_preserved": parsed_metadata["raw_metadata"] == raw_metadata,
    }
    return {
        "schema": "flydsl.sonic_e896_forward_self_test.v1",
        "cpu_only": True,
        "checks": checks,
        "workspace": {
            "baseline_static": baseline_workspace,
            "baseline_mock_actual": mock_accounting,
            "aliased_mock_actual": alias_accounting,
            "reduce_static": reduce_workspace,
        },
        "resource_gate_examples": {
            "passing": passing_resource_gate,
            "absent": absent_resource_gate,
            "missing_expected": missing_expected_resource_gate,
            "partial_artifact": partial_artifact_resource_gate,
            "missing_field": missing_resource_gate,
            "spilling": spilling_resource_gate,
            "private_segment": private_resource_gate,
            "oversized_lds": oversized_lds_resource_gate,
        },
        "source_identity": source_identity,
        "passed": all(checks.values()),
    }


def _execute(args: argparse.Namespace, repo: Path) -> dict[str, Any]:
    if args.dump_dir is not None:
        args.dump_dir = args.dump_dir.resolve()
        os.environ["FLYDSL_DUMP_IR"] = "1"
        os.environ["FLYDSL_DUMP_DIR"] = str(args.dump_dir)

    source_identity = _collect_source_identity(
        repo,
        allow_dirty=args.allow_dirty,
        enforce_provenance=True,
    )
    if not source_identity["passed"]:
        return {
            "schema": "flydsl.sonic_e896_forward_acceptance.v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "failure_stage": "git-provenance-preflight",
            "source_identity": source_identity,
            "codegen_environment": _codegen_environment(),
            "passed": False,
        }

    import torch

    import kernels.moe.sonic as sonic
    from flydsl.runtime.device import get_rocm_arch

    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")
    if args.device >= torch.cuda.device_count():
        raise RuntimeError(f"requested cuda:{args.device}, but only {torch.cuda.device_count()} device(s) are visible")
    torch.cuda.set_device(args.device)
    arch = str(get_rocm_arch())
    if not arch.startswith("gfx950"):
        raise RuntimeError(f"this acceptance sweep requires gfx950, found {arch!r}")

    selected = _selected_profiles(args)
    profile_records = {name: _profile_record(PROFILE_BY_NAME[name]) for name in ("baseline", *selected)}
    configs = {name: sonic.SonicMoEConfig(**profile_records[name]["config"]) for name in profile_records}
    if not all(profile_records[name]["prepared_weight_compatibility"]["matches_baseline"] for name in profile_records):
        raise RuntimeError("selected tile candidates do not share the baseline prepared-weight ABI")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    x = torch.empty((TOKENS, HIDDEN), device="cuda", dtype=torch.bfloat16).uniform_(
        -0.02,
        0.02,
        generator=generator,
    )
    w1 = torch.empty(
        (EXPERTS, 2 * INTERMEDIATE, HIDDEN),
        device="cuda",
        dtype=torch.bfloat16,
    ).uniform_(-0.02, 0.02, generator=generator)
    w2 = torch.empty(
        (EXPERTS, HIDDEN, INTERMEDIATE),
        device="cuda",
        dtype=torch.bfloat16,
    ).uniform_(-0.02, 0.02, generator=generator)
    weights = sonic.prepare_sonic_bf16_weights(w1, w2, configs["baseline"])
    routing_inputs = {}
    oracle_results = {}
    retain_oracle_state = "training" in dict.fromkeys(args.apis)
    for case_index, case in enumerate(dict.fromkeys(args.cases)):
        route_generator = torch.Generator(device="cuda").manual_seed(args.seed + 1000 + case_index)
        ids, scores, routing_summary = _make_routing(torch, case, route_generator)
        oracle_output, oracle_state, oracle_metadata = _chunked_torch_oracle(
            torch,
            x,
            w1,
            w2,
            ids,
            scores,
            retain_interleaved_state=retain_oracle_state,
        )
        routing_inputs[case] = (ids, scores, routing_summary)
        oracle_results[case] = {
            "output": oracle_output,
            "state": oracle_state,
            "metadata": oracle_metadata,
        }
    torch.cuda.synchronize()
    setup_peak = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    del w1, w2
    gc.collect()
    torch.cuda.empty_cache()

    baseline_op = sonic.SonicMoE(configs["baseline"], weights)
    baseline_out = torch.empty_like(x)
    report: dict[str, Any] = {
        "schema": "flydsl.sonic_e896_forward_acceptance.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "exclusive_gpu_asserted": args.exclusive_gpu,
        "timing_status": "skipped (--correctness-only)" if args.correctness_only else "measured",
        "device": _device_identity(torch, args.device, arch),
        "source_identity": source_identity,
        "codegen_environment": _codegen_environment(),
        "contract": {
            "shape": {"T": TOKENS, "H": HIDDEN, "I": INTERMEDIATE, "E": EXPERTS, "K": TOPK},
            "dtype": "torch.bfloat16",
            "activation": "swiglu",
            "bias": False,
            "training_state_interleaved_w1": True,
            "routing_cases": list(dict.fromkeys(args.cases)),
            "apis": list(dict.fromkeys(args.apis)),
            "input_seed": args.seed,
            "routing_seeds": {case: args.seed + 1000 + index for index, case in enumerate(dict.fromkeys(args.cases))},
        },
        "numerical_limits": {"output": OUTPUT_LIMITS, "training_preactivation": STATE_LIMITS},
        "oracle": {case: oracle_results[case]["metadata"] for case in dict.fromkeys(args.cases)},
        "setup_peak_memory": setup_peak,
        "baseline": profile_records["baseline"],
        "prepared_weight_reuse_gate": {
            "fields": list(WEIGHT_COMPATIBILITY_FIELDS),
            "all_selected_profiles_match_baseline": True,
            "validated_by_sonic_moe_constructors": True,
        },
        "selected_candidates": list(selected),
        "candidates": {},
    }

    correctness_all = True
    resource_all = True
    memory_isolation_all = True
    performance_all = True
    performance_required = not args.correctness_only and not args.skip_performance_gate
    for profile_name in selected:
        print(f"[sonic-e896-forward] profile={profile_name}", file=sys.stderr, flush=True)
        profile = profile_records[profile_name]
        candidate_op = sonic.SonicMoE(configs[profile_name], weights)
        candidate_out = torch.empty_like(x)
        profile_result: dict[str, Any] = {
            "profile": profile,
            "cases": {},
        }
        for case in dict.fromkeys(args.cases):
            ids, scores, routing_summary = routing_inputs[case]
            case_result: dict[str, Any] = {"routing": routing_summary, "apis": {}}
            for api in dict.fromkeys(args.apis):
                print(
                    f"[sonic-e896-forward] profile={profile_name} case={case} api={api}",
                    file=sys.stderr,
                    flush=True,
                )
                correctness = _run_correctness(
                    torch,
                    sonic,
                    x,
                    baseline_op,
                    candidate_op,
                    baseline_out,
                    candidate_out,
                    api,
                    case,
                    ids,
                    scores,
                    oracle_results[case]["output"],
                    oracle_results[case]["state"],
                    args.correctness_repeats,
                    args.dump_dir,
                )

                def baseline_call(
                    api=api,
                    ids=ids,
                    scores=scores,
                    op=baseline_op,
                    out=baseline_out,
                    hidden=x,
                ):
                    return _run_api(op, api, hidden, ids, scores, out)

                def candidate_call(
                    api=api,
                    ids=ids,
                    scores=scores,
                    op=candidate_op,
                    out=candidate_out,
                    hidden=x,
                ):
                    return _run_api(op, api, hidden, ids, scores, out)

                calls = {"baseline": baseline_call, "candidate": candidate_call}
                api_result: dict[str, Any] = {"correctness": correctness}
                if args.skip_peak_memory:
                    api_result["peak_memory_status"] = "skipped (--skip-peak-memory)"
                    memory_isolation_passed = True
                else:
                    peak_memory = _measure_peaks(
                        torch,
                        calls,
                        {"baseline": baseline_op, "candidate": candidate_op},
                        args.peak_samples,
                    )
                    peak_memory["correctness_run_owned_workspace_comparison"] = correctness[
                        "owned_workspace_comparison"
                    ]
                    api_result["peak_memory"] = peak_memory
                    memory_isolation_passed = bool(peak_memory["isolation_passed"])

                if args.correctness_only:
                    api_result["timing_status"] = "skipped (--correctness-only)"
                    performance_passed = True
                else:
                    _warmup(torch, calls, args.warmup)
                    timing = _measure_abba(torch, calls, args.pairs)
                    summary = timing["summary"]
                    performance_gate = {
                        "minimum_speedup": args.min_speedup,
                        "minimum_paired_win_rate": args.min_paired_win_rate,
                        "median_speedup_passed": summary["median_speedup"] >= args.min_speedup,
                        "paired_speedup_passed": summary["paired_speedup_median"] >= args.min_speedup,
                        "paired_win_rate_passed": summary["paired_win_rate"] >= args.min_paired_win_rate,
                    }
                    performance_gate["passed"] = all(
                        value for key, value in performance_gate.items() if key.endswith("_passed")
                    )
                    api_result["timing"] = timing
                    api_result["performance_gate"] = performance_gate
                    performance_passed = bool(performance_gate["passed"])
                api_result["memory_isolation_passed"] = memory_isolation_passed
                api_result["acceptance_passed"] = (
                    correctness["passed"]
                    and memory_isolation_passed
                    and (not performance_required or performance_passed)
                )
                case_result["apis"][api] = api_result
                correctness_all &= correctness["functional_passed"]
                resource_all &= correctness["resource_gate_passed"]
                memory_isolation_all &= memory_isolation_passed
                performance_all &= performance_passed
            profile_result["cases"][case] = case_result
        profile_result["correctness_passed"] = all(
            api_result["correctness"]["functional_passed"]
            for case_result in profile_result["cases"].values()
            for api_result in case_result["apis"].values()
        )
        profile_result["resource_gate_passed"] = all(
            api_result["correctness"]["resource_gate_passed"]
            for case_result in profile_result["cases"].values()
            for api_result in case_result["apis"].values()
        )
        profile_result["memory_isolation_passed"] = all(
            api_result["memory_isolation_passed"]
            for case_result in profile_result["cases"].values()
            for api_result in case_result["apis"].values()
        )
        profile_result["acceptance_passed"] = all(
            api_result["acceptance_passed"]
            for case_result in profile_result["cases"].values()
            for api_result in case_result["apis"].values()
        )
        report["candidates"][profile_name] = profile_result
        candidate_op.clear_workspace()
        del candidate_op, candidate_out
        gc.collect()
        torch.cuda.empty_cache()

    baseline_op.clear_workspace()
    report["correctness_passed"] = bool(correctness_all)
    report["resource_gate_passed"] = bool(resource_all)
    report["memory_isolation_passed"] = bool(memory_isolation_all)
    report["performance_passed"] = None if args.correctness_only else bool(performance_all)
    report["performance_required"] = performance_required
    report["performance_gate_opt_out"] = bool(args.skip_performance_gate)
    report["passed"] = bool(
        correctness_all and resource_all and memory_isolation_all and (not performance_required or performance_all)
    )
    return report


def main() -> None:
    args = _parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.list_profiles:
        _emit(_plan_payload(args, repo), args.output)
        return
    if args.self_test:
        report = _self_test_payload(repo)
        _emit(report, args.output)
        if not report["passed"]:
            raise SystemExit(2)
        return
    report = _execute(args, repo)
    _emit(report, args.output)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
