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
        --require-performance --output /tmp/sonic-e896-forward.json
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
        "m128-equal",
        "bn256-bk64",
        "pipeline2",
    ),
    "locality": ("xcd8-cached", "non-temporal", "persistent"),
    "output": ("reduce-output",),
    "extended": CANDIDATE_NAMES,
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _dict_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(repo: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", "-C", str(repo), *args),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


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
    if requested is None:
        # This E896 shape does not match sonic.py's sole automatic two-stage gate.
        return 1
    if config["stage2_output_mode"] != "atomic":
        return 1
    return int(requested)


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

    k_tiles_per_wave = HIDDEN // (k_wave * bk1)
    stage1_stages = 2 if k_tiles_per_wave > 1 else 1
    stage1_a_lds = k_wave * stage1_stages * bm1 * bk1 * 2
    if k_wave > 1:
        n_waves = 4 // k_wave
        acc_n = (bn1 // n_waves) // 16
        m_repeat = bm1 // 16
        stage1_reduce_lds = 4 * (acc_n * m_repeat) * 64 * 4 * 4
        stage1_lds = max(stage1_a_lds, stage1_reduce_lds)
    else:
        stage1_lds = stage1_a_lds
    stage2_stages = _effective_stage2_stages(config)
    stage2_lds = max(stage2_stages * bm2 * bk2 * 2, bm2 * bn2 * 4)
    if stage1_lds > GFX950_LDS_BYTES:
        errors.append(f"Stage 1 LDS {stage1_lds} exceeds {GFX950_LDS_BYTES}")
    if stage2_lds > GFX950_LDS_BYTES:
        errors.append(f"Stage 2 LDS {stage2_lds} exceeds {GFX950_LDS_BYTES}")
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
    stage1_launch_grid = capacity_stage1_m_blocks * (INTERMEDIATE // bn1)
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
            "launch_grid": stage1_launch_grid,
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


def _static_footprint(config: dict[str, Any]) -> dict[str, int]:
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
        },
        "topology": {case: _static_topology(config, case) for case in ROUTING_CASES},
        "static_footprint": _static_footprint(config),
    }


def _collect_source_identity(repo: Path) -> dict[str, Any]:
    files: dict[str, dict[str, str]] = {}
    aggregate = hashlib.sha256()
    for relative in (*RUNTIME_SOURCE_FILES, "tools/accept_sonic_e896_forward.py"):
        path = repo / relative
        digest = _sha256(path)
        files[relative] = {"path": str(path.resolve()), "sha256": digest}
        aggregate.update(relative.encode())
        aggregate.update(bytes.fromhex(digest))
    return {
        "repo": str(repo.resolve()),
        "head": _git_value(repo, "rev-parse", "HEAD"),
        "branch": _git_value(repo, "branch", "--show-current"),
        "status": _git_value(repo, "status", "--short"),
        "runtime_sha256": aggregate.hexdigest(),
        "files": files,
        "comparison_model": "config-only; baseline and candidates execute these identical runtime sources",
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
        "source_identity": _collect_source_identity(repo),
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
    parser.add_argument("--pairs", type=int, default=7, help="alternating ABBA/BAAB timing blocks")
    parser.add_argument("--peak-samples", type=int, default=2)
    parser.add_argument(
        "--skip-peak-memory",
        action="store_true",
        help="skip peak-memory sampling (normally retained even with --correctness-only)",
    )
    parser.add_argument("--min-speedup", type=float, default=1.0)
    parser.add_argument("--min-paired-win-rate", type=float, default=0.75)
    parser.add_argument(
        "--require-performance",
        action="store_true",
        help="make performance thresholds part of the process exit gate",
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
    if args.require_performance and args.correctness_only:
        parser.error("--require-performance cannot be combined with --correctness-only")
    if not args.list_profiles and not args.correctness_only and not args.exclusive_gpu:
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
    stage1_grid = workspace.stage1_max_m_blocks * (INTERMEDIATE // training_tile_n)
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
    return {
        "actual_padded_rows": actual_padded,
        "predicted_padded_rows": expected["actual_padded_rows"],
        "padded_rows_match_prediction": actual_padded == expected["actual_padded_rows"],
        "workspace_gates": workspace_gates,
        "workspace_gates_passed": all(workspace_gates.values()),
        "workspace_capacity_padded_rows": workspace.max_padded_tokens,
        "route_metadata_blocks": actual_padded // config.route_tile_m,
        "stage1": {
            "effective_tile_n": training_tile_n,
            "active_logical_workgroups": active_stage1,
            "capacity_m_blocks": workspace.stage1_max_m_blocks,
            "launch_grid": stage1_grid,
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
        "persistent_stage2": config.persistent_stage2,
        "stage2_pipeline_stages": config.stage2_pipeline_stages,
        "stage2_output_mode": config.stage2_output_mode,
        "activation": config.activation,
        "compute_dtype": config.compute_dtype,
    }


def _artifact_resources(launcher) -> dict[str, Any]:
    artifacts = list(getattr(launcher, "_mem_cache", {}).values())
    if not artifacts:
        return {}
    ir_text = artifacts[-1].ir
    matches = list(re.finditer(r'gpu\.kernel_metadata<"([^"]+)"', ir_text))
    if not matches:
        return {}
    start = matches[-1].start()
    metadata = ir_text[start : start + 2500]
    result: dict[str, Any] = {"kernel": matches[-1].group(1)}
    for key in (
        "agpr_count",
        "group_segment_fixed_size",
        "private_segment_fixed_size",
        "sgpr_count",
        "sgpr_spill_count",
        "vgpr_count",
        "vgpr_spill_count",
        "wavefront_size",
    ):
        match = re.search(rf"{key} = (\d+) : i64", metadata)
        if match:
            result[key] = int(match.group(1))
    return result


def _isa_resources(dump_dir: Path | None, kernel_name: str | None) -> dict[str, Any]:
    if dump_dir is None or kernel_name is None:
        return {}
    files = sorted((dump_dir / kernel_name).glob("*_final_isa.s"))
    if not files:
        return {}
    text = files[-1].read_text(encoding="utf-8")
    result: dict[str, Any] = {"isa_path": str(files[-1])}
    for field in (
        "group_segment_fixed_size",
        "private_segment_fixed_size",
        "next_free_vgpr",
        "next_free_sgpr",
        "accum_offset",
    ):
        match = re.search(rf"\.amdhsa_{field}\s+(\d+)", text)
        if match:
            result[f"isa_{field}"] = int(match.group(1))
    result["isa_uses_flat_scratch"] = bool(re.search(r"\.uses_flat_scratch,\s+1", text))
    result["isa_scratch_instructions"] = len(re.findall(r"\b(?:buffer_|flat_)?scratch_(?:load|store)\w*", text))
    result["isa_mfma_instructions"] = len(re.findall(r"\bv_mfma_", text))
    result["isa_buffer_load_lds_instructions"] = len(re.findall(r"\bbuffer_load_\w+.*\blds\b", text))
    return result


def _launcher_resources(sonic, config, api: str, dump_dir: Path | None, device_index: int) -> dict[str, Any]:
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
    resources = {}
    for name, launcher in (("stage1", stage1), ("stage2", stage2)):
        entry = _artifact_resources(launcher)
        entry.update(_isa_resources(dump_dir, entry.get("kernel")))
        resources[name] = entry
    return resources


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
    repeats: int,
    dump_dir: Path | None,
) -> dict[str, Any]:
    baseline_output, baseline_state = _run_api(baseline_op, api, x, ids, scores, baseline_out)
    candidate_output, candidate_state = _run_api(candidate_op, api, x, ids, scores, candidate_out)
    torch.cuda.synchronize()
    output_metrics = _tensor_metrics(torch, candidate_output, baseline_output, OUTPUT_LIMITS)
    state_metrics = None
    if api == "training":
        assert baseline_state is not None and candidate_state is not None
        state_metrics = _tensor_metrics(
            torch,
            candidate_state.preactivation,
            baseline_state.preactivation,
            STATE_LIMITS,
        )

    first_output = candidate_output.clone()
    first_state = None if candidate_state is None else candidate_state.preactivation
    repeatability = []
    for repeat in range(1, repeats):
        repeated_output, repeated_state = _run_api(candidate_op, api, x, ids, scores, candidate_out)
        torch.cuda.synchronize()
        entry = {
            "repeat": repeat,
            "output_bitwise_equal": bool(torch.equal(repeated_output, first_output)),
        }
        if api == "training":
            assert repeated_state is not None and first_state is not None
            entry["state_bitwise_equal"] = bool(torch.equal(repeated_state.preactivation, first_state))
        repeatability.append(entry)
        del repeated_state

    output_bitwise_required = candidate_op.config.stage2_output_mode == "reduce"
    output_bitwise_passed = all(entry["output_bitwise_equal"] for entry in repeatability)
    state_bitwise_required = api == "training"
    state_bitwise_passed = all(entry.get("state_bitwise_equal", True) for entry in repeatability)
    repeatability_passed = (not output_bitwise_required or output_bitwise_passed) and (
        not state_bitwise_required or state_bitwise_passed
    )
    runtime_topology = {
        "baseline": _runtime_topology(sonic, baseline_op.config, baseline_op.workspace, case, api),
        "candidate": _runtime_topology(sonic, candidate_op.config, candidate_op.workspace, case, api),
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
    passed = bool(
        output_metrics["passed"]
        and (state_metrics is None or state_metrics["passed"])
        and repeatability_passed
        and topology_passed
    )
    result = {
        "candidate_vs_baseline": {
            "output": output_metrics,
            "training_preactivation": state_metrics,
        },
        "candidate_repeatability": {
            "samples": repeatability,
            "output_bitwise_required": output_bitwise_required,
            "output_bitwise_passed": output_bitwise_passed,
            "training_state_bitwise_required": state_bitwise_required,
            "training_state_bitwise_passed": state_bitwise_passed,
            "passed": repeatability_passed,
        },
        "runtime_topology": runtime_topology,
        "resources": resources,
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
    blocks = []
    by_mode = {
        "baseline": {"device": [], "host": []},
        "candidate": {"device": [], "host": []},
    }
    orders = (
        ("ABBA", ("baseline", "candidate", "candidate", "baseline")),
        ("BAAB", ("candidate", "baseline", "baseline", "candidate")),
    )
    for pair in range(pairs):
        order_name, order = orders[pair % 2]
        block_samples = {"baseline": [], "candidate": []}
        for position, mode in enumerate(order):
            sample = _event_time(torch, calls[mode])
            by_mode[mode]["device"].append(sample["device_ms"])
            by_mode[mode]["host"].append(sample["host_ms"])
            block_samples[mode].append(sample["device_ms"])
            raw_samples.append(
                {
                    "pair": pair,
                    "order": order_name,
                    "position": position,
                    "mode": mode,
                    **sample,
                }
            )
        baseline_mean = statistics.mean(block_samples["baseline"])
        candidate_mean = statistics.mean(block_samples["candidate"])
        blocks.append(
            {
                "pair": pair,
                "order": order_name,
                "baseline_mean_device_ms": baseline_mean,
                "candidate_mean_device_ms": candidate_mean,
                "speedup": baseline_mean / candidate_mean,
                "candidate_faster": candidate_mean < baseline_mean,
            }
        )
    baseline_device = _stats(by_mode["baseline"]["device"])
    candidate_device = _stats(by_mode["candidate"]["device"])
    paired_speedups = [block["speedup"] for block in blocks]
    return {
        "raw_samples": raw_samples,
        "blocks": blocks,
        "summary": {
            "baseline_device": baseline_device,
            "candidate_device": candidate_device,
            "baseline_host": _stats(by_mode["baseline"]["host"]),
            "candidate_host": _stats(by_mode["candidate"]["host"]),
            "median_speedup": baseline_device["median_ms"] / candidate_device["median_ms"],
            "paired_speedup_median": statistics.median(paired_speedups),
            "paired_win_rate": sum(block["candidate_faster"] for block in blocks) / len(blocks),
        },
    }


def _peak_sample(torch, call: Callable[[], Any]) -> dict[str, int]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    resident_allocated = torch.cuda.memory_allocated()
    resident_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    value = call()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    del value
    gc.collect()
    torch.cuda.synchronize()
    return {
        "resident_allocated_bytes": resident_allocated,
        "resident_reserved_bytes": resident_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_delta_bytes": peak_allocated - resident_allocated,
        "peak_reserved_delta_bytes": peak_reserved - resident_reserved,
    }


def _measure_peaks(torch, calls: dict[str, Callable[[], Any]], samples: int) -> dict[str, Any]:
    raw = {"baseline": [], "candidate": []}
    for sample in range(samples):
        order = ("baseline", "candidate") if sample % 2 == 0 else ("candidate", "baseline")
        for mode in order:
            raw[mode].append(
                {
                    "sample": sample,
                    "order": "AB" if sample % 2 == 0 else "BA",
                    **_peak_sample(torch, calls[mode]),
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
    return {"raw_samples": raw, "summary": summary}


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


def _execute(args: argparse.Namespace, repo: Path) -> dict[str, Any]:
    if args.dump_dir is not None:
        args.dump_dir = args.dump_dir.resolve()
        os.environ["FLYDSL_DUMP_IR"] = "1"
        os.environ["FLYDSL_DUMP_DIR"] = str(args.dump_dir)

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
    routing_inputs = {}
    for case_index, case in enumerate(dict.fromkeys(args.cases)):
        route_generator = torch.Generator(device="cuda").manual_seed(args.seed + 1000 + case_index)
        routing_inputs[case] = _make_routing(torch, case, route_generator)
    report: dict[str, Any] = {
        "schema": "flydsl.sonic_e896_forward_acceptance.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "exclusive_gpu_asserted": args.exclusive_gpu,
        "timing_status": "skipped (--correctness-only)" if args.correctness_only else "measured",
        "device": _device_identity(torch, args.device, arch),
        "source_identity": _collect_source_identity(repo),
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
    performance_all = True
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
                else:
                    api_result["peak_memory"] = _measure_peaks(torch, calls, args.peak_samples)

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
                api_result["acceptance_passed"] = correctness["passed"] and performance_passed
                case_result["apis"][api] = api_result
                correctness_all &= correctness["passed"]
                performance_all &= performance_passed
            profile_result["cases"][case] = case_result
        profile_result["correctness_passed"] = all(
            api_result["correctness"]["passed"]
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
    report["performance_passed"] = None if args.correctness_only else bool(performance_all)
    report["performance_required"] = args.require_performance
    report["passed"] = bool(correctness_all and (not args.require_performance or performance_all))
    return report


def main() -> None:
    args = _parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.list_profiles:
        _emit(_plan_payload(args, repo), args.output)
        return
    report = _execute(args, repo)
    _emit(report, args.output)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
