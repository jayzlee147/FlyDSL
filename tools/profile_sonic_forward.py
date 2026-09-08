#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

r"""Profile one steady-state SonicMoE forward launch on gfx950.

Run this under rocprofv3 with selected-region collection enabled.  Setup,
weight preparation, compilation, routing, and warmup happen while collection
is paused; only the requested launch is placed in the selected region::

  PYTHONPATH=. rocprofv3 --kernel-trace --selected-regions -- \
    python tools/profile_sonic_forward.py \
      --case e128 --profile winner --kernel stage1 --output manifest.json

``stage1`` and ``stage2`` each enqueue exactly one grouped-GEMM dispatch per
iteration.  ``full`` enqueues one complete public forward call (router/sort,
both GEMMs, and output initialization) per iteration.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class Case:
    name: str
    tokens: int
    hidden: int
    intermediate: int
    experts: int
    topk: int


CASES = {
    "e128": Case("e128", 4096, 2048, 768, 128, 8),
    "e8": Case("e8", 4096, 4096, 14336, 8, 2),
}


def _config(sonic, case: Case, profile: str):
    common: dict[str, Any] = {
        "hidden_size": case.hidden,
        "intermediate_size": case.intermediate,
        "num_experts": case.experts,
        "top_k": case.topk,
        "stage1_k_wave": 1,
        "waves_per_eu": None,
        "persistent_stage1": False,
        "persistent_stage2": False,
        "stage2_output_mode": "atomic",
        "activation": "swiglu",
        "compute_dtype": "bf16",
        "stage1_lds_swizzle": False,
    }
    profiles: dict[tuple[str, str], dict[str, Any]] = {
        # Original production profile used by the formal E128 comparison.
        ("e128", "baseline"): {
            "tile_m": 64,
            "tile_n": 128,
            "tile_k": 128,
            "down_tile_m": 64,
            "down_tile_n": 128,
            "down_tile_k": 128,
            "stage1_b_cache_mod": None,
            "stage2_b_cache_mod": None,
            "stage1_xcd_swizzle": 0,
            "stage2_xcd_swizzle": 1,
            "stage2_pipeline_stages": 1,
            "stage1_write_padded_rows": False,
        },
        # Final dense E128 candidate.  The automatic pipeline gate resolves to
        # two stages only for this exact T4096 production bucket.
        ("e128", "winner"): {
            "tile_m": 128,
            "tile_n": 192,
            "tile_k": 64,
            "down_tile_m": 64,
            "down_tile_n": 256,
            "down_tile_k": 128,
            "stage1_b_cache_mod": 0,
            "stage2_b_cache_mod": 0,
            "stage1_xcd_swizzle": 8,
            "stage2_xcd_swizzle": 0,
            "stage2_pipeline_stages": None,
            "stage1_write_padded_rows": True,
        },
        # Throughput profile immediately preceding the final E8 XCD/pipeline
        # changes; the tile shapes already match the E8 winner.
        ("e8", "baseline"): {
            "tile_m": 128,
            "tile_n": 256,
            "tile_k": 64,
            "down_tile_m": 128,
            "down_tile_n": 128,
            "down_tile_k": 64,
            "stage1_b_cache_mod": 0,
            "stage2_b_cache_mod": 0,
            "stage1_xcd_swizzle": 0,
            "stage2_xcd_swizzle": 8,
            "stage2_pipeline_stages": 1,
            "stage1_write_padded_rows": False,
        },
        ("e8", "winner"): {
            "tile_m": 128,
            "tile_n": 256,
            "tile_k": 64,
            "down_tile_m": 128,
            "down_tile_n": 128,
            "down_tile_k": 64,
            "stage1_b_cache_mod": 0,
            "stage2_b_cache_mod": 0,
            "stage1_xcd_swizzle": 8,
            "stage2_xcd_swizzle": 8,
            "stage2_pipeline_stages": None,
            "stage1_write_padded_rows": False,
        },
    }
    return sonic.SonicMoEConfig(**common, **profiles[(case.name, profile)])


def _artifact_resources(launcher) -> dict[str, Any]:
    """Extract code-object metadata without triggering another launch."""

    last = getattr(launcher, "_last_compiled", None)
    artifact = last[1] if last is not None else None
    if artifact is None:
        artifacts = list(getattr(launcher, "_mem_cache", {}).values())
        artifact = artifacts[-1] if artifacts else None
    if artifact is None:
        return {"available": False}

    ir_text = getattr(artifact, "ir", "")
    matches = list(re.finditer(r'gpu\.kernel_metadata<"([^"]+)"', ir_text))
    if not matches:
        return {"available": False}
    match = matches[-1]
    metadata = ir_text[match.start() : match.start() + 3000]
    result: dict[str, Any] = {"available": True, "kernel": match.group(1)}
    for field in (
        "agpr_count",
        "group_segment_fixed_size",
        "private_segment_fixed_size",
        "sgpr_count",
        "sgpr_spill_count",
        "vgpr_count",
        "vgpr_spill_count",
        "wavefront_size",
    ):
        value = re.search(rf"{field} = (\d+) : i64", metadata)
        if value:
            result[field] = int(value.group(1))
    return result


def _roctx_control():
    names: list[str] = []
    python_lib = Path(sys.prefix) / "lib"
    for directory in python_lib.glob("python*/site-packages/_rocm_sdk_devel/lib"):
        names.extend(
            str(directory / library)
            for library in ("librocprofiler-sdk-roctx.so", "libroctx64.so")
        )
    names.extend(
        (
            "/opt/rocm/lib/librocprofiler-sdk-roctx.so",
            "/opt/rocm/lib/libroctx64.so",
            "librocprofiler-sdk-roctx.so",
            "libroctx64.so",
        )
    )
    for short_name in ("rocprofiler-sdk-roctx", "roctx64"):
        found = ctypes.util.find_library(short_name)
        if found:
            names.append(found)

    errors = []
    for name in dict.fromkeys(names):
        try:
            library = ctypes.CDLL(name)
            resume = library.roctxProfilerResume
            pause = library.roctxProfilerPause
            for function in (resume, pause):
                function.argtypes = [ctypes.c_uint64]
                function.restype = ctypes.c_int32
            return name, library
        except (AttributeError, OSError) as error:
            errors.append(f"{name}: {error}")
    raise RuntimeError(
        "cannot load ROCTx profiler control; run in a ROCm profiling image "
        "with librocprofiler-sdk-roctx.so available\n" + "\n".join(errors)
    )


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--profile", choices=("baseline", "winner"), required=True)
    parser.add_argument("--kernel", choices=("stage1", "stage2", "full"), required=True)
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="complete forward calls before profiling (also establish routing)",
    )
    parser.add_argument("--iters", type=int, default=1, help="selected launches to enqueue")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--output", type=Path, help="write the JSON manifest here instead of stdout")
    args = parser.parse_args()
    if args.warmup <= 0 or args.iters <= 0:
        parser.error("--warmup and --iters must be positive")
    return args


def _run(args: argparse.Namespace) -> dict[str, Any]:
    # Heavy imports stay here so ``--help`` works outside the ROCm image.
    import torch

    import kernels.moe.sonic as sonic
    from kernels.common.tensor_shim import _run_compiled

    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")
    properties = torch.cuda.get_device_properties(0)
    if "gfx950" not in properties.gcnArchName:
        raise RuntimeError(f"gfx950 is required, found {properties.gcnArchName}")

    case = CASES[args.case]
    config = _config(sonic, case, args.profile)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    x = torch.randn((case.tokens, case.hidden), device=device, dtype=torch.bfloat16)
    logits = torch.randn((case.tokens, case.experts), device=device, dtype=torch.bfloat16)
    w1 = torch.randn(
        (case.experts, 2 * case.intermediate, case.hidden),
        device=device,
        dtype=torch.bfloat16,
    )
    w1.mul_(1.0 / math.sqrt(case.hidden))
    w2 = torch.randn(
        (case.experts, case.hidden, case.intermediate),
        device=device,
        dtype=torch.bfloat16,
    )
    w2.mul_(1.0 / math.sqrt(case.intermediate))
    weights = sonic.prepare_sonic_bf16_weights(w1, w2, config)
    del w1, w2
    torch.cuda.empty_cache()

    op = sonic.SonicMoE(config, weights)
    output = torch.empty((case.tokens, case.hidden), device=device, dtype=torch.bfloat16)
    for _ in range(args.warmup):
        op(x, logits, out=output)
    torch.cuda.synchronize()
    workspace = op.workspace
    assert workspace is not None

    device_index = x.device.index or 0
    stage1_cache_mod = sonic._stage1_cache_mod(config, case.tokens)
    stage2_cache_mod = sonic._stage2_cache_mod(config, case.tokens)
    stage2_stages = sonic._stage2_stages(config, case.tokens)
    stage1_launcher = sonic._get_stage1_launcher(
        config, stage1_cache_mod, weights.weight_dtype, weights.has_bias, device_index
    )
    stage2_launcher = sonic._get_stage2_launcher(
        config,
        stage2_cache_mod,
        weights.weight_dtype,
        weights.has_bias,
        config.stage2_output_mode,
        stage2_stages,
        device_index,
    )
    grid1 = sonic.gemm1_a16w4_grid(
        config.tile_m,
        INTER=config.intermediate_size,
        TILE_N=config.tile_n,
        max_m_blocks=workspace.stage1_max_m_blocks,
        persist=config.persistent_stage1,
    )
    grid2 = sonic.gemm2_a16w4_grid(
        config.stage2_tile_m,
        N_OUT=config.hidden_size,
        TILE_N=config.stage2_tile_n,
        max_m_blocks=workspace.stage2_max_m_blocks,
        persist=config.persistent_stage2,
    )
    stream = torch.cuda.current_stream(x.device)

    def launch_stage1() -> None:
        _run_compiled(
            stage1_launcher,
            x.data_ptr(),
            weights.gate_up.data_ptr(),
            weights.dummy_scale.data_ptr(),
            weights.dummy_scale.data_ptr(),
            workspace.sorted_expert_ids.data_ptr(),
            workspace.num_valid_ids.data_ptr(),
            workspace.sorted_token_ids.data_ptr(),
            case.tokens,
            int(grid1),
            1.0,
            1.0,
            1.0,
            1.0,
            float("inf"),
            workspace.intermediate.data_ptr(),
            stream,
        )

    # Zeroing stays outside the selected region, so a Stage-2 profile contains
    # only the grouped down-projection dispatch.  Repeated iterations reuse the
    # target and therefore do not inject memset dispatches into the capture.
    # Do not disturb post-warmup cache state for Stage 1/full unnecessarily.
    stage2_output = torch.empty_like(output) if args.kernel == "stage2" else output
    if args.kernel == "stage2":
        stage2_output.zero_()

    def launch_stage2() -> None:
        _run_compiled(
            stage2_launcher,
            workspace.intermediate.data_ptr(),
            weights.down.data_ptr(),
            weights.dummy_scale.data_ptr(),
            weights.dummy_scale.data_ptr(),
            workspace.sorted_expert_ids.data_ptr(),
            workspace.num_valid_ids.data_ptr(),
            workspace.sorted_token_ids.data_ptr(),
            workspace.sorted_weights.data_ptr(),
            case.tokens,
            workspace.stage2_max_m_blocks,
            int(grid2),
            stage2_output.data_ptr(),
            stream,
        )

    def launch_full() -> None:
        op(x, logits, out=output)

    launches: dict[str, Callable[[], Any]] = {
        "stage1": launch_stage1,
        "stage2": launch_stage2,
        "full": launch_full,
    }
    selected = launches[args.kernel]

    padded_routes = int(workspace.num_valid_ids.reshape(-1)[0].item())
    resources = {
        "stage1": _artifact_resources(stage1_launcher),
        "stage2": _artifact_resources(stage2_launcher),
    }
    roctx_library, roctx = _roctx_control()
    torch.cuda.synchronize()
    rc = roctx.roctxProfilerResume(0)
    if rc:
        raise RuntimeError(f"roctxProfilerResume failed: {rc}")
    launch_error: BaseException | None = None
    try:
        for _ in range(args.iters):
            selected()
        torch.cuda.synchronize()
    except BaseException as error:
        launch_error = error
    finally:
        rc = roctx.roctxProfilerPause(0)
    if rc:
        raise RuntimeError(f"roctxProfilerPause failed: {rc}")
    if launch_error is not None:
        raise launch_error

    return {
        "schema_version": 1,
        "case": asdict(case),
        "profile": args.profile,
        "seed": args.seed,
        "selected_region": {
            "kernel": args.kernel,
            "iterations": args.iters,
            "scope_per_iteration": (
                "one_grouped_gemm_dispatch"
                if args.kernel in ("stage1", "stage2")
                else "one_complete_public_forward_call"
            ),
            "roctx_library": roctx_library,
        },
        "warmup": {"complete_forward_calls": args.warmup},
        "config": asdict(config),
        "resolved": {
            "route_tile_m": config.route_tile_m,
            "stage1_cache_mod": stage1_cache_mod,
            "stage2_cache_mod": stage2_cache_mod,
            "stage2_pipeline_stages": stage2_stages,
            "stage2_effective_pipeline_stages": config.stage2_effective_pipeline_stages,
        },
        "routing": {
            "logical_routes": case.tokens * case.topk,
            "padded_routes": padded_routes,
            "padding_rows": padded_routes - case.tokens * case.topk,
            "stage1_max_m_blocks": workspace.stage1_max_m_blocks,
            "stage2_max_m_blocks": workspace.stage2_max_m_blocks,
        },
        "grid": {
            "stage1_launch": int(grid1),
            "stage2_launch": int(grid2),
            "stage1_actual_work_tiles": padded_routes
            // config.tile_m
            * (config.intermediate_size // config.tile_n),
            "stage2_actual_work_tiles": padded_routes
            // config.stage2_tile_m
            * (config.hidden_size // config.stage2_tile_n),
        },
        "resources": resources,
        "environment": {
            "command": [sys.executable, *sys.argv],
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "device": {
                "name": properties.name,
                "arch": properties.gcnArchName,
                "multiprocessor_count": properties.multi_processor_count,
            },
            "visibility": {
                name: os.environ.get(name)
                for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
            },
            "git": {
                "commit": _git("rev-parse", "HEAD"),
                "branch": _git("branch", "--show-current"),
                "dirty": bool(_git("status", "--porcelain")),
            },
        },
    }


def main() -> None:
    args = _parse_args()
    manifest = _run(args)
    encoded = json.dumps(manifest, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="", flush=True)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(f"JSON: {args.output}", flush=True)


if __name__ == "__main__":
    main()
