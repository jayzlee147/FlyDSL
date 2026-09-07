#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Reproducible gfx950 sweep for SonicMoE's grouped dX kernel.

The benchmark constructs the exact sorter/compact-queue layout consumed by
``compile_sonic_grouped_a16w16_nn`` without timing the sorter or queue builder.
It reports both useful work and the work expanded by per-expert BM padding.

Examples (run from the repository root):

  PYTHONPATH=. python tools/bench_sonic_grouped_dx.py \
      --cases t128-balanced t128-hot16 --tile-sweep --grid-caps 1024

  PYTHONPATH=. python tools/bench_sonic_grouped_dx.py \
      --cases t1 t4096 --config 16,128,64,2 --grid-caps 256 512 1024 0

Set ``--dump-dir /tmp/dx-isa`` in a fresh process to dump final ``.s`` files.
The JSON rows also contain compiler-reported VGPR, SGPR, LDS, and spill counts.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    name: str
    tokens: int
    hidden: int
    intermediate: int
    experts: int
    topk: int
    distribution: str
    metadata_direct: bool = False


CASES = {
    "t1": Case("t1", 1, 3584, 512, 896, 16, "balanced", True),
    "t128-balanced": Case("t128-balanced", 128, 3584, 512, 896, 16, "balanced"),
    "t128-hot16": Case("t128-hot16", 128, 3584, 512, 896, 16, "hot16"),
    "t4096": Case("t4096", 4096, 4096, 2048, 64, 8, "balanced"),
}

CURRENT_CONFIGS = {
    "t1": (16, 128, 64, 2),
    "t128-balanced": (16, 256, 64, 4),
    "t128-hot16": (16, 128, 64, 2),
    # This case is intentionally experimental: production currently keeps the
    # general per-expert GEMM for long segments.
    "t4096": (16, 128, 64, 2),
}


def _parse_config(text: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(value) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("config must be BM,BN,BK,NW") from exc
    if len(values) != 4 or min(values) <= 0:
        raise argparse.ArgumentTypeError("config must contain four positive integers: BM,BN,BK,NW")
    return values


def _tile_sweep() -> list[tuple[int, int, int, int]]:
    # Keep wave count tied to the usual N tile, then use --config for targeted
    # cross-wave comparisons.  Invalid LDS/load-layout combinations are
    # recorded as skipped rows rather than terminating the sweep.
    return [
        (bm, bn, bk, 2 if bn <= 128 else 4)
        for bm in (16, 32, 64)
        for bn in (64, 128, 256, 512)
        for bk in (32, 64, 128)
    ]


def _frequencies(case: Case) -> list[int]:
    routes = case.tokens * case.topk
    active = min(case.experts, 16 if case.distribution == "hot16" else case.experts)
    base, remainder = divmod(routes, active)
    return [base + (expert < remainder) for expert in range(active)] + [0] * (
        case.experts - active
    )


def _make_metadata(torch, case: Case, block_m: int, device):
    frequencies = _frequencies(case)
    sorted_experts: list[int] = []
    descriptors: list[int] = []
    first_rows: dict[int, int] = {}
    padded_rows = 0
    for expert, frequency in enumerate(frequencies):
        if not frequency:
            continue
        first_rows[expert] = padded_rows
        padded = math.ceil(frequency / 64) * 64
        sorted_experts.extend([expert] * (padded // 64))
        descriptors.extend(
            padded_rows // block_m + tile for tile in range(math.ceil(frequency / block_m))
        )
        padded_rows += padded

    frequency_tensor = torch.tensor(frequencies, dtype=torch.int32, device=device)
    expert_ids = torch.tensor(sorted_experts, dtype=torch.int32, device=device)
    num_valid = torch.tensor(
        [padded_rows, case.tokens * case.topk],
        dtype=torch.int32,
        device=device,
    )
    schedule = torch.tensor(
        [len(descriptors), *descriptors],
        dtype=torch.int32,
        device=device,
    )
    return frequency_tensor, expert_ids, num_valid, schedule, frequencies, first_rows, padded_rows


def _artifact_resources(launcher) -> dict[str, Any]:
    artifacts = list(launcher._mem_cache.values())
    if not artifacts:
        return {}
    ir = artifacts[-1].ir
    marker = 'gpu.kernel_metadata<"sonic_grouped_nn_'
    start = ir.find(marker)
    if start < 0:
        return {}
    metadata = ir[start : start + 2500]
    result: dict[str, Any] = {}
    name_match = re.search(r'gpu\.kernel_metadata<"([^"]+)"', metadata)
    if name_match:
        result["kernel"] = name_match.group(1)
    for key in (
        "agpr_count",
        "group_segment_fixed_size",
        "max_flat_workgroup_size",
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
    """Read occupancy/spill evidence from a ``FLYDSL_DUMP_IR`` ISA file."""

    if dump_dir is None or kernel_name is None:
        return {}
    isa_files = sorted((dump_dir / kernel_name).glob("*_final_isa.s"))
    if not isa_files:
        return {}
    text = isa_files[-1].read_text(encoding="utf-8")
    result: dict[str, Any] = {"isa_path": str(isa_files[-1])}
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
    result["isa_uses_flat_scratch"] = bool(
        re.search(r"\.uses_flat_scratch,\s+1", text)
    )
    result["isa_scratch_instructions"] = len(
        re.findall(r"\b(?:buffer_|flat_)?scratch_(?:load|store)\w*", text)
    )
    result["isa_mfma_instructions"] = len(re.findall(r"\bv_mfma_", text))
    result["isa_buffer_load_lds_instructions"] = len(
        re.findall(r"\bbuffer_load_\w+.*\blds\b", text)
    )
    result["isa_ds_read_instructions"] = len(re.findall(r"\bds_read_", text))
    return result


def _time_ms(torch, launch, warmup: int, reps: int) -> list[float]:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    timings = []
    for _ in range(reps):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        launch()
        end.record()
        end.synchronize()
        timings.append(begin.elapsed_time(end))
    return timings


def _valid_config(case: Case, config: tuple[int, int, int, int]) -> str | None:
    bm, bn, bk, nw = config
    contraction = 2 * case.intermediate
    if bm % 16 or 64 % bm:
        return "BM must be a multiple of 16 dividing sorter BM64"
    if case.hidden % bn:
        return "BN must divide output N"
    if contraction % bk or bk % 32:
        return "BK must be an MFMA-compatible divisor of contraction K"
    if nw not in (2, 4) or bn % (nw * 16):
        return "BN must be divisible by NW*16, with NW in {2,4}"
    threads = nw * 64
    vector = 8
    a_threads = min(threads, bm * bk // vector)
    b_threads = min(threads, bn * bk // vector)
    if a_threads % 64 or b_threads % 64:
        return "async-copy participants must contain whole waves"
    if bm * bk % (a_threads * vector) or bn * bk % (b_threads * vector):
        return "tile is not exactly covered by 128-bit loads"
    lds = max(2 * (bm + bn) * bk * 2, bm * bn * 2)
    if lds > 163840:
        return "LDS exceeds gfx950's 160 KiB workgroup limit"
    return None


def _check_rows(torch, a, b, out, first_rows, sample_columns: int = 32):
    experts = list(first_rows)
    for expert in {experts[0], experts[-1]}:
        row = first_rows[expert]
        expected = a[row].float() @ b[expert, :, :sample_columns].float()
        torch.testing.assert_close(
            out[row, :sample_columns].float(),
            expected,
            rtol=3e-2,
            atol=5e-2,
        )


def _run_case(torch, compile_kernel, run_compiled, case, configs, grid_caps, args):
    device = torch.device("cuda")
    contraction = 2 * case.intermediate
    routes = case.tokens * case.topk
    # Values are initialized only when checking numerics. Uninitialized values
    # exercise identical loads/MFMA/stores and make large sweeps start quickly.
    a = torch.empty((sum(math.ceil(f / 64) * 64 for f in _frequencies(case)), contraction),
                    dtype=torch.bfloat16, device=device)
    b = torch.empty((case.experts, contraction, case.hidden), dtype=torch.bfloat16, device=device)
    out = torch.empty((a.shape[0], case.hidden), dtype=torch.bfloat16, device=device)
    if args.check:
        a.uniform_(-0.02, 0.02)
        b.uniform_(-0.02, 0.02)

    metadata_by_bm = {}
    results = []
    for config in configs:
        bm, bn, bk, nw = config
        invalid = _valid_config(case, config)
        if invalid:
            results.append({"case": case.name, "config": config, "status": "skipped", "reason": invalid})
            continue
        if bm not in metadata_by_bm:
            metadata_by_bm[bm] = _make_metadata(torch, case, bm, device)
        frequency, expert_ids, num_valid, schedule, frequencies, first_rows, padded_rows = metadata_by_bm[bm]
        active_experts = sum(value > 0 for value in frequencies)
        if case.metadata_direct:
            schedule_arg = frequency
            descriptors = len(expert_ids)
        else:
            schedule_arg = schedule
            descriptors = int(schedule.numel()) - 1
        workgroups = descriptors * (case.hidden // bn)
        scheduled_rows = descriptors * bm

        try:
            launcher = compile_kernel(
                contraction_size=contraction,
                output_size=case.hidden,
                num_experts=case.experts,
                block_m=bm,
                block_n=bn,
                block_k=bk,
                stages=2,
                n_waves=nw,
                sorted_block_m=64,
                compact_grid=not case.metadata_direct,
                device_index=device.index or 0,
            )
            for grid_cap in grid_caps:
                grid = workgroups if grid_cap == 0 else min(workgroups, grid_cap)

                def launch():
                    run_compiled(
                        launcher,
                        a.data_ptr(),
                        b.data_ptr(),
                        schedule_arg.data_ptr(),
                        expert_ids.data_ptr(),
                        num_valid.data_ptr(),
                        out.data_ptr(),
                        grid,
                        torch.cuda.current_stream(device),
                    )

                timings = _time_ms(torch, launch, args.warmup, args.reps)
                if args.check:
                    _check_rows(torch, a, b, out, first_rows)
                resources = _artifact_resources(launcher)
                isa = _isa_resources(args.dump_dir, resources.get("kernel"))
                useful_flops = 2 * routes * contraction * case.hidden
                executed_flops = 2 * scheduled_rows * contraction * case.hidden
                unique_weight_bytes = active_experts * contraction * case.hidden * 2
                streamed_weight_bytes = descriptors * contraction * case.hidden * 2
                lds_bytes = resources.get("group_segment_fixed_size", 0)
                results.append(
                    {
                        "case": case.name,
                        "status": "ok",
                        "bm": bm,
                        "bn": bn,
                        "bk": bk,
                        "n_waves": nw,
                        "grid_cap": grid_cap,
                        "grid": grid,
                        "persistent_rounds": math.ceil(workgroups / grid),
                        "routes": routes,
                        "active_experts": active_experts,
                        "sorter_padded_rows": padded_rows,
                        "scheduled_rows": scheduled_rows,
                        "sorter_padding_x": padded_rows / routes,
                        "compute_padding_x": scheduled_rows / routes,
                        "descriptors": descriptors,
                        "workgroups": workgroups,
                        "unique_weight_gib": unique_weight_bytes / 2**30,
                        "streamed_weight_gib": streamed_weight_bytes / 2**30,
                        "weight_reload_x": streamed_weight_bytes / unique_weight_bytes,
                        "median_ms": statistics.median(timings),
                        "min_ms": min(timings),
                        "timings_ms": timings,
                        "useful_tflops": useful_flops / (statistics.median(timings) * 1e9),
                        "executed_tflops": executed_flops / (statistics.median(timings) * 1e9),
                        "launch_waves_per_cu": grid * nw / 256,
                        "lds_limited_workgroups_per_cu": 163840 // lds_bytes if lds_bytes else None,
                        **resources,
                        **isa,
                    }
                )
        except Exception as exc:
            results.append(
                {
                    "case": case.name,
                    "config": config,
                    "status": "error",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    return results


def _print_summary(results):
    ok = [row for row in results if row.get("status") == "ok"]
    for case_name in dict.fromkeys(row["case"] for row in ok):
        print(f"\n{case_name}")
        print(" BM  BN  BK NW  cap  grid  pad-x  rounds    ms   useful-TF  VGPR  LDS-KiB spills")
        rows = sorted(
            (row for row in ok if row["case"] == case_name),
            key=lambda row: row["median_ms"],
        )
        for row in rows:
            spills = row.get("vgpr_spill_count", 0) + row.get("sgpr_spill_count", 0)
            cap = "full" if row["grid_cap"] == 0 else str(row["grid_cap"])
            print(
                f"{row['bm']:3d} {row['bn']:3d} {row['bk']:3d} {row['n_waves']:2d} "
                f"{cap:>5} {row['grid']:5d} {row['compute_padding_x']:6.2f} "
                f"{row['persistent_rounds']:6d} {row['median_ms']:7.3f} "
                f"{row['useful_tflops']:10.1f} {row.get('vgpr_count', -1):5d} "
                f"{row.get('group_segment_fixed_size', 0) / 1024:7.1f} {spills:6d}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--config", action="append", type=_parse_config, default=[])
    parser.add_argument("--tile-sweep", action="store_true")
    parser.add_argument(
        "--grid-caps",
        nargs="+",
        type=int,
        default=[1024],
        help="0 means launch the full logical workgroup count",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dump-dir",
        type=Path,
        help="enable FlyDSL IR/final-ISA dumps; use a fresh directory/process",
    )
    args = parser.parse_args()
    if min(args.warmup, args.reps) < 0 or args.reps == 0:
        parser.error("warmup must be non-negative and reps must be positive")
    if any(cap < 0 for cap in args.grid_caps):
        parser.error("grid caps must be non-negative")
    if args.dump_dir is not None:
        os.environ["FLYDSL_DUMP_IR"] = "1"
        os.environ["FLYDSL_DUMP_DIR"] = str(args.dump_dir.resolve())

    import torch

    from kernels.common.tensor_shim import _run_compiled
    from kernels.moe.sonic_grouped_a16w16 import compile_sonic_grouped_a16w16_nn

    if not torch.cuda.is_available():
        parser.error("a ROCm GPU is required")
    properties = torch.cuda.get_device_properties(0)
    if "gfx950" not in properties.gcnArchName:
        parser.error(f"gfx950 is required, found {properties.gcnArchName}")
    torch.manual_seed(args.seed)

    explicit_configs = list(dict.fromkeys(args.config))
    results = []
    for case_name in args.cases:
        case = CASES[case_name]
        if args.tile_sweep:
            configs = _tile_sweep()
        elif explicit_configs:
            configs = explicit_configs
        else:
            configs = [CURRENT_CONFIGS[case_name]]
        results.extend(
            _run_case(
                torch,
                compile_sonic_grouped_a16w16_nn,
                _run_compiled,
                case,
                configs,
                list(dict.fromkeys(args.grid_caps)),
                args,
            )
        )
        gc.collect()
        torch.cuda.empty_cache()

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    payload = {
        "command": [sys.executable, *sys.argv],
        "git_commit": git_commit,
        "device": properties.name,
        "arch": properties.gcnArchName,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "seed": args.seed,
        "warmup": args.warmup,
        "reps": args.reps,
        "cases": [asdict(CASES[name]) for name in args.cases],
        "results": results,
    }
    _print_summary(results)
    encoded = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"\nJSON: {args.output}")
    else:
        print("\n" + encoded)


if __name__ == "__main__":
    main()
