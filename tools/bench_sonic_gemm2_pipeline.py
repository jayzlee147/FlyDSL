#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compare serial and two-stage gfx950 SonicMoE stage-2 kernels.

The four cases are the production buckets used while tuning SonicMoE.  Both
orders are measured to expose cache/thermal drift.  Stage-2 timings contain
only the grouped down projection; end-to-end timings contain routing, sorting,
both GEMMs, and output initialization.

Example::

  PYTHONPATH=. python tools/bench_sonic_gemm2_pipeline.py \
      --cases t1 t128-balanced t128-hot16 t4096 --check
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
from typing import Any, Callable


@dataclass(frozen=True)
class Case:
    name: str
    tokens: int
    hidden: int
    intermediate: int
    experts: int
    topk: int
    distribution: str


CASES = {
    "t1": Case("t1", 1, 3584, 512, 896, 16, "balanced"),
    "t128-balanced": Case("t128-balanced", 128, 3584, 512, 896, 16, "balanced"),
    "t128-hot16": Case("t128-hot16", 128, 3584, 512, 896, 16, "hot16"),
    "t4096": Case("t4096", 4096, 4096, 2048, 64, 8, "balanced"),
}


def _config(sonic, case: Case):
    common = dict(
        hidden_size=case.hidden,
        intermediate_size=case.intermediate,
        num_experts=case.experts,
        top_k=case.topk,
        compute_dtype="bf16",
    )
    if case.name == "t1":
        return sonic.SonicMoEConfig(
            **common,
            tile_m=16,
            tile_n=64,
            tile_k=128,
            down_tile_m=16,
            down_tile_n=128,
            down_tile_k=128,
            stage1_k_wave=2,
            stage2_xcd_swizzle=1,
        )
    if case.name.startswith("t128"):
        return sonic.SonicMoEConfig(
            **common,
            tile_m=16,
            tile_n=128,
            tile_k=128,
            down_tile_m=16,
            down_tile_n=64,
            down_tile_k=128,
            stage1_k_wave=4,
            stage2_xcd_swizzle=1,
        )
    return sonic.SonicMoEConfig(
        **common,
        tile_m=128,
        tile_n=256,
        tile_k=64,
        down_tile_m=128,
        down_tile_n=128,
        down_tile_k=64,
        stage1_k_wave=1,
        stage2_xcd_swizzle=8,
    )


def _accuracy(actual, expected) -> dict[str, float]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    expected_norm = expected_f32.norm()
    return {
        "cosine": __import__("torch").nn.functional.cosine_similarity(
            actual_f32.flatten(), expected_f32.flatten(), dim=0
        ).item(),
        "relative_l2": ((actual_f32 - expected_f32).norm() / expected_norm).item(),
        "max_abs": (actual_f32 - expected_f32).abs().max().item(),
    }


def _artifact_resources(launcher) -> dict[str, Any]:
    artifacts = list(launcher._mem_cache.values())
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
    result["isa_scratch_instructions"] = len(
        re.findall(r"\b(?:buffer_|flat_)?scratch_(?:load|store)\w*", text)
    )
    result["isa_mfma_instructions"] = len(re.findall(r"\bv_mfma_", text))
    result["isa_buffer_load_lds_instructions"] = len(
        re.findall(r"\bbuffer_load_\w+.*\blds\b", text)
    )
    result["isa_ds_read_instructions"] = len(re.findall(r"\bds_read_", text))
    return result


def _measure_ordered_pair(
    torch,
    first_name: str,
    first: Callable[[], None],
    second_name: str,
    second: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    repeats: int,
    reset: dict[str, Callable[[], None]] | None = None,
) -> dict[str, list[float]]:
    for _ in range(warmup):
        first()
        second()
    torch.cuda.synchronize()
    samples = {first_name: [], second_name: []}
    for _ in range(repeats):
        for name, fn in ((first_name, first), (second_name, second)):
            if reset is not None:
                reset[name]()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(iters):
                fn()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000.0 / iters)
    return samples


def _summarize(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def _run_case(torch, sonic, compile_gemm2, gemm2_grid, run_compiled, case: Case, args):
    cfg = _config(sonic, case)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    x = torch.randn((case.tokens, case.hidden), device=device, dtype=torch.bfloat16)
    w1 = torch.randn(
        (case.experts, 2 * case.intermediate, case.hidden),
        device=device,
        dtype=torch.bfloat16,
    ) / math.sqrt(case.hidden)
    w2 = torch.randn(
        (case.experts, case.hidden, case.intermediate),
        device=device,
        dtype=torch.bfloat16,
    ) / math.sqrt(case.intermediate)
    logits = torch.randn((case.tokens, case.experts), device=device, dtype=torch.bfloat16)
    if case.distribution == "hot16":
        logits[:, 16:] = -100.0

    weights = sonic.prepare_sonic_bf16_weights(w1, w2, cfg)
    del w1, w2
    torch.cuda.empty_cache()

    serial_op = sonic.SonicMoE(cfg, weights)
    pipeline_op = sonic.SonicMoE(cfg, weights)
    serial_out = torch.empty((case.tokens, case.hidden), device=device, dtype=torch.bfloat16)
    pipeline_out = torch.empty_like(serial_out)

    # Prime routing, Stage 1, and the established serial Stage 2.
    serial_op(x, logits, out=serial_out)
    pipeline_op(x, logits, out=pipeline_out)
    torch.cuda.synchronize()

    candidate_stages = 2
    if args.production_gate:
        candidate_stages = sonic._stage2_stages(cfg, case.tokens)
        if weights.weight_dtype != "bf16" or weights.has_bias or cfg.stage2_output_mode != "atomic":
            candidate_stages = 1
    pipeline_launcher = compile_gemm2(
        BM=cfg.stage2_tile_m,
        SORTED_BM=cfg.route_tile_m,
        NE=cfg.num_experts,
        N_OUT=cfg.hidden_size,
        D_INTER=cfg.intermediate_size,
        TILE_N=cfg.stage2_tile_n,
        TILE_K=cfg.stage2_tile_k,
        xcd_swizzle=cfg.stage2_xcd_swizzle,
        b_cache_mod=sonic._stage2_cache_mod(cfg, case.tokens),
        waves_per_eu=cfg.waves_per_eu,
        w_dtype=weights.weight_dtype,
        a_dtype=cfg.compute_dtype,
        persist=cfg.persistent_stage2,
        has_bias=weights.has_bias,
        round_projection_bf16=True,
        output_mode="atomic",
        TOPK=cfg.top_k,
        stages=candidate_stages,
    )
    serial_launcher = sonic._get_stage2_launcher(
        cfg,
        sonic._stage2_cache_mod(cfg, case.tokens),
        weights.weight_dtype,
        weights.has_bias,
        "atomic",
        1,
        device.index or 0,
    )
    stage1_launcher = sonic._get_stage1_launcher(
        cfg,
        sonic._stage1_cache_mod(cfg, case.tokens),
        weights.weight_dtype,
        weights.has_bias,
        device.index or 0,
    )

    def stage1(op):
        ws = op.workspace
        assert ws is not None
        grid = sonic.gemm1_a16w4_grid(
            cfg.tile_m,
            INTER=cfg.intermediate_size,
            TILE_N=cfg.tile_n,
            max_m_blocks=ws.stage1_max_m_blocks,
        )
        run_compiled(
            stage1_launcher,
            x.data_ptr(),
            weights.gate_up.data_ptr(),
            weights.dummy_scale.data_ptr(),
            weights.dummy_scale.data_ptr(),
            ws.sorted_expert_ids.data_ptr(),
            ws.num_valid_ids.data_ptr(),
            ws.sorted_token_ids.data_ptr(),
            case.tokens,
            int(grid),
            1.0,
            1.0,
            1.0,
            1.0,
            float("inf"),
            ws.intermediate.data_ptr(),
            torch.cuda.current_stream(device),
        )

    def stage2(op, launcher, out):
        ws = op.workspace
        assert ws is not None
        grid = gemm2_grid(
            cfg.stage2_tile_m,
            N_OUT=cfg.hidden_size,
            TILE_N=cfg.stage2_tile_n,
            max_m_blocks=ws.stage2_max_m_blocks,
            persist=cfg.persistent_stage2,
        )
        run_compiled(
            launcher,
            ws.intermediate.data_ptr(),
            weights.down.data_ptr(),
            weights.dummy_scale.data_ptr(),
            weights.dummy_scale.data_ptr(),
            ws.sorted_expert_ids.data_ptr(),
            ws.num_valid_ids.data_ptr(),
            ws.sorted_token_ids.data_ptr(),
            ws.sorted_weights.data_ptr(),
            case.tokens,
            ws.stage2_max_m_blocks,
            int(grid),
            out.data_ptr(),
            torch.cuda.current_stream(device),
        )

    def serial_stage2():
        stage2(serial_op, serial_launcher, serial_out)

    def pipeline_stage2():
        stage2(pipeline_op, pipeline_launcher, pipeline_out)

    def full(op, launcher, out):
        """Enqueue the production router and both GEMMs with an explicit Stage 2.

        Keeping this chain local avoids mutating ``SonicMoE`` instances while
        still matching the device work of the public forward path exactly.
        """

        ws = op.workspace
        assert ws is not None
        with ws._launch_lock:
            sonic.moe_softmax_sort_flydsl(
                logits,
                ws.sorted_token_ids,
                ws.sorted_weights,
                ws.sorted_expert_ids,
                ws.num_valid_ids,
                out,
                cfg.num_experts,
                cfg.top_k,
                sonic._SUPPORTED_ROUTER_DTYPES[logits.dtype],
                unit_size=cfg.route_tile_m,
                renormalize=cfg.renormalize,
                workspace=ws.sorting_workspace,
                topk_scratch=(
                    ws.router_topk_weights,
                    ws.router_topk_ids,
                    ws.router_topk_expert_indices,
                ),
                direct_single_token=True,
            )
            stage1(op)
            stage2(op, launcher, out)
        return out

    def serial_full():
        return full(serial_op, serial_launcher, serial_out)

    def pipeline_full():
        return full(pipeline_op, pipeline_launcher, pipeline_out)

    serial_full()
    pipeline_full()
    torch.cuda.synchronize()
    accuracy = _accuracy(pipeline_out, serial_out)
    accuracy_passed = (
        math.isfinite(accuracy["cosine"])
        and accuracy["cosine"] >= 0.999
        and accuracy["relative_l2"] <= 0.05
    )
    if args.check and not accuracy_passed:
        raise AssertionError(f"pipeline correctness failed for {case.name}: {accuracy}")

    kernel_ab = _measure_ordered_pair(
        torch,
        "serial",
        serial_stage2,
        "pipeline",
        pipeline_stage2,
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
        reset={"serial": serial_out.zero_, "pipeline": pipeline_out.zero_},
    )
    kernel_ba = _measure_ordered_pair(
        torch,
        "pipeline",
        pipeline_stage2,
        "serial",
        serial_stage2,
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
        reset={"serial": serial_out.zero_, "pipeline": pipeline_out.zero_},
    )
    full_ab = _measure_ordered_pair(
        torch,
        "serial",
        serial_full,
        "pipeline",
        pipeline_full,
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
    )
    full_ba = _measure_ordered_pair(
        torch,
        "pipeline",
        pipeline_full,
        "serial",
        serial_full,
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
    )

    serial_resources = _artifact_resources(serial_launcher)
    pipeline_resources = _artifact_resources(pipeline_launcher)
    serial_resources.update(_isa_resources(args.dump_dir, serial_resources.get("kernel")))
    pipeline_resources.update(_isa_resources(args.dump_dir, pipeline_resources.get("kernel")))
    ws = serial_op.workspace
    assert ws is not None
    result = {
        "case": asdict(case),
        "tiles": {
            "stage1": [cfg.tile_m, cfg.tile_n, cfg.tile_k, cfg.stage1_k_wave],
            "stage2": [cfg.stage2_tile_m, cfg.stage2_tile_n, cfg.stage2_tile_k],
        },
        "candidate_stages": candidate_stages,
        "padded_rows": int(ws.num_valid_ids[0].item()),
        "correctness_vs_serial": accuracy,
        "correctness_passed": accuracy_passed,
        "kernel_us": {
            "ab": {name: _summarize(values) for name, values in kernel_ab.items()},
            "ba": {name: _summarize(values) for name, values in kernel_ba.items()},
        },
        "end_to_end_us": {
            "ab": {name: _summarize(values) for name, values in full_ab.items()},
            "ba": {name: _summarize(values) for name, values in full_ba.items()},
        },
        "resources": {"serial": serial_resources, "pipeline": pipeline_resources},
    }
    print(json.dumps(result), flush=True)
    serial_op.clear_workspace()
    pipeline_op.clear_workspace()
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--production-gate",
        action="store_true",
        help="benchmark the production-selected stage depth instead of forcing stages=2",
    )
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters <= 0 or args.repeats <= 0:
        parser.error("warmup must be non-negative; iters/repeats must be positive")
    if args.dump_dir is not None:
        args.dump_dir = args.dump_dir.resolve()
        os.environ["FLYDSL_DUMP_IR"] = "1"
        os.environ["FLYDSL_DUMP_DIR"] = str(args.dump_dir)

    import torch

    import kernels.moe.sonic as sonic
    from kernels.common.tensor_shim import _run_compiled
    from kernels.moe.moe_2stage_a16wmix.gemm2 import (
        compile_gemm2_a16w4_port,
        gemm2_a16w4_grid,
    )

    if not torch.cuda.is_available():
        parser.error("a ROCm GPU is required")
    properties = torch.cuda.get_device_properties(0)
    if "gfx950" not in properties.gcnArchName:
        parser.error(f"gfx950 is required, found {properties.gcnArchName}")

    results = [
        _run_case(
            torch,
            sonic,
            compile_gemm2_a16w4_port,
            gemm2_a16w4_grid,
            _run_compiled,
            CASES[name],
            args,
        )
        for name in args.cases
    ]
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
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
        "results": results,
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"JSON: {args.output}")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
