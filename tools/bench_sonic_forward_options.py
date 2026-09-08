#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Paired gfx950 benchmarks for SonicMoE forward tuning options.

The default cases are the two large forward shapes currently being tuned.  A
case prepares its BF16 weights exactly once; every candidate in that process
reuses those weights, the same input, and the same router logits.  Each metric
is measured in both requested orders (``AB`` and ``BA`` by default) so that a
winner is not inferred from a single cache/thermal ordering.

Built-in comparisons change one option at a time::

  PYTHONPATH=. python tools/bench_sonic_forward_options.py --case e128 \\
      --comparison stage2-stages --check
  PYTHONPATH=. python tools/bench_sonic_forward_options.py --case e8 \\
      --comparison padded-rows lds-swizzle atomic-reduce \\
      --orders abba baab --output /tmp/sonic-forward.json

Arbitrary tile/cache/XCD experiments use the ``custom`` comparison.  Values
are JSON scalars when possible, with unquoted strings accepted as a fallback::

  PYTHONPATH=. python tools/bench_sonic_forward_options.py \\
      --case e128 --comparison custom \\
      --variant-a tile_m=64 --variant-a stage1_b_cache_mod=0 \\
      --variant-b tile_m=128 --variant-b stage1_b_cache_mod=2

For formal results, repeat each command in a fresh process with both
``--allocation-order ab`` and ``--allocation-order ba``.  Launch order and
allocation order are independent controls: reversing only launch order cannot
remove stable address-placement effects.

``--compile-only`` still performs one smoke launch per variant.  That is
intentional: it materializes the device artifact, validates the complete call,
and makes kernel metadata/ISA resource reporting reliable, while skipping all
timed loops.
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
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
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

METRICS = ("stage1", "stage2-kernel", "stage2-path", "end-to-end")

COMPARISONS: dict[str, tuple[str, dict[str, Any], str, dict[str, Any]]] = {
    "stage2-stages": (
        "stages1",
        {"stage2_pipeline_stages": 1},
        "stages2",
        {"stage2_pipeline_stages": 2},
    ),
    "padded-rows": (
        "masked-epilogue",
        {"stage1_write_padded_rows": False},
        "write-padded-rows",
        {"stage1_write_padded_rows": True},
    ),
    "lds-swizzle": (
        "linear-lds",
        {"stage1_lds_swizzle": False},
        "xor-lds",
        {"stage1_lds_swizzle": True},
    ),
    "atomic-reduce": (
        "atomic",
        {"stage2_output_mode": "atomic"},
        "reduce",
        {"stage2_output_mode": "reduce"},
    ),
    "custom": ("a", {}, "b", {}),
}

# Shape/dtype/activation and routing semantics intentionally stay fixed.  The
# remaining fields are compile-time or scheduling choices that may be swept.
TUNING_FIELDS = frozenset(
    {
        "tile_m",
        "tile_n",
        "tile_k",
        "down_tile_m",
        "down_tile_n",
        "down_tile_k",
        "stage1_b_cache_mod",
        "stage2_b_cache_mod",
        "stage1_xcd_swizzle",
        "stage1_k_wave",
        "stage2_xcd_swizzle",
        "waves_per_eu",
        "persistent_stage1",
        "persistent_stage2",
        "stage2_pipeline_stages",
        "stage1_write_padded_rows",
        "stage1_lds_swizzle",
        "stage2_output_mode",
    }
)


def _parse_overrides(values: list[str], parser: argparse.ArgumentParser, option: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for assignment in values:
        if "=" not in assignment:
            parser.error(f"{option} expects FIELD=VALUE, got {assignment!r}")
        field, raw = assignment.split("=", 1)
        if field not in TUNING_FIELDS:
            parser.error(
                f"{option} cannot change {field!r}; allowed tuning fields are {', '.join(sorted(TUNING_FIELDS))}"
            )
        if not raw:
            parser.error(f"{option} has an empty value for {field!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        result[field] = value
    return result


def _base_config(sonic, case: Case, overrides: dict[str, Any]):
    config = sonic.SonicMoEConfig(
        hidden_size=case.hidden,
        intermediate_size=case.intermediate,
        num_experts=case.experts,
        top_k=case.topk,
        tile_m=64,
        tile_n=128,
        tile_k=128,
        down_tile_m=64,
        down_tile_n=128,
        down_tile_k=128,
        stage1_k_wave=1,
        stage1_xcd_swizzle=0,
        stage2_xcd_swizzle=1,
        stage2_pipeline_stages=1,
        stage1_write_padded_rows=False,
        stage1_lds_swizzle=False,
        stage2_output_mode="atomic",
        activation="swiglu",
        compute_dtype="bf16",
    )
    return replace(config, **overrides)


def _accuracy(actual, expected) -> dict[str, float | bool]:
    import torch

    actual_f32 = actual.float()
    expected_f32 = expected.float()
    delta = actual_f32 - expected_f32
    expected_norm = expected_f32.norm()
    relative_l2 = delta.norm() / expected_norm if expected_norm.item() else delta.norm()
    return {
        "finite": bool(torch.isfinite(actual_f32).all().item()),
        "cosine": torch.nn.functional.cosine_similarity(actual_f32.flatten(), expected_f32.flatten(), dim=0).item(),
        "relative_l2": relative_l2.item(),
        "max_abs": delta.abs().max().item(),
    }


def _accuracy_passed(accuracy: dict[str, float | bool]) -> bool:
    return bool(
        accuracy["finite"]
        and math.isfinite(float(accuracy["cosine"]))
        and float(accuracy["cosine"]) >= 0.999
        and float(accuracy["relative_l2"]) <= 0.05
    )


def _artifact_resources(launcher) -> dict[str, Any]:
    if launcher is None:
        return {}
    artifacts = list(getattr(launcher, "_mem_cache", {}).values())
    if not artifacts:
        return {}
    ir_text = artifacts[-1].ir
    matches = list(re.finditer(r'gpu\.kernel_metadata<"([^"]+)"', ir_text))
    if not matches:
        return {}
    start = matches[-1].start()
    metadata = ir_text[start : start + 3000]
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
    if dump_dir is None or not kernel_name:
        return {}
    files = sorted((dump_dir / kernel_name).glob("*_final_isa.s"))
    if not files:
        return {}
    text = files[-1].read_text(encoding="utf-8")
    result: dict[str, Any] = {
        "isa_path": str(files[-1]),
        "isa_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "isa_kernel_symbol_matches": bool(
            re.search(rf"\.amdhsa_kernel\s+{re.escape(kernel_name)}(?:\s|$)", text)
        ),
    }
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
    result.update(
        {
            "isa_uses_flat_scratch": bool(re.search(r"\.uses_flat_scratch,\s+1", text)),
            "isa_scratch_instructions": len(re.findall(r"\b(?:buffer_|flat_)?scratch_(?:load|store)\w*", text)),
            "isa_mfma_instructions": len(re.findall(r"\bv_mfma_", text)),
            "isa_buffer_load_lds_instructions": len(re.findall(r"\bbuffer_load_\w+.*\blds\b", text)),
            "isa_ds_read_instructions": len(re.findall(r"\bds_read_", text)),
            "isa_global_load_instructions": len(re.findall(r"\b(?:buffer|global|flat)_load_", text)),
            "isa_waitcnt_instructions": len(re.findall(r"\bs_waitcnt\b", text)),
            "isa_barrier_instructions": len(re.findall(r"\bs_barrier\b", text)),
        }
    )
    return result


def _resources(launcher, dump_dir: Path | None) -> dict[str, Any]:
    result = _artifact_resources(launcher)
    result.update(_isa_resources(dump_dir, result.get("kernel")))
    return result


def _summarize(samples: list[float], unit: str = "us") -> dict[str, float | list[float]]:
    return {
        f"median_{unit}": statistics.median(samples),
        f"mean_{unit}": statistics.mean(samples),
        f"min_{unit}": min(samples),
        f"max_{unit}": max(samples),
        f"samples_{unit}": samples,
    }


def _measure_orders(
    torch,
    functions: dict[str, Callable[[], Any]],
    resets: dict[str, Callable[[], None]],
    *,
    warmup: int,
    iters: int,
    repeats: int,
    orders: list[str],
    batch_iters: dict[str, int] | None = None,
) -> dict[str, Any]:
    capacities = {name: iters for name in ("a", "b")}
    if batch_iters is not None:
        capacities.update({name: max(1, int(value)) for name, value in batch_iters.items()})
    result: dict[str, Any] = {}
    for order in orders:
        resets["a"]()
        resets["b"]()
        warmup_counts = {"a": 0, "b": 0}
        for _ in range(warmup):
            for name in order:
                if warmup_counts[name] and warmup_counts[name] % capacities[name] == 0:
                    resets[name]()
                functions[name]()
                warmup_counts[name] += 1
        torch.cuda.synchronize()

        samples = {"a": [], "b": []}
        rounds: list[dict[str, Any]] = []
        paired_ratios: list[float] = []
        paired_deltas: list[float] = []
        for _ in range(repeats):
            launches: list[dict[str, float | int | str]] = []
            round_samples: dict[str, list[float]] = {"a": [], "b": []}
            for position, name in enumerate(order):
                remaining = iters
                segments = []
                while remaining:
                    resets[name]()
                    segment_iters = min(remaining, capacities[name])
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(segment_iters):
                        functions[name]()
                    end.record()
                    segments.append((begin, end))
                    remaining -= segment_iters
                segments[-1][1].synchronize()
                elapsed_us = sum(begin.elapsed_time(end) for begin, end in segments) * 1000.0 / iters
                samples[name].append(elapsed_us)
                round_samples[name].append(elapsed_us)
                launches.append({"position": position, "variant": name, "us": elapsed_us})
            a_us = statistics.median(round_samples["a"])
            b_us = statistics.median(round_samples["b"])
            ratio = b_us / a_us
            delta = b_us - a_us
            paired_ratios.append(ratio)
            paired_deltas.append(delta)
            rounds.append(
                {
                    "launches": launches,
                    "paired": {
                        "a_us": a_us,
                        "b_us": b_us,
                        "b_over_a": ratio,
                        "b_minus_a_us": delta,
                    },
                }
            )
        result[order] = {
            "rounds": rounds,
            "variants": {name: _summarize(values) for name, values in samples.items()},
            "paired": {
                "b_over_a": _summarize(paired_ratios, "ratio"),
                "b_minus_a": _summarize(paired_deltas),
                "b_vs_a_percent": _summarize(
                    [(ratio - 1.0) * 100.0 for ratio in paired_ratios],
                    "percent",
                ),
            },
        }
    return result


@dataclass
class _Runtime:
    label: str
    config: Any
    op: Any
    workspace: Any
    output: Any
    stage2_outputs: list[Any]
    stage1_launcher: Any
    stage2_launcher: Any
    reduction_launcher: Any
    grid: dict[str, int]
    resolved: dict[str, Any]
    stage1: Callable[[], None]
    stage2_kernel: Callable[[], None]
    stage2_path: Callable[[], Any]
    full: Callable[[], Any]
    reset_stage2: Callable[[], None]


def _make_runtime(torch, sonic, run_compiled, case: Case, label: str, config, weights, x, logits, iters: int):
    op = sonic.SonicMoE(config, weights)
    output = torch.empty((case.tokens, case.hidden), device=x.device, dtype=torch.bfloat16)

    # The smoke call creates/routs the workspace and materializes every regular
    # forward launcher.  The resulting routing metadata is then reused by the
    # isolated Stage-1 and Stage-2 measurements.
    op(x, logits, out=output)
    workspace = op.workspace
    assert workspace is not None

    device_index = x.device.index or 0
    stage1_launcher = sonic._get_stage1_launcher(
        config,
        sonic._stage1_cache_mod(config, case.tokens),
        weights.weight_dtype,
        weights.has_bias,
        device_index,
    )
    output_mode = config.stage2_output_mode
    stages = sonic._stage2_stages(config, case.tokens)
    if weights.weight_dtype != "bf16" or weights.has_bias or output_mode != "atomic":
        stages = 1
    stage2_launcher = sonic._get_stage2_launcher(
        config,
        sonic._stage2_cache_mod(config, case.tokens),
        weights.weight_dtype,
        weights.has_bias,
        output_mode,
        stages,
        device_index,
    )
    reduction_launcher = None
    if output_mode == "reduce":
        reduction_launcher = sonic.compile_moe_reduction(
            topk=config.top_k,
            model_dim=config.hidden_size,
            dtype_str="bf16",
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
    stream = lambda: torch.cuda.current_stream(x.device)

    def stage1() -> None:
        run_compiled(
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
            stream(),
        )

    # Cycling output buffers prevents repeated atomic additions from overflowing
    # while keeping output clears outside the timed event interval.
    pool_size = min(max(1, iters), 8) if output_mode == "atomic" else 1
    stage2_outputs = [torch.empty_like(output) for _ in range(pool_size)]
    cursor = [0]

    def _next_output():
        target = stage2_outputs[cursor[0] % len(stage2_outputs)]
        cursor[0] += 1
        return target

    def _launch_stage2(target) -> None:
        run_compiled(
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
            target.data_ptr(),
            stream(),
        )

    def stage2_kernel() -> None:
        target = workspace.route_output if output_mode == "reduce" else _next_output()
        assert target is not None
        _launch_stage2(target)

    def _reduce(target) -> None:
        assert workspace.route_output is not None and reduction_launcher is not None
        route_output_ptr = sonic.flyc.from_c_void_p(sonic.fx.Uint8, workspace.route_output.data_ptr())
        target_ptr = sonic.flyc.from_c_void_p(sonic.fx.Uint8, target.data_ptr())
        unused_ptr = sonic.flyc.from_c_void_p(sonic.fx.Uint8, weights.dummy_scale.data_ptr())
        run_compiled(
            reduction_launcher,
            route_output_ptr,
            target_ptr,
            unused_ptr,
            unused_ptr,
            case.tokens,
            stream(),
        )

    def stage2_path():
        target = _next_output()
        if output_mode == "reduce":
            assert workspace.route_output is not None
            _launch_stage2(workspace.route_output)
            _reduce(target)
        else:
            _launch_stage2(target)
        return target

    def full():
        return op(x, logits, out=output)

    def reset_stage2() -> None:
        cursor[0] = 0
        if output_mode == "atomic":
            for target in stage2_outputs:
                target.zero_()

    padded_routes = int(workspace.num_valid_ids[0].item())
    grid = {
        "stage1_launch": int(grid1),
        "stage2_launch": int(grid2),
        "stage1_actual_work_tiles": padded_routes // config.tile_m * (config.intermediate_size // config.tile_n),
        "stage2_actual_work_tiles": padded_routes
        // config.stage2_tile_m
        * (config.hidden_size // config.stage2_tile_n),
    }
    return _Runtime(
        label=label,
        config=config,
        op=op,
        workspace=workspace,
        output=output,
        stage2_outputs=stage2_outputs,
        stage1_launcher=stage1_launcher,
        stage2_launcher=stage2_launcher,
        reduction_launcher=reduction_launcher,
        grid=grid,
        resolved={
            "stage1_cache_mod": sonic._stage1_cache_mod(config, case.tokens),
            "stage2_cache_mod": sonic._stage2_cache_mod(config, case.tokens),
            "stage2_pipeline_stages": stages,
            "stage2_output_pool_size": pool_size,
        },
        stage1=stage1,
        stage2_kernel=stage2_kernel,
        stage2_path=stage2_path,
        full=full,
        reset_stage2=reset_stage2,
    )


def _routing_report(torch, case: Case, a: _Runtime, b: _Runtime) -> dict[str, Any]:
    def details(runtime: _Runtime) -> dict[str, Any]:
        padded_routes = int(runtime.workspace.num_valid_ids[0].item())
        logical_routes = case.tokens * case.topk
        return {
            "logical_routes": logical_routes,
            "padded_routes": padded_routes,
            "padding_rows": padded_routes - logical_routes,
            "padding_ratio": padded_routes / logical_routes,
            "route_tile_m": runtime.config.route_tile_m,
            "stage1_max_m_blocks": runtime.workspace.stage1_max_m_blocks,
            "stage2_max_m_blocks": runtime.workspace.stage2_max_m_blocks,
            "grid": runtime.grid,
        }

    topk_ids_equal = bool(torch.equal(a.workspace.router_topk_ids, b.workspace.router_topk_ids))
    topk_weight_max_abs = float((a.workspace.router_topk_weights - b.workspace.router_topk_weights).abs().max().item())
    sorted_metadata_equal: bool | None = None
    a_padded = int(a.workspace.num_valid_ids[0].item())
    b_padded = int(b.workspace.num_valid_ids[0].item())
    if a_padded == b_padded:
        metadata_blocks = a_padded // a.config.route_tile_m
        sorted_metadata_equal = bool(
            torch.equal(
                a.workspace.sorted_token_ids[:a_padded],
                b.workspace.sorted_token_ids[:b_padded],
            )
            and a.config.route_tile_m == b.config.route_tile_m
            and torch.equal(
                a.workspace.sorted_expert_ids[:metadata_blocks],
                b.workspace.sorted_expert_ids[:metadata_blocks],
            )
            and torch.equal(
                a.workspace.sorted_weights[:a_padded],
                b.workspace.sorted_weights[:b_padded],
            )
        )
    return {
        "a": details(a),
        "b": details(b),
        "same_router_logits": True,
        "topk_ids_equal": topk_ids_equal,
        "topk_weight_max_abs": topk_weight_max_abs,
        "sorted_metadata_equal": sorted_metadata_equal,
    }


def _variant_report(runtime: _Runtime, dump_dir: Path | None) -> dict[str, Any]:
    return {
        "label": runtime.label,
        "config": asdict(runtime.config),
        "resolved": {
            "route_tile_m": runtime.config.route_tile_m,
            **runtime.resolved,
            "stage2_effective_pipeline_stages": runtime.config.stage2_effective_pipeline_stages,
            "stage2_output_mode": runtime.config.stage2_output_mode,
        },
        "resources": {
            "stage1": _resources(runtime.stage1_launcher, dump_dir),
            "stage2": _resources(runtime.stage2_launcher, dump_dir),
            "reduction": _resources(runtime.reduction_launcher, dump_dir),
        },
    }


def _run_comparison(
    torch,
    sonic,
    run_compiled,
    case: Case,
    comparison: str,
    base_config,
    weights,
    x,
    logits,
    args,
    variant_a_overrides: dict[str, Any],
    variant_b_overrides: dict[str, Any],
) -> dict[str, Any]:
    label_a, preset_a, label_b, preset_b = COMPARISONS[comparison]
    config_a = replace(base_config, **preset_a, **variant_a_overrides)
    config_b = replace(base_config, **preset_b, **variant_b_overrides)
    output_pool_iters = 1 if args.compile_only else args.iters
    specs = {
        "a": (label_a, config_a),
        "b": (label_b, config_b),
    }
    runtimes = {}
    for name in args.allocation_order:
        label, config = specs[name]
        runtimes[name] = _make_runtime(
            torch,
            sonic,
            run_compiled,
            case,
            label,
            config,
            weights,
            x,
            logits,
            output_pool_iters,
        )
    a = runtimes["a"]
    b = runtimes["b"]
    torch.cuda.synchronize()

    correctness = _accuracy(b.output, a.output)
    correctness["passed"] = _accuracy_passed(correctness)
    routing = _routing_report(torch, case, a, b)
    isolated_correctness = {}
    if args.check:
        for name, runtime in (("a", a), ("b", b)):
            runtime.stage1()
            runtime.reset_stage2()
            isolated_output = runtime.stage2_path()
            accuracy = _accuracy(isolated_output, runtime.output)
            accuracy["passed"] = _accuracy_passed(accuracy)
            isolated_correctness[name] = accuracy
    if args.check:
        if not correctness["passed"]:
            raise AssertionError(f"{case.name}/{comparison} correctness failed: {correctness}")
        if not routing["topk_ids_equal"] or routing["topk_weight_max_abs"] != 0.0:
            raise AssertionError(f"{case.name}/{comparison} did not use identical logical routing: {routing}")
        if a.config.route_tile_m == b.config.route_tile_m and routing["sorted_metadata_equal"] is not True:
            raise AssertionError(f"{case.name}/{comparison} did not use identical sorted routing: {routing}")
        for name, accuracy in isolated_correctness.items():
            if not accuracy["passed"]:
                raise AssertionError(
                    f"{case.name}/{comparison}/{name} isolated path correctness failed: {accuracy}"
                )

    result: dict[str, Any] = {
        "comparison": comparison,
        "allocation_order": args.allocation_order.upper(),
        "variants": {
            "a": _variant_report(a, args.dump_dir),
            "b": _variant_report(b, args.dump_dir),
        },
        "routing": routing,
        "correctness_b_vs_a": correctness,
        "isolated_path_vs_public": isolated_correctness or None,
        "shared_stage2_buffers": args.shared_stage2_buffers,
        "shared_stage2_correctness": None,
        "timings": None,
    }
    if not args.compile_only:
        no_reset = {"a": lambda: None, "b": lambda: None}
        reset_stage2 = {"a": a.reset_stage2, "b": b.reset_stage2}
        stage2_kernel_functions = {"a": a.stage2_kernel, "b": b.stage2_kernel}
        stage2_path_functions = {"a": a.stage2_path, "b": b.stage2_path}
        if args.shared_stage2_buffers:
            if a.config.stage2_output_mode != "atomic" or b.config.stage2_output_mode != "atomic":
                raise ValueError("--shared-stage2-buffers currently supports only atomic output")
            if a.grid["stage2_launch"] != b.grid["stage2_launch"]:
                raise ValueError("--shared-stage2-buffers requires identical Stage-2 grids")
            if a.workspace.stage2_max_m_blocks != b.workspace.stage2_max_m_blocks:
                raise ValueError(
                    "--shared-stage2-buffers requires identical Stage-2 max-M bounds"
                )
            if routing["sorted_metadata_equal"] is not True:
                raise ValueError(
                    "--shared-stage2-buffers requires identical sorted routing metadata"
                )
            shared_workspace = a.workspace
            shared_outputs = a.stage2_outputs
            shared_cursor = [0]

            def shared_reset_stage2() -> None:
                shared_cursor[0] = 0
                for target in shared_outputs:
                    target.zero_()

            def make_shared_stage2(runtime: _Runtime):
                def launch():
                    target = shared_outputs[shared_cursor[0] % len(shared_outputs)]
                    shared_cursor[0] += 1
                    run_compiled(
                        runtime.stage2_launcher,
                        shared_workspace.intermediate.data_ptr(),
                        weights.down.data_ptr(),
                        weights.dummy_scale.data_ptr(),
                        weights.dummy_scale.data_ptr(),
                        shared_workspace.sorted_expert_ids.data_ptr(),
                        shared_workspace.num_valid_ids.data_ptr(),
                        shared_workspace.sorted_token_ids.data_ptr(),
                        shared_workspace.sorted_weights.data_ptr(),
                        case.tokens,
                        shared_workspace.stage2_max_m_blocks,
                        int(runtime.grid["stage2_launch"]),
                        target.data_ptr(),
                        torch.cuda.current_stream(x.device),
                    )
                    return target

                return launch

            shared_a = make_shared_stage2(a)
            shared_b = make_shared_stage2(b)
            shared_correctness = {}
            for name, launch in (("a", shared_a), ("b", shared_b)):
                shared_reset_stage2()
                shared_output = launch()
                torch.cuda.synchronize()
                accuracy = _accuracy(shared_output, a.output)
                accuracy["passed"] = _accuracy_passed(accuracy)
                shared_correctness[name] = accuracy
                if not accuracy["passed"]:
                    raise AssertionError(
                        f"{case.name}/{comparison}/{name} shared Stage-2 correctness failed: "
                        f"{accuracy}"
                    )
            result["shared_stage2_correctness"] = shared_correctness
            stage2_kernel_functions = {"a": shared_a, "b": shared_b}
            stage2_path_functions = stage2_kernel_functions
            reset_stage2 = {"a": shared_reset_stage2, "b": shared_reset_stage2}
        common = {
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "orders": args.orders,
        }
        timings = {}
        if "stage1" in args.metrics:
            timings["stage1_kernel_us"] = _measure_orders(
                torch,
                {"a": a.stage1, "b": b.stage1},
                no_reset,
                **common,
            )
        if "end-to-end" in args.metrics:
            timings["end_to_end_us"] = _measure_orders(
                torch,
                {"a": a.full, "b": b.full},
                no_reset,
                **common,
            )
        same_output_mode = a.config.stage2_output_mode == b.config.stage2_output_mode
        requested_stage2_metrics = {
            "stage2-kernel",
            "stage2-path",
        }.intersection(args.metrics)
        if same_output_mode and requested_stage2_metrics:
            batch_iters = {
                "a": len(a.stage2_outputs) if a.config.stage2_output_mode == "atomic" else args.iters,
                "b": len(b.stage2_outputs) if b.config.stage2_output_mode == "atomic" else args.iters,
            }
            # This is only the grouped down-projection kernel.  For reduce mode
            # it intentionally excludes the following top-k reduction.
            if "stage2-kernel" in args.metrics:
                timings["stage2_kernel_us"] = _measure_orders(
                    torch,
                    stage2_kernel_functions,
                    reset_stage2,
                    batch_iters=batch_iters,
                    **common,
                )
            # This is the complete output path: projection+atomic for atomic,
            # and projection+top-k reduction for reduce.
            if "stage2-path" in args.metrics:
                timings["stage2_path_us"] = _measure_orders(
                    torch,
                    stage2_path_functions,
                    reset_stage2,
                    batch_iters=batch_iters,
                    **common,
                )
            timings["stage2_isolated_comparison"] = "valid_same_output_mode"
        elif requested_stage2_metrics:
            # Atomic needs pre-zeroed output storage while reduce overwrites a
            # route tensor and then runs a separate reduction.  Pre-clearing
            # only the atomic buffers would create asymmetric cache/TLB state;
            # compare these modes through the production end-to-end path.
            if "stage2-kernel" in args.metrics:
                timings["stage2_kernel_us"] = None
            if "stage2-path" in args.metrics:
                timings["stage2_path_us"] = None
            timings["stage2_isolated_comparison"] = "omitted_asymmetric_output_cache_conditioning"
        result["timings"] = timings
    print(json.dumps({"case": case.name, **result}), flush=True)

    a.op.clear_workspace()
    b.op.clear_workspace()
    del a, b
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _git_value(args: list[str], default: str = "unknown") -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return default


def _environment(torch, properties) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "device": {
            "name": properties.name,
            "arch": properties.gcnArchName,
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
        },
        "visibility": {
            name: os.environ.get(name)
            for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
        },
        "git": {
            "commit": _git_value(["rev-parse", "HEAD"]),
            "branch": _git_value(["branch", "--show-current"]),
            "remote": _git_value(["remote", "get-url", "lsj"]),
            "dirty": bool(_git_value(["status", "--porcelain"], default="")),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument(
        "--comparison",
        nargs="+",
        choices=COMPARISONS,
        default=["stage2-stages", "padded-rows", "lds-swizzle", "atomic-reduce"],
    )
    parser.add_argument("--base", action="append", default=[], metavar="FIELD=VALUE")
    parser.add_argument("--variant-a", action="append", default=[], metavar="FIELD=VALUE")
    parser.add_argument("--variant-b", action="append", default=[], metavar="FIELD=VALUE")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--orders",
        nargs="+",
        choices=("ab", "ba", "abba", "baab"),
        default=["ab", "ba"],
        help="paired execution orders to report independently",
    )
    parser.add_argument(
        "--allocation-order",
        choices=("ab", "ba"),
        default="ab",
        help="runtime/workspace allocation order; repeat formal runs with both values",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRICS,
        default=list(METRICS),
        help="timing groups to run; correctness smoke always covers the full isolated path",
    )
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--shared-stage2-buffers",
        action="store_true",
        help="time Stage-2 variants against identical input/metadata/output addresses",
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters <= 0 or args.repeats <= 0:
        parser.error("warmup must be non-negative; iters/repeats must be positive")
    args.base = _parse_overrides(args.base, parser, "--base")
    args.variant_a = _parse_overrides(args.variant_a, parser, "--variant-a")
    args.variant_b = _parse_overrides(args.variant_b, parser, "--variant-b")
    if args.dump_dir is not None:
        args.dump_dir = args.dump_dir.resolve()
        os.environ["FLYDSL_DUMP_IR"] = "1"
        os.environ["FLYDSL_DUMP_DIR"] = str(args.dump_dir)
    return args


def main() -> None:
    args = _parse_args()

    # Set dump-related environment variables before importing FlyDSL modules.
    import torch

    import kernels.moe.sonic as sonic
    from kernels.common.tensor_shim import _run_compiled

    if not torch.cuda.is_available():
        raise SystemExit("a ROCm GPU is required")
    properties = torch.cuda.get_device_properties(0)
    if "gfx950" not in properties.gcnArchName:
        raise SystemExit(f"gfx950 is required, found {properties.gcnArchName}")

    results: list[dict[str, Any]] = []
    for case_index, case_name in enumerate(args.case):
        case = CASES[case_name]
        base_config = _base_config(sonic, case, args.base)
        torch.manual_seed(args.seed + case_index)
        device = torch.device("cuda")
        x = torch.randn((case.tokens, case.hidden), device=device, dtype=torch.bfloat16)
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
        logits = torch.randn((case.tokens, case.experts), device=device, dtype=torch.bfloat16)
        weights = sonic.prepare_sonic_bf16_weights(w1, w2, base_config)
        del w1, w2
        torch.cuda.empty_cache()

        comparisons = [
            _run_comparison(
                torch,
                sonic,
                _run_compiled,
                case,
                comparison,
                base_config,
                weights,
                x,
                logits,
                args,
                args.variant_a,
                args.variant_b,
            )
            for comparison in args.comparison
        ]
        results.append(
            {
                "case": asdict(case),
                "seed": args.seed + case_index,
                "shared_prepared_weights": True,
                "comparisons": comparisons,
            }
        )
        del weights, x, logits
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "environment": _environment(torch, properties),
        "measurement": {
            "warmup": args.warmup,
            "iterations_per_sample": args.iters,
            "repeats": args.repeats,
            "orders": [order.upper() for order in args.orders],
            "allocation_order": args.allocation_order.upper(),
            "metrics": args.metrics,
            "compile_only": args.compile_only,
            "correctness_thresholds": {"cosine_min": 0.999, "relative_l2_max": 0.05},
            "isolated_stage2_atomic_output_pool_cap": 8,
            "atomic_reduce_selection_metric": "end_to_end_us",
        },
        "fixed_defaults": {
            "stage1_tiles": [64, 128, 128],
            "stage2_tiles": [64, 128, 128],
            "stage1_k_wave": 1,
            "stage1_xcd_swizzle": 0,
            "stage2_xcd_swizzle": 1,
            "compute_dtype": "bf16",
            "activation": "swiglu",
        },
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
