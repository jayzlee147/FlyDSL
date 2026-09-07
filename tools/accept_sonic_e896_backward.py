#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Acceptance benchmark for the retained-state E896 SonicMoE backward.

This tool compares the current checkout against an explicitly loaded baseline
``sonic_backward.py`` for the default adapter-equivalent
T4096/H3584/I512/E896/K16 BF16 SwiGLU contract.  It deliberately exercises the
public non-concatenated GLU layout: forward receives separated ``[gate | up]``
weights, while backward consumes native interleaved ``[g0, u0, ...]`` weights
and a retained forward state with the same interleaved layout.

Timing is refused unless ``--exclusive-gpu`` is supplied.  On a shared/busy
machine use ``--correctness-only``; this still checks both routing regimes,
four-gradient accuracy, repeated bitwise determinism, code isolation, and the
expected launch topology.

Examples
--------
Correctness and launch audit only::

    PYTHONPATH=. python tools/accept_sonic_e896_backward.py \
        --correctness-only --cases balanced hot16 \
        --output /tmp/sonic-e896-correctness.json

Formal timing on an independently reserved gfx950::

    PYTHONPATH=. python tools/accept_sonic_e896_backward.py \
        --exclusive-gpu --pairs 11 --cases balanced hot16 \
        --output /tmp/sonic-e896-abba.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

import kernels.moe.sonic_backward as candidate_module
from flydsl.runtime.device import get_rocm_arch
from kernels.moe.sonic import SonicMoE, SonicMoEConfig, prepare_sonic_bf16_weights

TOKENS = 4096
HIDDEN = 3584
INTERMEDIATE = 512
EXPERTS = 896
TOPK = 16
GRADIENT_NAMES = ("dx", "dw1", "dw2", "dtopk_weights")
ROUTING_CASES = ("balanced", "hot16")
CHUNK_ELEMENTS = 32 * 1024 * 1024

# These are the production large-shape A16 bounds already used by the opt-in
# T4096 backward reference test.  The candidate changes contraction ordering:
# elementwise equality with the legacy path is not the numerical contract, but
# relative-L2, maximum absolute drift, and norm preservation all are.
RELATIVE_L2_LIMITS = {
    "dx": 7.5e-4,
    "dw1": 3.0e-4,
    "dw2": 1.0e-4,
    "dtopk_weights": 1.0e-4,
}
MAX_ABS_LIMITS = {
    "dx": 0.0625,
    "dw1": 0.5,
    "dw2": 0.5,
    "dtopk_weights": 0.25,
}
NORM_RATIO_TOLERANCE = 1.0e-5

# The benchmark isolates sonic_backward.py only.  Require all runtime sources
# shared by the baseline and candidate to be byte-identical so that an ABBA
# result cannot silently compare two different forward/sorter/GEMM stacks.
SHARED_RUNTIME_SOURCES = (
    "kernels/common/buffer_ops.py",
    "kernels/common/kernels_common.py",
    "kernels/common/mem_ops.py",
    "kernels/common/tensor_shim.py",
    "kernels/gemm/gemm_a16w16_gfx950.py",
    "kernels/moe/grouped_da_gfx950.py",
    "kernels/moe/moe_2stage_a16wmix/gemm1.py",
    "kernels/moe/moe_2stage_a16wmix/gemm2.py",
    "kernels/moe/moe_gemm_2stage/moe_reduce.py",
    "kernels/moe/moe_ragged_sorting_kernel.py",
    "kernels/moe/moe_sorting_kernel.py",
    "kernels/moe/sonic.py",
    "kernels/moe/sonic_grouped_a16w16.py",
    "kernels/moe/sonic_grouped_scheduler.py",
    "kernels/moe/sonic_grouped_tn.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_identity(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


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


def _repo_identity(repo: Path, backward_path: Path) -> dict[str, Any]:
    return {
        "root": str(repo.resolve()),
        "head": _git_value(repo, "rev-parse", "HEAD"),
        "branch": _git_value(repo, "branch", "--show-current"),
        "backward_status": _git_value(repo, "status", "--short", "--", str(backward_path)),
    }


def _load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_isolated_baseline(path: Path):
    module = _load_source_module("kernels.moe._e896_acceptance_baseline", path)
    loaded_from = Path(inspect.getsourcefile(module.sonic_moe_backward) or "").resolve()
    if loaded_from != path:
        raise RuntimeError(f"baseline sonic_moe_backward isolation failed: expected {path}, loaded {loaded_from}")
    return module


def _collect_code_identity(baseline_path: Path) -> dict[str, Any]:
    candidate_path = Path(candidate_module.__file__).resolve()
    candidate_repo = Path(__file__).resolve().parents[1]
    baseline_repo = baseline_path.parents[2]
    expected_candidate = candidate_repo / "kernels/moe/sonic_backward.py"
    if candidate_path != expected_candidate:
        raise RuntimeError(
            "candidate import does not resolve to this checkout: "
            f"expected {expected_candidate}, loaded {candidate_path}. "
            "Run with PYTHONPATH=. from the candidate repository root."
        )
    if baseline_path == candidate_path:
        raise RuntimeError("candidate and baseline sonic_backward.py resolve to the same file")
    if _sha256(baseline_path) == _sha256(candidate_path):
        raise RuntimeError("candidate and baseline sonic_backward.py are byte-identical")

    shared_sources: dict[str, Any] = {}
    mismatches = []
    for relative in SHARED_RUNTIME_SOURCES:
        baseline_source = baseline_repo / relative
        candidate_source = candidate_repo / relative
        baseline_identity = _file_identity(baseline_source)
        candidate_identity = _file_identity(candidate_source)
        identical = baseline_identity["sha256"] == candidate_identity["sha256"]
        shared_sources[relative] = {
            "baseline": baseline_identity,
            "candidate": candidate_identity,
            "identical": identical,
        }
        if not identical:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(
            "this single-file-isolation benchmark requires byte-identical shared "
            "runtime sources; mismatches: " + ", ".join(mismatches)
        )

    return {
        "acceptance_tool": _file_identity(Path(__file__)),
        "baseline": {
            "sonic_backward": _file_identity(baseline_path),
            "git": _repo_identity(baseline_repo, baseline_path),
        },
        "candidate": {
            "sonic_backward": _file_identity(candidate_path),
            "git": _repo_identity(candidate_repo, candidate_path),
        },
        "shared_runtime_sources": shared_sources,
        "shared_runtime_sources_identical": True,
    }


def _adapter_config() -> SonicMoEConfig:
    """Return the adapter's automatic conservative profile for this shape."""

    return SonicMoEConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts=EXPERTS,
        top_k=TOPK,
        tile_m=64,
        tile_n=128,
        tile_k=128,
        down_tile_m=64,
        down_tile_n=128,
        down_tile_k=128,
        renormalize=False,
        stage1_xcd_swizzle=0,
        stage1_k_wave=1,
        stage2_xcd_swizzle=1,
        activation="swiglu",
        compute_dtype="bf16",
    )


def _config_dict(config: SonicMoEConfig) -> dict[str, Any]:
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
        "stage1_xcd_swizzle": config.stage1_xcd_swizzle,
        "stage1_k_wave": config.stage1_k_wave,
        "stage2_xcd_swizzle": config.stage2_xcd_swizzle,
        "activation": config.activation,
        "compute_dtype": config.compute_dtype,
    }


def _interleave_glu_rows(separated: torch.Tensor) -> torch.Tensor:
    gate, up = separated.chunk(2, dim=1)
    return torch.stack((gate, up), dim=2).flatten(1, 2).contiguous()


def _routing(case: str, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
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
    if case == "balanced":
        minimum = (TOKENS * TOPK) // EXPERTS
        maximum = minimum + int((TOKENS * TOPK) % EXPERTS != 0)
        active = EXPERTS
    else:
        minimum = maximum = TOKENS
        active = TOPK
    return (
        ids,
        scores,
        {
            "routes": TOKENS * TOPK,
            "active_experts": active,
            "active_frequency_min": minimum,
            "active_frequency_max": maximum,
        },
    )


def _compare_gradients(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    if len(actual) != len(GRADIENT_NAMES) or len(expected) != len(GRADIENT_NAMES):
        raise RuntimeError(f"expected four gradients, got candidate={len(actual)} baseline={len(expected)}")

    comparison: dict[str, Any] = {}
    for name, actual_gradient, expected_gradient in zip(GRADIENT_NAMES, actual, expected):
        if actual_gradient.shape != expected_gradient.shape:
            raise RuntimeError(
                f"{name} shape mismatch: candidate={tuple(actual_gradient.shape)}, "
                f"baseline={tuple(expected_gradient.shape)}"
            )
        actual_flat = actual_gradient.detach().reshape(-1)
        expected_flat = expected_gradient.detach().reshape(-1)
        actual_sq = 0.0
        reference_sq = 0.0
        difference_sq = 0.0
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
            reference_sq += float(torch.sum(expected_chunk * expected_chunk))
            difference_sq += float(torch.sum(difference * difference))
            if difference.numel():
                max_abs = max(max_abs, float(difference.abs().max()))

        relative_l2 = (difference_sq / reference_sq) ** 0.5 if reference_sq else difference_sq**0.5
        if reference_sq:
            norm_ratio = (actual_sq / reference_sq) ** 0.5
        else:
            norm_ratio = 1.0 if actual_sq == 0.0 else float("inf")
        passed = bool(
            finite
            and max_abs <= MAX_ABS_LIMITS[name]
            and relative_l2 <= RELATIVE_L2_LIMITS[name]
            and abs(norm_ratio - 1.0) <= NORM_RATIO_TOLERANCE
        )
        comparison[name] = {
            "shape": list(actual_gradient.shape),
            "dtype": str(actual_gradient.dtype),
            "bitwise_equal": bool(torch.equal(actual_gradient, expected_gradient)),
            "finite": finite,
            "max_abs": max_abs,
            "max_abs_limit": MAX_ABS_LIMITS[name],
            "relative_l2": relative_l2,
            "relative_l2_limit": RELATIVE_L2_LIMITS[name],
            "norm_ratio": norm_ratio,
            "norm_ratio_tolerance": NORM_RATIO_TOLERANCE,
            "passed": passed,
        }
    return comparison


def _repeatability(first: tuple[torch.Tensor, ...], repeated: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    result = {}
    for name, first_gradient, repeated_gradient in zip(GRADIENT_NAMES, first, repeated):
        result[name] = {
            "bitwise_equal": bool(torch.equal(first_gradient, repeated_gradient)),
        }
    return result


def _audit_launches(module, fn: Callable[[], tuple[torch.Tensor, ...]]):
    """Run once while classifying generic, projection, and dX launches."""

    original_run = module._run_compiled
    original_gemm = module.gemm_a16w16
    compile_names = (
        "_compile_grouped_w1_recompute",
        "_compile_grouped_w2_recompute",
        "_compile_grouped_dx",
    )
    original_compilers = {name: getattr(module, name) for name in compile_names}
    compiled_labels: dict[int, str] = {}
    counts = {
        "generic_gemm_launches": 0,
        "generic_projection_launches": 0,
        "grouped_w1_projection_launches": 0,
        "grouped_w2_projection_launches": 0,
        "legacy_dx_gemm_launches": 0,
        "grouped_dx_launches": 0,
    }
    generic_gemms = []

    def tracking_compiler(label: str, compiler):
        def wrapped(*args, **kwargs):
            compiled = compiler(*args, **kwargs)
            compiled_labels[id(compiled)] = label
            return compiled

        return wrapped

    def wrapped_run(compiled, *args, **kwargs):
        label = compiled_labels.get(id(compiled))
        if label is not None:
            counts[f"{label}_launches"] += 1
        return original_run(compiled, *args, **kwargs)

    def wrapped_gemm(a, b, *args, **kwargs):
        layout = kwargs.get("layout")
        output = kwargs.get("out")
        a_shape = tuple(a.shape)
        b_shape = tuple(b.shape)
        output_shape = tuple(output.shape) if output is not None else None
        counts["generic_gemm_launches"] += 1
        if layout == "nt":
            counts["generic_projection_launches"] += 1
        is_legacy_dx = (
            layout == "nn"
            and len(a_shape) == 2
            and len(b_shape) == 2
            and a_shape[1] == 2 * INTERMEDIATE
            and b_shape == (2 * INTERMEDIATE, HIDDEN)
            and output_shape is not None
            and output_shape[-1] == HIDDEN
        )
        if is_legacy_dx:
            counts["legacy_dx_gemm_launches"] += 1
        generic_gemms.append(
            {
                "layout": layout,
                "a_shape": list(a_shape),
                "b_shape": list(b_shape),
                "out_shape": None if output_shape is None else list(output_shape),
                "classified_as_legacy_dx": is_legacy_dx,
            }
        )
        return original_gemm(a, b, *args, **kwargs)

    module._run_compiled = wrapped_run
    module.gemm_a16w16 = wrapped_gemm
    module._compile_grouped_w1_recompute = tracking_compiler(
        "grouped_w1_projection", original_compilers["_compile_grouped_w1_recompute"]
    )
    module._compile_grouped_w2_recompute = tracking_compiler(
        "grouped_w2_projection", original_compilers["_compile_grouped_w2_recompute"]
    )
    module._compile_grouped_dx = tracking_compiler("grouped_dx", original_compilers["_compile_grouped_dx"])
    try:
        gradients = fn()
        torch.cuda.synchronize()
    finally:
        module._run_compiled = original_run
        module.gemm_a16w16 = original_gemm
        for name, compiler in original_compilers.items():
            setattr(module, name, compiler)

    counts["projection_launches"] = (
        counts["generic_projection_launches"]
        + counts["grouped_w1_projection_launches"]
        + counts["grouped_w2_projection_launches"]
    )
    counts["generic_dx_gemm_launches"] = counts["legacy_dx_gemm_launches"]
    counts["total_dx_launches"] = counts["legacy_dx_gemm_launches"] + counts["grouped_dx_launches"]
    counts["zero_launch_gates"] = {
        "generic_gemm_zero": counts["generic_gemm_launches"] == 0,
        "projection_zero": counts["projection_launches"] == 0,
        "generic_dx_zero": counts["generic_dx_gemm_launches"] == 0,
    }
    counts["generic_gemms"] = generic_gemms
    return gradients, counts


def _event_time(fn: Callable[[], tuple[torch.Tensor, ...]]):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    host_begin = time.perf_counter()
    begin.record()
    value = fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end), (time.perf_counter() - host_begin) * 1e3, value


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "samples_ms": values,
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _peak_delta(fn: Callable[[], tuple[torch.Tensor, ...]]) -> dict[str, int]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    value = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del value
    gc.collect()
    torch.cuda.synchronize()
    return {
        "resident_before_bytes": resident,
        "peak_allocated_bytes": peak,
        "peak_delta_bytes": peak - resident,
    }


def _summarize_peaks(samples: dict[str, dict[str, list[dict[str, int]]]]) -> dict[str, Any]:
    summary = {}
    for scope, modes in samples.items():
        baseline = statistics.median(entry["peak_delta_bytes"] for entry in modes["baseline"])
        candidate = statistics.median(entry["peak_delta_bytes"] for entry in modes["candidate"])
        summary[scope] = {
            "baseline_peak_delta_median_bytes": baseline,
            "candidate_peak_delta_median_bytes": candidate,
            "candidate_minus_baseline_bytes": candidate - baseline,
        }
    return summary


def _measure_abba(
    calls: dict[str, Callable[[str], tuple[torch.Tensor, ...]]],
    *,
    pairs: int,
) -> dict[str, Any]:
    measurements = {
        scope: {
            "baseline": [],
            "candidate": [],
            "baseline_host": [],
            "candidate_host": [],
        }
        for scope in calls
    }
    blocks = {scope: [] for scope in calls}
    orders = (
        ("ABBA", ("baseline", "candidate", "candidate", "baseline")),
        ("BAAB", ("candidate", "baseline", "baseline", "candidate")),
    )
    for pair in range(pairs):
        scope_items = tuple(calls.items())
        if pair % 2:
            scope_items = tuple(reversed(scope_items))
        order_name, order = orders[pair % len(orders)]
        for scope, fn in scope_items:
            samples = {"baseline": [], "candidate": []}
            for mode in order:
                device_ms, host_ms, value = _event_time(lambda fn=fn, mode=mode: fn(mode))
                measurements[scope][mode].append(device_ms)
                measurements[scope][f"{mode}_host"].append(host_ms)
                samples[mode].append(device_ms)
                del value
            baseline_mean = statistics.mean(samples["baseline"])
            candidate_mean = statistics.mean(samples["candidate"])
            blocks[scope].append(
                {
                    "index": pair,
                    "order": order_name,
                    "baseline_mean_ms": baseline_mean,
                    "candidate_mean_ms": candidate_mean,
                    "speedup": baseline_mean / candidate_mean,
                    "candidate_faster": candidate_mean < baseline_mean,
                }
            )

    result = {}
    for scope, values in measurements.items():
        baseline = _stats(values["baseline"])
        candidate = _stats(values["candidate"])
        paired_speedups = [block["speedup"] for block in blocks[scope]]
        result[scope] = {
            "baseline": baseline,
            "candidate": candidate,
            "speedup": baseline["median_ms"] / candidate["median_ms"],
            "reduction_pct": 100.0 * (baseline["median_ms"] - candidate["median_ms"]) / baseline["median_ms"],
            "host_baseline": _stats(values["baseline_host"]),
            "host_candidate": _stats(values["candidate_host"]),
            "abba_baab_blocks": blocks[scope],
            "paired_speedup_median": statistics.median(paired_speedups),
            "paired_win_rate": sum(block["candidate_faster"] for block in blocks[scope]) / len(blocks[scope]),
        }
    return result


def _emit_report(report: dict[str, Any], output: str | None) -> None:
    payload = json.dumps(report, indent=2)
    print(payload)
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="/home/sijieli2/FlyDSL/kernels/moe/sonic_backward.py",
        help="mainline sonic_backward.py loaded under a private module name",
    )
    parser.add_argument("--cases", nargs="+", choices=ROUTING_CASES, default=list(ROUTING_CASES))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=11, help="alternating ABBA/BAAB blocks per case and scope")
    parser.add_argument(
        "--correctness-repeats",
        type=int,
        default=2,
        help="candidate invocations retained for strict bitwise repeatability (minimum 2)",
    )
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="run correctness and launch gates, but skip timing and peak-memory sampling",
    )
    parser.add_argument(
        "--exclusive-gpu",
        action="store_true",
        help="assert that the selected gfx950 is independently verified idle and reserved",
    )
    parser.add_argument("--min-backward-speedup", type=float, default=1.0)
    parser.add_argument("--min-full-step-speedup", type=float, default=1.0)
    parser.add_argument("--min-paired-win-rate", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    if args.correctness_repeats < 2:
        parser.error("--correctness-repeats must be at least 2 for the bitwise gate")
    if args.min_backward_speedup <= 0 or args.min_full_step_speedup <= 0:
        parser.error("speedup thresholds must be positive")
    if not 0.0 <= args.min_paired_win_rate <= 1.0:
        parser.error("--min-paired-win-rate must be between 0 and 1")
    if not args.correctness_only and not args.exclusive_gpu:
        parser.error("timing requires --exclusive-gpu after independently reserving the selected GPU")
    return args


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")
    arch = str(get_rocm_arch())
    if not arch.startswith("gfx950"):
        raise RuntimeError(f"this acceptance gate requires gfx950, found {arch!r}")

    baseline_path = Path(args.baseline).resolve()
    if not baseline_path.is_file():
        raise FileNotFoundError(f"baseline sonic_backward.py does not exist: {baseline_path}")
    code_identity = _collect_code_identity(baseline_path)
    baseline_module = _load_isolated_baseline(baseline_path)
    candidate_source = Path(inspect.getsourcefile(candidate_module.sonic_moe_backward) or "").resolve()
    if candidate_source != Path(candidate_module.__file__).resolve():
        raise RuntimeError(
            "candidate sonic_moe_backward was rebound after import: "
            f"module={candidate_module.__file__}, function={candidate_source}"
        )

    policy_kwargs = {
        "tokens": TOKENS,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_experts": EXPERTS,
        "topk": TOPK,
        "flat_routes": False,
    }
    policy_probe = {
        "baseline_large_grouped_dx": baseline_module._use_large_grouped_dx_descriptor_queue(**policy_kwargs),
        "candidate_large_grouped_dx": candidate_module._use_large_grouped_dx_descriptor_queue(**policy_kwargs),
    }
    policy_probe["passed"] = (
        policy_probe["baseline_large_grouped_dx"] is False and policy_probe["candidate_large_grouped_dx"] is True
    )
    if not policy_probe["passed"]:
        raise RuntimeError(f"baseline/candidate policy split is not the intended E896 comparison: {policy_probe}")

    config = _adapter_config()
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    x = torch.empty((TOKENS, HIDDEN), device="cuda", dtype=torch.bfloat16).uniform_(-0.02, 0.02, generator=generator)
    separated_w1 = torch.empty(
        (EXPERTS, 2 * INTERMEDIATE, HIDDEN),
        device="cuda",
        dtype=torch.bfloat16,
    ).uniform_(-0.02, 0.02, generator=generator)
    interleaved_w1 = _interleave_glu_rows(separated_w1)
    w2 = torch.empty(
        (EXPERTS, HIDDEN, INTERMEDIATE),
        device="cuda",
        dtype=torch.bfloat16,
    ).uniform_(-0.02, 0.02, generator=generator)
    dout = torch.empty_like(x).uniform_(-0.02, 0.02, generator=generator)
    op = SonicMoE(config, prepare_sonic_bf16_weights(separated_w1, w2, config))
    modules = {"baseline": baseline_module, "candidate": candidate_module}

    report: dict[str, Any] = {
        "schema": "flydsl.sonic_e896_backward_acceptance.v1",
        "device": {
            "name": torch.cuda.get_device_name(),
            "arch": arch,
            "index": torch.cuda.current_device(),
            "torch": torch.__version__,
        },
        "code_identity": code_identity,
        "policy_probe": policy_probe,
        "contract": {
            "shape": {"T": TOKENS, "H": HIDDEN, "I": INTERMEDIATE, "E": EXPERTS, "K": TOPK},
            "config": _config_dict(config),
            "dtype": "torch.bfloat16",
            "activation": "swiglu",
            "interleaved_w1": True,
            "forward_state": "retained",
            "adapter_layout": {
                "forward_w1": "expert-major [gate | up]",
                "backward_w1": "expert-major [g0, u0, ...]",
            },
        },
        "numerical_limits": {
            name: {
                "relative_l2": RELATIVE_L2_LIMITS[name],
                "max_abs": MAX_ABS_LIMITS[name],
                "norm_ratio_tolerance": NORM_RATIO_TOLERANCE,
            }
            for name in GRADIENT_NAMES
        },
        "correctness_repeats": args.correctness_repeats,
        "cases": {},
    }

    overall_passed = True
    for case in dict.fromkeys(args.cases):
        ids, scores, routing_summary = _routing(case, generator)
        forward_output, fixed_state = op.forward_topk_training(
            x,
            ids,
            scores,
            interleaved_w1=True,
        )
        del forward_output
        torch.cuda.synchronize()

        def backward(mode: str):
            return modules[mode].sonic_moe_backward(
                x,
                interleaved_w1,
                w2,
                ids,
                scores,
                dout,
                config,
                interleaved_w1=True,
                forward_state=fixed_state,
            )

        def full_step(mode: str):
            output, state = op.forward_topk_training(
                x,
                ids,
                scores,
                interleaved_w1=True,
            )
            gradients = modules[mode].sonic_moe_backward(
                x,
                interleaved_w1,
                w2,
                ids,
                scores,
                dout,
                config,
                interleaved_w1=True,
                forward_state=state,
            )
            del output, state
            return gradients

        baseline_gradients, baseline_launches = _audit_launches(baseline_module, lambda: backward("baseline"))
        candidate_gradients, candidate_launches = _audit_launches(candidate_module, lambda: backward("candidate"))
        accuracy = _compare_gradients(candidate_gradients, baseline_gradients)
        repeatability = []
        for repeat in range(1, args.correctness_repeats):
            repeated = backward("candidate")
            torch.cuda.synchronize()
            repeatability.append(
                {
                    "repeat": repeat,
                    "candidate_vs_first": _repeatability(candidate_gradients, repeated),
                }
            )
            del repeated

        accuracy_passed = all(entry["passed"] for entry in accuracy.values())
        repeatability_passed = all(
            entry["bitwise_equal"] for repeat in repeatability for entry in repeat["candidate_vs_first"].values()
        )
        candidate_zero_launches = all(candidate_launches["zero_launch_gates"].values())
        launch_topology_passed = bool(
            candidate_zero_launches
            and candidate_launches["grouped_dx_launches"] == 1
            and candidate_launches["total_dx_launches"] == 1
            and baseline_launches["legacy_dx_gemm_launches"] > 0
            and baseline_launches["projection_launches"] > 0
        )
        case_report: dict[str, Any] = {
            "routing": routing_summary,
            "candidate_vs_baseline": accuracy,
            "candidate_repeatability": repeatability,
            "launches": {
                "baseline": baseline_launches,
                "candidate": candidate_launches,
                "gate": {
                    "candidate_generic_gemm_zero": candidate_launches["generic_gemm_launches"] == 0,
                    "candidate_projection_zero": candidate_launches["projection_launches"] == 0,
                    "candidate_legacy_dx_zero": candidate_launches["legacy_dx_gemm_launches"] == 0,
                    "candidate_single_grouped_dx": candidate_launches["grouped_dx_launches"] == 1,
                    "baseline_exercises_legacy_dx": baseline_launches["legacy_dx_gemm_launches"] > 0,
                    "baseline_exercises_projection": baseline_launches["projection_launches"] > 0,
                    "passed": launch_topology_passed,
                },
            },
            "gates": {
                "accuracy": accuracy_passed,
                "bitwise_repeatability": repeatability_passed,
                "launch_topology": launch_topology_passed,
            },
        }
        del baseline_gradients, candidate_gradients
        gc.collect()

        correctness_passed = accuracy_passed and repeatability_passed and launch_topology_passed
        if args.correctness_only:
            case_report["timing_status"] = "skipped (--correctness-only)"
            case_report["passed"] = correctness_passed
            report["cases"][case] = case_report
            overall_passed &= correctness_passed
            continue

        for _ in range(args.warmup):
            for scope in (backward, full_step):
                for mode in ("baseline", "candidate"):
                    value = scope(mode)
                    torch.cuda.synchronize()
                    del value

        peaks = {scope: {mode: [] for mode in ("baseline", "candidate")} for scope in ("backward", "full_step")}
        peak_calls = {"backward": backward, "full_step": full_step}
        for order in (("baseline", "candidate"), ("candidate", "baseline")):
            for scope, fn in peak_calls.items():
                for mode in order:
                    peaks[scope][mode].append(_peak_delta(lambda fn=fn, mode=mode: fn(mode)))

        measurements = _measure_abba(
            {"backward": backward, "full_step": full_step},
            pairs=args.pairs,
        )
        speedup_limits = {
            "backward": args.min_backward_speedup,
            "full_step": args.min_full_step_speedup,
        }
        performance_gates = {}
        for scope, minimum_speedup in speedup_limits.items():
            measured = measurements[scope]
            performance_gates[scope] = {
                "minimum_speedup": minimum_speedup,
                "minimum_paired_win_rate": args.min_paired_win_rate,
                "median_speedup_passed": measured["speedup"] >= minimum_speedup,
                "paired_speedup_passed": measured["paired_speedup_median"] >= minimum_speedup,
                "paired_win_rate_passed": measured["paired_win_rate"] >= args.min_paired_win_rate,
            }
            performance_gates[scope]["passed"] = all(
                value for key, value in performance_gates[scope].items() if key.endswith("_passed")
            )
        performance_passed = all(entry["passed"] for entry in performance_gates.values())
        case_report.update(
            {
                "exclusive_gpu_asserted": args.exclusive_gpu,
                "warmup": args.warmup,
                "abba_baab_blocks": args.pairs,
                "peak_memory": {
                    "samples": peaks,
                    "summary": _summarize_peaks(peaks),
                },
                "measurements": measurements,
                "performance_gates": performance_gates,
                "passed": correctness_passed and performance_passed,
            }
        )
        report["cases"][case] = case_report
        overall_passed &= case_report["passed"]
        gc.collect()

    report["timing_status"] = "skipped" if args.correctness_only else "measured"
    report["exclusive_gpu_asserted"] = args.exclusive_gpu
    report["passed"] = bool(overall_passed and policy_probe["passed"])
    _emit_report(report, args.output)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
