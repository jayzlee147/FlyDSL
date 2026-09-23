# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Reproduce Lumen's Qwen3-30B-A3B SonicMoE adapter timing boundary.

The production path documented by Lumen is::

    SonicMoEExperts.forward
      -> flydsl_pre_routed
      -> _FlyDSLNativeRoutes.forward
      -> SonicMoE.forward_routes_training

Lumen creates ``cu_seqlens``, identity token ids, and expert ids inside that
forward range.  ``benchmark_sonic_e16_dynamic.py`` intentionally constructs
those tensors before timing, so its forward number is an operator-only number
and cannot be compared directly with Lumen's ``SonicMoE.experts.fwd`` range.

This tool reports both legacy and implicit-ID boundaries:

* legacy route metadata: cumsum, identity arange, and repeat_interleave;
* legacy and implicit-ID operator forward with prebuilt offsets/output;
* legacy simulated adapter work, including cumsum/arange/repeat_interleave;
* implicit-ID simulated adapter work, which constructs only expert offsets;
* counts-native work, which passes device counts and constructs no metadata;
* paired legacy/implicit forward, retained backward, and end-to-end timings;
* paired offsets-only/counts-native operator, adapter, and end-to-end timings;
* optionally, Lumen's actual legacy autograd adapter timing;
* optionally, AITER's complete ``moe_pre_routed_inputs`` baseline.

The default counts use the same seed-0 ``softmax(randn)`` generator as Lumen's
single-GPU benchmark.  They are synthetic Lumen-shaped counts, not a capture
of an EP=8 training run.  Supply ``--counts`` to replay one exact rank-local
distribution.  This is still a single-stream, single-GPU microbenchmark: it
does not reproduce EP all-to-all, NCCL overlap, 48-layer state pressure, or
cross-rank arrival skew.

Examples::

    PYTHONPATH=. python tools/benchmark_sonic_e16_lumen_repro.py
    PYTHONPATH=. python tools/benchmark_sonic_e16_lumen_repro.py --routes 8192
    PYTHONPATH=. python tools/benchmark_sonic_e16_lumen_repro.py \
      --counts 48,61,75,91,117,1404,83,512,720,604,198,687,901,1130,756,804
    PYTHONPATH=.:/path/to/aiter python tools/benchmark_sonic_e16_lumen_repro.py \
      --lumen-root /path/to/Lumen --aiter-root /path/to/aiter \
      --routes 8191,8192,9000

The Lumen adapter behavior mirrored here was audited at
``WuLei-AMD/Lumen@c8338ddf10e9``.  ``--aiter-root`` should point at the AITER
checkout being compared; the original model report used upstream
``ccd9200b022f``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable

import torch

from kernels.moe.sonic import (
    SonicMoE,
    SonicMoEConfig,
    prepare_sonic_bf16_weights,
)
from kernels.moe.sonic_backward import (
    sonic_moe_backward_expert_major,
    sonic_moe_backward_routes,
)


E = 16
H = 2048
I = 768
DEFAULT_ROUTES = (8191, 8192, 9000)
LUMEN_AUDIT_COMMIT = "c8338ddf10e9cf70cb4c35db8e4751b5d848a750"
_LUMEN_EXPERT_BWD_RANGES: list[Any] = []


class _LumenExpertsBwdClose(torch.autograd.Function):
    """Mirror Lumen's inner backward profiling marker."""

    @staticmethod
    def forward(ctx, hidden: torch.Tensor) -> torch.Tensor:
        return hidden

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        if _LUMEN_EXPERT_BWD_RANGES:
            _LUMEN_EXPERT_BWD_RANGES.pop().__exit__(None, None, None)
        return grad


class _LumenExpertsBwdOpen(torch.autograd.Function):
    """Mirror Lumen's outer backward profiling marker."""

    @staticmethod
    def forward(ctx, hidden: torch.Tensor) -> torch.Tensor:
        return hidden

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        record = torch.autograd.profiler.record_function("SonicMoE.experts.bwd")
        record.__enter__()
        _LUMEN_EXPERT_BWD_RANGES.append(record)
        return grad


def _lumen_profiled_experts(hidden: torch.Tensor, run: Callable[[torch.Tensor], torch.Tensor]):
    """Mirror Lumen's full expert timing wrapper around either backend."""

    hidden = _LumenExpertsBwdClose.apply(hidden)
    with torch.profiler.record_function("SonicMoE.experts.fwd"):
        output = run(hidden)
    return _LumenExpertsBwdOpen.apply(output)


@dataclass
class Case:
    routes: int
    counts_host: list[int]
    counts_cpu: torch.Tensor
    counts: torch.Tensor
    hidden: torch.Tensor
    scores: torch.Tensor
    grad_output: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    operator: SonicMoE
    config: SonicMoEConfig


def _production_config() -> SonicMoEConfig:
    """Return the exact E16 profile selected by Lumen's FlyDSL adapter."""

    return SonicMoEConfig(
        hidden_size=H,
        intermediate_size=I,
        num_experts=E,
        top_k=1,
        tile_m=128,
        tile_n=128,
        tile_k=64,
        down_tile_m=64,
        down_tile_n=256,
        down_tile_k=64,
        renormalize=False,
        stage1_b_cache_mod=0,
        stage2_b_cache_mod=0,
        stage1_xcd_swizzle=8,
        stage1_k_wave=1,
        stage2_xcd_swizzle=0,
        stage2_pipeline_stages=2,
        stage1_write_padded_rows=True,
        stage1_lds_swizzle=True,
        activation="swiglu",
        compute_dtype="bf16",
    )


def _lumen_synthetic_counts(routes: int, seed: int) -> list[int]:
    """Mirror Lumen's benchmark-only seed/softmax/multinomial generator."""

    if routes == 0:
        return [0] * E
    generator = torch.Generator(device="cpu").manual_seed(seed)
    probabilities = torch.softmax(torch.randn(E, generator=generator), dim=0)
    assignments = torch.multinomial(
        probabilities,
        routes,
        replacement=True,
        generator=generator,
    )
    return torch.bincount(assignments, minlength=E).tolist()


def _parse_int_list(value: str, *, name: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    return result


def _git_head(root: str | None) -> str | None:
    """Return a checkout's actual commit without making provenance mandatory."""

    if root is None:
        return None
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={os.path.abspath(root)}",
                "-C",
                root,
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _git_dirty(root: str | None) -> bool | None:
    if root is None:
        return None
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={os.path.abspath(root)}",
                "-C",
                root,
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(completed.stdout.strip())


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_counts(counts: list[int]) -> None:
    if len(counts) != E:
        raise ValueError(f"expected {E} expert counts, got {len(counts)}")
    if any(count < 0 for count in counts):
        raise ValueError("expert counts must be non-negative")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _summarize(samples: list[float]) -> dict[str, Any]:
    return {
        "median_ms": float(statistics.median(samples)),
        "min_ms": min(samples),
        "p10_ms": _percentile(samples, 0.10),
        "p90_ms": _percentile(samples, 0.90),
    }


def _measure_gpu(
    fn: Callable[[], object],
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    """Match Lumen's one-call-at-a-time CUDA-event measurement."""

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def one() -> float:
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))

    for _ in range(warmup):
        one()
    return _summarize([one() for _ in range(iters)])


def _measure_gpu_with_setup(
    setup: Callable[[], object],
    fn: Callable[[object], object],
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    """Run and synchronize setup outside a backward-only event range."""

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def one() -> float:
        payload = setup()
        torch.cuda.synchronize()
        start.record()
        fn(payload)
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))

    for _ in range(warmup):
        one()
    return _summarize([one() for _ in range(iters)])


def _paired_summary(
    reference_samples: list[float],
    candidate_samples: list[float],
) -> dict[str, Any]:
    """Summarize sample-aligned A/B timings and their per-pair deltas."""

    deltas = [
        candidate - reference
        for reference, candidate in zip(reference_samples, candidate_samples)
    ]
    speedups = [
        reference / candidate
        for reference, candidate in zip(reference_samples, candidate_samples)
        if candidate > 0.0
    ]
    return {
        "protocol": "alternating_ab_ba_synchronized_calls",
        "reference": _summarize(reference_samples),
        "candidate": _summarize(candidate_samples),
        "candidate_minus_reference_ms": _summarize(deltas),
        "candidate_speedup_x": _summarize(speedups) if speedups else None,
    }


def _measure_gpu_paired(
    reference: Callable[[], object],
    candidate: Callable[[], object],
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    """Measure one synchronized A/B pair per sample, alternating call order."""

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def one(fn: Callable[[], object]) -> float:
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))

    for index in range(warmup):
        ordered = (reference, candidate) if index % 2 == 0 else (candidate, reference)
        for fn in ordered:
            one(fn)

    reference_samples: list[float] = []
    candidate_samples: list[float] = []
    for index in range(iters):
        if index % 2 == 0:
            reference_samples.append(one(reference))
            candidate_samples.append(one(candidate))
        else:
            candidate_samples.append(one(candidate))
            reference_samples.append(one(reference))
    return _paired_summary(reference_samples, candidate_samples)


def _measure_gpu_paired_with_setup(
    reference_setup: Callable[[], object],
    reference: Callable[[object], object],
    candidate_setup: Callable[[], object],
    candidate: Callable[[object], object],
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    """Measure paired backward calls while keeping each forward setup untimed."""

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def one(setup: Callable[[], object], fn: Callable[[object], object]) -> float:
        payload = setup()
        torch.cuda.synchronize()
        start.record()
        fn(payload)
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))

    reference_call = lambda: one(reference_setup, reference)
    candidate_call = lambda: one(candidate_setup, candidate)
    for index in range(warmup):
        ordered = (
            (reference_call, candidate_call)
            if index % 2 == 0
            else (candidate_call, reference_call)
        )
        for fn in ordered:
            fn()

    reference_samples: list[float] = []
    candidate_samples: list[float] = []
    for index in range(iters):
        if index % 2 == 0:
            reference_samples.append(reference_call())
            candidate_samples.append(candidate_call())
        else:
            candidate_samples.append(candidate_call())
            reference_samples.append(reference_call())
    return _paired_summary(reference_samples, candidate_samples)


def _build_case(routes: int, counts_host: list[int], seed: int) -> Case:
    device = torch.device("cuda")
    config = _production_config()
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    counts_cpu = torch.tensor(counts_host, dtype=torch.int32)
    counts = counts_cpu.to(device=device)
    hidden = torch.randn(
        routes,
        H,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    scores = torch.rand(
        routes,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    grad_output = torch.randn(
        routes,
        H,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w1 = torch.randn(
        E,
        2 * I,
        H,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(H**-0.5)
    w2 = torch.randn(
        E,
        H,
        I,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(I**-0.5)
    operator = SonicMoE(
        config,
        prepare_sonic_bf16_weights(w1, w2, config),
        # Lumen uses one reusable workspace per layer to avoid 48-layer
        # first-forward workspace retention.
        max_cached_workspaces=1,
    )
    return Case(
        routes=routes,
        counts_host=counts_host,
        counts_cpu=counts_cpu,
        counts=counts,
        hidden=hidden,
        scores=scores,
        grad_output=grad_output,
        w1=w1,
        w2=w2,
        operator=operator,
        config=config,
    )


def _cu_seqlens(counts: torch.Tensor) -> torch.Tensor:
    cu = torch.empty(E + 1, dtype=torch.int32, device=counts.device)
    cu[0] = 0
    cu[1:] = counts.cumsum(0, dtype=torch.int32)
    return cu


def _token_indices(routes: int, device: torch.device) -> torch.Tensor:
    return torch.arange(routes, dtype=torch.int32, device=device)


def _expert_indices(
    counts: torch.Tensor,
    routes: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.repeat_interleave(
        torch.arange(E, dtype=torch.int32, device=device),
        counts,
        output_size=routes,
    )


def _route_metadata(case: Case) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = case.hidden.device
    return (
        _cu_seqlens(case.counts),
        _token_indices(case.routes, device),
        _expert_indices(case.counts, case.routes, device),
    )


def _operator_forward(
    case: Case,
    cu_seqlens: torch.Tensor,
    token_indices: torch.Tensor,
    expert_indices: torch.Tensor,
    output: torch.Tensor,
):
    return case.operator.forward_routes_training(
        case.hidden,
        token_indices,
        expert_indices,
        case.scores,
        out=output,
        expert_offsets=cu_seqlens,
        token_indices_identity=True,
    )


def _implicit_operator_forward(
    case: Case,
    cu_seqlens: torch.Tensor,
    output: torch.Tensor,
):
    """Call the expert-major training ABI without flat token/expert IDs."""

    return case.operator.forward_expert_major_training(
        case.hidden,
        cu_seqlens,
        case.scores,
        out=output,
    )


def _counts_native_operator_forward(
    case: Case,
    output: torch.Tensor,
):
    """Call the expert-major training ABI directly from device expert counts."""

    return case.operator.forward_expert_major_counts_training(
        case.hidden,
        case.counts,
        case.scores,
        out=output,
    )


def _simulated_adapter_forward(case: Case):
    """Mirror the GPU-visible work in Lumen's _FlyDSLNativeRoutes.forward."""

    hidden = case.hidden.contiguous()
    scores = case.scores.reshape(-1).float().contiguous()
    cu_seqlens, token_indices, expert_indices = _route_metadata(case)
    output = torch.empty_like(hidden)
    output, state = case.operator.forward_routes_training(
        hidden,
        token_indices,
        expert_indices,
        scores,
        out=output,
        expert_offsets=cu_seqlens,
        token_indices_identity=True,
    )
    return output, state, token_indices, expert_indices


def _implicit_adapter_forward(case: Case):
    """Model an updated adapter that materializes offsets, but no flat IDs."""

    hidden = case.hidden.contiguous()
    scores = case.scores.reshape(-1).float().contiguous()
    cu_seqlens = _cu_seqlens(case.counts)
    output = torch.empty_like(hidden)
    output, state = case.operator.forward_expert_major_training(
        hidden,
        cu_seqlens,
        scores,
        out=output,
    )
    return output, state


def _counts_native_adapter_forward(case: Case):
    """Model an adapter that forwards device counts without any metadata op."""

    hidden = case.hidden.contiguous()
    scores = case.scores.reshape(-1).float().contiguous()
    output = torch.empty_like(hidden)
    output, state = case.operator.forward_expert_major_counts_training(
        hidden,
        case.counts,
        scores,
        out=output,
    )
    return output, state


def _flydsl_backward(case: Case, payload: object):
    _, state, token_indices, expert_indices = payload
    return sonic_moe_backward_routes(
        case.hidden,
        case.w1,
        case.w2,
        token_indices,
        expert_indices,
        case.scores,
        case.grad_output,
        case.config,
        # This matches Lumen's retained-state gate. Small diagnostic cases
        # still run, but T<64 intentionally exercises standalone backward.
        forward_state=state if case.routes >= 64 else None,
        token_indices_sorted=True,
    )


def _flydsl_backward_implicit(case: Case, payload: object):
    _, state = payload
    return sonic_moe_backward_expert_major(
        case.hidden,
        case.w1,
        case.w2,
        case.scores,
        case.grad_output,
        case.config,
        forward_state=state,
    )


def _correctness(
    case: Case,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    torch.Tensor,
    tuple[torch.Tensor, ...],
]:
    cu_seqlens, token_indices, expert_indices = _route_metadata(case)
    direct_output = torch.empty_like(case.hidden)
    direct_output, direct_state = _operator_forward(
        case,
        cu_seqlens,
        token_indices,
        expert_indices,
        direct_output,
    )
    adapter_payload = _simulated_adapter_forward(case)
    adapter_output, adapter_state, _, _ = adapter_payload
    adapter_gradients = _flydsl_backward(case, adapter_payload)
    implicit_output = torch.empty_like(case.hidden)
    implicit_output, implicit_state = _implicit_operator_forward(
        case,
        cu_seqlens,
        implicit_output,
    )
    implicit_gradients = _flydsl_backward_implicit(
        case,
        (implicit_output, implicit_state),
    )
    counts_native_output = torch.empty_like(case.hidden)
    counts_native_output, counts_native_state = _counts_native_operator_forward(
        case,
        counts_native_output,
    )
    counts_native_gradients = _flydsl_backward_implicit(
        case,
        (counts_native_output, counts_native_state),
    )
    torch.cuda.synchronize()
    frequency_matches = bool(
        torch.equal(adapter_state.expert_frequency, case.counts)
        and torch.equal(direct_state.expert_frequency, case.counts)
        and torch.equal(implicit_state.expert_frequency, case.counts)
        and torch.equal(counts_native_state.expert_frequency, case.counts)
    )
    implicit_difference = (adapter_output.float() - implicit_output.float()).abs()
    counts_native_difference = (
        implicit_output.float() - counts_native_output.float()
    ).abs()
    return (
        {
            "operator_vs_simulated_adapter_bitwise": bool(
                torch.equal(direct_output, adapter_output)
            ),
            "legacy_vs_implicit_operator_bitwise": bool(
                torch.equal(adapter_output, implicit_output)
            ),
            "legacy_vs_implicit_operator_max_abs": (
                float(implicit_difference.max())
                if implicit_difference.numel()
                else 0.0
            ),
            "implicit_offsets_vs_counts_native_operator_bitwise": bool(
                torch.equal(implicit_output, counts_native_output)
            ),
            "implicit_offsets_vs_counts_native_operator_max_abs": (
                float(counts_native_difference.max())
                if counts_native_difference.numel()
                else 0.0
            ),
            "retained_expert_frequency_matches_counts": frequency_matches,
        },
        {
            "legacy_vs_implicit_retained": _compare_gradients(
                adapter_gradients,
                implicit_gradients,
            ),
            "implicit_offsets_vs_counts_native_retained": _compare_gradients(
                implicit_gradients,
                counts_native_gradients,
            ),
        },
        adapter_output,
        adapter_gradients,
    )


def _compare_gradients(
    reference: tuple[torch.Tensor, ...],
    candidate: tuple[torch.Tensor, ...],
    *,
    candidate_aiter_layout: bool = False,
) -> dict[str, Any]:
    names = ("dx", "dw1", "dw2", "dscores")
    if len(reference) != len(names) or len(candidate) != len(names):
        raise RuntimeError("bias-free SonicMoE backward must return four gradients")
    values: dict[str, Any] = {}
    for index, (name, expected, actual) in enumerate(
        zip(names, reference, candidate)
    ):
        if candidate_aiter_layout and index in (1, 2):
            actual = actual.transpose(1, 2)
        difference = (expected.float() - actual.float()).abs()
        denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        values[name] = {
            "bitwise": bool(torch.equal(expected, actual)),
            "max_abs": float(difference.max()) if difference.numel() else 0.0,
            "relative_l2": float(torch.linalg.vector_norm(difference) / denominator),
        }
    return values


def _flydsl_timings(case: Case, warmup: int, iters: int) -> dict[str, Any]:
    cu_seqlens, token_indices, expert_indices = _route_metadata(case)
    legacy_output = torch.empty_like(case.hidden)
    implicit_output = torch.empty_like(case.hidden)
    counts_native_output = torch.empty_like(case.hidden)

    # Compile every forward/backward family outside timed samples.
    _operator_forward(
        case,
        cu_seqlens,
        token_indices,
        expert_indices,
        legacy_output,
    )
    _implicit_operator_forward(case, cu_seqlens, implicit_output)
    _counts_native_operator_forward(case, counts_native_output)
    legacy_payload = _simulated_adapter_forward(case)
    _flydsl_backward(case, legacy_payload)
    implicit_payload = _implicit_adapter_forward(case)
    _flydsl_backward_implicit(case, implicit_payload)
    counts_native_payload = _counts_native_adapter_forward(case)
    _flydsl_backward_implicit(case, counts_native_payload)
    torch.cuda.synchronize()

    metadata_pair = _measure_gpu_paired(
        lambda: _route_metadata(case),
        lambda: _cu_seqlens(case.counts),
        warmup=warmup,
        iters=iters,
    )
    metadata_pair.update(
        {
            "reference_name": "legacy_cumsum_arange_repeat_interleave",
            "candidate_name": "implicit_expert_offsets_only",
        }
    )
    operator_pair = _measure_gpu_paired(
        lambda: _operator_forward(
            case,
            cu_seqlens,
            token_indices,
            expert_indices,
            legacy_output,
        ),
        lambda: _implicit_operator_forward(
            case,
            cu_seqlens,
            implicit_output,
        ),
        warmup=warmup,
        iters=iters,
    )
    operator_pair.update(
        {
            "reference_name": "legacy_flat_route_operator",
            "candidate_name": "implicit_expert_major_operator",
        }
    )
    adapter_forward_pair = _measure_gpu_paired(
        lambda: _simulated_adapter_forward(case),
        lambda: _implicit_adapter_forward(case),
        warmup=warmup,
        iters=iters,
    )
    adapter_forward_pair.update(
        {
            "reference_name": "legacy_simulated_lumen_adapter",
            "candidate_name": "implicit_simulated_adapter",
        }
    )
    backward_pair = _measure_gpu_paired_with_setup(
        lambda: _simulated_adapter_forward(case),
        lambda state_payload: _flydsl_backward(case, state_payload),
        lambda: _implicit_adapter_forward(case),
        lambda state_payload: _flydsl_backward_implicit(case, state_payload),
        warmup=warmup,
        iters=iters,
    )
    backward_pair.update(
        {
            "reference_name": "legacy_retained_backward",
            "candidate_name": "implicit_retained_backward",
        }
    )
    e2e_pair = _measure_gpu_paired(
        lambda: _flydsl_backward(case, _simulated_adapter_forward(case)),
        lambda: _flydsl_backward_implicit(case, _implicit_adapter_forward(case)),
        warmup=warmup,
        iters=iters,
    )
    e2e_pair.update(
        {
            "reference_name": "legacy_simulated_adapter_e2e",
            "candidate_name": "implicit_simulated_adapter_e2e",
        }
    )
    counts_operator_pair = _measure_gpu_paired(
        lambda: _implicit_operator_forward(
            case,
            cu_seqlens,
            implicit_output,
        ),
        lambda: _counts_native_operator_forward(case, counts_native_output),
        warmup=warmup,
        iters=iters,
    )
    counts_operator_pair.update(
        {
            "reference_name": "implicit_expert_offsets_operator",
            "candidate_name": "counts_native_operator",
        }
    )
    counts_adapter_forward_pair = _measure_gpu_paired(
        lambda: _implicit_adapter_forward(case),
        lambda: _counts_native_adapter_forward(case),
        warmup=warmup,
        iters=iters,
    )
    counts_adapter_forward_pair.update(
        {
            "reference_name": "implicit_offsets_simulated_adapter",
            "candidate_name": "counts_native_simulated_adapter",
        }
    )
    counts_e2e_pair = _measure_gpu_paired(
        lambda: _flydsl_backward_implicit(case, _implicit_adapter_forward(case)),
        lambda: _flydsl_backward_implicit(
            case,
            _counts_native_adapter_forward(case),
        ),
        warmup=warmup,
        iters=iters,
    )
    counts_e2e_pair.update(
        {
            "reference_name": "implicit_offsets_simulated_adapter_e2e",
            "candidate_name": "counts_native_simulated_adapter_e2e",
        }
    )

    timings: dict[str, Any] = {
        "route_cumsum": _measure_gpu(
            lambda: _cu_seqlens(case.counts),
            warmup=warmup,
            iters=iters,
        ),
        "route_token_arange": _measure_gpu(
            lambda: _token_indices(case.routes, case.hidden.device),
            warmup=warmup,
            iters=iters,
        ),
        "route_expert_repeat_interleave": _measure_gpu(
            lambda: _expert_indices(case.counts, case.routes, case.hidden.device),
            warmup=warmup,
            iters=iters,
        ),
        "route_metadata_total": metadata_pair["reference"],
        "implicit_metadata_total": metadata_pair["candidate"],
        "operator_forward": operator_pair["reference"],
        "implicit_operator_forward": operator_pair["candidate"],
        "simulated_adapter_forward": adapter_forward_pair["reference"],
        "implicit_simulated_adapter_forward": adapter_forward_pair["candidate"],
        "retained_backward": backward_pair["reference"],
        "implicit_retained_backward": backward_pair["candidate"],
        "simulated_adapter_e2e": e2e_pair["reference"],
        "implicit_simulated_adapter_e2e": e2e_pair["candidate"],
        "counts_native_operator_forward": counts_operator_pair["candidate"],
        "counts_native_simulated_adapter_forward": counts_adapter_forward_pair[
            "candidate"
        ],
        "counts_native_simulated_adapter_e2e": counts_e2e_pair["candidate"],
        "paired_ab": {
            "metadata": metadata_pair,
            "operator_forward": operator_pair,
            "simulated_adapter_forward": adapter_forward_pair,
            "retained_backward": backward_pair,
            "simulated_adapter_e2e": e2e_pair,
            "offsets_vs_counts_operator_forward": counts_operator_pair,
            "offsets_vs_counts_simulated_adapter_forward": (
                counts_adapter_forward_pair
            ),
            "offsets_vs_counts_simulated_adapter_e2e": counts_e2e_pair,
        },
    }
    timings["legacy_adapter_overhead_unpaired_median_ms"] = (
        timings["simulated_adapter_forward"]["median_ms"]
        - timings["operator_forward"]["median_ms"]
    )
    timings["implicit_adapter_overhead_unpaired_median_ms"] = (
        timings["implicit_simulated_adapter_forward"]["median_ms"]
        - timings["implicit_operator_forward"]["median_ms"]
    )
    timings["counts_native_adapter_overhead_unpaired_median_ms"] = (
        timings["counts_native_simulated_adapter_forward"]["median_ms"]
        - timings["counts_native_operator_forward"]["median_ms"]
    )
    return timings


def _load_aiter(aiter_root: str | None):
    root = None
    if aiter_root:
        root = os.path.realpath(os.path.expanduser(aiter_root))
        if not os.path.isdir(os.path.join(root, "aiter")):
            raise FileNotFoundError(f"AITER package directory not found: {root}/aiter")
        if root not in sys.path:
            sys.path.insert(0, root)
    os.environ["SONIC_MOE_GEMM_BACKEND"] = "triton"
    os.environ["SONIC_MOE_GROUPED_GEMM_BACKEND"] = "triton"
    os.environ["SONIC_MOE_USE_QWEN3_TUNED_GEMM"] = "1"
    from aiter.ops.triton.sonicmoe import (  # noqa: PLC0415
        SonicMoEActivationType,
        moe_pre_routed_inputs,
    )

    source_module = sys.modules.get(moe_pre_routed_inputs.__module__)
    source = getattr(source_module, "__file__", None)
    if root is not None and source is not None:
        resolved_source = os.path.realpath(source)
        if os.path.commonpath((root, resolved_source)) != root:
            raise ImportError(
                f"requested AITER root {root}, but imported {resolved_source}"
            )
    return (moe_pre_routed_inputs, SonicMoEActivationType.SWIGLU), source


def _load_lumen_adapter(lumen_root: str):
    """Load only Lumen's adapter module, without importing its Megatron stack."""

    module_path = os.path.join(
        os.path.abspath(os.path.expanduser(lumen_root)),
        "lumen",
        "ops",
        "moe",
        "flydsl_grouped.py",
    )
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"Lumen FlyDSL adapter not found: {module_path}")
    module_name = "_flydsl_lumen_repro_adapter"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Lumen FlyDSL adapter: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.flydsl_pre_routed, module_path


def _lumen_adapter_timings(
    case: Case,
    flydsl_pre_routed,
    warmup: int,
    iters: int,
) -> tuple[dict[str, Any], torch.Tensor, tuple[torch.Tensor, ...]]:
    """Measure Lumen's actual autograd adapter rather than its device-work model."""

    os.environ["SONIC_MOE_GEMM_BACKEND"] = "flydsl"
    os.environ["SONIC_MOE_FLYDSL_NATIVE"] = "1"
    hidden = case.hidden.detach().requires_grad_(True)
    scores = case.scores.detach().requires_grad_(True)
    w1 = case.w1.detach().requires_grad_(True)
    w2 = case.w2.detach().requires_grad_(True)

    def native_forward(value=hidden):
        return flydsl_pre_routed(
            value,
            case.counts,
            scores,
            w1,
            w2,
            native_weight_layout=True,
        )

    def profiled_forward():
        def run(value):
            return native_forward(value)

        return _lumen_profiled_experts(hidden, run)

    def inference_native_forward():
        with torch.no_grad():
            return native_forward()

    def inference_profiled_forward():
        with torch.no_grad():
            return profiled_forward()

    def backward(output: object):
        return torch.autograd.grad(
            output,
            (hidden, w1, w2, scores),
            case.grad_output,
        )

    compile_output = profiled_forward()
    reference_gradients = backward(compile_output)
    torch.cuda.synchronize()
    with torch.no_grad():
        reference_output = profiled_forward()
    torch.cuda.synchronize()

    actual_legacy_native_forward = _measure_gpu(
        inference_native_forward,
        warmup=warmup,
        iters=iters,
    )
    actual_legacy_profiled_forward = _measure_gpu(
        inference_profiled_forward,
        warmup=warmup,
        iters=iters,
    )
    timings = {
        "implementation": "actual_lumen_legacy_flat_route_autograd_adapter",
        "actual_legacy_native_adapter_forward": actual_legacy_native_forward,
        "actual_legacy_profiled_adapter_forward": actual_legacy_profiled_forward,
        # Compatibility aliases retained for consumers of the v1 JSON schema.
        "native_adapter_forward": actual_legacy_native_forward,
        "adapter_forward": actual_legacy_profiled_forward,
        "backward": _measure_gpu_with_setup(
            profiled_forward,
            backward,
            warmup=warmup,
            iters=iters,
        ),
        "e2e": _measure_gpu(
            lambda: backward(profiled_forward()),
            warmup=warmup,
            iters=iters,
        ),
    }
    return timings, reference_output, reference_gradients


def _aiter_timings(
    case: Case,
    aiter_api,
    activation,
    warmup: int,
    iters: int,
    counts_device: str,
) -> tuple[dict[str, Any], torch.Tensor, tuple[torch.Tensor, ...]]:
    os.environ["SONIC_MOE_GEMM_BACKEND"] = "triton"
    os.environ["SONIC_MOE_GROUPED_GEMM_BACKEND"] = "triton"
    os.environ["SONIC_MOE_USE_QWEN3_TUNED_GEMM"] = "1"
    # AITER's grouped-weight ABI is [E, K, N].
    w1 = case.w1.transpose(1, 2).contiguous().detach().requires_grad_(True)
    w2 = case.w2.transpose(1, 2).contiguous().detach().requires_grad_(True)
    hidden = case.hidden.detach().requires_grad_(True)
    scores = case.scores.detach().requires_grad_(True)
    expert_counts = case.counts_cpu if counts_device == "cpu" else case.counts

    def native_forward(value=hidden):
        output, _ = aiter_api(
            value,
            scores,
            expert_counts,
            w1,
            None,
            w2,
            None,
            torch.cuda.current_stream().cuda_stream,
            activation,
            False,
            True,
        )
        return output

    def profiled_forward():
        def run(value):
            return native_forward(value)

        return _lumen_profiled_experts(hidden, run)

    def inference_native_forward():
        with torch.no_grad():
            return native_forward()

    def inference_profiled_forward():
        with torch.no_grad():
            return profiled_forward()

    def backward(output: object):
        return torch.autograd.grad(
            output,
            (hidden, w1, w2, scores),
            case.grad_output,
        )

    # Compile/autotune before measurement.
    compile_output = profiled_forward()
    reference_gradients = backward(compile_output)
    torch.cuda.synchronize()
    with torch.no_grad():
        reference_output = profiled_forward()
    torch.cuda.synchronize()

    timings = {
        "native_pre_routed_forward": _measure_gpu(
            inference_native_forward,
            warmup=warmup,
            iters=iters,
        ),
        "adapter_forward": _measure_gpu(
            inference_profiled_forward,
            warmup=warmup,
            iters=iters,
        ),
        "backward": _measure_gpu_with_setup(
            profiled_forward,
            backward,
            warmup=warmup,
            iters=iters,
        ),
        "e2e": _measure_gpu(
            lambda: backward(profiled_forward()),
            warmup=warmup,
            iters=iters,
        ),
    }
    return timings, reference_output, reference_gradients


def _comparison(
    flydsl_output: torch.Tensor,
    aiter_output: torch.Tensor,
    flydsl_timings: dict[str, Any],
    aiter_timings: dict[str, Any],
    lumen_timings: dict[str, Any] | None,
) -> dict[str, Any]:
    difference = (flydsl_output.float() - aiter_output.float()).abs()
    denominator = torch.linalg.vector_norm(aiter_output.float())
    relative_l2 = torch.linalg.vector_norm(difference) / denominator.clamp_min(1e-12)
    result = {
        "forward_max_abs": float(difference.max()) if difference.numel() else 0.0,
        "forward_relative_l2": float(relative_l2),
        "flydsl_operator_over_aiter_native_pre_routed_forward": (
            flydsl_timings["operator_forward"]["median_ms"]
            / aiter_timings["native_pre_routed_forward"]["median_ms"]
        ),
        "flydsl_simulated_adapter_over_aiter_native_pre_routed_forward": (
            flydsl_timings["simulated_adapter_forward"]["median_ms"]
            / aiter_timings["native_pre_routed_forward"]["median_ms"]
        ),
        "flydsl_implicit_operator_over_aiter_native_pre_routed_forward": (
            flydsl_timings["implicit_operator_forward"]["median_ms"]
            / aiter_timings["native_pre_routed_forward"]["median_ms"]
        ),
        "flydsl_implicit_adapter_over_aiter_native_pre_routed_forward": (
            flydsl_timings["implicit_simulated_adapter_forward"]["median_ms"]
            / aiter_timings["native_pre_routed_forward"]["median_ms"]
        ),
        "flydsl_counts_native_operator_over_aiter_native_pre_routed_forward": (
            flydsl_timings["counts_native_operator_forward"]["median_ms"]
            / aiter_timings["native_pre_routed_forward"]["median_ms"]
        ),
        "flydsl_counts_native_adapter_over_aiter_native_pre_routed_forward": (
            flydsl_timings["counts_native_simulated_adapter_forward"]["median_ms"]
            / aiter_timings["native_pre_routed_forward"]["median_ms"]
        ),
        "flydsl_direct_retained_over_aiter_profiled_backward": (
            flydsl_timings["retained_backward"]["median_ms"]
            / aiter_timings["backward"]["median_ms"]
        ),
        "flydsl_implicit_retained_over_aiter_profiled_backward": (
            flydsl_timings["implicit_retained_backward"]["median_ms"]
            / aiter_timings["backward"]["median_ms"]
        ),
        "flydsl_simulated_adapter_over_aiter_profiled_e2e": (
            flydsl_timings["simulated_adapter_e2e"]["median_ms"]
            / aiter_timings["e2e"]["median_ms"]
        ),
        "flydsl_implicit_adapter_over_aiter_profiled_e2e": (
            flydsl_timings["implicit_simulated_adapter_e2e"]["median_ms"]
            / aiter_timings["e2e"]["median_ms"]
        ),
        "flydsl_counts_native_adapter_over_aiter_profiled_e2e": (
            flydsl_timings["counts_native_simulated_adapter_e2e"]["median_ms"]
            / aiter_timings["e2e"]["median_ms"]
        ),
    }
    if lumen_timings is not None:
        result.update(
            {
                "lumen_adapter_over_aiter_forward": (
                    lumen_timings["adapter_forward"]["median_ms"]
                    / aiter_timings["adapter_forward"]["median_ms"]
                ),
                "lumen_adapter_over_aiter_backward": (
                    lumen_timings["backward"]["median_ms"]
                    / aiter_timings["backward"]["median_ms"]
                ),
                "lumen_adapter_over_aiter_e2e": (
                    lumen_timings["e2e"]["median_ms"]
                    / aiter_timings["e2e"]["median_ms"]
                ),
            }
        )
    return result


def _run_case(
    routes: int,
    counts_host: list[int],
    *,
    counts_source: str,
    seed: int,
    warmup: int,
    iters: int,
    aiter_bundle,
    aiter_error: str | None,
    aiter_counts_device: str,
    lumen_adapter,
    lumen_error: str | None,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    case = _build_case(routes, counts_host, seed)
    (
        forward_correctness,
        backward_correctness,
        flydsl_output,
        flydsl_gradients,
    ) = _correctness(case)
    flydsl_timings = _flydsl_timings(case, warmup, iters)
    result: dict[str, Any] = {
        "schema": "flydsl.sonic_e16.lumen_repro.v2",
        "audited_lumen_reference_commit": LUMEN_AUDIT_COMMIT,
        "provenance": provenance,
        "device": torch.cuda.get_device_name(case.hidden.device),
        "routes": routes,
        "experts": E,
        "hidden_size": H,
        "intermediate_size": I,
        "counts_source": counts_source,
        "counts_seed": seed if counts_source == "synthetic-lumen-softmax-randn" else None,
        "expert_counts": counts_host,
        "active_experts": sum(count > 0 for count in counts_host),
        "expert_count_min": min(counts_host),
        "expert_count_max": max(counts_host),
        "expert_max_to_mean": max(counts_host) / (routes / E) if routes else 0.0,
        "warmup": warmup,
        "iters": iters,
        "single_gpu_only": True,
        "includes_ep_all_to_all": False,
        "production_tiles": {
            "stage1": [128, 128, 64],
            "stage2": [64, 256, 64],
        },
        "max_cached_workspaces": 1,
        "aiter_counts_device": aiter_counts_device,
        "forward_correctness": forward_correctness,
        "backward_correctness": backward_correctness,
        "legacy_route_metadata": [
            "expert_offsets_cumsum",
            "identity_token_arange",
            "expert_repeat_interleave",
        ],
        "implicit_route_metadata": ["expert_offsets_cumsum"],
        "counts_native_route_metadata": [],
        "route_policy_size": None,
        "route_policy_source": "local routes; audited Lumen adapter passes no EP-wide hint",
        "simulated_adapter_matches_lumen_state_gate": routes >= 64,
        "flydsl": flydsl_timings,
    }
    lumen_timings = None
    if lumen_adapter is None:
        result["lumen_adapter"] = None
        result["lumen_adapter_error"] = lumen_error
    else:
        lumen_timings, lumen_output, lumen_gradients = _lumen_adapter_timings(
            case,
            lumen_adapter,
            warmup,
            iters,
        )
        lumen_difference = (flydsl_output.float() - lumen_output.float()).abs()
        result["lumen_adapter"] = lumen_timings
        result["forward_correctness"].update(
            {
                "operator_vs_lumen_adapter_bitwise": bool(
                    torch.equal(flydsl_output, lumen_output)
                ),
                "operator_vs_lumen_adapter_max_abs": (
                    float(lumen_difference.max()) if lumen_difference.numel() else 0.0
                ),
            }
        )
        result["backward_correctness"]["flydsl_vs_lumen_adapter"] = (
            _compare_gradients(flydsl_gradients, lumen_gradients)
        )
    if routes == 0 and aiter_bundle is not None:
        # The audited ccd9200b pre-routed activation path launches a zero
        # grid. FlyDSL and Lumen have explicit empty-route contracts, while
        # this AITER baseline does not, so preserve the useful FlyDSL row.
        result["aiter"] = None
        result["aiter_error"] = "unsupported_empty_routes_by_audited_aiter"
    elif aiter_bundle is None:
        result["aiter"] = None
        result["aiter_error"] = aiter_error
    else:
        aiter_timings, aiter_output, aiter_gradients = _aiter_timings(
            case,
            *aiter_bundle,
            warmup,
            iters,
            aiter_counts_device,
        )
        result["aiter"] = aiter_timings
        result["comparison"] = _comparison(
            flydsl_output,
            aiter_output,
            flydsl_timings,
            aiter_timings,
            lumen_timings,
        )
        result["backward_correctness"]["flydsl_vs_aiter"] = _compare_gradients(
            flydsl_gradients,
            aiter_gradients,
            candidate_aiter_layout=True,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--routes",
        default=",".join(map(str, DEFAULT_ROUTES)),
        help="comma-separated synthetic route counts (default: 8191,8192,9000)",
    )
    parser.add_argument(
        "--counts",
        default=None,
        help="exact comma-separated 16-expert counts; implies one route case",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--lumen-root",
        default=os.environ.get("LUMEN_ROOT"),
        help=(
            "optional Lumen checkout; when set, also measures the actual "
            "lumen/ops/moe/flydsl_grouped.py autograd adapter"
        ),
    )
    parser.add_argument(
        "--require-lumen",
        action="store_true",
        help="fail instead of reporting lumen_adapter_error when Lumen cannot be loaded",
    )
    parser.add_argument(
        "--aiter-root",
        default=os.environ.get("AITER_ROOT"),
        help="optional AITER checkout; omit to use an importable installed AITER",
    )
    parser.add_argument(
        "--aiter-counts-device",
        choices=("cpu", "gpu"),
        default="gpu",
        help=(
            "AITER count placement: gpu matches Lumen's standalone benchmark; "
            "cpu replays tuple-based production dispatch when host counts are present"
        ),
    )
    parser.add_argument(
        "--no-aiter",
        action="store_true",
        help="skip the optional AITER comparison",
    )
    parser.add_argument(
        "--require-aiter",
        action="store_true",
        help="fail instead of reporting aiter_error when AITER cannot be imported",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("a CUDA/HIP device is required")
    if args.warmup < 0 or args.iters <= 0:
        parser.error("warmup must be non-negative and iters must be positive")
    if args.no_aiter and args.require_aiter:
        parser.error("--no-aiter and --require-aiter are mutually exclusive")

    if args.counts is not None:
        try:
            exact_counts = _parse_int_list(args.counts, name="counts")
            _validate_counts(exact_counts)
        except (argparse.ArgumentTypeError, ValueError) as exc:
            parser.error(str(exc))
        cases = [(sum(exact_counts), exact_counts, "explicit-expert-counts")]
    else:
        try:
            routes_values = _parse_int_list(args.routes, name="routes")
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
        if any(routes < 0 for routes in routes_values):
            parser.error("routes must be non-negative")
        cases = [
            (
                routes,
                _lumen_synthetic_counts(routes, args.seed),
                "synthetic-lumen-softmax-randn",
            )
            for routes in routes_values
        ]

    aiter_bundle = None
    aiter_error = None
    aiter_source = None
    if not args.no_aiter:
        try:
            aiter_bundle, aiter_source = _load_aiter(args.aiter_root)
        except Exception as exc:  # noqa: BLE001 - optional comparison by design
            aiter_error = f"{type(exc).__name__}: {exc}"
            if args.require_aiter:
                raise

    lumen_adapter = None
    lumen_error = None
    lumen_source = None
    if args.lumen_root:
        try:
            lumen_adapter, lumen_source = _load_lumen_adapter(args.lumen_root)
        except Exception as exc:  # noqa: BLE001 - optional exact adapter by design
            lumen_error = f"{type(exc).__name__}: {exc}"
            if args.require_lumen:
                raise
    elif args.require_lumen:
        parser.error("--require-lumen requires --lumen-root or LUMEN_ROOT")

    flydsl_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lumen_root = (
        os.path.abspath(os.path.expanduser(args.lumen_root))
        if args.lumen_root
        else None
    )
    aiter_root = (
        os.path.abspath(os.path.expanduser(args.aiter_root))
        if args.aiter_root
        else None
    )
    provenance = {
        "flydsl_root": flydsl_root,
        "flydsl_head": _git_head(flydsl_root),
        "flydsl_dirty": _git_dirty(flydsl_root),
        "benchmark_sha256": _file_sha256(os.path.abspath(__file__)),
        "lumen_root": lumen_root,
        "lumen_head": _git_head(lumen_root),
        "lumen_adapter_source": lumen_source,
        "aiter_root": aiter_root,
        "aiter_head": _git_head(aiter_root),
        "aiter_source": aiter_source,
    }

    for routes, counts_host, counts_source in cases:
        _validate_counts(counts_host)
        if sum(counts_host) != routes:
            raise RuntimeError("internal count generation did not sum to routes")
        result = _run_case(
            routes,
            counts_host,
            counts_source=counts_source,
            seed=args.seed,
            warmup=args.warmup,
            iters=args.iters,
            aiter_bundle=aiter_bundle,
            aiter_error=aiter_error,
            aiter_counts_device=args.aiter_counts_device,
            lumen_adapter=lumen_adapter,
            lumen_error=lumen_error,
            provenance=provenance,
        )
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
