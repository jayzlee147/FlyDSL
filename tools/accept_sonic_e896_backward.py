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

The default ``legacy-vs-hostless`` comparison retains the original acceptance
contract: the baseline must exercise legacy projection, dX, and host segment
materialization while the candidate must use the fully grouped hostless path.
The ``incremental`` comparison is for tuning the current hostless
implementation against a small candidate change.  It requires both sources to
select the large grouped-dX descriptor queue and retained-state hostless policy,
and launch-audits one grouped dX for each source without requiring legacy work
from the baseline.  Candidate generic GEMM and host materialization remain
forbidden in both modes.

Formal timing is refused unless ``--exclusive-gpu`` is supplied.  On a
shared/busy machine use ``--correctness-only`` for the acceptance checks or
explicit ``--diagnostic-shared-gpu`` for non-acceptance ABBA/BAAB observations.
The diagnostic mode can never report formal acceptance even when its observed
thresholds pass.

Examples
--------
Correctness and launch audit only::

    PYTHONPATH=. python tools/accept_sonic_e896_backward.py \
        --baseline /path/to/baseline/kernels/moe/sonic_backward.py \
        --correctness-only --cases balanced hot16 \
        --output /tmp/sonic-e896-correctness.json

Formal timing on an independently reserved gfx950::

    PYTHONPATH=. python tools/accept_sonic_e896_backward.py \
        --baseline /path/to/baseline/kernels/moe/sonic_backward.py \
        --exclusive-gpu --pairs 11 --cases balanced hot16 \
        --output /tmp/sonic-e896-abba.json

Strict hostless-to-hostless incremental timing::

    PYTHONPATH=. python tools/accept_sonic_e896_backward.py \
        --comparison-mode incremental \
        --baseline /path/to/hostless-baseline/kernels/moe/sonic_backward.py \
        --exclusive-gpu --pairs 11 --cases balanced hot16 \
        --output /tmp/sonic-e896-incremental-abba.json

Incremental timing with one privately isolated baseline runtime source::

    PYTHONPATH=. python tools/accept_sonic_e896_backward.py \
        --comparison-mode incremental \
        --baseline /path/to/baseline/kernels/moe/sonic_backward.py \
        --baseline-runtime-override kernels/moe/sonic_grouped_tn.py \
        --exclusive-gpu --pairs 11 --cases balanced hot16 \
        --output /tmp/sonic-e896-runtime-override-abba.json

Non-acceptance timing diagnostics on a shared gfx950::

    PYTHONPATH=. python tools/accept_sonic_e896_backward.py \
        --comparison-mode incremental \
        --baseline /path/to/baseline/kernels/moe/sonic_backward.py \
        --diagnostic-shared-gpu --pairs 3 --cases balanced hot16 \
        --output /tmp/sonic-e896-shared-gpu-diagnostic.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import inspect
import json
import statistics
import subprocess
import sys
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
COMPARISON_MODES = ("legacy-vs-hostless", "incremental")
CHUNK_ELEMENTS = 32 * 1024 * 1024

# The E896 adapter initialization keeps activations and weights near zero, so
# several gradients have very small norms: a one-ULP A16 contraction-order
# change is about 0.2--0.5% in relative L2 even though max-absolute drift is
# below 1.2e-7.  Keep a 1% global-error gate (still 3x tighter than the regular
# 3% elementwise reference tolerance), together with explicit max-absolute and
# norm-preservation gates.  dW2 is bitwise identical and retains its tighter
# historical limit.
RELATIVE_L2_LIMITS = {
    "dx": 1.0e-2,
    "dw1": 1.0e-2,
    "dw2": 1.0e-4,
    "dtopk_weights": 1.0e-2,
}
MAX_ABS_LIMITS = {
    "dx": 0.0625,
    "dw1": 0.5,
    "dw2": 0.5,
    "dtopk_weights": 0.25,
}
NORM_RATIO_TOLERANCE = 5.0e-5

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

# A same-process comparison normally isolates only sonic_backward.py.  These
# are the complete imported-symbol boundaries that may instead be rebound to a
# privately loaded baseline runtime source when explicitly requested.  Keep
# the relative paths exact: this is an allowlist, not a filename/glob filter.
BASELINE_RUNTIME_OVERRIDE_SYMBOLS = {
    "kernels/moe/grouped_da_gfx950.py": ("compile_grouped_da_gfx950",),
    "kernels/moe/sonic_grouped_a16w16.py": ("compile_sonic_grouped_a16w16_nn",),
    "kernels/moe/sonic_grouped_tn.py": (
        "active_expert_descriptor_capacity",
        "active_expert_queue_elements",
        "build_active_expert_queue_flydsl",
        "grouped_tn_from_metadata_flydsl",
        "grouped_tn_from_queue_flydsl",
        "zero_inactive_weight_grads_flydsl",
        "zero_weight_grads_adaptive_flydsl",
    ),
}
BASELINE_RUNTIME_OVERRIDE_CHOICES = tuple(BASELINE_RUNTIME_OVERRIDE_SYMBOLS)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_identity(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _verify_file_identity(expected: dict[str, str], *, label: str) -> dict[str, str]:
    observed = _file_identity(Path(expected["path"]))
    if observed != expected:
        raise RuntimeError(f"{label} changed after code identity collection: expected={expected}, observed={observed}")
    return observed


def _git_value(repo: Path, *args: str) -> str | None:
    resolved_repo = repo.resolve()
    try:
        completed = subprocess.run(
            ("git", "-C", str(resolved_repo), "-c", f"safe.directory={resolved_repo}", *args),
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


def _load_isolated_baseline(path: Path, *, expected_identity: dict[str, str] | None = None):
    module = _load_source_module("kernels.moe._e896_acceptance_baseline", path)
    loaded_from = Path(inspect.getsourcefile(module.sonic_moe_backward) or "").resolve()
    if loaded_from != path:
        raise RuntimeError(f"baseline sonic_moe_backward isolation failed: expected {path}, loaded {loaded_from}")
    if expected_identity is not None:
        _verify_file_identity(expected_identity, label="baseline sonic_backward.py")
    return module


def _normalize_baseline_runtime_overrides(values: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Validate and return runtime overrides in stable allowlist order."""

    requested = tuple(values or ())
    unknown = sorted(set(requested) - set(BASELINE_RUNTIME_OVERRIDE_CHOICES))
    if unknown:
        raise ValueError("unknown baseline runtime override(s): " + ", ".join(unknown))
    duplicates = sorted({value for value in requested if requested.count(value) > 1})
    if duplicates:
        raise ValueError("duplicate baseline runtime override(s): " + ", ".join(duplicates))
    requested_set = set(requested)
    return tuple(relative for relative in BASELINE_RUNTIME_OVERRIDE_CHOICES if relative in requested_set)


def _runtime_module_name(relative: str) -> str:
    return relative.removesuffix(".py").replace("/", ".")


def _private_runtime_module_name(relative: str, path: Path) -> str:
    fingerprint_source = f"{path.resolve()}:{_sha256(path)}".encode()
    fingerprint = hashlib.sha256(fingerprint_source).hexdigest()[:12]
    return f"kernels.moe._e896_baseline_runtime_{Path(relative).stem}_{fingerprint}"


def _load_private_runtime_module(
    relative: str,
    path: Path,
    *,
    expected_identity: dict[str, str] | None = None,
):
    """Load one allowlisted baseline source without replacing its canonical module."""

    if relative not in BASELINE_RUNTIME_OVERRIDE_SYMBOLS:
        raise ValueError(f"unknown baseline runtime override: {relative}")
    private_name = _private_runtime_module_name(relative, path)
    spec = importlib.util.spec_from_file_location(private_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load baseline runtime {relative} from {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(private_name)
    sys.modules[private_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(private_name, None)
        else:
            sys.modules[private_name] = previous
        raise
    loaded_from = Path(module.__file__ or "").resolve()
    if loaded_from != path.resolve():
        if previous is None:
            sys.modules.pop(private_name, None)
        else:
            sys.modules[private_name] = previous
        raise RuntimeError(f"baseline runtime isolation failed for {relative}: expected {path}, loaded {loaded_from}")
    loaded_identity = _file_identity(loaded_from)
    if expected_identity is not None and loaded_identity != expected_identity:
        if previous is None:
            sys.modules.pop(private_name, None)
        else:
            sys.modules[private_name] = previous
        raise RuntimeError(
            f"baseline runtime override {relative} changed after code identity collection: "
            f"expected={expected_identity}, observed={loaded_identity}"
        )
    module.__e896_loaded_source_identity__ = loaded_identity
    return module


def _load_baseline_runtime_overrides(
    baseline_repo: Path,
    runtime_overrides: tuple[str, ...],
    *,
    expected_sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_overrides = _normalize_baseline_runtime_overrides(runtime_overrides)
    modules = {}
    for relative in runtime_overrides:
        path = (baseline_repo / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"baseline runtime override does not exist: {path}")
        expected_identity = None if expected_sources is None else expected_sources[relative]["baseline"]
        modules[relative] = _load_private_runtime_module(
            relative,
            path,
            expected_identity=expected_identity,
        )
    return modules


def _callable_identity(value: Any) -> dict[str, str | None]:
    source = inspect.getsourcefile(inspect.unwrap(value))
    return {
        "module": getattr(value, "__module__", None),
        "name": getattr(value, "__name__", None),
        "qualname": getattr(value, "__qualname__", None),
        "source": str(Path(source).resolve()) if source else None,
    }


def _validate_runtime_symbol_origin(
    value: Any,
    *,
    symbol: str,
    expected_module: str,
    expected_path: Path,
    role: str,
) -> None:
    if not callable(value):
        raise RuntimeError(f"{role} runtime symbol {expected_module}.{symbol} is not callable")
    unwrapped = inspect.unwrap(value)
    observed_module = getattr(unwrapped, "__module__", None)
    source = inspect.getsourcefile(unwrapped)
    observed_path = Path(source).resolve() if source else None
    if observed_module != expected_module or observed_path != expected_path.resolve():
        raise RuntimeError(
            f"{role} runtime symbol {symbol} has the wrong origin: "
            f"expected module={expected_module} source={expected_path.resolve()}, "
            f"observed module={observed_module} source={observed_path}"
        )


def _patch_baseline_runtime_symbols(
    baseline_module: Any,
    runtime_modules: dict[str, Any],
    *,
    expected_sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preflight and atomically rebind baseline imports to private modules."""

    pending = []
    report = {}
    runtime_overrides = _normalize_baseline_runtime_overrides(tuple(runtime_modules))
    for relative in runtime_overrides:
        private_module = runtime_modules[relative]
        canonical_name = _runtime_module_name(relative)
        canonical_module = importlib.import_module(canonical_name)
        canonical_path = Path(canonical_module.__file__ or "").resolve()
        expected_canonical_path = (Path(__file__).resolve().parents[1] / relative).resolve()
        if canonical_path != expected_canonical_path:
            raise RuntimeError(
                f"canonical runtime import does not resolve to this checkout for {relative}: "
                f"expected {expected_canonical_path}, loaded {canonical_path}"
            )
        private_path = Path(private_module.__file__ or "").resolve()
        canonical_identity = _file_identity(canonical_path)
        private_identity = getattr(private_module, "__e896_loaded_source_identity__", None)
        if private_identity is None:
            private_identity = _file_identity(private_path)
        if expected_sources is not None:
            expected = expected_sources[relative]
            if canonical_identity != expected["candidate"]:
                raise RuntimeError(
                    f"candidate runtime {relative} changed after code identity collection: "
                    f"expected={expected['candidate']}, observed={canonical_identity}"
                )
            if private_identity != expected["baseline"]:
                raise RuntimeError(
                    f"loaded baseline runtime {relative} does not match the authorized identity: "
                    f"expected={expected['baseline']}, observed={private_identity}"
                )
        symbol_reports = []
        for symbol in BASELINE_RUNTIME_OVERRIDE_SYMBOLS[relative]:
            if not hasattr(baseline_module, symbol):
                raise RuntimeError(f"baseline sonic_backward.py is missing imported runtime symbol {symbol}")
            if not hasattr(canonical_module, symbol):
                raise RuntimeError(f"canonical runtime {canonical_name} is missing symbol {symbol}")
            if not hasattr(private_module, symbol):
                raise RuntimeError(f"private baseline runtime {private_module.__name__} is missing symbol {symbol}")

            original = getattr(baseline_module, symbol)
            canonical = getattr(canonical_module, symbol)
            replacement = getattr(private_module, symbol)
            if original is not canonical:
                raise RuntimeError(
                    f"baseline runtime symbol {symbol} was not imported from the expected canonical module "
                    f"{canonical_name}"
                )
            _validate_runtime_symbol_origin(
                canonical,
                symbol=symbol,
                expected_module=canonical_name,
                expected_path=canonical_path,
                role="canonical",
            )
            _validate_runtime_symbol_origin(
                replacement,
                symbol=symbol,
                expected_module=private_module.__name__,
                expected_path=private_path,
                role="private baseline",
            )
            pending.append((symbol, replacement))
            symbol_reports.append(
                {
                    "symbol": symbol,
                    "original": _callable_identity(original),
                    "replacement": _callable_identity(replacement),
                }
            )
        report[relative] = {
            "canonical_module": canonical_name,
            "canonical_source": canonical_identity,
            "private_module": private_module.__name__,
            "loaded_source": private_identity,
            "patched_symbols": symbol_reports,
        }

    for symbol, replacement in pending:
        setattr(baseline_module, symbol, replacement)
    return report


def _evaluate_policy_probe(
    comparison_mode: str,
    *,
    baseline_large_grouped_dx: bool,
    candidate_large_grouped_dx: bool,
    baseline_hostless_retained_backward: bool | None,
    candidate_hostless_retained_backward: bool,
) -> dict[str, Any]:
    """Evaluate mode-specific policy observations without touching a GPU."""

    if comparison_mode not in COMPARISON_MODES:
        raise ValueError(f"unknown comparison mode {comparison_mode!r}")

    if comparison_mode == "legacy-vs-hostless":
        checks = {
            "baseline_uses_legacy_dx": baseline_large_grouped_dx is False,
            "candidate_uses_large_grouped_dx": candidate_large_grouped_dx is True,
            "candidate_uses_hostless_retained_backward": candidate_hostless_retained_backward is True,
        }
    else:
        checks = {
            "baseline_uses_large_grouped_dx": baseline_large_grouped_dx is True,
            "candidate_uses_large_grouped_dx": candidate_large_grouped_dx is True,
            "baseline_uses_hostless_retained_backward": baseline_hostless_retained_backward is True,
            "candidate_uses_hostless_retained_backward": candidate_hostless_retained_backward is True,
        }

    return {
        "comparison_mode": comparison_mode,
        "baseline_large_grouped_dx": baseline_large_grouped_dx,
        "candidate_large_grouped_dx": candidate_large_grouped_dx,
        "baseline_hostless_retained_backward": baseline_hostless_retained_backward,
        "candidate_hostless_retained_backward": candidate_hostless_retained_backward,
        "required_checks": checks,
        "passed": all(checks.values()),
    }


def _collect_code_identity(
    baseline_path: Path,
    baseline_runtime_overrides: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    runtime_overrides = _normalize_baseline_runtime_overrides(baseline_runtime_overrides)
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
    backward_identical = _sha256(baseline_path) == _sha256(candidate_path)
    if not runtime_overrides and backward_identical:
        raise RuntimeError("candidate and baseline sonic_backward.py are byte-identical")

    shared_sources: dict[str, Any] = {}
    mismatches = []
    redundant_overrides = []
    for relative in SHARED_RUNTIME_SOURCES:
        baseline_source = baseline_repo / relative
        candidate_source = candidate_repo / relative
        if not baseline_source.is_file():
            raise FileNotFoundError(f"baseline shared runtime source does not exist: {baseline_source}")
        if not candidate_source.is_file():
            raise FileNotFoundError(f"candidate shared runtime source does not exist: {candidate_source}")
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
        elif relative in runtime_overrides:
            redundant_overrides.append(relative)
    if not runtime_overrides and mismatches:
        raise RuntimeError(
            "this single-file-isolation benchmark requires byte-identical shared "
            "runtime sources; mismatches: " + ", ".join(mismatches)
        )
    if runtime_overrides:
        unlisted_mismatches = [relative for relative in mismatches if relative not in runtime_overrides]
        failures = []
        if unlisted_mismatches:
            failures.append("unlisted shared runtime mismatches: " + ", ".join(unlisted_mismatches))
        if redundant_overrides:
            failures.append("selected runtime overrides are byte-identical: " + ", ".join(redundant_overrides))
        if failures:
            raise RuntimeError("invalid baseline runtime override identity: " + "; ".join(failures))

    identity = {
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
        "shared_runtime_sources_identical": not mismatches,
    }
    if runtime_overrides:
        identity["baseline_runtime_overrides"] = {
            "requested": list(runtime_overrides),
            "sonic_backward_identical": backward_identical,
            "sources": {
                relative: {
                    "baseline": shared_sources[relative]["baseline"],
                    "candidate": shared_sources[relative]["candidate"],
                    "identical": shared_sources[relative]["identical"],
                }
                for relative in runtime_overrides
            },
        }
    return identity


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
    original_materialize = module._materialize_expert_segments
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
        "host_segment_materializations": 0,
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

    def wrapped_materialize(*args, **kwargs):
        counts["host_segment_materializations"] += 1
        return original_materialize(*args, **kwargs)

    module._run_compiled = wrapped_run
    module.gemm_a16w16 = wrapped_gemm
    module._materialize_expert_segments = wrapped_materialize
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
        module._materialize_expert_segments = original_materialize
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
        "host_segment_materialization_zero": counts["host_segment_materializations"] == 0,
    }
    counts["generic_gemms"] = generic_gemms
    return gradients, counts


def _evaluate_launch_topology(
    comparison_mode: str,
    baseline_launches: dict[str, Any],
    candidate_launches: dict[str, Any],
) -> dict[str, Any]:
    """Return the launch gate while keeping non-required observations visible."""

    if comparison_mode not in COMPARISON_MODES:
        raise ValueError(f"unknown comparison mode {comparison_mode!r}")

    checks = {
        "candidate_generic_gemm_zero": candidate_launches["generic_gemm_launches"] == 0,
        "candidate_projection_zero": candidate_launches["projection_launches"] == 0,
        "candidate_legacy_dx_zero": candidate_launches["legacy_dx_gemm_launches"] == 0,
        "candidate_host_segment_materialization_zero": (candidate_launches["host_segment_materializations"] == 0),
        "candidate_single_grouped_dx": candidate_launches["grouped_dx_launches"] == 1,
        "candidate_single_total_dx": candidate_launches["total_dx_launches"] == 1,
        "baseline_single_grouped_dx": baseline_launches["grouped_dx_launches"] == 1,
        "baseline_single_total_dx": baseline_launches["total_dx_launches"] == 1,
        "baseline_generic_gemm_zero": baseline_launches["generic_gemm_launches"] == 0,
        "baseline_host_segment_materialization_zero": baseline_launches["host_segment_materializations"] == 0,
        "baseline_exercises_legacy_dx": baseline_launches["legacy_dx_gemm_launches"] > 0,
        "baseline_exercises_projection": baseline_launches["projection_launches"] > 0,
        "baseline_exercises_host_segment_materialization": (baseline_launches["host_segment_materializations"] > 0),
    }
    candidate_requirements = (
        "candidate_generic_gemm_zero",
        "candidate_projection_zero",
        "candidate_legacy_dx_zero",
        "candidate_host_segment_materialization_zero",
        "candidate_single_grouped_dx",
        "candidate_single_total_dx",
    )
    if comparison_mode == "legacy-vs-hostless":
        required_checks = candidate_requirements + (
            "baseline_exercises_legacy_dx",
            "baseline_exercises_projection",
            "baseline_exercises_host_segment_materialization",
        )
    else:
        required_checks = candidate_requirements + ("baseline_single_grouped_dx",)

    return {
        **checks,
        "required_checks": list(required_checks),
        "passed": all(checks[name] for name in required_checks),
    }


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


def _acceptance_report_fields(*, observed_passed: bool, diagnostic_shared_gpu: bool) -> dict[str, bool]:
    """Keep shared-GPU observations distinct from formal acceptance."""

    if diagnostic_shared_gpu:
        return {
            "acceptance": False,
            "observed_passed": bool(observed_passed),
            "passed": False,
        }
    return {"passed": bool(observed_passed)}


def _timing_status(*, correctness_only: bool, diagnostic_shared_gpu: bool) -> str:
    if correctness_only:
        return "skipped"
    if diagnostic_shared_gpu:
        return "diagnostic-shared-gpu"
    return "measured"


def _performance_report_fields(
    performance_gates: dict[str, Any],
    *,
    observed_passed: bool,
    diagnostic_shared_gpu: bool,
) -> dict[str, Any]:
    if diagnostic_shared_gpu:
        fields = {
            "timing_status": "diagnostic-shared-gpu",
            "diagnostic_shared_gpu_asserted": True,
            "observed_performance_gates": performance_gates,
        }
    else:
        fields = {"performance_gates": performance_gates}
    fields.update(
        _acceptance_report_fields(
            observed_passed=observed_passed,
            diagnostic_shared_gpu=diagnostic_shared_gpu,
        )
    )
    return fields


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-mode",
        choices=COMPARISON_MODES,
        default="legacy-vs-hostless",
        help=(
            "legacy-vs-hostless preserves the original migration gate; incremental compares two "
            "large-grouped-dX hostless implementations"
        ),
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="baseline sonic_backward.py loaded under a private module name",
    )
    parser.add_argument(
        "--baseline-runtime-override",
        action="append",
        choices=BASELINE_RUNTIME_OVERRIDE_CHOICES,
        default=[],
        help=(
            "repeatable exact relative path whose baseline implementation is privately loaded and rebound; "
            "all unlisted shared runtime sources must remain byte-identical"
        ),
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
    parser.add_argument(
        "--diagnostic-shared-gpu",
        action="store_true",
        help=(
            "run non-acceptance ABBA/BAAB diagnostics on a shared GPU; cannot be combined with "
            "--correctness-only or --exclusive-gpu"
        ),
    )
    parser.add_argument("--min-backward-speedup", type=float, default=1.01)
    parser.add_argument("--min-full-step-speedup", type=float, default=1.01)
    parser.add_argument("--min-paired-win-rate", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        args.baseline_runtime_override = list(
            _normalize_baseline_runtime_overrides(args.baseline_runtime_override)
        )
    except ValueError as error:
        parser.error(str(error))
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
    if args.diagnostic_shared_gpu and (args.correctness_only or args.exclusive_gpu):
        parser.error("--diagnostic-shared-gpu cannot be combined with --correctness-only or --exclusive-gpu")
    if not args.correctness_only and not args.exclusive_gpu and not args.diagnostic_shared_gpu:
        parser.error(
            "timing requires --exclusive-gpu after independently reserving the selected GPU, "
            "or explicit non-acceptance --diagnostic-shared-gpu"
        )
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
    runtime_overrides = tuple(args.baseline_runtime_override)
    code_identity = _collect_code_identity(baseline_path, runtime_overrides)
    baseline_module = _load_isolated_baseline(
        baseline_path,
        expected_identity=code_identity["baseline"]["sonic_backward"],
    )
    if runtime_overrides:
        baseline_repo = baseline_path.parents[2]
        override_sources = code_identity["baseline_runtime_overrides"]["sources"]
        runtime_modules = _load_baseline_runtime_overrides(
            baseline_repo,
            runtime_overrides,
            expected_sources=override_sources,
        )
        patch_report = _patch_baseline_runtime_symbols(
            baseline_module,
            runtime_modules,
            expected_sources=override_sources,
        )
        for relative, details in patch_report.items():
            override_sources[relative].update(details)
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
    baseline_large_grouped_dx = baseline_module._use_large_grouped_dx_descriptor_queue(**policy_kwargs)
    candidate_large_grouped_dx = candidate_module._use_large_grouped_dx_descriptor_queue(**policy_kwargs)
    hostless_kwargs = {
        "flat_routes": False,
        "has_bias": False,
        "reuse_forward_preactivation": True,
        "use_grouped_w1": False,
        "use_grouped_w2": False,
        "use_grouped_dw2": True,
        "use_grouped_da": True,
        "use_grouped_dw1": True,
        "use_grouped_dx": True,
        **{key: value for key, value in policy_kwargs.items() if key != "flat_routes"},
    }
    candidate_hostless_retained_backward = candidate_module._use_hostless_grouped_backward(
        use_large_grouped_dx=candidate_large_grouped_dx,
        **hostless_kwargs,
    )
    # A historical baseline may predate this policy helper entirely.  Preserve
    # legacy-mode compatibility by probing it only when incremental mode makes
    # the baseline's hostless policy part of the acceptance contract.
    baseline_hostless_retained_backward = None
    if args.comparison_mode == "incremental":
        if not hasattr(baseline_module, "_use_hostless_grouped_backward"):
            raise RuntimeError("incremental comparison requires the baseline to expose _use_hostless_grouped_backward")
        baseline_hostless_retained_backward = baseline_module._use_hostless_grouped_backward(
            use_large_grouped_dx=baseline_large_grouped_dx,
            **hostless_kwargs,
        )
    policy_probe = _evaluate_policy_probe(
        args.comparison_mode,
        baseline_large_grouped_dx=baseline_large_grouped_dx,
        candidate_large_grouped_dx=candidate_large_grouped_dx,
        baseline_hostless_retained_backward=baseline_hostless_retained_backward,
        candidate_hostless_retained_backward=candidate_hostless_retained_backward,
    )
    if not policy_probe["passed"]:
        raise RuntimeError(f"baseline/candidate policies do not satisfy the selected E896 comparison: {policy_probe}")

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
    # The prepared operator owns its preshuffled Stage-1 storage.  Keeping the
    # separated source tensor alive adds another multi-GiB copy but is not part
    # of either backward implementation under test.
    del separated_w1
    gc.collect()
    modules = {"baseline": baseline_module, "candidate": candidate_module}

    report: dict[str, Any] = {
        "schema": "flydsl.sonic_e896_backward_acceptance.v2",
        "comparison": {
            "mode": args.comparison_mode,
            "source_isolation": (
                "sonic_backward.py plus explicit private baseline runtime overrides; "
                "all unlisted shared runtime sources must be byte-identical"
                if runtime_overrides
                else "sonic_backward.py only; shared runtime sources must be byte-identical"
            ),
            "timing_design": "paired ABBA/BAAB for backward and full-step",
            "baseline_role": (
                "legacy projection/readback path"
                if args.comparison_mode == "legacy-vs-hostless"
                else "large-grouped-dX retained-state hostless reference"
            ),
            "candidate_role": "large-grouped-dX retained-state hostless implementation",
        },
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
    overall_correctness_passed = True
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
        # Accuracy has been reduced to scalar metrics.  Release the roughly
        # 10-GiB baseline gradient tuple before allocating repeatability output.
        del baseline_gradients
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
        launch_gate = _evaluate_launch_topology(args.comparison_mode, baseline_launches, candidate_launches)
        launch_topology_passed = launch_gate["passed"]
        case_report: dict[str, Any] = {
            "routing": routing_summary,
            "candidate_vs_baseline": accuracy,
            "candidate_repeatability": repeatability,
            "launches": {
                "baseline": baseline_launches,
                "candidate": candidate_launches,
                "gate": launch_gate,
            },
            "gates": {
                "accuracy": accuracy_passed,
                "bitwise_repeatability": repeatability_passed,
                "launch_topology": launch_topology_passed,
            },
        }
        del candidate_gradients
        gc.collect()

        correctness_passed = accuracy_passed and repeatability_passed and launch_topology_passed
        overall_correctness_passed &= correctness_passed
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
        case_observed_passed = correctness_passed and performance_passed
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
            }
        )
        case_report.update(
            _performance_report_fields(
                performance_gates,
                observed_passed=case_observed_passed,
                diagnostic_shared_gpu=args.diagnostic_shared_gpu,
            )
        )
        report["cases"][case] = case_report
        overall_passed &= case_observed_passed
        gc.collect()

    report["timing_status"] = _timing_status(
        correctness_only=args.correctness_only,
        diagnostic_shared_gpu=args.diagnostic_shared_gpu,
    )
    report["exclusive_gpu_asserted"] = args.exclusive_gpu
    if args.diagnostic_shared_gpu:
        report["diagnostic_shared_gpu_asserted"] = True
        report["diagnostic_valid"] = bool(overall_correctness_passed and policy_probe["passed"])
    report.update(
        _acceptance_report_fields(
            observed_passed=bool(overall_passed and policy_probe["passed"]),
            diagnostic_shared_gpu=args.diagnostic_shared_gpu,
        )
    )
    _emit_report(report, args.output)
    if args.diagnostic_shared_gpu:
        if not report["diagnostic_valid"]:
            raise SystemExit(2)
    elif not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
