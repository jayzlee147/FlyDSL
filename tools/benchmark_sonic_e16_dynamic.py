# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Benchmark dynamic Qwen3-30B-A3B E16 flat-route training on gfx950.

The three modes isolate the production dispatch differences:

* ``generic``: no retained state in backward (legacy dynamic-R fallback);
* ``retained``: retain forward sorter metadata and use segmented dX;
* ``expert-major``: consume expert offsets, skip counting sort, use identity
  Stage-2 stores, and write dX directly.

Run modes in separate processes for allocator and JIT isolation.

``--matrix`` runs the canonical Qwen3 route-count/load matrix.  The separate
``--workspace-sequence`` mode keeps one operator alive and reports the
growable workspace's resident and transient allocation as route counts grow.
``--row-grid-abba`` compares 1024- and 2048-CTA hostless row-grid caps with
paired backward-only GPU-event timings.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys

import torch

from kernels.moe import sonic_backward as sonic_backward_module
from kernels.moe.sonic import (
    SonicMoE,
    SonicMoEConfig,
    SonicMoEDynamicWorkspacePool,
    prepare_sonic_bf16_weights,
)
from kernels.moe.sonic_backward import sonic_moe_backward_routes


_QWEN3_MATRIX_ROUTES = (4096, 8191, 8192, 16384, 32768, 65536)
_QWEN3_MATRIX_LOADS = ("balanced", "softmax-multinomial", "hot4")
_QWEN3_WORKSPACE_ROUTES = (8192, 65536, 4096, 16384)


def _counts(routes: int, experts: int, load: str, seed: int) -> list[int]:
    if load in ("uniform", "balanced"):
        quotient, remainder = divmod(routes, experts)
        return [quotient + (expert < remainder) for expert in range(experts)]
    if load == "softmax-multinomial":
        # A fixed logit vector lets route-count sweeps scale the same skew
        # profile.  The 1.5 sigma produces a realistic all-expert imbalance
        # (roughly 18% hottest probability for this seed) without degenerating
        # into the deliberately sparse hot-expert stress cases below.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        probabilities = torch.softmax(
            1.5 * torch.randn(experts, generator=generator),
            dim=0,
        )
        samples = torch.multinomial(
            probabilities,
            routes,
            replacement=True,
            generator=generator,
        )
        return torch.bincount(samples, minlength=experts).tolist()
    if load == "hot1":
        return [routes, *([0] * (experts - 1))]
    if load == "hot2":
        first = (routes * 4 + 4) // 5
        return [first, routes - first, *([0] * (experts - 2))]
    if load == "hot4":
        quotient, remainder = divmod(routes, 4)
        return [
            *[quotient + (expert < remainder) for expert in range(4)],
            *([0] * (experts - 4)),
        ]
    if load == "long-tail":
        first = (routes + 1) // 2
        quotient, remainder = divmod(routes - first, experts - 1)
        return [first, *[quotient + (expert < remainder) for expert in range(experts - 1)]]
    raise ValueError(f"unknown load {load!r}")


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _e16_config() -> SonicMoEConfig:
    return SonicMoEConfig(
        hidden_size=2048,
        intermediate_size=768,
        num_experts=16,
        top_k=1,
        tile_m=128,
        tile_n=192,
        tile_k=64,
        down_tile_m=64,
        down_tile_n=256,
        down_tile_k=64,
        stage1_xcd_swizzle=8,
        stage2_xcd_swizzle=0,
        stage2_pipeline_stages=2,
        stage1_write_padded_rows=True,
        stage1_lds_swizzle=True,
    )


def _build_case(
    routes: int,
    load: str,
    seed: int,
    mode: str,
    *,
    shared_dynamic_workspace: bool = False,
):
    device = torch.device("cuda")
    config = _e16_config()
    experts = config.num_experts
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    counts = _counts(routes, experts, load, seed)
    offsets_host = [0]
    for count in counts:
        offsets_host.append(offsets_host[-1] + count)
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        (routes, hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.2)
    w1 = torch.randn(
        (experts, 2 * intermediate_size, hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.02)
    w2 = torch.randn(
        (experts, hidden_size, intermediate_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.02)
    grad_output = torch.randn(
        (routes, hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.2)
    token_indices = torch.arange(routes, dtype=torch.int32, device=device)
    expert_indices = torch.repeat_interleave(
        torch.arange(experts, dtype=torch.int32, device=device),
        torch.tensor(counts, dtype=torch.int64, device=device),
    ).contiguous()
    expert_offsets = torch.tensor(offsets_host, dtype=torch.int32, device=device)
    route_weights = torch.linspace(
        0.25,
        1.0,
        routes,
        dtype=torch.float32,
        device=device,
    )
    output = torch.empty_like(x)
    pool = SonicMoEDynamicWorkspacePool() if shared_dynamic_workspace else None
    operator = SonicMoE(
        config,
        prepare_sonic_bf16_weights(w1, w2, config),
        shared_dynamic_workspace_pool=pool,
    )

    def forward():
        kwargs = {}
        if mode == "expert-major":
            kwargs = {
                "expert_offsets": expert_offsets,
                "token_indices_identity": True,
            }
        return operator.forward_routes_training(
            x,
            token_indices,
            expert_indices,
            route_weights,
            out=output,
            **kwargs,
        )

    def backward(state):
        return sonic_moe_backward_routes(
            x,
            w1,
            w2,
            token_indices,
            expert_indices,
            route_weights,
            grad_output,
            config,
            forward_state=(None if mode == "generic" else state),
            token_indices_sorted=True,
        )

    return device, config, counts, x, operator, forward, backward


def _run_qwen3_matrix(args: argparse.Namespace) -> None:
    """Run canonical cases in isolated processes and emit one JSON row each."""

    for routes in _QWEN3_MATRIX_ROUTES:
        for load in _QWEN3_MATRIX_LOADS:
            command = [
                sys.executable,
                __file__,
                "--routes",
                str(routes),
                "--mode",
                args.mode,
                "--load",
                load,
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--seed",
                str(args.seed),
            ]
            if args.shared_dynamic_workspace:
                command.append("--shared-dynamic-workspace")
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            if not lines:
                raise RuntimeError(f"matrix child produced no result: {command!r}")
            # Re-serialize to validate every child row and keep JSONL output
            # free of any incidental compiler logging.
            result = json.loads(lines[-1])
            print(json.dumps(result, sort_keys=True), flush=True)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _compare_gradients(reference, candidate) -> list[dict[str, object]]:
    names = ("dx", "dw1", "dw2", "droute_weights")
    if len(reference) != len(names) or len(candidate) != len(names):
        raise RuntimeError("bias-free E16 backward must return four gradients")
    checks = []
    for name, expected, actual in zip(names, reference, candidate):
        bitwise_equal = bool(torch.equal(expected, actual))
        max_abs = 0.0
        reference_sq = 0.0
        difference_sq = 0.0
        if not bitwise_equal:
            expected_flat = expected.reshape(-1)
            actual_flat = actual.reshape(-1)
            chunk_elements = 8 * 1024 * 1024
            for start in range(0, expected_flat.numel(), chunk_elements):
                end = min(start + chunk_elements, expected_flat.numel())
                expected_chunk = expected_flat[start:end].float()
                difference = actual_flat[start:end].float() - expected_chunk
                max_abs = max(max_abs, float(difference.abs().max()))
                reference_sq += float(torch.sum(expected_chunk * expected_chunk))
                difference_sq += float(torch.sum(difference * difference))
        checks.append(
            {
                "bitwise_equal": bitwise_equal,
                "dtype": str(expected.dtype),
                "max_abs": max_abs,
                "name": name,
                "relative_l2": math.sqrt(difference_sq / max(reference_sq, 1.0e-30)),
                "shape": list(expected.shape),
            }
        )
    return checks


def _run_row_grid_abba_case(args: argparse.Namespace) -> None:
    """Compare E16 policy caps by switching its dedicated module global."""

    device, config, counts, x, _, forward, backward = _build_case(
        args.routes,
        args.load,
        args.seed,
        args.mode,
    )
    _, state = forward()
    original_cap = sonic_backward_module._E16_EXACT_ROW_GRID_CAP
    baseline_cap = args.baseline_cap
    candidate_cap = args.candidate_cap
    pair_orders = (
        (baseline_cap, candidate_cap, candidate_cap, baseline_cap),
        (candidate_cap, baseline_cap, baseline_cap, candidate_cap),
    )
    baseline_pair_ms: list[float] = []
    candidate_pair_ms: list[float] = []
    pairwise_speedup_pct: list[float] = []
    order_speedup_pct: list[list[float]] = [[], []]
    last_gradients = None
    try:
        # Compile both launch variants and validate the four outputs once.
        sonic_backward_module._E16_EXACT_ROW_GRID_CAP = baseline_cap
        reference = backward(state)
        torch.cuda.synchronize(device)
        sonic_backward_module._E16_EXACT_ROW_GRID_CAP = candidate_cap
        candidate = backward(state)
        torch.cuda.synchronize(device)
        gradient_checks = _compare_gradients(reference, candidate)
        del reference, candidate

        for _ in range(args.warmup):
            for cap in pair_orders[0]:
                sonic_backward_module._E16_EXACT_ROW_GRID_CAP = cap
                last_gradients = backward(state)
        torch.cuda.synchronize(device)

        for pair_index in range(args.pairs):
            order = pair_orders[pair_index % len(pair_orders)]
            events = []
            for cap in order:
                sonic_backward_module._E16_EXACT_ROW_GRID_CAP = cap
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                last_gradients = backward(state)
                end.record()
                events.append((start, end))
            events[-1][1].synchronize()
            elapsed = [start.elapsed_time(end) for start, end in events]
            baseline_ms = statistics.mean(
                value for cap, value in zip(order, elapsed) if cap == baseline_cap
            )
            candidate_ms = statistics.mean(
                value for cap, value in zip(order, elapsed) if cap == candidate_cap
            )
            baseline_pair_ms.append(baseline_ms)
            candidate_pair_ms.append(candidate_ms)
            speedup = 100.0 * (baseline_ms / candidate_ms - 1.0)
            pairwise_speedup_pct.append(speedup)
            order_speedup_pct[pair_index % len(pair_orders)].append(speedup)
    finally:
        sonic_backward_module._E16_EXACT_ROW_GRID_CAP = original_cap

    assert last_gradients is not None and last_gradients[0].shape == x.shape
    median_speedup = _median(pairwise_speedup_pct)
    cycle_speedup_pct = []
    for index in range(0, len(baseline_pair_ms) - 1, 2):
        baseline_ms = statistics.mean(baseline_pair_ms[index : index + 2])
        candidate_ms = statistics.mean(candidate_pair_ms[index : index + 2])
        cycle_speedup_pct.append(100.0 * (baseline_ms / candidate_ms - 1.0))
    result = {
        "active_experts": sum(count > 0 for count in counts),
        "baseline_cap": baseline_cap,
        "baseline_effective_cap": (
            baseline_cap
            if args.mode == "expert-major"
            and args.routes >= sonic_backward_module._E16_EXACT_ROW_GRID_MIN_ROUTES
            else sonic_backward_module._HOSTLESS_ROW_GRID_CAP
        ),
        "baseline_ms_median": _median(baseline_pair_ms),
        "candidate_cap": candidate_cap,
        "candidate_effective_cap": (
            candidate_cap
            if args.mode == "expert-major"
            and args.routes >= sonic_backward_module._E16_EXACT_ROW_GRID_MIN_ROUTES
            else sonic_backward_module._HOSTLESS_ROW_GRID_CAP
        ),
        "candidate_ms_median": _median(candidate_pair_ms),
        "device": torch.cuda.get_device_name(device),
        "expert_counts": counts,
        "expert_max_to_mean": max(counts) / (args.routes / config.num_experts),
        "gradient_checks": gradient_checks,
        "kind": "hostless-row-grid-abba",
        "load": args.load,
        "mode": args.mode,
        "order_conditioned_speedup_pct_median": {
            "abba": _median(order_speedup_pct[0]),
            "baab": _median(order_speedup_pct[1]),
        },
        "pairs": args.pairs,
        "pair_orders": [list(order) for order in pair_orders],
        "pairwise_speedup_pct_mad": _median(
            [abs(value - median_speedup) for value in pairwise_speedup_pct]
        ),
        "pairwise_speedup_pct_median": median_speedup,
        "pairwise_speedup_pct_p10": _percentile(pairwise_speedup_pct, 0.10),
        "pairwise_speedup_pct_p90": _percentile(pairwise_speedup_pct, 0.90),
        "pairwise_win_rate_pct": 100.0
        * sum(value > 0.0 for value in pairwise_speedup_pct)
        / len(pairwise_speedup_pct),
        "position_neutral_cycles": len(cycle_speedup_pct),
        "position_neutral_speedup_pct_median": _median(cycle_speedup_pct),
        "position_neutral_speedup_pct_p10": _percentile(cycle_speedup_pct, 0.10),
        "position_neutral_speedup_pct_p90": _percentile(cycle_speedup_pct, 0.90),
        "position_neutral_win_rate_pct": 100.0
        * sum(value > 0.0 for value in cycle_speedup_pct)
        / len(cycle_speedup_pct),
        "routes": args.routes,
        "seed": args.seed,
    }
    print(json.dumps(result, sort_keys=True), flush=True)


def _run_row_grid_abba_matrix(args: argparse.Namespace) -> None:
    """Run each ABBA case in its own process while pairing caps in-process."""

    for routes in _QWEN3_MATRIX_ROUTES:
        for load in _QWEN3_MATRIX_LOADS:
            command = [
                sys.executable,
                __file__,
                "--row-grid-abba-case",
                "--routes",
                str(routes),
                "--mode",
                args.mode,
                "--load",
                load,
                "--warmup",
                str(args.warmup),
                "--pairs",
                str(args.pairs),
                "--baseline-cap",
                str(args.baseline_cap),
                "--candidate-cap",
                str(args.candidate_cap),
                "--seed",
                str(args.seed),
            ]
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            if not lines:
                raise RuntimeError(f"ABBA child produced no result: {command!r}")
            result = json.loads(lines[-1])
            print(json.dumps(result, sort_keys=True), flush=True)


def _run_workspace_sequence(
    seed: int,
    instances: int,
    shared_dynamic_workspace: bool,
) -> None:
    """Report growable-workspace memory while T changes in one process."""

    device = torch.device("cuda")
    config = _e16_config()
    generator = torch.Generator(device=device).manual_seed(seed)
    w1 = torch.randn(
        (config.num_experts, 2 * config.intermediate_size, config.hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.02)
    w2 = torch.randn(
        (config.num_experts, config.hidden_size, config.intermediate_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.02)
    weights = prepare_sonic_bf16_weights(w1, w2, config)
    pool = SonicMoEDynamicWorkspacePool() if shared_dynamic_workspace else None
    operators = [
        SonicMoE(
            config,
            weights,
            shared_dynamic_workspace_pool=pool,
        )
        for _ in range(instances)
    ]
    torch.cuda.synchronize(device)
    baseline_allocated = torch.cuda.memory_allocated(device)

    # Grow to a production outlier, then return to ordinary route counts.  The
    # latter two samples make a retained high-water allocation visible instead
    # of conflating it with one cache entry per exact route count.
    for routes in _QWEN3_WORKSPACE_ROUTES:
        torch.cuda.reset_peak_memory_stats(device)
        for operator in operators:
            operator.reserve_dynamic_routes(routes, routes)
        torch.cuda.synchronize(device)
        capacity = (
            next(iter(pool._entries.values())).capacity
            if pool is not None
            else next(iter(operators[-1]._dynamic_route_workspaces.values()))
        )
        assert capacity is not None
        allocated = torch.cuda.memory_allocated(device)
        result = {
            "baseline_allocated_mib": baseline_allocated / (1024 * 1024),
            "current_allocated_mib": allocated / (1024 * 1024),
            "device": torch.cuda.get_device_name(device),
            "dynamic_workspace_cache_entries": (
                len(pool)
                if pool is not None
                else sum(len(operator._dynamic_route_workspaces) for operator in operators)
            ),
            "instances": instances,
            "kind": "same-process-workspace-sequence",
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
            "routes": routes,
            "seed": seed,
            "shared_dynamic_workspace": shared_dynamic_workspace,
            "workspace_allocated_mib": (allocated - baseline_allocated) / (1024 * 1024),
            "workspace_capacity_routes": capacity.routes,
            "workspace_capacity_tokens": capacity.tokens,
        }
        print(json.dumps(result, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=int, default=65536)
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument(
        "--matrix",
        action="store_true",
        help="run the canonical Qwen3 T-by-load matrix in isolated processes",
    )
    run_group.add_argument(
        "--workspace-sequence",
        action="store_true",
        help="measure growable workspace memory for the canonical T sequence",
    )
    run_group.add_argument(
        "--row-grid-abba",
        action="store_true",
        help="compare 1024/2048 hostless row-grid caps over the Qwen3 matrix",
    )
    run_group.add_argument(
        "--row-grid-abba-case",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mode",
        choices=("generic", "retained", "expert-major"),
        default="expert-major",
    )
    parser.add_argument(
        "--load",
        choices=(
            "uniform",
            "balanced",
            "softmax-multinomial",
            "hot1",
            "hot2",
            "hot4",
            "long-tail",
        ),
        default="uniform",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--baseline-cap", type=int, default=1024)
    parser.add_argument("--candidate-cap", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument(
        "--shared-dynamic-workspace",
        action="store_true",
        help="opt in to a dynamic scratch pool shared by all benchmark operators",
    )
    parser.add_argument(
        "--workspace-instances",
        type=int,
        default=1,
        help="number of operators retained by --workspace-sequence",
    )
    args = parser.parse_args()
    if (
        args.routes <= 0
        or args.warmup < 0
        or args.iters <= 0
        or args.pairs <= 0
        or args.baseline_cap <= 0
        or args.candidate_cap <= 0
        or args.workspace_instances <= 0
    ):
        parser.error(
            "routes/iters/pairs/caps/workspace-instances must be positive "
            "and warmup non-negative"
        )
    if args.baseline_cap == args.candidate_cap:
        parser.error("row-grid baseline and candidate caps must differ")
    if (args.row_grid_abba or args.row_grid_abba_case) and args.pairs % 2:
        parser.error("row-grid ABBA/BAAB comparison requires an even --pairs value")
    if args.matrix:
        _run_qwen3_matrix(args)
        return
    if args.workspace_sequence:
        _run_workspace_sequence(
            args.seed,
            args.workspace_instances,
            args.shared_dynamic_workspace,
        )
        return
    if args.row_grid_abba:
        _run_row_grid_abba_matrix(args)
        return
    if args.row_grid_abba_case:
        _run_row_grid_abba_case(args)
        return

    routes = args.routes
    device, config, counts, x, operator, forward, backward = _build_case(
        routes,
        args.load,
        args.seed,
        args.mode,
        shared_dynamic_workspace=args.shared_dynamic_workspace,
    )

    # Untimed compile followed by warmup.  Keep only the latest invocation's
    # state/gradients so allocator reuse matches normal autograd turnover.
    _, state = forward()
    gradients = backward(state)
    torch.cuda.synchronize(device)
    for _ in range(args.warmup):
        _, state = forward()
        gradients = backward(state)
    torch.cuda.synchronize(device)

    forward_ms: list[float] = []
    backward_ms: list[float] = []
    e2e_ms: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.iters):
        start = torch.cuda.Event(enable_timing=True)
        middle = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _, state = forward()
        middle.record()
        gradients = backward(state)
        end.record()
        end.synchronize()
        forward_ms.append(start.elapsed_time(middle))
        backward_ms.append(middle.elapsed_time(end))
        e2e_ms.append(start.elapsed_time(end))
    # Keep the final result live until all event timings and memory readings
    # are complete; this mirrors the autograd consumer lifetime.
    assert gradients[0].shape == x.shape
    result = {
        "active_experts": sum(count > 0 for count in counts),
        "device": torch.cuda.get_device_name(device),
        "experts": config.num_experts,
        "expert_counts": counts,
        "expert_max_to_mean": max(counts) / (routes / config.num_experts),
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "mode": args.mode,
        "load": args.load,
        "routes": routes,
        "seed": args.seed,
        "shared_dynamic_workspace": args.shared_dynamic_workspace,
        "forward_ms_median": _median(forward_ms),
        "backward_ms_median": _median(backward_ms),
        "e2e_ms_median": _median(e2e_ms),
        "forward_ms_min": min(forward_ms),
        "backward_ms_min": min(backward_ms),
        "e2e_ms_min": min(e2e_ms),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
        "exact_workspace_cache_entries": len(operator._workspaces),
        "dynamic_workspace_cache_entries": (
            len(operator._shared_dynamic_workspace_pool)
            if operator._shared_dynamic_workspace_pool is not None
            else len(operator._dynamic_route_workspaces)
        ),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
