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
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys

import torch

from kernels.moe.sonic import SonicMoE, SonicMoEConfig, prepare_sonic_bf16_weights
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


def _run_workspace_sequence(seed: int) -> None:
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
    operator = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    torch.cuda.synchronize(device)
    baseline_allocated = torch.cuda.memory_allocated(device)

    # Grow to a production outlier, then return to ordinary route counts.  The
    # latter two samples make a retained high-water allocation visible instead
    # of conflating it with one cache entry per exact route count.
    for routes in _QWEN3_WORKSPACE_ROUTES:
        torch.cuda.reset_peak_memory_stats(device)
        operator.reserve_dynamic_routes(routes, routes)
        torch.cuda.synchronize(device)
        capacity = next(iter(operator._dynamic_route_workspaces.values()))
        allocated = torch.cuda.memory_allocated(device)
        result = {
            "baseline_allocated_mib": baseline_allocated / (1024 * 1024),
            "current_allocated_mib": allocated / (1024 * 1024),
            "device": torch.cuda.get_device_name(device),
            "dynamic_workspace_cache_entries": len(operator._dynamic_route_workspaces),
            "kind": "same-process-workspace-sequence",
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
            "routes": routes,
            "seed": seed,
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
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    if args.routes <= 0 or args.warmup < 0 or args.iters <= 0:
        parser.error("routes/iters must be positive and warmup non-negative")
    if args.matrix:
        _run_qwen3_matrix(args)
        return
    if args.workspace_sequence:
        _run_workspace_sequence(args.seed)
        return

    device = torch.device("cuda")
    routes = args.routes
    experts, hidden_size, intermediate_size = 16, 2048, 768
    counts = _counts(routes, experts, args.load, args.seed)
    offsets_host = [0]
    for count in counts:
        offsets_host.append(offsets_host[-1] + count)
    generator = torch.Generator(device=device).manual_seed(args.seed)
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
    config = _e16_config()
    operator = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))

    def forward():
        kwargs = {}
        if args.mode == "expert-major":
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
            forward_state=(None if args.mode == "generic" else state),
            token_indices_sorted=True,
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
        "experts": experts,
        "expert_counts": counts,
        "expert_max_to_mean": max(counts) / (routes / experts),
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "mode": args.mode,
        "load": args.load,
        "routes": routes,
        "seed": args.seed,
        "forward_ms_median": _median(forward_ms),
        "backward_ms_median": _median(backward_ms),
        "e2e_ms_median": _median(e2e_ms),
        "forward_ms_min": min(forward_ms),
        "backward_ms_min": min(backward_ms),
        "e2e_ms_min": min(e2e_ms),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
        "exact_workspace_cache_entries": len(operator._workspaces),
        "dynamic_workspace_cache_entries": len(operator._dynamic_route_workspaces),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
