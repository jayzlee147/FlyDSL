# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Benchmark dynamic Qwen3-30B-A3B E16 flat-route training on gfx950.

The three modes isolate the production dispatch differences:

* ``generic``: no retained state in backward (legacy dynamic-R fallback);
* ``retained``: retain forward sorter metadata and use segmented dX;
* ``expert-major``: consume expert offsets, skip counting sort, use identity
  Stage-2 stores, and write dX directly.

Run modes in separate processes for allocator and JIT isolation.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from kernels.moe.sonic import SonicMoE, SonicMoEConfig, prepare_sonic_bf16_weights
from kernels.moe.sonic_backward import sonic_moe_backward_routes


def _counts(routes: int, experts: int, load: str) -> list[int]:
    if load == "uniform":
        quotient, remainder = divmod(routes, experts)
        return [quotient + (expert < remainder) for expert in range(experts)]
    if load == "hot1":
        return [routes, *([0] * (experts - 1))]
    if load == "hot2":
        first = (routes * 4 + 4) // 5
        return [first, routes - first, *([0] * (experts - 2))]
    if load == "long-tail":
        first = (routes + 1) // 2
        quotient, remainder = divmod(routes - first, experts - 1)
        return [first, *[quotient + (expert < remainder) for expert in range(experts - 1)]]
    raise ValueError(f"unknown load {load!r}")


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=int, default=65536)
    parser.add_argument(
        "--mode",
        choices=("generic", "retained", "expert-major"),
        default="expert-major",
    )
    parser.add_argument(
        "--load",
        choices=("uniform", "hot1", "hot2", "long-tail"),
        default="uniform",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    if args.routes <= 0 or args.warmup < 0 or args.iters <= 0:
        parser.error("routes/iters must be positive and warmup non-negative")

    device = torch.device("cuda")
    routes = args.routes
    experts, hidden_size, intermediate_size = 16, 2048, 768
    counts = _counts(routes, experts, args.load)
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
    config = SonicMoEConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=experts,
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
        "mode": args.mode,
        "load": args.load,
        "routes": routes,
        "forward_ms_median": _median(forward_ms),
        "backward_ms_median": _median(backward_ms),
        "e2e_ms_median": _median(e2e_ms),
        "forward_ms_min": min(forward_ms),
        "backward_ms_min": min(backward_ms),
        "e2e_ms_min": min(e2e_ms),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "exact_workspace_cache_entries": len(operator._workspaces),
        "dynamic_workspace_cache_entries": len(operator._dynamic_route_workspaces),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
