#!/usr/bin/env python3
"""ABBA the device-guarded E16 direct-dW1 tile pair on gfx950."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from kernels.moe.sonic import SonicMoE, SonicMoEConfig, prepare_sonic_bf16_weights
from kernels.moe import sonic_backward as backward_module
from kernels.moe.sonic_backward import sonic_moe_backward_routes


def counts_for(routes: int, experts: int, load: str) -> list[int]:
    if load == "uniform":
        quotient, remainder = divmod(routes, experts)
        return [quotient + int(expert < remainder) for expert in range(experts)]
    if load == "hot1":
        return [routes, *([0] * (experts - 1))]
    if load == "hot2":
        first = (4 * routes + 4) // 5
        return [first, routes - first, *([0] * (experts - 2))]
    if load == "long-tail":
        first = (routes + 1) // 2
        quotient, remainder = divmod(routes - first, experts - 1)
        return [
            first,
            *[
                quotient + int(expert < remainder)
                for expert in range(experts - 1)
            ],
        ]
    raise ValueError(load)


def summary(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "median_ms": float(statistics.median(samples)),
        "mean_ms": float(statistics.fmean(samples)),
        "min_ms": float(ordered[0]),
        "p10_ms": float(ordered[max(0, int(0.1 * (len(ordered) - 1)))]),
        "p90_ms": float(ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))]),
        "max_ms": float(ordered[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=int, required=True)
    parser.add_argument(
        "--load",
        choices=("uniform", "hot1", "hot2", "long-tail"),
        required=True,
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--profile-once", action="store_true")
    parser.add_argument(
        "--profile-variant",
        choices=("baseline", "candidate"),
        default="candidate",
        help="variant enclosed by the rocprofv3 selected region",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    routes = args.routes
    experts, hidden_size, intermediate_size = 16, 2048, 768
    counts = counts_for(routes, experts, args.load)
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    generator = torch.Generator(device=device).manual_seed(20260917)

    def randn(shape: tuple[int, ...], scale: float) -> torch.Tensor:
        return torch.randn(
            shape,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).mul_(scale)

    x = randn((routes, hidden_size), 0.2)
    w1 = randn((experts, 2 * intermediate_size, hidden_size), 0.02)
    w2 = randn((experts, hidden_size, intermediate_size), 0.02)
    grad_output = randn((routes, hidden_size), 0.2)
    token_indices = torch.arange(routes, dtype=torch.int32, device=device)
    expert_indices = torch.repeat_interleave(
        torch.arange(experts, dtype=torch.int32, device=device),
        torch.tensor(counts, dtype=torch.int64, device=device),
    ).contiguous()
    expert_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    route_weights = torch.linspace(
        0.25,
        1.0,
        routes,
        dtype=torch.float32,
        device=device,
    )
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
    _, state = operator.forward_routes_training(
        x,
        token_indices,
        expert_indices,
        route_weights,
        expert_offsets=expert_offsets,
        token_indices_identity=True,
    )

    original_policy = backward_module._use_e16_dw1_dual_profile

    def backward(candidate: bool):
        backward_module._use_e16_dw1_dual_profile = (
            original_policy if candidate else lambda **_kwargs: False
        )
        return sonic_moe_backward_routes(
            x,
            w1,
            w2,
            token_indices,
            expert_indices,
            route_weights,
            grad_output,
            config,
            forward_state=state,
            token_indices_sorted=True,
        )

    baseline = backward(False)
    candidate = backward(True)
    torch.cuda.synchronize(device)
    correctness: dict[str, dict[str, float | bool]] = {}
    for name, actual, expected in zip(
        ("dx", "dw1", "dw2", "droute"),
        candidate,
        baseline,
    ):
        difference = actual.float() - expected.float()
        reference_norm = max(float(torch.linalg.vector_norm(expected.float())), 1.0e-12)
        metrics = {
            "exact": bool(torch.equal(actual, expected)),
            "finite": bool(torch.isfinite(actual).all()),
            "max_abs": float(difference.abs().max()),
            "relative_l2": float(torch.linalg.vector_norm(difference)) / reference_norm,
        }
        if not metrics["finite"]:
            raise AssertionError(f"{name} contains non-finite values")
        if name != "dw1" and not metrics["exact"]:
            raise AssertionError(f"{name} unexpectedly changed")
        if name == "dw1" and (
            metrics["max_abs"] > 0.02 or metrics["relative_l2"] > 2.0e-4
        ):
            raise AssertionError(f"{name} mismatch: {metrics}")
        correctness[name] = metrics
    del baseline, candidate

    if args.profile_once:
        from tools.profile_sonic_forward import _roctx_control

        candidate_enabled = args.profile_variant == "candidate"
        _, roctx = _roctx_control()
        torch.cuda.synchronize(device)
        rc = roctx.roctxProfilerResume(0)
        if rc:
            raise RuntimeError(f"roctxProfilerResume failed: {rc}")
        profile_error = None
        range_pushed = False
        try:
            torch.cuda.nvtx.range_push(f"dw1_{args.profile_variant}")
            range_pushed = True
            result = backward(candidate_enabled)
            torch.cuda.nvtx.range_pop()
            range_pushed = False
            del result
            torch.cuda.synchronize(device)
        except BaseException as error:
            profile_error = error
        finally:
            if range_pushed:
                try:
                    torch.cuda.nvtx.range_pop()
                except BaseException as error:
                    if profile_error is None:
                        profile_error = error
            backward_module._use_e16_dw1_dual_profile = original_policy
            rc = roctx.roctxProfilerPause(0)
        if rc:
            raise RuntimeError(f"roctxProfilerPause failed: {rc}")
        if profile_error is not None:
            raise profile_error
        return

    for iteration in range(args.warmup):
        result = backward(bool(iteration & 1))
        del result
    torch.cuda.synchronize(device)

    samples = {"baseline": [], "candidate": []}
    for pair in range(args.pairs):
        order = (
            (False, True, True, False)
            if pair % 2 == 0
            else (True, False, False, True)
        )
        for candidate_enabled in order:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            result = backward(candidate_enabled)
            end.record()
            end.synchronize()
            samples["candidate" if candidate_enabled else "baseline"].append(
                begin.elapsed_time(end)
            )
            del result
    backward_module._use_e16_dw1_dual_profile = original_policy
    baseline_summary = summary(samples["baseline"])
    candidate_summary = summary(samples["candidate"])
    print(
        json.dumps(
            {
                "routes": routes,
                "load": args.load,
                "counts": counts,
                "active_experts": sum(count > 0 for count in counts),
                "correctness": correctness,
                "baseline": baseline_summary,
                "candidate": candidate_summary,
                "speedup": baseline_summary["median_ms"]
                / candidate_summary["median_ms"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
