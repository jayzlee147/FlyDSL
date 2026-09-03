#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Correctness smoke test and warm-cache benchmark for gfx950 SonicMoE.

Run from the repository root with a ROCm PyTorch environment, for example::

    PYTHONPATH=. python examples/06-sonicMoE.py --check

Weight preparation and first-call JIT compilation are intentionally excluded
from the reported latency.
"""

import argparse
import math

import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.moe.sonic import (
    SonicMoE,
    SonicMoEConfig,
    prepare_sonic_bf16_weights,
    prepare_sonic_fp16_weights,
    prepare_sonic_mxfp4_weights,
    sonic_moe_mxfp4_reference,
    sonic_moe_reference,
)
from kernels.moe.sonic_autotune import SonicMoEAutotuner


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--tile-m", type=int, default=32)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--down-tile-n", type=int, default=None)
    parser.add_argument("--down-tile-k", type=int, default=None)
    parser.add_argument("--weight-dtype", choices=("bf16", "fp16", "mxfp4"), default="bf16")
    parser.add_argument(
        "--activation",
        choices=("swiglu", "geglu", "reglu", "gelu_tanh_approx", "relu", "silu", "relu_sq"),
        default="swiglu",
    )
    parser.add_argument("--autotune", action="store_true", help="search and cache a shape-bucket tile")
    parser.add_argument("--autotune-warmup", type=int, default=3)
    parser.add_argument("--autotune-iters", type=int, default=10)
    parser.add_argument(
        "--autotune-profile-key",
        type=str,
        default=None,
        help="distinguish representative router-distribution profiles in the disk cache",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--check", action="store_true", help="compare once with the FP32 reference")
    return parser.parse_args()


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A ROCm GPU is required")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be >= 0 and --iters must be > 0")

    compute_dtype = "bf16" if args.weight_dtype == "mxfp4" else args.weight_dtype
    torch_dtype = torch.bfloat16 if compute_dtype == "bf16" else torch.float16
    config = SonicMoEConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.experts,
        top_k=args.top_k,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        down_tile_n=args.down_tile_n,
        down_tile_k=args.down_tile_k,
        activation=args.activation,
        compute_dtype=compute_dtype,
    )
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    x = torch.randn(args.tokens, args.hidden_size, device=device, dtype=torch_dtype)
    w1 = torch.randn(
        args.experts,
        args.intermediate_size * (2 if args.activation in ("swiglu", "geglu", "reglu") else 1),
        args.hidden_size,
        device=device,
        dtype=torch_dtype,
    ) / math.sqrt(args.hidden_size)
    w2 = torch.randn(
        args.experts,
        args.hidden_size,
        args.intermediate_size,
        device=device,
        dtype=torch_dtype,
    ) / math.sqrt(args.intermediate_size)
    router_logits = torch.randn(args.tokens, args.experts, device=device, dtype=torch_dtype)

    if args.weight_dtype == "mxfp4":
        prepared = prepare_sonic_mxfp4_weights(w1, w2, config)
    elif args.weight_dtype == "fp16":
        prepared = prepare_sonic_fp16_weights(w1, w2, config)
    else:
        prepared = prepare_sonic_bf16_weights(w1, w2, config)
    if args.autotune:
        op = SonicMoEAutotuner(
            config,
            prepared,
            warmup=args.autotune_warmup,
            rep=args.autotune_iters,
            profile_key=args.autotune_profile_key,
        )
    else:
        op = SonicMoE(config, prepared)
    out = torch.empty((args.tokens, args.hidden_size), device=device, dtype=torch_dtype)

    # Cold call performs JIT compilation. Subsequent calls dispatch cached binaries.
    op(x, router_logits, out=out)
    torch.cuda.synchronize()

    if isinstance(op, SonicMoEAutotuner):
        if op.last_results:
            print("autotune candidates (complete forward):")
            for candidate, elapsed_ms in sorted(op.last_results, key=lambda item: item[1]):
                print(
                    "  "
                    f"({candidate.tile_m},{candidate.tile_n},{candidate.tile_k})/"
                    f"({candidate.tile_m},{candidate.stage2_tile_n},"
                    f"{candidate.stage2_tile_k}): {elapsed_ms * 1000.0:.2f} us"
                )
        else:
            print("autotune: winner loaded from the in-memory/disk shape-profile cache")

    if args.check:
        original_ref = sonic_moe_reference(x, w1, w2, router_logits, config)
        ref = (
            sonic_moe_mxfp4_reference(x, w1, w2, router_logits, config)
            if args.weight_dtype == "mxfp4"
            else original_ref
        )
        out_f32 = out.float()
        ref_f32 = ref.float()
        cosine = torch.nn.functional.cosine_similarity(out_f32.flatten(), ref_f32.flatten(), dim=0).item()
        ref_norm = torch.linalg.vector_norm(ref_f32)
        out_norm = torch.linalg.vector_norm(out_f32)
        if ref_norm.item() == 0.0:
            relative_l2 = out_norm.item()
            norm_ratio = 1.0 if out_norm.item() == 0.0 else float("inf")
            if out_norm.item() == 0.0:
                cosine = 1.0
        else:
            relative_l2 = (torch.linalg.vector_norm(out_f32 - ref_f32) / ref_norm).item()
            norm_ratio = (out_norm / ref_norm).item()
        max_abs = (out_f32 - ref_f32).abs().max().item()
        max_reference = ref_f32.abs().max().item()
        reference_name = "quantized-weight" if args.weight_dtype == "mxfp4" else args.weight_dtype.upper()
        print(
            f"kernel-vs-{reference_name}: cosine={cosine:.7f}, "
            f"relative_l2={relative_l2:.6f}, norm_ratio={norm_ratio:.6f}, "
            f"max_abs={max_abs:.6f}"
        )
        if (
            not all(math.isfinite(value) for value in (cosine, relative_l2, norm_ratio))
            or cosine < 0.999
            or relative_l2 > 0.05
            or not 0.98 <= norm_ratio <= 1.02
            or max_abs > 0.15 * max(max_reference, 1.0e-2)
        ):
            raise AssertionError(
                "SonicMoE kernel correctness check failed: "
                f"cosine={cosine}, relative_l2={relative_l2}, norm_ratio={norm_ratio}"
            )
        if args.weight_dtype == "mxfp4":
            quant_cosine = torch.nn.functional.cosine_similarity(
                ref_f32.flatten(), original_ref.float().flatten(), dim=0
            ).item()
            print(
                "quantized-weight-vs-BF16-model: "
                f"cosine={quant_cosine:.7f} (model-dependent, not a kernel-error gate)"
            )

    for _ in range(args.warmup):
        op(x, router_logits, out=out)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        op(x, router_logits, out=out)
    end.record()
    end.synchronize()

    latency_us = start.elapsed_time(end) * 1000.0 / args.iters
    # GLU stage 1 has two projections; pointwise activations have one. Stage 2 has one.
    projection_count = 3 if args.activation in ("swiglu", "geglu", "reglu") else 2
    useful_flops = 2.0 * projection_count * args.tokens * args.top_k * args.hidden_size * args.intermediate_size
    useful_tflops = useful_flops / (latency_us * 1e-6) / 1e12
    if op.workspace is None:
        raise RuntimeError("SonicMoE workspace was not initialized")
    padded = int(op.workspace.num_valid_ids[0].item())
    padding_ratio = padded / (args.tokens * args.top_k)
    run_config = op.best_config if isinstance(op, SonicMoEAutotuner) else config
    if run_config is None:
        raise RuntimeError("SonicMoE autotuner did not select a configuration")
    props = torch.cuda.get_device_properties(device)
    print(
        f"device={props.name}, arch={get_rocm_arch()}, "
        f"shape=T{args.tokens} H{args.hidden_size} I{args.intermediate_size} "
        f"E{args.experts} K{args.top_k} W={args.weight_dtype} A={args.activation}"
    )
    print(
        f"tiles=({run_config.tile_m},{run_config.tile_n},{run_config.tile_k})/"
        f"({run_config.tile_m},{run_config.stage2_tile_n},{run_config.stage2_tile_k}), "
        f"padded_rows={padded}, padding_ratio={padding_ratio:.3f}"
    )
    print(f"latency={latency_us:.2f} us, useful_throughput={useful_tflops:.2f} TFLOP/s")


if __name__ == "__main__":
    main()
