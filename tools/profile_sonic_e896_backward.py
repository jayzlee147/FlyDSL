#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

r"""Profile the production E896 retained-state backward on gfx950.

Setup, weight preparation, compilation, and warmup run while rocprofiler
collection is paused.  Only the requested steady-state calls are enclosed by
``roctxProfilerResume/Pause``::

  PYTHONPATH=. rocprofv3 --kernel-trace --selected-regions -- \
    python tools/profile_sonic_e896_backward.py \
      --case balanced --scope backward --iters 1 \
      --output /tmp/e896-backward-manifest.json

The fixed contract is T4096/H3584/I512/E896/K16 BF16 SwiGLU with native
interleaved W1 and a retained forward state.  ``backward`` reuses one retained
state; ``full-step`` includes a fresh training forward and its backward.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--case", choices=("balanced", "hot16"), required=True)
    parser.add_argument("--scope", choices=("backward", "full-step"), default="backward")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup <= 0 or args.iters <= 0:
        parser.error("--warmup and --iters must be positive")
    return args


def _run(args: argparse.Namespace) -> dict[str, Any]:
    # Keep heavyweight imports here so --help also works without PyTorch/ROCm.
    import torch

    import kernels.moe.sonic_backward as backward_module
    from kernels.moe.sonic import SonicMoE, prepare_sonic_bf16_weights
    from tools.accept_sonic_e896_backward import (
        EXPERTS,
        GRADIENT_NAMES,
        HIDDEN,
        INTERMEDIATE,
        TOKENS,
        TOPK,
        _adapter_config,
        _interleave_glu_rows,
        _routing,
    )
    from tools.profile_sonic_forward import _roctx_control

    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")
    properties = torch.cuda.get_device_properties(0)
    if "gfx950" not in properties.gcnArchName:
        raise RuntimeError(f"gfx950 is required, found {properties.gcnArchName}")

    config = _adapter_config()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.empty(
        (TOKENS, HIDDEN), device=device, dtype=torch.bfloat16
    ).uniform_(-0.02, 0.02, generator=generator)
    separated_w1 = torch.empty(
        (EXPERTS, 2 * INTERMEDIATE, HIDDEN),
        device=device,
        dtype=torch.bfloat16,
    ).uniform_(-0.02, 0.02, generator=generator)
    interleaved_w1 = _interleave_glu_rows(separated_w1)
    w2 = torch.empty(
        (EXPERTS, HIDDEN, INTERMEDIATE),
        device=device,
        dtype=torch.bfloat16,
    ).uniform_(-0.02, 0.02, generator=generator)
    dout = torch.empty_like(x).uniform_(-0.02, 0.02, generator=generator)
    ids, scores, routing = _routing(args.case, generator)

    weights = prepare_sonic_bf16_weights(separated_w1, w2, config)
    del separated_w1
    gc.collect()
    torch.cuda.empty_cache()
    op = SonicMoE(config, weights)
    forward_output, fixed_state = op.forward_topk_training(
        x,
        ids,
        scores,
        interleaved_w1=True,
    )
    del forward_output
    torch.cuda.synchronize()

    def backward() -> tuple[torch.Tensor, ...]:
        return backward_module.sonic_moe_backward(
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

    def full_step() -> tuple[torch.Tensor, ...]:
        output, state = op.forward_topk_training(
            x,
            ids,
            scores,
            interleaved_w1=True,
        )
        gradients = backward_module.sonic_moe_backward(
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

    selected: Callable[[], tuple[torch.Tensor, ...]] = {
        "backward": backward,
        "full-step": full_step,
    }[args.scope]

    gradient_shapes: list[list[int]] | None = None
    gradient_dtypes: list[str] | None = None
    for _ in range(args.warmup):
        gradients = selected()
        if gradient_shapes is None:
            if len(gradients) != len(GRADIENT_NAMES):
                raise RuntimeError(
                    f"expected {len(GRADIENT_NAMES)} gradients, got {len(gradients)}"
                )
            gradient_shapes = [list(gradient.shape) for gradient in gradients]
            gradient_dtypes = [str(gradient.dtype) for gradient in gradients]
        del gradients
    torch.cuda.synchronize()

    roctx_library, roctx = _roctx_control()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    rc = roctx.roctxProfilerResume(0)
    if rc:
        raise RuntimeError(f"roctxProfilerResume failed: {rc}")
    launch_error: BaseException | None = None
    try:
        begin.record()
        for _ in range(args.iters):
            gradients = selected()
            del gradients
        end.record()
        end.synchronize()
    except BaseException as error:
        launch_error = error
    finally:
        rc = roctx.roctxProfilerPause(0)
    if rc:
        raise RuntimeError(f"roctxProfilerPause failed: {rc}")
    if launch_error is not None:
        raise launch_error

    return {
        "schema": "flydsl.sonic_e896_backward_profile.v1",
        "contract": {
            "shape": {
                "T": TOKENS,
                "H": HIDDEN,
                "I": INTERMEDIATE,
                "E": EXPERTS,
                "K": TOPK,
            },
            "dtype": "bf16",
            "activation": "swiglu",
            "bias": False,
            "interleaved_w1": True,
            "forward_state": "retained",
        },
        "case": args.case,
        "routing": routing,
        "selected_region": {
            "scope": args.scope,
            "iterations": args.iters,
            "device_elapsed_ms": float(begin.elapsed_time(end)),
            "mean_device_ms": float(begin.elapsed_time(end)) / args.iters,
            "roctx_library": roctx_library,
        },
        "warmup_calls": args.warmup,
        "gradients": {
            "names": list(GRADIENT_NAMES),
            "shapes": gradient_shapes,
            "dtypes": gradient_dtypes,
        },
        "environment": {
            "command": [sys.executable, *sys.argv],
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "device": {
                "name": properties.name,
                "arch": properties.gcnArchName,
                "uuid": str(properties.uuid),
                "pci_bus_id": properties.pci_bus_id,
                "multiprocessor_count": properties.multi_processor_count,
            },
            "visibility": {
                name: os.environ.get(name)
                for name in (
                    "ROCR_VISIBLE_DEVICES",
                    "HIP_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES",
                )
            },
            "git": {
                "commit": _git("rev-parse", "HEAD"),
                "branch": _git("branch", "--show-current"),
                "dirty": bool(_git("status", "--porcelain")),
            },
        },
    }


def main() -> None:
    args = _parse_args()
    result = _run(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
