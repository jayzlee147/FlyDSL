#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
TopK Gating Softmax Operator Test

Fused softmax + top-K expert selection for Mixture-of-Experts gating.
Validates:
  - topk_weights match torch.softmax -> torch.topk reference
  - topk_indices match the reference top-K expert indices
  - token_expert_indices follow the k * num_tokens + token_row convention
  - Optional renormalization (selected weights sum to 1.0)
"""

import os

import pytest

from kernels.moe.topk_gating_softmax_kernel import (
    _compute_topk_gating_layout,
    build_topk_gating_softmax_module,
)
from tests.kernels.benchmark_common import (
    PerfRow,
    bench_gpu_us_torch,
    print_perf_table,
)
from tests.test_common import run_perftest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

DTYPE_FP32 = torch.float32
DTYPE_FP16 = torch.float16
DTYPE_BF16 = torch.bfloat16

WARMUP_ITERS = 10
BENCH_ITERS = 100


def _torch_dtype(dtype_str):
    return {"f32": DTYPE_FP32, "f16": DTYPE_FP16, "bf16": DTYPE_BF16}[dtype_str]


def run_test(num_tokens, num_experts, topk, dtype_str, renormalize=True):
    print(
        f"\nTesting TopK Gating Softmax: "
        f"tokens={num_tokens}, experts={num_experts}, topk={topk}, "
        f"dtype={dtype_str}, renorm={renormalize}"
    )

    try:
        launch_fn = build_topk_gating_softmax_module(
            num_experts=num_experts,
            topk=topk,
            dtype_str=dtype_str,
            renormalize=renormalize,
        )
    except Exception as e:
        print(f"Compilation Failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None

    torch.manual_seed(42)
    torch_dtype = _torch_dtype(dtype_str)
    gating_fp32 = (torch.rand((num_tokens, num_experts), device="cuda", dtype=DTYPE_FP32) * 4.0) - 2.0
    # An all-equal row makes the tie-break observable.  The kernel contract is
    # deterministic: lower expert indices win ties, in ascending order.
    gating_fp32[0].zero_()
    # Quantize to the kernel's input dtype FIRST so the reference sees the
    # exact bytes the kernel sees. Otherwise the K-th expert can flip at
    # bf16/f16 precision boundaries (both vLLM and FlyDSL pick a different
    # near-tie expert than fp32-softmax does).
    gating_dev = gating_fp32.to(torch_dtype).contiguous()
    gating_for_ref = gating_dev.to(DTYPE_FP32)

    # --- PyTorch reference ---
    probs_ref = torch.softmax(gating_for_ref, dim=1)
    ref_weights, ref_indices = torch.topk(probs_ref, topk, dim=1)
    if renormalize:
        ref_weights = ref_weights / ref_weights.sum(dim=1, keepdim=True).clamp(min=1e-20)
    ref_weights = ref_weights.to(DTYPE_FP32)
    ref_indices = ref_indices.to(torch.int32)

    # token_expert_indices reference: k * num_tokens + row
    ref_tei = torch.zeros_like(ref_indices)
    for k in range(topk):
        ref_tei[:, k] = k * num_tokens + torch.arange(num_tokens, device="cuda", dtype=torch.int32)

    # --- Device tensors ---
    topk_weights_dev = torch.empty((num_tokens, topk), device="cuda", dtype=DTYPE_FP32)
    topk_indices_dev = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)
    token_expert_indices_dev = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)

    stream = torch.cuda.current_stream()

    def kernel_launch():
        launch_fn(
            gating_dev,
            topk_weights_dev,
            topk_indices_dev,
            token_expert_indices_dev,
            num_tokens,
            stream=stream,
        )

    kernel_launch()
    torch.cuda.synchronize()

    _, avg_us = run_perftest(
        lambda: (kernel_launch(), torch.cuda.synchronize()),
        num_iters=BENCH_ITERS,
        num_warmup=WARMUP_ITERS,
    )
    torch.cuda.synchronize()
    flydsl_gpu_us = None
    if os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1":
        flydsl_gpu_us = bench_gpu_us_torch(kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)

    avg_ms = avg_us / 1000.0
    print(f"Kernel avg time: {avg_ms:.4f} ms (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    if flydsl_gpu_us is not None:
        print(f"[Perf] FlyDSL topk_gating_softmax gpu: {flydsl_gpu_us:.1f} us")

    # --- Verification ---
    atol_weight = 2e-2 if dtype_str in ("bf16", "f16") else 1e-5
    passed = True

    # 1. Check topk_indices: every kernel-selected expert must be a valid
    #    top-K choice. We compare against the K-th largest reference
    #    probability rather than torch.topk's specific index set, since
    #    torch.topk and the kernel may legitimately disagree on which
    #    experts to take when several share the boundary probability
    #    (a common bf16/f16 quantization artifact).
    got_indices = topk_indices_dev.cpu()
    exp_indices = ref_indices.cpu()

    probs_ref_cpu = probs_ref.cpu()
    # K-th largest reference probability per row (the top-K threshold).
    kth_threshold, _ = torch.topk(probs_ref_cpu, topk, dim=1)
    kth_threshold = kth_threshold[:, -1]
    # Probability tolerance: bf16/f16 representable gap between adjacent values
    # near the typical softmax magnitude is ~1e-3; f32 is much tighter.
    prob_tol = 1e-3 if dtype_str in ("bf16", "f16") else 1e-6

    indices_match = 0  # strict set equality with torch.topk (informational)
    indices_valid = 0  # every selected expert is at-or-above the K-th threshold
    for row in range(num_tokens):
        got_list = got_indices[row].tolist()
        if set(got_list) == set(exp_indices[row].tolist()):
            indices_match += 1
        if len(set(got_list)) == topk:
            row_thr = kth_threshold[row].item() - prob_tol
            if all(probs_ref_cpu[row, idx].item() >= row_thr for idx in got_list):
                indices_valid += 1
    indices_pct = 100.0 * indices_match / num_tokens
    valid_pct = 100.0 * indices_valid / num_tokens
    print(
        f"  Indices match torch.topk: {indices_match}/{num_tokens} rows "
        f"({indices_pct:.1f}%; ties at the K-th boundary may diverge)"
    )
    print(f"  Indices valid (>= K-th prob): {indices_valid}/{num_tokens} rows " f"({valid_pct:.1f}%)")
    if valid_pct < 100.0:
        print("  FAILED: kernel selected experts below the top-K threshold")
        passed = False

    expected_tie_indices = torch.arange(topk, dtype=torch.int32)
    tie_break_match = torch.equal(got_indices[0], expected_tie_indices)
    print(f"  Lower-index tie break correct: {tie_break_match}")
    if not tie_break_match:
        print(f"  FAILED: expected tie indices {expected_tie_indices.tolist()}, got {got_indices[0].tolist()}")
        passed = False

    # 2. Check topk_weights: for matching rows, compare sorted weights
    got_weights = topk_weights_dev.cpu().to(DTYPE_FP32)
    exp_weights = ref_weights.cpu().to(DTYPE_FP32)

    got_sorted, _ = got_weights.sort(dim=1, descending=True)
    exp_sorted, _ = exp_weights.sort(dim=1, descending=True)
    weight_err = (got_sorted - exp_sorted).abs().max().item()
    print(f"  Max weight error (sorted): {weight_err:.2e} (atol={atol_weight})")
    if weight_err > atol_weight:
        print("  FAILED: weight error too large")
        passed = False

    # 3. Check token_expert_indices
    got_tei = token_expert_indices_dev.cpu()
    tei_match = (got_tei == ref_tei.cpu()).all().item()
    print(f"  token_expert_indices correct: {tei_match}")
    if not tei_match:
        print("  FAILED: token_expert_indices mismatch")
        print(f"    Expected first row: {ref_tei[0].tolist()}")
        print(f"    Got first row:      {got_tei[0].tolist()}")
        passed = False

    # 4. Check renormalization (weights sum to ~1.0)
    if renormalize:
        row_sums = got_weights.sum(dim=1)
        max_sum_err = (row_sums - 1.0).abs().max().item()
        print(f"  Max renorm sum error: {max_sum_err:.2e}")
        if max_sum_err > atol_weight:
            print("  FAILED: renormalized weights don't sum to 1")
            passed = False

    if passed:
        print("  PASSED")
    else:
        print("  FAILED")

    return passed, flydsl_gpu_us


def test_e896_layout():
    layout = _compute_topk_gating_layout(896, 16, "bf16")
    assert layout["VPT"] == 14
    assert layout["THREADS_PER_TOKEN"] == 64
    assert layout["TOKENS_PER_BLOCK"] == 4
    assert layout["ELEMS_PER_ATOM"] == 2
    assert layout["ATOMS_PER_THREAD"] == 7


def test_renormalized_topk_keeps_masked_negative_infinity_indices_valid():
    """A row with fewer than K finite logits must still emit K unique IDs."""

    num_experts = 8
    topk = 4
    launch = build_topk_gating_softmax_module(
        num_experts=num_experts,
        topk=topk,
        dtype_str="bf16",
        renormalize=True,
    )
    logits = torch.full((1, num_experts), float("-inf"), device="cuda", dtype=torch.bfloat16)
    logits[0, 5] = 2.0
    logits[0, 2] = 1.0
    weights = torch.empty((1, topk), device="cuda", dtype=torch.float32)
    indices = torch.empty((1, topk), device="cuda", dtype=torch.int32)
    token_expert_indices = torch.empty_like(indices)

    launch(
        logits,
        weights,
        indices,
        token_expert_indices,
        1,
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    expected_indices = torch.tensor([[5, 2, 0, 1]], device="cuda", dtype=torch.int32)
    expected_weights = torch.softmax(
        torch.tensor([[2.0, 1.0, float("-inf"), float("-inf")]], device="cuda"),
        dim=-1,
    )
    assert torch.equal(indices, expected_indices)
    torch.testing.assert_close(weights, expected_weights, rtol=1e-5, atol=1e-6)
    assert torch.equal(
        token_expert_indices,
        torch.arange(topk, device="cuda", dtype=torch.int32).unsqueeze(0),
    )


def test_all():
    print("=" * 80)
    print("Running TopK Gating Softmax Tests")
    print("=" * 80)

    shapes_env = os.environ.get("ROCDSL_TOPK_GATING_SHAPES", "").strip()
    if shapes_env:
        configs = []
        for part in shapes_env.split(";"):
            p = part.strip()
            if not p:
                continue
            toks, exps, k, dt = [x.strip() for x in p.split(",")]
            configs.append((int(toks), int(exps), int(k), dt, True))
    else:
        configs = [
            (1024, 128, 6, "bf16", True),
            (512, 64, 2, "bf16", True),
            (256, 8, 2, "f32", True),
            (128, 128, 6, "f16", True),
            # Kimi-K3 production expert count.  Cover all input dtypes and a
            # non-multiple-of-TOKENS_PER_BLOCK tail (65 % 4 == 1).
            (65, 896, 16, "bf16", True),
            (65, 896, 16, "f16", True),
            (65, 896, 16, "f32", True),
            # Non-renormalized routing must retain full E-way softmax weights.
            (65, 896, 16, "bf16", False),
        ]

    do_compare = os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1"
    perf_rows = []

    failures = 0
    for num_tokens, num_experts, topk, dtype_str, renormalize in configs:
        ok, flydsl_gpu_us = run_test(
            num_tokens,
            num_experts,
            topk,
            dtype_str,
            renormalize=renormalize,
        )
        if not ok:
            failures += 1

        if do_compare:
            perf_rows.append(
                PerfRow(
                    op="topk_gating_softmax",
                    shape=f"{num_tokens}x{num_experts}xk{topk}-renorm{int(renormalize)}",
                    dtype=dtype_str,
                    flydsl_gpu_us=flydsl_gpu_us,
                    aiter_gpu_us=None,
                )
            )

    print("\n" + "=" * 80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("=" * 80)
    if do_compare and perf_rows:
        print_perf_table(perf_rows)
    if failures != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    test_all()
