# Pre-built kernel library guide

This guide covers the available FlyDSL kernels — normalization, softmax, GEMM, attention, and MoE — along with their configuration options, supported data types, pipeline designs, and shared utilities.

## Quick reference

| Kernel | Builder function | API style | Dtypes | Key feature |
|---|---|---|---|---|
| **LayerNorm** | `build_layernorm_module(N, dtype)` | Layout API (`@flyc.kernel`) | f32, f16, bf16 | Two-pass vectorized normalization |
| **RMSNorm** | `build_rmsnorm_module(N, dtype)` | Layout API (`@flyc.kernel`) | f32, f16, bf16; optional fp32 weight | LDS-cached 3-pass pipeline |
| **Softmax** | `build_softmax_module(M, N, dtype)` | Layout API (`@flyc.kernel`) | f32, f16, bf16 | Online softmax, adaptive block size |
| **Softmax backward** | `build_softmax_bwd_module(N, dtype)` | Layout API (`@flyc.kernel`) | f32, f16, bf16 | fp32 dot reduction, native-dtype register buffering |
| **GEMM** | `compile_preshuffle_gemm(...)` | `@flyc.kernel` | fp8, int8, fp16, bf16 | Preshuffle B, ping-pong LDS, MFMA 16x16 |
| **FlashAttention** | `build_flash_attn_func_module(...)` | `@flyc.kernel` | bf16, f16 (any arch); fp8 e4m3fn (gfx950, D=128, dense) | Dual-wave SWP fwd, GQA/MQA, causal, descale ABI |
| **SonicMoE forward** | `SonicMoE(config, weights)` | Host-composed FlyDSL | BF16/FP16 activation and dense weight; MXFP4 weight with BF16 activation | Routing/top-k + sort, fused activation, weighted down scatter |
| **SonicMoE backward** | `sonic_moe_backward(...)`, `sonic_moe_backward_routes(...)` | Host-composed FlyDSL | BF16/FP16 activation, dense weight, optional bias | Fixed-K and flat ragged-route gradients for all seven Sonic activations |

All kernels use the `@flyc.kernel`/`@flyc.jit` API from `flydsl.compiler` and `flydsl.expr` (`python/flydsl/`).

---

## 1. Normalization kernels

### 1.1 LayerNorm (`kernels/norm/layernorm_kernel.py`)

Computes `LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta` for each row.

**Builder:**
```python
from kernels.norm.layernorm_kernel import build_layernorm_module

executor = build_layernorm_module(N=8192, dtype_str="bf16")
```

**Configuration constants:**
| Constant | Value | Description |
|---|---|---|
| `BLOCK_THREADS` | 256 | Threads per block |
| `WARP_SIZE` | 64 | AMD wavefront size |
| `VEC_WIDTH` | 8 | Vector load/store width |
| `VEC_ALIGN` | 16 | Alignment for vector ops (bytes) |
| `EPS` | 1e-5 | Numerical stability epsilon |
| `USE_NONTEMPORAL` | True | Non-temporal stores for output |

**Algorithm:**
- **Two-pass normalization**: Pass 1 computes mean and variance, Pass 2 applies affine transform
- **Fast path**: When `N == BLOCK_THREADS * VEC_WIDTH * 4` (for example, N=8192), uses fully register-resident computation with no scalar tail
- **Generic path**: Handles arbitrary N with vector body + scalar tail
- **bf16 handling**: Software round-to-nearest-even (RNE) pack on gfx942; hardware `cvt_pk_bf16_f32` on gfx950+
- **Warp reduction**: XOR-shuffle-based intra-wave reduction (shifts: 32, 16, 8, 4, 2, 1), then LDS-based cross-wave synchronization

**Kernel signature** (using `@flyc.kernel` API):
```
GPU_MODULE_NAME = "layernorm_module"

@kernel
layernorm_kernel(self, Input, Gamma, Beta, Output, m_in)

@jit
__call__(self, Input, Gamma, Beta, Output, m_in)
```

### 1.2 RMSNorm (`kernels/norm/rmsnorm_kernel.py`)

Computes `RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma`.

**Builder:**
```python
from kernels.norm.rmsnorm_kernel import build_rmsnorm_module

executor = build_rmsnorm_module(N=8192, dtype_str="bf16", store_rstd=False)
```

`build_rmsnorm_module(N, dtype_str, store_rstd=False, eps=EPS,
BLOCK_THREADS=BLOCK_THREADS, weight_dtype_str=None)` optionally writes the
per-row reciprocal std (`rstd`) for use by the backward pass.
`weight_dtype_str` defaults to `dtype_str`; FP16/BF16 activations additionally
support FP32 weights.

**Backward:** `build_rmsnorm_bwd_module(N, dtype_str,
weight_dtype_str=None)` builds the fused RMSNorm backward kernel (grid `(M,)`,
one block per row). Kernel signature
`rmsnorm_bwd_kernel(Input, Gamma, DY, Rstd, DX, DWeight)`: reads the forward
`Rstd`, writes `DX` (input grad), and atomic-adds into `DWeight` (fp32 weight
grad). The forward bakes `eps` into `Rstd`, so the backward does not need it.
The public plain and fused-add training wrappers return `dweight` in the
original weight dtype.

**Configuration constants:** Same as LayerNorm (BLOCK_THREADS=256, VEC_WIDTH=8, etc.)

**Algorithm (3-pass with LDS caching):**
1. **Pass 0**: Global → LDS row cache (one-pass global read, vectorized)
2. **Pass 1**: Sum-of-squares computation from LDS row cache
3. **Pass 2**: Normalize + gamma multiply + store with software pipeline for Gamma prefetch

**Kernel signature:**
```
GPU_MODULE_NAME = "rmsnorm_module"

@kernel
rmsnorm_kernel(self, Input, Gamma, Output, m_in)
```

---

## 2. Softmax kernel

### 2.1 Softmax (`kernels/norm/softmax_kernel.py`)

Computes row-wise softmax: `softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))`.

**Builder:**
```python
from kernels.norm.softmax_kernel import build_softmax_module

executor = build_softmax_module(M=32768, N=8192, dtype_str="bf16")
```

**Configuration:**
| Parameter | Value | Description |
|---|---|---|
| `BLOCK_SIZE` | `min(256, next_power_of_2(N))`, min 32 | Adaptive block size |
| `VEC_WIDTH` | 8 | Vector load/store width |
| `WARP_SIZE` | 64 | AMD wavefront size |

**Algorithm (6 stages):**
1. **Load data**: Vectorized global loads into register buffer with validity masks
2. **Local max**: Per-thread vector reduction (`maxnumf`)
3. **Global max**: Block-wide shuffle reduction (intra-wave XOR → wave0 finalize via LDS)
4. **Local exp + sum**: `exp2(x * log2(e))` approximation, accumulate partial sums
5. **Global sum**: Block-wide reduction for sum
6. **Normalize + store**: Divide by sum, convert to output dtype, vectorized store

**Kernel signature:**
```
GPU_MODULE_NAME = f"softmax_{dtype_str}"

@kernel
softmax_kernel(self, A, C, m_in)
```

### 2.2 Softmax backward (`kernels/norm/softmax_bwd_kernel.py`)

Computes the row-wise Softmax gradient: `dx = y * (dy - sum(dy * y))`, with the
dot reduction accumulated in fp32.

**Builder:**
```python
from kernels.norm.softmax_bwd_kernel import build_softmax_bwd_module

launch = build_softmax_bwd_module(N=8192, dtype_str="bf16")
launch(dy, y, dx, M, stream=torch.cuda.current_stream())
```

The builder takes `N` only; the row count is the runtime `m_in` launch argument.
Inputs must be **contiguous 2-D** tensors — reshape a 4-D attention gradient to
`(B*H*S, S)` before calling, since the buffer-tensor path assumes row-major rows.

**Paths:**
| Condition | Behaviour |
|---|---|
| `N >= tile_cols and N % tile_cols == 0` | 128-bit vectorized load/store (`tile_cols` = 1024 for f32, 2048 for 16-bit) |
| otherwise | masked scalar path for arbitrary `N` |
| `N <= 16384` | both operands register-resident across the reduction — ideal 3-unit traffic |
| `16384 < N <= 32768` | `Y` resident, `DY` re-read — 4 units |
| `N > 32768` | neither resident — 5 units |

Ideal traffic is 3 units (read `Y`, read `DY`, write `DX`); each operand dropped
from registers adds one more. The residency cap is on elements held per thread
(`N / BLOCK_THREADS`), so the tier boundaries fall at the same `N` for every
dtype. Use `softmax_bwd_buffered_operands(N, dtype_str)` to query the tier.

Both bounds are measured on an idle gfx950, not assumed. Pushing the middle tier
out to `N = 65536` spills and costs 29% (337.4 µs vs 261.8 µs at 2048x65536
bf16); dropping the middle tier costs 30-38% on the shapes it covers (4096x32768
bf16: 169.3 µs with `Y` resident vs 220.4 µs without).

Benchmark these on an **idle** GPU. A neighbouring tenant on the same device
distorts results by 20-35%, and single-sample idleness checks miss bursty
neighbours — sample repeatedly and reject a device that is busy in any sample.

**Notes:**
- One block per row. Small `M`/`N` are launch-bound rather than bandwidth-bound;
  effective bandwidth reads as a few percent of peak there and that is expected.
- The generic path unrolls `2 * ceil(N / 256)` scalar bodies, so compile time
  grows with `N` for large non-aligned rows.

---

## 3. GEMM kernel

### 3.1 Preshuffle GEMM (`kernels/gemm/preshuffle_gemm.py`)

MFMA 16x16-based GEMM with B-matrix preshuffle layout: `C[M,N] = A[M,K] @ B[N,K]^T`.

Uses the `@flyc.kernel` / `@flyc.jit` API.

**Builder:**
```python
from kernels.gemm.preshuffle_gemm import compile_preshuffle_gemm

launch_fn = compile_preshuffle_gemm(
    N=5120, K=8192,
    tile_m=16, tile_n=128, tile_k=256,
    in_dtype="fp8",
    out_dtype="bf16",
    epilogue="none",
    lds_stage=2,
)
```

Returns a `@flyc.jit`-decorated function that auto-compiles on first call.

**Parameters** (keyword-only):
| Parameter | Type | Description |
|---|---|---|
| `N, K` | int | GEMM dimensions: A[M,K], B[N,K], C[M,N]. M is a runtime arg, not a compile-time parameter. |
| `tile_m, tile_n, tile_k` | int | Block tile sizes |
| `in_dtype` | str | `"fp8"`, `"int8"`, `"fp16"`, `"bf16"` (default `"fp8"`) |
| `out_dtype` | str | Output dtype (default `"bf16"`) |
| `epilogue` | str | Fused epilogue: `"none"`, `"bias"`, `"bias_relu"`, `"bias_silu"`, `"bias_gelu"` (default `"none"`) |
| `lds_stage` | int | `2` = ping-pong LDS (tuned), `1` = single LDS buffer |
| `waves_per_eu` | int | Occupancy hint (None = default, 1-4 = limit occupancy) |
| `enable_scheduler` | bool | Enable the MLIR instruction scheduler (default `True`) |
| `use_async_copy` | bool | Use async DMA for A tile global-to-LDS transfer |
| `xcd_swizzle` | int | XCD remap factor for grid launch (0 = disabled) |

**Key constraints:**
- `tile_k` must be a positive divisor of `K`
- MX (block-scaled) GEMM is a separate kernel (`kernels/gemm/mxfp4_preshuffle.py`, `kernels/gemm/fp4_gemm_4wave.py`); INT4 is not supported by this kernel.

**MX A x MXFP4 B GEMM (`kernels/gemm/mxfp4_preshuffle.py`, gfx950):** the
`launch_gemm` `@flyc.jit` launcher runs `A x preshuffled MXFP4 B` with per-32
E8M0 scales, selecting the A element type via `a_dtype` (`"fp4"`, `"fp6"`, or
`"fp8"`; B is always MXFP4). This unified `launch_gemm` is the current gfx950
entry point (it replaced the earlier standalone `compile_mxfp6_gemm` from #780);
the separate `compile_mxfp4_gemm` in `kernels/gemm/gemm_fp8fp4_gfx1250.py` is the
distinct gfx1250 kernel. `batch>1` runs a strided-batched GEMM over `grid.z`.
Covered by `tests/kernels/test_preshuffle_gemm.py`.

**Pipeline details:**
- **lds_stage=2 (ping-pong)**: Two LDS buffers for A tiles. Cross-tile A0 prefetch overlaps VMEM with LDS reads
- **lds_stage=1 (single)**: CK-style intrawave schedule with single LDS buffer
- **K64-byte micro-step**: Each step issues 2x K32 MFMA operations
- **XOR16 swizzle**: Byte-level swizzle on LDS to avoid bank conflicts
- **B-preshuffle**: Shape (N0, K0, KLane, NLane, KPackBytes) = (N/16, K/64, 4, 16, kpack_bytes)
- **Fused epilogue**: selected via `epilogue=` (bias add + optional relu/silu/gelu activation)

**Launch function signature:**
```python
launch_fn(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, arg_bias, M_val, N_val, stream)
```

- `arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, arg_bias`: PyTorch tensors (auto-converted to memref). `arg_bias` is the fused epilogue bias (per-N, `out_dtype`); unused when `epilogue == "none"`.
- `M_val, N_val`: Python int (auto-converted to Int32)
- `stream`: `fx.Stream` (default stream if omitted)

---

## 3b. FlashAttention forward (`kernels/attention/flash_attn_generic.py`, `kernels/attention/flash_attn_gfx950.py`, `kernels/attention/flash_attn_fp8_gfx950.py`)

Dense FlashAttention forward. `build_flash_attn_func_module(num_heads, head_dim,
causal=..., dtype_str=..., num_kv_heads=...)` is the public builder; on
gfx950 + `head_dim == 128` it routes to the dual-wave software-pipelined fast path
(`build_flash_attn_dualwave_swp_module`), otherwise to the generic fallback.
Supports MHA and GQA/MQA (`num_kv_heads <= num_heads`), causal and non-causal,
arbitrary sequence length, and (bf16/f16) packed varlen + split-K.

### fp8 (e4m3fn) forward

| Property | Value |
|---|---|
| Arch / shape | gfx950 (CDNA4) only; `head_dim == 128`; dense only |
| Inputs | **pre-quantized** Q/K/V in `torch.float8_e4m3fn` (OCP e4m3fn, not fnuz); no in-kernel quantization |
| Descales | per-tensor shape-`[1]` fp32 `q_descale`, `k_descale`, `v_descale` (launch kwargs) |
| Math | QK on native `mfma_f32_32x32x16_fp8_fp8`, with `q_descale*k_descale*sm_scale` on fp32 logits; fp32 online softmax; PV applies `v_descale`; **fp32 accumulation** throughout |
| Output | `bf16` only |
| Unsupported (rejected with a clear error) | fp8 split-K (`num_kv_splits > 1`) and fp8 packed varlen (`cu_seqlens`) |

The PV path dequantizes fp8 V to bf16 in-kernel and accumulates P*V in bf16, keeping
the softmax probabilities at high precision. Build/launch example:

```python
from kernels.attention.flash_attn_generic import build_flash_attn_func_module

exe = build_flash_attn_func_module(num_heads=H, head_dim=128, causal=False,
                                   dtype_str="fp8", num_kv_heads=H_kv)
# Q/K/V are e4m3fn [B,S,H,D]; O is bf16; descales are shape-[1] fp32.
exe(q_fp8.view(-1), k_fp8.view(-1), v_fp8.view(-1), o_bf16.view(-1), B, S,
    q_descale=q_descale, k_descale=k_descale, v_descale=v_descale)
```

Reproduce the fp8 correctness sweep and the FlyDSL-fp8 vs aiter-ASM-fp8 comparison:

```bash
python3 tests/kernels/test_flash_attn_fwd.py --dtype fp8 --warmup 3 --iters 3
python3 tests/kernels/test_flash_attn_fwd.py --dtype fp8 --compare --warmup 10 --iters 50
```

---

## 3c. SonicMoE A16W16/A16W4 (`kernels/moe/sonic.py`)

The gfx950 inference path composes the existing FlyDSL routing and
`moe_2stage_a16wmix` MFMA kernels. The routing stage rounds each expert's rows to
`route_tile_m = lcm(tile_m, down_tile_m)` and records packed token/slot indices.
Stage 1 uses `tile_m`, gathers the original BF16/FP16 rows while loading A, and
fuses the selected activation. Stage 2 independently uses `down_tile_m`, consumes
the sorted A16 intermediate, and performs routing-weighted packed A16 atomic
scatter. No explicit gathered activation tensor is materialized. Supported
activations are SwiGLU, GEGLU, ReGLU, GELU-tanh, ReLU, SiLU, and ReLU squared.

```python
from dataclasses import replace

from kernels.moe.sonic import (
    SonicMoE,
    SonicMoEConfig,
    prepare_sonic_bf16_weights,
    prepare_sonic_fp16_weights,
    prepare_sonic_mxfp4_weights,
    sonic_moe_backward,
    sonic_moe_backward_routes,
)

cfg = SonicMoEConfig(
    hidden_size=4096, intermediate_size=14336,
    num_experts=256, top_k=8,
    tile_m=32, tile_n=128, tile_k=128,
    stage1_k_wave=1,
    activation="swiglu",
)
# GLU w1: [E, 2*I, H] in [gate | up] order.
# Non-GLU w1: [E, I, H]. w2 is always [E, H, I].
# Optional b1/b2 are [E, 2*I or I] and [E, H], respectively.
# Choose one prepared format. Weight preparation is outside the hot path.
weights = prepare_sonic_bf16_weights(w1, w2, cfg, b1=b1, b2=b2)
# weights = prepare_sonic_mxfp4_weights(w1, w2, cfg, b1=b1, b2=b2)
op = SonicMoE(cfg, weights)
out = op(hidden_states_bf16, router_logits_bf16)

# The fixed-K BF16 SwiGLU training path can retain the exact route-order
# preactivation produced by this invocation and reuse it in backward.  This
# avoids the backward W1 recomputation while inference keeps using
# ``forward_topk``/``__call__`` without allocating training state.
out, forward_state = op.forward_topk_training(
    hidden_states_bf16,
    topk_ids_i32,
    topk_scores_f32,
)

# Dense FP16 uses the same logical layouts and native FP16 MFMA.
cfg_fp16 = replace(cfg, compute_dtype="fp16")
weights_fp16 = prepare_sonic_fp16_weights(w1_fp16, w2_fp16, cfg_fp16)
out_fp16 = SonicMoE(cfg_fp16, weights_fp16)(hidden_states_fp16, router_logits_fp16)

# Initial training API: logical dense expert-major weights, explicit fixed-K
# routes, BF16/FP16, any supported activation, and optional expert bias. Omitting
# b1/b2 keeps the original four-result return contract.
dx, dw1, dw2, droute_scores, db1, db2 = sonic_moe_backward(
    hidden_states_bf16,
    w1,
    w2,
    topk_ids_i32,
    topk_scores_f32,
    grad_output_bf16,
    cfg,
    b1=b1,
    b2=b2,
    forward_state=forward_state,
)

# Flat routes preserve one score-gradient destination per original edge,
# including duplicate (token, expert) pairs and tokens with zero routes.
dx, dw1, dw2, droute_scores, db1, db2 = sonic_moe_backward_routes(
    hidden_states_bf16,
    w1,
    w2,
    token_indices_i32,
    expert_indices_i32,
    route_scores_f32,
    grad_output_bf16,
    cfg,
    b1=b1,
    b2=b2,
)
```

`forward_topk_training` currently supports dense BF16 SwiGLU with fixed-K
routes.  The state is tied to the exact forward invocation (including W1, B1,
route IDs, and W1 layout), is immutable, and may be reused with
`retain_graph=True`.  FP16, non-SwiGLU, ragged routing, and unsupported layout
combinations continue through the standalone backward path or are rejected
explicitly.  Passing `forward_state=None` always selects that tested fallback.

For GLU backward, both entry points also accept ``interleaved_w1=True``. In
that mode each expert's raw W1 rows, optional B1 entries, and returned
W1/B1 gradients use ``[g0, u0, g1, u1, ...]`` order instead of the default
``[gate | up]`` order. This is a logical dense layout and is distinct from the
quantized ``guinterleave`` preshuffle; pointwise activations reject the option.

Dense BF16/FP16 shapes may use a 64-wide intermediate dimension. For example,
`H=128, I=64` uses `tile_n=64`, `tile_k=128`, `down_tile_n=128`, and
`down_tile_k=64`. MXFP4 retains its packed-load requirement that both K tiles
are at least 128, so this `I=64` configuration is dense-only.

Stage 1 can repartition its fixed four-wave workgroup with
`stage1_k_wave={1,2,4}`. Values above one assign multiple waves to independent K
slices and reduce their FP32 partials through LDS; this can help small-M,
long-contraction shapes at the cost of fewer N partitions and more LDS traffic.
The hidden size must be divisible by `stage1_k_wave * tile_k`, and the
constructor rejects combinations whose A buffers or reduction scratch exceed
gfx950's 160 KiB LDS limit.

For multi-subtile N shapes, the Stage-1 K loop rotates the next B fragments into
the registers released immediately after each current N subtile is consumed.
This preserves one-tile prefetching while avoiding simultaneous whole-tile
current/next B lifetimes. Single-subtile shapes retain the original prefetch
order because there is no following N work available to hide a delayed load.

Weights are preshuffled once during preparation. Workspaces and compiled launchers
are reused. Expert counts with an exact single-wave layout use the FlyDSL router;
this includes the production E=896 shape (`VPT=14`, 64 threads per token).
Unsupported counts retain a PyTorch softmax/top-k fallback followed by the same
FlyDSL sort and grouped GEMMs. With `renormalize=True`, the native router ranks
raw logits and evaluates exponentials only for the selected K entries; the full-E
softmax path is retained when non-renormalized probabilities are requested. Call
`forward_topk` to supply routing directly.

The Sonic inference entry points additionally specialize unmasked single-token
routing. Top-k guarantees distinct experts, so the T=1 path emits one padded
expert block per route slot directly and skips the E-wide histogram, prefix scan,
and scatter. The logits entry point performs top-k selection and metadata
emission in the same kernel; `forward_topk` uses the corresponding direct sorter.
Both kernels overlap output zeroing in separate CTAs. This route-slot order is an
internal opt-in: the standalone `moe_softmax_sort_flydsl` and
`moe_sorting_flydsl` APIs retain ascending expert-ID order by default, as does
backward's segment reconstruction.

Stage 2 defaults to the faster, lower-memory `stage2_output_mode="atomic"`.
The experimental `"reduce"` mode is available only for fixed-K routing: it
writes one A16 row per `(token, slot)` and then reduces those rows in FP32, so it
allocates an additional `tokens * top_k * hidden_size` A16 scratch tensor. Flat
ragged routes always use atomic scatter even when the config requests reduce.

`prepare_sonic_mxfp4_weights` is the validated weight-only A16W4 path. It quantizes
each contiguous 32-value weight block to packed E2M1 FP4 with one E8M0 scale,
then converts both values and scales to the gfx950 kernel layouts. Activations and
the stage-1 intermediate remain BF16; the kernels upconvert weights and execute
BF16 MFMA. It is therefore **not** activation MXFP8/A8W4 and does not use the
CDNA4 scaled-MFMA instruction. Compare kernel correctness against the dequantized
quantized weights; a comparison with the original BF16 weights additionally
contains the expected model-quantization error.

Packed standard-layout weights use a 64-bit per-expert resource base, so the
complete expert tensor may exceed the 4 GiB 32-bit buffer-offset range. Each
individual expert must still satisfy that range, and the current preshuffled
E8M0 scale resource has a 4 GiB whole-tensor span limit. Prepared buffers are
checked for exact shape, padded scale length, contiguity, dtype, device, and
alignment before any raw pointer reaches a kernel. MXFP4 tiles require
`tile_k >= 128` in both stages.

Shape-bucket autotuning is available as a separate wrapper:

```python
from kernels.moe.sonic_autotune import SonicMoEAutotuner

op = SonicMoEAutotuner(
    cfg,
    weights,
    warmup=5,
    rep=20,
)
out = op(hidden_states_bf16, router_logits_bf16)
print(op.best_config, op.last_results)
```

The tuner benchmarks the complete router + sort + two-GEMM forward, validates
candidate output against the base configuration, and keys winners by a
power-of-two token bucket, model shape, dtypes, device/architecture, FlyDSL
and PyTorch/ROCm versions, candidate set, and kernel source hash. Candidate
workspaces are released after every measurement so search memory does not grow
with the candidate count. Only correctness-validated searches are persisted; the
disk cache uses a lock plus atomic replacement for concurrent processes. Its
default location is `~/.flydsl/autotune/sonic_moe.json`; set
`FLYDSL_AUTOTUNE_CACHE_DIR` or pass `cache_dir=` to relocate it. Pass
`force_tune=True` on a call to remeasure a key.

Tile choice also depends on the expert-load distribution, not only tensor shape.
Tune with representative router logits. If one shape has materially different
traffic profiles, construct separate tuners with `profile_key="uniform"`,
`profile_key="decode-skew"`, or another stable application label so their disk
cache entries do not collide.

Optional expert-major BF16/FP16 `b1`/`b2` are prepared with the weights and fused
before the activation and route weighting, respectively. The backward paths
support dense BF16/FP16 weights, fixed-K or flat ragged
routing, every supported activation, and optional expert bias. They independently
re-sort routes and recompute the materialized pre-activation and projection, so
they do not retain or alias an inference workspace across calls. The bring-up
implementation uses device-driven grouped MFMA kernels for BF16 SwiGLU W1 and
W2 recompute when every expert segment is bounded by 128 rows. The other four
matrix products still use per-expert A16W16 GEMMs and one host synchronization
to read expert frequencies. Its independent sort unit remains 64 rows because
those generic GEMMs currently require contraction-K blocks aligned to 64;
forward `route_tile_m` tuning does not alter that invariant. Activation,
routing, reduction, and every tensor calculation remain FlyDSL device kernels.
The `dout * route_score` input is rounded to the selected A16 dtype before the
backward GEMMs, so this is not bitwise parity with a legacy FP32-scaled Triton
grouped GEMM.

Flat-route backward accumulates routed input gradients through an FP32 atomic
buffer, so high-fan-in tokens can be non-deterministic at the last few bits. The
packed A16 atomic forward output has the same property. See
`examples/06-sonicMoE.py` for correctness and warm-cache benchmarking.

### Scaled-MFMA status

The existing `kernels/moe/mxfp_moe/` A4W4/A8W4 implementation is not exposed by
`SonicMoE`: its repository tests mark the fused end-to-end modes as known-broken
and unsafe to run after observed low cosine and illegal-address/JIT corruption.
The next activation-low-precision mode should be a distinct `a8w4_mx` compute
mode, with per-1x32 MXFP8 payload/scales and a scaled-MFMA local pipeline adapted
from `kernels/mega_moe/`. It also needs indexed token/scale gather and a local
weighted-scatter epilogue. Calling the validated A16W4 path “A8W4” would hide both
the numerical and performance distinction.

### gfx950 tuning notes

Tune against the complete `(tokens, H, I, E, top_k)` bucket rather than choosing
the M tiles from padding alone. `tile_m` and `down_tile_m` independently control
the two MFMA workgroups, while their least common multiple controls per-expert
routing padding. `tile_n`/`tile_k` and their `down_` counterparts trade loop count
against LDS and VGPR pressure. The default tuner uses a bounded profile list,
not a Cartesian product. It covers equal M tiles in `{16,32,64,128}`, the useful
stage-1/stage-2 pairs `(32,128)` and `(64,128)`, dense-only K64 profiles including
the measured `S1=(128,256,64) / S2=(128,128,64)` point, Stage-2 XCD swizzle 8,
and BM16 `stage1_k_wave={2,4}` decode variants. The dense decode set also
contains the measured asymmetric-N points `S1 BN64/S2 BN128/k_wave=2` and
`S1 BN128/S2 BN64/k_wave=4`. Packed MXFP4/INT4 candidates keep both K tiles at
least 128. Illegal DMA, divisibility, and 160 KiB LDS combinations are pruned.
Cache policy, XCD swizzle, waves-per-EU, and persistent Stage 2 remain available
for a custom candidate sweep and are included in the autotune cache identity.

For decode, workspace sizing is based on the number of routes that can actually
activate experts. With `R=tokens*top_k`, `A=min(E,R)`, and distinct top-k IDs per
token, the padded block bound is the smaller of
`floor((R + A*(route_tile_m-1))/route_tile_m)` and
`A*ceil(tokens/route_tile_m)`. Each GEMM launch then converts that padded-row
bound to its own M tile. Thus `T=1, E=896, top_k=2, route_tile_m=32` reserves two
route blocks (64 rows), not 896 empty expert blocks.

On one MI355X, the dense BF16 decode shape
`T=1, H=3584, I=512, E=896, top_k=16` with
`S1=(BM16,BN64,BK128,k_wave=2)` and `S2=(BM16,BN128,BK128)` measured the
following warm-cache, same-device changes after enabling direct T=1 metadata:

| Entry point | Generic routing | Direct T=1 | Speedup |
|---|---:|---:|---:|
| logits, complete MoE forward | 54.598 us | 44.692 us | 1.222x |
| precomputed top-k, complete MoE forward | 43.283 us | 33.905 us | 1.277x |
| precomputed top-k sorter kernel | 9.748 us | 2.059 us | 4.73x |

These numbers include output clearing in both complete forwards. A router-only
CTA was slightly slower end to end because Stage 2's atomic output still needs
the clear; keeping the concurrent clear CTAs hides that work under routing.

For one MI355X warm-cache run at `T=128, H=4096, I=14336, E=8, top_k=2`, with
weight preparation and JIT excluded, the measured points were:

| Stage-1/2 tile `(M,N,K)` | Padding ratio | Latency | Useful throughput |
|---|---:|---:|---:|
| `(16,128,128)` | 1.25 | 999.39 us | 90.25 TFLOP/s |
| `(32,128,128)` | 1.50 | 708.96 us | 127.22 TFLOP/s |
| `(64,128,128)` | 2.00 | 488.86 us | 184.50 TFLOP/s |
| `(64,128,256)` | 2.00 | 494.98 us | 182.22 TFLOP/s |
| `(64,256,128)` | 2.00 | 533.49 us | 169.06 TFLOP/s |
| `(128,128,128)` | 4.00 | 619.74 us | 145.54 TFLOP/s |

This is a tuning example, not a universal default: expert imbalance and token
bucket size change the rounding cost substantially. Reproduce a point with:

```bash
PYTHONPATH=. python examples/06-sonicMoE.py \
  --tokens 128 --hidden-size 4096 --intermediate-size 14336 \
  --experts 8 --top-k 2 --tile-m 64 --tile-n 128 --tile-k 128 \
  --weight-dtype bf16
```

Use the independent Stage-2 and XCD controls to reproduce the current gfx950
throughput point:

```bash
PYTHONPATH=. python examples/06-sonicMoE.py \
  --tokens 4096 --hidden-size 4096 --intermediate-size 2048 \
  --experts 64 --top-k 8 --tile-m 128 --tile-n 256 --tile-k 64 \
  --down-tile-m 128 --down-tile-n 128 --down-tile-k 64 \
  --stage2-xcd-swizzle 8 \
  --weight-dtype bf16 --check
```

Add `--stage1-k-wave 2` to benchmark a legal slice-K variant of the same shape.
`--stage1-k-wave 4` is also available for tile shapes whose larger reduction
scratch fits in LDS.

For the command above, a same-device MI355X run with a fixed seed and 7x100
warm-cache measurements reduced median end-to-end latency from `2021.448 us` to
`1871.134 us` after Stage 1 adopted N-subtile B-register rotation (`1.080x`).
Useful BF16 MoE throughput increased from approximately `815.9` to
`881.4 TFLOP/s`; routing, padding (`36,480` rows), and Stage 2 were unchanged.

The first backward grouped specialization reuses the gfx950 Stage-1 MFMA body
with logical `[E, 2I, H]` weights and a raw preactivation store. T1 uses a
device-side expert grid directly. Short-route calls with at least 64 tokens
first build a compact device queue of real M tiles, then launch each tile as an
independent CTA; neither schedule reads expert frequencies back to the host.
For `T=128, H=3584, I=512, E=896, top_k=16`, paired warm-cache
measurements reduced W1 recompute from `48.595 ms` to `1.196 ms` (`40.62x`) and
the complete backward from `299.630 ms` to `253.449 ms` (`1.182x`). Empty
experts do not launch matrix work, and 65-row and bias cases are covered by the
backward tests. The T1 expert-grid profile is `BM16/BN64/BK64/k_wave4`; the
compact profile is described below. Both are intentionally limited to fixed-K
calls with at most 128 tokens, or ragged calls with at most 128 total routes. A
balanced `T4096/E64` case has 512 rows per expert: forcing the short-M kernel
made the full backward `30.66 ms`, whereas retaining the BM64 fallback measured
`22.33 ms`. Later grouped kernels should use separate short- and long-M
schedules rather than extending this threshold blindly.

The compact W1 profile uses `BM16/BN128/BK64/k_wave2`; its two-launch descriptor
builder costs about `12-14 us` on MI355X. Including that cost, it changed W1
latency from `156.32` to `76.16 us` for T64 with 16 hot experts, from `305.40`
to `120.84 us` for the corresponding T128 skew, and from `1127.73` to
`1108.45 us` for balanced T128 routing. The host launches a proven-safe upper
bound, while a device counter and compact descriptors suppress empty work and
expose a hot expert's M tiles to separate CTAs.

The matching W2 recompute specialization reads logical `[E, H, I]` weights and
writes the unweighted projection directly by sorted row. Its measured
`BM32/BN256/BK64` profile includes zeroing untouched projection padding: on the
same `T128/H3584/I512/E896/K16` target it reduced this phase from `48.956 ms`
to `0.667 ms` (`73.4x`), and reduced complete backward on top of grouped W1
from about `251.0 ms` to `201.0 ms` (`1.249x`). At T1 the W2 phase measured
`885.1 us` versus `24.4 us` (`36.3x`). The complete T1 backward measured
`5.27 ms`, down from the original `6.99 ms`. The same conservative 128-row
policy keeps the large-T BM64 path; the independently remeasured
`T4096/H4096/I2048/E64/K8` full backward remained `22.28 ms`.

The grouped dA specialization completes the backward down-projection pair by
reading public row-major `W2[E,H,I]` directly and computing
`dY[sorted,H] @ W2[e,H,I]`. It uses the gfx950 NN pipeline: 16-byte async
global-to-LDS loads, an LDS transpose plus `LDSReadTrans16_64b` for B, and
BF16 `MFMA 16x16x32`. Since the backward already synchronizes to obtain expert
frequencies for dW, the kernel chooses among three measured profiles using the
actual largest expert segment: `BM16/BN64/BK128/w1x4` for one row,
`BM32/BN64/BK64/w2x2` through 16 rows, and `BM64/BN64/BK64/w2x2` above that.
The output is pre-zeroed because real-M tiles intentionally omit sorter
padding. FP16 and unsupported shapes retain the general GEMM path.

On the same MI355X, isolated dA latency fell from `901.5 us` to `21.5 us` at
`T1/H3584/I512/E896/K16`, from `49.549 ms` to `0.601 ms` for balanced T128,
from `921.3 us` to `57.7 us` for T128 routed to 16 hot experts, and from
`3.446 ms` to `1.611 ms` at balanced `T4096/H4096/I2048/E64/K8`. Paired
end-to-end backward medians (all four returned gradients checked against the
legacy path) were `5.266 -> 4.408 ms`, `202.854 -> 153.552 ms`,
`5.381 -> 4.512 ms`, and `22.360 -> 18.853 ms`, respectively. A prototype
that reused W1's compact BM16 descriptor queue reached `0.603 ms` on balanced
T128, slightly behind the selected `0.601 ms` expert-grid profile, so it was
not retained.

The grouped dX specialization computes `dZ[sorted,2I] @ W1[e,2I,H]` with a
gfx950-native NN MFMA pipeline and keeps the public row-major weight layout.
T1 consumes sorter metadata directly, while short-route calls reuse W1's
existing BM16 compact descriptor queue; no second builder launch is added.
The persistent grid is capped at 1024 workgroups and selects `BN128/2-wave`
below 256 active experts or `BN256/4-wave` for dense expert sets when the
hidden dimension permits it. On `H3584/I512/E896/K16`, isolated dX changed
from `876.36` to `18.95 us` at T1, from `852.21` to `57.01 us` for T128 with
16 hot experts, and from `48.410 ms` to `1.209 ms` for balanced T128. The
corresponding complete backward medians changed from `4.426` to `3.549 ms`,
`4.592` to `3.735 ms`, and `154.230` to `103.507 ms`. Long T4096 calls retain
the general BM64 path; paired measurements stayed within 0.4% noise.

Grouped BF16 SwiGLU dW1/dW2 initialization also avoids a dense fill when the
active-expert set is sufficiently dense. Both output tensors are allocated
uninitialized; grouped TN overwrites every element of every active expert,
while one expert-grid kernel issues 128-bit stores only for inactive slabs. A
64-bit expert base plus an exact expert-local BRSRC keeps the production
`dW1[E,2I,H]` correct even when its total allocation exceeds 4 GiB. If fewer
than one eighth of experts are active, the original dense fill remains faster
and is retained; FP16, non-SwiGLU, and non-grouped paths are unchanged.

On an otherwise idle MI355X, same-device warm-cache medians were
`8.555 -> 7.188 ms` for balanced `T128/H3584/I512/E896/K16`. A GPU profile
reduced five fill kernels totaling `1.492 ms` to the three unrelated scratch
fills totaling `0.084 ms`. The sparse guard kept T1 at `1.810 -> 1.797 ms` and
T128/hot16 at `2.116 -> 2.114 ms`; the all-active
`T4096/H4096/I2048/E64/K8` case improved from `15.005` to `14.570 ms`.

Run the validated A16W4 path or let the shape-bucket tuner choose the tiles with:

```bash
PYTHONPATH=. python examples/06-sonicMoE.py \
  --tokens 128 --hidden-size 1024 --intermediate-size 1024 \
  --experts 8 --top-k 2 --weight-dtype mxfp4 --check

PYTHONPATH=. python examples/06-sonicMoE.py \
  --tokens 128 --hidden-size 1024 --intermediate-size 1024 \
  --experts 8 --top-k 2 --weight-dtype mxfp4 --autotune \
  --autotune-warmup 3 --autotune-iters 10 \
  --autotune-profile-key representative-prefill --check
```

For A16W4, `--check` reports two separate quantities: kernel output versus a
dequantized-weight oracle (the correctness gate), and that quantized oracle versus
the original BF16 model (model-dependent quantization quality). Weight preparation,
reference computation, autotuning, and first-call JIT are excluded from the final
warm-cache timing.

---

## 4. Shared utilities

### 4.1 Common kernel helpers (`kernels/common/kernels_common.py`)

Shared kernel utilities used across GEMM/MoE/norm kernels.

| Function | Description |
|---|---|
| `get_warp_size(arch=None)` | Wave size for the arch: `32` on gfx10/11/12, else `64` |
| `dtype_to_elem_type(dtype_str)` | Map a dtype string to the Fly element type |
| `validate_moe_dtypes(a_dtype, b_dtype)` | Validate an allowed MoE A/B dtype pairing |
| `get_llvm_ptr(ptr, offset, dtype_bytes, ...)` | Compute a byte-offset LLVM pointer |
| `atomic_add(...)` | Emit an atomic add |
| `_if_then(if_op, scf=None)` / `_if_else(if_op, scf=None)` | SCF `if`/`else` region context managers |

### 4.2 MFMA epilogues (`kernels/mma/mfma_epilogues.py`)

Configurable epilogue strategies for MFMA 16x16 kernels.

| Function | Description |
|---|---|
| `default_epilog(...)` | Standard row-iterator: `row = bx_m + mi*16 + lane_div_16*4 + ii` |
| `c_shuffle_epilog(...)` | CK-style LDS CShuffle: write to LDS → barrier → remap threads → half2 store |
| `mfma_epilog(use_cshuffle, ...)` | Dispatcher: calls default or CShuffle based on flag |

### 4.3 Preshuffle pipeline (`kernels/mma/mfma_preshuffle_pipeline.py`)

Shared data movement and layout utilities for preshuffle GEMM kernels.

| Function | Description |
|---|---|
| `make_preshuffle_b_layout(...)` | Build B-preshuffle layout: (N/16, K/64, 4, 16, kpack_bytes) |
| `load_b_pack_k32(...)` | Load B pack for K32 MFMA micro-step (returns i64) |
| `tile_chunk_coord_i32(...)` | Map (thread, chunk) → (row, col) for tile loads |
| `buffer_copy_gmem16_dwordx4(...)` | 16-byte global load via buffer-load dwordx4 |
| `lds_store_16b_xor16(...)` | Store 16B to LDS with XOR16 swizzle |
| `lds_load_pack_k32(...)` | Load A-pack from LDS for K32 micro-step |
| `swizzle_xor16(...)` | XOR-based swizzle for LDS bank-conflict avoidance |

### 4.4 Layout coordinate helpers

Native Fly dialect coordinate mapping (in `flydsl.expr` and `kernels/mma/mfma_preshuffle_pipeline.py`):

| Function | Description |
|---|---|
| `fx.crd2idx(crd, layout)` | Coordinate → flat index (Fly dialect op) |
| `fx.idx2crd(idx, layout)` | Flat index → coordinate tuple (Fly dialect op) |
| `fx.get(int_tuple, mode)` | Extract element at index from `!fly.int_tuple` |
| `crd2idx(crd, layout)` | Wrapper in `kernels/mma/mfma_preshuffle_pipeline.py` (auto index cast) |

---

## 5. Kernel API comparison

### New API (GEMM)

Used by `kernels/gemm/preshuffle_gemm.py`:

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, rocdl

@flyc.kernel
def gemm_kernel(arg_c: fx.Tensor, arg_a: fx.Tensor, ...):
    tid = gpu.thread_idx.x
    # ... uses fx.*, Numeric/Vector, gpu.*, rocdl.* ...

@flyc.jit
def launch_fn(arg_c: fx.Tensor, ..., stream: fx.Stream = fx.Stream(None)):
    gemm_kernel(arg_c, ...).launch(grid=..., block=..., stream=stream)
```

---

## 6. Kernel decision tree

```
What operation do you need?
│
├── Normalization
│   ├── Need bias (beta) term? → LayerNorm (kernels/norm/layernorm_kernel.py)
│   └── No bias term?         → RMSNorm (kernels/norm/rmsnorm_kernel.py)
│
├── Softmax
│   ├── Row-wise softmax      → Softmax (kernels/norm/softmax_kernel.py)
│   └── Softmax gradient      → Softmax backward (kernels/norm/softmax_bwd_kernel.py)
│
├── Matrix Multiply (GEMM)
│   ├── Standard GEMM (uniform precision)
│   │   ├── FP8 / INT8 / FP16 / BF16
│   │   └── → compile_preshuffle_gemm()
│   │
│   └── Uses new @flyc.kernel API
│       └── See kernels/gemm/preshuffle_gemm.py
│
├── MoE (Mixture of Experts)
│   ├── SonicMoE A16W16/A16W4 forward → SonicMoE (kernels/moe/sonic.py)
│   ├── Blockscale MoE (gate+up+reduce)
│   └── Standard MoE (fp8/f16/bf16/int8/int4)
│       └── → kernels/moe/moe_gemm_2stage.py
│
└── Building blocks
    ├── Common kernel helpers    → kernels/common/kernels_common.py
    ├── MFMA epilogue selection  → kernels/mma/mfma_epilogues.py
    └── Preshuffle data movement → kernels/mma/mfma_preshuffle_pipeline.py
```

---

## 7. Source files

| File | Description |
|---|---|
| `kernels/gemm/preshuffle_gemm.py` | GEMM (preshuffle layout) |
| `kernels/moe/moe_gemm_2stage.py` | MoE GEMM 2-stage (gate/up + reduce) |
| `kernels/moe/mxfp_moe/` | Fused a4w4/a8w4 MoE 2-stage GEMM (device fp4 re-quant) |
| `kernels/moe/sonic.py` | gfx950 SonicMoE A16W16/A16W4 inference forward orchestration |
| `kernels/moe/sonic_autotune.py` | Shape-bucket SonicMoE tile autotuner and disk cache |
| `kernels/attention/pa_decode_fp8.py` | Paged attention decode (FP8) |
| `kernels/attention/flash_attn_generic.py` | FlashAttention generic fallback |
| `kernels/attention/flash_attn_gfx950.py` | FlashAttention gfx950 bf16/f16 fast path |
| `kernels/attention/flash_attn_fp8_gfx950.py` | FlashAttention gfx950 fp8 dense fast path |
| `kernels/norm/layernorm_kernel.py` | LayerNorm (layout API) |
| `kernels/norm/rmsnorm_kernel.py` | RMSNorm (layout API) |
| `kernels/norm/softmax_kernel.py` | Softmax (layout API) |
| `kernels/norm/softmax_bwd_kernel.py` | Softmax backward (layout API) |
| `kernels/attention/fused_rope_cache_kernel.py` | Fused RoPE + KV cache |
| `kernels/comm/custom_all_reduce.py` | Multi-GPU all-reduce |
| `kernels/gemm/rdna_f16_gemm.py` | RDNA FP16 GEMM |
| `kernels/gemm/rdna_fp8_preshuffle_gemm.py` | RDNA FP8 GEMM |
| `kernels/gemm/gemm_common_gfx1250.py` | GFX1250 GEMM common |
| `kernels/gemm/gemm_fp8fp4_gfx1250.py` | GFX1250 FP8/FP4 GEMM |
| `kernels/gemm/wmma_gemm_gfx1250.py` | GFX1250 WMMA GEMM |
| `kernels/mma/mfma_epilogues.py` | MFMA epilogue helpers |
| `kernels/mma/mfma_preshuffle_pipeline.py` | Preshuffle data movement and layout utilities |
| `kernels/mma/pipeline_utils.py` | Pipeline utility helpers |
| `kernels/common/kernels_common.py` | Common kernel utilities |
| `kernels/common/tensor_shim.py` | GTensor/STensor abstraction |

## 8. Test files

| File | Tests |
|---|---|
| `tests/kernels/test_preshuffle_gemm.py` | GEMM fp8/int8/fp16/bf16 |
| `tests/kernels/test_moe_gemm.py` | MoE GEMM |
| `tests/kernels/test_moe_reduce.py` | MoE reduce kernel |
| `tests/kernels/test_sonic_moe.py` | SonicMoE BF16/FP16/A16W4 correctness, autotuning, routing, workspace, validation |
| `tests/kernels/test_pa.py` | Paged attention decode |
| `tests/kernels/test_flash_attn_fwd.py` | FlashAttention |
| `tests/kernels/test_layernorm.py` | LayerNorm |
| `tests/kernels/test_rmsnorm.py` | RMSNorm |
| `tests/kernels/test_softmax.py` | Softmax |
| `tests/kernels/test_softmax_bwd.py` | Softmax backward |
| `tests/kernels/test_fused_rope_cache.py` | Fused RoPE + KV cache |
| `tests/kernels/test_allreduce.py` | Multi-GPU all-reduce |
| `tests/kernels/test_rdna_gemm.py` | RDNA GEMM |
| `tests/kernels/test_gemm_fp8fp4_gfx1250.py` | GFX1250 FP8/FP4 GEMM |
| `tests/kernels/test_wmma_gemm_gfx1250.py` | GFX1250 WMMA GEMM |
| `tests/kernels/test_vec_add.py` | Vector addition |
| `tests/kernels/test_quant.py` | Quantization utilities |
| `tests/kernels/benchmark_common.py` | Shared benchmark infrastructure |
