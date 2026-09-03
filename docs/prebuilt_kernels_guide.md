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
| **SonicMoE forward** | `SonicMoE(config, weights)` | Host-composed FlyDSL | BF16 activation/bias; BF16 or MXFP4 weight | Routing/top-k + sort, fused activation, weighted down scatter |

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

## 3c. SonicMoE BF16/A16W4 forward (`kernels/moe/sonic.py`)

The gfx950 inference path composes the existing FlyDSL routing and
`moe_2stage_a16wmix` MFMA kernels. The routing stage rounds each expert's rows to
`tile_m` and records packed token/slot indices. Stage 1 gathers the original BF16
rows while loading A and fuses the selected activation; stage 2 consumes the
sorted BF16 intermediate and performs routing-weighted BF16 atomic scatter. No
explicit gathered activation tensor is materialized. Supported activations are
SwiGLU, GEGLU, ReGLU, GELU-tanh, ReLU, SiLU, and ReLU squared.

```python
from kernels.moe.sonic import (
    SonicMoE,
    SonicMoEConfig,
    prepare_sonic_bf16_weights,
    prepare_sonic_mxfp4_weights,
)

cfg = SonicMoEConfig(
    hidden_size=4096, intermediate_size=14336,
    num_experts=256, top_k=8,
    tile_m=32, tile_n=128, tile_k=128,
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
```

Weights are preshuffled once during preparation. Workspaces and compiled launchers
are reused. Power-of-two expert counts up to 1024 use the FlyDSL router; other
counts (for example E=896) use a PyTorch softmax/top-k fallback followed by the
same FlyDSL sort and grouped GEMMs. Call `forward_topk` to supply routing directly.

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

This API is inference-forward only. Optional expert-major BF16 `b1`/`b2` are
prepared with the weights and fused before the activation and route weighting,
respectively. Saved pre-activation, backward, varlen-K dW, and activation
derivatives are not yet provided. The BF16 atomic output is non-deterministic at
the last few bits. See `examples/06-sonicMoE.py` for correctness and warm-cache
benchmarking.

An optimized training path needs more than a generic GEMM fallback: it must retain
or recompute gate/up pre-activations, add dSwiGLU, provide transposed grouped dX
GEMMs, expose actual and padded expert offsets, implement expert-ragged-K dW, and
perform segmented bias reductions. The current sorter and forward workspace do
not expose or lifetime-manage that state, so the inference API deliberately
rejects tensors requiring gradients.

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
`tile_m` from padding alone. `tile_m` controls both per-expert rounding and MFMA
workgroup efficiency; `tile_n`/`tile_k` trade loop count against LDS and VGPR
pressure. The default tuner starts from `tile_m={16,32,64,128}` and
`tile_n,tile_k={128,256}`, pruning candidates that fail the constructor's DMA,
divisibility, or 160 KiB LDS guards. The cache policy, XCD swizzle, waves-per-EU,
and persistent stage-2 switches are also exposed on `SonicMoEConfig` for a custom
candidate sweep.

For decode, workspace sizing is based on the number of routes that can actually
activate experts. With `R=tokens*top_k`, `A=min(E,R)`, and distinct top-k IDs per
token, the padded block bound is the smaller of
`floor((R + A*(tile_m-1))/tile_m)` and `A*ceil(tokens/tile_m)`. Thus
`T=1, E=896, top_k=2, tile_m=32` reserves and launches two blocks (64 rows), not
896 empty expert blocks.

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
│   ├── SonicMoE BF16/A16W4 forward → SonicMoE (kernels/moe/sonic.py)
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
| `kernels/moe/sonic.py` | gfx950 SonicMoE BF16/A16W4 inference forward orchestration |
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
| `tests/kernels/test_sonic_moe.py` | SonicMoE BF16/A16W4 correctness, autotuning, routing, workspace, validation |
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
