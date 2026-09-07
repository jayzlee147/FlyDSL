# SonicMoE forward-state reuse for backward

Status: phase 1 and the first gfx950 exact-row consumer are implemented and
measured.  Forward saves compact route-order BF16 preactivation; backward can
consume it to skip W1 recomputation and, on the compact hostless path, avoid
BM64-padded gather/state materialization.  Routing-metadata reuse and fused
dA/dscore remain follow-up work.

## Summary

The standalone compatibility path treats `SonicMoE.forward_topk` as an
inference operation and invokes backward later.  Consequently standalone
backward sorts the routes again and materializes both matrix products again:

1. W1 is recomputed into A16 preactivation, then the activation is materialized.
2. W2 is recomputed into an unweighted A16 projection solely for the route-score
   gradient.

The production T128/H3584/I512/E896/K16 measurements put these two recomputes at
roughly 1.17 ms and 0.56 ms respectively (the older numbers recorded in
`prebuilt_kernels_guide.md` are 1.196 ms and 0.667 ms).  They are avoidable.

The recommended training contract is:

- emit immutable, invocation-owned routing metadata and W1 preactivation from
  forward;
- regenerate the inexpensive elementwise activation in backward by default;
- fuse route-score calculation into the unscaled dA contraction, using
  `dscore = dot(dout @ W2, activation) + dot(dout, b2)`, instead of materializing
  `[routes, H]` projection;
- retain an opt-in saved-activation mode, and a strict-numerics saved-projection
  mode for validation, but do not make either the default;
- keep inference `forward_topk()` and standalone backward without a forward
  state source-compatible.

This is the same high-level lifetime choice as upstream SonicMoE: its
`_UpProjection` saves the preactivation `h`, while `_DownProjection.backward`
reconstructs/fuses the activation and score-gradient work.  It does not retain
the down projection.

## Implemented phase 1

The BF16 SwiGLU fixed-K training path now exposes:

```python
output, state = operator.forward_topk_training(
    x,
    topk_ids,
    topk_weights,
    interleaved_w1=interleaved_w1,
)

gradients = sonic_moe_backward(
    x, w1, w2, topk_ids, topk_weights, grad_output, config,
    b1=b1, b2=b2, interleaved_w1=interleaved_w1,
    forward_state=state,
)
```

Stage 1 keeps its existing sorted activation output for Stage 2 and scatters
the exact rounded gate/up values into an invocation-owned contiguous BF16
`[T, K, 2I]` tensor.  Backward independently sorts routes, gathers that compact
state into its sorter order while regenerating activation and `dy`, and skips
both grouped and generic W1 recomputation.  `forward_state=None` retains the
standalone implementation.  FP16, non-SwiGLU, ragged routing, and unsupported
training-forward combinations continue to use an explicit fallback.

The state is immutable and includes shape/layout/dtype, bias-presence, producer
stream, and ready-event metadata.  It never aliases the reusable forward
workspace.  The adapter saves the state tensor through `save_for_backward`, so
version checks, saved-tensor hooks, `retain_graph=True`, and tensor lifetime keep
their PyTorch semantics.  Graph capture remains explicitly rejected.

The implementation structurally validates a supplied state.  Semantic source
identity remains the same-invocation hot-path precondition: a state must be
paired with the exact `x`, W1, B1, and route IDs that produced it.  Proving that
identity in the low-level API would require retaining or hashing large inputs
or adding synchronization.

### Eager performance

The table uses an AMD Instinct MI355X (`gfx950`), PyTorch
`2.13.0+rocm7.14.0`, BF16, real expert-major `MoE` leaf parameters exposed
through the public `permute(1, 2, 0)` views, two warmups, and 11 AB plus 11 BA
pairs.  JIT and weight preparation are excluded.  The baseline is the same
tree with only the training-state call disabled.  “Full” starts from fixed
route IDs/scores and includes adapter forward plus backward; it excludes router
logits and top-k.  Times are event medians.

| Bucket | Compact state | Forward baseline/state | Backward baseline/state | Full baseline/state | Full reduction |
|---|---:|---:|---:|---:|---:|
| T1, H3584/I512/E896/K16 | 32 KiB | 0.118 / 0.151 ms | 1.765 / 1.733 ms | 1.874 / 1.849 ms | 1.31% |
| T128 balanced, H3584/I512/E896/K16 | 4 MiB | 1.509 / 1.545 ms | 6.567 / 5.424 ms | 7.860 / 6.761 ms | 13.98% |
| T128 hot16, H3584/I512/E896/K16 | 4 MiB | 0.243 / 0.256 ms | 2.211 / 2.132 ms | 2.344 / 2.288 ms | 2.38% |
| T4096, H4096/I2048/E64/K8 | 256 MiB | 1.645 / 2.305 ms | 11.453 / 7.948 ms | 12.995 / 10.095 ms | 22.32% |

The forward store is deliberately visible in these numbers: it costs about
0.66 ms for the 256 MiB T4096 state, but removing W1 from backward saves about
3.51 ms.  Across all four buckets, the complete eager step is faster.  Relative
L2 differences versus the standalone recompute path are at most `3.91e-4`, and
the largest gradient absolute difference is `2.39e-7`.

Against the ROCm Triton retained-forward backward on the same device and
module-view layout, FlyDSL's state path measures 1.733/5.424/2.132/7.948 ms for
T1/T128-balanced/T128-hot16/T4096 versus Triton's
9.562/19.150/10.798/15.684 ms, or 5.52x/3.53x/5.06x/1.97x faster.  Triton already
retains routing metadata while this FlyDSL phase still re-sorts, so metadata
reuse remains a material opportunity rather than an accounting advantage.

### gfx950 compact-row preparation

The no-bias BF16/SwiGLU fixed-K path for 64--128 tokens already builds a BM16
device descriptor queue for W1/dX.  Backward now reuses that queue to fuse the
live-row hidden-state gather, activation reconstruction, and routed `dy`
preparation.  It reads `grad_output` directly in token order for both `dy` and
`dscore`, and the activation derivative reads the compact route-order state
directly.  Consequently this path does not allocate or materialize the padded
`dout_sorted[P64,H]` or `preactivation[P64,2I]` tensors.  Legacy, bias, ragged,
long-token, and standalone paths are unchanged.

For balanced `T128/H3584/I512/E896/K16`, only 2,048 routes are real while the
BM64 sorter extent is 57,344 rows.  The two removed tensors account for
528,482,304 bytes (504 MiB) of allocation, and the old gather plus state
prepare plus derivative measured about 0.69 ms.  With the compact-row path,
the 11-pair event median for retained-state backward is 4.809 ms, down from the
phase-1 5.424 ms (11.3%); full adapter forward plus backward is 6.143 ms, down
from 6.761 ms (9.1%).  T1 deliberately keeps the original row kernels because
its corresponding work is only about 18 us and the descriptor queue is not
built there.

## Standalone baseline data flow and redundant work

The compatibility autograd function currently does the following:

```text
_FlyDSLExpertFunction.forward
  -> _run_flydsl
     -> operator.forward_topk(...)
        -> sort
        -> W1 + activation             workspace.intermediate[P, I]
        -> W2 + weighted scatter       caller-owned output[T, H]
  -> save(x, w1, w2, ids, scores, b1, b2)

_FlyDSLExpertFunction.backward
  -> sonic_moe_backward(...)
     -> histogram + sort               new metadata
     -> gather x and dout
     -> W1                              preactivation[P64, Q]
     -> activation + dout*score         activation[P64, I], dy[P64, H]
     -> W2                              projection[P64, H], only for dscore
     -> dW2, dA, activation derivative, dW1, dX, reductions
```

Here `Q = 2I` for a GLU and `Q = I` for a pointwise activation.  `P` is the
forward-sort padded row count.  `P64` is independently padded to backward's
current fixed sort unit of 64.

`gemm1.py` already has a `store_preactivation=True` specialization.  It stores
gate and up separately as A16 and deliberately does not apply the activation.
The regular forward specialization instead rounds gate/up through A16 when
`round_preact_bf16=True`, applies the activation in FP32 to those rounded
values, and stores an A16 activation.  A dual-output training epilogue can
therefore produce both values without another GEMM and without changing the
forward rounding boundary:

```text
g16 = A16(acc_gate + bias_gate)
u16 = A16(acc_up   + bias_up)
saved_preact = [g16, u16]
activation16 = A16(activation(FP32(g16), FP32(u16)))
```

The existing `store_preactivation` boolean is mutually exclusive with the
activation output.  It should become an independent optional preactivation
pointer/compile-time store flag; the ordinary stage-1 output remains the
activation consumed by stage 2.

## What backward actually needs

| Backward result/stage | Required forward values | Values that need not be saved |
|---|---|---|
| `dy = A16(dout * route_weight)` | sorted token/route mapping and sorted weights | x, preactivation, projection |
| dW2 | activation and `dy`; activation can be regenerated from preactivation | projection |
| dA | W2 and `dout`/route weight | forward projection |
| activation derivative (`dZ`) | preactivation and dA | activation, projection |
| dW1 | dZ, x, expert segment metadata | activation, projection |
| dX | dZ, W1, token mapping | activation, projection |
| db1/db2 | dZ/dy and expert segment metadata | activation, projection |
| route-score gradient | `dout`, W2, activation, optional b2 | materialized projection, if fused with dA |

For one sorted route `r` assigned to expert `e`, forward computes

```text
y_r = activation_r @ W2_e^T + b2_e
out[token(r)] += score_r * y_r .
```

Therefore

```text
q_r      = dout[token(r)] @ W2_e
dscore_r = dot(q_r, activation_r) + dot(dout[token(r)], b2_e)
dA_r     = score_r * q_r .
```

The grouped dA kernel already owns the `dout @ W2` accumulators.  Its epilogue
can multiply them by the route weight for dA and reduce their dot product with
the activation for dscore.  With the current `BN=64` split, each N tile can
atomically add one FP32 partial per real route after a device-side zero, or a
future row-owned schedule can loop over all N tiles and write the scalar once.
The optional bias term is a separate `H` reduction or a fused contribution in
a row-owned dA schedule.

This changes the placement of A16 rounding relative to today's explicit
`dy=A16(dout*score); dA=GEMM(dy,W2)` and today's
`projection=A16(activation@W2^T+bias); dscore=dot(dout,projection)`.  It matches
the algebraic/upstream fused formulation, but exact bitwise identity is not
expected.  Correctness should use the existing BF16/FP16 tolerance contract.
If bitwise compatibility with the current standalone backward is required, the
only general solution is to save or recompute the rounded projection; an
optional saved-projection validation mode is specified below.

## State ownership and API

Do not save `SonicMoEWorkspace` or aliases of any tensor owned by it.  Workspaces
are keyed by `(device, stream, token_count, route_count)` and reused.  The
per-workspace Python lock serializes enqueue, not tensor lifetime: a later
forward with the same key overwrites `sorted_*`, `num_valid_ids`, and
`intermediate` long before an earlier autograd graph is necessarily consumed.

Add a training-only entry point while preserving the inference return type:

```python
output, state = operator.forward_topk_training(
    x,
    ids,
    scores,
    out=output,
    save_policy="preactivation",
    state_buffers=None,       # required during graph capture
)

grads = sonic_moe_backward(
    x, w1, w2, ids, scores, grad_output, config,
    b1=b1, b2=b2,
    forward_state=state,      # None keeps the existing standalone path
)
```

The analogous `forward_routes_training`/`sonic_moe_backward_routes` extension
should share the representation.  A lightweight frozen state descriptor should
contain only invocation-owned tensors plus immutable scalar metadata:

```text
sorted_token_ids       int32 [capacity]       (token + fixed-K slot when applicable)
sorted_route_ids       int32 [capacity]       (ragged routes only)
sorted_weights         fp32  [capacity]
sorted_expert_ids      int32 [block_capacity] (padded layout only)
expert_offsets         int32 [E+1]            (compact layout only)
num_valid_ids          int32 [2]
expert_frequency       int32 [E]
preactivation          A16   [capacity, Q]     (policy-dependent)
activation             A16   [capacity, I]     (policy-dependent)
projection             A16   [capacity, H]     (strict/debug policy only)
layout_kind            direct_slots | padded_sorted | compact_sorted
sort_unit, tokens, routes, top_k, config_signature
producer_stream, ready_event                         (eager cross-stream use)
```

Sorting scratch can remain workspace-owned.  The final metadata must be written
directly to the invocation-owned state (preferred), or copied into it before the
workspace may be reused.  Stage 1 similarly writes its optional preactivation
directly to state.  Under the activation-saving policy, the normal stage-1
activation output itself is invocation-owned and stage 2 consumes that tensor;
there is no extra activation copy.

The state is immutable after the ready event.  Backward may allocate and mutate
its own scratch (`x_sorted`, `dout_sorted`, `dy`, `dA`, `dZ`, gradient outputs,
and scheduling queues), but must never use a saved-state tensor as output.

### Adapter integration and saved-tensor checks

`_run_flydsl` should return `(output, state)` only on the training path.
`_FlyDSLExpertFunction.forward` must flatten every tensor in `state` into
`ctx.save_for_backward(...)`, together with the existing user tensors.  Only
non-tensor shape/layout tags belong on ordinary `ctx` attributes.  Reconstruct
the lightweight descriptor from `ctx.saved_tensors` in backward.

This is important for three reasons:

1. PyTorch checks version counters when `ctx.saved_tensors` is unpacked, so an
   in-place edit of x, weights, scores, biases, or an exposed state tensor fails
   rather than silently producing a stale gradient.
2. saved-tensor hooks (including offload hooks) see the state tensors.
3. tensor lifetime follows the autograd graph, so `retain_graph=True` and a
   second backward remain valid.  A global pool cannot safely reclaim a state
   slot after the first backward because the same graph may be re-entered.

Raw-pointer writes do not increment a PyTorch tensor's version counter.  Version
checking is therefore not a substitute for unique ownership: a reused workspace
could silently overwrite saved bytes while retaining the same version.  Unique
invocation storage is the eager-mode correctness mechanism.

The adapter should continue saving the original user tensors even when backward
could consume only their forward copies.  In particular, mutating `w1`, `w2`,
or route scores between forward and backward must retain the current PyTorch
error behavior rather than silently differentiating a snapshot.

## Sort layout compatibility

Forward and backward do not currently have the same metadata ABI:

- forward pads to `route_tile_m = lcm(stage1_tile_m, stage2_tile_m)`, commonly
  16 for the high-E decode/sparse profiles and 128 for the throughput profile;
- standalone backward always pads to 64;
- the direct T=1 forward path emits one block per top-k slot in route order,
  while several backward schedulers binary-search an ascending expert-block
  list.

Passing forward metadata into the current backward unchanged would therefore
be incorrect.

For the first optimized BF16 SwiGLU path, make `sort_unit` part of each compiled
backward launcher's cache key and state validation.  Device-driven dA/dX/dW
kernels must consume `state.sort_unit`; their compute BM must divide it.  The
current production pairs satisfy this (`16 -> BM16`, `128 -> BM64/128`).  Replace
host reconstruction of segment offsets with `expert_frequency`, device queues,
or expert offsets from the state.

Direct T=1 state should be tagged `direct_slots`.  T1-specialized backward can
launch from the block/route descriptors directly and must not binary-search as
if expert IDs were ascending.  Disabling direct T=1 sorting would be simpler but
would give back much of the measured forward routing win, so it should only be
a temporary correctness fallback.

FP16, non-SwiGLU, and any generic GEMM path that still requires 64-row
contraction alignment may keep the independent sorter initially.  Reusing only
part of an incompatible state is preferable to silently interpreting it as
BM64.  The long-term compact ABI below removes this coupling.

## Save policies

### 1. Metadata only

Save routing metadata and frequencies, but leave stage-1 activation in the
reusable forward workspace.

- Eliminates the backward histogram and sort.
- Retains W1, activation, and (until dA/dscore fusion lands) W2 recomputation.
- Smallest integration step and useful for validating state ownership, streams,
  and layout plumbing.

### 2. Metadata + preactivation (recommended default)

Add A16 `[capacity, Q]` preactivation written by the forward stage-1 epilogue.

- Eliminates W1 recomputation.
- Backward regenerates A16 activation with an elementwise kernel and combines
  that pass with `dy=A16(dout*score)`.
- After fused dA/dscore is enabled, eliminates W2 recomputation without saving a
  projection.
- Matches upstream SonicMoE's decision to retain preactivation rather than both
  stage outputs.

### 3. Metadata + preactivation + activation

Make the normal forward activation buffer invocation-owned and save it too.

- Eliminates W1 and activation recomputation; only the `dy` portion of today's
  activation-prepare kernel remains.
- Adds no extra forward store beyond the store stage 1 already performs, but
  extends the lifetime of `2 * capacity * I` bytes.
- Worth enabling through a policy/autotune threshold when the extra residency
  is cheaper than the backward activation pass.  It should not be assumed to
  win for the large T4096 shape.

### Optional strict projection

Saving unweighted A16 projection adds `2 * capacity * H` bytes in padded layout
or `2 * routes * H` bytes in compact layout.  It permits the existing score
kernel to run without W2 recompute and preserves its current rounded-projection
numerics.  This is useful as an A/B oracle and perhaps for an application that
already needs per-route output, but is too large for the default policy.

Policy can also respect `ctx.needs_input_grad`: no route-score gradient means no
projection or fused score reduction; no W2 gradient means activation is needed
only by score fusion; and no x/W1/b1 gradient means preactivation need not be
retained.  The selected mask must be static for graph capture.

## Memory cost

For fixed-K padded state with capacity `C`, padding unit `B`, and GLU
preactivation (`Q=2I`):

```text
metadata ~= 8C + 4(C/B) + 4E + 8 bytes
preactivation = 4CI bytes
activation    = 2CI bytes
projection    = 2CH bytes
```

Metadata includes packed token/slot IDs, FP32 weights, one expert ID per route
block, frequencies, and two counters.  For a compact state of exactly `R=T*K`
real rows, packed token/slot IDs plus weights and `E+1` expert offsets cost
approximately `8R + 4(E+1) + 8` bytes.

The table reports MiB.  `C_fwd` is the distribution-independent allocation
bound used without a host synchronization.  `C_64` shows today's independent
BM64 backward allocation.  A compact state is a future exact-`R` layout.

| Shape | Layout/capacity | Metadata | Preactivation | Activation | Projection | Metadata + preact + activation |
|---|---:|---:|---:|---:|---:|---:|
| T1/H3584/I512/E896/K16 | forward B16, C=256 | 0.005 | 0.500 | 0.250 | 1.750 | 0.755 |
|  | current B64, C=1024 | 0.011 | 2.000 | 1.000 | 7.000 | 3.011 |
|  | compact R=16 | 0.004 | 0.031 | 0.016 | 0.109 | 0.050 |
| T128/H3584/I512/E896/K16 | forward B16, C=15488 | 0.125 | 30.250 | 15.125 | 105.875 | 45.500 |
|  | current B64, C=58496 | 0.453 | 114.250 | 57.125 | 399.875 | 171.828 |
|  | compact R=2048 | 0.019 | 4.000 | 2.000 | 14.000 | 6.019 |
| T4096/H4096/I2048/E64/K8 | forward B128, C=40832 | 0.313 | 319.000 | 159.500 | 319.000 | 478.813 |
|  | current B64, C=36800 | 0.283 | 287.500 | 143.750 | 287.500 | 431.533 |
|  | compact R=32768 | 0.250 | 256.000 | 128.000 | 256.000 | 384.250 |

Actual used padded rows can be lower than capacity.  For T128 balanced routing,
forward B16 uses 14,336 rows (28 MiB preactivation and 14 MiB activation); hot16
uses only 2,048 rows (4 MiB and 2 MiB).  A graph/eager implementation that must
allocate before reading device counts still reserves C=15,488 for either
distribution.  This makes an exact-R compact state especially attractive for
high-E sparse routing.

Compact state requires more work than trimming a tensor view: expert padding is
interleaved between segments.  A training sorter can emit compact expert offsets
and compact row IDs, allowing stage 1 to store saved preactivation directly at
the real-row destination while retaining its padded activation for forward.
Ultimately, backward grouped kernels should use those offsets as a varlen
schedule, matching upstream SonicMoE.  A post-forward compaction kernel is a
valid intermediate implementation but its extra `O(RI)` traffic must be included
in end-to-end timing.

## Streams, overlap, and re-entry

The following rules are required, not optional optimizations:

1. Every eager forward invocation owns distinct saved tensor storage.  Two
   same-shape forwards may share sorter scratch but never saved metadata or
   preactivation/activation storage.
2. Keep the workspace launch lock around sort, both GEMMs, and ready-event
   recording.  It guarantees a complete same-key enqueue sequence.  Same-stream
   ordering then makes reuse of workspace-only scratch safe.
3. Different streams continue to receive different workspaces.  State tensors
   are invocation-owned regardless of stream.
4. Record a per-invocation ready event after the last forward state write.  The
   autograd path normally executes backward work on the corresponding forward
   stream, but standalone backward accepting `forward_state` must wait on this
   event when its current stream differs from `producer_stream`.
5. `record_stream` is needed for every tensor passed through a raw pointer on
   the stream that consumes it: inputs, prepared weights, output, saved state,
   and backward scratch/results.  `record_stream` protects allocator lifetime;
   it does not establish an execution dependency, so it does not replace the
   event wait.
6. Backward is read-only with respect to saved state.  A second backward with
   `retain_graph=True`, nested/re-entrant autograd, or reverse-order backward of
   overlapping forwards therefore observes the same bytes.

An object pool with a slot returned after the first backward does not satisfy
rule 6.  Prefer allocator-owned tensors whose references live in the autograd
graph.  Introduce pooling only with an explicit lease whose lifetime is the
saved graph, not the backward call.

## CUDA/HIP graph capture

The compatibility adapter currently rejects graph capture, and backward also
performs `expert_frequency.cpu().tolist()`.  Forward-state reuse alone does not
make the path graph-safe.  Enable capture only after all of these conditions are
met:

- prepared weights, operator, compiled launchers, workspace, saved-state slot,
  gradient outputs, and device scheduling queues are warmed up and allocated
  outside capture;
- capture performs no tensor/event allocation, LRU insertion/eviction, JIT,
  `.item()`, or CPU frequency readback;
- all scheduling and active/inactive-expert initialization is device-driven;
- bias reductions and every dtype/activation path admitted to capture have a
  device-driven grouped implementation;
- pointers and the `needs_input_grad`/save-policy mask are fixed for the graph.

A capture-safe API should require an explicit `SonicMoETrainingBuffers` slot.
For the simplest supported contract, capture forward and backward together
(for example through a graphed callable) and bind one private slot to that graph.
Each replay overwrites the slot only after the preceding replay's backward has
consumed it.

Capturing forward alone and running backward later is not safe with one static
slot.  Neither are overlapping replays that share the slot.  Support those
modes only with graph-private slots/rings and an explicit lease/release protocol,
or reject them.  A separate captured graph executable per slot is the clearest
fixed-address implementation.

Raw-pointer replay writes do not bump version counters, so saved-tensor version
checks cannot detect a graph slot overwritten by a later replay.  The graph-slot
lifetime contract is what guarantees correctness.

## Implementation sequence

1. **Dual stage-1 epilogue (complete).** The training entry point saves exact
   compact route-order A16 preactivation while preserving the existing
   activation/output path.  Backward validates the state, skips W1 recompute,
   and retains `forward_state=None` as the tested standalone fallback.
2. **Exact-row state consumers (complete for compact hostless).** Reuse the
   existing BM16 device queue to fuse gather/activation/`dy`, read state
   directly in the derivative, and remove the padded sorted dout/preactivation
   tensors.  Decode keeps its measured lower-latency row kernels.
3. **Metadata state (next).** Make forward emit invocation-owned routing
   metadata and frequency, then skip backward histogram/sorting.  Reconcile
   forward's B16/B128 and backward's B64 layouts without a host readback.
4. **Variable sort-unit consumers.** Parameterize grouped dA/dX/dW schedulers by
   the state sort unit and add direct-slot T1 scheduling.  Remove assumptions
   that every saved layout is ascending BM64.
5. **Fused dA/dscore.** Compute unscaled `dout @ W2`, form dA and score partials
   in its epilogue, and add the bias contribution.  Delete projection allocation
   and W2 recompute from this path.  Keep saved-projection mode as a numerical
   oracle until tolerances and performance are established.
6. **Policy tuning.** Compare preactivation-only with saved activation.  Measure
   the extra forward stores, forward latency, backward latency, peak allocated
   memory, and full forward+backward time on T1, T128 balanced/hot16, and T4096.
7. **Compact state.** Emit exact-R expert offsets/mappings and teach grouped
   backward kernels the varlen layout.  This is highest priority for E896 sparse
   routing, where padded-state residency dominates.
8. **Capture mode.** Remove host frequency decisions, provide preallocated
   graph slots, and enable only paired forward+backward capture first.

Each step should be separately guarded so `forward_state=None` remains a tested
fallback and inference performance is unchanged.

## Required tests

- Compare all returned gradients against the standalone path for BF16/FP16,
  every activation, bias/no-bias, fixed-K and ragged routes, duplicate ragged
  routes, zero-route tokens, and boundary expert counts.  Optimized support may
  land incrementally, but unsupported combinations must fall back explicitly.
- Run forward A, forward B with the same operator/key, then backward A; repeat in
  reverse backward order and verify that A's state was not overwritten.
- Repeat the overlap test across two streams, and exercise standalone backward
  on a third stream without a host synchronization.
- Run backward twice with `retain_graph=True`, including re-entrant autograd from
  a hook, and compare gradients.
- Mutate each saved user input with a legal `no_grad` in-place update between
  forward and backward and assert PyTorch's saved-tensor version error.
- In a test-only exposed-state path, mutate a saved metadata/preactivation tensor
  through PyTorch and assert the same version error.
- Stress allocator reuse after dropping local forward references; this catches
  missing `record_stream` calls around raw-pointer launches.
- Test each state layout tag and reject mismatched sort units, shapes, devices,
  dtypes, operator configs, and forward generations before launching kernels.
- After capture prerequisites land, warm up outside capture, capture paired
  forward+backward, replay with changing inputs, and compare eager results.
  Also verify that forward-only capture and overlapping replay are either given
  distinct slots or rejected deterministically.
- Benchmark stage timings and end-to-end time, not only isolated removed GEMMs;
  a saved state is a win only after accounting for its forward stores and memory
  pressure.

## Acceptance criteria

- No training saved tensor aliases `SonicMoEWorkspace` storage.
- Overlapping forwards and re-entrant backward are deterministic within the
  existing fixed/ragged reduction guarantees.
- The implemented phase-1 BF16 SwiGLU state path launches no W1 recompute.
- After fused dA/dscore lands, the optimized path also launches no W2
  projection recompute.
- Inference entry points and their workspace reuse remain unchanged.
- Standalone backward without a state remains available and numerically tested.
- Graph capture remains explicitly rejected until the complete preallocation
  and device-driven checklist is satisfied; once enabled, unsafe slot reuse is
  rejected rather than silently accepted.
