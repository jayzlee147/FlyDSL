# SonicMoE grouped dX tuning on gfx950

This note records an isolated tuning study of the grouped dX contraction

```text
dZ[sorted M, 2I] @ W1[expert, 2I, H] -> dX_sorted[sorted M, H]
```

on an AMD Instinct MI355X (`gfx950`).  It compares every recommendation with
the configuration selected by commit `e0656bd`.  The study does not change a
production kernel or include sorter/descriptor-builder latency.

## Reproduction

The measurements used `rocm/pytorch:latest`, PyTorch
`2.13.0+rocm7.14.0`, and HIP `7.14.60850`.  The benchmark constructs the same
sorter metadata or compact descriptor layout consumed by
`compile_sonic_grouped_a16w16_nn`, checks sampled rows against PyTorch when
`--check` is present, and records raw event samples plus compiler resources in
JSON.

```bash
PYTHONPATH=. python tools/bench_sonic_grouped_dx.py \
  --cases t128-balanced \
  --config 16,256,64,4 --config 16,512,64,4 \
  --grid-caps 1024 --warmup 5 --reps 51 --check \
  --output /tmp/dx-balanced.json

PYTHONPATH=. python tools/bench_sonic_grouped_dx.py \
  --cases t128-hot16 \
  --config 16,128,64,2 --config 64,64,64,4 \
  --grid-caps 1024 --warmup 7 --reps 101 --check \
  --output /tmp/dx-hot16.json
```

Run each command once in the listed order and once with the two `--config`
arguments reversed.  The table below averages the two order-specific medians;
the individual medians are retained to expose drift.

## Results against the current configuration

| Routing bucket | Current BM/BN/BK/NW | Current ms (AB/BA) | Candidate BM/BN/BK/NW | Candidate ms (AB/BA) | Reduction |
|---|---:|---:|---:|---:|---:|
| T1, 16 active | 16/128/64/2 | 0.0215 in the tile sweep | unchanged | unchanged | no repeatable gain |
| T128 balanced, 896 active | 16/256/64/4 | 1.1325 / 1.1356 | 16/512/64/4 | 1.1149 / 1.1131 | 1.77% |
| T128 hot16, 16 active | 16/128/64/2 | 0.0608 / 0.0612 | 64/64/64/4 | 0.0356 / 0.0362 | 41.06% (1.70x) |
| T4096 balanced, 64 active | production uses 64 general GEMMs | about 2.903 for the dX phase | 64/256/64/4 | 1.2571 isolated | 56.70% (2.31x) potential |

The T4096 comparison is not yet an end-to-end result.  The candidate passed
the sampled correctness check, but production integration must include and
time a new BM64 descriptor build before claiming the 1.646 ms saving.

## Why the buckets differ

For balanced T128 there are 2,048 routes and 896 active experts, each with two
or three real rows.  The BM64 sorter materializes 57,344 rows (`28x` route
padding), while the existing BM16 compact queue schedules 896 descriptors or
14,336 rows (`7x` executed-row padding).  The grouped kernel therefore avoids
three quarters of the sorter padding, but BM16 is already the smallest gfx950
MFMA M tile.  BN512 halves the logical workgroups from 12,544 to 6,272 and
produces the measured 1.77% improvement over the current BN256 kernel.

For hot16, every active expert has 128 real rows.  BM16 has no row padding, but
its eight descriptors per expert reload the complete W1 expert slab eight
times.  BM64 reduces this to two descriptors and two W1 slab reads per expert:
the modeled streamed W1 traffic falls from 0.875 GiB to 0.21875 GiB.  This is
the reason BM64/BN64 is 41% faster despite using the same arithmetic count.

T1 already uses one metadata block per active expert and is launch/weight
traffic dominated.  Raising BM only increases its existing `16x` row padding,
so the current BM16/BN128 kernel remains the correct choice.

## Grid and resource evidence

The persistent grid cap should remain 1024 unless a later end-to-end sweep
proves otherwise:

- Balanced BN512 was nearly flat from cap 256 through a full 6,272-WG grid
  (about 1.157--1.184 ms in the grid sweep); cap 128 was underfilled at about
  1.63 ms.  Its 132 KiB LDS footprint already limits it to one workgroup per
  CU, so cap 256 is sufficient in isolation but offers no material reason to
  specialize the production launch.
- Hot16 BM64/BN64 was most stable at cap 1024 (about 0.037 ms); cap 512 was
  about 0.045 ms and cap 256 about 0.064 ms.
- T1 needs its complete 448-WG logical grid.  A cap of 128 increased latency
  from about 0.025 ms to 0.041 ms.
- T4096 BM64/BN256 was best at cap 1024 (1.257 ms); its full 8,192-WG grid was
  about 1.421 ms.

Final ISA and compiler metadata show no private segment or SGPR/VGPR spills:

| Kernel | metadata VGPR | ISA next-free VGPR | accum offset | LDS |
|---|---:|---:|---:|---:|
| balanced current, BM16/BN256 | 74 | 169 | 76 | 68 KiB |
| balanced candidate, BM16/BN512 | 126 | 257 | 128 | 132 KiB |
| hot16 current, BM16/BN128 | 84 | 169 | 84 | 36 KiB |
| hot16 candidate, BM64/BN64 | 68 | 97 | 68 | 32 KiB |

`vgpr_count` from kernel metadata and `.amdhsa_next_free_vgpr` are intentionally
reported separately: the latter includes unified accumulator register
numbering and must not be interpreted as the same occupancy field.

## Hostless selection constraints

Only choices determined from public tensor shapes can be made by the host
after removing the expert-frequency D2H copy:

- T1 can keep BM16/BN128 statically; it needs no distribution statistic.
- The T4096 BM64/BN256 candidate can also be selected from the host-known token
  and dimension bucket.  A device builder must create its independent BM64
  queue, and the queue header can bound useful work without a D2H copy.
- The two T128 candidates are distribution-specific and cannot be selected
  safely from `T/H/I/E/K` alone.  Balanced and hot16 have identical public
  shapes.  Unconditionally selecting BN512 loses the hot case, while
  unconditionally selecting BM64 amplifies the many-short-expert case.

The current device-resident queues already expose descriptor and active counts,
but a robust hot predicate also needs a device-produced maximum segment length.
The recommended follow-up is for the builder to emit
`active_experts`, `max_expert_rows`, and separate BM16/BM64 queue counts.  Then
use the following policy entirely on device:

```text
T1:
    BM16 / BN128 / BK64 / NW2
T128 and max_expert_rows >= 64 and active_experts < 256:
    BM64 / BN64 / BK64 / NW4
T128 otherwise, when the dense/many-expert predicate is true:
    BM16 / BN512 / BK64 / NW4
fallback:
    current BM16 profile
T4096 production bucket:
    BM64 / BN256 / BK64 / NW4
```

Because BM/BN and LDS layouts are compile-time properties, removing the D2H
copy requires either guarded launches of the specialized kernels or a later
kernel design that consumes mixed-size descriptors.  A dual-launch scheme must
be benchmarked end to end: an empty launch is significant beside the 36 us hot
kernel.  Until that gate exists, keep the current hostless conservative profile
rather than applying either T128 candidate unconditionally.

## T4096 production integration

The fixed-K `T4096/H4096/I2048/E64/K8` shape now builds an independent BM64
descriptor queue and launches BM64/BN256/BK64/NW4 with a 1024-workgroup cap.
The selection uses only public tensor shapes.  The builder also emits the
active-expert queue already needed by dW1/dW2, so integration does not add a
host readback or a second active-expert scan.

An end-to-end backward AB/BA run on the same MI355X class device used two
warmups and 11 event-timed repetitions per ordering.  Baseline was the same
tree with the large-dX static gate disabled.  The smaller buckets execute
identical kernels and remain within measurement noise; T4096 improves by
12.4% (`1.142x`).

| Bucket | Baseline ms (AB/BA) | Integrated ms (AB/BA) | Result |
|---|---:|---:|---:|
| T1 | 1.6964 / 1.6949 | 1.6863 / 1.6835 | unchanged path |
| T128 balanced | 6.3097 / 6.3327 | 6.2822 / 6.3070 | unchanged path |
| T128 hot16 | 2.1074 / 2.0967 | 2.0977 / 2.0901 | unchanged path |
| T4096 balanced | 12.9251 / 12.8789 | 11.3130 / 11.2892 | 12.4% lower latency |

For the T4096 inputs, all four returned gradients were bitwise identical to
the generic-dX baseline.  A separate sampled kernel check measured the BM64
candidate at 1.374 ms and verified its partial-tile results.

## Production order

1. Add device-side distribution statistics and BM64 descriptors without a host
   readback.
2. Integrate the hot16 BM64 kernel behind that device predicate; require the
   launch overhead to preserve the approximately 25 us isolated saving.
3. Integrate balanced BN512 only if end-to-end AB/BA still shows a stable gain;
   its isolated 1.77% margin is small and its 132 KiB LDS footprint is a
   regression risk under concurrent workloads.
