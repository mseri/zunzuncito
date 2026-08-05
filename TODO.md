# TODO

## Metal for the ternary kernel

`gpu.h` speaks q4_0 and q8_0. `maple` is ternary throughout, so `--metal` currently
accepts the flag and declines, and the whole model runs on the CPU.

The shape of the work is a `tq2` kernel alongside the two in `metal.mm`, plus a `q4a`
one for the lm_head and the FlashHead centroids. Ternary is the easiest case that
backend will ever get: the codes unpack with two shifts and a mask, there is no
per-block scale to decode, and the row reduces to an integer accumulation times one
f32, so a threadgroup per output row with a simdgroup reduction is close to the whole
kernel.

Whether it is worth building is the open question, and the same one `gpu.h` already
argues about. The README's numbers are on an M1 where 4 threads beat 8 (`--ram 4`:
34.2 tok/s at 4, degrading past that), which says the CPU path is already near its
memory roofline rather than compute-starved. Prefill is the honest target: at a
128-token batch it is genuinely compute-bound and runs at 49 tok/s, so that is where a
dispatch amortises. Measure prefill, not decode, before deciding.

## Router bytes

At `--ram 4` with the FlashHead on, a decode step reads roughly 63 MiB of attention
weights, 150 MiB of experts, 24 MiB of head, and then 50 MiB of router, because the
router is f32 at 256 x 2048 per layer across 24 layers. That is 17% of the step for
0.05% of the parameters.

Both sibling engines keep the router f32 on the grounds that a routing error is not a
small numeric error, it picks a different expert, and Maple's own config sets
`router_dtype: fp32` for exactly that reason. But fp32 is a statement about the
*accumulation*, not the storage: the checkpoint ships these weights as bf16 and we
widen them at conversion time. Storing them bf16 and widening in the kernel would halve
the bytes with no change to the arithmetic that the config is actually asking for.

Worth about 25 MiB/token, so ~8% of the step. Needs a bf16 matvec that accumulates in
f32, and a check that top-8 selection does not move on a real prompt set.

## Partial q3 experts, and the kernel that needs

For low-RAM systems (`--ram 4` and below), where the engine is disk-bound and
bytes-per-token is the only lever that matters.

### Where the idea comes from

[WASTE](https://github.com/sqliteai/waste)
([writeup](https://marcobambini.substack.com/p/the-waste-inference-engine)) runs Kimi
K3 (2.78 T params in a 982 GB container) on a 64 GB MacBook at ~0.5 tok/s, on the
same bet this engine makes: trunk resident, experts streamed from NVMe, the rest of
RAM as a bounded expert cache. Most of its machinery we already have, and in places
have better. It bypasses the page cache (so do we, `F_NOCACHE`). It reads one
positional record per expert (so do we, 4096-aligned, one `pread`). It lists
"improving prefetching and cache persistence" as future work, while gemma4 already
routes off the raw post-attention residual, so its prefetch is exact rather than
predicted, and both engines persist a pin set in `usage.bin`.

What it has that we don't is a 3-bit expert tier: residual vector quantisation, with
the matvec done from the RVQ representation via per-codebook partial dot-product
tables so the expert matrix is never materialised. That's the one transferable win,
and squarely a low-RAM feature: fewer bytes per expert means both less I/O per token
and more experts resident per GB of cache.

### Why partial

Extend the apex-quant gradient downward by one tier rather than switching wholesale:

| tensors | today | proposed |
|---------|-------|----------|
| attention, conv, dense MLP (always on) | q8_0 | q8_0, unchanged |
| routed experts, edge layers | q8_0 | q8_0, unchanged |
| routed experts, middle layers | q4_0 | q3 |
| norms, router, expert bias | f32 | f32, unchanged |

The trunk boundary isn't a guess. WASTE measured a 3-bit trunk: it freed over 6 GB
and raised the cache hit rate, and it still lost, partly on unpacking cost but mainly
because generation quality collapsed. K3's QAT covered its experts and not the rest
of the model. Our always-on tensors are q8_0 for the same reason, so leave them be.

`--expert-edge` keeps its current meaning; q3 applies only to the middle MoE layers
it already excludes.

### What it buys

Container sizes at 3.5 bit/weight (a plain 3-bit block codec: 32 weights, fp16 scale,
12 bytes of codes) and at 3.0 bpw (RVQ, no per-block scale):

| | today | 3.5 bpw | 3.0 bpw |
|---|---|---|---|
| gemma4 experts (all q4_0) | 12.9 GB | ~10.0 GB | ~8.6 GB |
| lfm25 experts (`--expert-edge 2`) | 4.72 GiB | ~3.98 GiB | ~3.61 GiB |

This does not apply to `maple`, whose experts are already ternary at 2.0 bpw and were
trained that way. Going below it is not a quantisation choice left to make.

lfm25 gains less in relative terms because only 18 of its 22 MoE layers are q4_0 to
begin with. The four q8_0 edge layers are 30% of the expert bytes and stay put.

This doesn't make either model resident at 4 GB. It buys roughly 16–22% fewer bytes
per token and a proportional rise in slots/layer, which matters where the hit rate is
low (`gemma4 --ram 4`, `lfm25 --ram 4`) and not at all where it is already around 90%
(`lfm25 --ram 8`, which is compute-bound rather than disk-bound).

### What it costs, measured

Weight reconstruction error, rel-rms, on Gaussian-ish random weights at I=2048. These
show the shape of the penalty, not what real tensors will do:

```
q4_0  sym  blk32   4.5 bpw : 0.0800   <- shipping today
q3    sym  blk32   3.5 bpw : 0.1635   <- 2.05x
q3   asym  blk64   3.5 bpw : 0.1789   (RTN min/max)
q3   asym  blk64   3.5 bpw : 0.1671   (HQQ-style, 8 iters)
q3   asym  blk32   4.0 bpw : 0.1472   (HQQ-style, 8 iters)
```

No calibration-free scheme closes that gap. Asymmetric coding with a zero-point comes
out worse than plain symmetric at a matched bit budget, because the coarser block it
forces costs more than the zero-point gains on near-symmetric zero-mean weights, and
HQQ-style refinement only claws it back. Even at 4.0 bpw, half a bit under q4_0, it
is still 1.84x. The binding constraint is the 3-bit codebook, not the parameter fit.
Beating it needs activation calibration (GPTQ/AWQ, offline, needs a calibration set)
or QAT in the checkpoint. K3 had expert QAT. Neither Gemma-4's expert tier nor LFM2.5
gives us that, so this is a real quality gamble, and it wants a perplexity measurement
on the real checkpoints before it ships rather than a rel-rms proxy.

Do it in the converter, not at runtime. Requantising an existing q4_0 container to 3
bits measured 0.2113, which is 1.29x the error of going direct from the checkpoint,
for no benefit beyond not re-reading the safetensors. `tools/convert_*.py` already
reads bf16 and already dispatches per-tensor formats, so `--expert-bits {3,4}` there
costs nothing extra. ISQ-style in-situ quantisation does not apply here: its output
lives in RAM, ours has to live on disk in expert-contiguous layout, so the persisted
container already is the in-situ result, computed once instead of every load. If a
requant-from-container path is wanted purely as a convenience for someone who deleted
their checkpoint, it is cheap (~51 ms/expert, ~39 s for lfm25's whole set,
single-threaded scalar), but it has to be an eager pass rather than lazy on cache
miss, and documented as costing that 1.29x.

### The kernel

A 3-bit unpack straddles byte boundaries (8 codes per 3 bytes) rather than splitting
cleanly like q4_0's nibbles, so this is where the LUT / bit-serial approach
(T-MAC-style, and WASTE's own partial-dot-product tables) becomes worth building, and
only here.

At 4 bits it isn't worth it, and we measured that rather than assuming it. The q4_0 matvec
runs at ~20% of the streaming roofline, so there is arithmetic headroom, but the
binding cost turned out to be the accumulator dependency chain rather than the op
count: hoisting the redundant per-row `sum(x)` out of the loop, about 15% of the
inner-loop uops, gave 1.01x, while four independent accumulators gave 1.12–1.21x
and is what `q40.h` now does. Both target ISAs already do the int8 dot in about one
instruction (AVX2 `maddubs`+`madd`, ARMv8.2 `sdot`), which is exactly the situation a
LUT doesn't improve. At 3 bits that changes: the unpack gets expensive, and the table
amortises well since it is built once per activation vector and reused across
O = 1792..2048 rows.

Two costs to weigh when building it. It breaks the bit-identical-to-`block_q4_0`
property `q40.h` calls out as a feature, where GGUF expert tensors `memcpy` straight
in; that's for the q3 tier only, but the file stops being uniformly GGUF-compatible.
And `pshufb`-based tables have to be int8-quantised, which stacks a second activation
quantisation error on top of the ~1.3e-2 the int8-activation build already carries.

### Don't bother with

Measured and rejected, either by WASTE or here. Recorded so nobody spends a week
rediscovering them.

Per-expert importance-weighted bit allocation. `usage.bin` makes this tempting. WASTE
measured quantisation error as near-uniform across experts and layers, with optimised
allocation beating random by 1.01–1.15x, and since the low-precision experts would
be the rarely-routed ones, it saves disk capacity rather than bytes read. Disk
capacity isn't our constraint either. (This doesn't touch `--expert-edge`, which is
per-layer and quality-motivated.)

Metal for decode. WASTE implemented and verified a Metal backend and measured it 22%
slower than CPU on this workload; `gpu.h` already argues the same for the 4–8 GB case.

Trusting a microbenchmark. WASTE got a 1.44x isolated win from an index-layout change
that produced no end-to-end gain, because the microbenchmark did not model threads
sharing cache. Measure the engine.
