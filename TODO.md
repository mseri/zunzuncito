# TODO

## Partial q3 experts, and the kernel that needs

For low-RAM systems (`--ram 4` and below), where the engine is disk-bound and
bytes-per-token is the only lever that matters.

### Where the idea comes from

[WASTE](https://github.com/sqliteai/waste) ([writeup](https://marcobambini.substack.com/p/the-waste-inference-engine))
runs Kimi K3 — 2.78 T params, a 982 GB container — on a 64 GB MacBook at ~0.5 tok/s,
on the same bet this engine makes: trunk resident, experts streamed from NVMe, the
rest of RAM as a bounded expert cache. Most of its machinery we already have, and in
places have better: it bypasses the page cache (we do, `F_NOCACHE`), reads one
positional record per expert (we do, 4096-aligned, one `pread`), and lists
"improving prefetching and cache persistence" as future work (gemma4 already routes
off the raw post-attention residual, so its prefetch is exact rather than predicted,
and both engines persist a pin set in `usage.bin`).

What it has that we don't is **3-bit experts** — residual vector quantisation, with
the matvec done from the RVQ representation via per-codebook partial dot-product
tables so the expert matrix is never materialised. That is the one transferable win,
and it is squarely a low-RAM feature: fewer bytes per expert means both less I/O per
token and more experts resident per GB of cache.

### Why *partial*

Extend the apex-quant gradient downward by one tier rather than switching wholesale:

| tensors | today | proposed |
|---------|-------|----------|
| attention, conv, dense MLP (always on) | q8_0 | q8_0, unchanged |
| routed experts, edge layers | q8_0 | q8_0, unchanged |
| routed experts, middle layers | q4_0 | **q3** |
| norms, router, expert bias | f32 | f32, unchanged |

The trunk boundary is not a guess. WASTE measured a 3-bit trunk: it freed >6 GB and
raised the cache hit rate, and it *still* lost, partly on unpacking cost but mainly
because generation quality collapsed — K3's QAT covered its experts and not the rest
of the model. Our always-on tensors are q8_0 for the same reason. Leave them alone.

`--expert-edge` keeps its current meaning; q3 applies only to the middle MoE layers
it already excludes.

### What it buys

Container sizes, computed at 3.5 bit/weight (a plain 3-bit block codec: 32 weights,
fp16 scale, 12 bytes of codes) and at 3.0 bpw (RVQ, no per-block scale):

| | today | 3.5 bpw | 3.0 bpw |
|---|---|---|---|
| gemma4 experts (all q4_0) | 12.9 GB | ~10.0 GB | ~8.6 GB |
| lfm25 experts (`--expert-edge 2`) | 4.72 GiB | ~3.98 GiB | ~3.61 GiB |

lfm25 gains less in relative terms because only 18 of its 22 MoE layers are q4_0 to
begin with; the four q8_0 edge layers are 30% of the expert bytes and stay put.

Note this does **not** make either model resident at 4 GB. It buys roughly 16–22%
fewer bytes per token and a proportional rise in slots/layer, which matters where the
hit rate is low (`gemma4 --ram 4`, `lfm25 --ram 4`) and not at all where it is
already ~90% (`lfm25 --ram 8`, which is compute-bound, not disk-bound).

### What it costs, measured

Weight reconstruction error, rel-rms, on Gaussian-ish random weights at I=2048.
These are indicative of the *shape* of the penalty, not predictions for real tensors:

```
q4_0  sym  blk32   4.5 bpw : 0.0800   <- shipping today
q3    sym  blk32   3.5 bpw : 0.1635   <- 2.05x
q3   asym  blk64   3.5 bpw : 0.1789   (RTN min/max)
q3   asym  blk64   3.5 bpw : 0.1671   (HQQ-style, 8 iters)
q3   asym  blk32   4.0 bpw : 0.1472   (HQQ-style, 8 iters)
```

Two things follow. **No calibration-free scheme closes the gap.** Asymmetric coding
with a zero-point is worse than plain symmetric at a matched bit budget (the coarser
block it forces costs more than the zero-point gains on near-symmetric zero-mean
weights), and HQQ-style refinement only claws it back. Even at 4.0 bpw — half a bit
under q4_0 — it is still 1.84x. The 3-bit codebook is the binding constraint, not the
parameter fit. Beating this needs activation calibration (GPTQ/AWQ: offline, needs a
calibration set) or QAT in the checkpoint. K3 had expert QAT. Neither Gemma-4's
expert tier nor LFM2.5 gives us that, so **this is a real quality gamble and wants a
perplexity measurement on the real checkpoints before it ships**, not a rel-rms proxy.

**Do it in the converter, not at runtime.** Requantising an existing q4_0 container
to 3 bits measured 0.2113 — 1.29x the error of going direct from the checkpoint, for
no benefit other than not re-reading the safetensors. `tools/convert_*.py` already
reads bf16 and already dispatches per-tensor formats, so `--expert-bits {3,4}` there
costs nothing extra. ISQ-style in-situ quantisation does not apply here: its output
lives in RAM, ours has to live on disk in expert-contiguous layout, so the persisted
container already *is* the in-situ result, computed once instead of every load. (If a
requant-from-container path is wanted purely as a convenience for someone who deleted
their checkpoint, it is cheap — ~51 ms/expert, ~39 s for lfm25's whole set,
single-threaded scalar — but it must be an eager pass, never lazy on cache miss, and
documented as costing that 1.29x.)

### The kernel

A 3-bit unpack is byte-straddling (8 codes per 3 bytes) rather than q4_0's clean
nibble split, so this is where the LUT / bit-serial approach (T-MAC-style, and WASTE's
own partial-dot-product tables) becomes worth building — and only here.

At 4 bits it is not worth it, which we measured rather than assumed: the q4_0 matvec
runs at ~20% of the streaming roofline, so there *is* arithmetic headroom, but the
binding cost was the accumulator dependency chain, not the op count (hoisting the
redundant per-row `sum(x)` out of the loop — ~15% of inner-loop uops — gave 1.01x;
four independent accumulators gave 1.12–1.21x and is what `q40.h` now does). Both
target ISAs already do the int8 dot in about one instruction (AVX2 `maddubs`+`madd`,
ARMv8.2 `sdot`), which is exactly the situation LUT does not improve. At 3 bits that
changes: the unpack gets expensive, and the table amortises well since it is built
once per activation vector and reused across O = 1792..2048 rows.

Two costs to weigh when building it:

- it breaks the bit-identical-to-`block_q4_0` property `q40.h` calls out as a feature
  (GGUF expert tensors `memcpy` straight in) — for the q3 tier only, but the file
  stops being uniformly GGUF-compatible;
- `pshufb`-based tables must be int8-quantised, which stacks a second activation
  quantisation error on top of the ~1.3e-2 the int8-activation build already carries.

### Don't bother with

Measured and rejected, either by WASTE or here — recorded so nobody spends a week
rediscovering them:

- **Per-expert importance-weighted bit allocation.** `usage.bin` makes this tempting.
  WASTE measured quantisation error as near-uniform across experts and layers, with
  optimised allocation beating random by 1.01–1.15x; and since the low-precision
  experts would be the rarely-routed ones, it saves disk capacity, not bytes read.
  Disk capacity is not our constraint either. (This does not touch `--expert-edge`,
  which is per-layer and quality-motivated.)
- **Metal for decode.** WASTE implemented and verified a Metal backend and measured it
  22% slower than CPU on this workload; `gpu.h` already argues the same for the
  4–8 GB case.
- **Trusting a microbenchmark.** WASTE got a 1.44x isolated win from an index-layout
  change that produced no end-to-end gain, because the microbenchmark did not model
  threads sharing cache. Measure the engine.
