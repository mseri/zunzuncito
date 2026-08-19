# zunzuncito — colibrì-style MoE engines for a small-RAM machine

Four engines share this repo, along with the q4_0/q8_0 kernels (`q40.h`), the ternary
kernels (`tq2.h`), the KVarN KV cache (`kvarn.h`), the OpenAI server, and one idea:
stream the routed experts from disk under an expert-granular cache instead of leaving
it to the OS page cache.

| binary | model | notes |
|--------|-------|-------|
| `gemma4` | Gemma-4 26B-A4B | 30 attention layers, 128 experts/layer, MTP + DFlash speculation, optional FlashHead |
| `lfm25`  | [LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) | hybrid 18 short-conv + 6 attention layers, 32 tiny experts/layer, apex-quant mixed precision, optional FlashHead |
| `maple`  | [Maple-preview 20B-A1B](https://github.com/deepgrove-ai/mlx-lm-deepgrove) | ternary throughout, 256 experts/layer, sliding + full attention, FlashHead |
| `ling`   | [Ling-3.0-tiny 7.9B-A1.3B](https://huggingface.co/inclusionAI/Ling-3.0-tiny) | 3:1 KDA/MLA hybrid, 128 experts/layer + a shared one, absorbed MLA cache, optional FlashHead |

Most of this README is about `gemma4`; see [LFM2.5-8B-A1B](#lfm25--lfm25-8b-a1b),
[Maple](#maple--maple-preview-20b-a1b) and [Ling](#ling--ling-30-tiny-79b-a13b) for the
other three.

## gemma4 — Gemma-4 26B-A4B

A Gemma-4 26B-A4B [colibrì-style inference engine](https://github.com/JustVugg/colibri)
for macOS (may work on Linux, but I never tried). The OpenAI webserver follows closely
[samosa-chat, a Qwen3.6-35b colibrì-style inference engine](https://github.com/deepanwadhwa/samosa-chat).
It runs the model, quantized or unquantized, on very RAM-constrained systems: it works
fine on a Mac with 8 GB of RAM while doing other things, and on older Intel Macs.

Gemma-4 26B-A4B is ~25 B params of which only ~3.8 B activate per token. At q4_0 that
splits into 1.3 GB of dense weights (resident) and 12.9 GB of routed experts (3840 of
them, 3.19 MiB each). On a 4–8 GB machine the experts don't fit, so we stream them
from disk with an expert-granular cache instead of leaving it to the OS page cache.

Three things make that fast enough to use:

1. The prefetch is exact rather than predicted. Gemma-4's router reads the raw
   post-attention residual, so the 8 expert IDs for a layer are known before the dense
   MLP runs. We route first, fire the reads, then compute the MLP while they fly.
2. The MoE runs over the batch union. The S tokens of a prefill batch collectively
   route to at most `min(128, 8·S)` distinct experts per layer. We read each once, so a
   512-token prompt drops from 122,880 expert reads to ≤3,840.
3. The pin set is learned. Expert usage is heavily skewed and stable across prompts, so
   routing counts persist to `usage.bin` and the next run pins the hot set into slots
   the LRU may never evict. On a constrained cache this easily reaches 80% hits.

MTP speculative decoding is implemented, but on my system the IO slowdown makes it
useless.

All of this is only worth it if you have 4–8 GB of RAM. With 16 GB you are better off
with llama.cpp — though I get comparable performance by tuning the IO threads (`--io`)
and the compute threads (`--threads`), see below, and I keep a lot more RAM free at
high context (KVarN for the KV works quite well).

This is the pure-text model: vision is not implemented at all, and the CLI does not
implement tool calling.


## Build

```sh
make                 # auto-detects Intel vs Apple silicon
make check           # full regression suite
```

Everything is overridable:

```sh
make CC=gcc-14 OMPFLAGS=-fopenmp OMPLIBS=-fopenmp
make ARCHFLAGS="-mcpu=apple-m3"
make OMP=0           # single-threaded
```

On macOS the Makefile looks for MacPorts' or Homebrew's libomp and applies Apple
clang's `-Xpreprocessor -fopenmp`.

Check you got the right kernel. `q40.h` has AVX2, NEON (with an ARMv8.2 dotprod fast
path, M1 and later) and a portable fallback. If `ARCHFLAGS` is wrong you land silently
on the scalar path and lose several times the speed. `./test_q40` prints which one you
got.

## Convert the model

You need the `-unquantized` QAT checkpoint (safetensors); we re-quantise the weights
ourselves, in the block structure the engine wants. Pass the maximum context length
you expect to use and the RAM budget to configure things for:

```sh
python3 tools/convert_gemma4.py /path/to/gemma-4-26B-A4B-it-qat-unquantized ./g4 \
        --ram 4 --ctx 4096
python3 tools/convert_tokenizer.py /path/to/checkpoint/tokenizer.json ./g4/tok.bin
```

Use a corrected `tokenizer.json`: last time I checked, the official Google repo did
not yet contain the recent fixes.

That writes the quantized model and everything else needed into `./g4`, which is then
the only thing you need to run it.

`--ram` sizes the expert cache and writes `slots_per_layer` into the container:

| RAM | expert cache | slots/layer (of 128) | % of expert set resident |
|-----|--------------|----------------------|--------------------------|
| 4 GB  | 1.96 GB  | 21  | 16%  |
| 6 GB  | 4.02 GB  | 43  | 34%  |
| 8 GB  | 5.98 GB  | 64  | 50%  |
| 12 GB | 10.0 GB  | 107 | 84%  |
| 16 GB | 11.96 GB | 128 | 100% (fully resident) |

The minimum viable budget is 2.70 GB (the floor is `topk` = 8 slots/layer).

The converter needs the full checkpoint readable, but it streams it, so it needs very
little RAM — only disk space. Expect it to take a while, since it re-quantises every
tensor.

It also builds a [FlashHead](#flashhead) unless you pass `--no-flash`. That is a
k-means over the embedding table and the one step that is not streaming: it holds the
head in RAM (2.8 GiB at Gemma-4's dimensions) for the duration. `--flash-iters` and
`--flash-probes` tune it. The engine ignores the result unless run with `--flash`.

## Run

```sh
./gemma4 ./g4 "explain MoE routing"
```

The prompt is a positional argument (the first non-flag one). For an interactive
multi-turn session, use `--chat`:

```sh
./gemma4 ./g4 --chat
```

Omitting the prompt entirely also enters interactive mode. A positional prompt given
alongside `--chat` is used as the first user turn, with subsequent turns read from
stdin. `--chat` cannot be combined with `--serve`.

Generation defaults to 2048 tokens unless `--max_tokens` overrides it:

```sh
./gemma4 ./g4 --max_tokens 100
```

### OpenAI-compatible server

Start the model as a local, loopback-only OpenAI-compatible server:

```sh
./gemma4 ./g4 --serve --port 8484
```

This exposes:

- `GET /v1/models`
- `POST /v1/chat/completions` (JSON and `stream: true` SSE responses)
- `GET /healthz`
- `POST /v1/cancel`
- `POST /v1/shutdown`

Example:

```sh
curl -N http://127.0.0.1:8484/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-26b-a4b","messages":[{"role":"user","content":"Explain MoE routing."}],"stream":true,"max_tokens":128}'
```

The supported generation controls are `messages` (text `system` and `user` messages),
`stream`, `max_tokens`/`max_completion_tokens`, `temperature`, `top_p` and `top_k`.

Requests are serialized, since the KV state is not concurrently mutable. The server
caches the most recently completed tokenized conversation and reuses its matching
prefix on the next request, so send the prior `assistant` response back in `messages`
to continue incrementally; if the prefix differs, the KV state is rebuilt from the
first differing token. The cache is in memory only and dies with the server.

Useful flags for the CLI generation:

| flag | meaning |
|------|---------|
| `--chat` | enter interactive multi-turn chat mode |
| `--system S` | system prompt |
| `--think` | enable reasoning (injects `<\|think\|>`) |
| `--raw` | skip the chat template, feed the prompt verbatim |
| `--temp F` `--topp F` `--topk N` | sampling; defaults are Gemma-4's own: 1.0 / 0.95 / 64. `--temp 0` = greedy |
| `--pin N` | pin N experts/layer from `usage.bin`. Needs one prior run to learn |
| `--draft DIR` `--ndraft N` | MTP speculative decoding against a drafter container |
| `--io N` | I/O threads (default 8) |

Pinning only kicks in from the second run: the first one writes the routing statistics
to `usage.bin` for later runs to use. `--pin` is capped at `slots_per_layer - 1`, so a
miss always has somewhere to load its expert.

To use MTP you have to convert it first, into the target container directory (`./g4`
in the examples so far):

```sh
python3 tools/convert_gemma4_mtp.py /path/to/gemma4-assistant-safetensors ./g4
```

Then run it with `--mtp`:

```sh
./gemma4 ./g4 --mtp --ndraft 4 "..."
```

There is also `--draft DIR`, if you want the drafter in a separate directory with its
own KV.

MTP doesn't pay off on this engine. It would if verification used the same experts,
but here it usually has to load and unload others, which costs at least a factor of 2
in my measurements. It may help if everything fits in RAM, but I have no system with
enough RAM to check.

### Metal

Metal support is off by default, because it makes everything slower on both Intel and
M1. The kernel issues one dispatch and one `waitUntilCompleted` per matvec, and a
token needs thousands of them: at batch size 1 you pay ~0.5 ms of dispatch latency
against ~40 ms of real arithmetic. A GPU only wins here with far more work per
dispatch (whole layers fused into one command buffer, or large prefill batches), and
even then the engine is disk-bound at a 4–8 GB budget. Fixing it properly would take
time, so it is opt-in with `--metal`, and `make METAL=0` builds a pure-CPU binary.
Any Metal failure falls back to the CPU silently, so there may be issues I never
noticed.

At a small RAM budget this engine is IO bound anyway, roughly 800 MB of expert reads
per token against 7.6 GFLOP of compute, so a faster matmul does not help you wait on
NVMe. Metal could still help for:

- prefill, which is batched and genuinely compute-bound;
- a 16 GB budget, where the whole container is resident and there is no disk in the
  loop, so unified memory bandwidth may beat the CPU bandwidth.

### FlashHead

Gemma-4 ties the lm_head to the embedding table: 262144 x 2816, which at q4_0 is
**415 MiB read for a single matvec per decode step**, more than the attention weights
and the routed experts of that step put together. Of the three engines this is where
the idea should pay best.

`--flash` cuts that down. The converter groups the head rows into 8192 clusters of 32
(balanced spherical k-means, `tools/flashhead.py`), decode scores the 13 MiB of
centroids, and exact logits are computed only for the 883 best clusters: 28256 rows,
about 58 MiB, 10.8% of the vocabulary. Everything else is -inf, so greedy decoding is
exact whenever the true argmax lies in a probed cluster.

It is off by default, unlike `maple`'s, because there the clustering ships in the
checkpoint and here it is built post-hoc from the embedding table.

Two Gemma-4 specifics. The softcap is applied to the probed logits only:
`tanh(-inf/30)*30` is -30, not -inf, so capping the whole vector afterwards would turn
every pruned token into a perfectly samplable one at the floor of the distribution.
And `--flash` is refused under `--mtp` and `--dflash`: the draft verifier needs real
logits for tokens the head would prune, or acceptance stops measuring the drafter.

`--flash-check` runs the exact head alongside on the same row and reports argmax
agreement plus `max |probed - exact|`, which must be 0. Use it before trusting the
approximation on a prompt set you care about. On LFM2.5 the same construction agrees
100/100 on greedy decoding, but that is a different vocabulary and a different
embedding geometry, and the decode speedup there was inside the noise. I do not have a
Gemma-4 checkpoint on this machine, so the byte counts above are arithmetic and the
tok/s is unmeasured.

`--probes N` overrides the cluster count. An already-converted container can be
upgraded in place with `python3 tools/add_flashhead.py ./g4 --src /path/to/checkpoint`
instead of re-converting.

## KV-cache compression (KVarN)

On by default at upstream's shipped preset, and it's what buys you context length.
`--kv off` gets you back to an f32 cache. Gemma-4's 25 sliding layers cap their KV at
1024 positions, so they cost a fixed 400 MiB whatever the context; all the growth is
in the 5 global layers, and on a constrained system you feel it.

`--kv PRESET` enables [KVarN](https://github.com/huawei-csl/KVarN) (Hadamard rotation
+ log-domain Sinkhorn variance normalisation + asymmetric RTN). One flag, one of
upstream's four `--kv-cache-dtype` values, and nothing inside a preset is separately
settable. The bit widths and the tile are one calibrated recipe; taking them apart
gets you a configuration nobody measured.

The point of the Sinkhorn step is outliers. A handful of KV channels carry an order of
magnitude more variance than the rest, in every layer, and a per-vector quantiser has
to set its scale by them. The rotation smears that energy across all coordinates
rather than removing it, so the bulk of the vector gets a fraction of the codebook.
KVarN quantises a *tile* of 128 consecutive tokens instead, and before rounding it
drives that tile towards equal variance along both axes by alternating column-wise and
row-wise std normalisation in log space. A hot channel is divided down by its own row
scale, a hot token by its own column scale, both scales are kept, and what reaches the
rounder has no outlier left to waste resolution on. `kvarn.h` has the details.

### The presets

```sh
./gemma4 ./g4                         # kvarn_k4v2_g128, the default
./gemma4 ./g4 --kv kvarn_k4v4_g128    # 4-bit values, ~25% more KV
./gemma4 ./g4 --kv kvarn_k4v2_g64     # 64-token tiles
./gemma4 ./g4 --kv kvarn_k4v4_g64
./gemma4 ./g4 --kv off                # f32 KV
```

`--check` is the exception: it diffs the forward pass against a stored oracle to
~1e-4, which is tighter than any KV quantiser reproduces, so it uses an f32 cache
unless `--kv` is given explicitly.

Every preset is K4: KVarN spends its bits on keys, because a key error moves every
attention score while a value error is averaged out by the softmax weights. 128 is
upstream's design point and the default; 64 gives finer granularity for a little
more scale overhead per token, and on the fixture is marginally the more accurate
of the two.

KV bytes at each context, all four presets, exact (`tools/convert_gemma4.py`'s planner
and the engine's own sizing agree to the byte):

| ctx  | f32 KV | k4v2_g128 | k4v4_g128 | k4v2_g64 | k4v4_g64 |
|------|--------|-----------|-----------|----------|----------|
| 4K   | 560M   | 123M      | 144M      | 121M     | 141M     |
| 32K  | 1680M  | 237M      | 293M      | 241M     | 295M     |
| 128K | 5520M  | 625M      | 801M      | 652M     | 826M     |
| 256K | 10640M | 1142M     | 1478M     | 1199M    | 1534M    |

At 4K it buys you almost nothing. At 32K it triples the expert cache.
At 128K+ it makes the difference between running and not running: 256K of KV drops
from 10.4 GiB to 1.1 GiB.

Every layer runs the preset's own widths. There is no protected-layer carve-out, on
the grounds that a per-layer override would mean asking for `kvarn_k4v2_g128` and
getting something else.

What KVarN spends, against a per-vector codec, is scale bytes: one fp16 scale per
channel and one per token per tile (2·d + group halves for K, 2·group + d for V)
where TurboQuant carried a single f32 norm per vector. At d=512 and group 128 that is
18 bytes per token rather than 4. A per-channel scale is what the variance
normalisation needs, so this is the method's cost rather than an implementation tax,
and it shrinks as a fraction of the store as the context grows.

Below 4 bits the per-row RTN range comes from the [q, 1-q] percentiles rather than
from the row's extremes, and the few values outside get clamped. This is upstream's
`KVARN_RTN_QUANTILE`, at its q=0.005. With four levels to spend, one loud coordinate
across the full min..max costs the other 127 samples most of their resolution, and
after the balancing there is usually exactly one left: Sinkhorn equalises variance,
not kurtosis. It is worth 6% of the round-trip error at 2 bits and 5% at 3, and it
is why the shipped preset's 2-bit values hold up. At 4 bits and above the extremes
are worth keeping exactly, so min/max stands.

`./test_kvarn` reports what the codec costs on synthetic outlier-heavy KV: at K4 the
attention scores keep 0.995 cosine against f32, at K8 0.99998, at K2 0.90. Treat those
as regression tripwires rather than accuracy claims. Upstream makes the same point
from the other side: a high score similarity does not by itself prove generation
works.

The residual window is this engine's own addition, fixed at 128 positions and not
exposed. It exists because a tile cannot be sealed until all of its tokens exist, so
the newest ones need somewhere to live. It is rounded up to
whole tiles, and with speculation on (`--mtp`, `--draft`, `--dflash`) gemma4 widens it
by one tile plus the draft length, since a sealed tile is final and a rejected draft
must never be baked into one.

### Making it fast

A compressed cache only pays if reading it is cheaper than reading the f32 one it
replaces, so most of the work here went into the read path rather than into the
codec's arithmetic. Decoding one d=512 key costs 385 ns and encoding a 128-token
tile costs 1817 us. Four decisions account for almost all of that.

The largest single factor is that the code plane is stored token-major even for K,
which the quantiser balances channels-major. Storing it the way it was quantised
turns reading one token into a 64-byte-stride gather, 512 cache lines to rebuild a
single vector. Transposing on the way in costs one blocked pass per 128 tokens and
buys a contiguous read on every attention step afterwards.

Attention then stays in the rotated frame. H is orthonormal and symmetric, so
`q.k = (Hq).(Hk)` and `sum_t w_t v_t = H(sum_t w_t Hv_t)`: rotate the query once per
token, transform the accumulated output once, and the inverse Hadamard leaves the
per-position path entirely. The positions still in the f32 ring get rotated on the
way in to join it.

The transform itself is vectorised. From len=8 up a stage is a plain add and
subtract of two vectors; the three stages below that live inside a single register
and are one shuffle plus one multiply-add against a sign pattern. It comes out
bit-identical to the scalar version, which `tests/test_kvarn.c` checks.

Two loops were bounded by float-add latency rather than by throughput. The Sinkhorn's
row standard deviation and the attention dot product each accumulated into one
running sum, and at four cycles per add that bounds a loop at four cycles an element
however wide the vector unit is: 561 ns for a head_dim 512 dot against 52 once the
chain is split eight ways. The Sinkhorn also reduced down columns with a
`for c { for r }` loop, which takes a cache miss per element where C accumulators
take none, and it ran the reference's eight sweeps per iteration where two carry the
same arithmetic in the same summation order.

A tile's per-channel scale plane is converted once per tile rather than once per
position. It is the same for all `group` of the tile's tokens, so rebuilding it per
position did `group` times the work the tile needed and touched a second region of
the record tens of KiB from the codes on every read. Sequential decode went from
111 ns to 68 at d=512. This one mattered far more than the microbenchmark suggested:
on an Intel MacBook Pro it was the difference between maple running at 12 tok/s and
at 29, and it only bit the 128-token tile, where that second region sits twice as
far from the codes. On a Xeon with a 1 MiB L2 the same change is worth a few percent
and both tile sizes behave identically, which is a decent argument against trusting
one machine's cache behaviour.

The dot split (`head_dot`) is not really a KV change. The attention dot is there
either way, so `--kv off` gets the same speedup from it.

On one kv head's attention over T cached positions:

| d / T / q heads | plain dot, model frame | rotated frame | `head_dot` | both |
|---|---|---|---|---|
| 512 / 4096 / 8   | 53.5 ms | 46.0 | 33.6 | **27.2 (1.97x)** |
| 512 / 16384 / 8  | 220.1 ms | 191.7 | 144.2 | **116.1 (1.90x)** |
| 256 / 1024 / 4   | 2.05 ms | 1.76 | 0.91 | **0.60 (3.38x)** |
| 128 / 4096 / 4   | 4.21 ms | 3.46 | 1.89 | **1.22 (3.44x)** |

The rotated frame is worth 1.16x on its own and 1.5x once the dot stops being the
bottleneck, so the two compound. Both reorder float summations, which moves logits
by around 1e-6 relative. That is well under q4_0's own weight error, and `--check`'s
1e-4 tolerance against the numpy oracle is tight enough to still be a real test of
them rather than a rubber stamp.

One thing that did not work: fusing the Sinkhorn's two remaining sweeps into one is
about 10% slower. A tile is 256 KiB and lives in L2, so nothing was bandwidth-bound
to begin with, and making each row wait on its own standard deviation drains the
pipeline.

## Context length

Both engines take `--ctx N` to override the context the container was converted with.
It is applied before the KV cache is sized, so lowering it just returns RAM. Raising
it is allowed too, with a warning: the container's `ctx` only fixed `slots_per_layer`,
the weights do not care, but the total will exceed the RAM budget the conversion
planned for.

The startup line reports what you actually got, and it accounts for `--kv`:

```
kv:  123 MiB for ctx 4096  (KVarN K4/V2, tile 128, f32 ring 128)   # gemma4 default
kv:  172 MiB for ctx 16384 (KVarN K4/V2, tile 128, f32 ring 128)   # gemma4 --ctx 16384
kv:  567 MiB for ctx 4096  (f32, by --kv off)
kv: 1047 MiB for ctx 16384 (f32, by --kv off)
   warning: that is 480 MiB more KV than the container's plan budgeted ...
```

The warning is about bytes rather than the context number: it fires only when the KV
you are actually allocating exceeds what the conversion budgeted (an f32 KV at the
container's own `ctx`). With `--kv` a 4x longer context often costs less than the
plan assumed, as in the 4x-context-for-a-third-of-the-RAM row above, and stays quiet.

Under `--serve` this is the server's context window, advertised as `context_length`
and `max_model_len` on `GET /v1/models`; a prompt that leaves no room for a
completion is rejected with both figures in the error. There is no per-request
context parameter, since the KV cache is allocated once at startup: changing the
window means restarting with a different `--ctx`.

## RAM budget

Both engines also take `--ram F` (GB) to re-plan the expert cache at startup, without
reconverting. `slots_per_layer` is the only thing the conversion's `--ram` fixed, and
nothing in the container depends on it: experts are read from `experts.bin` one at a
time by offset and the cache is a plain LRU. So the same container runs on a smaller
or a larger machine than it was converted for.

This re-runs the converter's planner (`plan()` in `tools/convert_*.py`) against the
resident dense blob and the KV it just allocated, instead of the estimates the
conversion had to work from:

```
$ ./lfm25 ./lfm-ct --ram 4
kv: 14 MiB for ctx 4096 (KVarN K4/V2, tile 128, f32 ring 128)
ram: 4 GB budget -> 13 slots/layer (dense 606 MiB + kv 14 MiB + cache 3191 MiB)
```

It composes with `--ctx` and `--kv`, in that order: the KV is sized first and the
cache gets what is left, so a longer context costs slots. Below `topk` slots per layer
the cache would thrash inside a single forward, so rather than run unusably slow it
exits and tells you the minimum budget for the current context:

```
$ ./gemma4 ./g4 --ram 2
--ram 2 GB leaves room for 3 experts per layer, below topk=8: this model needs 2.38 GB
```

Fewer slots costs hit rate but never correctness: with `--ram` low the engine just
streams more experts per token from disk. Check the `expert cache: NN% hit` line at
exit, and see `--pin` under Tuning. Without `--ram` the container's own plan is used.

## Tuning

You can improve generation speed quite a lot by playing with these:

- `--io` sets the I/O threads, default 8. We do 240 expert reads per token at 3.19 MiB
  each, so on a good SSD you can go considerably higher.
- `--pin` decides how much of the cache to freeze rather than leave to the LRU. At 4 GB
  there are only 21 slots/layer, and the right split depends on the routing
  distribution of your own requests.
- `--threads` sets the compute threads, default 2. Raise it according to your core count.

The engine prints hit rate, expert reads and speculation acceptance after every run,
so you can see what the flags did.

`make check` runs a few scripts to validate the implementation. You can ignore it
unless you are modifying the code.

```
modeling_gemma4.py (HF)  ──►  gemma4_oracle.py   3.7e-7   (numpy, from the architecture)
gemma4_oracle.py         ──►  gemma4.c           2.6e-7   (exact-activation build)
batch-union prefill      ──►  sequential         bit-identical
Python converter         ──►  C q40 kernel       bit-identical
HF tokenizer             ──►  g4tok.h            416/416 exact (incl. 400 fuzzed)
KVarN (huawei-csl)       ──►  kvarn.h            round-trip + attention cosine, 2..8-bit
chat_template.jinja      ──►  chat_prompt()      5/5 exact
```

## lfm25 — LFM2.5-8B-A1B

`lfm25` runs [LiquidAI/LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B)
(8.3 B total, 1.5 B active). The bet is different from Gemma-4's: the experts are tiny
(32 per layer at `moe_intermediate_size` 1792, ~5.9 MiB each at q4_0, of which only 4
fire per token), so a miss is cheap and a layer's whole expert set is only 32 of them.
At an 8 GB budget essentially the entire model is cacheable.

Being a hybrid changes the engine in two ways. Only 6 of the 24 layers are attention,
and the other 18 hold no KV at all, so the KV cache costs a quarter of what the layer
count suggests (96 MiB at ctx 4096). And those 18 carry a recurrent conv state, which
unlike a KV cache cannot be rewound: prompt-prefix reuse (chat, server) is only taken
when the new prompt strictly extends what was already absorbed, and anything else is
reprocessed from scratch.

No MTP, no DFlash (this model ships neither). No Metal.

### Mixed precision (apex-quant, ported)

[apex-quant](https://github.com/localai-org/apex-quant) is a llama.cpp recipe. It
invents no format; it assigns different quant types per tensor class, exploiting that
routed experts are ~97% idle and so tolerate far more error than the always-on
tensors. We keep the idea and drop the GGUF dependency, mapping its Q3_K..Q8_0
gradient onto the two block formats this engine has kernels for:

| tensors | format |
|---------|--------|
| attention, conv, dense MLP (always on) | q8_0 |
| routed experts, edge layers | q8_0 |
| routed experts, middle layers | q4_0 |
| norms, router, expert bias, depthwise conv | f32 |

`--expert-edge N` is the size/quality dial (default 2, `0` = uniform q4_0). Each
q8_0 layer costs ~1.9x its q4_0 equivalent, straight out of the expert cache.

### Convert and run

```sh
python3 tools/convert_lfm25.py /path/to/LFM2.5-8B-A1B ./lfm-ct --ram 8 --ctx 4096
python3 tools/convert_lfm_tokenizer.py /path/to/LFM2.5-8B-A1B/tokenizer.json ./lfm-ct/tok.bin
./lfm25 ./lfm-ct "explain MoE routing"
```

The converter also builds a [FlashHead](#flashhead-1) unless given `--no-flash`; the
engine ignores it unless run with `--flash`.

At `--ram 8 --ctx 4096 --expert-edge 2` that gives 606 MiB dense resident, 4.72 GiB
of experts, 96 MiB KV and 29 of 32 slots/layer. Measured on an Intel Mac at 8 threads:
~24 tok/s prefill, ~16 tok/s decode, ~90% expert-cache hit.

Flags mirror `gemma4` where they mean the same thing (`--chat`, `--serve`, `--system`,
`--raw`, `--temp/--topp/--topk`, `--pin`, `--io`, `--threads`, `--ctx`, `--ram`,
`--kv`). Sampling defaults are LFM2.5's
own: temp 0.2, top_k 80. `--batch N` (default 128) sets the prefill batch, which is
what gives each streamed weight row its reuse, so lowering it saves scratch and costs
prefill speed. `--think` forces a reasoning block by pre-filling `<think>`; the chat
template has no thinking toggle, the model decides for itself.

The tokenizer has its own container (`lfmtok.h`, magic `LFTK`) rather than reusing
`g4tok.h`.

### FlashHead

The same approximation Maple uses, built here instead of read out of the checkpoint.
The head is tied to the embedding table, 128000 x 2048, which at q4_0 is 140 MiB read
per decode step for a single matvec. The converter clusters those rows into 4000
groups of 32 and decode scores the centroids first, computing exact logits only for
the 431 best clusters (10.8% of the vocabulary). Everything else is -inf.

It is off by default; `--flash` turns it on. `maple` differs because there the
clustering is the checkpoint's own and the model was released with it enabled. Here
`tools/flashhead.py` runs balanced spherical k-means over the embedding table after
the fact, so the exact head stays the default.

Measured on an M1 (8 GB, 4 threads), 100 greedy tokens, interleaved run-by-run,
best of 4 pairs:

| `--ram` | head | decode | argmax agreement |
|---------|------|--------|------------------|
| 6 | exact | 14.2 tok/s | -- |
| 6 | flash, 431 probes | 14.9 tok/s | 100/100 |
| 4 | exact | 12.0 tok/s | -- |
| 4 | flash, 431 probes | 12.3 tok/s | 100/100 |

So: correct, and worth almost nothing on this machine. The reason is the byte
accounting rather than the head. At `--ram 4` a decode step is dominated by streaming
~150 MiB of experts off disk at a 77% hit rate, and 140 MiB of *resident* head is a
small share of a 100 ms step; Maple gets +15% from the same idea because its step is
6 ms and the head is a quarter of it. Run-to-run drift on this machine is larger than
the effect, which is why the runs are interleaved and why the table is a best of four.

What the approximation costs is agreement, and that is measurable. `--flash-check`
runs the exact head alongside on the same row and reports how often the two pick the
same token, plus `max |probed - exact|`, which must be 0: a probed logit comes out of
the same kernel and is not approximated at all, only the candidate set is.

| probes | vocabulary scored | argmax agreement |
|--------|-------------------|------------------|
| 431 (default) | 10.8% | 100/100 |
| 128 | 3.2% | 100/100 |
| 64 | 1.6% | 99/100 |

`--probes N` overrides the count. Control tokens (EOS and the chat template's markers,
120 ids for LFM2.5) are always scored, since a token at -inf cannot be sampled and a
model that cannot emit EOS does not stop.

An existing container can be upgraded without re-converting:

```sh
python3 tools/add_flashhead.py ./lfm-ct --src /path/to/LFM2.5-8B-A1B
```

That clusters the container's own (dequantised) head and appends 4.9 MiB to
`dense.bin`, ~50 s. `--src` is read only for the forced control-token ids.

### Checking it

```sh
make check-lfm25          # add PYTHON=... if numpy/tokenizers live in a venv
```

That runs the tokenizer against HF's (falling back to a bundled reference
implementation, loudly, if `tokenizers` is not installed), then builds a tiny random
fixture and diffs the engine against `tools/lfm25_oracle.py`, an independent numpy
forward pass run on the dequantised container weights, so quantisation error cannot
mask an engine bug.

```
numpy oracle          ──►  lfm25.c            1.8e-7   (exact-activation build)
batch prefill         ──►  sequential         bit-identical
expert pinning        ──►  unpinned           bit-identical
HF tokenizer          ──►  lfmtok.h           475/475 exact (incl. 400 fuzzed)
```

## maple — Maple-preview 20B-A1B

`maple` runs DeepGrove's Maple preview, a 20 B-parameter MoE with ~1 B active. The
premise is different from the other two: the checkpoint is ternary. Every matrix in
the model, the four attention projections and all three expert projections, is stored
as 2-bit codes with one scale per output row, and across the whole checkpoint those
codes only ever take the values 0, 1 and 2:

```
w = alpha_row * (code - 1)          so every weight is -alpha, 0 or +alpha
```

That is trained, not something the converter chose, and it changes what this engine
has to decide. There is no `--expert-edge` and no q8_0 tier: apex-quant's premise is
that routed experts tolerate more error than always-on tensors, and here both are
already at the floor. The converter's job is to repack losslessly, and `--verify`
checks exactly that: `max |w - dequant(pack(w))| = 0`, not "small".

20 B of parameters land in a 4.57 GiB expert container plus 449 MiB resident, so an
expert is 780 KiB against gemma4's 3.19 MiB and lfm25's 5.9 MiB. On an 8 GB machine
roughly two thirds of the expert set stays cached at `--ram 4`.

### The ternary kernel

`tq2.h` is where the interesting part lives. Two things fall out of a per-row scale:

1. The scale leaves the inner loop. q4_0 and q8_0 carry an fp16 scale per 32 weights,
   so their kernels decode a scale and issue an FMA per block. Here a whole row is one
   integer accumulation and one multiply at the end. So the layout is per tensor
   rather than per row, `alpha[O]` then `codes[O][I/4]`, which also gives every code
   row a clean power-of-two stride instead of q4_0's 18-bytes-per-block interleave.
2. The -1 offset is free. With u8 codes and i8 activations,
   `<c-1, x> = <c, x> - sum(x)`, and `sum(x)` depends on neither the row nor the
   tensor nor the expert. It is one scalar per activation vector, computed once by
   `tq2_quant_act` and subtracted at the end of every row of every tensor that
   consumes it. q4_0 pays the equivalent correction (its -8) per block, per row.

On NEON it is cheaper still: `sdot` multiplies signed by signed, so the codes become
{-1,0,+1} with one `vsubq_s8` and the scalar is not needed at all. Both routes are
exact integer arithmetic on the same operands, so the ISAs agree bit for bit.

Three tensor classes are not ternary. The embedding table, the lm_head and the
FlashHead centroids are 4-bit affine with a scale and a bias per group of 64, so
`tq2.h` carries that format through too (`q4a`) rather than re-quantising onto q4_0,
which would cost the same 36 bytes per 64 weights while throwing the zero point away.
Norms and the router stay f32, since a routing error is not a small numeric error, it
picks a different expert.

### Attention

Gemma-shaped rather than Llama-shaped: three sliding-window layers (512) to every full
one, RoPE on the sliding layers only and NoPE on the full ones, partial rotary over
the first half of each head, and qk-norm. The sliding layers cap their KV at 512
positions however long the context is, so the KV costs a quarter of what 24 layers
suggests: 132 MiB at ctx 4096, since only the 6 full layers store 4096 deep.
`--kv` therefore applies to the full-attention layers only; compressing a layer that
is already bounded buys nothing and costs accuracy.

One consequence worth stating because getting it wrong does not crash: a prefill batch
cannot publish all its keys before attending. On a sliding layer the ring is exactly
512 long, so writing position `p+127` would evict `p-385`, which row 0 of that same
batch still needs. `attn_fwd` interleaves publish-and-attend per position; the
projections above it are still batched.

### FlashHead

The one approximation on offer, and it is the checkpoint's own. The vocabulary is
clustered into 4748 groups of 32; decode scores the centroids, takes the best 512
clusters, and computes exact logits only for those 16384 tokens. Everything else is
-inf, so greedy decoding is exact whenever the true argmax lies in a probed cluster.

At 4 bits the lm_head is 151936 x 2048 = 175 MiB, the single biggest read of a decode
step, more bytes than the rest of the model put together. Probing 512 clusters reads
about 24 MiB instead.

Measured on an M1, `--ram 4`, 4 threads, 100 tokens, best of 5, with the three
configurations interleaved run-by-run. Consecutive blocks of runs drift by more than
the effect being measured, which is enough to invert the answer:

| head | decode |
|------|--------|
| exact | 30.8 tok/s |
| flash, 512 probes (default) | 35.3 tok/s |
| flash, 128 probes | 37.0 tok/s |

So +15% at the shipped setting, not the 7x the byte count suggests: at ~1 B active
params a decode step is not purely bandwidth-bound, and the fixed costs (attention over
the full-attention layers, ~50 OpenMP fork/joins, the vocabulary-wide sampling scan)
do not move. Sorting the probed ids to make the gather monotonic looked like an obvious
further win and was worth 30.4 vs 30.3 tok/s, i.e. nothing, because a probed row is
already 1152 contiguous bytes and the prefetcher had nothing left to gain from ordering
the rows themselves. That negative result is recorded in `flash_logits` rather than
the code.

`--noflash` uses the exact head; `--probes N` overrides the cluster count.

### Convert and run

```sh
python3 tools/convert_maple.py /path/to/maple-preview-mlx ./maple-ct --ram 4 --ctx 4096
python3 tools/convert_lfm_tokenizer.py /path/to/maple-preview-mlx/tokenizer.json ./maple-ct/tok.bin
./maple ./maple-ct --ram 4 "explain how a rainbow forms"
```

Maple reuses `lfmtok.h`: its tokenizer is the same Qwen2 byte-level BPE LFM2.5 uses.
Two things differ and both are now container fields rather than assumptions: the
digit run (`\p{N}` here, `\p{N}{1,3}` there) and an NFC normaliser. The engine does not
implement NFC; instead the converter ships the set of codepoints whose NFC quick-check
is not Yes, so `lfmtok_nfc_safe()` can prove the ids match HF's when none of them
occurs (which covers ASCII, precomposed accents and every CJK script) and warn when
they do. Existing `LFTK` containers still load unchanged.

Measured on an M1 (8 GB, 4 threads), prefill on a 1605-token prompt:

| `--ram` | slots/layer | prefill | decode | expert-cache hit |
|---------|-------------|---------|--------|------------------|
| 4 | 181/256 | 51.8 tok/s | 34.1 tok/s | 83% |
| 3 | 125/256 | 52.0 tok/s | 34.5 tok/s | 82% |
| 2 | 69/256 | 48.1 tok/s | 28.5 tok/s | 73% |

Note there is no row for `--ram 8`: this is an 8 GB machine, so asking for the whole
container resident costs more in memory pressure than the cache hits are worth, and it
measures *slower* than `--ram 4` (30.9 against 34.2 tok/s). The budget is a budget for
the whole machine, not for the process.

`--pin` earns its keep here as it does elsewhere: at `--ram 3`, pinning 40 experts per
layer from `usage.bin` (65.9% of past routing) lifts the hit rate from 77.8% to 84.5%
and decode from 28.8 to 34.5 tok/s.

Flags mirror the other two engines. Sampling defaults follow the model's own
recommendation, and they depend on the head: `--temp 1.0 --topp 0.95 --topk 20` with
the FlashHead, and `--temp 1.0 --topp 0.95` with top-k off under `--noflash`. The
FlashHead has already restricted the candidates to the probed clusters, so top-k 20
is the tail cut on top of that; with the exact head all 151936 logits are real and
top-p does the cutting alone. An explicit `--topk` overrides either. The chat
template opens a `<think>` block unconditionally (this is a reasoning model and the
template gives no way to turn it off), so the flag is `--nothink` rather than
`--think`. `--metal` works and is off by default, as on the other two: `metal.mm`
carries a `tq2` and a `q4a` kernel next to the q4_0/q8_0 pair, and `--check-gpu`
diffs them against the CPU. The ternary kernel has the cheapest inner loop of the
four, since a per-row scale leaves nothing to decode per block, but that does not buy
any decode throughput (see [Metal](#metal) for why). Prefill and `--noflash` are
where it wins.

### Checking it

```sh
make check-maple PYTHON=.venv/bin/python
```

That runs the ternary and q4a kernels against an independent scalar reference written
from the format description (all three ISA paths), then builds a tiny random fixture
and diffs the engine against `tools/maple_oracle.py`, a numpy forward pass over the
dequantised container weights.

```
numpy oracle          ──►  maple.c            2.5e-7   (exact-activation build)
batch prefill         ──►  sequential         both match the oracle
expert pinning        ──►  unpinned           bit-identical
tq2/q4a SIMD          ──►  scalar reference   NEON, AVX2 and portable all agree
HF tokenizer          ──►  lfmtok.h           475/475 exact (incl. 400 fuzzed)
```

The oracle is a second transcription of the same reading of `maple.py`, so it cannot
catch a systematic misunderstanding of the architecture: if the RoPE convention or the
sliding-window boundary were read wrongly, both would be wrong together.
`tools/maple_mlx_check.py` closes that gap by diffing greedy generation against the
actual mlx-lm reference on the real checkpoint. It needs `mlx-lm` and a Metal GPU, so
it is not part of `make check-maple`; run it by hand after any architecture change.

## ling — Ling-3.0-tiny 7.9B-A1.3B

`ling` runs [inclusionAI/Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny)
(7.9 B total, 1.3 B active). It is the smallest model here and the one whose context is
cheapest, and both come from the same place: it is a 3:1 hybrid of two attention
mechanisms, neither of which is ordinary softmax attention over a per-head KV cache.

```
layers 0..23,  MLA where (idx+1) % 4 == 0     →  3, 7, 11, 15, 19, 23
               KDA everywhere else            →  the other 18
layer 0        dense SwiGLU MLP (4608)
layers 1..23   128 routed experts (512) + 1 always-on shared expert (512), top-8
```

**The 18 KDA layers hold no cache at all.** Kimi Delta Attention is linear attention:
each layer carries a `16 x 128 x 128` recurrent state, 1 MiB, and that is its size at
position 1 and at position 131072 alike. 19.3 MiB for the whole model, whatever the
context. Like `lfm25`'s short convolutions it is a recurrence rather than a cache, so it
only moves forwards: prompt-prefix reuse (chat, server) is taken only when the new
prompt strictly extends what was already absorbed, and anything else is reprocessed from
scratch.

**The 6 MLA layers are run absorbed.** Multi-head Latent Attention projects the input
down to a 512-wide latent, and the textbook forward pass expands that back out into
per-head keys and values before attending — 16 heads x (192 + 128) floats, 20 KiB per
position per layer. The converter instead splits `kv_b_proj` per head, transposes the
key half, and the engine folds the two halves into the ends of the attention:

```
score_t = (W_k[h]^T q_nope[h]) . c_t  +  q_rot[h] . k_rot_t
out[h]  = W_v[h] (sum_t a_t c_t)
```

so the cache holds the latent itself — 512 + 64 floats, 2.3 KiB per position per layer.
Both folds cost one `kv_b_proj`'s worth of arithmetic per token, and in exchange the
per-cached-position work drops from a 512x256 expansion to a 576-wide dot. Over the
whole model the context costs **13.8 KiB a token instead of 120 KiB**: 108 MiB at
8 K context, 27 MiB with KVarN on, against ~960 MiB unabsorbed.

The engine's `--kv` applies KVarN to the latent, with one codec rather than two: in
absorbed form the latent *is* simultaneously the key (what the folded query dots
against) and the value (what the attention weights average), so it gets the more
accurate of the two bit budgets rather than being split down the middle. The 64-wide
RoPE key stays f32 — it is a ninth of the bytes and contributes to the score only.

Two details of KDA are worth naming because both are easy to get plausibly wrong, and
wrong in a way that still generates fluent text. The decay uses fla's *bounded* branch,
`g = lower_bound * sigmoid(exp(A_log) * (f_proj(x) + dt_bias))` with `lower_bound = -5`,
not the `-exp(A_log) * softplus(...)` the same kernel uses when no bound is supplied.
And `FusedRMSNormGated` applies its gate *after* the normalisation, not before.

No MTP: this checkpoint declares `num_nextn_predict_layers` 0 and ships no MTP tensors.

### Grouped routing, and the shared expert

The router is `noaux_tc`, which is not a plain top-k:

1. `scores = sigmoid(router . x)`, in f32 — the checkpoint sets `router_dtype` fp32
   because near-ties at top-8 flip a few percent of picks in bf16, over 23 layers.
2. the 128 experts are cut into 8 contiguous groups; a group is ranked by the **sum of
   its top two** biased scores, and only the best 4 groups survive. An expert with the
   second highest score overall does not fire if its group's other members are weak.
3. top-8 among the survivors.

`expert_bias` selects but never weights: the weight applied is the *unbiased* sigmoid,
renormalised over the 8 kept and scaled by `routed_scaling_factor` 2.5. It is a
load-balancing nudge, and letting it into the weight would rescale every expert's
contribution by an amount unrelated to the input.

The shared expert is the one thing this architecture gives the streaming machinery that
the other three do not have. `gemma4` hides expert reads behind its dense MLP; `lfm25`
and `maple` have nothing beside the MoE and are left with chunk pipelining alone. Here
there is an always-on 512-wide expert reading the same normed hidden the router did, so
`layer_fwd` runs it *between* submitting the routed reads and applying them.

### Mixed precision

The source is fp8: `e4m3` weights with `ue8m0` block scales over `[128,128]` tiles, plus
a `modules_to_not_convert` list that left the MLA LoRAs, `kv_b_proj`, `lm_head` and the
embeddings in bf16. There is no way to avoid a round trip — q4_0/q8_0 are per-32-element
*row* blocks and fp8's tiles are 128x128 across both axes — so the converter dequantises
to f32 and requantises. (Point it at the bf16 checkpoint instead and it just works: it
branches on whether a `_scale_inv` companion exists. That is the better container, since
q8_0 then quantises the original weights rather than an already-lossy fp8 copy.)

| tensors | format |
|---------|--------|
| KDA/MLA projections, shared experts, dense MLP, embeddings, lm_head | q8_0 |
| routed experts | q4_0 |
| norms, `A_log`, `dt_bias`, `b_proj`, MLA `g_proj`, router, expert bias, conv taps | f32 |

The always-on floor is q8_0 rather than q4_0 for a reason specific to this model: the
KDA recurrence carries a 128x128 state across the whole sequence, so projection error
compounds there in a way it does not inside a softmax layer. `--expert-edge N` and
`--embed-q4` move off the defaults.

### Convert and run

```sh
.venv/bin/python tools/convert_ling.py ./ling-tiny-fp8 ./ling-ct --ram 8 --ctx 8192
./ling ./ling-ct --chat
```

The conversion takes a few minutes, most of it the FlashHead clustering, and produces:

```
dense resident :    980.8 MiB
latent kv      :    108.0 MiB   (ctx 8192, 6 MLA layers, f32; 27 MiB with --kv on)
kda state      :     19.3 MiB   (context-independent)
experts        :  3.64 GiB total, 1.27 MiB each, 23 layers x 128
flash head     :  4912 clusters x 32, probe 530 -> 16960/157184 rows scored (10.8%)
```

Thinking is **on** by default, which is what the chat template does when the caller says
nothing; `--nothink` pre-closes the think block. The server takes the same switch as
`enable_thinking` in the request body. Roles are literal `<role>HUMAN</role>` text rather
than special tokens, and the turn ends at `<|role_end|>`.

The tokenizer needed no new code. It is a byte-level BPE whose split regex is a
notational variant of the one `lfmtok.h` already hand-codes — the contractions factored
into one group, possessive quantifiers, `\s*[\r\n]` for `\s*[\r\n]+` — none of which
changes what is matched, so `convert_lfm_tokenizer.py` just learned to accept the second
spelling.

On an Intel Mac with the container resident: ~17 tok/s prefill, ~12 tok/s decode.

About a token a second of that is recent, from two places, neither of them new
arithmetic. The KDA update used to sweep its `128x128` state twice per head, once to
decay it and form `S k` and once to apply the delta rule, but nothing crosses rows
between the sweeps, so they fuse: the state, 1 MiB a layer in 18 of the 24 layers, is
read and written once instead of twice. And the router accumulated `n_experts x hidden`
of f32 into a single `double`, a serial chain of adds on one thread, sitting in front of
the expert reads it delays. Eight accumulators (still `double` — the width is about the
dependency chain, the type is about `router_dtype`) and an OpenMP loop took that from
4.5 ms a token to under half of one. Greedy generation is identical token for token
afterwards, and the expert-cache read count does not move.

### FlashHead

Available and correct, and on this machine it is a wash: 11.28 tok/s with it against
11.27 without. Unlike `gemma4` and `lfm25` it clusters a real `lm_head` rather than a
tied embedding table — this checkpoint does not tie them — so the clustering sees the
matrix it is actually approximating, and greedy decoding agreed with the exact head on
99/100 tokens at the default 530 probes (98/100 at 64).

The reason it does not pay here is that the head is 157184 x 1536 and lives in
`dense.bin`, which is always fully resident: the exact head is a straight 245 MiB
sequential read at full memory bandwidth, and the FlashHead trades it for a 12 MiB
centroid matvec plus ~17 000 *scattered* row reads. The scatter costs about what the
bytes save. It should pay on a machine where memory bandwidth binds harder than latency.

### Checking it

```sh
make check-ling PYTHON=.venv/bin/python
```

```
numpy oracle          ──►  ling.c             1.4e-6   (exact-activation build)
routing rule          ──►  numpy oracle       168/168 picks identical
batch prefill         ──►  sequential         both match the oracle
expert pinning        ──►  unpinned           bit-identical
HF tokenizer          ──►  lfmtok.h           475/475 exact (incl. 400 fuzzed)
```

`--check` reports the routing separately from the arithmetic, because grouped top-k is a
*discrete* function of the hidden state. On exact activations the engine sees the very
numbers the oracle saw, so every pick must match and a disagreement is asserted as a bug.
On int8 activations the hidden has moved by ~1%, and near a decision boundary either
answer is defensible — but a flipped pick puts a different expert in the sum and moves
the logits by far more than the int8 arithmetic alone would. So `ling --check` counts the
disagreements and then *remeasures* with the oracle's picks pinned, and the tolerance it
asserts is about the arithmetic only. It also reports argmax agreement two ways: exact,
and "within the row's own error", since on a random fixture whose 256 logits sit in a
band narrower than the quantisation error exact agreement is not something the int8 build
can promise.

The int8 tolerance is looser here than in the other three (1.2e-1 against 3e-2) and the
reason is depth times chain length, not this engine. Each layer runs five or six chained
int8 matmuls against two or four in `lfm25`, and the fixture is 64 wide. Measured, the
relative error grows smoothly with depth — `.019 .026 .035 .041 .052 .057 .060 .082` for
one through eight layers — with no step anywhere, which is what says accumulation rather
than a bug.

As with `maple`, the oracle is a second transcription of the same reading of
`modeling_bailing_moe_v3.py` and of fla's KDA kernel, so it cannot catch a systematic
misunderstanding: if the decay branch or the gate order were read wrongly, both would be
wrong together. `tools/ling_hf_check.py` closes that gap by diffing greedy generation
against HF transformers on the real checkpoint. It needs `torch`, `transformers` and
`fla-core`, so it is not part of `make check-ling`; run it by hand after any architecture
change. **I have not been able to run it** — there is no torch on the machine this was
written on — so the architecture is validated by construction and by the model
generating sensible text, not by a reference implementation.

## Not implemented yet

See [TODO.md](TODO.md). The main open item is 3-bit experts for the middle MoE layers,
plus the kernel that needs. It's the one idea worth taking from
[WASTE](https://github.com/sqliteai/waste), and it's specifically a low-RAM feature:
worth 16–22% fewer bytes per token at `--ram 4`, and nothing at all where the cache
already holds the model.

## GenAI use warning

Part of the code comes from [colibrì](https://github.com/JustVugg/colibri) and
[samosa-chat](https://github.com/deepanwadhwa/samosa-chat). The gemma4 implementation
follows transformers closely, diverging for the various optimizations and features and
to adapt to the colibrì idea. There is a serious amount of vibecoding involved in this
project: I used it to explore the differences between Opus 4.8 and Claude 4.6,
ChatGPT-5.6 Luna and Terra, Deepseek V4 Pro and Flash, and their respective costs.
