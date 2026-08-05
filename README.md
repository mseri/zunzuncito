# zunzuncito — colibrì-style MoE engines for a small-RAM machine

Three engines share this repo, along with the q4_0/q8_0 kernels (`q40.h`), the ternary
kernels (`tq2.h`), the TurboQuant KV cache (`kvq.h`), the OpenAI server, and one idea:
stream the routed experts from disk under an expert-granular cache instead of leaving
it to the OS page cache.

| binary | model | notes |
|--------|-------|-------|
| `gemma4` | Gemma-4 26B-A4B | 30 attention layers, 128 experts/layer, MTP + DFlash speculation |
| `lfm25`  | [LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) | hybrid 18 short-conv + 6 attention layers, 32 tiny experts/layer, apex-quant mixed precision |
| `maple`  | [Maple-preview 20B-A1B](https://github.com/deepgrove-ai/mlx-lm-deepgrove) | ternary throughout, 256 experts/layer, sliding + full attention, FlashHead |

Most of this README is about `gemma4`; see [LFM2.5-8B-A1B](#lfm25--lfm25-8b-a1b) and
[Maple](#maple--maple-preview-20b-a1b) for the other two.

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
high context (TurboQuant for the KV works quite well).

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

## KV-cache compression (TurboQuant V3)

Optional, off by default, and it's what buys you context length. Gemma-4's 25 sliding
layers cap their KV at 1024 positions, so they cost a fixed 400 MiB whatever the
context; all the growth is in the 5 global layers, and on a constrained system you
feel it.

`--kvq` enables TurboQuant V3 (random rotation + Lloyd-Max, MSE-only, no QJL),
asymmetric K/V bits, high-bit protected layers, and an f32 residual window.

The slots/layer distribution at a 4 GB budget (0 = will not run):

| ctx  | f32 KV      | K6/V4 p2      | K4/V2 p4      |
|------|-------------|---------------|---------------|
| 4K   | 560M → 21   | 152M → 25     | 132M → 25     |
| 32K  | 1680M → 9   | 350M → **23** | 274M → 24     |
| 128K | 5520M → **0** | 1030M → **16** | 762M → 19   |
| 256K | 10640M → **0** | 1936M → **0** | 1412M → **12** |

At 4K it buys you almost nothing. At 32K it triples the expert cache.
At 128K+ it makes the difference between running and not running.

### Which preset

```sh
./gemma4 ./g4 --kv off      # f32 KV (default)
./gemma4 ./g4 --kv k6v4     # K6/V4, rwin 128, 2 protected layers @ 8 bits
./gemma4 ./g4 --kv k4v2     # K4/V2, rwin 128, 4 protected layers @ 8 bits
```

Pick k6v4 unless you need more than 128K context, in which case k4v2 costs you some
quality. K6's fidelity is very high (0.9997 cos / 94% top-1, against K4's 0.995 / 81%)
and you can check both with `./test_kvq`.

To explore further, set `--kbits`, `--vbits`, `--pbits`, `--protect` and `--rwin` by
hand. `--protect N` protects the first and last N layers at `--pbits` (default 8)
rather than f32: Gemma-4's last layer is global, so protecting it with f32 would cost
1.07 GiB at 128K and undo most of the saving. The residual window `--rwin` isn't
optional at low bit-widths; 3-4 bit compression without one just gives you garbage.

## Context length

Both engines take `--ctx N` to override the context the container was converted with.
It is applied before the KV cache is sized, so lowering it just returns RAM. Raising
it is allowed too, with a warning: the container's `ctx` only fixed `slots_per_layer`,
the weights do not care, but the total will exceed the RAM budget the conversion
planned for.

The startup line reports what you actually got, and it accounts for `--kvq`:

```
kv:  567 MiB for ctx 4096  (f32; --kvq would cut this a lot)   # gemma4, container default
kv: 1047 MiB for ctx 16384 (f32; --kvq would cut this a lot)   # gemma4 --ctx 16384
   warning: that is 480 MiB more KV than the container's plan budgeted ...
kv:  238 MiB for ctx 16384 (K6/V4, rwin 128, 2 protected layers at 8 bits)
```

The warning is about bytes rather than the context number: it fires only when the KV
you are actually allocating exceeds what the conversion budgeted (an f32 KV at the
container's own `ctx`). With `--kvq` a 4x longer context often costs less than the
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
$ ./lfm25 ./lfm-ct --ram 4 --kvq
kv: 20 MiB for ctx 4096 (K6/V4, rwin 128, 2 protected layers at 8 bits)
ram: 4 GB budget -> 13 slots/layer (dense 606 MiB + kv 20 MiB + cache 3191 MiB)
```

It composes with `--ctx` and `--kvq`, in that order: the KV is sized first and the
cache gets what is left, so a longer context costs slots. Below `topk` slots per layer
the cache would thrash inside a single forward, so rather than run unusably slow it
exits and tells you the minimum budget for the current context:

```
$ ./gemma4 ./g4 --ram 2 --kvq
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
TurboQuant paper bounds  ──►  kvq.h              MSE within bounds at 1/2/3/4-bit
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

At `--ram 8 --ctx 4096 --expert-edge 2` that gives 606 MiB dense resident, 4.72 GiB
of experts, 96 MiB KV and 29 of 32 slots/layer. Measured on an Intel Mac at 8 threads:
~24 tok/s prefill, ~16 tok/s decode, ~90% expert-cache hit.

Flags mirror `gemma4` where they mean the same thing (`--chat`, `--serve`, `--system`,
`--raw`, `--temp/--topp/--topk`, `--pin`, `--io`, `--threads`, `--ctx`, `--ram`,
`--kv`/`--kvq` and the individual TurboQuant knobs). Sampling defaults are LFM2.5's
own: temp 0.2, top_k 80. `--batch N` (default 128) sets the prefill batch, which is
what gives each streamed weight row its reuse, so lowering it saves scratch and costs
prefill speed. `--think` forces a reasoning block by pre-filling `<think>`; the chat
template has no thinking toggle, the model decides for itself.

The tokenizer has its own container (`lfmtok.h`, magic `LFTK`) rather than reusing
`g4tok.h`.

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
`--kvq` therefore applies to the full-attention layers only; compressing a layer that
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
