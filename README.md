# zunzuncito — colibrì-style MoE engines for a small-RAM machine

Two engines share this repo, the same q4_0/q8_0 kernels (`q40.h`), the same
TurboQuant KV cache (`kvq.h`), the same OpenAI server, and the same idea — stream
the routed experts from disk under an expert-granular cache instead of leaving it
to the OS page cache:

| binary | model | notes |
|--------|-------|-------|
| `gemma4` | Gemma-4 26B-A4B | 30 attention layers, 128 experts/layer, MTP + DFlash speculation |
| `lfm25`  | [LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) | hybrid 18 short-conv + 6 attention layers, 32 tiny experts/layer, apex-quant mixed precision |

Most of this README is about `gemma4`; see [LFM2.5-8B-A1B](#lfm25--lfm25-8b-a1b) for
the other one.

## gemma4 — Gemma-4 26B-A4B

A Gemma-4 26B-4B [colibrì-style inference engine](https://github.com/JustVugg/colibri) for MacOS (may work on linux, but I never tried). The openai webserver follows closely [samosa-chat (a Qwen3.6-35b colibrì-style inference engine)](https://github.com/deepanwadhwa/samosa-chat). It allows to run the model (quantized or unquantized) also on very RAM-constrained systems (e.g. runs fine on a mac with 8Gb of ram, while doing other things, and also older intel macs).

Gemma-4 26B-A4B is ~25 B params of which only ~3.8 B activate per token.
At q4_0 that splits into **1.3 GB of dense weights** (resident) and **12.9 GB of routed experts** (3840 of them, 3.19 MiB each).
On a 4–8 GB machine the experts don't fit, so we stream them from disk with an expert-granular cache instead of leaving it to the OS page cache.

This engine tries to make it as fast as possible mainly with the following tricks (and more):

1. **Exact prefetch.** Gemma-4's router reads the *raw post-attention residual*, so
   the 8 expert IDs for a layer are known before the dense MLP runs. We route
   first, fire the reads, then compute the MLP while they're in flight. In
   this way there is no need to make a prediction.
2. **Batch-union MoE.** The S tokens of a prefill batch collectively route to at most
   `min(128, 8·S)` distinct experts per layer. We read each once so a 512-token
   prompt drops from 122,880 expert reads to ≤3,840.
3. **A learned pin set.** Expert usage is heavily skewed and stable across prompts.
   Routing counts persist to `usage.bin` and the next run pins the hot set into slots the
   LRU may never evict. Measured on a constrained cache one can easily reach 80% cache hits.

MTP speculative decoding is implemented, but on my system the IO slowdown makes it useless.

Note that this is useful only if you have 4–8 GB of RAM.
With 16 GB you are better off using llama.cpp, though I get similar performances on my system by tuning the IO threads (`--io`) and the compute threads (`--threads`), see below, and I can keep a lot more ram free also at high context (TurboQuant is implemented for the KV and works quite well).

This is the pure-text model: vision is not implemented at all. Moreover the cli does NOT implement tool calling.


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

On macOS the Makefile looks for Macport's or Homebrew's libomp and applies Apple clang's `-Xpreprocessor -fopenmp`.

**Check you got the right kernel.** `q40.h` has AVX2, NEON (with an ARMv8.2 dotprod
fast path — M1 and later), and a portable fallback. If `ARCHFLAGS` is wrong
you silently land on the scalar path and lose several×. `./test_q40` prints which.

## Convert the model

You need the **`-unquantized` QAT checkpoint** (safetensors). From those we re-quantise the weights ourselves, preserving the block structure as appropriate for the engine. When you prepare the model you need to pass the max context length you expect tot use and for how much RAM you want to configure things

```sh
python3 tools/convert_gemma4.py /path/to/gemma-4-26B-A4B-it-qat-unquantized ./g4 \
        --ram 4 --ctx 4096
python3 tools/convert_tokenizer.py /path/to/checkpoint/tokenizer.json ./g4/tok.bin  # make sure you use the corrected one, the official google repo did not yet contain the recent fixes last time I checked
```

This will create the quantized model and all the necessary files in the `g4` folder, that's the only thing you need to run this model.

The `--ram` flag sizes the expert cache and writes `slots_per_layer`
into the container:

| RAM | expert cache | slots/layer (of 128) | % of expert set resident |
|-----|--------------|----------------------|--------------------------|
| 4 GB  | 1.96 GB  | 21  | 16%  |
| 6 GB  | 4.02 GB  | 43  | 34%  |
| 8 GB  | 5.98 GB  | 64  | 50%  |
| 12 GB | 10.0 GB  | 107 | 84%  |
| 16 GB | 11.96 GB | 128 | 100% (fully resident) |

The minimum viable is **2.70 GB** (the floor is `topk`=8 slots/layer).

The converter needs the full checkpoint readable but streams it, so don't
worry about resource usages, it takes very little (beside HDD space).
Expect it to take a while though, since it has to re-quantise every tensor.

## Run

```sh
./gemma4 ./g4 "explain MoE routing"
```

The prompt is a positional argument (the last non-flag argument). To start an
interactive multi-turn session explicitly, use `--chat`:

```sh
./gemma4 ./g4 --chat
```

For compatibility, omitting the prompt also still enters interactive mode. If a
positional prompt is supplied with `--chat`, it is used as the first user turn
before reading subsequent turns from stdin. `--chat` cannot be combined with
`--serve`.
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
curl -N http://127.0.0.1:8642/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"gemma-4-26b-a4b","messages":[{"role":"user","content":"Explain MoE routing."}],"stream":true,"max_tokens":128}'
```

Supported generation controls are `messages` (text `system` and `user` messages),
`stream`, `max_tokens`/`max_completion_tokens`, `temperature`, `top_p`, `top_k`.
Requests are serialized because the Gemma KV/cache state is not concurrently
mutable. The server caches the most recently completed tokenized conversation and
reuses its matching prefix on the next request; send the prior `assistant` response
in `messages` to continue a conversation incrementally. If the prefix differs, the
KV state is rebuilt from the first differing token. The cache is in-memory only and
is discarded when the server exits. The listener binds to `127.0.0.1`.

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

Pinning starts happening from the second run: on the first run the `usage.bin`
is written, containing the necessary routing statistics for later runs to use.
Note that `--pin` is capped at `slots_per_layer - 1` so an expert miss
always has some space to load the expert.

You can use the MTP but have to convert it first. This should be doune into the target's container directory (`./g4` in the examples so far):
```sh
python3 tools/convert_gemma4_mtp.py /path/to/gemma4-assistant-safetensors ./g4
```

After this is done, you can run it with the `--mtp` flag:

```sh
./gemma4 ./g4 --mtp --ndraft 4 "..."
```

There is also `--draft DIR` if you want to put it in a separate directory and keep the KV separate.

Unfortunately MTP does NOT pay off on this engine. It would only if the verification were to use the same experts, but here it may (and usually does) require to load and unload other experts, slowing things down a lot (At least a factor of 2 in my measurements). It may help if you can load everything in RAM, but I don't have any system with enough ram to check.


There is Metal support, which is also off by default (since it also makes everything slower on both Intel and M1).
Our kernel issues one dispatch and one `waitUntilCompleted` **per matvec**, and a token needs thousands of them. At batch size
1 you are paying ~0.5 ms of dispatch latency against ~40 ms of real arithmetic. A GPU
only wins here with far more work per dispatch — whole layers fused into one command
buffer, or large prefill batches — and even then the engine is disk-bound at a 4–8 GB
budget. Fixing it properly would take time, so it is **opt-in** with `--metal`. You can completely turn it off, by building a pure-CPU binary with `make METAL=0`. Any Metal failure falls back to the CPU silently, so there may be issues I did not even notice.

Especially with small RAM budget, this engine is IO bound — roughly 800 MB of
expert reads per token against 7.6 GFLOP of compute, so having a faster matmul
does not help you wait on NVMe. Metal can help still for:

- **prefill**, which is batched and genuinely compute-bound;
- **16 GB**, where the whole container is resident in memory, so there is no
  disk in the loop, and unified memory bandwidth may beat the CPU bandwidth.

## KV-cache compression (TurboQuant V3)

Optional, off by default.
This is gread for context-length: Gemma-4's 25 sliding layers cap their KV at
1024 positions, so they cost a fixed 400 MiB no matter the context. All KV
growth is in the 5 global layers and in constrained systems you feel it.

The `--kvq` flag enables TurboQuant V3 (random rotation + Lloyd-Max, MSE-only, no QJL),
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

Pick k6v4 unless you need >128K context, in which case use k4v2 but you will pay some price for it. Its fidelity is incredible (next to no compression: 0.9997 cos / 94% top-1 vs K4's 0.995 / 81%, you can check with `./test_kvq`).

You can try other configurations by setting manually `--kbits`,
`--vbits`, `--pbits`, `--protect`, `--rwin` if you want to explore more.
Here: `--protect N` protects the first and last N layers with `--pbits` (default 8) rather than f32. This helps because Gemma-4's last layer is global, so protecting it with f32 would cost 1.07 GiB at 128K and undo most of the saving.

Flags: `--kbits` `--vbits` `--pbits` `--protect` `--rwin`.
The residual window (rwin) is necessary at low bit-widths, if you try a 3-4 bit compression without a residual window, you just get garbage.

## Context length

Both engines take `--ctx N` to override the context the container was converted with.
It is applied before the KV cache is sized, so lowering it simply returns RAM and
raising it is allowed with a warning — the container's `ctx` only fixed
`slots_per_layer` (the expert-cache plan), the weights do not care. Raising it does
mean the total exceeds the RAM budget the conversion planned for, hence the warning.

The startup line reports what you actually got, and it accounts for `--kvq`:

```
kv:  567 MiB for ctx 4096  (f32; --kvq would cut this a lot)   # gemma4, container default
kv: 1047 MiB for ctx 16384 (f32; --kvq would cut this a lot)   # gemma4 --ctx 16384
   warning: that is 480 MiB more KV than the container's plan budgeted ...
kv:  238 MiB for ctx 16384 (K6/V4, rwin 128, 2 protected layers at 8 bits)
```

The warning is about BYTES, not about the context number: it fires only when the KV
you are actually allocating exceeds what the conversion budgeted (an f32 KV at the
container's own `ctx`). With `--kvq` a 4x longer context routinely costs *less* than
the plan assumed — 4x the context for a third of the RAM, above — and that is
silent, as it should be.

Under `--serve` this is the server's context window: it is advertised as
`context_length` / `max_model_len` on `GET /v1/models`, and a prompt that leaves no
room for a completion is rejected with both figures in the error. There is no
per-request context parameter — changing the window means restarting with a
different `--ctx`, because the KV cache is allocated once.

## RAM budget

Both engines also take `--ram F` (GB) to re-plan the expert cache at startup, without
reconverting. `slots_per_layer` is the only thing the conversion's `--ram` fixed, and
nothing in the container depends on it: experts are read from `experts.bin` one at a
time by offset and the cache is a plain LRU. So the same container runs on a smaller
or a larger machine than it was converted for.

The runtime re-runs the converter's planner (`plan()` in `tools/convert_*.py`) with
numbers it knows *exactly* rather than had to assume — the resident dense blob and the
KV it just allocated:

```
$ ./lfm25 ./lfm-ct --ram 4 --kvq
kv: 20 MiB for ctx 4096 (K6/V4, rwin 128, 2 protected layers at 8 bits)
ram: 4 GB budget -> 13 slots/layer (dense 606 MiB + kv 20 MiB + cache 3191 MiB)
```

It composes with `--ctx` and `--kvq`, in that order: the KV is sized first and the
cache gets what is left, so a longer context costs slots. Below `topk` slots per layer
the cache would thrash inside a single forward, so that exits with the minimum budget
for the current context instead of running unusably slow:

```
$ ./gemma4 ./g4 --ram 2 --kvq
--ram 2 GB leaves room for 3 experts per layer, below topk=8: this model needs 2.38 GB
```

Fewer slots is a hit-rate cost, not a correctness one: with `--ram` low the engine
streams more experts per token from disk. Check the `expert cache: NN% hit` line at
exit, and see `--pin` under Tuning — the learned hot-expert pin set is what makes a
small cache bearable. Without `--ram` the container's own plan is used and nothing
changes.

## Tuning

You can improve the generation speed quite a lot by playing around with these flags

- **`--io`**: we do 240 expert reads per token at 3.19 MiB each. We default the io threads to 8 but on a good SSD you can go quite higher and improve the expert loading speed.
- **`--pin`**: this sets how much of the cache to freeze vs leave to LRU. At 4 GB there are only 21 slots/layer; the right split depends on the routing distribution of your requests.
- **`--threads`**: sets the compute threads, defaults to 2 but you can go higher depending on how many cores you have.

The engine prints hit rate, expert reads, and speculation acceptance after every run so you can check the effects.

The `make check` runs a few scripts to validate the implementation, you probably can ignore it unless you are modifying the code.

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
(8.3 B total / 1.5 B active). The bet is different from Gemma-4's: the experts are
**tiny** — 32 per layer at `moe_intermediate_size` 1792, ~5.9 MiB each at q4_0, of
which only 4 fire per token — so a miss is cheap and a layer's whole expert set is
only 32 of them. At an 8 GB budget essentially the entire model is cacheable.

It is also a **hybrid** architecture, and that shapes the engine:

- 24 layers, of which only **6 are attention**; the other 18 mix along the sequence
  with a short causal depthwise convolution (`y = C * conv(B * x)`, kernel 3).
  Conv layers hold **no KV at all**, so the KV cache costs a quarter of what the
  layer count suggests (96 MiB at ctx 4096).
- The conv carries a **recurrent state**, and unlike a KV cache it cannot be
  rewound. So prompt-prefix reuse (chat, server) is only taken when the new prompt
  strictly *extends* what was already absorbed; anything else is reprocessed.
- The router is **sigmoid, not softmax**, and `expert_bias` affects *selection
  only* — the weight applied to an expert is the unbiased sigmoid, renormalised.
- First 2 layers use a dense SwiGLU MLP rather than the MoE.

No MTP, no DFlash (this model ships neither). No Metal.

### Mixed precision (apex-quant, ported)

[apex-quant](https://github.com/localai-org/apex-quant) is a llama.cpp recipe: it
does not invent a format, it assigns *different* quant types per tensor class,
exploiting that routed experts are ~97% idle and so tolerate far more error than
the always-on tensors. We keep the idea and drop the GGUF dependency, mapping its
Q3_K..Q8_0 gradient onto the two block formats this engine has kernels for:

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
`--kv`/`--kvq` and the individual TurboQuant knobs). Sampling defaults are LFM2.5's own: temp 0.2,
top_k 80. `--batch N` (default 128) sets the prefill batch: it is what gives each
streamed weight row its reuse, so lowering it costs prefill speed and saves scratch.
`--think` forces a reasoning block by pre-filling `<think>` — the chat template has
no thinking toggle, the model decides for itself.

The tokenizer is a **different family** from Gemma's: GPT-2-style ByteLevel BPE with
the Qwen2/Llama-3 pre-tokenizer split regex and `ignore_merges`, so it has its own
container (`lfmtok.h`, magic `LFTK`) rather than reusing `g4tok.h`.

### Checking it

```sh
make check-lfm25          # add PYTHON=... if numpy/tokenizers live in a venv
```

That runs the tokenizer against HF's (falling back to a bundled reference
implementation, loudly, if `tokenizers` is not installed), then builds a tiny random
fixture and diffs the engine against `tools/lfm25_oracle.py` — an independent numpy
forward pass run on the **dequantised container weights**, so quantisation error
cannot mask an engine bug.

```
numpy oracle          ──►  lfm25.c            1.8e-7   (exact-activation build)
batch prefill         ──►  sequential         bit-identical
expert pinning        ──►  unpinned           bit-identical
HF tokenizer          ──►  lfmtok.h           475/475 exact (incl. 400 fuzzed)
```

## GenAI use warning

Part of the code comes from [colibrì](https://github.com/JustVugg/colibri) and [samosa-chat)](https://github.com/deepanwadhwa/samosa-chat). The implementation for gemma4 follows closely transformer, diverging to implement various optimizations and features and to adapt to the colibrì idea. There is a serious amount of vibecoding involved in this project. I used it to explore the differences between Opus 4.8 and Claude 4.6, ChatGPT-5.6 Luna and Terra, Deepseek V4 Pro and Flash, and their respective costs.
