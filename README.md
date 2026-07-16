# zunzuncito — Gemma-4 26B-A4B colibrì engine for a small-RAM machine

A Gemma-4 26B-4B [colibrì-style inference engine](https://github.com/JustVugg/colibri) for MacOS (may work on linux, but I never tried). The openai webserver follows closely [samosa-chat (a Qwen3.6-35b colibrì-style inference engine)](https://github.com/deepanwadhwa/samosa-chat). It allows to run the model (quantized or unquantized) also on very RAM-constrained systems (e.g. runs fine on a mac with 8Gb of ram, while doing other things, and also older intel macs).

Gemma-4 26B-A4B is ~25 B params of which only ~3.8 B activate per token.
At q4_0 that splits into **1.3 GB of dense weights** (resident) and **12.9 GB of routed experts** (3840 of them, 3.19 MiB each).
On a 4–8 GB machine the experts don't fit, so we stream them from disk with an expert-granular cache instead of leaving it to the OS page cache.

This engine tries to make it as fast as possible mainly with the following tricks (and more):

1. **Exact prefetch.** Gemma-4's router reads the *raw post-attention residual*, so
   the 8 expert IDs for a layer are known **before** the dense MLP runs. We route
   first, fire the reads, then compute the MLP while they're in flight. In
   this way there is no need to make a prediction.
2. **Batch-union MoE.** The S tokens of a prefill batch collectively route to at most
   `min(128, 8·S)` distinct experts per layer. We read each **once**. A 512-token
   prompt drops from 122,880 expert reads to ≤3,840.
3. **A learned pin set.** Expert usage is heavily skewed and stable across prompts.
   Routing counts persist to `usage.bin` and the next run pins the hot set into slots the
   LRU may never evict. Measured on a constrained cache: **49.8% → 74.6% hit rate**.

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

The prompt is a positional argument (the last non-flag argument), and if 
missing enters into an interactive multi-turn mode.
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
| `--system S` | system prompt |
| `--think` | enable reasoning (injects `<\|think\|>`) |
| `--raw` | skip the chat template, feed the prompt verbatim |
| `--temp F` `--topp F` `--topk N` | sampling; defaults are Gemma-4's own: 1.0 / 0.95 / 64. `--temp 0` = greedy |
| `--pin N` | pin N experts/layer from `usage.bin`. Needs one prior run to learn |
| `--draft DIR` `--ndraft N` | MTP speculative decoding against a drafter container |
| `--io N` | I/O threads (default 8) |
| `--check` | teacher-forcing validation against a reference |

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

## GenAI use warning

Part of the code comes from [colibrì](https://github.com/JustVugg/colibri) and [samosa-chat)](https://github.com/deepanwadhwa/samosa-chat). The implementation for gemma4 follows closely transformer, diverging to implement various optimizations and features and to adapt to the colibrì idea. There is a serious amount of vibecoding involved in this project. I used it to explore the differences between Opus 4.8 and Claude 4.6, ChatGPT-5.6 Luna and Terra, Deepseek V4 Pro and Flash, and their respective costs.
