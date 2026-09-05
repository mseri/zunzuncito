# zunzuncito

**zunzuncito** is a fast C/C++ inference engine for Mixture-of-Experts (MoE) models on RAM-constrained machines (macOS and Linux).

Instead of relying on the OS page cache, zunzuncito streams routed experts directly from disk using an expert-granular cache. Dense weights and hot experts remain in memory, while inactive experts are streamed as needed.

Ready-made containers and converted weights are published at [huggingface.co/mseri](https://huggingface.co/mseri), pre-configured for 4Gb of RAM and 4096 token of context (this can be changed at runtime using the flags described later in this README).

## Supported models

| Program | Model ID from `/v1/models` | Shape and extras |
|---|---|---|
| `gemma4` | `gemma-4-26b-a4b` | Gemma-4 26B-A4B; 30 attention layers, 128 experts per layer, top-8 routing; MTP and DFlash drafting |
| `lfm25` | `lfm2.5-8b-a1b` | LFM2.5-8B-A1B; 18 short-convolution layers plus 6 attention layers, 32 experts per layer; DSpark drafting |
| `maple` | `maple-preview` | Maple-preview 20B-A1B; native ternary `tq2` weights, 256 experts per layer, sliding and full attention |
| `ling` | `ling-3.0-tiny` | Ling-3.0-tiny 7.9B-A1.3B; 3:1 KDA/MLA mix, 128 routed experts plus one shared expert per layer, absorbed MLA cache |

Since this is meant for tight memory setups, we follow a few technical tricks:

1. **Exact MoE Prefetching**: Routers evaluate the residual before the dense/attention block finishes, so disk reads for selected experts are scheduled ahead of time.
2. **Batch-Union Streaming**: During prompt prefill, duplicate expert reads across tokens in the batch are deduplicated, drastically cutting disk I/O.
3. **Learned Hot-Expert Pinning**: Expert activation patterns are skewed. Routing statistics persist across runs in `usage.bin`, allowing frequently used experts to remain pinned in RAM.
4. **KVarN KV Cache Compression**: Outlier-aware tile-based KV quantization (`kvarn.h`) keeps long contexts lightweight.
5. **FlashHead**: Optional clustered centroid approximation for large embedding/lm_head projections.

## How It Works

## Building

You only need a C99 compiler (clang or gcc). OpenMP is recommended for multi-threading and Metal is optional (and recommended for Apple Silicon).

### Quick Build

```sh
make
```

### Build Options & Custom Flags

```sh
# Override compiler and OpenMP flags
make CC=gcc-14 OMPFLAGS=-fopenmp OMPLIBS=-fopenmp

# Optimize for Apple Silicon architecture
make ARCHFLAGS="-mcpu=apple-m3"

# Build without OpenMP (single-threaded)
make OMP=0

# Build CPU-only (disable Metal kernels on macOS)
make METAL=0

# Run full test suite
make check
```

## Get a model

## Model Conversion & Setup

You can download ready-to-use containers from [huggingface.co/mseri](https://huggingface.co/mseri), or convert Hugging Face checkpoints locally using the provided Python scripts in `tools/`:

### Gemma-4

```sh
python3 tools/convert_gemma4.py /path/to/gemma-4-26B-A4B-it-qat-unquantized ./g4-ct --ram 4 --ctx 4096
python3 tools/convert_tokenizer.py /path/to/checkpoint/tokenizer.json ./g4-ct/tok.bin
```

### LFM2.5

```sh
python3 tools/convert_lfm25.py /path/to/LFM2.5-8B-A1B ./lfm-ct --ram 4 --ctx 4096
python3 tools/convert_lfm_tokenizer.py /path/to/LFM2.5-8B-A1B/tokenizer.json ./lfm-ct/tok.bin
```

### Maple

```sh
python3 tools/convert_maple.py /path/to/maple-preview-mlx ./maple-ct --ram 4 --ctx 4096
python3 tools/convert_lfm_tokenizer.py /path/to/maple-preview-mlx/tokenizer.json ./maple-ct/tok.bin
```

### Ling

```sh
python3 tools/convert_ling.py /path/to/Ling-3.0-tiny ./ling-ct --ram 8 --ctx 8192
```

Note that ram and ctx precompute how to distribute objects in memory but can be modified at runtime (see below for the appropriate flags).

## Usage & CLI Flags

All four executables follow a common command line interface:

```sh
./gemma4 <model_dir> [flags...] [prompt]
./lfm25  <model_dir> [flags...] [prompt]
./maple  <model_dir> [flags...] [prompt]
./ling   <model_dir> [flags...] [prompt]
```

### Common Modes

```sh
# Generate one answer.
./gemma4 ./g4 "Explain Mixture of Experts in simple terms."

# Keep a local chat going.
./gemma4 ./g4 --chat

# Serve the OpenAI-style API on port 8484.
./gemma4 ./g4 --serve --port 8484
```

## Command-Line Flags Reference

Use `--help` on the program you are running. We collect all the flags below.

### Conversation and sampling

| Flag | Description |
|---|---|
| `<dir>` | Path to the directory containing model weights and manifest (*required*). |
| `[prompt]` | Initial user prompt (positional). If omitted, interactive chat mode is started. |
| `--chat` | Launch interactive multi-turn session on stdin/stdout with context caching. |
| `--system S` | Set system prompt. |
| `--nothink` | Disable thinking/reasoning output (reasoning is on by default). |
| `--raw` | Feed the prompt verbatim, bypassing the chat template formatting. |
| `--max_tokens N` | Maximum number of tokens to generate (default: `2048`). |
| `--temp F` | Sampling temperature. `0` selects greedy argmax. |
| `--topp F` | Nucleus-sampling cutoff, such as `0.95`. |
| `--topk N` | Top-k limit. |
| `--penalty F` | Repetition penalty; `1.0` turns it off. |

### Memory, I/O, and compute

| Flag | Description |
|---|---|
| `--ram F` | Set the total RAM budget in GB and recalculate resident expert slots. |
| `--ctx N` | Replace the container's context length. |
| `--pin N` | Keep the top `N` experts from `usage.bin` resident for each layer. |
| `--kv PRESET` | Choose KV quantization: `off`, `kvarn_k4v2_g128`, `kvarn_k4v4_g128`, `kvarn_k4v2_g64`, or `kvarn_k4v4_g64` (default: `kvarn_k4v2_g128`). |
| `--threads N` | Number of OpenMP compute threads. |
| `--io N` | Background I/O threads for expert reads; default `8`. |
| `--batch N` | Prefill batch size (default: `128`). |
| `--nobatch` | Prefill tokens one at a time. |
| `--metal` | Use Apple Metal offload. It starts disabled. |

### FlashHead and drafting

| Flag | Available in | Description |
|---|---|---|
| `--flash` | `gemma4`, `lfm25`, `ling` | Approximate `lm_head` with centroid clusters. |
| `--noflash` | `maple` | Force the exact `lm_head`. |
| `--probes N` | all programs | Evaluate this many FlashHead clusters per token. |
| `--mtp` | `gemma4` | Turn on Multi-Token Prediction drafting. |
| `--dflash` | `gemma4` | Turn on DFlash block drafting. |
| `--dspark` | `lfm25` | Turn on DSpark block-parallel drafting. |
| `--draft DIR` | `gemma4` | Use a separate drafter container. |
| `--ndraft N` | `gemma4`, `lfm25` | Draft tokens proposed per step. |
| `--drefine N` | `gemma4`, `lfm25` | Extra denoising passes through a draft block. |
| `--dfreeze F` | `lfm25` | Confidence needed to freeze a token between refinement passes. |
| `--dconf F` | `lfm25` | Confidence cutoff for leaving a proposal early. |
| `--no-markov` | `lfm25` | Do not use the bigram Markov head while DSpark drafts. |

#### HTTP Server
| Flag | Description |
|---|---|
| `--serve` | Launch an OpenAI-compatible HTTP server. |
| `--port N` | TCP port for the HTTP server (default: `8484`). |

## HTTP server

Start the OpenAI server with `--serve`. It loads the model and offers the following endpoints.

| Method and path | Behavior |
|---|---|
| `GET /v1/models` | Returns the loaded model and its ID. |
| `POST /v1/chat/completions` | Handles chat completions, including `stream: true` Server-Sent Events. `content` accepts a string or text-part array; image and other non-text parts are ignored. |
| `GET /props` | Returns llama.cpp-style server properties, including context size and model alias. |
| `POST /props` | Returns `501 not_supported_error`; global properties cannot be changed. |
| `GET /models` | Returns a one-model llama.cpp router-style catalog. |
| `GET /models/sse` | Holds an idle router-style status connection open. |
| `GET /healthz` | Health check. |
| `POST /v1/cancel` | Stops a generation in progress. |
| `POST /v1/shutdown` | Stops the server cleanly. |

`/v1/models` reports `gemma-4-26b-a4b`, `lfm2.5-8b-a1b`, `maple-preview`, or `ling-3.0-tiny`, according to the executable.



All four servers accept OpenAI-style `tools`, `tool_calls`, and `role: "tool"` results.
When tools are supplied, the server waits for a complete tool call instead of streaming raw model tokens. A real call yields `"finish_reason": "tool_calls"`.

### Example Request

```sh
curl -N http://127.0.0.1:8484/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-26b-a4b",
    "messages": [
      {"role": "user", "content": "Explain how MoE routing works."}
    ],
    "stream": true,
    "max_tokens": 128
  }'
```

## Optional speculative-decoding

Convert the target mode, then run the matching converter below with the
checkpoint directory and target container directory. The extra files can go into
the target container, so there is no second model directory to pass at runtime.
If you download the pre-made containers from Hugging Face, the drafter files are already included.

#### Gemma-4 MTP

[Download the Gemma-4 assistant checkpoint](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant).
It supplies the MTP head used by `gemma4` and shares the target model's KV
cache. Convert it into `./g4-ct`:

```sh
python3 tools/convert_gemma4_mtp.py \
  /path/to/gemma-4-26B-A4B-it-assistant ./g4-ct
```

Run it with `--mtp`:

```sh
./gemma4 ./g4-ct --mtp --ndraft 4
```

The converter writes `mtp.bin`, `mtp.idx`, `mtp.cfg.json`, and
`mtp.manifest.txt`. The assistant must have the same hidden size and vocabulary
as the Gemma-4 target.

#### LFM2.5 DSpark

[Download the LFM2.5 DSpark checkpoint](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-DSpark).
The checkpoint is a drafter for LFM2.5. Install it in `./lfm-ct`:

```sh
python3 tools/convert_lfm25_dspark.py \
  /path/to/LFM2.5-8B-A1B-DSpark ./lfm-ct
```

Then enable DSpark:

```sh
./lfm25 ./lfm-ct --dspark --ndraft 9
```

The converter writes `dspark.bin`, `dspark.idx`, `dspark.cfg.json`, and
`dspark.manifest.txt`. It uses q4 drafter matrices by default. For q8 matrices,
which use more resident memory, add `--draft-bits 8` to the conversion command:

```sh
python3 tools/convert_lfm25_dspark.py \
  /path/to/LFM2.5-8B-A1B-DSpark ./lfm-ct --draft-bits 8
```


## License

See [LICENSE](LICENSE) for details.
