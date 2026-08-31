# zunzuncito

**zunzuncito** is a fast C/C++ inference engine for Mixture-of-Experts (MoE) models on RAM-constrained machines (macOS and Linux).

Instead of relying on the OS page cache, zunzuncito streams routed experts directly from disk using an expert-granular cache. Dense weights and hot experts remain in memory, while inactive experts are streamed as needed.

Pre-converted model weights and containers are available on Hugging Face:
👉 **[https://huggingface.co/mseri](https://huggingface.co/mseri)**

---

## Supported Models & Binaries

| Executable | Target Architecture | Model `id` (`/v1/models`) | Key Features |
|---|---|---|---|
| `gemma4` | **Gemma-4 26B-A4B** | `gemma-4-26b-a4b` | 30 attention layers, 128 experts/layer, top-8 routing, MTP & DFlash speculation |
| `lfm25` | **LFM2.5-8B-A1B** | `lfm2.5-8b-a1b` | Hybrid 18 short-conv + 6 attention layers, 32 experts/layer, DSpark speculation |
| `maple` | **Maple-preview 20B-A1B** | `maple-preview` | Native ternary (`tq2`) weights, 256 experts/layer, sliding + full attention |
| `ling` | **Ling-3.0-tiny 7.9B-A1.3B** | `ling-3.0-tiny` | 3:1 KDA/MLA hybrid, 128 experts + 1 shared expert/layer, absorbed MLA cache |

---

## How It Works

1. **Exact MoE Prefetching**: Routers evaluate the residual before the dense/attention block finishes, so disk reads for selected experts are scheduled ahead of time.
2. **Batch-Union Streaming**: During prompt prefill, duplicate expert reads across tokens in the batch are deduplicated, drastically cutting disk I/O.
3. **Learned Hot-Expert Pinning**: Expert activation patterns are skewed. Routing statistics persist across runs in `usage.bin`, allowing frequently used experts to remain pinned in RAM.
4. **KVarN KV Cache Compression**: Outlier-aware tile-based KV quantization (`kvarn.h`) keeps long contexts lightweight.
5. **FlashHead**: Optional clustered centroid approximation for large embedding/lm_head projections.

---

## Building

A C99 compiler (clang or gcc) and OpenMP (recommended for multi-threading) are required.

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

---

## Model Conversion & Setup

You can download ready-to-use containers from [huggingface.co/mseri](https://huggingface.co/mseri), or convert Hugging Face checkpoints locally using the provided Python scripts in `tools/`:

### Gemma-4
```sh
python3 tools/convert_gemma4.py /path/to/gemma-4-26B-A4B-it-qat-unquantized ./g4 --ram 4 --ctx 4096
python3 tools/convert_tokenizer.py /path/to/checkpoint/tokenizer.json ./g4/tok.bin
```

### LFM2.5
```sh
python3 tools/convert_lfm25.py /path/to/LFM2.5-8B-A1B ./lfm-ct --ram 8 --ctx 4096
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

---

## Usage & CLI Flags

All four executables follow a consistent command line interface:

```sh
./gemma4 <model_dir> [flags...] [prompt]
./lfm25  <model_dir> [flags...] [prompt]
./maple  <model_dir> [flags...] [prompt]
./ling   <model_dir> [flags...] [prompt]
```

### Common Modes

- **One-shot generation:**
  ```sh
  ./gemma4 ./g4 "Explain Mixture of Experts in simple terms."
  ```
- **Interactive multi-turn chat:**
  ```sh
  ./gemma4 ./g4 --chat
  ```
- **OpenAI-compatible HTTP Server:**
  ```sh
  ./gemma4 ./g4 --serve --port 8484
  ```

---

### Command-Line Flags Reference

#### General & Chat Flags
| Flag | Description |
|---|---|
| `<dir>` | Path to the directory containing model weights and manifest (*required*). |
| `[prompt]` | Initial user prompt (positional). If omitted, interactive chat mode is started. |
| `--chat` | Launch interactive multi-turn session on stdin/stdout with context caching. |
| `--system S` | Set system prompt. |
| `--nothink` | Disable thinking/reasoning output (reasoning is on by default). |
| `--raw` | Feed the prompt verbatim, bypassing the chat template formatting. |
| `--max_tokens N` | Maximum number of tokens to generate (default: `2048`). |

#### Sampling Parameters
| Flag | Description |
|---|---|
| `--temp F` | Sampling temperature (`--temp 0` selects greedy argmax). |
| `--topp F` | Top-p nucleus sampling threshold (e.g. `0.95`). |
| `--topk N` | Top-k sampling limit. |
| `--penalty F` | Repetition penalty (`1.0` = disabled). |

#### Memory & Cache Management
| Flag | Description |
|---|---|
| `--ram F` | Total RAM budget in GB. Recalculates resident expert cache slots. |
| `--ctx N` | Override context length. |
| `--pin N` | Pin the top-N hot experts per layer into RAM from `usage.bin`. |
| `--kv PRESET` | Select KVarN KV cache quantization preset (`off`, `kvarn_k4v2_g128`, `kvarn_k4v4_g128`, `kvarn_k4v2_g64`, `kvarn_k4v4_g64`). |

#### Performance & Hardware
| Flag | Description |
|---|---|
| `--threads N` | Number of compute threads (OpenMP). |
| `--io N` | Number of background asynchronous I/O threads for streaming experts (default: `8`). |
| `--batch N` | Prefill batch size (default: `128`). |
| `--nobatch` | Process prefill tokens sequentially one-by-one. |
| `--metal` | Enable Apple Metal GPU offloading (disabled by default). |

#### FlashHead Approximation
| Flag | Description |
|---|---|
| `--flash` | Enable approximate `lm_head` using centroid clustering. |
| `--noflash` | Force exact `lm_head` computation (Maple). |
| `--probes N` | Number of FlashHead clusters to evaluate per token. |
| `--flash-check` | Run exact head side-by-side with FlashHead to report agreement rate. |

#### Speculative Decoding
| Flag | Executable | Description |
|---|---|---|
| `--mtp` | `gemma4` | Enable Multi-Token Prediction speculation. |
| `--dflash` | `gemma4` | Enable DFlash block speculation. |
| `--dspark` | `lfm25` | Enable DSpark block-parallel speculative decoding. |
| `--draft DIR` | `gemma4` | Path to separate drafter model container. |
| `--ndraft N` | `gemma4`/`lfm25` | Number of speculative draft tokens proposed per step. |
| `--drefine N` | `gemma4`/`lfm25` | Extra denoising passes across the draft block. |
| `--dfreeze F` | `lfm25` | Confidence threshold to freeze tokens between refinement passes. |
| `--dconf F` | `lfm25` | Early-exit confidence cutoff for speculative proposals. |
| `--no-markov` | `lfm25` | Disable bigram Markov head during DSpark drafting. |

#### HTTP Server
| Flag | Description |
|---|---|
| `--serve` | Launch an OpenAI-compatible HTTP server. |
| `--port N` | TCP port for the HTTP server (default: `8484`). |

#### Validation & Diagnostics
| Flag | Description |
|---|---|
| `--check` | Verify numerical correctness against stored reference logits/oracle. |
| `--check-gpu` | Verify Metal GPU kernel computations against CPU reference. |
| `--help`, `-h` | Display help and usage summary. |

---

## OpenAI Server Endpoints

When running with `--serve`:

- `GET /v1/models` — Lists the loaded model and its advertised `id`:
  - `gemma4` $\rightarrow$ `"id": "gemma-4-26b-a4b"`
  - `lfm25` $\rightarrow$ `"id": "lfm2.5-8b-a1b"`
  - `maple` $\rightarrow$ `"id": "maple-preview"`
  - `ling` $\rightarrow$ `"id": "ling-3.0-tiny"`
- `POST /v1/chat/completions` — Handles chat requests (supports both standard JSON and `stream: true` Server-Sent Events).
- `GET /healthz` — Health check endpoint.
- `POST /v1/cancel` — Abort ongoing generation.
- `POST /v1/shutdown` — Gracefully stop server.

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

---

## License

See [LICENSE](LICENSE) for details.
