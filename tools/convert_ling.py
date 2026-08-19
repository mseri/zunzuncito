#!/usr/bin/env python3
"""
convert_ling.py — inclusionAI/Ling-3.0-tiny (fp8 HF safetensors) -> ling container.

Same container shape as convert_lfm25.py (manifest.txt + dense.bin + experts.bin, one
expert = one contiguous 4096-aligned pread), and the same mixed-precision idea. Three
things differ, and each of them is where the work is.

1. THE SOURCE IS ALREADY QUANTISED. The checkpoint is fp8 e4m3 with ue8m0 block scales
   over [128,128] tiles (`<tensor>.weight_scale_inv`), except for the tensors listed in
   quantization_config.modules_to_not_convert, which stayed bf16. So every weight is
   dequantised to f32 first and then re-quantised into the two block formats this engine
   has kernels for. There is no way around the round trip: q4_0/q8_0 are per-32-element
   row blocks and fp8's tiles are 128x128 across BOTH axes.

2. THE ARCHITECTURE IS HYBRID, but along a different seam than LFM2.5's. Layers with
   (idx+1) % layer_group_size == 0 are Multi-head Latent Attention; the rest are Kimi
   Delta Attention, a linear/recurrent operator whose state is 16x128x128 per layer and
   does not grow with the context. `layer_types` here means MLA-vs-KDA. Layer 0 carries a
   dense SwiGLU MLP (first_k_dense_replace) and 1..23 carry a router + 128 routed experts
   + one always-on shared expert.

3. kv_b_proj IS PRE-SPLIT AND PRE-TRANSPOSED. The engine runs MLA in absorbed form: it
   caches the 512-wide latent rather than expanded per-head keys and values, so it needs
   W_k TRANSPOSED (to fold into the query) and W_v as-is (to unfold the attention output).
   Doing the split here means the engine never reshapes at runtime, and the numpy oracle
   and the C engine consume the same bytes.

The precision gradient follows lfm25's reasoning -- routed experts are ~94% idle and
tolerate error the always-on path does not -- with one addition specific to this model:

    always-on (KDA/MLA projections, shared experts, dense MLP)  -> q8_0
    embeddings and lm_head                                      -> q8_0
    routed experts                                              -> q4_0
    norms, A_log, dt_bias, b_proj, g_proj (MLA), router,
    expert_bias, depthwise conv taps                            -> f32

The KDA layers are the reason the always-on floor is q8_0 rather than q4_0: their
recurrence carries a 128x128 state across the whole sequence, so projection error
compounds there in a way it does not inside a softmax layer. --expert-edge and
--embed-q4 exist to move off those defaults.

The depthwise conv taps are f32 for a structural reason, not a quality one: the tensor is
[2048, 1, 4] and a block format needs the contracted dimension to be a multiple of 32.

    python3 tools/convert_ling.py ./ling-tiny-fp8 ./ling-ct --ram 8 --ctx 8192
    python3 tools/convert_ling.py --fixture ./ling-fix          # tiny random model
"""
import argparse, json, os, struct, subprocess, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flashhead

QK = 32
BLK40, BLK80 = 18, 34
FMT_F32, FMT_Q40, FMT_Q80, FMT_I32 = 0, 1, 2, 5


# q4_0 / q8_0 (mirror q40.h bit-for-bit; same code as convert_lfm25.py)
def q40_quant_rows(w: np.ndarray) -> bytes:
    O, I = w.shape
    assert I % QK == 0, f"I={I} not a multiple of {QK}"
    nb = I // QK
    x = w.reshape(O, nb, QK).astype(np.float32)
    ai = np.abs(x).argmax(axis=2)
    mx = np.take_along_axis(x, ai[:, :, None], axis=2)[:, :, 0]
    d = (mx / -8.0).astype(np.float32)
    d = d.astype(np.float16).astype(np.float32)     # round d to fp16 before deriving id
    idv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0).astype(np.float32)
    q = np.floor(x * idv[:, :, None] + 8.5).astype(np.int32)
    np.clip(q, 0, 15, out=q)
    nib = (q[:, :, :16].astype(np.uint8) | (q[:, :, 16:].astype(np.uint8) << 4))
    out = np.empty((O, nb, BLK40), dtype=np.uint8)
    out[:, :, 0:2] = d.astype(np.float16).view(np.uint8).reshape(O, nb, 2)
    out[:, :, 2:] = nib
    return out.tobytes()


def q40_dequant_rows(buf, O, I):
    nb = I // QK
    a = np.frombuffer(buf, np.uint8).reshape(O, nb, BLK40)
    d = a[:, :, 0:2].copy().view(np.float16).astype(np.float32).reshape(O, nb)
    nib = a[:, :, 2:]
    lo = (nib & 0x0F).astype(np.int32) - 8
    hi = (nib >> 4).astype(np.int32) - 8
    w = np.concatenate([lo, hi], 2).astype(np.float32)
    return (w * d[:, :, None]).reshape(O, I)


def q80_quant_rows(w: np.ndarray) -> bytes:
    O, I = w.shape
    assert I % QK == 0, f"I={I} not a multiple of {QK}"
    nb = I // QK
    x = w.reshape(O, nb, QK).astype(np.float32)
    amax = np.abs(x).max(axis=2)
    d = (amax / 127.0).astype(np.float16).astype(np.float32)
    idv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0).astype(np.float32)
    q = np.rint(x * idv[:, :, None]).astype(np.int32)   # rint = half-to-even, as lrintf
    np.clip(q, -127, 127, out=q)
    out = np.empty((O, nb, BLK80), dtype=np.uint8)
    out[:, :, 0:2] = d.astype(np.float16).view(np.uint8).reshape(O, nb, 2)
    out[:, :, 2:] = q.astype(np.int8).view(np.uint8)
    return out.tobytes()


def q80_dequant_rows(buf, O, I):
    nb = I // QK
    a = np.frombuffer(buf, np.uint8).reshape(O, nb, BLK80)
    d = a[:, :, 0:2].copy().view(np.float16).astype(np.float32).reshape(O, nb)
    q = a[:, :, 2:].view(np.int8).astype(np.float32)
    return (q * d[:, :, None]).reshape(O, I)


def qbytes(fmt, O, I):
    if fmt == FMT_Q40:
        return O * (I // QK) * BLK40
    if fmt == FMT_Q80:
        return O * (I // QK) * BLK80
    return O * I * 4


def quant_rows(fmt, w):
    if fmt == FMT_Q40:
        return q40_quant_rows(w)
    if fmt == FMT_Q80:
        return q80_quant_rows(w)
    return np.ascontiguousarray(w, dtype=np.float32).tobytes()


def dequant_rows(fmt, buf, O, I):
    if fmt == FMT_Q40:
        return q40_dequant_rows(buf, O, I)
    if fmt == FMT_Q80:
        return q80_dequant_rows(buf, O, I)
    return np.frombuffer(buf, np.float32).reshape(O, I)


# fp8
def _e4m3_table():
    """byte -> f32 for OCP float8_e4m3fn: 1-4-3, exponent bias 7, no infinities,
    S.1111.111 is NaN. Built once as a 256-entry LUT so dequantising a 6 GB checkpoint
    is a gather rather than bit arithmetic per element."""
    t = np.zeros(256, np.float32)
    for b in range(256):
        s = -1.0 if (b >> 7) else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0xF and m == 0x7:
            t[b] = np.nan
        elif e == 0:
            t[b] = s * (m / 8.0) * 2.0 ** -6          # subnormal
        else:
            t[b] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return t


E4M3 = _e4m3_table()


def _e8m0_table():
    """byte -> f32 for ue8m0: a bare power of two, 2^(e-127). 255 is NaN. Values below
    2^-126 would be subnormal as f32 exponents but the scales here never go there; the
    table keeps them exact anyway by construction."""
    t = np.zeros(256, np.float32)
    for b in range(256):
        t[b] = np.nan if b == 255 else np.float32(2.0) ** (b - 127)
    return t


E8M0 = _e8m0_table()


def fp8_dequant(codes: np.ndarray, scales: np.ndarray, block) -> np.ndarray:
    """[O,I] fp8 codes with [ceil(O/bo), ceil(I/bi)] block scales -> f32 [O,I].

    The scale grid is a ceiling, so the last row/column block may be partial. Repeating
    then trimming is simpler than an index gather and costs one temporary."""
    bo, bi = block
    O, I = codes.shape
    assert scales.shape == ((O + bo - 1) // bo, (I + bi - 1) // bi), \
        f"scale grid {scales.shape} does not match [{O},{I}] at block {block}"
    s = np.repeat(np.repeat(scales, bo, axis=0), bi, axis=1)[:O, :I]
    return codes * s


# safetensors
_DT = {"F32": np.float32, "F16": np.float16, "I8": np.int8, "U8": np.uint8,
       "I32": np.int32, "I64": np.int64}


class Shards:
    """Streaming reader. Keeps one open handle per shard and reads a tensor's byte range
    on demand -- the checkpoint is 8 GB and must never be resident.

    Unlike convert_lfm25.py's, this one knows about fp8: `get` returns the raw decoded
    values and `getw` applies the block scales when a `_scale_inv` companion exists."""

    def __init__(self, d):
        self.d = d
        p = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(p):
            self.wmap = json.load(open(p))["weight_map"]
        else:
            self.wmap = {}
            for f in sorted(os.listdir(d)):
                if f.endswith(".safetensors"):
                    with open(os.path.join(d, f), "rb") as fh:
                        n = struct.unpack("<Q", fh.read(8))[0]
                        h = json.loads(fh.read(n))
                    for k in h:
                        if k != "__metadata__":
                            self.wmap[k] = f
        self.hdr, self.fh = {}, {}
        self.block = (128, 128)

    def _open(self, fn):
        if fn not in self.fh:
            f = open(os.path.join(self.d, fn), "rb")
            n = struct.unpack("<Q", f.read(8))[0]
            self.hdr[fn] = (json.loads(f.read(n)), 8 + n)
            self.fh[fn] = f
        return self.fh[fn], self.hdr[fn]

    def has(self, n):
        return n in self.wmap

    def get(self, n):
        if n not in self.wmap:
            sys.exit(f"missing tensor {n}")
        f, (hdr, base) = self._open(self.wmap[n])
        e = hdr[n]
        s, t = e["data_offsets"]
        f.seek(base + s)
        raw = f.read(t - s)
        dt = e["dtype"]
        if dt == "BF16":
            a = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dt == "F8_E4M3":
            a = E4M3[np.frombuffer(raw, np.uint8)]
        elif dt == "F8_E8M0":
            a = E8M0[np.frombuffer(raw, np.uint8)]
        else:
            a = np.frombuffer(raw, _DT[dt]).astype(np.float32)
        return a.reshape(e["shape"])

    def getw(self, n):
        """A weight, dequantised. fp8 tensors carry a `<n>_scale_inv` block-scale grid;
        the ones in modules_to_not_convert do not and come back verbatim."""
        w = self.get(n)
        sn = n + "_scale_inv"
        if self.has(sn):
            w = fp8_dequant(w, self.get(sn), self.block)
        if not np.isfinite(w).all():
            sys.exit(f"{n}: dequantised to non-finite values")
        return w


# dense writer
class Dense:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.idx, self.off = {}, 0

    def add(self, name, w, fmt):
        w = np.ascontiguousarray(w, dtype=np.float32)
        if w.ndim == 1:
            w = w.reshape(1, -1)
        assert w.ndim == 2, f"{name}: ndim {w.ndim}"
        b = quant_rows(fmt, w)
        self.f.write(b)
        self.idx[name] = {"off": self.off, "len": len(b), "fmt": fmt,
                          "shape": [int(w.shape[0]), int(w.shape[1])]}
        self.off += len(b)
        return b

    def add_i32(self, name, w):
        w = np.ascontiguousarray(w, dtype=np.int32)
        if w.ndim == 1:
            w = w.reshape(1, -1)
        b = w.tobytes()
        self.f.write(b)
        self.idx[name] = {"off": self.off, "len": len(b), "fmt": FMT_I32,
                          "shape": [int(w.shape[0]), int(w.shape[1])]}
        self.off += len(b)
        return b

    def close(self):
        self.f.close()


# RAM planner
def plan(cfg, dense_bytes, esz_by_layer, ctx, ram_gb):
    """slots/layer for the per-layer LRU expert cache under a RAM budget.

    The resident cost of the context is unusual here and worth spelling out: only the
    MLA layers hold anything that grows with it, and in absorbed form each caches
    kv_lora + qk_rope floats per position rather than expanded per-head keys and values.
    The KDA layers hold a state that is the same size at position 1 and position 131072.
    """
    NE = cfg["n_experts"]
    moe = [li for li, e in enumerate(esz_by_layer) if e]
    esz_max = max((esz_by_layer[li] for li in moe), default=0)
    nmoe = len(moe)

    n_mla = sum(cfg["layer_types"])
    n_kda = cfg["n_layers"] - n_mla
    # f32 accounting: --kv turns the latent into KVarN codes and shrinks this, so
    # planning against the uncompressed size is the conservative direction.
    kvbytes = n_mla * ctx * (cfg["kv_lora"] + cfg["qk_rope"]) * 4
    # KDA recurrent state + short-conv state, both context-independent
    kda_state = n_kda * cfg["n_heads"] * cfg["head_dim"] * cfg["head_dim"] * 4
    kda_state += n_kda * 3 * max(0, cfg["conv_L"] - 1) * cfg["n_heads"] * cfg["head_dim"] * 4

    scratch = 192 << 20
    avail = int(ram_gb * (1 << 30)) - dense_bytes - kvbytes - kda_state - scratch
    per = min(NE, max(0, avail // esz_max) // max(1, nmoe)) if esz_max else NE
    total = sum(esz_by_layer[li] * NE for li in moe)
    cache = int(per) * sum(esz_by_layer[li] for li in moe)
    return {
        "expert_bytes_max": esz_max, "dense_bytes": dense_bytes, "kv_bytes": kvbytes,
        "kda_state_bytes": kda_state,
        "slots_per_layer": int(per), "cache_bytes": cache,
        "total_expert_bytes": total, "n_moe_layers": nmoe,
        "ok": per >= cfg["topk"],
        "min_ram_gb": (dense_bytes + kvbytes + kda_state + scratch
                       + cfg["topk"] * nmoe * esz_max) / (1 << 30),
    }


# fixture
def make_fixture(dst):
    """A tiny random BailingMoeV3 with the same STRUCTURE -- the 3:1 KDA/MLA stack, a
    dense layer 0, MoE layers with a shared expert, and the grouped sigmoid router --
    and absurd dimensions. It exists so the numpy oracle and the C engine can be diffed
    without an 8 GB download and without torch.

    8 layers so the (idx+1) % 4 pattern produces two MLA layers and six KDA ones, and so
    the dense layer and the MoE layers both appear. n_experts 8 over n_group 2 /
    topk_group 1 exercises the group-limited top-k on a grid small enough to check by
    hand.

    Every dimension that any matmul CONTRACTS over is a multiple of 32, because the
    q4_0/q8_0 block is 32 wide and the activation quantiser refuses anything else.
    That is why q_lora and qk_nope are 32 rather than something smaller, and kv_lora is
    64: it also has to be a power of two for kvarn_init."""
    rng = np.random.default_rng(11)
    cfg = dict(hidden=64, n_layers=8, n_heads=4, head_dim=32, conv_L=4,
               q_lora=32, kv_lora=64, qk_nope=32, qk_rope=16, v_head=32,
               n_experts=8, topk=2, moe_inter=32, shared_inter=32, dense_inter=96,
               n_dense_layers=1, n_group=2, topk_group=1, routed_scale=2.5,
               norm_topk_prob=1, kda_lower_bound=-5.0,
               vocab=256, eps=1e-6, rope_theta=1000000.0,
               layer_types=[0, 0, 0, 1, 0, 0, 0, 1])
    T = {}
    n = lambda *s: rng.standard_normal(s).astype(np.float32) * 0.05
    D = cfg["hidden"]
    H, HD = cfg["n_heads"], cfg["head_dim"]
    P = H * HD                                       # KDA projection width
    NE, MI, SI, DI = cfg["n_experts"], cfg["moe_inter"], cfg["shared_inter"], cfg["dense_inter"]
    T["model.word_embeddings.weight"] = n(cfg["vocab"], D)
    T["lm_head.weight"] = n(cfg["vocab"], D)
    T["model.norm.weight"] = 1.0 + n(D) * 2
    for li in range(cfg["n_layers"]):
        p = f"model.layers.{li}."
        T[p + "input_layernorm.weight"] = 1.0 + n(D) * 2
        T[p + "post_attention_layernorm.weight"] = 1.0 + n(D) * 2
        if cfg["layer_types"][li]:
            T[p + "attention.q_a_proj.weight"] = n(cfg["q_lora"], D)
            T[p + "attention.q_a_layernorm.weight"] = 1.0 + n(cfg["q_lora"]) * 2
            T[p + "attention.q_b_proj.weight"] = n(H * (cfg["qk_nope"] + cfg["qk_rope"]),
                                                   cfg["q_lora"])
            T[p + "attention.kv_a_proj_with_mqa.weight"] = n(cfg["kv_lora"] + cfg["qk_rope"], D)
            T[p + "attention.kv_a_layernorm.weight"] = 1.0 + n(cfg["kv_lora"]) * 2
            T[p + "attention.kv_b_proj.weight"] = n(H * (cfg["qk_nope"] + cfg["v_head"]),
                                                    cfg["kv_lora"])
            T[p + "attention.g_proj.weight"] = n(H, D)
            T[p + "attention.dense.weight"] = n(D, H * cfg["v_head"])
        else:
            for t in "qkv":
                T[p + f"attention.{t}_proj.weight"] = n(P, D)
                T[p + f"attention.{t}_conv1d.weight"] = n(P, 1, cfg["conv_L"])
            T[p + "attention.f_proj.weight"] = n(P, D)
            T[p + "attention.g_proj.weight"] = n(P, D)
            T[p + "attention.b_proj.weight"] = n(H, D)
            T[p + "attention.A_log"] = np.log(rng.uniform(1, 16, H)).astype(np.float32)
            T[p + "attention.dt_bias"] = n(P)
            T[p + "attention.o_norm.weight"] = 1.0 + n(HD) * 2
            T[p + "attention.o_proj.weight"] = n(D, P)
        if li < cfg["n_dense_layers"]:
            T[p + "mlp.gate_proj.weight"] = n(DI, D)
            T[p + "mlp.up_proj.weight"] = n(DI, D)
            T[p + "mlp.down_proj.weight"] = n(D, DI)
        else:
            # The router is deliberately 20x louder than everything else. With
            # unit-scale weights the sigmoid scores of eight random experts land within
            # a few hundredths of each other, and the int8-activation build then routes
            # DIFFERENTLY from the f32 oracle on a good fraction of rows -- not a small
            # numeric difference but a different expert entirely, which puts --check
            # nowhere near its tolerance for reasons that have nothing to do with the
            # engine. A loud router spreads the scores over most of (0,1), so the top-k
            # is stable under the ~1% perturbation int8 activations cause, exactly as a
            # trained router's is. The bias stays large enough that using it as the
            # weight (rather than only for selection) would still be plainly visible.
            T[p + "mlp.gate.weight"] = n(NE, D) * 20.0
            T[p + "mlp.gate.expert_bias"] = n(NE) * 0.15
            T[p + "mlp.shared_experts.gate_proj.weight"] = n(SI, D)
            T[p + "mlp.shared_experts.up_proj.weight"] = n(SI, D)
            T[p + "mlp.shared_experts.down_proj.weight"] = n(D, SI)
            # Experts share a base and differ by a third of their magnitude, rather
            # than being independent draws. This is about the int8-activation check,
            # not about the exact one.
            #
            # Grouped top-k is a DISCRETE decision, and with fully independent random
            # experts it is also a knife-edge one: int8 activations perturb the hidden
            # by ~1%, that flips a top-2-of-8 pick somewhere around the fifth MoE
            # layer, and the flipped expert -- being unrelated to the one the oracle
            # chose -- moves the logits by 30%. So `ling --check` would be asserting
            # that a chaotic function is insensitive, which it is not, and no tolerance
            # makes that claim true.
            #
            # A trained MoE is not chaotic here: its experts share most of their
            # structure, which is precisely why the routing boundary sits where it
            # does, and a flip near it costs little. Correlated experts make the
            # fixture representative of that. Nothing is weakened -- ling-exact
            # --check still runs both the engine and the oracle on identical
            # activations, so a genuine bug in the selection rule shows up there at
            # full magnitude.
            base = {w: n(*s) for w, s in
                    (("gate_proj", (MI, D)), ("up_proj", (MI, D)), ("down_proj", (D, MI)))}
            for e in range(NE):
                q = p + f"mlp.experts.{e}."
                for w, s in (("gate_proj", (MI, D)), ("up_proj", (MI, D)),
                             ("down_proj", (D, MI))):
                    T[q + w + ".weight"] = base[w] + n(*s) * 0.35
    os.makedirs(dst, exist_ok=True)
    ids = rng.integers(0, cfg["vocab"], size=12).tolist()
    json.dump({"prompt": ids}, open(os.path.join(dst, "ref.json"), "w"))
    return cfg, T


class DictShards:
    def __init__(self, T):
        self.T = T

    def has(self, n):
        return n in self.T

    def get(self, n):
        if n not in self.T:
            sys.exit(f"missing tensor {n}")
        return self.T[n]

    getw = get


# build
def build(src, dst, ctx, ram, expert_edge, embed_q4, verify, want_flash=True,
          flash_probes=0, flash_iters=10, fixture=False):
    os.makedirs(dst, exist_ok=True)

    if fixture:
        cfg, T = make_fixture(dst)
        S = DictShards(T)
    else:
        t = json.load(open(os.path.join(src, "config.json")))
        if t.get("model_type") != "bailing_hybrid":
            sys.exit(f"expected model_type bailing_hybrid, got {t.get('model_type')}")
        # Every one of these changes the arithmetic rather than the shapes, and the
        # engine hardcodes the branch it was written against. Fail loudly.
        for k, want in (("score_function", "sigmoid"), ("topk_method", "noaux_tc"),
                        ("hidden_act", "silu"), ("router_dtype", "fp32")):
            if t.get(k) != want:
                sys.exit(f"{k}={t.get(k)!r}, expected {want!r}")
        for k in ("rope_interleave", "moe_router_enable_expert_bias", "norm_topk_prob",
                  "no_kda_lora", "kda_safe_gate"):
            if not t.get(k):
                sys.exit(f"{k} is false; the engine only implements the true branch")
        for k in ("use_bias", "use_qkv_bias", "use_nGPT", "use_mla_nope", "value_norm",
                  "up_proj_norm", "scale_router_input", "tie_word_embeddings"):
            if t.get(k):
                sys.exit(f"{k} is set; not supported")
        if t.get("num_nextn_predict_layers"):
            sys.exit("MTP layers are not implemented")
        if t.get("rope_scaling"):
            sys.exit("rope_scaling is not implemented")
        if t.get("gated_attention_proj_granularity_type") != "head_wise":
            sys.exit("only head_wise MLA gating is implemented")
        if t.get("num_kv_heads_for_linear_attn"):
            sys.exit("KDA with fewer k heads than q heads is not implemented")
        if t.get("group_norm_size", 1) != 1:
            sys.exit("group_norm_size != 1 is not implemented")
        if t.get("num_shared_experts") != 1:
            sys.exit("exactly one shared expert is assumed")
        if t.get("num_attention_heads") != t.get("num_key_value_heads"):
            sys.exit("MLA with grouped KV heads is not implemented")
        qc = t.get("quantization_config") or {}
        if qc and (qc.get("quant_method") != "fp8" or qc.get("fmt") != "e4m3"
                   or qc.get("scale_fmt") != "ue8m0"):
            sys.exit(f"unsupported quantization_config {qc}")

        G = t["layer_group_size"]
        L = t["num_hidden_layers"]
        # BailingMoeV3DecoderLayer: MLA when (idx+1) % group == 0, or in the ragged tail
        # past the last whole group.
        types = [1 if ((li + 1) % G == 0 or li >= L // G * G) else 0 for li in range(L)]
        cfg = dict(
            hidden=t["hidden_size"], n_layers=L,
            n_heads=t["num_attention_heads"], head_dim=t["head_dim"],
            conv_L=t["short_conv_kernel_size"],
            q_lora=t["q_lora_rank"], kv_lora=t["kv_lora_rank"],
            qk_nope=t["qk_nope_head_dim"], qk_rope=t["qk_rope_head_dim"],
            v_head=t["v_head_dim"],
            n_experts=t["num_experts"], topk=t["num_experts_per_tok"],
            moe_inter=t["moe_intermediate_size"],
            shared_inter=t["moe_shared_expert_intermediate_size"],
            dense_inter=t["intermediate_size"],
            n_dense_layers=t["first_k_dense_replace"],
            n_group=t["n_group"], topk_group=t["topk_group"],
            routed_scale=float(t["routed_scaling_factor"]),
            norm_topk_prob=1, kda_lower_bound=float(t["kda_lower_bound"]),
            vocab=t["vocab_size"], eps=t["rms_norm_eps"],
            rope_theta=float(t["rope_theta"]), layer_types=types)
        if cfg["qk_nope"] + cfg["qk_rope"] != t["qk_head_dim"]:
            sys.exit("qk_nope + qk_rope != qk_head_dim")
        S = Shards(src)

    cfg["ctx"] = ctx
    D = cfg["hidden"]
    L, NE = cfg["n_layers"], cfg["n_experts"]
    H, HD, CL = cfg["n_heads"], cfg["head_dim"], cfg["conv_L"]
    P = H * HD                                       # KDA projection width
    MI, SI, DI = cfg["moe_inter"], cfg["shared_inter"], cfg["dense_inter"]
    ND = cfg["n_dense_layers"]
    KL, RD, NOPE, VH = cfg["kv_lora"], cfg["qk_rope"], cfg["qk_nope"], cfg["v_head"]

    if cfg["n_experts"] % cfg["n_group"]:
        sys.exit("n_experts must be a multiple of n_group")
    if cfg["topk_group"] > cfg["n_group"]:
        sys.exit("topk_group > n_group")
    if cfg["topk"] > cfg["topk_group"] * (NE // cfg["n_group"]):
        sys.exit("topk exceeds the experts reachable through topk_group groups")
    # Everything a matmul contracts over has to be a whole number of q4_0/q8_0 blocks.
    # Checking here beats discovering it as a wrong answer from the engine.
    for what, n in (("hidden", D), ("kda projection", P), ("q_lora", cfg["q_lora"]),
                    ("kv_lora", KL), ("qk_nope", NOPE), ("n_heads*v_head", H * VH),
                    ("moe_inter", MI), ("shared_inter", SI), ("dense_inter", DI)):
        if n % QK:
            sys.exit(f"{what}={n} is not a multiple of the {QK}-wide quant block")
    if KL < 2 or (KL & (KL - 1)):
        sys.exit(f"kv_lora={KL} must be a power of two (the KVarN codec requires it)")

    moe_layers = [li for li in range(L) if li >= ND]
    edge = set(moe_layers[:expert_edge] + moe_layers[-expert_edge:]) if expert_edge else set()
    expert_fmt = [0] * L
    for li in moe_layers:
        expert_fmt[li] = FMT_Q80 if li in edge else FMT_Q40

    # dense
    dn = Dense(os.path.join(dst, "dense.bin"))
    ALWAYS = FMT_Q80
    EMB = FMT_Q40 if embed_q4 else FMT_Q80
    dn.add("embed", S.getw("model.word_embeddings.weight"), EMB)
    head = S.getw("lm_head.weight")
    dn.add("lm_head", head, EMB)
    dn.add("final_norm", S.getw("model.norm.weight"), FMT_F32)

    # FlashHead. Unlike gemma4 and lfm25, which tie the head to the embedding table and
    # therefore have to cluster the embeddings, this checkpoint ships a real lm_head --
    # so the clustering sees the matrix it is actually approximating. It is built here,
    # right after the head is written, because flashhead.build normalises `head` IN
    # PLACE and consumes it.
    flash = {}
    if want_flash:
        if not fixture:
            print(f"flash head: clustering {head.shape[0]} x {head.shape[1]} lm_head "
                  f"rows", flush=True)
        log = (lambda *a, **k: None) if fixture else print
        fh = flashhead.build(head, n_probes=flash_probes, iters=flash_iters, log=log)
        force = np.asarray(flashhead.force_tokens(src, cfg["vocab"]), np.int32)
        dn.add("flash_centroids", fh["centroids"], FMT_Q40)
        dn.add_i32("flash_token_map", fh["token_map"])
        dn.add("flash_cluster_scale", fh["cluster_scale"].reshape(1, -1), FMT_F32)
        dn.add_i32("flash_force", force.reshape(1, -1) if force.size
                   else np.zeros((1, 1), np.int32))
        flash = dict(fh["cfg"], n_force=int(force.size))
    del head

    for li in range(L):
        p = f"model.layers.{li}.attention."
        o = f"layers.{li}."
        dn.add(o + "input_layernorm",
               S.getw(f"model.layers.{li}.input_layernorm.weight"), FMT_F32)
        dn.add(o + "post_attention_layernorm",
               S.getw(f"model.layers.{li}.post_attention_layernorm.weight"), FMT_F32)

        if cfg["layer_types"][li]:
            dn.add(o + "q_a_proj", S.getw(p + "q_a_proj.weight"), ALWAYS)
            dn.add(o + "q_a_norm", S.getw(p + "q_a_layernorm.weight"), FMT_F32)
            dn.add(o + "q_b_proj", S.getw(p + "q_b_proj.weight"), ALWAYS)
            dn.add(o + "kv_a_proj", S.getw(p + "kv_a_proj_with_mqa.weight"), ALWAYS)
            dn.add(o + "kv_a_norm", S.getw(p + "kv_a_layernorm.weight"), FMT_F32)
            # kv_b_proj is [H*(nope+v_head), kv_lora], head-major with the nope rows
            # first inside each head. The engine runs MLA absorbed, so ship W_k
            # TRANSPOSED (fold into the query: qc[h] = W_k[h]^T q_nope[h], contracting
            # over nope) and W_v as-is (unfold the output: o[h] = W_v[h] acc[h],
            # contracting over kv_lora).
            kvb = np.ascontiguousarray(S.getw(p + "kv_b_proj.weight")).reshape(H, NOPE + VH, KL)
            wk = kvb[:, :NOPE, :]                            # [H, nope, kv_lora]
            wv = kvb[:, NOPE:, :]                            # [H, v_head, kv_lora]
            dn.add(o + "kv_b_kt",
                   np.ascontiguousarray(wk.transpose(0, 2, 1)).reshape(H * KL, NOPE),
                   ALWAYS)
            dn.add(o + "kv_b_v", np.ascontiguousarray(wv).reshape(H * VH, KL), ALWAYS)
            # [H, hidden]: one gate scalar per head, so tiny and f32
            dn.add(o + "g_proj", S.getw(p + "g_proj.weight"), FMT_F32)
            dn.add(o + "dense", S.getw(p + "dense.weight"), ALWAYS)
        else:
            for t in "qkv":
                dn.add(o + t + "_proj", S.getw(p + f"{t}_proj.weight"), ALWAYS)
                # [P,1,CL] -> [P,CL]; f32 because CL is not a multiple of the block size
                dn.add(o + t + "_conv",
                       np.ascontiguousarray(S.getw(p + f"{t}_conv1d.weight")).reshape(P, CL),
                       FMT_F32)
            dn.add(o + "f_proj", S.getw(p + "f_proj.weight"), ALWAYS)
            dn.add(o + "kg_proj", S.getw(p + "g_proj.weight"), ALWAYS)
            dn.add(o + "b_proj", S.getw(p + "b_proj.weight"), FMT_F32)  # [H, hidden]
            dn.add(o + "A_log", S.getw(p + "A_log"), FMT_F32)
            dn.add(o + "dt_bias", S.getw(p + "dt_bias"), FMT_F32)
            dn.add(o + "o_norm", S.getw(p + "o_norm.weight"), FMT_F32)
            dn.add(o + "o_proj", S.getw(p + "o_proj.weight"), ALWAYS)

        q = f"model.layers.{li}.mlp."
        if li < ND:
            dn.add(o + "mlp_gate", S.getw(q + "gate_proj.weight"), ALWAYS)
            dn.add(o + "mlp_up", S.getw(q + "up_proj.weight"), ALWAYS)
            dn.add(o + "mlp_down", S.getw(q + "down_proj.weight"), ALWAYS)
        else:
            # f32: a routing error is not a small numeric error, it picks a different
            # expert entirely -- and the checkpoint sets router_dtype fp32 for the same
            # reason, since near-ties at top-8 flip a few percent of picks in bf16.
            dn.add(o + "router", S.getw(q + "gate.weight"), FMT_F32)
            dn.add(o + "expert_bias", S.getw(q + "gate.expert_bias"), FMT_F32)
            dn.add(o + "shared_gate", S.getw(q + "shared_experts.gate_proj.weight"), ALWAYS)
            dn.add(o + "shared_up", S.getw(q + "shared_experts.up_proj.weight"), ALWAYS)
            dn.add(o + "shared_down", S.getw(q + "shared_experts.down_proj.weight"), ALWAYS)

    dense_bytes = dn.off
    dn.close()

    # experts
    ALIGN = 4096
    gb_l, db_l, esz_l = [0] * L, [0] * L, [0] * L
    for li in moe_layers:
        f = expert_fmt[li]
        gb_l[li] = qbytes(f, MI, D)
        db_l[li] = qbytes(f, D, MI)
        esz_l[li] = 2 * gb_l[li] + db_l[li]

    idx, off, worst = {}, 0, 0.0
    with open(os.path.join(dst, "experts.bin"), "wb") as out:
        for li in moe_layers:
            f = expert_fmt[li]
            for e in range(NE):
                q = f"model.layers.{li}.mlp.experts.{e}."
                g = np.ascontiguousarray(S.getw(q + "gate_proj.weight"))
                u = np.ascontiguousarray(S.getw(q + "up_proj.weight"))
                d = np.ascontiguousarray(S.getw(q + "down_proj.weight"))
                bg, bu, bd = quant_rows(f, g), quant_rows(f, u), quant_rows(f, d)
                assert len(bg) == gb_l[li] and len(bd) == db_l[li]
                if verify:
                    worst = max(worst,
                                float(np.abs(g - dequant_rows(f, bg, MI, D)).max()),
                                float(np.abs(d - dequant_rows(f, bd, D, MI)).max()))
                pad = (-off) % ALIGN
                if pad:
                    out.write(b"\0" * pad)
                    off += pad
                out.write(bg); out.write(bu); out.write(bd)
                idx[f"{li}.{e}"] = off
                off += esz_l[li]
            if not fixture:
                print(f"  moe layer {li}  ({'q8_0' if f == FMT_Q80 else 'q4_0'})  "
                      f"{off / 2**30:.2f} GiB", flush=True)
    if verify:
        print(f"verify: max |w - dequant(quant(w))| = {worst:.6g}")

    # tokenizer
    if not fixture:
        tj = os.path.join(src, "tokenizer.json")
        if os.path.exists(tj):
            here = os.path.dirname(os.path.abspath(__file__))
            subprocess.check_call([sys.executable,
                                   os.path.join(here, "convert_lfm_tokenizer.py"),
                                   tj, os.path.join(dst, "tok.bin")])
        else:
            print("warning: no tokenizer.json in the source; tok.bin not written")

    # plan
    p = plan(cfg, dense_bytes, esz_l, ctx, ram)
    cfg["slots_per_layer"] = p["slots_per_layer"]
    cfg.update({f"flash_{k}": v for k, v in flash.items()})
    json.dump(cfg, open(os.path.join(dst, "cfg.json"), "w"), indent=1)

    with open(os.path.join(dst, "manifest.txt"), "w") as m:
        for k, v in cfg.items():
            if k == "layer_types":
                m.write("cfg layer_types " + " ".join(str(x) for x in v) + "\n")
            else:
                m.write(f"cfg {k} {v}\n")
        for li in moe_layers:
            m.write(f"eszl {li} {esz_l[li]} {gb_l[li]} {db_l[li]} {expert_fmt[li]}\n")
        m.write(f"ndense {len(dn.idx)}\n")
        for k, v in dn.idx.items():
            sh = v["shape"]
            m.write(f"dense {k} {v['off']} {v['len']} {v['fmt']} {sh[0]} {sh[1]}\n")
        m.write(f"nexpert {len(idx)}\n")
        for k, v in idx.items():
            li, e = k.split(".")
            m.write(f"expert {li} {e} {v}\n")

    n_mla = sum(cfg["layer_types"])
    nq8 = sum(1 for li in moe_layers if expert_fmt[li] == FMT_Q80)
    print(f"\nlayers         : {L} ({n_mla} MLA, {L - n_mla} KDA; "
          f"{ND} dense, {len(moe_layers)} MoE + shared)")
    print(f"dense resident : {p['dense_bytes']/2**20:8.1f} MiB")
    print(f"latent kv      : {p['kv_bytes']/2**20:8.1f} MiB  (ctx {ctx}, {n_mla} MLA layers)")
    print(f"kda state      : {p['kda_state_bytes']/2**20:8.1f} MiB  (context-independent)")
    print(f"experts        : {p['total_expert_bytes']/2**30:.2f} GiB total, "
          f"{nq8} layers q8_0 / {len(moe_layers)-nq8} q4_0, "
          f"max {p['expert_bytes_max']/2**20:.2f} MiB each")
    print(f"plan @ {ram:g} GB : cache {p['cache_bytes']/2**30:.2f} GiB = "
          f"{p['slots_per_layer']}/{NE} slots/layer "
          f"({100.0*p['cache_bytes']/max(1,p['total_expert_bytes']):.0f}% of the expert set)")
    if flash:
        print(f"flash head     : {flash['n_clusters']} clusters x "
              f"{flash['cluster_size']}, probe {flash['n_probes']} -> "
              f"{flash['n_probes']*flash['cluster_size']}/{cfg['vocab']} rows scored, "
              f"{flash['n_force']} forced")
    if not p["ok"]:
        print(f"  !! BELOW FLOOR: a layer needs >= topk={cfg['topk']} slots. "
              f"Minimum viable budget {p['min_ram_gb']:.2f} GB.")
    return cfg, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst")
    ap.add_argument("--ram", type=float, default=8.0, help="RAM budget, GB")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--expert-edge", type=int, default=0, metavar="N",
                    help="MoE layers at each end whose experts get q8_0 instead of "
                         "q4_0. 0 (the default) = uniform q4_0 experts.")
    ap.add_argument("--embed-q4", action="store_true",
                    help="q4_0 the embedding and lm_head instead of q8_0 "
                         "(saves ~230 MiB of resident set)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--no-flash", dest="flash", action="store_false",
                    help="do not build the FlashHead (exact lm_head only)")
    ap.add_argument("--flash-probes", type=int, default=0,
                    help="clusters probed per decode step (default: ~11%% of them, "
                         "the fraction Maple's shipped head uses)")
    ap.add_argument("--flash-iters", type=int, default=10,
                    help="k-means iterations for the FlashHead clustering")
    ap.add_argument("--fixture", action="store_true",
                    help="build a tiny random model for the oracle check")
    a = ap.parse_args()
    if a.fixture:
        build(None, a.dst, a.ctx, a.ram, a.expert_edge, a.embed_q4, a.verify,
              a.flash, a.flash_probes, a.flash_iters, fixture=True)
    else:
        if not a.src:
            ap.error("src is required unless --fixture")
        build(a.src, a.dst, a.ctx, a.ram, a.expert_edge, a.embed_q4, a.verify,
              a.flash, a.flash_probes, a.flash_iters)


if __name__ == "__main__":
    main()
