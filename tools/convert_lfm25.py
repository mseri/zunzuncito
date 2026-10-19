#!/usr/bin/env python3
"""
convert_lfm25.py — LiquidAI LFM2.5-8B-A1B / -24B-A2B (HF safetensors, or a
llama.cpp GGUF) -> lfm25 container.

Same container shape as convert_gemma4.py (manifest.txt + dense.bin + experts.bin,
one expert = one contiguous 4096-aligned pread), but two things differ.

The architecture is hybrid: 24 layers of which only 6 are attention and 18 are short
convolutions, and the first `num_dense_layers` use a dense SwiGLU MLP instead of the
MoE. So `layer_types` here means conv-vs-attention rather than sliding-vs-global, and
layers 0..num_dense_layers-1 carry an MLP in dense.bin while the rest carry a router
+ 32 experts in experts.bin.

Mixed precision comes from apex-quant, which assigns different quant types per tensor
class, exploiting that routed experts are ~97% idle and tolerate far more error than
the always-on tensors. Its Q3_K..Q8_0 gradient maps onto the two block formats this
engine has kernels for:

    always-on  (attention, conv, dense MLP)  -> q8_0    (apex: Q6_K/Q8_0)
    routed experts, edge layers              -> q8_0    (apex: Q6_K)
    routed experts, middle layers            -> q4_0    (apex: Q4_K/IQ4_XS)
    norms, router, expert_bias, depthwise    -> f32     (tiny, and routing errors
                                                         are not small numeric
                                                         errors -- they pick a
                                                         different expert)

--expert-edge N sets how many MoE layers at each end get q8_0 experts. It is the
main size/quality dial: each q8_0 layer costs ~1.9x its q4_0 equivalent, and on a
RAM-constrained box that comes straight out of the expert cache. N=0 reproduces the
uniform-q4_0 behaviour of the gemma4 converter.

The depthwise conv weight is f32 for a structural reason, not a quality one: it is
[hidden, 1, conv_L] with conv_L = 3, and a block format needs the contracted
dimension to be a multiple of 32.

Weights can also come from a GGUF (--gguf FILE; src still supplies config.json and
the tokenizer). Its Q4_0/Q8_0 tensors are copied block for block and keep their
format whatever the policy above says, so --expert-edge does nothing to a Q4_0
layer. That is the path for LiquidAI's QAD-Q4_0 checkpoints: QAD (quantisation-aware
distillation) retrains the weights to sit on the plain q4_0 grid, and the quantiser
itself is stock ggml Q4_0, which q40_quant_rows already reproduces bit for bit on
bf16 inputs. So there is nothing to borrow from the quantiser; the gain is in the
weights, and only survives if they are not re-rounded. Any other GGUF type is
dequantised and re-encoded, to q8_0 when the policy wants a block format, so the
second rounding is as small as it can be.

The 24B checkpoint may be saved in transformers 5's fused expert layout
(experts.gate_up_proj / experts.down_proj per layer); both layouts are read.

    python3 tools/convert_lfm25.py ./lfm ./lfm-ct --ram 8 --ctx 4096
    python3 tools/convert_lfm25.py ./lfm24 ./lfm24-ct --ram 8 \
        --gguf ./LFM2.5-24B-A2B-QAD-Q4_0.gguf
    python3 tools/convert_lfm25.py --fixture ./lfm-fix          # tiny random model
"""
import argparse, json, os, re, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flashhead

QK = 32
BLK40, BLK80 = 18, 34
FMT_F32, FMT_Q40, FMT_Q80, FMT_I32 = 0, 1, 2, 5


# q4_0 (mirrors q40.h bit-for-bit)
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


# q8_0 (mirrors q40.h's q80_*)
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


# safetensors
_DT = {"F32": np.float32, "F16": np.float16, "I8": np.int8, "U8": np.uint8,
       "I32": np.int32, "I64": np.int64}


class Shards:
    """Streaming reader. Keeps one open handle per shard and reads a tensor's byte
    range on demand -- the checkpoint is 16 GB and must never be resident."""

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
        if e["dtype"] == "BF16":
            a = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            a = np.frombuffer(raw, _DT[e["dtype"]]).astype(np.float32)
        return a.reshape(e["shape"])

    def get_slab(self, n, i):
        """The i-th slice along the leading axis, reading only its bytes: a fused
        expert tensor is [n_experts, ...] and ~800 MB a layer at 24B."""
        if n not in self.wmap:
            sys.exit(f"missing tensor {n}")
        f, (hdr, base) = self._open(self.wmap[n])
        e = hdr[n]
        sh = e["shape"]
        isz = 2 if e["dtype"] in ("BF16", "F16") else np.dtype(_DT[e["dtype"]]).itemsize
        slab = int(np.prod(sh[1:])) * isz
        f.seek(base + e["data_offsets"][0] + i * slab)
        raw = f.read(slab)
        if e["dtype"] == "BF16":
            a = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            a = np.frombuffer(raw, _DT[e["dtype"]]).astype(np.float32)
        return a.reshape(sh[1:])

    def fmt_of(self, n):
        return None                    # every tensor here is floating point


# gguf
# HF checkpoint name -> llama.cpp lfm2moe name, per layer and global. Mirrors
# gguf-py's tensor_mapping for LFM2; embedding_norm lands in token_embd_norm (it is
# applied at the output, llama.cpp's own "fix for wrong tensor name").
_GG_GLOBAL = {"model.embed_tokens.weight": "token_embd.weight",
              "model.embedding_norm.weight": "token_embd_norm.weight"}
_GG_LAYER = {
    "operator_norm.weight": "attn_norm.weight", "ffn_norm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight", "self_attn.out_proj.weight": "attn_output.weight",
    "self_attn.q_layernorm.weight": "attn_q_norm.weight",
    "self_attn.k_layernorm.weight": "attn_k_norm.weight",
    "conv.in_proj.weight": "shortconv.in_proj.weight", "conv.conv.weight": "shortconv.conv.weight",
    "conv.out_proj.weight": "shortconv.out_proj.weight",
    "feed_forward.w1.weight": "ffn_gate.weight", "feed_forward.w3.weight": "ffn_up.weight",
    "feed_forward.w2.weight": "ffn_down.weight",
    "feed_forward.gate.weight": "ffn_gate_inp.weight",
    "feed_forward.expert_bias": "exp_probs_b.bias"}
_GG_EXPERT = {"w1": "ffn_gate_exps.weight", "w3": "ffn_up_exps.weight",
              "w2": "ffn_down_exps.weight"}


class GGUFSource:
    """Weights from a llama.cpp GGUF of arch lfm2moe, served under the HF checkpoint's
    tensor names so build() does not care where they came from.

    The point is the blocks themselves. q40.h is bit-identical to ggml's block_q4_0
    and q8_0 likewise, and a row of either is contiguous over the contracted dim in
    both layouts, so a Q4_0/Q8_0 tensor is copied byte for byte (fmt_of / raw) and
    never touches a float. That matters for LiquidAI's QAD-Q4_0 release: its weights
    were distilled to sit on the q4_0 grid, so the blocks ARE the model, and any
    requantisation -- even up to q8_0 -- moves them off it. Every other type is
    dequantised by gguf-py (get) and re-encoded by the caller."""

    def __init__(self, path):
        try:
            from gguf import GGUFReader, GGMLQuantizationType, quants
        except ImportError:
            sys.exit("--gguf needs the gguf package (pip install gguf)")
        self.r = GGUFReader(path)
        self.quants, Q = quants, GGMLQuantizationType
        arch = self.r.fields["general.architecture"].contents()
        if arch != "lfm2moe":
            sys.exit(f"expected a GGUF of arch lfm2moe, got {arch}")
        self.t = {t.name: t for t in self.r.tensors}
        self.native = {Q.Q4_0: FMT_Q40, Q.Q8_0: FMT_Q80}
        self.floats = {Q.F32, Q.F16, Q.BF16}

    def meta(self, key):
        f = self.r.fields.get(f"lfm2moe.{key}")
        return None if f is None else f.contents()

    def check(self, cfg):
        for key, want in (("block_count", cfg["n_layers"]),
                          ("embedding_length", cfg["hidden"]),
                          ("expert_count", cfg["n_experts"]),
                          ("expert_used_count", cfg["topk"]),
                          ("expert_feed_forward_length", cfg["moe_inter"])):
            got = self.meta(key)
            if got is not None and got != want:
                sys.exit(f"GGUF {key}={got} but config.json says {want}: "
                         f"the GGUF and the config are not the same model")

    def _resolve(self, n):
        """-> (tensor, expert index or None, row range or None)"""
        if n in _GG_GLOBAL:
            return self.t.get(_GG_GLOBAL[n]), None, None
        m = re.fullmatch(r"model\.layers\.(\d+)\.feed_forward\.experts\.(\d+)\.(w[123])\.weight", n)
        if m:
            return self.t.get(f"blk.{m[1]}.{_GG_EXPERT[m[3]]}"), int(m[2]), None
        m = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", n)
        if not m or m[2] not in _GG_LAYER:
            return None, None, None
        t = self.t.get(f"blk.{m[1]}.{_GG_LAYER[m[2]]}")
        if t is None and m[2] in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                                  "self_attn.v_proj.weight"):
            # fused attn_qkv: q is n_embd rows, then k and v split the rest evenly
            t = self.t.get(f"blk.{m[1]}.attn_qkv.weight")
            if t is not None:
                O, D = int(t.shape[1]), int(t.shape[0])
                kv = (O - D) // 2
                r = {"q": (0, D), "k": (D, D + kv), "v": (D + kv, O)}[m[2][10]]
                return t, None, r
        return t, None, None

    def _data(self, n):
        t, e, r = self._resolve(n)
        if t is None:
            sys.exit(f"missing tensor {n} in the GGUF")
        a = t.data if e is None else t.data[e]
        if r is not None:
            a = a[r[0]:r[1]]
        return t, a

    def has(self, n):
        return self._resolve(n)[0] is not None

    def fmt_of(self, n):
        """FMT_Q40/FMT_Q80 for a tensor raw() can hand over as is, -1 for any other
        quantised type, None for floating point."""
        t = self._resolve(n)[0]
        if t is None or t.tensor_type in self.floats:
            return None
        return self.native.get(t.tensor_type, -1)

    def raw(self, n):
        t, a = self._data(n)
        return np.ascontiguousarray(a).tobytes()

    def get(self, n):
        t, a = self._data(n)
        if t.tensor_type.name not in ("F32", "F16"):
            a = self.quants.dequantize(a, t.tensor_type)
        return np.asarray(a, dtype=np.float32)


def pick_fmt(S, n, fmt):
    """The format weight `n` lands in, given the policy's `fmt`. A source block that
    is already q4_0/q8_0 stays what it is (see GGUFSource); one in some other quantised
    type goes to q8_0, so the second rounding adds as little as it can. Policy f32 is
    always honoured: those tensors are f32 for structural reasons, not size."""
    if fmt == FMT_F32:
        return fmt
    f = S.fmt_of(n)
    if f is None:
        return fmt
    return f if f in (FMT_Q40, FMT_Q80) else FMT_Q80


def encode(S, n, fmt):
    """-> (bytes, O, I) of weight `n` in `fmt`, copying the source's own blocks when
    they are already in it."""
    if fmt != FMT_F32 and S.fmt_of(n) == fmt:
        b = S.raw(n)
        t, a = S._data(n)
        O = a.shape[0] if a.ndim > 1 else 1
        I = len(b) // O // (BLK40 if fmt == FMT_Q40 else BLK80) * QK
        return b, O, I
    w = np.ascontiguousarray(S.get(n), dtype=np.float32)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    return quant_rows(fmt, w), int(w.shape[0]), int(w.shape[1])


def expert_names(li, e):
    q = f"model.layers.{li}.feed_forward.experts.{e}."
    return q + "w1.weight", q + "w3.weight", q + "w2.weight"


def expert_floats(S, li, e):
    """(gate, up, down) of one expert, from either checkpoint layout: one tensor per
    expert, or the fused [n_experts, 2*moe_inter, hidden] gate_up_proj and
    [n_experts, hidden, moe_inter] down_proj that transformers >= 5 saves (gate rows
    first, as its WeightConverter concatenates w1 then w3)."""
    p = f"model.layers.{li}.feed_forward.experts."
    if S.has(p + "gate_up_proj"):
        gu = S.get_slab(p + "gate_up_proj", e)
        h = gu.shape[0] // 2
        return gu[:h], gu[h:], S.get_slab(p + "down_proj", e)
    return tuple(S.get(n) for n in expert_names(li, e))


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

    def put(self, name, b, fmt, O, I):
        """an already-encoded weight (see encode)"""
        self.f.write(b)
        self.idx[name] = {"off": self.off, "len": len(b), "fmt": fmt, "shape": [O, I]}
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
def plan(cfg, dense_bytes, esz_by_layer, ctx, ram_gb, kv=None):
    """slots/layer for the per-layer LRU expert cache under a RAM budget.

    Only the MoE layers hold experts, and with a mixed precision gradient their
    expert size differs per layer, so the cache is sized off the LARGEST -- a single
    slots/layer number that is safe everywhere beats a per-layer one that is only
    correct on average.
    """
    NE = cfg["n_experts"]
    moe = [li for li, e in enumerate(esz_by_layer) if e]
    esz_max = max((esz_by_layer[li] for li in moe), default=0)
    nmoe = len(moe)

    kvbytes = 0
    for li, t in enumerate(cfg["layer_types"]):
        if not t:
            continue                              # conv layers hold no KV
        hd, nkv = cfg["head_dim"], cfg["n_kv_heads"]
        if not kv or kv["kbits"] <= 0:
            kvbytes += 2 * ctx * nkv * hd * 4
            continue
        prot = li < kv["protect"] or li >= cfg["n_layers"] - kv["protect"]
        kb = kv["pbits"] if prot else kv["kbits"]
        vb = kv["pbits"] if prot else kv["vbits"]
        W = min(kv["rwin"], ctx)
        vecb = lambda d, bits: (d * bits + 7) // 8 + 4
        kvbytes += 2 * W * nkv * hd * 4
        kvbytes += ctx * nkv * (vecb(hd, kb) + vecb(hd, vb))

    # conv state is [n_conv_layers][conv_L-1][hidden] f32: kilobytes, but count it
    conv = sum(1 for t in cfg["layer_types"] if not t)
    convstate = conv * max(0, cfg["conv_L"] - 1) * cfg["hidden"] * 4

    scratch = 192 << 20
    avail = int(ram_gb * (1 << 30)) - dense_bytes - kvbytes - convstate - scratch
    per = min(NE, max(0, avail // esz_max) // max(1, nmoe)) if esz_max else NE
    total = sum(esz_by_layer[li] * NE for li in moe)
    # Budgeted against the largest expert so the plan holds on every layer, but
    # reported at the true per-layer sizes: otherwise a mixed-precision container
    # claims a cache bigger than the expert set it is caching.
    cache = int(per) * sum(esz_by_layer[li] for li in moe)
    return {
        "expert_bytes_max": esz_max, "dense_bytes": dense_bytes, "kv_bytes": kvbytes,
        "slots_per_layer": int(per), "cache_bytes": cache,
        "total_expert_bytes": total, "n_moe_layers": nmoe,
        "ok": per >= cfg["topk"],
        "min_ram_gb": (dense_bytes + kvbytes + convstate + scratch
                       + cfg["topk"] * nmoe * esz_max) / (1 << 30),
    }


# fixture
def make_fixture(dst):
    """A tiny random LFM2-MoE with the same STRUCTURE (conv + attention layers,
    dense + MoE layers, sigmoid router with bias) and absurd dimensions. It exists
    so the numpy oracle and the C engine can be diffed without a 16 GB download and
    without torch."""
    rng = np.random.default_rng(7)
    cfg = dict(hidden=64, n_layers=6, n_heads=4, head_dim=16, n_kv_heads=2,
               n_experts=6, topk=2, moe_inter=32, dense_inter=96, n_dense_layers=2,
               conv_L=3, vocab=256, eps=1e-5, rope_theta=5000000.0,
               routed_scaling=1.0, norm_topk_prob=1, use_expert_bias=1,
               layer_types=[0, 0, 1, 0, 1, 0])
    T = {}
    n = lambda *s: rng.standard_normal(s).astype(np.float32) * 0.05
    T["model.embed_tokens.weight"] = n(cfg["vocab"], cfg["hidden"])
    T["model.embedding_norm.weight"] = 1.0 + n(cfg["hidden"]) * 2
    D, MI, DI, NE = cfg["hidden"], cfg["moe_inter"], cfg["dense_inter"], cfg["n_experts"]
    for li in range(cfg["n_layers"]):
        p = f"model.layers.{li}."
        T[p + "operator_norm.weight"] = 1.0 + n(D) * 2
        T[p + "ffn_norm.weight"] = 1.0 + n(D) * 2
        if cfg["layer_types"][li]:
            T[p + "self_attn.q_proj.weight"] = n(cfg["n_heads"] * cfg["head_dim"], D)
            T[p + "self_attn.k_proj.weight"] = n(cfg["n_kv_heads"] * cfg["head_dim"], D)
            T[p + "self_attn.v_proj.weight"] = n(cfg["n_kv_heads"] * cfg["head_dim"], D)
            T[p + "self_attn.out_proj.weight"] = n(D, cfg["n_heads"] * cfg["head_dim"])
            T[p + "self_attn.q_layernorm.weight"] = 1.0 + n(cfg["head_dim"]) * 2
            T[p + "self_attn.k_layernorm.weight"] = 1.0 + n(cfg["head_dim"]) * 2
        else:
            T[p + "conv.in_proj.weight"] = n(3 * D, D)
            T[p + "conv.conv.weight"] = n(D, 1, cfg["conv_L"])
            T[p + "conv.out_proj.weight"] = n(D, D)
        if li < cfg["n_dense_layers"]:
            T[p + "feed_forward.w1.weight"] = n(DI, D)
            T[p + "feed_forward.w3.weight"] = n(DI, D)
            T[p + "feed_forward.w2.weight"] = n(D, DI)
        else:
            T[p + "feed_forward.gate.weight"] = n(NE, D)
            T[p + "feed_forward.expert_bias"] = n(NE) * 0.1
            for e in range(NE):
                q = p + f"feed_forward.experts.{e}."
                T[q + "w1.weight"] = n(MI, D)
                T[q + "w3.weight"] = n(MI, D)
                T[q + "w2.weight"] = n(D, MI)
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

    def get_slab(self, n, i):
        return self.get(n)[i]

    def fmt_of(self, n):
        return None


# build
# what /v1/models reports, by layer count; anything else falls back to the dir name
MODEL_IDS = {24: "lfm2.5-8b-a1b", 40: "lfm2.5-24b-a2b"}


def build(src, dst, ctx, ram, expert_edge, embed_q8, verify, want_flash=True,
          flash_probes=0, flash_iters=10, fixture=False, gguf=None, model_id=None,
          fixture_cfg=None):
    os.makedirs(dst, exist_ok=True)

    if fixture:
        cfg, T = fixture_cfg or make_fixture(dst)
        S = DictShards(T)
    else:
        t = json.load(open(os.path.join(src, "config.json")))
        if t.get("model_type") != "lfm2_moe":
            sys.exit(f"expected model_type lfm2_moe, got {t.get('model_type')}")
        types = [1 if x == "full_attention" else 0 for x in t["layer_types"]]
        for x in t["layer_types"]:
            if x not in ("full_attention", "conv"):
                sys.exit(f"unknown layer type {x}")
        rp = t.get("rope_parameters", {})
        cfg = dict(
            hidden=t["hidden_size"], n_layers=t["num_hidden_layers"],
            n_heads=t["num_attention_heads"],
            head_dim=t.get("head_dim", t["hidden_size"] // t["num_attention_heads"]),
            n_kv_heads=t["num_key_value_heads"],
            n_experts=t["num_experts"], topk=t["num_experts_per_tok"],
            moe_inter=t["moe_intermediate_size"], dense_inter=t["intermediate_size"],
            n_dense_layers=t["num_dense_layers"], conv_L=t["conv_L_cache"],
            vocab=t["vocab_size"], eps=t["norm_eps"],
            rope_theta=float(rp.get("rope_theta", t.get("rope_theta", 10000.0))),
            routed_scaling=float(t.get("routed_scaling_factor", 1.0)),
            norm_topk_prob=1 if t.get("norm_topk_prob", True) else 0,
            use_expert_bias=1 if t.get("use_expert_bias", False) else 0,
            layer_types=types)
        if t.get("conv_bias"):
            sys.exit("conv_bias is not supported")
        # older LFM2 configs spell it tie_embedding
        if not t.get("tie_word_embeddings", t.get("tie_embedding", True)):
            sys.exit("untied embeddings are not supported (lm_head reuses embed_tokens)")
        S = Shards(src)

    if gguf:
        S = GGUFSource(gguf)
        S.check(cfg)
    cfg["model_id"] = model_id or MODEL_IDS.get(cfg["n_layers"]) or (
        os.path.basename(os.path.normpath(src)).lower() if src else "lfm2.5-moe")
    cfg["ctx"] = ctx
    D, MI, DI = cfg["hidden"], cfg["moe_inter"], cfg["dense_inter"]
    L, NE, CL = cfg["n_layers"], cfg["n_experts"], cfg["conv_L"]
    ND = cfg["n_dense_layers"]

    # the apex precision gradient, resolved to a per-layer expert format
    moe_layers = [li for li in range(L) if li >= ND]
    edge = set(moe_layers[:expert_edge] + moe_layers[-expert_edge:]) if expert_edge else set()
    # A layer's three expert matrices share one format (the manifest has one per
    # layer), so a source that mixes them gets the widest of the three.
    expert_fmt = [0] * L
    for li in moe_layers:
        pol = FMT_Q80 if li in edge else FMT_Q40
        fs = {pick_fmt(S, n, pol) for n in expert_names(li, 0)}
        expert_fmt[li] = FMT_Q80 if FMT_Q80 in fs else FMT_Q40

    # dense
    dn = Dense(os.path.join(dst, "dense.bin"))
    ALWAYS = FMT_Q80                          # attention / conv / dense MLP

    def w(oname, hfname, fmt):
        f = pick_fmt(S, hfname, fmt)
        b, O, I = encode(S, hfname, f)
        dn.put(oname, b, f, O, I)

    w("embed_tokens", "model.embed_tokens.weight", FMT_Q80 if embed_q8 else FMT_Q40)
    w("embedding_norm", "model.embedding_norm.weight", FMT_F32)
    emb = S.get("model.embed_tokens.weight") if want_flash else None

    # FlashHead, built from the same table before anything else needs it: the
    # clustering normalises `emb` in place, so this consumes it.
    flash = {}
    if want_flash:
        if not fixture:
            print(f"flash head: clustering {emb.shape[0]} x {emb.shape[1]} lm_head "
                  f"rows", flush=True)
        log = (lambda *a, **k: None) if fixture else print
        fh = flashhead.build(emb, n_probes=flash_probes, iters=flash_iters, log=log)
        force = np.asarray(flashhead.force_tokens(src, cfg["vocab"]), np.int32)
        dn.add("flash_centroids", fh["centroids"], FMT_Q40)
        dn.add_i32("flash_token_map", fh["token_map"])
        dn.add("flash_cluster_scale", fh["cluster_scale"].reshape(1, -1), FMT_F32)
        dn.add_i32("flash_force", force.reshape(1, -1) if force.size
                   else np.zeros((1, 1), np.int32))
        flash = dict(fh["cfg"], n_force=int(force.size))
    del emb

    for li in range(L):
        p = f"model.layers.{li}."
        o = f"layers.{li}."
        w(o + "operator_norm", p + "operator_norm.weight", FMT_F32)
        w(o + "ffn_norm", p + "ffn_norm.weight", FMT_F32)

        if cfg["layer_types"][li]:
            w(o + "q_proj", p + "self_attn.q_proj.weight", ALWAYS)
            w(o + "k_proj", p + "self_attn.k_proj.weight", ALWAYS)
            w(o + "v_proj", p + "self_attn.v_proj.weight", ALWAYS)
            w(o + "o_proj", p + "self_attn.out_proj.weight", ALWAYS)
            w(o + "q_norm", p + "self_attn.q_layernorm.weight", FMT_F32)
            w(o + "k_norm", p + "self_attn.k_layernorm.weight", FMT_F32)
        else:
            w(o + "conv_in", p + "conv.in_proj.weight", ALWAYS)
            # [D,1,CL] -> [D,CL]; f32 because CL=3 is not a multiple of the block size
            dn.add(o + "conv_w",
                   np.ascontiguousarray(S.get(p + "conv.conv.weight")).reshape(D, CL),
                   FMT_F32)
            w(o + "conv_out", p + "conv.out_proj.weight", ALWAYS)

        if li < ND:
            w(o + "mlp_gate", p + "feed_forward.w1.weight", ALWAYS)
            w(o + "mlp_up", p + "feed_forward.w3.weight", ALWAYS)
            w(o + "mlp_down", p + "feed_forward.w2.weight", ALWAYS)
        else:
            # f32: a routing error is not a small numeric error, it picks a
            # different expert entirely.
            w(o + "router", p + "feed_forward.gate.weight", FMT_F32)
            bias = (S.get(p + "feed_forward.expert_bias") if cfg["use_expert_bias"]
                    else np.zeros(NE, np.float32))
            dn.add(o + "expert_bias", bias, FMT_F32)

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
                if S.fmt_of(expert_names(li, e)[0]) is None:
                    g, u, d = (np.ascontiguousarray(x) for x in expert_floats(S, li, e))
                    bg, bu, bd = quant_rows(f, g), quant_rows(f, u), quant_rows(f, d)
                else:
                    bg, bu, bd = (encode(S, n, f)[0] for n in expert_names(li, e))
                    if verify:
                        g, u, d = (S.get(n) for n in expert_names(li, e))
                assert len(bg) == gb_l[li] and len(bu) == gb_l[li] and len(bd) == db_l[li]
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

    nq8 = sum(1 for li in moe_layers if expert_fmt[li] == FMT_Q80)
    print(f"\nlayers         : {L} ({sum(cfg['layer_types'])} attention, "
          f"{L - sum(cfg['layer_types'])} conv; {ND} dense, {len(moe_layers)} MoE)")
    print(f"dense resident : {p['dense_bytes']/2**20:8.1f} MiB")
    print(f"kv cache       : {p['kv_bytes']/2**20:8.1f} MiB  (ctx {ctx})")
    print(f"experts        : {p['total_expert_bytes']/2**30:.2f} GiB total, "
          f"{nq8} layers q8_0 / {len(moe_layers)-nq8} q4_0, "
          f"max {p['expert_bytes_max']/2**20:.2f} MiB each")
    print(f"plan @ {ram:g} GB : cache {p['cache_bytes']/2**30:.2f} GiB = "
          f"{p['slots_per_layer']}/{NE} slots/layer "
          f"({100.0*p['cache_bytes']/max(1,p['total_expert_bytes']):.0f}% of the expert set)")
    if not p["ok"]:
        print(f"  !! BELOW FLOOR: a layer needs >= topk={cfg['topk']} slots. "
              f"Minimum viable budget {p['min_ram_gb']:.2f} GB.")
    return cfg, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst")
    ap.add_argument("--ram", type=float, default=8.0, help="RAM budget, GB")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--expert-edge", type=int, default=2, metavar="N",
                    help="MoE layers at each end whose experts get q8_0 instead of "
                         "q4_0 (apex gradient). 0 = uniform q4_0.")
    ap.add_argument("--embed-q8", action="store_true",
                    help="q8_0 the (tied) embedding/lm_head instead of q4_0")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--no-flash", dest="flash", action="store_false",
                    help="do not build the FlashHead (exact lm_head only)")
    ap.add_argument("--flash-probes", type=int, default=0,
                    help="clusters probed per decode step (default: ~11%% of them, "
                         "the fraction Maple's shipped head uses)")
    ap.add_argument("--flash-iters", type=int, default=10,
                    help="k-means iterations for the FlashHead clustering")
    ap.add_argument("--gguf", metavar="FILE",
                    help="take the weights from this llama.cpp GGUF (arch lfm2moe) "
                         "instead of src's safetensors; src still supplies config.json "
                         "and the tokenizer. Q4_0/Q8_0 blocks are copied as they are, "
                         "which is what LiquidAI's QAD-Q4_0 release wants")
    ap.add_argument("--model-id", help="the id /v1/models reports (default: from "
                    "the layer count, else the src directory name)")
    ap.add_argument("--fixture", action="store_true",
                    help="build a tiny random model for the oracle check")
    a = ap.parse_args()
    if a.fixture:
        build(None, a.dst, a.ctx, a.ram, a.expert_edge, a.embed_q8, a.verify,
              a.flash, a.flash_probes, a.flash_iters, fixture=True,
              model_id=a.model_id)
    else:
        if not a.src:
            ap.error("src is required unless --fixture")
        build(a.src, a.dst, a.ctx, a.ram, a.expert_edge, a.embed_q8, a.verify,
              a.flash, a.flash_probes, a.flash_iters, gguf=a.gguf, model_id=a.model_id)


if __name__ == "__main__":
    main()
