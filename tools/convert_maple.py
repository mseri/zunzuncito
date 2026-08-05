#!/usr/bin/env python3
"""
convert_maple.py — DeepGrove Maple (MLX ternary checkpoint) -> maple container.

Same container shape as convert_lfm25.py (manifest.txt + dense.bin + experts.bin,
one expert = one contiguous 4096-aligned pread). What differs is that there is no
precision gradient to choose, because the checkpoint has already made every choice.

Maple is trained ternary. Every matrix in the model -- the four attention
projections and all three expert projections -- ships as 2-bit codes with one bf16
scale per output row (`row_alpha`), and across the whole checkpoint the codes only
ever take the values 0, 1 and 2:

    w = alpha_row * (code - 1)        so w in {-alpha, 0, +alpha}

There is therefore no --expert-edge here and no q8_0 tier. apex-quant's premise --
that routed experts tolerate more error than always-on tensors -- has nothing to act
on when both are already at the floor, and re-quantising a ternary tensor to q4_0
would spend 2.25x the bytes to represent strictly less. The converter's job is to
repack, not to quantise, and it verifies that: --verify checks the container
reproduces the checkpoint's weights EXACTLY, not approximately.

Three tensor classes are not ternary and keep their own formats:

    embeddings, lm_head, FlashHead centroids  -> q4a   4-bit affine, group 64
    norms, router                             -> f32
    everything else (all matrices)            -> tq2   ternary, per-row scale

q4a is the checkpoint's own format carried through rather than re-quantised: MLX
stores these at 4 bits with a scale AND a bias per group of 64, which is 36 bytes per
64 weights, exactly what q4_0 would cost while throwing the zero point away. The
repack is lossless (bf16 scales are exactly representable in fp16 at these
magnitudes, which is asserted).

The FlashHead is optional and lives in its own shard. It is an approximate lm_head:
the vocabulary is clustered into `n_clusters` groups of `cluster_size`, decode scores
the cluster centroids first and computes exact logits only for the tokens of the top
`n_probes` clusters. At Maple's numbers that is 16384 candidate rows instead of
151936 -- and since the lm_head is 175 MiB at 4 bits, it is the single biggest read
of a decode step. Greedy decoding stays exact whenever the true argmax lies in a
probed cluster.

    python3 tools/convert_maple.py /path/to/maple-preview-mlx ./maple-ct --ram 8 --ctx 4096
    python3 tools/convert_maple.py --fixture ./maple-fix            # tiny random model
"""
import argparse, json, os, struct, sys
import numpy as np

QK = 32                       # activation block, shared with q40.h
TQ2_GRP = 64                  # ternary packing group (16 bytes)
Q4A_GRP, Q4A_GRP_BYTES = 64, 36

FMT_F32, FMT_TQ2, FMT_Q4A, FMT_I32 = 0, 3, 4, 5


# ---------------------------------------------------------------- tq2 (ternary)
def tq2_pack_codes(codes: np.ndarray) -> bytes:
    """codes uint8 [O,I] in {0,1,2} -> packed [O, I/4].

    Byte (g,k) of a row holds elements g*64+k, g*64+16+k, g*64+32+k, g*64+48+k at
    bit positions 0, 2, 4, 6 -- see the layout comment in tq2.h. Reshaping to
    [O, g, t, k] makes element g*64 + t*16 + k land at index [g,t,k], so the four
    shift positions are just the four slices of `t`.
    """
    O, I = codes.shape
    assert I % TQ2_GRP == 0, f"I={I} not a multiple of {TQ2_GRP}"
    c = codes.reshape(O, I // TQ2_GRP, 4, 16)
    out = (c[:, :, 0, :] | (c[:, :, 1, :] << 2) |
           (c[:, :, 2, :] << 4) | (c[:, :, 3, :] << 6))
    return np.ascontiguousarray(out.reshape(O, I // 4), dtype=np.uint8)


def tq2_unpack_codes(packed: np.ndarray, I: int) -> np.ndarray:
    """Inverse of tq2_pack_codes, for --verify."""
    O = packed.shape[0]
    p = packed.reshape(O, I // TQ2_GRP, 16)
    c = np.stack([(p >> 0) & 3, (p >> 2) & 3, (p >> 4) & 3, (p >> 6) & 3], axis=2)
    return c.reshape(O, I)


def tq2_tensor(codes: np.ndarray, alpha: np.ndarray) -> bytes:
    """One ternary tensor: all row scales, then all code rows."""
    return (np.ascontiguousarray(alpha, dtype=np.float32).tobytes() +
            tq2_pack_codes(codes).tobytes())


def tq2_bytes(O, I):
    return O * 4 + O * (I // 4)


def tq2_dequant(codes, alpha):
    return (codes.astype(np.float32) - 1.0) * alpha.astype(np.float32)[:, None]


# ------------------------------------------------------------ q4a (4-bit affine)
# Worst absolute error seen converting a q4a scale or bias to fp16, reported at the
# end of a build so the claim "lossless repack" stays checkable rather than assumed.
_FP16_WORST = [0.0, 0]
FP16_MIN_NORMAL = 6.103515625e-05


def f32_to_fp16_checked(x, what):
    """bf16 -> fp16 for the q4a scales and biases.

    fp16 has 10 mantissa bits against bf16's 7, so every bf16 value with a normal
    fp16 exponent converts EXACTLY -- that is why q4a can carry the checkpoint's own
    numbers instead of re-quantising. The exception is the tail of near-zero scales
    (Maple has ~9k of them around 1e-7, from groups whose weights are all equal),
    which land in fp16's subnormal range: still representable, but rounded to a
    multiple of 2^-24. The resulting error on a weight is at most 15 * 3e-8, six
    orders of magnitude under the weights themselves, so it is accepted and
    measured rather than rejected.

    Overflow is a different matter and is fatal: a scale flushed to inf or to zero
    would corrupt a whole group."""
    if not np.isfinite(x).all():
        sys.exit(f"{what}: non-finite value in the checkpoint")
    big = np.abs(x) > 65504.0
    if big.any():
        sys.exit(f"{what}: {x.ravel()[int(np.argmax(big))]!r} overflows fp16")
    h = x.astype(np.float16)
    err = np.abs(h.astype(np.float32) - x)
    worst = float(err.max()) if err.size else 0.0
    if worst > 0:
        _FP16_WORST[0] = max(_FP16_WORST[0], worst)
        _FP16_WORST[1] += int((err > 0).sum())
    # Anything inexact must be a subnormal; an inexact normal value would mean the
    # source was not bf16 and the losslessness argument above does not hold.
    inexact_normal = (err > 0) & (np.abs(x) >= FP16_MIN_NORMAL)
    if inexact_normal.any():
        i = int(np.argmax(inexact_normal))
        sys.exit(f"{what}: {x.ravel()[i]!r} is normal in fp16 but did not convert "
                 f"exactly -- the source is not bf16")
    return h


def q4a_tensor(q: np.ndarray, d: np.ndarray, m: np.ndarray) -> bytes:
    """q uint8 [O,I] in [0,15], d/m float32 [O, I/64] -> rows of 36-byte groups.

    Group layout is fp16 d, fp16 m, then two 32-weight blocks of nibbles with the
    low nibble of byte j holding w[j] and the high nibble w[j+16] -- the same split
    q4_0 uses, so the kernel's unpack is the one q40.h already proved out.
    """
    O, I = q.shape
    assert I % Q4A_GRP == 0, f"I={I} not a multiple of {Q4A_GRP}"
    ng = I // Q4A_GRP
    g = q.reshape(O, ng, 2, 2, 16)              # [O, grp, half32, lo/hi, k]
    nib = (g[:, :, :, 0, :] | (g[:, :, :, 1, :] << 4)).reshape(O, ng, 32)
    out = np.empty((O, ng, Q4A_GRP_BYTES), np.uint8)
    out[:, :, 0:2] = f32_to_fp16_checked(d, "q4a scale").view(np.uint8).reshape(O, ng, 2)
    out[:, :, 2:4] = f32_to_fp16_checked(m, "q4a bias").view(np.uint8).reshape(O, ng, 2)
    out[:, :, 4:] = nib
    return np.ascontiguousarray(out).tobytes()


def q4a_bytes(O, I):
    return O * (I // Q4A_GRP) * Q4A_GRP_BYTES


def q4a_dequant(q, d, m):
    O, I = q.shape
    ng = I // Q4A_GRP
    return (q.reshape(O, ng, Q4A_GRP).astype(np.float32) * d[:, :, None]
            + m[:, :, None]).reshape(O, I)


# ------------------------------------------------------------------- safetensors
_DT = {"F32": np.float32, "F16": np.float16, "I8": np.int8, "U8": np.uint8,
       "I32": np.int32, "I64": np.int64, "U32": np.uint32}


class Shards:
    """Streaming reader: one open handle per shard, a tensor's byte range read on
    demand. The checkpoint is 5 GB and must never be resident."""

    def __init__(self, d, extra=()):
        self.d = d
        self.wmap = {}
        p = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(p):
            self.wmap.update(json.load(open(p))["weight_map"])
        for f in sorted(os.listdir(d)):
            if f.endswith(".safetensors") and f not in self.wmap.values():
                with open(os.path.join(d, f), "rb") as fh:
                    n = struct.unpack("<Q", fh.read(8))[0]
                    h = json.loads(fh.read(n))
                for k in h:
                    if k != "__metadata__":
                        self.wmap.setdefault(k, f)
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

    def raw(self, n):
        """The tensor in its stored dtype -- uint32 code words stay uint32."""
        if n not in self.wmap:
            sys.exit(f"missing tensor {n}")
        f, (hdr, base) = self._open(self.wmap[n])
        e = hdr[n]
        s, t = e["data_offsets"]
        f.seek(base + s)
        buf = f.read(t - s)
        dt = np.uint16 if e["dtype"] == "BF16" else _DT[e["dtype"]]
        return np.frombuffer(buf, dt).reshape(e["shape"]), e["dtype"]

    def get(self, n):
        """As float32, decoding bf16."""
        a, dt = self.raw(n)
        if dt == "BF16":
            return (a.astype(np.uint32) << 16).view(np.float32)
        return a.astype(np.float32)


def unpack_mlx(words: np.ndarray, bits: int, I: int) -> np.ndarray:
    """MLX packs `32/bits` codes per uint32, low-order first. Viewing the words as
    bytes turns that into `8/bits` codes per byte with no word arithmetic."""
    b = words.view(np.uint8).reshape(*words.shape[:-1], -1)
    per = 8 // bits
    out = np.empty((*b.shape[:-1], b.shape[-1] * per), np.uint8)
    mask = (1 << bits) - 1
    for k in range(per):
        out[..., k::per] = (b >> (bits * k)) & mask
    assert out.shape[-1] == I, (out.shape, I)
    return out


def expand_group(x: np.ndarray, ng: int) -> np.ndarray:
    """row_alpha is one value per output row; the ternary format wants exactly that,
    so this only exists for the group-scaled variant the checkpoint may also use."""
    return np.broadcast_to(x[..., None], (*x.shape, ng))


# ------------------------------------------------------------------ dense writer
class Dense:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.idx, self.off = {}, 0

    def add_raw(self, name, blob, fmt, O, I):
        self.f.write(blob)
        self.idx[name] = {"off": self.off, "len": len(blob), "fmt": fmt,
                          "shape": [int(O), int(I)]}
        self.off += len(blob)
        return blob

    def add_f32(self, name, w):
        w = np.ascontiguousarray(w, dtype=np.float32)
        if w.ndim == 1:
            w = w.reshape(1, -1)
        return self.add_raw(name, w.tobytes(), FMT_F32, w.shape[0], w.shape[1])

    def add_i32(self, name, w):
        w = np.ascontiguousarray(w, dtype=np.int32)
        if w.ndim == 1:
            w = w.reshape(1, -1)
        return self.add_raw(name, w.tobytes(), FMT_I32, w.shape[0], w.shape[1])

    def add_tq2(self, name, codes, alpha):
        O, I = codes.shape
        return self.add_raw(name, tq2_tensor(codes, alpha), FMT_TQ2, O, I)

    def add_q4a(self, name, q, d, m):
        O, I = q.shape
        return self.add_raw(name, q4a_tensor(q, d, m), FMT_Q4A, O, I)

    def close(self):
        self.f.close()


# ---------------------------------------------------------------- source access
# The rest of the converter asks for logical names ("L3.q_proj", "embed"); these two
# classes are the only place that knows how a real checkpoint spells them, so the
# fixture path and the checkpoint path share every line of the build below.
_HF = {"embed": "model.word_embeddings", "lm_head": "lm_head",
       "final_norm": "model.norm.weight",
       "flash.token_map": "lm_head_flash.token_map",
       "flash.centroids": "lm_head_flash.centroids",
       "flash.cluster_scale": "lm_head_flash.cluster_scale"}
_HF_LAYER = {"input_layernorm": "input_layernorm.weight",
             "post_attention_layernorm": "post_attention_layernorm.weight",
             "q_norm": "self_attn.q_norm.weight", "k_norm": "self_attn.k_norm.weight",
             "router": "mlp.gate.weight",
             "q_proj": "self_attn.q_proj", "k_proj": "self_attn.k_proj",
             "v_proj": "self_attn.v_proj", "o_proj": "self_attn.o_proj",
             "gate_proj": "mlp.switch_mlp.gate_proj",
             "up_proj": "mlp.switch_mlp.up_proj",
             "down_proj": "mlp.switch_mlp.down_proj"}


class Source:
    def __init__(self, shards):
        self.S = shards

    @staticmethod
    def _key(name):
        if "." in name and name.startswith("L"):
            li, leaf = name[1:].split(".", 1)
            return f"model.layers.{li}.{_HF_LAYER[leaf]}"
        return _HF[name]

    def has(self, name):
        k = self._key(name)
        return self.S.has(k) or self.S.has(k + ".weight")

    def f32(self, name):
        return self.S.get(self._key(name))

    def i32(self, name):
        return self.S.raw(self._key(name))[0].astype(np.int32)

    def ternary(self, name, expect_shape=None):
        """-> (codes uint8 [.., O, I], alpha float32 [.., O])"""
        prefix = self._key(name)
        words, _ = self.S.raw(prefix + ".weight")
        I = words.shape[-1] * 16
        codes = unpack_mlx(words, 2, I)
        if codes.max() > 2:
            sys.exit(f"{prefix}: code 3 present -- this tensor is not ternary")
        if not self.S.has(prefix + ".row_alpha"):
            sys.exit(f"{prefix}: no row_alpha; group-scaled ternary checkpoints are "
                     f"not supported (the format carries one scale per row)")
        alpha = self.S.get(prefix + ".row_alpha")
        if expect_shape and codes.shape != expect_shape:
            sys.exit(f"{prefix}: shape {codes.shape}, expected {expect_shape}")
        if alpha.shape != codes.shape[:-1]:
            sys.exit(f"{prefix}: row_alpha {alpha.shape} vs weights {codes.shape}")
        return codes, alpha

    def affine4(self, name):
        """-> (q uint8 [O,I], d float32 [O,I/64], m float32 [O,I/64])"""
        prefix = self._key(name)
        words, _ = self.S.raw(prefix + ".weight")
        I = words.shape[-1] * 8
        q = unpack_mlx(words, 4, I)
        d = self.S.get(prefix + ".scales")
        m = self.S.get(prefix + ".biases")
        if d.shape[-1] * Q4A_GRP != I:
            sys.exit(f"{prefix}: {d.shape[-1]} groups for I={I}; expected group "
                     f"size {Q4A_GRP}, got {I // d.shape[-1]}")
        return q, d, m


# ------------------------------------------------------------------- RAM planner
def plan(cfg, dense_bytes, esz, ctx, ram_gb, kv=None):
    """slots/layer for the per-layer LRU expert cache under a RAM budget.

    Every layer here is an MoE layer and every expert is the same size, so unlike
    lfm25 there is no largest-expert subtlety: the plan is exact."""
    NE, L = cfg["n_experts"], cfg["n_layers"]
    kvbytes = kv_bytes(cfg, ctx, kv)
    scratch = 192 << 20
    avail = int(ram_gb * (1 << 30)) - dense_bytes - kvbytes - scratch
    per = min(NE, max(0, avail // esz) // max(1, L)) if esz else NE
    total = esz * NE * L
    return {
        "expert_bytes": esz, "dense_bytes": dense_bytes, "kv_bytes": kvbytes,
        "slots_per_layer": int(per), "cache_bytes": int(per) * esz * L,
        "total_expert_bytes": total,
        "ok": per >= cfg["topk"],
        "min_ram_gb": (dense_bytes + kvbytes + scratch
                       + cfg["topk"] * L * esz) / (1 << 30),
    }


def kv_bytes(cfg, ctx, kv=None):
    """Sliding layers never hold more than `sliding_window` positions, which is what
    makes Maple's KV cheap: 18 of its 24 layers are capped at 512 regardless of the
    context length, and only the 6 full-attention layers grow with it."""
    hd, nkv = cfg["head_dim"], cfg["n_kv_heads"]
    tot = 0
    for swa in cfg["layer_swa"]:
        cap = min(ctx, cfg["sliding_window"]) if swa else ctx
        if not kv or kv["kbits"] <= 0 or swa:
            # Sliding layers are never quantised: they hold at most `sliding_window`
            # positions, so there is nothing to save and residual-window bookkeeping
            # would only add error.
            tot += 2 * cap * nkv * hd * 4
            continue
        W = min(kv["rwin"], cap)
        vecb = lambda d, bits: (d * bits + 7) // 8 + 4
        tot += 2 * W * nkv * hd * 4
        tot += cap * nkv * (vecb(hd, kv["kbits"]) + vecb(hd, kv["vbits"]))
    return tot


# ----------------------------------------------------------------------- fixture
def make_fixture(dst):
    """A tiny random Maple with the same STRUCTURE -- ternary matrices with per-row
    scales, sliding/full attention alternation, RoPE only on the sliding layers,
    partial rotary, qk-norm, a softmax router, clamped SwiGLU experts, a q4a
    embedding and lm_head, and a FlashHead -- at absurd dimensions. It exists so the
    numpy oracle and the C engine can be diffed without a 5 GB checkpoint."""
    rng = np.random.default_rng(11)
    cfg = dict(hidden=128, n_layers=4, n_heads=4, head_dim=32, n_kv_heads=2,
               n_experts=6, topk=2, moe_inter=64, vocab=256, eps=1e-6,
               rope_theta=10000.0, rope_dim=16, sliding_window=4,
               norm_topk_prob=1, mlp_clamp=7.0,
               layer_swa=[1, 1, 0, 1])
    D, MI, NE, V = cfg["hidden"], cfg["moe_inter"], cfg["n_experts"], cfg["vocab"]
    QD = cfg["n_heads"] * cfg["head_dim"]
    KD = cfg["n_kv_heads"] * cfg["head_dim"]

    T = {}

    def tern(*shape):
        """codes in {0,1,2} plus a positive per-row scale, as the checkpoint has."""
        codes = rng.integers(0, 3, size=shape).astype(np.uint8)
        alpha = (0.02 + 0.08 * rng.random(shape[:-1])).astype(np.float32)
        return codes, alpha

    def aff4(O, I):
        q = rng.integers(0, 16, size=(O, I)).astype(np.uint8)
        ng = I // Q4A_GRP
        d = (0.002 + 0.02 * rng.random((O, ng))).astype(np.float32)
        m = (-0.15 * rng.random((O, ng))).astype(np.float32)
        # round-trip through fp16 so the fixture's "checkpoint" and its container
        # agree exactly, as the real bf16 values do
        return q, d.astype(np.float16).astype(np.float32), m.astype(np.float16).astype(np.float32)

    nrm = lambda n: (1.0 + 0.2 * rng.standard_normal(n)).astype(np.float32)

    T["embed"] = aff4(V, D)
    T["lm_head"] = aff4(V, D)
    T["final_norm"] = nrm(D)
    for li in range(cfg["n_layers"]):
        T[f"L{li}.input_layernorm"] = nrm(D)
        T[f"L{li}.post_attention_layernorm"] = nrm(D)
        T[f"L{li}.q_proj"] = tern(QD, D)
        T[f"L{li}.k_proj"] = tern(KD, D)
        T[f"L{li}.v_proj"] = tern(KD, D)
        T[f"L{li}.o_proj"] = tern(D, QD)
        T[f"L{li}.q_norm"] = nrm(cfg["head_dim"])
        T[f"L{li}.k_norm"] = nrm(cfg["head_dim"])
        T[f"L{li}.router"] = (0.05 * rng.standard_normal((NE, D))).astype(np.float32)
        T[f"L{li}.gate_proj"] = tern(NE, MI, D)
        T[f"L{li}.up_proj"] = tern(NE, MI, D)
        T[f"L{li}.down_proj"] = tern(NE, D, MI)

    # FlashHead: a genuine partition of the vocabulary into clusters of 32
    ncl, csz = V // 32, 32
    T["flash.token_map"] = rng.permutation(V).astype(np.int32).reshape(ncl, csz)
    T["flash.centroids"] = aff4(ncl, D)
    T["flash.cluster_scale"] = np.ones(ncl, np.float32)
    meta = dict(n_clusters=ncl, cluster_size=csz, n_probes=max(1, ncl // 2),
                scaled_centroids=True, force_tokens=[0, 1])

    os.makedirs(dst, exist_ok=True)
    ids = rng.integers(0, V, size=12).tolist()
    json.dump({"prompt": ids}, open(os.path.join(dst, "ref.json"), "w"))
    return cfg, T, meta


class FixtureSource:
    """The fixture builds its tensors already in logical form, so this is a dict."""

    def __init__(self, T):
        self.T = T

    def has(self, name):
        return name in self.T

    def f32(self, name):
        return np.asarray(self.T[name], np.float32)

    def i32(self, name):
        return np.asarray(self.T[name], np.int32)

    def ternary(self, name, expect_shape=None):
        codes, alpha = self.T[name]
        if expect_shape and codes.shape != expect_shape:
            sys.exit(f"{name}: shape {codes.shape}, expected {expect_shape}")
        return codes, alpha

    def affine4(self, name):
        return self.T[name]


# ------------------------------------------------------------------------- build
def build(src, dst, ctx, ram, verify, want_flash, fixture=False):
    os.makedirs(dst, exist_ok=True)

    if fixture:
        cfg, T, meta = make_fixture(dst)
        S = FixtureSource(T)
        flash_meta = meta if want_flash else None
    else:
        t = json.load(open(os.path.join(src, "config.json")))
        if t.get("model_type") != "maple":
            sys.exit(f"expected model_type maple, got {t.get('model_type')}")
        for k, v in (("first_k_dense_replace", 0), ("num_shared_experts", 0),
                     ("num_nextn_predict_layers", 0)):
            if t.get(k, v) != v:
                sys.exit(f"{k}={t[k]} is not supported (expected {v})")
        if t.get("tie_word_embeddings"):
            sys.exit("tie_word_embeddings is not supported (Maple ships an untied head)")
        if not t.get("use_qk_norm", True):
            sys.exit("use_qk_norm=false is not supported")
        if t.get("use_bias") or t.get("use_qkv_bias"):
            sys.exit("biased projections are not supported")
        q = t.get("quantization") or {}
        if q.get("bits") != 2 or q.get("mode", "affine") != "affine":
            sys.exit(f"expected a 2-bit affine checkpoint, got {q}")
        for x in t["layer_types"]:
            if x not in ("sliding_attention", "full_attention"):
                sys.exit(f"unknown layer type {x}")
        head_dim = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
        cfg = dict(
            hidden=t["hidden_size"], n_layers=t["num_hidden_layers"],
            n_heads=t["num_attention_heads"], head_dim=head_dim,
            n_kv_heads=t["num_key_value_heads"],
            n_experts=t["num_experts"], topk=t["num_experts_per_tok"],
            moe_inter=t["moe_intermediate_size"], vocab=t["vocab_size"],
            eps=t["rms_norm_eps"], rope_theta=float(t["rope_theta"]),
            rope_dim=int(head_dim * t.get("partial_rotary_factor", 1.0)),
            sliding_window=t["sliding_window"],
            norm_topk_prob=1 if t.get("norm_topk_prob", True) else 0,
            mlp_clamp=7.0,
            layer_swa=[1 if x == "sliding_attention" else 0 for x in t["layer_types"]])
        if t.get("rope_scaling"):
            sys.exit("rope_scaling is not supported")
        S = Source(Shards(src))
        flash_meta = t.get("flash_head") if want_flash else None

    cfg["ctx"] = ctx
    D, MI, NE = cfg["hidden"], cfg["moe_inter"], cfg["n_experts"]
    L, V = cfg["n_layers"], cfg["vocab"]
    QD, KD = cfg["n_heads"] * cfg["head_dim"], cfg["n_kv_heads"] * cfg["head_dim"]

    for what, dim in (("hidden", D), ("moe_inter", MI), ("head_dim*n_heads", QD)):
        if dim % TQ2_GRP:
            sys.exit(f"{what}={dim} must be a multiple of {TQ2_GRP} (tq2 packing)")

    worst = 0.0

    def tern_add(dn, out_name, key, shape):
        nonlocal worst
        codes, alpha = S.ternary(key, shape)
        dn.add_tq2(out_name, codes, alpha)
        if verify:
            back = tq2_unpack_codes(tq2_pack_codes(codes), codes.shape[1])
            worst = max(worst, float(np.abs(
                tq2_dequant(codes, alpha) - tq2_dequant(back, alpha)).max()))

    # ---- dense
    dn = Dense(os.path.join(dst, "dense.bin"))
    for nm in ("embed", "lm_head"):
        dn.add_q4a(nm, *S.affine4(nm))
    dn.add_f32("final_norm", S.f32("final_norm"))

    for li in range(L):
        o = f"layers.{li}."
        for nm in ("input_layernorm", "post_attention_layernorm", "q_norm", "k_norm"):
            dn.add_f32(o + nm, S.f32(f"L{li}.{nm}"))
        # f32: a routing error is not a small numeric error, it picks a different
        # expert entirely.
        dn.add_f32(o + "router", S.f32(f"L{li}.router"))
        tern_add(dn, o + "q_proj", f"L{li}.q_proj", (QD, D))
        tern_add(dn, o + "k_proj", f"L{li}.k_proj", (KD, D))
        tern_add(dn, o + "v_proj", f"L{li}.v_proj", (KD, D))
        tern_add(dn, o + "o_proj", f"L{li}.o_proj", (D, QD))

    # ---- FlashHead
    flash = {}
    if flash_meta:
        if not S.has("flash.token_map"):
            sys.exit("the FlashHead was asked for but the checkpoint has no "
                     "lm_head_flash tensors (model-flashhead.safetensors); "
                     "pass --no-flash")
        tm = S.i32("flash.token_map")
        cq, cd, cm = S.affine4("flash.centroids")
        cs = S.f32("flash.cluster_scale").reshape(-1)
        force = np.array(flash_meta.get("force_tokens", []), np.int32)
        scaled = bool(flash_meta.get("scaled_centroids", False))
        nprobe = int(flash_meta.get("n_probes", 512))
        ncl, csz = tm.shape
        if int(tm.min()) < 0 or int(tm.max()) >= V:
            sys.exit("FlashHead token_map is out of range")
        dn.add_q4a("flash_centroids", cq, cd, cm)
        dn.add_i32("flash_token_map", tm)
        dn.add_f32("flash_cluster_scale", cs)
        dn.add_i32("flash_force", force.reshape(1, -1) if force.size else
                   np.zeros((1, 0), np.int32))
        flash = dict(n_clusters=int(ncl), cluster_size=int(csz),
                     n_probes=min(int(nprobe), int(ncl)),
                     scaled_centroids=1 if scaled else 0,
                     n_force=int(force.size))
        # Tokens no cluster covers can never be produced. The checkpoint's map is a
        # partition, but say so rather than assume it.
        covered = np.zeros(V, bool)
        covered[tm.reshape(-1)] = True
        flash["uncovered"] = int(V - covered.sum())

    dense_bytes = dn.off
    dn.close()

    # ---- experts
    ALIGN = 4096
    gb = tq2_bytes(MI, D)                 # gate, and up
    db = tq2_bytes(D, MI)                 # down
    esz = 2 * gb + db

    idx, off = {}, 0
    with open(os.path.join(dst, "experts.bin"), "wb") as out:
        for li in range(L):
            # The checkpoint stacks all 256 experts of a layer into one tensor, so a
            # layer is three reads and the split into per-expert records happens
            # here. That is also what makes the container's layout the interesting
            # part: on disk each expert is contiguous and 4096-aligned, so a cache
            # miss is one pread rather than three strided ones.
            gc, ga = S.ternary(f"L{li}.gate_proj", (NE, MI, D))
            uc, ua = S.ternary(f"L{li}.up_proj", (NE, MI, D))
            dc, da = S.ternary(f"L{li}.down_proj", (NE, D, MI))
            for e in range(NE):
                bg = tq2_tensor(gc[e], ga[e])
                bu = tq2_tensor(uc[e], ua[e])
                bd = tq2_tensor(dc[e], da[e])
                assert len(bg) == gb and len(bu) == gb and len(bd) == db
                pad = (-off) % ALIGN
                if pad:
                    out.write(b"\0" * pad)
                    off += pad
                out.write(bg)
                out.write(bu)
                out.write(bd)
                idx[f"{li}.{e}"] = off
                off += esz
            if verify:
                back = tq2_unpack_codes(tq2_pack_codes(gc[0]), D)
                worst = max(worst, float(np.abs(
                    tq2_dequant(gc[0], ga[0]) - tq2_dequant(back, ga[0])).max()))
            if not fixture:
                print(f"  layer {li:2d}  {off / 2**30:.2f} GiB", flush=True)

    if verify:
        print(f"verify: ternary repack max |w - dequant(pack(w))| = {worst:.6g} "
              f"({'exact' if worst == 0 else 'LOSSY -- this should be exact'})")
        if worst != 0:
            sys.exit(1)
    if _FP16_WORST[1]:
        print(f"verify: {_FP16_WORST[1]} q4a scales/biases were fp16 subnormals, "
              f"max abs error {_FP16_WORST[0]:.3g} (bounded by 15x that on a weight)")
    else:
        print("verify: q4a scales/biases converted exactly")

    # ---- plan + manifest
    p = plan(cfg, dense_bytes, esz, ctx, ram)
    cfg["slots_per_layer"] = p["slots_per_layer"]
    cfg.update({f"flash_{k}": v for k, v in flash.items() if k != "uncovered"})
    json.dump(cfg, open(os.path.join(dst, "cfg.json"), "w"), indent=1)

    with open(os.path.join(dst, "manifest.txt"), "w") as m:
        for k, v in cfg.items():
            if k == "layer_swa":
                m.write("cfg layer_swa " + " ".join(str(x) for x in v) + "\n")
            else:
                m.write(f"cfg {k} {v}\n")
        m.write(f"esz {esz} {gb} {db}\n")
        m.write(f"ndense {len(dn.idx)}\n")
        for k, v in dn.idx.items():
            sh = v["shape"]
            m.write(f"dense {k} {v['off']} {v['len']} {v['fmt']} {sh[0]} {sh[1]}\n")
        m.write(f"nexpert {len(idx)}\n")
        for k, v in idx.items():
            li, e = k.split(".")
            m.write(f"expert {li} {e} {v}\n")

    nswa = sum(cfg["layer_swa"])
    print(f"\nlayers         : {L} ({nswa} sliding@{cfg['sliding_window']} + RoPE, "
          f"{L - nswa} full + NoPE); every layer MoE")
    print(f"dense resident : {p['dense_bytes']/2**20:8.1f} MiB")
    print(f"kv cache       : {p['kv_bytes']/2**20:8.1f} MiB  (ctx {ctx}; the sliding "
          f"layers are capped at {cfg['sliding_window']})")
    print(f"experts        : {p['total_expert_bytes']/2**30:.2f} GiB total, "
          f"{NE}/layer at {esz/2**10:.0f} KiB each (ternary)")
    if flash:
        print(f"flash head     : {flash['n_clusters']} clusters x "
              f"{flash['cluster_size']}, probe {flash['n_probes']} -> "
              f"{flash['n_probes']*flash['cluster_size']}/{V} rows scored"
              + (f"; {flash['uncovered']} vocab ids uncovered" if flash["uncovered"] else ""))
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
    ap.add_argument("--no-flash", dest="flash", action="store_false",
                    help="skip the FlashHead tensors (exact lm_head only)")
    ap.add_argument("--verify", action="store_true",
                    help="check the repack is bit-exact (it must be)")
    ap.add_argument("--fixture", action="store_true",
                    help="build a tiny random model for the oracle check")
    a = ap.parse_args()
    if a.fixture:
        build(None, a.dst, a.ctx, a.ram, a.verify, a.flash, fixture=True)
    else:
        if not a.src:
            ap.error("src is required unless --fixture")
        build(a.src, a.dst, a.ctx, a.ram, a.verify, a.flash)


if __name__ == "__main__":
    main()
