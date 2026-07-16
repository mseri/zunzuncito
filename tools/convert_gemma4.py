#!/usr/bin/env python3
"""
convert_gemma4.py — Gemma-4 26B-A4B (QAT q4_0-unquantized) -> colibri container.

OUTPUT
  dense.bin / dense.idx     resident: embeddings, norms, attention, the per-layer
                            dense MLP (the "shared expert"), router. Matrices in
                            q4_0; norms/scalars in f32.
  experts.bin / experts.idx streamed: 30 x 128 experts, q4_0, each expert ONE
                            contiguous 4096-aligned blob -> one pread, no seek.
  cfg.json                  flattened config for the engine.

WHY q4_0 EVERYWHERE (dense too, not colibri's per-row fmt=2)
  One format, one kernel, and -- the reason that matters -- dense lands at 1.31 GB
  instead of ~2 GB, which is what makes a 4 GB budget feasible at all. q4_0 also
  carries its fp16 scales INSIDE the weight bytes (one per 32 weights), so an
  expert is a single contiguous byte range: no companion scale tensor to seek to.
  With 240 expert reads per token, halving the read COUNT matters more than
  anything we could do to bandwidth.

EXPERT LAYOUT (per expert): [ gate | up | down ], all q4_0, 4096-aligned.
  gate [MI, D], up [MI, D], down [D, MI]  ->  3,346,176 B (~3.19 MiB) at the real
  dims (D=2816, MI=704).

  gate_up_proj ships as [E, 2*MI, D] -- ALREADY [out, in], since Gemma4TextExperts
  does F.linear(x, gate_up_proj[e]) then .chunk(2, dim=-1) on the OUTPUT. So gate
  = rows [0:MI], up = rows [MI:2*MI], and NO transpose is needed. down_proj is
  [E, D, MI], likewise already [out, in]. Shapes are asserted: an accidental
  transpose here is silent garbage, not a crash.
"""
import argparse, json, os, struct, sys
import numpy as np

QK, BLK = 32, 18


# ------------------------------------------------- q4_0 (mirrors q40.h bit-for-bit)
def q40_quant_rows(w: np.ndarray) -> bytes:
    O, I = w.shape
    assert I % QK == 0, f"I={I} not a multiple of {QK}"
    nb = I // QK
    x = w.reshape(O, nb, QK).astype(np.float32)

    # q4_0's codebook is asymmetric (d*[-8..+7]), so the scale keys off the SIGNED
    # max-|.| element, not its magnitude.
    ai = np.abs(x).argmax(axis=2)
    mx = np.take_along_axis(x, ai[:, :, None], axis=2)[:, :, 0]

    d = (mx / -8.0).astype(np.float32)
    d = d.astype(np.float16).astype(np.float32)   # round to fp16 BEFORE deriving id:
                                                  # the decoder only ever sees the fp16
                                                  # value, so quantising against the
                                                  # unrounded scale would push the weight
                                                  # error past the d/2 bound.
    idv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0).astype(np.float32)

    q = np.floor(x * idv[:, :, None] + 8.5).astype(np.int32)
    np.clip(q, 0, 15, out=q)
    nib = (q[:, :, :16].astype(np.uint8) | (q[:, :, 16:].astype(np.uint8) << 4))

    out = np.empty((O, nb, BLK), dtype=np.uint8)
    out[:, :, 0:2] = d.astype(np.float16).view(np.uint8).reshape(O, nb, 2)
    out[:, :, 2:] = nib
    return out.tobytes()


def q40_dequant_rows(buf, O, I):
    nb = I // QK
    a = np.frombuffer(buf, np.uint8).reshape(O, nb, BLK)
    d = a[:, :, 0:2].copy().view(np.float16).astype(np.float32).reshape(O, nb)
    nib = a[:, :, 2:]
    lo = (nib & 0x0F).astype(np.int32) - 8
    hi = (nib >> 4).astype(np.int32) - 8
    w = np.concatenate([lo, hi], 2).astype(np.float32)
    return (w * d[:, :, None]).reshape(O, I)


def q40_bytes(O, I):
    return O * (I // QK) * BLK


# ------------------------------------------------------------------ safetensors
_DT = {"F32": np.float32, "F16": np.float16, "I8": np.int8, "U8": np.uint8}


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


class Shards:
    def __init__(self, d):
        p = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(p):
            self.wmap = json.load(open(p))["weight_map"]
        else:
            self.wmap = {}
            for f in sorted(os.listdir(d)):
                if f.endswith(".safetensors"):
                    h, _ = st_header(os.path.join(d, f))
                    for k in h:
                        if k != "__metadata__":
                            self.wmap[k] = f
        self.d, self.cache = d, {}

    def _h(self, fn):
        if fn not in self.cache:
            self.cache[fn] = st_header(os.path.join(self.d, fn))
        return self.cache[fn]

    def has(self, n):
        return n in self.wmap

    def get(self, n):
        fn = self.wmap[n]
        hdr, base = self._h(fn)
        e = hdr[n]
        s, t = e["data_offsets"]
        with open(os.path.join(self.d, fn), "rb") as f:
            f.seek(base + s)
            raw = f.read(t - s)
        if e["dtype"] == "BF16":
            a = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            a = np.frombuffer(raw, _DT[e["dtype"]]).astype(np.float32)
        return a.reshape(e["shape"])


# ------------------------------------------------------------------ dense writer
class Dense:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.idx, self.off = {}, 0

    def add_q40(self, name, w):
        assert w.ndim == 2
        b = q40_quant_rows(np.ascontiguousarray(w, dtype=np.float32))
        self.f.write(b)
        self.idx[name] = {"off": self.off, "len": len(b), "fmt": "q40",
                          "shape": [int(w.shape[0]), int(w.shape[1])]}
        self.off += len(b)

    def add_f32(self, name, w):
        w = np.ascontiguousarray(w, dtype=np.float32)
        b = w.tobytes()
        self.f.write(b)
        self.idx[name] = {"off": self.off, "len": len(b), "fmt": "f32",
                          "shape": [int(x) for x in w.shape]}
        self.off += len(b)

    def close(self, path):
        self.f.close()
        json.dump(self.idx, open(path, "w"))


# ------------------------------------------------------------------ RAM planner
def plan(cfg, dense_bytes, ctx, ram_gb, kv=None):
    """slots/layer for the per-layer LRU expert cache under a RAM budget.

    FLOOR: a layer must be able to hold the topk experts the current token routes
    to, or the cache thrashes within a single forward. Below that we say so rather
    than pretend.
    """
    D, L, NE, MI = cfg["hidden"], cfg["n_layers"], cfg["n_experts"], cfg["moe_inter"]
    esz = 2 * q40_bytes(MI, D) + q40_bytes(D, MI)

    # NOTE attention_k_eq_v does NOT halve storage: K and V share the k projection
    # but differ afterwards (K = rope(k_norm(raw)), V = v_norm(raw)), so the engine
    # caches both. An earlier version of this planner counted global layers once and
    # under-reported their KV by 2x.
    # All KV GROWTH lives in the global layers: sliding layers are permanently capped
    # at `sliding_window` positions, so they cost the same at 4K as at 256K ctx.
    # kv = None -> f32. Otherwise dict(kbits, vbits, rwin, protect, pbits): the f32
    # residual window plus a TurboQuant-packed store for everything older.
    def vecb(d, bits):
        return (d * bits + 7) // 8 + 4          # payload + f32 norm
    kvbytes = 0
    L = cfg["n_layers"]
    for li, t in enumerate(cfg["layer_types"]):
        hd  = cfg["global_head_dim"] if t else cfg["head_dim"]
        nkv = cfg["n_global_kv_heads"] if t else cfg["n_kv_heads"]
        cap = ctx if t else min(ctx, cfg["sliding_window"])
        if not kv or kv["kbits"] <= 0:
            kvbytes += 2 * cap * nkv * hd * 4
            continue
        prot = li < kv["protect"] or li >= L - kv["protect"]
        kb = kv["pbits"] if prot else kv["kbits"]
        vb = kv["pbits"] if prot else kv["vbits"]
        W = min(kv["rwin"], cap)
        kvbytes += 2 * W * nkv * hd * 4                       # f32 residual window
        kvbytes += cap * nkv * (vecb(hd, kb) + vecb(hd, vb))  # packed store
    kv = kvbytes

    scratch = 192 << 20
    avail = int(ram_gb * (1 << 30)) - dense_bytes - kv - scratch
    per = min(NE, max(0, avail // esz) // L)
    return {
        "expert_bytes": esz, "dense_bytes": dense_bytes, "kv_bytes": kv,
        "slots_per_layer": int(per), "cache_bytes": int(per) * L * esz,
        "total_expert_bytes": L * NE * esz,
        "ok": per >= cfg["topk"],
        "min_ram_gb": (dense_bytes + kv + scratch + cfg["topk"] * L * esz) / (1 << 30),
    }


# ------------------------------------------------------------------ build
def build(src, dst, ctx, ram, verify, fixture=None):
    os.makedirs(dst, exist_ok=True)

    if fixture is not None:
        S, t, P = fixture
    else:
        raw = json.load(open(os.path.join(src, "config.json")))

        # The MTP drafter (Gemma4AssistantForCausalLM) is NOT a standalone model and
        # cannot be converted to a container that gemma4.c can run. It is an
        # EAGLE/MTP-style head bolted onto the backbone:
        #   * its layers have NO k_proj / v_proj / k_norm at all -- it cannot compute
        #     its own K/V, and num_kv_shared_layers == num_hidden_layers means every
        #     layer reuses the BACKBONE's KV cache;
        #   * pre_projection is [hidden, 2 * backbone_hidden_size], i.e. it consumes
        #     the backbone's hidden states, and post_projection maps back into
        #     backbone space so it can iterate there across draft steps;
        #   * enable_moe_block is false and num_experts is null.
        # Running it as an independent model therefore produces garbage, not a speedup.
        # Supporting it means running the head INSIDE the target's forward, against the
        # target's KV -- a different design from --draft, not a conversion problem.
        arch = (raw.get("architectures") or [""])[0]
        if raw.get("model_type") == "gemma4_assistant" or "Assistant" in arch:
            sys.exit(
                "This is the Gemma-4 MTP drafter (%s), which is a draft HEAD on the\n"
                "backbone, not a standalone model: its layers have no k_proj/v_proj and\n"
                "it attends into the TARGET's KV cache, while pre/post_projection read and\n"
                "write the backbone's %s-dim hidden states.\n"
                "It cannot be converted to a runnable container, and using one as --draft\n"
                "will produce garbage. Speculative decoding with it needs the head to run\n"
                "inside the target's forward -- not yet implemented." %
                (arch or raw.get("model_type"), raw.get("backbone_hidden_size", "?")))

        t = raw["text_config"]
        S = Shards(src)
        P = "model.language_model."

        # A dense (non-MoE) Gemma-4 text model would also break the engine, which
        # assumes a routed MoE in every layer. Catch it here rather than at inference.
        if not t.get("enable_moe_block", True) or not t.get("num_experts"):
            sys.exit("config has no MoE block (enable_moe_block=%r, num_experts=%r); "
                     "gemma4.c assumes routed experts in every layer."
                     % (t.get("enable_moe_block"), t.get("num_experts")))

    L, D, NE, MI = (t["num_hidden_layers"], t["hidden_size"],
                    t["num_experts"], t["moe_intermediate_size"])
    types = [1 if x == "full_attention" else 0 for x in t["layer_types"]]

    cfg = dict(
        hidden=D, n_layers=L, n_heads=t["num_attention_heads"],
        head_dim=t["head_dim"], global_head_dim=t["global_head_dim"],
        n_kv_heads=t["num_key_value_heads"],
        n_global_kv_heads=t["num_global_key_value_heads"],
        # global layers reuse the k projection as v (attention_k_eq_v). The weight
        # map corroborates: full_attention layers carry NO v_proj tensor at all.
        k_eq_v_global=bool(t.get("attention_k_eq_v", False)),
        n_experts=NE, topk=t["top_k_experts"], moe_inter=MI,
        dense_inter=t["intermediate_size"], vocab=t["vocab_size"],
        eps=t["rms_norm_eps"], sliding_window=t["sliding_window"],
        layer_types=types,
        rope_theta_local=t["rope_parameters"]["sliding_attention"]["rope_theta"],
        rope_theta_global=t["rope_parameters"]["full_attention"]["rope_theta"],
        rope_partial_global=t["rope_parameters"]["full_attention"].get("partial_rotary_factor", 1.0),
        final_logit_softcap=float(t.get("final_logit_softcapping") or 0.0),
        embed_scale=float(D) ** 0.5,
        ctx=ctx,
    )

    # ---------------- dense ----------------
    dn = Dense(os.path.join(dst, "dense.bin"))
    dn.add_q40("embed_tokens", S.get(P + "embed_tokens.weight"))   # tied => lm_head too
    dn.add_f32("norm", S.get(P + "norm.weight"))

    for li in range(L):
        q, o = f"{P}layers.{li}.", f"layers.{li}."
        for nm in ("input_layernorm", "post_attention_layernorm",
                   "pre_feedforward_layernorm", "post_feedforward_layernorm",
                   "pre_feedforward_layernorm_2", "post_feedforward_layernorm_1",
                   "post_feedforward_layernorm_2"):
            dn.add_f32(o + nm, S.get(q + nm + ".weight"))
        dn.add_f32(o + "layer_scalar", np.asarray(S.get(q + "layer_scalar")).reshape(-1))

        dn.add_q40(o + "q_proj", S.get(q + "self_attn.q_proj.weight"))
        dn.add_q40(o + "k_proj", S.get(q + "self_attn.k_proj.weight"))
        if not (types[li] and cfg["k_eq_v_global"]):
            dn.add_q40(o + "v_proj", S.get(q + "self_attn.v_proj.weight"))
        dn.add_q40(o + "o_proj", S.get(q + "self_attn.o_proj.weight"))
        dn.add_f32(o + "q_norm", S.get(q + "self_attn.q_norm.weight"))
        dn.add_f32(o + "k_norm", S.get(q + "self_attn.k_norm.weight"))

        dn.add_q40(o + "mlp_gate", S.get(q + "mlp.gate_proj.weight"))
        dn.add_q40(o + "mlp_up", S.get(q + "mlp.up_proj.weight"))
        dn.add_q40(o + "mlp_down", S.get(q + "mlp.down_proj.weight"))

        # The router stays f32: it is tiny, and a routing error is not a small
        # numeric error -- it picks entirely different experts.
        dn.add_f32(o + "router_proj", S.get(q + "router.proj.weight"))
        dn.add_f32(o + "router_scale", S.get(q + "router.scale"))
        dn.add_f32(o + "router_pes", S.get(q + "router.per_expert_scale"))

    dense_bytes = dn.off
    dn.close(os.path.join(dst, "dense.idx"))

    # ---------------- experts ----------------
    gb, db = q40_bytes(MI, D), q40_bytes(D, MI)
    esz, ALIGN = 2 * gb + db, 4096
    idx, off, worst = {}, 0, 0.0
    with open(os.path.join(dst, "experts.bin"), "wb") as out:
        for li in range(L):
            gu = S.get(f"{P}layers.{li}.experts.gate_up_proj")   # [E, 2*MI, D]
            dw = S.get(f"{P}layers.{li}.experts.down_proj")      # [E, D, MI]
            assert gu.shape == (NE, 2 * MI, D), f"gate_up {gu.shape}"
            assert dw.shape == (NE, D, MI), f"down {dw.shape}"
            for e in range(NE):
                g = np.ascontiguousarray(gu[e, :MI, :])
                u = np.ascontiguousarray(gu[e, MI:, :])
                d = np.ascontiguousarray(dw[e])
                bg, bu, bd = q40_quant_rows(g), q40_quant_rows(u), q40_quant_rows(d)
                assert len(bg) == gb and len(bd) == db
                if verify:
                    worst = max(worst,
                                float(np.abs(g - q40_dequant_rows(bg, MI, D)).max()),
                                float(np.abs(d - q40_dequant_rows(bd, D, MI)).max()))
                pad = (-off) % ALIGN
                if pad:
                    out.write(b"\0" * pad)
                    off += pad
                out.write(bg); out.write(bu); out.write(bd)
                idx[f"{li}.{e}"] = off
                off += esz
            if L > 4:
                print(f"  layer {li+1}/{L}  {off/2**30:.2f} GiB", flush=True)

    json.dump({"expert_bytes": esz, "gate_bytes": gb, "down_bytes": db,
               "align": ALIGN, "offsets": idx},
              open(os.path.join(dst, "experts.idx"), "w"))
    if verify:
        print(f"verify: max |w - dequant(quant(w))| = {worst:.6g}")

    # ---------------- plan ----------------
    p = plan(cfg, dense_bytes, ctx, ram)
    cfg["slots_per_layer"] = p["slots_per_layer"]
    json.dump(cfg, open(os.path.join(dst, "cfg.json"), "w"), indent=1)

    # manifest.txt: the same information, flat, for the C engine. Parsing JSON in C
    # to recover a 3840-row offset table is fragility with no upside.
    FMT = {"f32": 0, "q40": 1}
    with open(os.path.join(dst, "manifest.txt"), "w") as m:
        for k, v in cfg.items():
            if k == "layer_types":
                m.write("cfg layer_types " + " ".join(str(x) for x in v) + "\n")
            else:
                m.write(f"cfg {k} {v}\n")
        m.write(f"esz {esz} {gb} {db}\n")
        m.write(f"ndense {len(dn.idx)}\n")
        for k, v in dn.idx.items():
            sh = v["shape"]
            O = sh[0]
            I = sh[1] if len(sh) > 1 else 1
            m.write(f"dense {k} {v['off']} {v['len']} {FMT[v['fmt']]} {O} {I}\n")
        m.write(f"nexpert {len(idx)}\n")
        for k, v in idx.items():
            li, e = k.split(".")
            m.write(f"expert {li} {e} {v}\n")

    print(f"\ndense resident : {p['dense_bytes']/2**20:8.1f} MiB")
    print(f"kv cache       : {p['kv_bytes']/2**20:8.1f} MiB  (ctx {ctx})")
    print(f"expert         : {esz/2**20:8.2f} MiB each, "
          f"{p['total_expert_bytes']/2**30:.2f} GiB total")
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
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--fixture", action="store_true",
                    help="build a tiny random Gemma-4 container for engine tests")
    a = ap.parse_args()

    if a.fixture:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from gemma4_fixture import make_fixture
        build(None, a.dst, a.ctx, a.ram, a.verify, fixture=make_fixture(a.dst))
    else:
        if not a.src:
            sys.exit("need src")
        build(a.src, a.dst, a.ctx, a.ram, a.verify)


if __name__ == "__main__":
    main()
