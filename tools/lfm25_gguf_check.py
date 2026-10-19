#!/usr/bin/env python3
"""
lfm25_gguf_check.py — the two new checkpoint layouts convert_lfm25.py reads, checked
against the one it already did, on the tiny random fixture.

1. GGUF (--gguf). The fixture is written as a llama.cpp lfm2moe GGUF with mostly
   Q4_0 blocks, as LiquidAI's QAD-Q4_0 release is, plus a Q8_0 expert layer and an
   F16 and a BF16 matrix to exercise the requantising path. Every Q4_0/Q8_0 tensor
   must land in the container byte for byte -- the whole point of the path -- and the
   rest must dequantise to what the GGUF dequantises to, within one q8_0 rounding.
   Tensor names follow llama.cpp's src/llama-arch.cpp for LLM_ARCH_LFM2MOE.

2. Fused-expert safetensors, as transformers >= 5 saves an lfm2_moe checkpoint
   (gate_up_proj / down_proj per layer instead of one tensor per expert), in bf16,
   through the real Shards reader. Must give the same container bytes as the per-
   expert fixture.

It also writes the GGUF-derived container (with ref.json) to DST, so the oracle and
the engine can be run on it:

    python3 tools/lfm25_gguf_check.py /tmp/lfmgguf
    python3 tools/lfm25_oracle.py /tmp/lfmgguf
    ./lfm25-exact /tmp/lfmgguf --check

Needs numpy and gguf (pip install gguf).
"""
import json, os, shutil, struct, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_lfm25 as C

try:
    import gguf
    from gguf import GGMLQuantizationType as Q, quants
except ImportError:
    sys.exit("needs the gguf package (pip install gguf)")


def bf16_round(a):
    """round-to-nearest-even to bf16 and back, as a real checkpoint's values are"""
    u = np.ascontiguousarray(a, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32).reshape(a.shape)


def hf_config(cfg):
    return {
        "model_type": "lfm2_moe", "hidden_size": cfg["hidden"],
        "num_hidden_layers": cfg["n_layers"], "num_attention_heads": cfg["n_heads"],
        "num_key_value_heads": cfg["n_kv_heads"], "num_experts": cfg["n_experts"],
        "num_experts_per_tok": cfg["topk"], "moe_intermediate_size": cfg["moe_inter"],
        "intermediate_size": cfg["dense_inter"], "num_dense_layers": cfg["n_dense_layers"],
        "conv_L_cache": cfg["conv_L"], "vocab_size": cfg["vocab"], "norm_eps": cfg["eps"],
        "rope_parameters": {"rope_theta": cfg["rope_theta"], "rope_type": "default"},
        "routed_scaling_factor": cfg["routed_scaling"], "norm_topk_prob": True,
        "use_expert_bias": True, "tie_embedding": True, "conv_bias": False,
        "bos_token_id": 1, "eos_token_id": 7, "pad_token_id": 0,
        "layer_types": ["full_attention" if t else "conv" for t in cfg["layer_types"]]}


def write_gguf(path, cfg, T):
    """T in HF names -> lfm2moe GGUF. Returns {gguf name: type} for the check."""
    D, NE, ND, L = cfg["hidden"], cfg["n_experts"], cfg["n_dense_layers"], cfg["n_layers"]
    w = gguf.GGUFWriter(path, "lfm2moe")
    w.add_block_count(L)
    w.add_embedding_length(D)
    w.add_expert_count(NE)
    w.add_expert_used_count(cfg["topk"])
    w.add_expert_feed_forward_length(cfg["moe_inter"])
    types = {}
    q8_layer = ND                         # first MoE layer's experts in Q8_0
    odd = {"blk.0.ffn_down.weight": Q.F16,                      # requant paths
           f"blk.{ND}.ffn_gate_inp.weight": Q.BF16}             # policy-f32, from bf16

    def put(name, a, t):
        a = np.ascontiguousarray(a, np.float32)
        types[name] = t
        if t == Q.F32:
            w.add_tensor(name, a)
        elif t == Q.F16:
            w.add_tensor(name, a.astype(np.float16))
        else:
            w.add_tensor(name, quants.quantize(a, t), raw_dtype=t)

    def mat(name, a):
        put(name, a, odd.get(name, Q.Q4_0))

    put("token_embd.weight", T["model.embed_tokens.weight"], Q.Q4_0)
    put("token_embd_norm.weight", T["model.embedding_norm.weight"], Q.F32)
    for li in range(L):
        p, b = f"model.layers.{li}.", f"blk.{li}."
        put(b + "attn_norm.weight", T[p + "operator_norm.weight"], Q.F32)
        put(b + "ffn_norm.weight", T[p + "ffn_norm.weight"], Q.F32)
        if cfg["layer_types"][li]:
            if li == L - 2:                   # one layer with llama.cpp's fused qkv
                mat(b + "attn_qkv.weight", np.concatenate(
                    [T[p + f"self_attn.{x}_proj.weight"] for x in "qkv"]))
            else:
                for x in "qkv":
                    mat(b + f"attn_{x}.weight", T[p + f"self_attn.{x}_proj.weight"])
            mat(b + "attn_output.weight", T[p + "self_attn.out_proj.weight"])
            put(b + "attn_q_norm.weight", T[p + "self_attn.q_layernorm.weight"], Q.F32)
            put(b + "attn_k_norm.weight", T[p + "self_attn.k_layernorm.weight"], Q.F32)
        else:
            mat(b + "shortconv.in_proj.weight", T[p + "conv.in_proj.weight"])
            put(b + "shortconv.conv.weight", T[p + "conv.conv.weight"][:, 0, :], Q.F32)
            mat(b + "shortconv.out_proj.weight", T[p + "conv.out_proj.weight"])
        if li < ND:
            mat(b + "ffn_gate.weight", T[p + "feed_forward.w1.weight"])
            mat(b + "ffn_up.weight", T[p + "feed_forward.w3.weight"])
            mat(b + "ffn_down.weight", T[p + "feed_forward.w2.weight"])
        else:
            put(b + "ffn_gate_inp.weight", T[p + "feed_forward.gate.weight"],
                odd.get(b + "ffn_gate_inp.weight", Q.F32))
            put(b + "exp_probs_b.bias", T[p + "feed_forward.expert_bias"], Q.F32)
            et = Q.Q8_0 if li == q8_layer else Q.Q4_0
            for hw, gw in (("w1", "ffn_gate_exps"), ("w3", "ffn_up_exps"),
                           ("w2", "ffn_down_exps")):
                put(b + gw + ".weight", np.stack(
                    [T[p + f"feed_forward.experts.{e}.{hw}.weight"] for e in range(NE)]), et)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return types


def write_fused_safetensors(d, cfg, T):
    """T -> one bf16 safetensors file with the experts fused per layer"""
    out = {}
    for n, a in T.items():
        if ".feed_forward.experts." in n:
            continue
        out[n] = a
    for li in range(cfg["n_dense_layers"], cfg["n_layers"]):
        p = f"model.layers.{li}.feed_forward.experts."
        E = range(cfg["n_experts"])
        out[p + "gate_up_proj"] = np.stack([np.concatenate(
            [T[p + f"{e}.w1.weight"], T[p + f"{e}.w3.weight"]]) for e in E])
        out[p + "down_proj"] = np.stack([T[p + f"{e}.w2.weight"] for e in E])
    hdr, blobs, off = {}, [], 0
    for n, a in out.items():
        b = (np.ascontiguousarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
        hdr[n] = {"dtype": "BF16", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(hdr).encode()
    h += b" " * ((-len(h)) % 8)
    with open(os.path.join(d, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def dense_index(d):
    idx = {}
    for line in open(os.path.join(d, "manifest.txt")):
        t = line.split()
        if t and t[0] == "dense":
            idx[t[1]] = (int(t[2]), int(t[3]), int(t[4]), int(t[5]), int(t[6]))
    return idx


def expert_index(d):
    esz, off = {}, {}
    for line in open(os.path.join(d, "manifest.txt")):
        t = line.split()
        if t and t[0] == "eszl":
            esz[int(t[1])] = (int(t[2]), int(t[3]), int(t[4]), int(t[5]))
        elif t and t[0] == "expert":
            off[(int(t[1]), int(t[2]))] = int(t[3])
    return esz, off


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    dst = sys.argv[1]
    tmp = tempfile.mkdtemp(prefix="lfmgguf")
    fail = 0

    def ok(cond, what):
        nonlocal fail
        print(f"  {'ok  ' if cond else 'FAIL'} {what}")
        fail |= not cond

    ref = os.path.join(tmp, "ref")
    cfg, T = C.make_fixture(ref)
    T = {n: bf16_round(a) for n, a in T.items()}
    cfg = dict(cfg)

    # the reference: the in-memory fixture through the existing path
    C.build(None, ref, 64, 8, 1, False, False, want_flash=False, fixture=True,
            fixture_cfg=(dict(cfg), T))

    src = os.path.join(tmp, "src")
    os.makedirs(src)
    json.dump(hf_config(cfg), open(os.path.join(src, "config.json"), "w"))

    # 2. fused safetensors
    print("fused-expert bf16 safetensors:")
    write_fused_safetensors(src, cfg, T)
    fused = os.path.join(tmp, "fused")
    C.build(src, fused, 64, 8, 1, False, False, want_flash=False)
    for f in ("dense.bin", "experts.bin"):
        ok(open(os.path.join(ref, f), "rb").read() == open(os.path.join(fused, f), "rb").read(),
           f"{f} identical to the per-expert fixture")
    os.remove(os.path.join(src, "model.safetensors"))

    # 1. GGUF
    print("GGUF:")
    gpath = os.path.join(tmp, "fix.gguf")
    types = write_gguf(gpath, cfg, T)
    C.build(src, dst, 64, 8, 1, False, False, want_flash=True, gguf=gpath)
    shutil.copy(os.path.join(ref, "ref.json"), os.path.join(dst, "ref.json"))

    G = C.GGUFSource(gpath)
    gt = {t.name: t for t in G.r.tensors}
    dense = open(os.path.join(dst, "dense.bin"), "rb").read()
    idx = dense_index(dst)
    names = {"embed_tokens": "model.embed_tokens.weight"}
    for li in range(cfg["n_layers"]):
        p, o = f"model.layers.{li}.", f"layers.{li}."
        for on, hn in (("q_proj", "self_attn.q_proj.weight"), ("k_proj", "self_attn.k_proj.weight"),
                       ("v_proj", "self_attn.v_proj.weight"), ("o_proj", "self_attn.out_proj.weight"),
                       ("conv_in", "conv.in_proj.weight"), ("conv_out", "conv.out_proj.weight"),
                       ("mlp_gate", "feed_forward.w1.weight"), ("mlp_up", "feed_forward.w3.weight"),
                       ("mlp_down", "feed_forward.w2.weight"), ("router", "feed_forward.gate.weight")):
            if o + on in idx:
                names[o + on] = p + hn
    copied, requant, worst = [], 0, 0.0
    for on, hn in names.items():
        off, ln, fmt, O, I = idx[on]
        mine = dense[off:off + ln]
        if G.fmt_of(hn) in (C.FMT_Q40, C.FMT_Q80):
            copied.append(fmt == G.fmt_of(hn) and mine == G.raw(hn))
            continue
        want = G.get(hn).reshape(O, I)
        got = C.dequant_rows(fmt, mine, O, I)
        worst = max(worst, float(np.abs(got - want).max() / (np.abs(want).max() + 1e-12)))
        requant += 1
    ok(copied and all(copied), f"{len(copied)} Q4_0 dense matrices copied byte for byte "
       f"(attention via fused qkv too)")
    ok(requant >= 3 and worst < 1e-2,
       f"{requant} re-encoded from F16/BF16/F32, worst rel err {worst:.2e}")

    esz, eoff = expert_index(dst)
    ex = open(os.path.join(dst, "experts.bin"), "rb").read()
    bad = 0
    for (li, e), off in eoff.items():
        sz, gb, db, fmt = esz[li]
        want = b"".join(np.ascontiguousarray(gt[f"blk.{li}.{n}.weight"].data[e]).tobytes()
                        for n in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"))
        gfmt = C.FMT_Q80 if gt[f"blk.{li}.ffn_gate_exps.weight"].tensor_type == Q.Q8_0 else C.FMT_Q40
        bad += ex[off:off + sz] != want or fmt != gfmt
    ok(bad == 0, f"{len(eoff)} experts copied byte for byte, formats kept "
       f"(the --expert-edge q8_0 policy does not override a native q4_0 layer)")

    # and the GGUF's own q4_0 is ours: quantising the same bf16 values agrees
    e = T["model.embed_tokens.weight"]
    ok(quants.quantize(e, Q.Q4_0).tobytes() == C.q40_quant_rows(e),
       "ggml's Q4_0 and q40_quant_rows agree on bf16 inputs")
    m = [l for l in open(os.path.join(dst, "manifest.txt")) if l.startswith("cfg model_id")]
    ok(m == ["cfg model_id src\n"], "model_id falls back to the src directory name")

    shutil.rmtree(tmp)
    print("PASS" if not fail else "FAILED")
    return fail


if __name__ == "__main__":
    sys.exit(main())
