#!/usr/bin/env python3
"""
gemma4_check.py — isolate ENGINE bugs from QUANTISATION error.

Comparing gemma4.c straight to the HF reference conflates two things: the error
q4_0 introduces, and any bug in the C. This reads the container back, dequantises
it, runs the numpy oracle on those exact weights, and dumps the logits. gemma4.c
must then match THIS to ~1e-4 -- anything worse is a bug, not quantisation.

Also reports the true quantisation cost (oracle-on-dequantised vs HF reference).
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_gemma4 import q40_dequant_rows
from gemma4_oracle import Gemma4Oracle, rmsnorm


def load(dst):
    cfg = json.load(open(os.path.join(dst, "cfg.json")))
    idx = json.load(open(os.path.join(dst, "dense.idx")))
    blob = np.fromfile(os.path.join(dst, "dense.bin"), dtype=np.uint8)
    ex = json.load(open(os.path.join(dst, "experts.idx")))
    eb = np.fromfile(os.path.join(dst, "experts.bin"), dtype=np.uint8)

    W = {}
    for name, e in idx.items():
        raw = blob[e["off"]:e["off"] + e["len"]].tobytes()
        if e["fmt"] == "q40":
            O, I = e["shape"]
            W[name] = q40_dequant_rows(raw, O, I)
        else:
            W[name] = np.frombuffer(raw, np.float32).reshape(e["shape"])

    D, MI, NE = cfg["hidden"], cfg["moe_inter"], cfg["n_experts"]
    gb = ex["gate_bytes"]
    for li in range(cfg["n_layers"]):
        gu = np.zeros((NE, 2 * MI, D), np.float32)
        dn = np.zeros((NE, D, MI), np.float32)
        for e in range(NE):
            o = ex["offsets"][f"{li}.{e}"]
            gu[e, :MI] = q40_dequant_rows(eb[o:o + gb].tobytes(), MI, D)
            gu[e, MI:] = q40_dequant_rows(eb[o + gb:o + 2 * gb].tobytes(), MI, D)
            dn[e] = q40_dequant_rows(eb[o + 2 * gb:o + 2 * gb + ex["down_bytes"]].tobytes(), D, MI)
        W[f"layers.{li}.experts.gate_up_proj"] = gu
        W[f"layers.{li}.experts.down_proj"] = dn
    return cfg, W


def rename(W):
    """container names -> the oracle's HF-ish names"""
    out = {}
    for k, v in W.items():
        k2 = (k.replace("q_proj", "self_attn.q_proj.weight")
               .replace("k_proj", "self_attn.k_proj.weight")
               .replace("v_proj", "self_attn.v_proj.weight")
               .replace("o_proj", "self_attn.o_proj.weight")
               .replace("q_norm", "self_attn.q_norm.weight")
               .replace("k_norm", "self_attn.k_norm.weight")
               .replace("mlp_gate", "mlp.gate_proj.weight")
               .replace("mlp_up", "mlp.up_proj.weight")
               .replace("mlp_down", "mlp.down_proj.weight")
               .replace("router_proj", "router.proj.weight")
               .replace("router_scale", "router.scale")
               .replace("router_pes", "router.per_expert_scale"))
        if k2 == k and not k.endswith(("gate_up_proj", "down_proj")):
            k2 = k + ".weight" if not k.endswith("layer_scalar") else k
        if k == "embed_tokens":
            k2 = "embed_tokens.weight"
        if k == "norm":
            k2 = "norm.weight"
        out[k2] = v
    return out


def main(dst):
    cfg, Wc = load(dst)
    W = rename(Wc)
    ref = json.load(open(os.path.join(dst, "ref.json")))
    ids = np.array(ref["prompt"])
    hf = np.load(os.path.join(dst, "ref_logits.npy"))

    ora = Gemma4Oracle(W, cfg)
    logits = ora.forward(ids)
    logits.astype(np.float32).tofile(os.path.join(dst, "deq_logits.f32"))
    json.dump({"prompt": ref["prompt"]}, open(os.path.join(dst, "deq.json"), "w"))

    d = np.abs(logits - hf).max() / np.abs(hf).max()
    am = (logits.argmax(-1) == hf.argmax(-1)).sum()
    print(f"quantisation cost (oracle-on-dequantised vs HF fp32):")
    print(f"  max rel logit diff {d:.3e}   argmax agree {am}/{len(ids)}")
    print(f"wrote deq_logits.f32 -- gemma4.c --check must match THIS to ~1e-4")


if __name__ == "__main__":
    main(sys.argv[1])
