#!/usr/bin/env python3
"""
convert_gemma4_dflash.py — Gemma-4 DFlash drafter (DFlashDraftModel) -> container.

DFlash is a block-parallel drafter: it drafts an entire block of tokens at once using
bidirectional attention, conditioned on hidden states extracted from specific layers of
the target backbone.

The model is Qwen3-based (not Gemma4), with 5 decoder layers, each having its own
q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, and a plain MLP (gate/up/down).
Attention is special: K/V come from BOTH the target hidden context AND the draft
block's own tokens (concatenated), and the mask is bidirectional within the block.

OUTPUT: dflash.bin / dflash.manifest.txt, same conventions as the main container
(q4_0 matrices, f32 norms).
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_gemma4 import Shards, Dense, q40_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="the DFlash checkpoint dir (with model.safetensors)")
    ap.add_argument("dst", help="the TARGET container dir (dflash.* is written into it)")
    a = ap.parse_args()

    raw = json.load(open(os.path.join(a.src, "config.json")))
    arch = (raw.get("architectures") or [""])[0]
    if "DFlash" not in arch and raw.get("model_type") != "qwen3":
        sys.exit(f"not a DFlash checkpoint (got {arch or raw.get('model_type')})")

    dc = raw.get("dflash_config", {})
    L = raw["num_hidden_layers"]
    D = raw["hidden_size"]
    types = [1 if x == "full_attention" else 0 for x in raw["layer_types"]]

    S = Shards(a.src)
    os.makedirs(a.dst, exist_ok=True)
    dn = Dense(os.path.join(a.dst, "dflash.bin"))

    # fc: projects concatenated target hidden states -> D
    # [D, num_target_layers * D] = [2816, 6*2816] = [2816, 16896]
    dn.add_q40("fc", S.get("fc.weight"))
    dn.add_f32("hidden_norm", S.get("hidden_norm.weight"))
    dn.add_f32("norm", S.get("norm.weight"))

    for li in range(L):
        q, o = f"layers.{li}.", f"layers.{li}."
        for nm in ("input_layernorm", "post_attention_layernorm"):
            dn.add_f32(o + nm, S.get(q + nm + ".weight"))

        # Qwen3-style attention: q_proj, k_proj, v_proj, o_proj, q_norm, k_norm
        dn.add_q40(o + "q_proj", S.get(q + "self_attn.q_proj.weight"))
        dn.add_q40(o + "k_proj", S.get(q + "self_attn.k_proj.weight"))
        dn.add_q40(o + "v_proj", S.get(q + "self_attn.v_proj.weight"))
        dn.add_q40(o + "o_proj", S.get(q + "self_attn.o_proj.weight"))
        dn.add_f32(o + "q_norm", S.get(q + "self_attn.q_norm.weight"))
        dn.add_f32(o + "k_norm", S.get(q + "self_attn.k_norm.weight"))

        # Plain MLP (no MoE)
        dn.add_q40(o + "mlp_gate", S.get(q + "mlp.gate_proj.weight"))
        dn.add_q40(o + "mlp_up", S.get(q + "mlp.up_proj.weight"))
        dn.add_q40(o + "mlp_down", S.get(q + "mlp.down_proj.weight"))

    dn.close(os.path.join(a.dst, "dflash.idx"))

    cfg = dict(
        hidden=D,
        n_layers=L,
        n_heads=raw["num_attention_heads"],
        head_dim=raw["head_dim"],
        n_kv_heads=raw["num_key_value_heads"],
        intermediate_size=raw["intermediate_size"],
        vocab_size=raw["vocab_size"],
        eps=raw["rms_norm_eps"],
        layer_types=types,
        sliding_window=raw.get("sliding_window", 2048),
        rope_theta=raw.get("rope_theta", 1000000),
        block_size=raw.get("block_size", 16),
        mask_token_id=dc.get("mask_token_id", 4),
        target_layer_ids=dc.get("target_layer_ids", [1, 6, 11, 17, 22, 27]),
        num_target_layers=raw.get("num_target_layers", 30),
        final_logit_softcap=float(raw.get("final_logit_softcapping") or 0.0),
    )
    json.dump(cfg, open(os.path.join(a.dst, "dflash.cfg.json"), "w"), indent=1)

    FMT = {"f32": 0, "q40": 1}
    with open(os.path.join(a.dst, "dflash.manifest.txt"), "w") as m:
        for k, v in cfg.items():
            if k == "layer_types":
                m.write("cfg layer_types " + " ".join(str(x) for x in v) + "\n")
            elif k == "target_layer_ids":
                m.write("cfg target_layer_ids " + " ".join(str(x) for x in v) + "\n")
            else:
                m.write(f"cfg {k} {v}\n")
        m.write(f"ndense {len(dn.idx)}\n")
        for k, v in dn.idx.items():
            sh = v["shape"]
            O = sh[0]
            I = sh[1] if len(sh) > 1 else 1
            m.write(f"dense {k} {v['off']} {v['len']} {FMT[v['fmt']]} {O} {I}\n")

    print(f"dflash: {L} layers, hidden {D}, {dn.off/2**20:.1f} MiB")
    print("layer types:", " ".join("full" if x else "slide" for x in types))
    print(f"target layers: {cfg['target_layer_ids']}")
    print(f"block size: {cfg['block_size']}")
    print("\nRun with:  ./gemma4 <dir> --dflash --ndraft 16")


if __name__ == "__main__":
    main()
