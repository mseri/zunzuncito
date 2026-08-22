#!/usr/bin/env python3
"""
convert_lfm25_dspark.py — LFM2.5-DSpark drafter (Lfm2DSparkDraftModel) -> container.

DSpark is DFlash's block-parallel drafter with two heads bolted on. The backbone is
the same idea as z-lab's DFlash and converts the same way: 5 Qwen3-style decoder
layers whose K/V come partly from the target's own hidden states (projected through
`fc` + `hidden_norm`) and partly from the draft block's tokens, with bidirectional
attention inside the block. The target here is LFM2.5-8B-A1B, so this writes into an
lfm25 container built by convert_lfm25.py and borrows that container's embedding
table for both the block's input embeddings and the draft logits -- the drafter ships
neither an embed_tokens nor an lm_head.

What DSpark adds over DFlash:

  markov head    A rank-256 factorisation of a bigram table: two [vocab, rank]
                 matrices, w1 read as an embedding of the PREVIOUS token and w2 as a
                 linear map back to the vocabulary, so the block-parallel drafter
                 recovers the immediate left-context dependency that masking took
                 away. logits_j += w2 @ w1[token at j-1].

  confidence     Linear(hidden + rank -> 1) over concat(final hidden, w1[prev]).
                 Squashed with a sigmoid it estimates whether this position's draft
                 will survive verification, which lfm25 --dconf uses to stop
                 proposing where the drafter itself does not believe the block.

Formats follow the sibling converters: q4_0 matrices, f32 norms. markov_w1 is q8_0
because it is only ever row-indexed (33 MiB resident, no streaming cost), while
markov_w2 is a full [vocab, rank] matvec per drafted position and so pays q4_0 rates.

    python3 tools/convert_lfm25_dspark.py ./lfm-dspark ./lfm-ct
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_lfm25 import Shards, Dense, FMT_F32, FMT_Q40, FMT_Q80


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="the DSpark checkpoint dir (with model.safetensors)")
    ap.add_argument("dst", help="the TARGET lfm25 container dir (dspark.* goes here)")
    ap.add_argument("--draft-bits", type=int, choices=(4, 8), default=4,
                    help="block format for the drafter's matrices (default 4). The "
                         "drafter is re-verified by the target every step, so its "
                         "error costs acceptance rather than output quality; 8 "
                         "roughly doubles its resident footprint.")
    a = ap.parse_args()

    raw = json.load(open(os.path.join(a.src, "config.json")))
    arch = (raw.get("architectures") or [""])[0]
    if "DSpark" not in arch:
        sys.exit(f"not a DSpark checkpoint (got {arch or raw.get('model_type')})")

    dc = raw.get("dflash_config", {})
    L = raw["num_hidden_layers"]
    D = raw["hidden_size"]
    WFMT = FMT_Q40 if a.draft_bits == 4 else FMT_Q80

    # The container's own cfg.json fixes what the drafter has to agree with: it reads
    # the target's hidden states and writes into the target's vocabulary, so a
    # mismatch on either is a wiring error rather than a tolerable difference.
    tcfg = json.load(open(os.path.join(a.dst, "cfg.json")))
    if D != tcfg["hidden"]:
        sys.exit(f"hidden {D} != target container's {tcfg['hidden']}")
    if raw["vocab_size"] != tcfg["vocab"]:
        sys.exit(f"vocab {raw['vocab_size']} != target container's {tcfg['vocab']}")
    ntl = dc.get("num_target_layers", raw.get("num_target_layers", tcfg["n_layers"]))
    if ntl != tcfg["n_layers"]:
        sys.exit(f"num_target_layers {ntl} != target container's {tcfg['n_layers']}")
    tids = dc.get("target_layer_ids", [])
    if not tids or max(tids) >= tcfg["n_layers"]:
        sys.exit(f"target_layer_ids {tids} out of range for {tcfg['n_layers']} layers")

    S = Shards(a.src)
    dn = Dense(os.path.join(a.dst, "dspark.bin"))

    # fc projects the concatenated target hidden states down to D: [D, len(tids) * D]
    fc = S.get("fc.weight")
    if fc.shape != (D, len(tids) * D):
        sys.exit(f"fc.weight is {fc.shape}, expected {(D, len(tids) * D)}")
    dn.add("fc", fc, WFMT)
    dn.add("hidden_norm", S.get("hidden_norm.weight"), FMT_F32)
    dn.add("norm", S.get("norm.weight"), FMT_F32)

    for li in range(L):
        p = f"layers.{li}."
        for nm in ("input_layernorm", "post_attention_layernorm"):
            dn.add(p + nm, S.get(p + nm + ".weight"), FMT_F32)
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            dn.add(p + nm, S.get(p + "self_attn." + nm + ".weight"), WFMT)
        for nm in ("q_norm", "k_norm"):
            dn.add(p + nm, S.get(p + "self_attn." + nm + ".weight"), FMT_F32)
        for nm in ("gate_proj", "up_proj", "down_proj"):
            dn.add(p + "mlp_" + nm[:-5], S.get(p + "mlp." + nm + ".weight"), WFMT)

    rank = raw.get("markov_rank", 0)
    has_markov = rank > 0 and S.has("markov_head.markov_w1.weight")
    if has_markov:
        if raw.get("markov_head_type", "vanilla") != "vanilla":
            sys.exit(f"unsupported markov_head_type {raw['markov_head_type']}")
        dn.add("markov_w1", S.get("markov_head.markov_w1.weight"), FMT_Q80)
        dn.add("markov_w2", S.get("markov_head.markov_w2.weight"), WFMT)

    has_conf = bool(raw.get("enable_confidence_head")) and \
        S.has("confidence_head.proj.weight")
    if has_conf:
        cw = S.get("confidence_head.proj.weight")
        if cw.shape != (1, D + rank):
            sys.exit(f"confidence proj is {cw.shape}, expected {(1, D + rank)}")
        # f32: one row, and it feeds a sigmoid whose output is compared against a
        # user-set threshold, where a quantisation wobble would move the cut.
        dn.add("conf_w", cw, FMT_F32)
        dn.add("conf_b", S.get("confidence_head.proj.bias"), FMT_F32)

    dn.close()

    cfg = dict(
        hidden=D,
        n_layers=L,
        n_heads=raw["num_attention_heads"],
        head_dim=raw["head_dim"],
        n_kv_heads=raw["num_key_value_heads"],
        intermediate_size=raw["intermediate_size"],
        vocab_size=raw["vocab_size"],
        eps=raw["rms_norm_eps"],
        rope_theta=raw.get("rope_theta", 1000000.0),
        # sglang's flag, and it is not decoration: false means the pairs rotated are
        # (x[2i], x[2i+1]) rather than (x[i], x[i + half]). Getting it wrong does not
        # crash, it just drafts badly.
        rope_neox=int(bool(raw.get("rope_is_neox_style", True))),
        block_size=raw.get("block_size", 9),
        mask_token_id=dc.get("mask_token_id", 0),
        target_layer_ids=tids,
        num_target_layers=ntl,
        markov_rank=rank if has_markov else 0,
        has_confidence=int(has_conf),
    )
    json.dump(cfg, open(os.path.join(a.dst, "dspark.cfg.json"), "w"), indent=1)

    with open(os.path.join(a.dst, "dspark.manifest.txt"), "w") as m:
        for k, v in cfg.items():
            if k == "target_layer_ids":
                m.write("cfg target_layer_ids " + " ".join(str(x) for x in v) + "\n")
            else:
                m.write(f"cfg {k} {v}\n")
        m.write(f"ndense {len(dn.idx)}\n")
        for k, v in dn.idx.items():
            O, I = v["shape"]
            m.write(f"dense {k} {v['off']} {v['len']} {v['fmt']} {O} {I}\n")

    print(f"dspark: {L} layers, hidden {D}, block_size {cfg['block_size']}, "
          f"{dn.off / 2**20:.1f} MiB")
    print(f"target layers: {tids} of {ntl}")
    print(f"markov rank: {cfg['markov_rank'] or 'none'}; "
          f"confidence head: {'yes' if has_conf else 'no'}; "
          f"rope: {'neox' if cfg['rope_neox'] else 'interleaved'}")
    print(f"\nRun with:  ./lfm25 {a.dst} --dspark")


if __name__ == "__main__":
    main()
