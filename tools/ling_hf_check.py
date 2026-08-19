#!/usr/bin/env python3
"""
ling_hf_check.py — diff the engine against HF transformers on the REAL checkpoint.

tools/ling_oracle.py already proves the engine reproduces a numpy forward pass over the
container weights to float precision. What it cannot catch is a systematic misreading of
the architecture, because the oracle is a second transcription of the SAME reading. Two
places in this model are easy to get plausibly wrong, and both would be wrong in the
oracle and the engine together:

  * the KDA decay. fla's kernel has two branches, and the model takes the bounded one:
        g = lower_bound * sigmoid(exp(A_log) * (f_proj(x) + dt_bias))
    not the -exp(A_log) * softplus(...) it uses when no lower bound is supplied. The
    wrong branch gives a model that runs, generates text, and decays wrong.
  * FusedRMSNormGated's order. The gate multiplies AFTER the normalisation, not before.

Also the interleaved RoPE convention, and the grouped top-k's "the bias selects but
never weights" rule. This script is the only thing in the repo that would notice.

The comparison is greedy generation, not logits. The two run different arithmetic --
HF in bf16, this in q8_0/q4_0 weights with int8 activations -- so their logits differ by
percent-level amounts that say nothing. What must agree is the token stream. A
divergence in the first few tokens is an architecture bug; one deep into a long
generation, at a position where the top two logits are near-tied, is not.

Needs torch, transformers and fla-core, and the 8 GB checkpoint. Deliberately NOT part
of `make check-ling`: run it once by hand after any change to the architecture.

    python3 tools/ling_hf_check.py ./ling-tiny-fp8 ./ling-ct
"""
import argparse, subprocess, sys

PROMPTS = [
    "What is the capital of France?",
    "Write the numbers one to five, comma separated.",
    "Explain in one sentence why the sky is blue.",
    "def fibonacci(n):",
    "Le chat s'est assis sur le tapis. Traduis en anglais.",
]


def hf_load(model_dir):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cpu")
    model.eval()
    return model, tok


def hf_generate(model, tok, prompts, n, think):
    import torch
    out = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=think)
        ids = tok(text, return_tensors="pt").input_ids
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=n, do_sample=False,
                               temperature=None, top_p=None, top_k=None)
        out.append(tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True))
    return out


def engine_generate(binary, container, prompts, n, extra):
    out = []
    for p in prompts:
        r = subprocess.run([binary, container, "--temp", "0", "--max_tokens", str(n),
                            *extra, "--", p], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"{binary} failed: {r.stderr}")
        txt = r.stdout
        cut = txt.find("\n\nprefill ")          # strip the CLI's timing block
        out.append(txt[:cut] if cut >= 0 else txt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="the HF checkpoint (ling-tiny-fp8)")
    ap.add_argument("container", help="the converted ling container")
    ap.add_argument("--binary", default="./ling")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--nothink", action="store_true",
                    help="compare with thinking off (shorter, easier to eyeball)")
    a = ap.parse_args()

    print("loading the HF model (this needs ~16 GB of RAM in bf16) ...", flush=True)
    model, tok = hf_load(a.model_dir)

    think = not a.nothink
    extra = [] if think else ["--nothink"]
    print(f"\ngenerating {a.tokens} greedy tokens for {len(PROMPTS)} prompts "
          f"(thinking {'on' if think else 'off'}) ...", flush=True)
    mine = engine_generate(a.binary, a.container, PROMPTS, a.tokens, extra)
    theirs = hf_generate(model, tok, PROMPTS, a.tokens, think)

    bad = 0
    for p, m, t in zip(PROMPTS, mine, theirs):
        k = min(len(m), len(t))                 # the two stop conditions differ
        if m[:k].strip() == t[:k].strip():
            print(f"  ok        {p!r}")
            continue
        bad += 1
        common = 0
        while common < k and m[common] == t[common]:
            common += 1
        print(f"  DIVERGES  {p!r}\n    after {common} chars\n"
              f"    hf   {t[max(0,common-40):common+80]!r}\n"
              f"    ling {m[max(0,common-40):common+80]!r}")

    print(f"\n{len(PROMPTS) - bad}/{len(PROMPTS)} prompts identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
