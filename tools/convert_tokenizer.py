#!/usr/bin/env python3
"""
convert_tokenizer.py — HF tokenizer.json -> flat binary for gemma4.c.

Gemma-4's tokenizer is a SentencePiece-style BPE: 262K vocab, byte_fallback, and a
Replace normaliser mapping ' ' -> U+2581. Nothing about it is exotic; what makes a
naive C port slow is doing STRING lookups in the merge loop. So we resolve every
merge rule to token IDs here, offline:

    merge rule  (id_a, id_b) -> id_ab,  rank = position in the merges list

and the engine's BPE inner loop becomes an integer hash probe. The 262K vocab and
~1M merge rules are then just two flat arrays.

FORMAT (little-endian):
  magic "G4TK"           u32
  n_vocab               u32
  n_merges              u32
  n_special             u32
  n_replace             u32
  unk_id                i32
  byte_fallback         u32
  byte_tok[256]         i32     id of "<0xNN>", or -1 (byte_fallback targets)
  vocab: n_vocab x { len u16, bytes }
  merges: n_merges x { a i32, b i32, ab i32 }   (rank = index)
  special: n_special x { id i32, len u16, bytes }
  replace: n_replace x { flen u16, from, tlen u16, to }
"""
import argparse, json, struct, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer_json")
    ap.add_argument("out")
    a = ap.parse_args()

    d = json.load(open(a.tokenizer_json, encoding="utf-8"))
    mdl = d["model"]
    if mdl["type"] != "BPE":
        sys.exit(f"expected a BPE model, got {mdl['type']}")

    vocab = mdl["vocab"]                       # token string -> id
    n = max(vocab.values()) + 1
    toks = [None] * n
    for s, i in vocab.items():
        toks[i] = s
    for aa in d.get("added_tokens", []):       # added tokens may sit outside `vocab`
        if aa["id"] < n:
            toks[aa["id"]] = aa["content"]
        else:
            toks += [None] * (aa["id"] - n + 1)
            n = aa["id"] + 1
            toks[aa["id"]] = aa["content"]
    for i in range(n):
        if toks[i] is None:
            toks[i] = ""                       # holes: unreachable, but keep ids dense

    # merges: strings -> ids. A merge whose product is not in the vocab is dead;
    # dropping it is correct and keeps the engine's table clean.
    merges = []
    dropped = 0
    for m in mdl["merges"]:
        x, y = (m if isinstance(m, list) else m.split(" ", 1))
        ab = vocab.get(x + y)
        ia, ib = vocab.get(x), vocab.get(y)
        if ab is None or ia is None or ib is None:
            dropped += 1
            continue
        merges.append((ia, ib, ab))

    byte_fallback = bool(mdl.get("byte_fallback", False))
    btok = [-1] * 256
    if byte_fallback:
        for i in range(256):
            btok[i] = vocab.get(f"<0x{i:02X}>", -1)

    unk = vocab.get(mdl.get("unk_token") or "", -1)

    special = [(aa["id"], aa["content"]) for aa in d.get("added_tokens", [])]

    # normaliser: we support the Replace rules, which is all Gemma uses (' ' -> ▁).
    rep = []

    def walk(nm):
        if not nm:
            return
        t = nm.get("type")
        if t == "Sequence":
            for x in nm["normalizers"]:
                walk(x)
        elif t == "Replace":
            p = nm["pattern"]
            if "String" not in p:
                sys.exit("Replace with a Regex pattern is not supported")
            rep.append((p["String"], nm["content"]))
        elif t in ("Prepend",):
            rep.append(("\x00PREPEND", nm["prepend"]))
        elif t in ("NFC", "NFD", "NFKC", "NFKD", "Nmt", "StripAccents", "Lowercase"):
            sys.exit(f"normaliser {t} not supported; add it to the C side first")

    walk(d.get("normalizer"))

    # the pre_tokenizer must not split: SentencePiece-BPE runs over the whole string.
    pt = d.get("pre_tokenizer")
    if pt and pt.get("type") not in (None, "Sequence"):
        print(f"note: pre_tokenizer {pt.get('type')} ignored (SP-BPE does not split)")

    with open(a.out, "wb") as f:
        f.write(b"G4TK")
        f.write(struct.pack("<IIIIiI", len(toks), len(merges), len(special),
                            len(rep), unk, int(byte_fallback)))
        f.write(struct.pack("<256i", *btok))
        for t in toks:
            b = t.encode("utf-8")
            f.write(struct.pack("<H", len(b)))
            f.write(b)
        for x, y, z in merges:
            f.write(struct.pack("<iii", x, y, z))
        for i, s in special:
            b = s.encode("utf-8")
            f.write(struct.pack("<iH", i, len(b)))
            f.write(b)
        for x, y in rep:
            bx, by = x.encode("utf-8"), y.encode("utf-8")
            f.write(struct.pack("<H", len(bx)))
            f.write(bx)
            f.write(struct.pack("<H", len(by)))
            f.write(by)

    print(f"vocab {len(toks)}  merges {len(merges)} (dropped {dropped})  "
          f"special {len(special)}  replace {len(rep)}  "
          f"byte_fallback {byte_fallback}  unk {unk}")


if __name__ == "__main__":
    main()
