#!/usr/bin/env python3
"""
lfmtok_check.py — diff lfmtok.h's ids against the HF tokenizer's, case by case.

The engine's tokenizer is a hand-written port of a ByteLevel BPE with a hand-coded
pre-tokenizer regex; every one of those pieces can be subtly wrong in a way that
only shows up on particular inputs (a contraction, a CJK run, an emoji, a run of
spaces before a newline). So the check is adversarial by construction: fixed cases
that target each regex alternative, plus fuzzed strings drawn from an alphabet
chosen to hit the boundaries.

    python3 tools/lfmtok_check.py ./lfm ./lfm-ct/tok.bin

Needs `tokenizers` (or `transformers`) for the reference.
"""
import argparse, json, random, subprocess, sys, os


# reference
# Used only when `tokenizers`/`transformers` is unavailable. It is a second,
# independent transcription of the same spec (straight-line Python against
# str.isalpha()/isnumeric(), rather than C ranges + a hand-rolled state machine),
# so it catches transcription bugs, but it is not authoritative. Run the check
# again with the real HF tokenizer installed before trusting the tokenizer.
WS = set([0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680,
          0x2028, 0x2029, 0x202F, 0x205F, 0x3000] + list(range(0x2000, 0x200B)))


class RefTokenizer:
    def __init__(self, path):
        d = json.load(open(path, encoding="utf-8"))
        self.vocab = d["model"]["vocab"]
        self.ignore_merges = bool(d["model"].get("ignore_merges"))
        self.ranks = {}
        for r, m in enumerate(d["model"]["merges"]):
            x, y = (m if isinstance(m, list) else m.split(" ", 1))
            self.ranks.setdefault((x, y), r)
        self.special = sorted(((a["content"], a["id"]) for a in d.get("added_tokens", [])),
                              key=lambda p: -len(p[0]))
        bs = (list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) +
              list(range(0xAE, 0xFF + 1)))
        cs, n = bs[:], 0
        for b in range(256):
            if b not in bs:
                bs.append(b); cs.append(256 + n); n += 1
        self.b2u = {b: chr(c) for b, c in zip(bs, cs)}

    @staticmethod
    def _is_l(c): return c.isalpha()          # \p{L}

    @staticmethod
    def _is_n(c): return c.isnumeric()        # \p{N} (Nd + Nl + No)

    @staticmethod
    def _is_s(c): return ord(c) in WS         # \p{White_Space}

    def split(self, s):
        """(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}
           | ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"""
        out, i, n = [], 0, len(s)
        L, N, S = self._is_l, self._is_n, self._is_s
        at = lambda k: s[k] if k < n else ""
        while i < n:
            c = s[i]
            k = None
            # 1. contractions
            if c == "'":
                two, three = at(i + 1).lower(), at(i + 2).lower()
                if two in ("s", "t", "m", "d"):
                    k = 2
                elif two + three in ("re", "ve", "ll"):
                    k = 3
            # 2. [^\r\n\p{L}\p{N}]?\p{L}+
            if k is None:
                off = 1 if (not L(c) and not N(c) and c not in "\r\n"
                            and at(i + 1) and L(at(i + 1))) else 0
                if at(i + off) and L(at(i + off)):
                    j = i + off
                    while j < n and L(s[j]):
                        j += 1
                    k = j - i
            # 3. \p{N}{1,3}
            if k is None and N(c):
                j = i
                while j < n and j - i < 3 and N(s[j]):
                    j += 1
                k = j - i
            # 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
            if k is None:
                off = 1 if c == " " else 0
                d = at(i + off)
                if d and not S(d) and not L(d) and not N(d):
                    j = i + off
                    while j < n and not S(s[j]) and not L(s[j]) and not N(s[j]):
                        j += 1
                    while j < n and s[j] in "\r\n":
                        j += 1
                    k = j - i
            # 5/6/7. whitespace runs
            if k is None and S(c):
                j, last_nl = i, 0
                while j < n and S(s[j]):
                    if s[j] in "\r\n":
                        last_nl = j - i + 1
                    j += 1
                run = j - i
                k = last_nl if last_nl else (run - 1 if run > 1 and j < n else run)
            out.append(s[i:i + (k or 1)])
            i += k or 1
        return out

    def _bpe(self, piece):
        word = [self.b2u[b] for b in piece.encode("utf-8")]
        if self.ignore_merges and "".join(word) in self.vocab:
            return [self.vocab["".join(word)]]
        while len(word) > 1:
            best, bi = None, -1
            for j in range(len(word) - 1):
                r = self.ranks.get((word[j], word[j + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, j
            if bi < 0:
                break
            word[bi:bi + 2] = [word[bi] + word[bi + 1]]
        return [self.vocab[w] for w in word if w in self.vocab]

    def encode(self, text):
        segs, out = [text], []
        for content, _ in self.special:               # longest first
            nxt = []
            for s in segs:
                if isinstance(s, tuple):
                    nxt.append(s); continue
                parts = s.split(content)
                for j, p in enumerate(parts):
                    if j:
                        nxt.append((content,))
                    if p:
                        nxt.append(p)
            segs = nxt
        sid = dict(self.special)
        for s in segs:
            if isinstance(s, tuple):
                out.append(sid[s[0]])
            else:
                for piece in self.split(s):
                    out += self._bpe(piece)
        return out


FIXED = [
    "",
    " ",
    "hello world",
    "Hello, world!",
    " leading space",
    "trailing space ",
    "multiple   spaces   here",
    "tabs\tand\tnewlines\nand\r\nCRLF",
    "\n\n\n",
    "   \n   ",
    "line1\nline2\n",
    # contractions: every alternative of (?i:'s|'t|'re|'ve|'m|'ll|'d)
    "it's a test",
    "IT'S UPPERCASE",
    "don't can't won't",
    "they're we've I'm you'll he'd",
    "THEY'RE WE'VE I'M YOU'LL HE'D",
    "'s'''t''re",
    # numbers: \p{N}{1,3} splits in threes
    "1", "12", "123", "1234", "12345", "1234567890",
    "3.14159", "1,000,000", "v1.2.3-beta4",
    "0x1F 0b1010 1e10",
    # punctuation / symbol runs
    "!!!", "?!?!", "-->", "a==b", "(x + y) * z;",
    "...ellipsis...",
    "@#$%^&*()",
    # unicode letters
    "café naïve résumé",
    "Ünïcödé",
    "Привет мир",
    "こんにちは世界",
    "中文测试文本",
    "한국어 텍스트",
    "العربية نص",
    "עברית טקסט",
    "ελληνικά κείμενο",
    "ไทย ข้อความ",
    # unicode numbers (\p{N} beyond ASCII)
    "١٢٣ ٤٥٦",
    "一二三四五",
    "Ⅷ Ⅸ Ⅹ",
    "½ ¾ ⅓",
    # emoji / astral plane
    "hello 👋 world 🌍",
    "👨‍👩‍👧‍👦 family",
    "🇬🇧🇮🇹",
    "a🎉b",
    # mixed scripts, no spaces
    "abc中文def",
    "1a2b3c",
    "test123test",
    "CamelCaseWordsHere",
    "snake_case_words",
    "kebab-case-words",
    # code-ish
    "def foo(x):\n    return x + 1\n",
    "{\"key\": \"value\", \"n\": 42}",
    "<html><body>hi</body></html>",
    "SELECT * FROM t WHERE a='b';",
    "https://example.com/path?q=1&r=2",
    "/usr/local/bin/python3",
    # special tokens must survive verbatim
    "<|im_start|>user\nhi<|im_end|>\n",
    "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n",
    "<|startoftext|>text<|endoftext|>",
    "<|pad|>",
    "text with <|im_end|> inside",
    # nasty whitespace around newlines: exercises \s*[\r\n]+ vs \s+(?!\S) vs \s+
    "a \n b",
    "a  \n  b",
    "a\n \nb",
    "word  ",
    "  word",
    " \t \n \t ",
]

ALPHABET = (
    " \n\t\r"
    "abcXYZ"
    "0123456789"
    "'\"!?.,;:-_+=*/\\()[]{}<>@#$%^&|~`"
    "éüñ"
    "中日한"
    "Привет"
    "👋🌍"
    "١٢"
    "\u00a0\u2009\u3000"   # exotic whitespace
    "<|im_end|>"
)


def fuzz(n, rng):
    out = []
    for _ in range(n):
        k = rng.randint(1, 60)
        out.append("".join(rng.choice(ALPHABET) for _ in range(k)))
    return out


def esc(s):
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="HF checkpoint dir (with tokenizer.json)")
    ap.add_argument("tok_bin", help="converted tok.bin")
    ap.add_argument("--driver", default="./test_lfmtok")
    ap.add_argument("--fuzz", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    tj = os.path.join(a.model_dir, "tokenizer.json")
    authoritative = True
    try:
        from tokenizers import Tokenizer
        ref = Tokenizer.from_file(tj)
        encode = lambda s: ref.encode(s, add_special_tokens=False).ids
    except ImportError:
        try:
            from transformers import AutoTokenizer
            ref = AutoTokenizer.from_pretrained(a.model_dir)
            encode = lambda s: ref.encode(s, add_special_tokens=False)
        except ImportError:
            authoritative = False
            ref = RefTokenizer(tj)
            encode = ref.encode
            print("WARNING: neither `tokenizers` nor `transformers` is installed.\n"
                  "         Falling back to the in-file reference implementation, which\n"
                  "         is NOT authoritative. Re-run with the HF tokenizer installed\n"
                  "         before trusting these results.\n")

    cases = FIXED + fuzz(a.fuzz, random.Random(a.seed))

    stdin = "\n".join(esc(c) for c in cases) + "\n"
    proc = subprocess.run([a.driver, a.tok_bin], input=stdin,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"driver failed: {proc.stderr}")
    got = proc.stdout.split("\n")

    bad = 0
    for i, case in enumerate(cases):
        want = encode(case)
        mine = [int(x) for x in got[i].split()] if i < len(got) and got[i].strip() else []
        if mine != want:
            bad += 1
            if bad <= 20:
                print(f"MISMATCH {esc(case)!r}\n   hf  {want}\n   c   {mine}")
                if len(want) == len(mine):
                    for j, (w, m) in enumerate(zip(want, mine)):
                        if w != m:
                            print(f"   first diff at {j}: hf {w} != c {m}")
                            break
    print(f"\n{len(cases) - bad}/{len(cases)} exact  ({bad} mismatches)"
          f"{'' if authoritative else '  [vs NON-AUTHORITATIVE reference]'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
