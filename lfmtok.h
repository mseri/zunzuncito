/* lfmtok.h — LFM2.5's tokenizer. Same container and integer-only BPE as g4tok.h;
 * what differs is the pre-tokenizer.
 *
 * The input is split on the Qwen2/Llama-3 regex before BPE, and merges never cross a
 * piece boundary. Skipping the split silently produces different ids, so it is not an
 * optimisation to drop.
 *
 * The regex is hand-coded, following the structure llama.cpp uses for the same
 * pattern. It needs Unicode \p{L} / \p{N} membership, which the converter ships as
 * sorted ranges in the container (binary-searched here); \s is the 25-codepoint
 * White_Space set and is hardcoded.
 */
#ifndef LFMTOK_H
#define LFMTOK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct { int32_t a, b, ab; } LfmMerge;
typedef struct { uint32_t lo, hi; } LfmRange;

typedef struct {
    int n_vocab, n_merges, n_special, n_lrange, n_nrange;
    int32_t ignore_merges, bos, eos;
    /* Longest run of digits one piece may take: 3 for LFM2.5's \p{N}{1,3}, 1 for
     * Maple/Qwen2's bare \p{N}. It is the only part of the split regex that varies
     * between the two checkpoints, and getting it wrong renumbers every multi-digit
     * number in the prompt, so the converter writes it rather than the engine
     * assuming it. */
    int32_t digit_max;
    /* Codepoints whose NFC quick-check is not Yes -- i.e. the only ones NFC could
     * rewrite. Empty for tokenizers with no normaliser; see lfmtok_nfc_safe(). */
    LfmRange *qr;
    int n_qcrange;

    char **tok;            /* [n_vocab] byte-level spellings, NUL-terminated */
    uint16_t *tlen;

    /* merge table: open-addressed map (a,b) -> rank, and rank -> merged id */
    uint64_t *mkey;        /* 0 = empty; key is ((a+1)<<32 | (b+1)) so 0 is free */
    int32_t *mrank;
    int32_t *mid;
    size_t mcap;

    int32_t *sp_id;  char **sp_str;  uint16_t *sp_len;      /* special tokens */
    LfmRange *lr, *nr;                                      /* \p{L}, \p{N} */

    /* id lookup for a literal byte-level string (used to seed symbols) */
    uint64_t *vkey;  int32_t *vval;  size_t vcap;

    /* GPT-2 byte<->unicode table */
    uint32_t b2u[256];
    int16_t u2b[512];      /* codepoint -> byte, or -1 */
} LfmTok;

/* hashing */
static uint64_t lfm_hash(const char *s, int n) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ULL; }
    return h ? h : 1;
}

static int32_t lfm_lookup(const LfmTok *t, const char *s, int n) {
    uint64_t h = lfm_hash(s, n);
    size_t i = h & (t->vcap - 1);
    for (;;) {
        if (!t->vkey[i]) return -1;
        if (t->vkey[i] == h) {
            int32_t id = t->vval[i];
            if (t->tlen[id] == n && !memcmp(t->tok[id], s, n)) return id;
        }
        i = (i + 1) & (t->vcap - 1);
    }
}

static int lfm_merge_find(const LfmTok *t, int32_t a, int32_t b, int32_t *id) {
    uint64_t k = ((uint64_t)(a + 1) << 32) | (uint32_t)(b + 1);
    size_t i = (k * 1099511628211ULL) & (t->mcap - 1);
    for (;;) {
        if (!t->mkey[i]) return -1;
        if (t->mkey[i] == k) { *id = t->mid[i]; return t->mrank[i]; }
        i = (i + 1) & (t->mcap - 1);
    }
}

static size_t lfm_pow2(size_t n) { size_t c = 8; while (c < n * 2) c <<= 1; return c; }

/* character classes */
static int lfm_in_ranges(const LfmRange *r, int n, uint32_t c) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (c < r[mid].lo) hi = mid - 1;
        else if (c > r[mid].hi) lo = mid + 1;
        else return 1;
    }
    return 0;
}
static int lfm_is_letter(const LfmTok *t, uint32_t c) {
    return lfm_in_ranges(t->lr, t->n_lrange, c);
}
static int lfm_is_number(const LfmTok *t, uint32_t c) {
    return lfm_in_ranges(t->nr, t->n_nrange, c);
}
/* Unicode White_Space. Small and fixed, so no table from the converter. */
static int lfm_is_space(uint32_t c) {
    return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 ||
           c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

/* UTF-8 */
/* Decode one codepoint. Invalid bytes decode as themselves (Latin-1), which keeps
 * the splitter total: byte-level BPE must never reject its input. */
static int lfm_utf8_next(const char *s, int n, int i, uint32_t *cp) {
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) { *cp = c; return 1; }
    int len = (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (len == 1 || i + len > n) { *cp = c; return 1; }
    uint32_t v = c & (0x7f >> len);
    for (int k = 1; k < len; k++) {
        unsigned char d = (unsigned char)s[i + k];
        if ((d & 0xc0) != 0x80) { *cp = c; return 1; }
        v = (v << 6) | (d & 0x3f);
    }
    *cp = v;
    return len;
}
static int lfm_utf8_put(char *o, uint32_t c) {
    if (c < 0x80) { o[0] = (char)c; return 1; }
    if (c < 0x800) { o[0] = (char)(0xc0 | (c >> 6)); o[1] = (char)(0x80 | (c & 0x3f)); return 2; }
    if (c < 0x10000) {
        o[0] = (char)(0xe0 | (c >> 12)); o[1] = (char)(0x80 | ((c >> 6) & 0x3f));
        o[2] = (char)(0x80 | (c & 0x3f)); return 3;
    }
    o[0] = (char)(0xf0 | (c >> 18)); o[1] = (char)(0x80 | ((c >> 12) & 0x3f));
    o[2] = (char)(0x80 | ((c >> 6) & 0x3f)); o[3] = (char)(0x80 | (c & 0x3f));
    return 4;
}

/* load */
static void lfm_byte_table(LfmTok *t) {
    int used[256] = {0};
    int n = 0;
    for (int b = '!'; b <= '~'; b++)   { t->b2u[b] = (uint32_t)b; used[b] = 1; }
    for (int b = 0xA1; b <= 0xAC; b++) { t->b2u[b] = (uint32_t)b; used[b] = 1; }
    for (int b = 0xAE; b <= 0xFF; b++) { t->b2u[b] = (uint32_t)b; used[b] = 1; }
    for (int b = 0; b < 256; b++)
        if (!used[b]) t->b2u[b] = (uint32_t)(256 + n++);
    for (int i = 0; i < 512; i++) t->u2b[i] = -1;
    for (int b = 0; b < 256; b++)
        if (t->b2u[b] < 512) t->u2b[t->b2u[b]] = (int16_t)b;
}

static LfmTok *lfmtok_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    char magic[4];
    /* "LFT2" adds digit_max after eos. "LFTK" is the original container and has no
     * such field, so it keeps the \p{N}{1,3} it was written against; existing lfm25
     * tok.bin files load unchanged. */
    int v2;
    if (fread(magic, 1, 4, f) != 4 ||
        (!(v2 = !memcmp(magic, "LFT2", 4)) && memcmp(magic, "LFTK", 4))) {
        fprintf(stderr, "%s: bad magic\n", path); fclose(f); return NULL;
    }
    LfmTok *t = calloc(1, sizeof *t);
    uint32_t nv, nm, ns, nl, nn, ig, nq = 0; int32_t bos, eos, dmax = 3;
    if (fread(&nv, 4, 1, f) != 1 || fread(&nm, 4, 1, f) != 1 ||
        fread(&ns, 4, 1, f) != 1 || fread(&nl, 4, 1, f) != 1 ||
        fread(&nn, 4, 1, f) != 1 || fread(&ig, 4, 1, f) != 1 ||
        fread(&bos, 4, 1, f) != 1 || fread(&eos, 4, 1, f) != 1) goto bad;
    if (v2 && (fread(&dmax, 4, 1, f) != 1 || fread(&nq, 4, 1, f) != 1)) goto bad;
    if (dmax < 1) dmax = 1;
    t->n_vocab = nv; t->n_merges = nm; t->n_special = ns;
    t->n_lrange = nl; t->n_nrange = nn; t->n_qcrange = nq;
    t->ignore_merges = ig; t->bos = bos; t->eos = eos; t->digit_max = dmax;

    t->tok = calloc(nv, sizeof(char *));
    t->tlen = calloc(nv, sizeof(uint16_t));
    for (uint32_t i = 0; i < nv; i++) {
        uint16_t L;
        if (fread(&L, 2, 1, f) != 1) goto bad;
        t->tok[i] = malloc(L + 1);
        if (L && fread(t->tok[i], 1, L, f) != L) goto bad;
        t->tok[i][L] = 0;
        t->tlen[i] = L;
    }

    t->mcap = lfm_pow2(nm ? nm : 1);
    t->mkey = calloc(t->mcap, 8);
    t->mrank = calloc(t->mcap, 4);
    t->mid = calloc(t->mcap, 4);
    for (uint32_t r = 0; r < nm; r++) {
        LfmMerge m;
        if (fread(&m, sizeof m, 1, f) != 1) goto bad;
        uint64_t k = ((uint64_t)(m.a + 1) << 32) | (uint32_t)(m.b + 1);
        size_t i = (k * 1099511628211ULL) & (t->mcap - 1);
        while (t->mkey[i] && t->mkey[i] != k) i = (i + 1) & (t->mcap - 1);
        if (!t->mkey[i]) {                 /* first rule for this pair wins: lowest rank */
            t->mkey[i] = k; t->mrank[i] = r; t->mid[i] = m.ab;
        }
    }

    t->sp_id = calloc(ns ? ns : 1, 4);
    t->sp_str = calloc(ns ? ns : 1, sizeof(char *));
    t->sp_len = calloc(ns ? ns : 1, 2);
    for (uint32_t i = 0; i < ns; i++) {
        uint16_t L;
        if (fread(&t->sp_id[i], 4, 1, f) != 1 || fread(&L, 2, 1, f) != 1) goto bad;
        t->sp_str[i] = malloc(L + 1);
        if (L && fread(t->sp_str[i], 1, L, f) != L) goto bad;
        t->sp_str[i][L] = 0;
        t->sp_len[i] = L;
    }

    t->lr = calloc(nl ? nl : 1, sizeof(LfmRange));
    if (nl && fread(t->lr, sizeof(LfmRange), nl, f) != nl) goto bad;
    t->nr = calloc(nn ? nn : 1, sizeof(LfmRange));
    if (nn && fread(t->nr, sizeof(LfmRange), nn, f) != nn) goto bad;
    t->qr = calloc(nq ? nq : 1, sizeof(LfmRange));
    if (nq && fread(t->qr, sizeof(LfmRange), nq, f) != nq) goto bad;
    fclose(f);

    t->vcap = lfm_pow2(nv);
    t->vkey = calloc(t->vcap, 8);
    t->vval = calloc(t->vcap, 4);
    for (uint32_t i = 0; i < nv; i++) {
        if (!t->tlen[i]) continue;
        uint64_t h = lfm_hash(t->tok[i], t->tlen[i]);
        size_t j = h & (t->vcap - 1);
        while (t->vkey[j]) j = (j + 1) & (t->vcap - 1);
        t->vkey[j] = h; t->vval[j] = i;
    }
    lfm_byte_table(t);
    return t;
bad:
    fprintf(stderr, "%s: truncated\n", path);
    fclose(f);
    return NULL;
}

/* 1 if NFC provably leaves `text` alone, so the ids below match HF's exactly.
 *
 * Always 1 for a tokenizer with no normaliser (n_qcrange == 0). Maple's declares
 * NFC, and implementing it here would mean canonical decomposition, combining-class
 * reordering and a composition table for a transform that is the identity on
 * essentially all prompt text. So the engine proves the identity case instead: NFC
 * leaves any string alone if none of its codepoints fails the NFC quick-check, and
 * that covers ASCII, precomposed accents and every CJK script. It fails on
 * decomposed input ("e" + U+0301), where HF would compose first and we do not; the
 * caller is expected to say so rather than pretend. */
__attribute__((unused))
static int lfmtok_nfc_safe(const LfmTok *t, const char *text) {
    if (!t->n_qcrange) return 1;
    int n = (int)strlen(text);
    for (int i = 0; i < n;) {
        uint32_t c;
        i += lfm_utf8_next(text, n, i, &c);
        if (c >= 0x300 && lfm_in_ranges(t->qr, t->n_qcrange, c)) return 0;
    }
    return 1;
}

/* id of a named special token, or -1. Used for the chat template's stop tokens:
 * hardcoding 124900 would silently break on a re-tokenised checkpoint. */
static int lfmtok_id(const LfmTok *t, const char *name) {
    size_t n = strlen(name);
    for (int i = 0; i < t->n_special; i++)
        if (t->sp_len[i] == n && !memcmp(t->sp_str[i], name, n)) return t->sp_id[i];
    return -1;
}

/* BPE */
typedef struct { int32_t id; int prev, next; } LfmSym;

/* BPE one pre-tokenizer piece, given as RAW bytes. */
static int lfm_bpe(const LfmTok *t, const char *s, int n, int *out, int cap, int nout) {
    if (!n) return nout;

    /* byte-level translate: each raw byte becomes one printable codepoint */
    char *tr = malloc((size_t)n * 4 + 4);
    int *coff = malloc(sizeof(int) * (size_t)(n + 1));
    int tn = 0;
    for (int i = 0; i < n; i++) {
        coff[i] = tn;
        tn += lfm_utf8_put(tr + tn, t->b2u[(unsigned char)s[i]]);
    }
    coff[n] = tn;

    /* ignore_merges: a piece that is itself a token is emitted verbatim */
    if (t->ignore_merges) {
        int32_t id = lfm_lookup(t, tr, tn);
        if (id >= 0) {
            if (nout < cap) out[nout++] = id;
            free(tr); free(coff);
            return nout;
        }
    }

    LfmSym *sy = malloc(sizeof(LfmSym) * (size_t)(n + 1));
    int ns = 0;
    for (int i = 0; i < n; i++) {
        int32_t id = lfm_lookup(t, tr + coff[i], coff[i + 1] - coff[i]);
        if (id < 0) continue;            /* cannot happen: all 256 are in the vocab */
        sy[ns].id = id; sy[ns].prev = ns - 1; sy[ns].next = ns + 1; ns++;
    }
    if (ns) sy[ns - 1].next = -1;

    for (;;) {
        int best = -1, brank = 0x7fffffff;
        int32_t bid = -1;
        for (int i = 0; i != -1 && i < ns; i = sy[i].next) {
            int j = sy[i].next;
            if (j == -1 || j >= ns) break;
            int32_t id;
            int r = lfm_merge_find(t, sy[i].id, sy[j].id, &id);
            if (r >= 0 && r < brank) { brank = r; best = i; bid = id; }
        }
        if (best < 0) break;
        int j = sy[best].next;
        sy[best].id = bid;
        sy[best].next = sy[j].next;
        if (sy[j].next != -1) sy[sy[j].next].prev = best;
    }

    for (int i = 0; i != -1 && i < ns; i = sy[i].next) {
        if (nout < cap) out[nout++] = sy[i].id;
        if (sy[i].next == i) break;
    }
    free(sy); free(tr); free(coff);
    return nout;
}

/* pre-tokenizer split
 *
 * Hand-coded equivalent of
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,digit_max}
 *   | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 * Alternatives are tried in order, exactly as the regex engine would. digit_max
 * comes from the container (see the struct).
 * Returns the number of codepoints consumed starting at `i` (always >= 1). */
static int lfm_split_one(const LfmTok *t, const uint32_t *cp, int n, int i) {
    #define CP(k) ((i + (k) < n) ? cp[i + (k)] : 0u)
    #define LOWER(c) (((c) >= 'A' && (c) <= 'Z') ? (c) + 32 : (c))

    /* 1. contractions, case-insensitive */
    if (CP(0) == '\'') {
        uint32_t a = LOWER(CP(1));
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
        uint32_t b = LOWER(CP(2));
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l'))
            return 3;
    }

    /* 2. [^\r\n\p{L}\p{N}]? \p{L}+ */
    {
        uint32_t c = CP(0);
        int off = 0;
        if (!lfm_is_letter(t, c) && !lfm_is_number(t, c) && c != '\r' && c != '\n' &&
            lfm_is_letter(t, CP(1)))
            off = 1;
        if (lfm_is_letter(t, CP(off))) {
            int k = off;
            while (i + k < n && lfm_is_letter(t, cp[i + k])) k++;
            return k;
        }
    }

    /* 3. \p{N}{1,digit_max} */
    if (lfm_is_number(t, CP(0))) {
        int k = 0;
        while (k < t->digit_max && i + k < n && lfm_is_number(t, cp[i + k])) k++;
        return k;
    }

    /* 4.  ?[^\s\p{L}\p{N}]+[\r\n]* */
    {
        int off = (CP(0) == ' ') ? 1 : 0;
        uint32_t c = CP(off);
        if (i + off < n && !lfm_is_space(c) && !lfm_is_letter(t, c) && !lfm_is_number(t, c)) {
            int k = off;
            while (i + k < n && !lfm_is_space(cp[i + k]) &&
                   !lfm_is_letter(t, cp[i + k]) && !lfm_is_number(t, cp[i + k])) k++;
            while (i + k < n && (cp[i + k] == '\r' || cp[i + k] == '\n')) k++;
            return k;
        }
    }

    /* 5/6/7. the whitespace runs. Measure the whole run once, then decide:
     *   \s*[\r\n]+   -> cut after the last newline in the run
     *   \s+(?!\S)    -> a run followed by a non-space keeps one space back, so the
     *                   next piece can start with " x" via alternative 2 or 4
     *   \s+          -> otherwise the whole run */
    {
        int k = 0, last_nl = 0;
        while (i + k < n && lfm_is_space(cp[i + k])) {
            if (cp[i + k] == '\r' || cp[i + k] == '\n') last_nl = k + 1;
            k++;
        }
        if (last_nl > 0) return last_nl;
        if (k > 1 && i + k < n) return k - 1;
        if (k > 0) return k;
    }

    return 1;
    #undef CP
    #undef LOWER
}

/* encode */
static int lfmtok_encode(const LfmTok *t, const char *text, int *out, int cap) {
    size_t bn = strlen(text);
    int nout = 0;
    size_t seg = 0, i = 0;

    /* Decode the segment [seg,end) to codepoints, split it, BPE each piece. */
    #define FLUSH(end) do {                                                        \
        size_t _s = (seg), _e = (end);                                             \
        if (_e > _s) {                                                             \
            int _n = (int)(_e - _s);                                               \
            uint32_t *_cp = malloc(sizeof(uint32_t) * (size_t)_n);                 \
            int *_off = malloc(sizeof(int) * (size_t)(_n + 1));                    \
            int _m = 0;                                                            \
            for (int _k = 0; _k < _n;) {                                           \
                _off[_m] = _k;                                                     \
                _k += lfm_utf8_next(text + _s, _n, _k, &_cp[_m]);                  \
                _m++;                                                              \
            }                                                                      \
            _off[_m] = _n;                                                         \
            for (int _p = 0; _p < _m;) {                                           \
                int _len = lfm_split_one(t, _cp, _m, _p);                          \
                if (_len < 1) _len = 1;                                            \
                if (_p + _len > _m) _len = _m - _p;                                \
                nout = lfm_bpe(t, text + _s + _off[_p],                            \
                               _off[_p + _len] - _off[_p], out, cap, nout);        \
                _p += _len;                                                        \
            }                                                                      \
            free(_cp); free(_off);                                                 \
        }                                                                          \
    } while (0)

    while (i < bn) {
        int hit = -1;
        for (int s = 0; s < t->n_special; s++) {
            uint16_t L = t->sp_len[s];
            if (L && i + L <= bn && !memcmp(text + i, t->sp_str[s], L)) {
                if (hit < 0 || L > t->sp_len[hit]) hit = s;   /* longest match wins */
            }
        }
        if (hit >= 0) {
            FLUSH(i);
            if (nout < cap) out[nout++] = t->sp_id[hit];
            i += t->sp_len[hit];
            seg = i;
        } else {
            i++;
        }
    }
    FLUSH(bn);
    #undef FLUSH
    return nout;
}

/* decode */
static int lfmtok_decode(const LfmTok *t, const int *ids, int n, char *out, int cap) {
    int m = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id < 0 || id >= t->n_vocab) continue;
        const char *s = t->tok[id];
        int L = t->tlen[id];

        /* A special token is raw text, not byte-level: emit it verbatim. */
        int spec = 0;
        for (int k = 0; k < t->n_special; k++)
            if (t->sp_id[k] == id) { spec = 1; break; }
        if (spec) {
            for (int k = 0; k < L && m < cap - 1; k++) out[m++] = s[k];
            continue;
        }
        /* otherwise undo the byte-level mapping, codepoint by codepoint */
        for (int k = 0; k < L;) {
            uint32_t c;
            k += lfm_utf8_next(s, L, k, &c);
            int b = (c < 512) ? t->u2b[c] : -1;
            if (b >= 0) { if (m < cap - 1) out[m++] = (char)b; }
            else if (m < cap - 5) m += lfm_utf8_put(out + m, c);
        }
    }
    out[m] = 0;
    return m;
}

#endif /* LFMTOK_H */
