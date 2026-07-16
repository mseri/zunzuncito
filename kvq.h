/* kvq.h — TurboQuant V3 KV-cache quantisation.
 *
 * From tonbistudio/turboquant-pytorch (Google's TurboQuant, ICLR 2026), V3 variant:
 * MSE-only (no QJL), asymmetric K/V bit-widths, bit-packed, layer-adaptive.
 *
 * WHY IT IS WORTH ANYTHING HERE. Gemma-4's sliding layers cap their KV at 1024
 * positions, so 25 of 30 layers cost a fixed 400 MiB no matter the context. ALL KV
 * growth is in the 5 global layers. At 4 GB and 4K ctx, quantising KV buys about one
 * extra expert slot -- nothing. At 32K it takes the cache from 9 slots/layer to 21,
 * and at 128K it is the difference between running and not running at all. This is a
 * CONTEXT-LENGTH feature, not a general RAM feature.
 *
 * THE ALGORITHM
 *   quantise:  x -> n = |x|, u = x/n -> RHT -> scale by sqrt(d) -> Lloyd-Max round
 *              each coordinate independently -> pack b bits/coord + store n
 *   restore:   unpack -> centroids -> /sqrt(d) -> inverse RHT -> * n
 *
 * The rotation is the whole trick: it makes every coordinate of a unit vector
 * approximately N(0, 1/d) regardless of what the original vector looked like, so ONE
 * precomputed scalar quantiser (Lloyd-Max for a standard normal) is optimal for every
 * coordinate. No per-tensor calibration, no outlier problem.
 *
 * We use a randomised Hadamard transform (sign flips + fast Walsh-Hadamard) rather
 * than the reference's dense random orthogonal matrix: same distributional effect,
 * but O(d log d) instead of O(d^2). At d=512 and 240 KV writes/token that difference
 * is not optional. Requires d to be a power of two -- Gemma-4's head dims are 256 and
 * 512, so this is exact, not a compromise.
 *
 * K4/V2 -- READ THIS.
 * The upstream README's own corrected generation table says K4/V2 with no residual
 * window MISSES the needle at both 2K and 4K, i.e. generation is broken; and that
 * "high attention score similarity (99.5%+) does not guarantee working generation".
 * The 99% top-1 figure people quote for "K4/V2 + protected layers" is from their
 * ATTENTION-SCORE table, which is the very table they warn is not predictive. Their
 * only config measured EXACT is K6/V4 with a 128-token fp32 residual window (~2x).
 * So: bits are configurable, K4/V2 is reachable, and the default is K6/V4 rw=128.
 * Measure before trusting either.
 */
#ifndef KVQ_H
#define KVQ_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#define KVQ_MAXBITS 8
/* Gemma-4 head dims are 256 (sliding) and 512 (global). Bounding this lets the codec
 * use fixed stack arrays instead of alloca -- which is a GNU-ism, and a stack-overflow
 * waiting to happen if a head dim ever grows. */
#define KVQ_MAXD 1024

/* ---------------------------------------------------------------- Lloyd-Max
 * Optimal scalar quantiser for a standard normal: centroids solving the
 * fixed point c_i = E[X | X in cell_i], boundaries at the midpoints. Solved once
 * at startup by Lloyd iteration over a fine grid -- a few ms, no tables to ship. */
typedef struct { int n; float c[1 << KVQ_MAXBITS]; float b[1 << KVQ_MAXBITS]; } KvqLM;

static void kvq_lloyd(KvqLM *q, int bits) {
    int n = 1 << bits;
    q->n = n;
    const int G = 20001;
    const float LO = -6.0f, HI = 6.0f;
    static float x[20001], p[20001];
    float dx = (HI - LO) / (G - 1);
    for (int i = 0; i < G; i++) {
        x[i] = LO + i * dx;
        p[i] = expf(-0.5f * x[i] * x[i]);        /* unnormalised N(0,1) */
    }
    for (int i = 0; i < n; i++)                  /* init: uniform over [-3,3] */
        q->c[i] = -3.0f + 6.0f * (i + 0.5f) / n;

    for (int it = 0; it < 200; it++) {
        double num[1 << KVQ_MAXBITS] = {0}, den[1 << KVQ_MAXBITS] = {0};
        int j = 0;
        for (int i = 0; i < G; i++) {
            while (j + 1 < n && fabsf(x[i] - q->c[j + 1]) < fabsf(x[i] - q->c[j])) j++;
            num[j] += (double)p[i] * x[i];
            den[j] += p[i];
        }
        float mx = 0;
        for (int k = 0; k < n; k++) {
            if (den[k] > 1e-12) {
                float c = (float)(num[k] / den[k]);
                if (fabsf(c - q->c[k]) > mx) mx = fabsf(c - q->c[k]);
                q->c[k] = c;
            }
        }
        if (mx < 1e-7f) break;
    }
    for (int i = 0; i + 1 < n; i++) q->b[i] = 0.5f * (q->c[i] + q->c[i + 1]);
    q->b[n - 1] = 1e30f;
}

static inline int kvq_round(const KvqLM *q, float v) {
    int lo = 0, hi = q->n - 1;                   /* boundaries are sorted: bisect */
    while (lo < hi) {
        int m = (lo + hi) >> 1;
        if (v <= q->b[m]) hi = m; else lo = m + 1;
    }
    return lo;
}

/* ------------------------------------------------- randomised Hadamard transform */
/* in-place fast Walsh-Hadamard; d must be a power of two */
static void kvq_fwht(float *a, int d) {
    for (int len = 1; len < d; len <<= 1)
        for (int i = 0; i < d; i += len << 1)
            for (int j = i; j < i + len; j++) {
                float u = a[j], v = a[j + len];
                a[j] = u + v;
                a[j + len] = u - v;
            }
}

/* H D x / sqrt(d): orthonormal, self-inverse up to the sign flips. */
static inline void kvq_rht(float *a, const int8_t *sign, int d) {
    for (int i = 0; i < d; i++) a[i] *= sign[i];
    kvq_fwht(a, d);
    float s = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) a[i] *= s;
}
static inline void kvq_irht(float *a, const int8_t *sign, int d) {
    kvq_fwht(a, d);                              /* H is symmetric and H*H = d*I */
    float s = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) a[i] *= s * sign[i];
}

/* ------------------------------------------------------------------ bit packing */
static inline void kvq_pack(uint8_t *dst, const int *q, int d, int bits) {
    memset(dst, 0, ((size_t)d * bits + 7) / 8);
    for (int i = 0; i < d; i++) {
        size_t bit = (size_t)i * bits;
        uint32_t v = (uint32_t)q[i];
        for (int b = 0; b < bits; b++)
            if (v & (1u << b)) dst[(bit + b) >> 3] |= (uint8_t)(1u << ((bit + b) & 7));
    }
}
static inline void kvq_unpack(const uint8_t *src, int *q, int d, int bits) {
    for (int i = 0; i < d; i++) {
        size_t bit = (size_t)i * bits;
        uint32_t v = 0;
        for (int b = 0; b < bits; b++)
            if (src[(bit + b) >> 3] & (1u << ((bit + b) & 7))) v |= 1u << b;
        q[i] = (int)v;
    }
}

/* ------------------------------------------------------------------ the codec */
typedef struct {
    int d, bits;
    int8_t *sign;          /* [d] the RHT's random sign flips */
    KvqLM lm;
    size_t bytes;          /* packed bytes per vector, excluding the norm */
} Kvq;

static inline size_t kvq_vec_bytes(int d, int bits) {
    return ((size_t)d * bits + 7) / 8 + sizeof(float);   /* payload + norm */
}

static void kvq_init(Kvq *k, int d, int bits, uint64_t seed) {
    if (d > KVQ_MAXD) { fprintf(stderr, "kvq: head_dim %d > KVQ_MAXD\n", d); exit(1); }
    k->d = d;
    k->bits = bits;
    k->sign = malloc(d);
    /* Fixed seed per (layer, K/V): the decoder must reproduce the same rotation, and
     * we never persist it -- it is regenerated identically from the seed. */
    uint64_t s = seed ? seed : 1;
    for (int i = 0; i < d; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        k->sign[i] = (s >> 33) & 1 ? 1 : -1;
    }
    kvq_lloyd(&k->lm, bits);
    k->bytes = kvq_vec_bytes(d, bits);
}

static inline void kvq_free(Kvq *k) { free(k->sign); }

/* x[d] -> dst (packed payload followed by the f32 norm) */
static void kvq_encode(const Kvq *k, const float *x, uint8_t *dst) {
    int d = k->d;
    float t[KVQ_MAXD];
    int q[KVQ_MAXD];

    double nn = 0;
    for (int i = 0; i < d; i++) nn += (double)x[i] * x[i];
    float n = (float)sqrt(nn);
    float inv = n > 0 ? 1.0f / n : 0.0f;
    for (int i = 0; i < d; i++) t[i] = x[i] * inv;

    kvq_rht(t, k->sign, d);
    /* after rotating a UNIT vector, coordinates are ~N(0, 1/d); scale to N(0,1) so
     * the single standard-normal Lloyd-Max quantiser is the right one for all of them */
    float s = sqrtf((float)d);
    for (int i = 0; i < d; i++) q[i] = kvq_round(&k->lm, t[i] * s);

    kvq_pack(dst, q, d, k->bits);
    memcpy(dst + k->bytes - sizeof(float), &n, sizeof(float));
}

static void kvq_decode(const Kvq *k, const uint8_t *src, float *x) {
    int d = k->d;
    int q[KVQ_MAXD];
    kvq_unpack(src, q, d, k->bits);

    float n;
    memcpy(&n, src + k->bytes - sizeof(float), sizeof(float));
    float s = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) x[i] = k->lm.c[q[i]] * s;

    kvq_irht(x, k->sign, d);
    for (int i = 0; i < d; i++) x[i] *= n;
}

#endif /* KVQ_H */
