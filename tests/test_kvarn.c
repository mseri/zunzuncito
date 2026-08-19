/* test_kvarn.c — validate the KVarN codec: the Hadamard, the Sinkhorn balancing,
 * round-trip fidelity at each bit width, and the attention-score agreement that is
 * the only figure that actually predicts whether generation survives.
 * Build: cc -O2 -I. tests/test_kvarn.c -lm -o /tmp/tk && /tmp/tk
 */
#include "kvarn.h"
#include <stdio.h>

static uint64_t rs = 88172645463325252ULL;
static double u01(void) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (rs >> 11) / (double)(1ULL << 53);
}
static float gauss(void) {
    double u = u01(), v = u01();
    if (u < 1e-12) u = 1e-12;
    return (float)(sqrt(-2 * log(u)) * cos(2 * M_PI * v));
}

/* A tile that looks like real KV rather than like noise: isotropic bulk plus a few
 * channels an order of magnitude hotter, constant across tokens, plus the odd loud
 * token. That structure is the entire reason a per-vector quantiser struggles here,
 * so a test on plain gaussians would say nothing. */
static void make_kv(float *t, int ntok, int d, size_t stride) {
    float chan[KVARN_MAXD];
    for (int i = 0; i < d; i++) chan[i] = 1.0f;
    for (int k = 0; k < 6; k++) chan[(int)(u01() * d) % d] = 8.0f + 12.0f * (float)u01();
    for (int i = 0; i < ntok; i++) {
        float tok = u01() < 0.05 ? 4.0f : 1.0f;
        float *v = t + (size_t)i * stride;
        for (int j = 0; j < d; j++) v[j] = gauss() * chan[j] * tok;
    }
}

static double relerr(const float *a, const float *b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) {
        double e = (double)a[i] - b[i];
        num += e * e; den += (double)a[i] * a[i];
    }
    return den > 0 ? sqrt(num / den) : 0.0;
}

int main(void) {
    int fail = 0;

    /* ---- 1. the Hadamard must be exactly orthonormal and self-inverse */
    {
        int d = 256;
        float x[256], y[256];
        for (int i = 0; i < d; i++) x[i] = gauss();
        memcpy(y, x, sizeof x);
        kvarn_rot(y, d);
        double n0 = 0, n1 = 0;
        for (int i = 0; i < d; i++) { n0 += (double)x[i] * x[i]; n1 += (double)y[i] * y[i]; }
        kvarn_rot(y, d);
        double err = 0;
        for (int i = 0; i < d; i++) err = fmax(err, fabs(x[i] - y[i]));
        printf("Hadamard: norm %.9f -> %.9f ; self-inverse max err %.2e\n",
               sqrt(n0), sqrt(n1), err);
        if (fabs(sqrt(n0) - sqrt(n1)) > 1e-3 || err > 1e-4) { printf("FAIL hadamard\n"); fail = 1; }
    }

    /* ---- 2. Sinkhorn must actually flatten the variance it was given, both on a
     * raw tile and on the rotated one the encoder really hands it. The rotation
     * already deals with per-channel outliers, so the second number starts much
     * lower; what Sinkhorn removes there is the token-level spread, which no
     * within-token rotation can touch. */
    {
        int d = 128, g = 128;
        float *tile = malloc(sizeof(float) * (size_t)g * d);
        float *csd = malloc(sizeof(float) * g), *rsd = malloc(sizeof(float) * d);
        float *ca = malloc(sizeof(float) * g), *cb = malloc(sizeof(float) * g);
        make_kv(tile, g, d, d);
        printf("\nSinkhorn imbalance (perfect is 2.0)\n");
        for (int rot = 0; rot < 2; rot++) {
            Kvarn q; kvarn_init(&q, d, 4, g, 1);
            for (int t = 0; t < g; t++) {
                float v[128];
                memcpy(v, tile + (size_t)t * d, sizeof v);
                if (rot) kvarn_rot(v, d);
                for (int i = 0; i < d; i++) q.work[(size_t)i * g + t] = v[i];
            }
            kvarn_stds(q.work, q.R, q.C, csd, rsd, ca, cb);
            double before = kvarn_spread(csd, rsd, q.R, q.C);
            kvarn_balance(&q);
            kvarn_stds(q.cur, q.R, q.C, csd, rsd, ca, cb);
            double after = kvarn_spread(csd, rsd, q.R, q.C);
            printf("  %-10s %6.1f -> %.1f\n", rot ? "rotated" : "raw", before, after);
            if (!(after < before) || after > 3.0) { printf("FAIL sinkhorn\n"); fail = 1; }
            kvarn_free(&q);
        }
        free(tile); free(csd); free(rsd); free(ca); free(cb);
    }

    /* ---- 3. round-trip fidelity, both orientations, every bit width we ship */
    printf("\nround-trip relative error (d=128, tile=128, outlier-heavy KV)\n");
    printf("  bits      K (per-channel)   V (per-token)   bytes/token\n");
    {
        /* Measured on this generator plus ~15% headroom, so a regression in the
         * balancer or the packing trips it but ordinary noise does not. The ratio
         * between adjacent rows is the figure to watch: it should stay near 2, one
         * bit's worth, all the way down. */
        const double kbound[9] = {0, 0, 0.50, 0.21, 0.100, 0, 0.025, 0, 0.007};
        for (int bits = 2; bits <= 8; bits++) {
            if (bits == 5 || bits == 7) continue;
            int d = 128, g = 128;
            double e[2] = {0, 0};
            size_t bytes = 0;
            for (int isk = 1; isk >= 0; isk--) {
                Kvarn q; kvarn_init(&q, d, bits, g, isk);
                float *tile = malloc(sizeof(float) * (size_t)g * d);
                float *out = malloc(sizeof(float) * d);
                uint8_t *rec = malloc(q.bytes);
                double acc = 0;
                for (int rep = 0; rep < 8; rep++) {
                    make_kv(tile, g, d, d);
                    kvarn_encode_tile(&q, tile, d, g, rec);
                    for (int t = 0; t < g; t++) {
                        kvarn_decode_vec(&q, rec, t, out);
                        acc += relerr(tile + (size_t)t * d, out, d);
                    }
                }
                e[isk] = acc / (8 * g);
                if (isk) bytes = q.bytes;
                free(tile); free(out); free(rec); kvarn_free(&q);
            }
            printf("   %d          %.4f            %.4f          %.2f\n",
                   bits, e[1], e[0], bytes / (double)128);
            if (kbound[bits] > 0 && e[1] > kbound[bits]) {
                printf("FAIL K relerr at %d bits (%.4f > %.4f)\n", bits, e[1], kbound[bits]);
                fail = 1;
            }
        }
    }

    /* ---- 4. attention-score agreement, which is what generation actually depends
     * on. Scores are a dot against a query, so a per-channel error that survives the
     * rotation shows up here even when the elementwise error looks respectable.
     *
     * Only the key width appears here: a score is q.k and V never enters it, which
     * is why every shipped preset is K4 and only the value width moves. V's own
     * error is the second column of the round-trip table above. The seed is
     * reset per row so the rows are comparable rather than each drawing fresh keys.
     * K4 is what the presets use; 8 and 2 bracket it to show the trend is monotone,
     * not because anything can select them. */
    printf("\nattention scores vs f32 (d=128, 512 keys, 64 queries)\n");
    printf("  key bits   top-1    cosine\n");
    {
        const int kbits[] = {8, 4, 2};
        for (unsigned ci = 0; ci < sizeof kbits / sizeof *kbits; ci++) {
            int d = 128, g = 128, ntile = 4, nk = g * ntile, nq = 64;
            rs = 88172645463325252ULL;
            Kvarn qk; kvarn_init(&qk, d, kbits[ci], g, 1);
            float *keys = malloc(sizeof(float) * (size_t)nk * d);
            float *deq  = malloc(sizeof(float) * (size_t)nk * d);
            uint8_t *rec = malloc(qk.bytes * ntile);
            make_kv(keys, nk, d, d);
            for (int t = 0; t < ntile; t++)
                kvarn_encode_tile(&qk, keys + (size_t)t * g * d, d, g,
                                  rec + (size_t)t * qk.bytes);
            for (int i = 0; i < nk; i++)
                kvarn_decode_vec(&qk, rec + (size_t)(i / g) * qk.bytes, i % g,
                                 deq + (size_t)i * d);

            int hit = 0;
            double cos_acc = 0;
            float *q = malloc(sizeof(float) * d);
            double *s0 = malloc(sizeof(double) * nk), *s1 = malloc(sizeof(double) * nk);
            for (int j = 0; j < nq; j++) {
                for (int i = 0; i < d; i++) q[i] = gauss();
                int a0 = 0, a1 = 0;
                double dot = 0, n0 = 0, n1 = 0;
                for (int i = 0; i < nk; i++) {
                    double x = 0, y = 0;
                    for (int c = 0; c < d; c++) {
                        x += (double)q[c] * keys[(size_t)i * d + c];
                        y += (double)q[c] * deq[(size_t)i * d + c];
                    }
                    s0[i] = x; s1[i] = y;
                    if (x > s0[a0]) a0 = i;
                    if (y > s1[a1]) a1 = i;
                    dot += x * y; n0 += x * x; n1 += y * y;
                }
                hit += (a0 == a1);
                cos_acc += dot / (sqrt(n0 * n1) + 1e-12);
            }
            printf("  %-9d  %5.1f%%   %.5f\n", kbits[ci],
                   100.0 * hit / nq, cos_acc / nq);
            /* Cosine is the assertion, not top-1: with 512 synthetic keys the top
             * two scores are routinely within noise of each other, so top-1 swings
             * several points between seeds while cosine does not. Upstream makes the
             * same point from the other side -- a high score similarity does not by
             * itself prove generation works, so treat both as regression tripwires
             * rather than as accuracy claims. */
            if (kbits[ci] >= 4 && cos_acc / nq < 0.99) {
                printf("FAIL attention cosine at K%d\n", kbits[ci]);
                fail = 1;
            }
            free(keys); free(deq); free(rec); free(q); free(s0); free(s1);
            kvarn_free(&qk);
        }
    }

    /* ---- 5. a short tile must still decode its real tokens */
    {
        int d = 128, g = 128, n = 37;
        Kvarn q; kvarn_init(&q, d, 4, g, 0);
        float *tile = malloc(sizeof(float) * (size_t)g * d);
        float *out = malloc(sizeof(float) * d);
        uint8_t *rec = malloc(q.bytes);
        make_kv(tile, n, d, d);
        kvarn_encode_tile(&q, tile, d, n, rec);
        double worst = 0;
        for (int t = 0; t < n; t++) {
            kvarn_decode_vec(&q, rec, t, out);
            worst = fmax(worst, relerr(tile + (size_t)t * d, out, d));
        }
        printf("\npartial tile (%d of %d tokens): worst relerr %.4f\n", n, g, worst);
        if (worst > 0.15) { printf("FAIL partial tile\n"); fail = 1; }
        free(tile); free(out); free(rec); kvarn_free(&q);
    }

    /* ---- 6. the ring geometry the models depend on */
    {
        int bad = 0;
        for (int g = 32; g <= 256; g <<= 1)
            for (int rw = 1; rw <= 1024; rw++) {
                int W = kvarn_window(rw, g);
                /* whole tiles, never narrower than asked, never wastefully wider */
                if (W % g || W < rw || W < g || W - rw >= g) bad = 1;
            }
        for (int g = 32; g <= 256; g <<= 1)
            for (int cap = g; cap <= 8192; cap++) {
                /* every window of `cap` live positions must fit in the tile ring */
                int need = (cap - 1) / g + 2;
                if (kvarn_ntiles(cap, g) < need) bad = 1;
            }
        printf("\nring geometry: %s\n", bad ? "BROKEN" : "ok");
        if (bad) { printf("FAIL ring geometry\n"); fail = 1; }
    }

    /* ---- 7. the ring discipline the models run, end to end.
     *
     * A replica of kv_write/kv_read: an f32 ring of kvarn_window() positions in
     * front of a tile store of kvarn_ntiles(), sealing the outgoing tile on every
     * tile boundary. Every live position has to read back, from the ring while it is
     * young and from the tile store afterwards, with no hole at the handover and no
     * aliasing when the store wraps. A sliding cap that is not a multiple of the tile
     * is the case that finds aliasing bugs, so that is the one tested. */
    {
        int d = 64, g = 64, rwin = 100, cap = 1042, hd = d;
        int W = kvarn_window(rwin, g), nt = kvarn_ntiles(cap, g);
        Kvarn qk; kvarn_init(&qk, d, 4, g, 1);
        float *ring = malloc(sizeof(float) * (size_t)W * hd);
        int *rpos = malloc(sizeof(int) * W);
        uint8_t *store = malloc(qk.bytes * nt);
        float *truth = malloc(sizeof(float) * (size_t)cap * hd);   /* last cap positions */
        float out[64];
        for (int i = 0; i < W; i++) rpos[i] = -1;

        double worst = 0;
        int checked = 0, holes = 0;
        for (int pos = 0; pos < 5000; pos++) {
            if (W < cap && pos >= W && pos % g == 0) {
                int base = pos - W, ready = 1;
                for (int i = 0; i < g; i++)
                    if (rpos[(base + i) % W] != base + i) { ready = 0; break; }
                if (ready)
                    kvarn_encode_tile(&qk, ring + (size_t)(base % W) * hd, hd, g,
                                      store + (size_t)((base / g) % nt) * qk.bytes);
            }
            rpos[pos % W] = pos;
            float *slot = ring + (size_t)(pos % W) * hd;
            for (int i = 0; i < hd; i++) slot[i] = gauss() * (1.0f + (i % 7));
            memcpy(truth + (size_t)(pos % cap) * hd, slot, sizeof(float) * hd);

            /* read every position this step is allowed to attend to */
            if (pos % 97) continue;
            int lo = pos - cap + 1 < 0 ? 0 : pos - cap + 1;
            for (int t = lo; t <= pos; t++) {
                const float *want = truth + (size_t)(t % cap) * hd;
                if (rpos[t % W] == t) {
                    if (memcmp(ring + (size_t)(t % W) * hd, want, sizeof(float) * hd))
                        holes++;
                    continue;
                }
                kvarn_decode_vec(&qk, store + (size_t)((t / g) % nt) * qk.bytes,
                                 t % g, out);
                double e = relerr(want, out, hd);
                if (e > worst) worst = e;
                if (e > 0.25) holes++;
                checked++;
            }
        }
        printf("\nring discipline (ring %d, cap %d, tile %d, %d tiles): "
               "%d quantised reads, worst relerr %.4f, %d holes\n",
               W, cap, g, nt, checked, worst, holes);
        if (holes || checked < 10000) { printf("FAIL ring discipline\n"); fail = 1; }
        free(ring); free(rpos); free(store); free(truth); kvarn_free(&qk);
    }

    /* ---- 8. the identity the engines' attention loops rely on.
     *
     * They never call kvarn_decode_vec: they rotate the query once, run the whole
     * softmax against raw (still-rotated) keys and values, and transform the
     * accumulated output once. That is only sound because H is orthonormal and
     * symmetric, which makes q.k invariant and lets the transform commute with the
     * weighted sum. Both halves are checked against the same attention done the
     * naive way, since an error in either is silent: it degrades output quality
     * without ever crashing. */
    {
        int d = 128, g = 128, nk = g * 3;
        Kvarn qk, qv;
        kvarn_init(&qk, d, 4, g, 1);
        kvarn_init(&qv, d, 2, g, 0);
        float *keys = malloc(sizeof(float) * (size_t)nk * d);
        uint8_t *rk = malloc(qk.bytes * 3), *rv = malloc(qv.bytes * 3);
        make_kv(keys, nk, d, d);
        for (int t = 0; t < 3; t++) {
            kvarn_encode_tile(&qk, keys + (size_t)t * g * d, d, g, rk + (size_t)t * qk.bytes);
            kvarn_encode_tile(&qv, keys + (size_t)t * g * d, d, g, rv + (size_t)t * qv.bytes);
        }
        float q[128], qr[128], kb[128], vb[128];
        KvarnPlanes pk, pv; pk.rec = pv.rec = NULL;
        float o_model[128] = {0}, o_rot[128] = {0};
        for (int i = 0; i < d; i++) q[i] = gauss();
        memcpy(qr, q, sizeof q);
        kvarn_rot(qr, d);

        double worst_score = 0, score_scale = 0;
        double m0 = -INFINITY, z0 = 0, m1 = -INFINITY, z1 = 0;
        for (int t = 0; t < nk; t++) {
            const uint8_t *K = rk + (size_t)(t / g) * qk.bytes;
            const uint8_t *V = rv + (size_t)(t / g) * qv.bytes;
            /* the naive way: bring every position back to the model frame */
            kvarn_decode_vec(&qk, K, t % g, kb);
            kvarn_decode_vec(&qv, V, t % g, vb);
            double s0 = 0;
            for (int i = 0; i < d; i++) s0 += (double)q[i] * kb[i];
            double nm = s0 > m0 ? s0 : m0, a = exp(m0 - nm), w = exp(s0 - nm), nz = a * z0 + w;
            double old = z0 ? a * z0 / nz : 0.0, add = w / nz;
            for (int i = 0; i < d; i++) o_model[i] = (float)(old * o_model[i] + add * vb[i]);
            m0 = nm; z0 = nz;
            /* the engines' way: stay in the rotated frame throughout */
            kvarn_decode_raw(&qk, K, t % g, &pk, kb);
            kvarn_decode_raw(&qv, V, t % g, &pv, vb);
            double s1 = 0;
            for (int i = 0; i < d; i++) s1 += (double)qr[i] * kb[i];
            /* Against the SCALE of the scores, not each score's own magnitude: the
             * softmax only cares about differences, and a score that happens to land
             * near zero would otherwise report a huge relative error for an absolute
             * one of no consequence. */
            worst_score = fmax(worst_score, fabs(s1 - s0));
            score_scale = fmax(score_scale, fabs(s0));
            nm = s1 > m1 ? s1 : m1; a = exp(m1 - nm); w = exp(s1 - nm); nz = a * z1 + w;
            old = z1 ? a * z1 / nz : 0.0; add = w / nz;
            for (int i = 0; i < d; i++) o_rot[i] = (float)(old * o_rot[i] + add * vb[i]);
            m1 = nm; z1 = nz;
        }
        kvarn_rot(o_rot, d);                    /* one transform for the whole sum */
        double eo = relerr(o_model, o_rot, d);
        double es = worst_score / (score_scale + 1e-12);
        printf("\nrotated-frame attention: score err %.2e of scale, output rel err %.2e\n",
               es, eo);
        if (es > 1e-5 || eo > 1e-5) { printf("FAIL rotated frame\n"); fail = 1; }
        free(keys); free(rk); free(rv); kvarn_free(&qk); kvarn_free(&qv);
    }

    printf("\n%s\n", fail ? "FAILURES" : "all good");
    return fail;
}
