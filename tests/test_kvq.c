/* test_kvq.c — validate TurboQuant V3 against the paper's bounds and reproduce the
 * upstream attention-fidelity table on synthetic data.
 * Build: cc -O2 -I. tests/test_kvq.c -lm -o /tmp/tk && /tmp/tk
 */
#include "kvq.h"
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

static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? 1 : x > y ? -1 : 0;
}

int main(void) {
    int fail = 0;

    /* ---- 1. RHT must be exactly orthonormal (else everything downstream is wrong) */
    {
        int d = 256;
        Kvq k; kvq_init(&k, d, 4, 12345);
        float *x = malloc(4 * d), *y = malloc(4 * d);
        for (int i = 0; i < d; i++) x[i] = gauss();
        memcpy(y, x, 4 * d);
        kvq_rht(y, k.sign, d);
        double n0 = 0, n1 = 0;
        for (int i = 0; i < d; i++) { n0 += (double)x[i] * x[i]; n1 += (double)y[i] * y[i]; }
        kvq_irht(y, k.sign, d);
        double err = 0;
        for (int i = 0; i < d; i++) err = fmax(err, fabs(x[i] - y[i]));
        printf("RHT: norm preserved %.9f -> %.9f ; roundtrip max err %.2e\n",
               sqrt(n0), sqrt(n1), err);
        if (fabs(sqrt(n0) - sqrt(n1)) > 1e-3 || err > 1e-4) { printf("FAIL rht\n"); fail = 1; }
        free(x); free(y); kvq_free(&k);
    }

    /* ---- 2. MSE vs the paper's upper bound (README table: d=128, unit vectors) */
    printf("\nMSE distortion (d=128, 1000 random unit vectors)\n");
    printf("  bits   measured   paper bound   ratio\n");
    const double bound[9] = {0, 0.680, 0.170, 0.043, 0.011, 0, 0, 0, 0};
    for (int bits = 1; bits <= 4; bits++) {
        int d = 128;
        Kvq k; kvq_init(&k, d, bits, 999);
        uint8_t *buf = malloc(k.bytes);
        float *x = malloc(4 * d), *y = malloc(4 * d);
        double mse = 0;
        for (int t = 0; t < 1000; t++) {
            double n = 0;
            for (int i = 0; i < d; i++) { x[i] = gauss(); n += (double)x[i] * x[i]; }
            n = sqrt(n);
            for (int i = 0; i < d; i++) x[i] /= (float)n;      /* unit vector */
            kvq_encode(&k, x, buf);
            kvq_decode(&k, buf, y);
            double e = 0;
            for (int i = 0; i < d; i++) e += (double)(x[i]-y[i]) * (x[i]-y[i]);
            mse += e;
        }
        mse /= 1000;
        printf("  %d-bit  %8.4f   %8.4f   %.2fx %s\n", bits, mse, bound[bits],
               mse / bound[bits], mse < bound[bits] ? "" : "  <-- ABOVE BOUND");
        if (mse > bound[bits]) { fail = 1; }
        free(buf); free(x); free(y); kvq_free(&k);
    }

    /* ---- 3. attention fidelity: the upstream table's actual metric.
     * d=128 keys, 8K context, compare softmax(q.K) against softmax(q.K_hat). */
    printf("\nattention fidelity (d=128, 8192 keys, 200 queries)\n");
    printf("  config                       cos sim   top-1   top-5   compression\n");
    struct { int kb, vb; const char *name; } cfg[] = {
        {8, 8, "K8/V8"}, {6, 4, "K6/V4  (upstream: EXACT)"},
        {4, 4, "K4/V4"}, {4, 2, "K4/V2  (upstream: MISS)"},
        {3, 3, "K3/V3 uniform"},
    };
    int T = 8192, d = 128, NQ = 200;
    float *K = malloc((size_t)4 * T * d);
    for (size_t i = 0; i < (size_t)T * d; i++) K[i] = gauss();

    for (unsigned c = 0; c < sizeof cfg / sizeof *cfg; c++) {
        Kvq kk; kvq_init(&kk, d, cfg[c].kb, 4242);
        uint8_t *enc = malloc(kk.bytes * T);
        float *Kh = malloc((size_t)4 * T * d);
        for (int t = 0; t < T; t++) {
            kvq_encode(&kk, K + (size_t)t * d, enc + (size_t)t * kk.bytes);
            kvq_decode(&kk, enc + (size_t)t * kk.bytes, Kh + (size_t)t * d);
        }
        double cs = 0;
        int t1 = 0, t5 = 0;
        float *q = malloc(4 * d);
        double *a = malloc(8 * T), *b = malloc(8 * T), *sa = malloc(8 * T);
        for (int Q = 0; Q < NQ; Q++) {
            for (int i = 0; i < d; i++) q[i] = gauss();
            double ma = -1e30, mb = -1e30;
            for (int t = 0; t < T; t++) {
                double x = 0, y = 0;
                for (int i = 0; i < d; i++) {
                    x += (double)q[i] * K[(size_t)t*d+i];
                    y += (double)q[i] * Kh[(size_t)t*d+i];
                }
                x /= sqrt(d); y /= sqrt(d);
                a[t] = x; b[t] = y;
                if (x > ma) ma = x;
                if (y > mb) mb = y;
            }
            double za = 0, zb = 0;
            for (int t = 0; t < T; t++) { a[t] = exp(a[t]-ma); b[t] = exp(b[t]-mb); za += a[t]; zb += b[t]; }
            double dot = 0, na = 0, nb = 0;
            for (int t = 0; t < T; t++) {
                a[t] /= za; b[t] /= zb;
                dot += a[t]*b[t]; na += a[t]*a[t]; nb += b[t]*b[t];
            }
            cs += dot / (sqrt(na)*sqrt(nb) + 1e-30);

            /* top-1 / top-5 agreement of the attended keys */
            int ba = 0, bb = 0;
            for (int t = 1; t < T; t++) { if (a[t] > a[ba]) ba = t; if (b[t] > b[bb]) bb = t; }
            t1 += (ba == bb);
            memcpy(sa, a, 8 * T);
            qsort(sa, T, 8, cmpd);
            double thr = sa[4];                  /* 5th largest true score */
            t5 += (a[bb] >= thr);                /* predicted top-1 inside true top-5 */
        }
        double comp = (32.0 * d) / (double)(kk.bits * d + 32);   /* payload+norm vs f32 */
        printf("  %-26s %7.4f  %5.0f%%  %5.0f%%   %6.2fx\n", cfg[c].name,
               cs / NQ, 100.0 * t1 / NQ, 100.0 * t5 / NQ, comp);
        free(enc); free(Kh); free(q); free(a); free(b); free(sa);
        kvq_free(&kk);
    }
    free(K);

    printf("\n%s\n", fail ? "FAILED" : "ok");
    return fail;
}
