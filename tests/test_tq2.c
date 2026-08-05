/* test_tq2.c — dependency-free checks for Maple's two weight formats.
 *
 * The SIMD kernels in tq2.h are the whole point of the format, so every check here
 * diffs them against an independent scalar reference written from the FORMAT
 * DESCRIPTION rather than from the header's code -- otherwise a packing bug that is
 * consistent between the two would pass.
 *
 * Build: cc -O2 -mavx2 -mfma tests/test_tq2.c -lm -o /tmp/t && /tmp/t
 *    or: cc -O2 -mcpu=apple-m1 tests/test_tq2.c -lm -o /tmp/t && /tmp/t
 *    or: cc -O2                tests/test_tq2.c -lm -o /tmp/t && /tmp/t   (scalar)
 */
#include "../tq2.h"
#include <stdio.h>
#include <stdlib.h>

static uint32_t rs = 12345u;
static float frand(void) {
    rs = rs * 1664525u + 1013904223u;
    return ((float)(rs >> 8) / 8388608.0f - 1.0f);
}
static int irand(int n) {
    rs = rs * 1664525u + 1013904223u;
    return (int)((rs >> 8) % (unsigned)n);
}

/* ---- reference implementations, written from the layout comment ---- */

/* code j of a row lives in byte (j%16) of group (j/64), at bit 2*((j/16)%4) */
static int ref_code(const uint8_t *codes, int j) {
    const uint8_t *grp = codes + (size_t)(j / 64) * 16;
    return (grp[j % 16] >> (2 * ((j / 16) % 4))) & 3;
}
static void ref_pack(uint8_t *codes, const int *c, int I) {
    memset(codes, 0, (size_t)I / 4);
    for (int j = 0; j < I; j++)
        codes[(size_t)(j / 64) * 16 + (j % 16)] |= (uint8_t)(c[j] << (2 * ((j / 16) % 4)));
}
static float ref_tq2_dot(const uint8_t *codes, float alpha, const int8_t *xq,
                         const float *sx, int I) {
    double acc = 0;
    for (int j = 0; j < I; j++)
        acc += (double)(ref_code(codes, j) - 1) * (double)sx[j / Q40_BLK] * (double)xq[j];
    return (float)(acc * alpha);
}

/* q4a: group of 64 = fp16 d, fp16 m, then two 32-blocks of nibbles */
static int ref_q4a_q(const uint8_t *w, int j) {
    const uint8_t *grp = w + (size_t)(j / Q4A_GRP) * Q4A_GRP_BYTES;
    int in = j % Q4A_GRP;
    const uint8_t *nib = grp + 4 + (in / 32) * 16;
    int k = in % 32;
    return k < 16 ? (nib[k] & 0x0f) : (nib[k - 16] >> 4);
}
static void ref_q4a_dm(const uint8_t *w, int j, float *d, float *m) {
    const uint8_t *grp = w + (size_t)(j / Q4A_GRP) * Q4A_GRP_BYTES;
    uint16_t hd, hm;
    memcpy(&hd, grp, 2);
    memcpy(&hm, grp + 2, 2);
    *d = q40_fp16_to_f32(hd);
    *m = q40_fp16_to_f32(hm);
}
static float ref_q4a_dot(const uint8_t *w, const int8_t *xq, const float *sx, int I) {
    double acc = 0;
    for (int j = 0; j < I; j++) {
        float d, m;
        ref_q4a_dm(w, j, &d, &m);
        acc += ((double)d * ref_q4a_q(w, j) + m) * (double)sx[j / Q40_BLK] * xq[j];
    }
    return (float)acc;
}

int main(void) {
    int fail = 0;
    const int dims[] = {64, 128, 512, 1792, 2048};

    for (unsigned t = 0; t < sizeof dims / sizeof *dims; t++) {
        int I = dims[t];
        int *c = malloc(sizeof(int) * I);
        float *x = malloc(sizeof(float) * I);
        float *w = malloc(sizeof(float) * I);
        float *wd = malloc(sizeof(float) * I);
        uint8_t *codes = malloc((size_t)tq2_row_bytes(I));
        int8_t *xq = malloc(I);
        float *sx = malloc(sizeof(float) * (I / Q40_BLK + 8));

        /* ---- 1. ternary dot, int8 activations ---- */
        for (int rep = 0; rep < 200; rep++) {
            for (int j = 0; j < I; j++) c[j] = irand(3);      /* only 0,1,2 exist */
            ref_pack(codes, c, I);
            for (int j = 0; j < I; j++) x[j] = frand();
            float alpha = 0.02f + 0.1f * fabsf(frand());
            float sxtot = tq2_quant_act(x, xq, sx, I);

            float got = tq2_dot(codes, alpha, xq, sx, sxtot, I);
            float want = ref_tq2_dot(codes, alpha, xq, sx, I);
            float tol = fabsf(want) * 1e-5f + 1e-5f;
            if (fabsf(got - want) > tol) {
                printf("FAIL tq2_dot I=%d rep=%d: %g vs %g\n", I, rep, got, want);
                fail = 1;
                break;
            }
        }

        /* ---- 2. the Sx identity: sum(x) recovered from the quantised vector ---- */
        {
            for (int j = 0; j < I; j++) x[j] = frand();
            float sxtot = tq2_quant_act(x, xq, sx, I);
            double direct = 0;
            for (int j = 0; j < I; j++) direct += (double)sx[j / Q40_BLK] * xq[j];
            if (fabsf(sxtot - (float)direct) > fabsf((float)direct) * 1e-5f + 1e-6f) {
                printf("FAIL tq2_quant_act total I=%d: %g vs %g\n", I, sxtot,
                       (float)direct);
                fail = 1;
            }
        }

        /* ---- 3. dequant agrees with the packing, and dot_f32 with dequant ---- */
        {
            for (int j = 0; j < I; j++) c[j] = irand(3);
            ref_pack(codes, c, I);
            float alpha = 0.037f;
            tq2_dequant_row(codes, alpha, w, I);
            for (int j = 0; j < I; j++) {
                float want = alpha * (float)(c[j] - 1);
                if (w[j] != want) {
                    printf("FAIL tq2_dequant I=%d j=%d: %g vs %g\n", I, j, w[j], want);
                    fail = 1;
                    break;
                }
            }
            for (int j = 0; j < I; j++) x[j] = frand();
            double want = 0;
            for (int j = 0; j < I; j++) want += (double)w[j] * x[j];
            float got = tq2_dot_f32(codes, alpha, x, I);
            if (fabsf(got - (float)want) > fabsf((float)want) * 1e-5f + 1e-5f) {
                printf("FAIL tq2_dot_f32 I=%d: %g vs %g\n", I, got, (float)want);
                fail = 1;
            }
        }

        /* ---- 4. q4a dot and dequant ---- */
        {
            uint8_t *qw = malloc((size_t)q4a_row_bytes(I));
            for (int g = 0; g < I / Q4A_GRP; g++) {
                uint8_t *grp = qw + (size_t)g * Q4A_GRP_BYTES;
                uint16_t hd = q40_f32_to_fp16(0.001f + 0.02f * fabsf(frand()));
                uint16_t hm = q40_f32_to_fp16(-0.15f * fabsf(frand()));
                memcpy(grp, &hd, 2);
                memcpy(grp + 2, &hm, 2);
                for (int k = 0; k < 32; k++) grp[4 + k] = (uint8_t)(irand(16) | (irand(16) << 4));
            }
            for (int rep = 0; rep < 50; rep++) {
                for (int j = 0; j < I; j++) x[j] = frand();
                q40_quant_act(x, xq, sx, I);
                float got = q4a_dot(qw, xq, sx, I);
                float want = ref_q4a_dot(qw, xq, sx, I);
                if (fabsf(got - want) > fabsf(want) * 1e-4f + 1e-4f) {
                    printf("FAIL q4a_dot I=%d rep=%d: %g vs %g\n", I, rep, got, want);
                    fail = 1;
                    break;
                }
            }
            q4a_dequant_row(qw, wd, I);
            for (int j = 0; j < I; j++) {
                float d, m;
                ref_q4a_dm(qw, j, &d, &m);
                float want = d * (float)ref_q4a_q(qw, j) + m;
                if (fabsf(wd[j] - want) > 1e-6f) {
                    printf("FAIL q4a_dequant I=%d j=%d: %g vs %g\n", I, j, wd[j], want);
                    fail = 1;
                    break;
                }
            }
            for (int j = 0; j < I; j++) x[j] = frand();
            double want = 0;
            for (int j = 0; j < I; j++) want += (double)wd[j] * x[j];
            float got = q4a_dot_f32(qw, x, I);
            if (fabsf(got - (float)want) > fabsf((float)want) * 1e-5f + 1e-5f) {
                printf("FAIL q4a_dot_f32 I=%d: %g vs %g\n", I, got, (float)want);
                fail = 1;
            }
            free(qw);
        }

        /* ---- 5. sizes ---- */
        if (tq2_row_bytes(I) != I / 4 ||
            tq2_tensor_bytes(7, I) != 7 * 4 + 7 * (I / 4) ||
            q4a_row_bytes(I) != (I / 64) * 36) {
            printf("FAIL size arithmetic I=%d\n", I);
            fail = 1;
        }

        free(c); free(x); free(w); free(wd); free(codes); free(xq); free(sx);
    }

    printf("%s\n", fail ? "test_tq2 FAILED" : "test_tq2 ok");
    return fail;
}
