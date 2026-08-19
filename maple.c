/* maple.c — DeepGrove Maple 20B-A1B on a small-RAM machine, by streaming experts.
 *
 * The third engine in this repo, sharing the streaming machinery of gemma4.c and
 * lfm25.c: an expert-granular per-layer LRU instead of the OS page cache,
 * batch-union prefill, a learned pin set in usage.bin, and I/O threads. What differs
 * is the model, and here it differs in a way that changes the arithmetic rather than
 * just the shapes.
 *
 * The checkpoint is ternary. Every matrix in the model, the four attention
 * projections and all three expert projections, is 2-bit codes with one scale per
 * output row, and the codes only ever take the values 0, 1, 2, so each weight is
 * -alpha, 0 or +alpha. That is trained, not chosen by a converter. Three things
 * follow:
 *
 *   * There is no precision gradient. lfm25 spends --expert-edge on giving the edge
 *     MoE layers q8_0 experts because routed experts tolerate error the always-on
 *     tensors do not; here both are already at the floor and there is nothing to
 *     trade. The one dial that remains is the FlashHead.
 *   * An expert is 780 KiB: 20 B of parameters in a 4.9 GiB container. At an 8 GB
 *     budget the whole expert set is resident; the streaming cache earns its keep at
 *     4 GB, where about two thirds of it fits.
 *   * The kernel changes shape. With a per-row scale the inner loop reduces to one
 *     integer accumulation, and the -1 offset factors out of the whole matvec into a
 *     single scalar per activation vector. See tq2.h, which is where the interesting
 *     part of this engine lives.
 *
 * The attention is Gemma-shaped rather than Llama-shaped: three sliding-window
 * layers (512) to every full-attention one, RoPE on the sliding layers only and NoPE
 * on the full ones, partial rotary over the first half of each head, and qk-norm.
 * The sliding layers cap their KV at 512 positions no matter the context length,
 * which is why the KV here costs a quarter of what 24 layers suggests.
 *
 * The FlashHead is the one approximation the engine offers, and it is the checkpoint's
 * own: the vocabulary is clustered, decode scores 4748 centroids and then computes
 * exact logits only for the tokens of the top 512 clusters. At 4 bits the lm_head is
 * 175 MiB, the single biggest read of a decode step, and this cuts it to ~24 MiB.
 * Greedy decoding is exact whenever the true argmax lies in a probed cluster.
 *
 * Build:  cc -O3 -march=native -fopenmp maple.c -lm -lpthread -o maple
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <limits.h>
#include <errno.h>
#include <stdatomic.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "q40.h"
#include "tq2.h"
#include "gpu.h"
#include "lfmtok.h"
#include "kvarn.h"
#include "openai_json.h"
#include "openai_http.h"

#define MAXL 64
#define MAXTOPK 16
#define MAXEXPERTS 512

/* tensor formats, as written by tools/convert_maple.py. The values continue
 * lfm25.c's numbering rather than restarting it, so a format number means the same
 * thing everywhere in the repo (1 = q4_0, 2 = q8_0, which this engine never sees). */
#define FMT_F32 0
#define FMT_TQ2 3
#define FMT_Q4A 4
#define FMT_I32 5

/* config */
typedef struct {
    int hidden, n_layers, n_heads, head_dim, n_kv_heads;
    int n_experts, topk, moe_inter;
    int vocab, ctx, slots_per_layer;
    int layer_swa[MAXL];                   /* 1 = sliding window + RoPE, 0 = full + NoPE */
    int norm_topk_prob;
    int rope_dim, sliding_window;
    float eps, rope_theta, mlp_clamp;
    /* FlashHead; n_clusters == 0 means the container has none */
    int flash_n_clusters, flash_cluster_size, flash_n_probes;
    int flash_scaled_centroids, flash_n_force;
} Cfg;

/* A weight: a view into the resident dense blob.
 *
 * For FMT_TQ2 the blob is the row scales followed by the code rows, so the two
 * halves are derived rather than stored -- see tq2.h on why they are split. */
typedef struct { int fmt; int O, I; const uint8_t *q; const float *f; } W;

#define W_ALPHA(w) ((const float *)(w)->q)
#define W_CODES(w) ((w)->q + (size_t)(w)->O * 4)

typedef struct {
    W in_norm, post_norm;                  /* input / post-attention layernorm */
    W q_proj, k_proj, v_proj, o_proj, q_norm, k_norm;
    W router;
} Layer;

typedef struct { int eid; uint64_t used; uint8_t *buf; int pinned, busy; } Slot;

typedef struct {
    Cfg c;
    Layer L[MAXL];
    const uint8_t *dense;   size_t dense_len;
    int efd;                                /* experts.bin */
    /* One expert size for the whole model: every layer is MoE and every expert is
     * ternary, so unlike lfm25 there is no per-layer variation to carry. */
    int64_t esz, gate_b, down_b;
    int64_t *eoff;                          /* [layer*n_experts + eid] -> file offset */

    W embed, lm_head, final_norm;
    W flash_centroids, flash_token_map, flash_cluster_scale, flash_force;
    int use_flash;

    Slot *slots;                            /* [n_layers][slots_per_layer] */
    uint64_t tick;
    int64_t hit, miss;

    /* Learned hot-expert pin set: routing counts persist to usage.bin and the next
     * run pins the top-N per layer into slots the LRU may never evict. */
    int64_t *ucount;
    int npin;
    char usage_path[4096];

    /* KV. Per layer because the cap differs: a sliding layer never holds more than
     * `sliding_window` positions however long the context is, so position t lives at
     * t % cap and the ring aliases only outside the window it is allowed to see. */
    int kvcap[MAXL];
    float **kv_k, **kv_v;
    int **ring_pos;
    uint8_t **pk, **pv;
    Kvarn *qk, *qv;
    int rwin;                               /* f32 ring, in positions: a whole number of tiles */
    int tile;                               /* KVarN tile width, in tokens */

    /* RoPE inverse frequencies, [rope_dim/2]. Partial rotary: only the first
     * rope_dim of each head_dim-wide head rotates, and only on sliding layers. */
    float *inv_freq;

    /* I/O threads */
    pthread_t *io;
    int n_io;
    pthread_mutex_t mu;
    pthread_cond_t cv, done;
    struct { int layer, eid, slot; } *q;
    int qcap, qhead, qtail, qcount, inflight, stop;
} M;

/* Metal, off unless --metal. It is compiled in (metal.mm has tq2 and q4a kernels
 * alongside the q4_0/q8_0 pair the other engines use) but stays opt-in, for the
 * reason gpu.h documents: at 1 B active params over 780 KiB experts, decode is
 * dispatch-latency- and disk-bound, so the GPU only pays off on prefill and on the
 * exact lm_head. Any failure anywhere clears this and the engine runs on the CPU. */
static int g_use_gpu = 0;

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static void *xmalloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 4096, n ? n : 1)) { fprintf(stderr, "OOM %zu\n", n); exit(1); }
    return p;
}

/* COLI_F32ACT picks one of the two dot paths below; the other is legitimately unused. */
#define MAYBE_UNUSED __attribute__((unused))

/* Batched int8 activations.
 *
 * `tot` is what the ternary kernel needs and q4_0 does not: <c-1,x> = <c,x> - sum(x),
 * and sum(x) is the same for every row of every tensor consuming this activation, so
 * it is computed once here rather than per row. */
typedef struct { int8_t *q; float *s; float *tot; } Act;

MAYBE_UNUSED static void act_quant(Act *a, const float *X, int S, int I) {
    int nb = I / Q40_BLK;
    for (int s = 0; s < S; s++)
        a->tot[s] = tq2_quant_act(X + (size_t)s * I, a->q + (size_t)s * I,
                                  a->s + (size_t)s * nb, I);
}

/* manifest */
typedef struct { char name[96]; int64_t off, len; int fmt, O, I; } DEnt;

static W dense_bind(const DEnt *dd, int ndense, const uint8_t *blob, const char *want,
                    int optional) {
    for (int i = 0; i < ndense; i++) {
        if (!strcmp(dd[i].name, want)) {
            W w;
            w.fmt = dd[i].fmt; w.O = dd[i].O; w.I = dd[i].I;
            w.q = blob + dd[i].off;
            w.f = (const float *)(blob + dd[i].off);
            return w;
        }
    }
    if (optional) { W w; memset(&w, 0, sizeof w); return w; }
    fprintf(stderr, "missing dense tensor %s\n", want);
    exit(1);
}

static void manifest(M *m, const char *dir) {
    char p[4096];
    snprintf(p, sizeof p, "%s/manifest.txt", dir);
    FILE *f = fopen(p, "r");
    if (!f) { perror(p); exit(1); }

    Cfg *c = &m->c;
    char line[1024];
    int ndense = 0, di = 0;
    DEnt *dd = NULL;

    while (fgets(line, sizeof line, f)) {
        char k[64];
        if (sscanf(line, "cfg %63s", k) == 1) {
            char *v = line + 4 + strlen(k) + 1;
            if (!strcmp(k, "layer_swa")) {
                int n = 0;
                for (char *t = strtok(v, " \n"); t && n < MAXL; t = strtok(NULL, " \n"))
                    c->layer_swa[n++] = atoi(t);
            }
            else if (!strcmp(k, "hidden"))          c->hidden = atoi(v);
            else if (!strcmp(k, "n_layers"))        c->n_layers = atoi(v);
            else if (!strcmp(k, "n_heads"))         c->n_heads = atoi(v);
            else if (!strcmp(k, "head_dim"))        c->head_dim = atoi(v);
            else if (!strcmp(k, "n_kv_heads"))      c->n_kv_heads = atoi(v);
            else if (!strcmp(k, "n_experts"))       c->n_experts = atoi(v);
            else if (!strcmp(k, "topk"))            c->topk = atoi(v);
            else if (!strcmp(k, "moe_inter"))       c->moe_inter = atoi(v);
            else if (!strcmp(k, "vocab"))           c->vocab = atoi(v);
            else if (!strcmp(k, "ctx"))             c->ctx = atoi(v);
            else if (!strcmp(k, "slots_per_layer")) c->slots_per_layer = atoi(v);
            else if (!strcmp(k, "norm_topk_prob"))  c->norm_topk_prob = atoi(v);
            else if (!strcmp(k, "rope_dim"))        c->rope_dim = atoi(v);
            else if (!strcmp(k, "sliding_window"))  c->sliding_window = atoi(v);
            else if (!strcmp(k, "eps"))             c->eps = atof(v);
            else if (!strcmp(k, "rope_theta"))      c->rope_theta = atof(v);
            else if (!strcmp(k, "mlp_clamp"))       c->mlp_clamp = atof(v);
            else if (!strcmp(k, "flash_n_clusters"))     c->flash_n_clusters = atoi(v);
            else if (!strcmp(k, "flash_cluster_size"))   c->flash_cluster_size = atoi(v);
            else if (!strcmp(k, "flash_n_probes"))       c->flash_n_probes = atoi(v);
            else if (!strcmp(k, "flash_scaled_centroids")) c->flash_scaled_centroids = atoi(v);
            else if (!strcmp(k, "flash_n_force"))        c->flash_n_force = atoi(v);
            continue;
        }
        long long a, bb, cc, eo;
        int li, e, nexp;
        if (sscanf(line, "esz %lld %lld %lld", &a, &bb, &cc) == 3) {
            m->esz = a; m->gate_b = bb; m->down_b = cc;
            continue;
        }
        if (sscanf(line, "ndense %d", &ndense) == 1) {
            dd = calloc(ndense, sizeof *dd); continue;
        }
        if (sscanf(line, "nexpert %d", &nexp) == 1) {
            m->eoff = calloc((size_t)c->n_layers * c->n_experts, sizeof(int64_t));
            continue;
        }
        char nm[96];
        long long off, len; int fmt, O, I;
        if (sscanf(line, "dense %95s %lld %lld %d %d %d", nm, &off, &len, &fmt, &O, &I) == 6) {
            snprintf(dd[di].name, sizeof dd[di].name, "%s", nm);
            dd[di].off = off; dd[di].len = len; dd[di].fmt = fmt;
            dd[di].O = O; dd[di].I = I; di++; continue;
        }
        if (sscanf(line, "expert %d %d %lld", &li, &e, &eo) == 3) {
            m->eoff[(int64_t)li * c->n_experts + e] = eo; continue;
        }
    }
    fclose(f);

    if (c->n_experts > MAXEXPERTS) {
        fprintf(stderr, "n_experts=%d exceeds MAXEXPERTS=%d\n", c->n_experts, MAXEXPERTS);
        exit(1);
    }
    if (c->n_layers > MAXL) {
        fprintf(stderr, "n_layers=%d exceeds MAXL=%d\n", c->n_layers, MAXL);
        exit(1);
    }
    if (c->topk > MAXTOPK || c->topk < 1) {
        fprintf(stderr, "topk=%d out of range (1..%d)\n", c->topk, MAXTOPK);
        exit(1);
    }
    if (c->hidden % TQ2_GRP || c->moe_inter % TQ2_GRP) {
        fprintf(stderr, "hidden=%d and moe_inter=%d must both be multiples of %d\n",
                c->hidden, c->moe_inter, TQ2_GRP);
        exit(1);
    }

    /* dense.bin: read whole -- it is the resident set */
    snprintf(p, sizeof p, "%s/dense.bin", dir);
    int fd = open(p, O_RDONLY);
    if (fd < 0) { perror(p); exit(1); }
    off_t sz = lseek(fd, 0, SEEK_END);
    uint8_t *blob = xmalloc(sz);
    lseek(fd, 0, SEEK_SET);
    for (off_t o = 0; o < sz;) {
        ssize_t r = pread(fd, blob + o, sz - o, o);
        if (r <= 0) { perror("read dense"); exit(1); }
        o += r;
    }
    close(fd);
    m->dense = blob; m->dense_len = sz;
    if (g_use_gpu && !gpu_map(blob, (size_t)sz)) {
        fprintf(stderr, "metal: could not map the dense blob; using CPU\n");
        g_use_gpu = 0;
    }

    m->embed      = dense_bind(dd, ndense, blob, "embed", 0);
    m->lm_head    = dense_bind(dd, ndense, blob, "lm_head", 0);
    m->final_norm = dense_bind(dd, ndense, blob, "final_norm", 0);
    if (c->flash_n_clusters) {
        m->flash_centroids     = dense_bind(dd, ndense, blob, "flash_centroids", 0);
        m->flash_token_map     = dense_bind(dd, ndense, blob, "flash_token_map", 0);
        m->flash_cluster_scale = dense_bind(dd, ndense, blob, "flash_cluster_scale", 0);
        m->flash_force         = dense_bind(dd, ndense, blob, "flash_force", 1);
    }

    char nm[128];
    for (int l = 0; l < c->n_layers; l++) {
        Layer *L = &m->L[l];
        #define B(f, s) do { snprintf(nm, sizeof nm, "layers.%d." s, l); \
                             L->f = dense_bind(dd, ndense, blob, nm, 0); } while (0)
        B(in_norm, "input_layernorm");
        B(post_norm, "post_attention_layernorm");
        B(q_proj, "q_proj"); B(k_proj, "k_proj"); B(v_proj, "v_proj");
        B(o_proj, "o_proj"); B(q_norm, "q_norm"); B(k_norm, "k_norm");
        B(router, "router");
        #undef B
    }
    free(dd);

    snprintf(p, sizeof p, "%s/experts.bin", dir);
    m->efd = open(p, O_RDONLY);
    if (m->efd < 0) { perror(p); exit(1); }

    /* We keep our own expert cache, so page-caching the same bytes double-buffers
     * them. F_NOCACHE is macOS's O_DIRECT analogue; POSIX_FADV_RANDOM stops Linux
     * doing readahead around random reads. */
#if defined(__APPLE__)
    fcntl(m->efd, F_NOCACHE, 1);
    fcntl(m->efd, F_RDAHEAD, 0);
#elif defined(POSIX_FADV_RANDOM)
    posix_fadvise(m->efd, 0, 0, POSIX_FADV_RANDOM);
#endif
}

/* kernels */
/* q . k over one head.
 *
 * Eight accumulators rather than one. A single running sum makes this a serial
 * chain of float adds, four cycles of latency each, which bounds the loop at four
 * cycles an element however wide the vector unit is: 561 ns at head_dim 512 against
 * 52 with the chain split. There is one of these per cached position per head, so
 * at long context it is the largest single cost in a decode. Eight is an AVX
 * vector's width and enough chains to cover the latency, and the compiler folds the
 * inner loop into one FMA. Reordering the sum moves the result by about 1e-6
 * relative, well under both the q4_0 weight error and the KV quantiser's. */
static inline float head_dot(const float *a, const float *b, int n) {
    float s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int i = 0;
    for (; i + 8 <= n; i += 8)
        for (int k = 0; k < 8; k++) s[k] += a[i + k] * b[i + k];
    for (; i < n; i++) s[0] += a[i] * b[i];
    return ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
}

static void rmsnorm(float *o, const float *x, const float *w, int D, float eps) {
    double s = 0;
    for (int i = 0; i < D; i++) s += (double)x[i] * x[i];
    float r = 1.0f / sqrtf((float)(s / D) + eps);
    if (w) for (int i = 0; i < D; i++) o[i] = x[i] * r * w[i];
    else   for (int i = 0; i < D; i++) o[i] = x[i] * r;
}
static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* one weight row . quantised activations */
MAYBE_UNUSED static inline float w_dot(const W *w, int o, const int8_t *xq, const float *sx,
                          float tot) {
    if (w->fmt == FMT_TQ2)
        return tq2_dot(W_CODES(w) + (size_t)o * tq2_row_bytes(w->I), W_ALPHA(w)[o],
                       xq, sx, tot, w->I);
    if (w->fmt == FMT_Q4A)
        return q4a_dot(w->q + (size_t)o * q4a_row_bytes(w->I), xq, sx, w->I);
    const float *f = w->f + (size_t)o * w->I;
    double s = 0;
    for (int i = 0; i < w->I; i++) s += (double)f[i] * (double)sx[i / Q40_BLK] * xq[i];
    return (float)s;
}
/* one weight row . f32 activations (COLI_F32ACT, and f32 tensors in either build) */
static inline float w_dot_f32(const W *w, int o, const float *x) {
    if (w->fmt == FMT_TQ2)
        return tq2_dot_f32(W_CODES(w) + (size_t)o * tq2_row_bytes(w->I),
                           W_ALPHA(w)[o], x, w->I);
    if (w->fmt == FMT_Q4A)
        return q4a_dot_f32(w->q + (size_t)o * q4a_row_bytes(w->I), x, w->I);
    const float *f = w->f + (size_t)o * w->I;
    double s = 0;
    for (int i = 0; i < w->I; i++) s += (double)f[i] * x[i];
    return (float)s;
}

/* y = W x. COLI_F32ACT keeps activations in f32 (weights stay quantised) so --check
 * can separate the int8-activation approximation from an actual bug. */
static void matvec(float *y, const W *w, const float *x, Act *a) {
    /* The Metal kernels consume f32 activations, so they reproduce the w_dot_f32
     * path -- more accurate than the int8 default rather than less. They decline for
     * f32/i32 tensors and for weights that are not GPU-mapped. */
    if (g_use_gpu && gpu_matmul(w->fmt, y, w->q, x, w->O, w->I, 1)) return;
    if (w->fmt == FMT_F32) {
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < w->O; o++) y[o] = w_dot_f32(w, o, x);
        return;
    }
#ifdef COLI_F32ACT
    (void)a;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < w->O; o++) y[o] = w_dot_f32(w, o, x);
#else
    act_quant(a, x, 1, w->I);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < w->O; o++) y[o] = w_dot(w, o, a->q, a->s, a->tot[0]);
#endif
}

/* Y[S,O] = X[S,I] * W^T.
 *
 * For weight reuse, not arithmetic. A matvec streams the whole [O,I] matrix per
 * output vector; here each row is loaded once and dotted against all S activations
 * while it is still in cache. */
static void matmul(float *Y, const W *w, const float *X, int S, Act *a) {
    if (S == 1) { matvec(Y, w, X, a); return; }
    /* One dispatch fills the whole O*S grid, which is where Metal has a chance:
     * prefill is batched and compute-bound. */
    if (g_use_gpu && gpu_matmul(w->fmt, Y, w->q, X, w->O, w->I, S)) return;
    int I = w->I, O = w->O, nb = I / Q40_BLK;
#ifdef COLI_F32ACT
    (void)a; (void)nb;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++)
        for (int s = 0; s < S; s++)
            Y[(size_t)s * O + o] = w_dot_f32(w, o, X + (size_t)s * I);
#else
    if (w->fmt == FMT_F32) {
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++)
            for (int s = 0; s < S; s++)
                Y[(size_t)s * O + o] = w_dot_f32(w, o, X + (size_t)s * I);
        return;
    }
    act_quant(a, X, S, I);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++)
        for (int s = 0; s < S; s++)
            Y[(size_t)s * O + o] = w_dot(w, o, a->q + (size_t)s * I,
                                         a->s + (size_t)s * nb, a->tot[s]);
#endif
}

/* Partial rotary: only the first `D` of each `stride`-wide head rotates, and only on
 * sliding layers (the full-attention layers are NoPE). The dimension loop is outside
 * the head loop on purpose: cos/sin do not depend on the head. */
static void rope(float *x, int H, int stride, int D, int pos, const float *invf) {
    int half = D / 2;
    for (int i = 0; i < half; i++) {
        float a = pos * invf[i], co = cosf(a), si = sinf(a);
        for (int h = 0; h < H; h++) {
            float *v = x + (size_t)h * stride;
            float x1 = v[i], x2 = v[i + half];
            v[i]        = x1 * co - x2 * si;
            v[i + half] = x2 * co + x1 * si;
        }
    }
}

/* expert I/O */
static void slot_read(M *m, int layer, int eid, uint8_t *buf) {
    int64_t off = m->eoff[(int64_t)layer * m->c.n_experts + eid];
    int64_t sz = m->esz;
    for (int64_t o = 0; o < sz;) {
        ssize_t r = pread(m->efd, buf + o, sz - o, off + o);
        if (r <= 0) { fprintf(stderr, "pread expert %d.%d\n", layer, eid); exit(1); }
        o += r;
    }
}
static void *io_worker(void *arg) {
    M *m = arg;
    for (;;) {
        pthread_mutex_lock(&m->mu);
        while (m->qcount == 0 && !m->stop) pthread_cond_wait(&m->cv, &m->mu);
        if (m->stop) { pthread_mutex_unlock(&m->mu); return NULL; }
        int i = m->qhead;
        int layer = m->q[i].layer, eid = m->q[i].eid, slot = m->q[i].slot;
        m->qhead = (m->qhead + 1) % m->qcap;
        m->qcount--;
        Slot *sl = &m->slots[(size_t)layer * m->c.slots_per_layer + slot];
        pthread_mutex_unlock(&m->mu);

        slot_read(m, layer, eid, sl->buf);

        pthread_mutex_lock(&m->mu);
        sl->eid = eid;
        if (--m->inflight == 0) pthread_cond_broadcast(&m->done);
        pthread_mutex_unlock(&m->mu);
    }
}
/* Find or evict a slot for (layer,eid). The LRU is per layer, so an expert only
 * competes with the other experts of its own layer -- under a global page LRU a hot
 * early-layer expert can be evicted by a cold late-layer one. Slots reserved by an
 * in-flight chunk are `busy` and never eligible. */
static int slot_for(M *m, int layer, int eid, int *need_io) {
    int S = m->c.slots_per_layer;
    Slot *base = &m->slots[(size_t)layer * S];
    m->ucount[(size_t)layer * m->c.n_experts + eid]++;      /* learn the hot set */
    for (int i = 0; i < S; i++)
        if (base[i].eid == eid) { base[i].used = ++m->tick; *need_io = 0; m->hit++; return i; }
    int lru = -1;
    for (int i = 0; i < S; i++)
        if (!base[i].pinned && !base[i].busy &&
            (lru < 0 || base[i].used < base[lru].used)) lru = i;
    if (lru < 0) {
        fprintf(stderr, "expert cache exhausted on layer %d: raise --ram or lower --pin\n",
                layer);
        exit(1);
    }
    /* Do not publish the eid yet: the buffer holds the evicted expert until the I/O
     * completes, and the worker sets s->eid once the bytes are actually there. */
    base[lru].eid = -1;
    base[lru].used = ++m->tick;
    *need_io = 1;
    m->miss++;
    return lru;
}

/* forward */
typedef struct {
    float *x, *xn, *q, *k, *v, *o, *tmp;
    float *eout, *h2, *cy;
    Act act;                          /* batched activations for matmul */
    Act eact;                         /* per-expert-row activations */
    Act hact;                         /* expert hidden -> down_proj */
    int moe_nu, moe_chunk_n, moe_prefetch, moe_slots[MAXEXPERTS];
    float *egate;                     /* [S, moe_inter] clamped SwiGLU hidden */
    int *eidx;        /* [S * topk] chosen expert per (row, slot) */
    float *ewt;       /* [S * topk] weight */
    float *rlog;      /* [S * n_experts] router logits, one batched matmul */
    int *rows;        /* scratch: rows routed to one expert */
    float *roww;      /* scratch: their weights */
    int *uniq;        /* distinct experts in the batch */
    /* GPU expert scratch: the rows routed to one expert are scattered through X, so
     * the Metal path gathers them contiguously and scatters the result back. Only
     * allocated when --metal is in effect. */
    float *gx, *gg, *gu, *gd;
    /* FlashHead scratch */
    float *fsim;      /* [n_clusters] centroid scores */
    int *fsel;        /* [n_clusters] index scratch for the top-probe selection */
    int *fids;        /* [n_probes*cluster_size + n_force] candidate token ids */
    int S;
} Buf;

/* Router: pick the top k experts for one row from its precomputed logits.
 *
 * The reference softmaxes all `n_experts` logits, takes the top k of THAT, and
 * renormalises the k it kept. Softmax is monotone, so the top k of the softmax are
 * the top k of the logits, and renormalising them is exactly a softmax over the k
 * logits alone -- the full softmax cancels. So this only has to find the top k, and
 * the exponentials are k of them rather than n_experts.
 *
 * The logits themselves are computed by the caller as one batched f32 matmul, and
 * the selection runs on them directly: the checkpoint sets router_dtype fp32 because
 * near-tied boundaries at top-8 flip a few percent of picks per layer in bf16, and
 * that compounds over 24 layers. */
static void route_row(M *m, const float *logits, int *idx, float *wts) {
    Cfg *c = &m->c;
    int E = c->n_experts, K = c->topk;

    float topv[MAXTOPK];
    for (int j = 0; j < K; j++) { idx[j] = -1; topv[j] = -INFINITY; }
    for (int e = 0; e < E; e++) {
        float sel = logits[e];
        int j = K;
        while (j > 0 && sel > topv[j - 1]) j--;
        if (j == K) continue;
        for (int t = K - 1; t > j; t--) { topv[t] = topv[t - 1]; idx[t] = idx[t - 1]; }
        topv[j] = sel; idx[j] = e;
    }
    /* softmax over the selected logits, shifted by the max for stability */
    float mx = topv[0], sum = 0.0f;
    for (int j = 0; j < K; j++) { wts[j] = expf(topv[j] - mx); sum += wts[j]; }
    if (c->norm_topk_prob)
        for (int j = 0; j < K; j++) wts[j] /= (sum + 1e-20f);
}

/* Apply one loaded expert to every row that routed to it.
 *
 * The loop order matters. Parallelising over token rows -- one thread per token --
 * walks the whole expert per thread, streaming its 780 KiB once per row. Parallelising
 * over output rows instead loads a weight row once and dots it against every
 * activation while it is still in L1: same flops, nrows-fold less bandwidth.
 *
 * The SwiGLU is clamped, which is part of the trained forward pass rather than a
 * numerical guard:  h = silu(min(gate, 7)) * clip(up, -7, 7). */
static void expert_apply(M *m, const uint8_t *blob, const float *X, float *OUT,
                         const int *rows, int nrows, const float *w, Buf *b) {
    Cfg *c = &m->c;
    int D = c->hidden, MI = c->moe_inter;
    float CL = c->mlp_clamp;
    W G = { FMT_TQ2, MI, D, blob, NULL };
    W U = { FMT_TQ2, MI, D, blob + m->gate_b, NULL };
    W Dn = { FMT_TQ2, D, MI, blob + 2 * m->gate_b, NULL };

    /* GPU path. The expert slot was gpu_map'd at allocation, so the weights are read
     * in place; only the participating activation rows move. A decline at any of the
     * three dispatches falls through to the CPU below, which recomputes from X --
     * OUT has not been touched yet, so there is nothing to undo. */
    if (g_use_gpu) {
        for (int r = 0; r < nrows; r++)
            memcpy(b->gx + (size_t)r * D, X + (size_t)rows[r] * D, sizeof(float) * D);
        if (gpu_matmul(FMT_TQ2, b->gg, G.q, b->gx, MI, D, nrows) &&
            gpu_matmul(FMT_TQ2, b->gu, U.q, b->gx, MI, D, nrows)) {
            for (size_t i = 0; i < (size_t)nrows * MI; i++) {
                float g = b->gg[i], u = b->gu[i];
                if (g > CL) g = CL;
                u = u < -CL ? -CL : (u > CL ? CL : u);
                b->gg[i] = silu(g) * u;
            }
            if (gpu_matmul(FMT_TQ2, b->gd, Dn.q, b->gg, D, MI, nrows)) {
                for (int r = 0; r < nrows; r++) {
                    float *out = OUT + (size_t)rows[r] * D;
                    const float *dr = b->gd + (size_t)r * D;
                    for (int o = 0; o < D; o++) out[o] += w[r] * dr[o];
                }
                return;
            }
        }
    }

#ifdef COLI_F32ACT
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < MI; o++)
        for (int r = 0; r < nrows; r++) {
            const float *x = X + (size_t)rows[r] * D;
            float g = w_dot_f32(&G, o, x), u = w_dot_f32(&U, o, x);
            if (g > CL) g = CL;
            u = u < -CL ? -CL : (u > CL ? CL : u);
            b->egate[(size_t)r * MI + o] = silu(g) * u;
        }
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < D; o++)
        for (int r = 0; r < nrows; r++)
            OUT[(size_t)rows[r] * D + o] +=
                w[r] * w_dot_f32(&Dn, o, b->egate + (size_t)r * MI);
#else
    /* quantise each participating activation row once, up front */
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < nrows; r++)
        b->eact.tot[r] = tq2_quant_act(X + (size_t)rows[r] * D,
                                       b->eact.q + (size_t)r * D,
                                       b->eact.s + (size_t)r * (D / Q40_BLK), D);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < MI; o++)
        for (int r = 0; r < nrows; r++) {
            const int8_t *xq = b->eact.q + (size_t)r * D;
            const float *sx = b->eact.s + (size_t)r * (D / Q40_BLK);
            float t = b->eact.tot[r];
            float g = w_dot(&G, o, xq, sx, t), u = w_dot(&U, o, xq, sx, t);
            if (g > CL) g = CL;
            u = u < -CL ? -CL : (u > CL ? CL : u);
            b->egate[(size_t)r * MI + o] = silu(g) * u;
        }
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < nrows; r++)
        b->hact.tot[r] = tq2_quant_act(b->egate + (size_t)r * MI,
                                       b->hact.q + (size_t)r * MI,
                                       b->hact.s + (size_t)r * (MI / Q40_BLK), MI);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < D; o++)
        for (int r = 0; r < nrows; r++)
            OUT[(size_t)rows[r] * D + o] +=
                w[r] * w_dot(&Dn, o, b->hact.q + (size_t)r * MI,
                             b->hact.s + (size_t)r * (MI / Q40_BLK), b->hact.tot[r]);
#endif
}

/* Batch-union MoE. Token-at-a-time prefill reads topk experts per layer per token,
 * but the S rows of a batch collectively route to at most min(n_experts, topk*S)
 * distinct experts, so the loop is inverted: gather expert -> {rows that chose it},
 * read each once, apply it to all of them.
 *
 * Chunking bounds that against slots_per_layer. Acquiring a slot bumps it to
 * most-recently-used and marks it busy, so nothing acquired within a chunk can be
 * evicted by a later acquisition in the same chunk. */
static void moe_wait(M *m) {
    pthread_mutex_lock(&m->mu);
    while (m->inflight) pthread_cond_wait(&m->done, &m->mu);
    pthread_mutex_unlock(&m->mu);
}

static void moe_submit_chunk(M *m, int li, Buf *b, int c0, int cn) {
    pthread_mutex_lock(&m->mu);
    for (int u = 0; u < cn; u++) {
        int need, slot = slot_for(m, li, b->uniq[c0 + u], &need);
        b->moe_slots[c0 + u] = slot;
        m->slots[(size_t)li * m->c.slots_per_layer + slot].busy = 1;
        if (need) {
            if (m->qcount == m->qcap) { fprintf(stderr, "expert I/O queue overflow\n"); exit(1); }
            m->q[m->qtail].layer = li;
            m->q[m->qtail].eid = b->uniq[c0 + u];
            m->q[m->qtail].slot = slot;
            m->qtail = (m->qtail + 1) % m->qcap;
            m->qcount++; m->inflight++;
        }
    }
    if (m->qcount) pthread_cond_broadcast(&m->cv);
    pthread_mutex_unlock(&m->mu);
}

static void moe_release_chunk(M *m, int li, Buf *b, int c0, int cn) {
    pthread_mutex_lock(&m->mu);
    for (int u = 0; u < cn; u++)
        m->slots[(size_t)li * m->c.slots_per_layer + b->moe_slots[c0 + u]].busy = 0;
    pthread_mutex_unlock(&m->mu);
}

/* The router reads the post-attention-normed hidden, which is also the experts'
 * input. As in lfm25 (and unlike gemma4, whose router reads the raw residual) there
 * is no earlier routing signal to prefetch from, and no dense branch sits beside the
 * MoE either, so chunk pipelining is the only latency hiding available. */
static void moe_start(M *m, int li, const float *X, float *out, int S, Buf *b) {
    Cfg *c = &m->c;
    int D = c->hidden, K = c->topk, E = c->n_experts;
    /* One batched matmul for the whole prefill batch, not a matvec per row: the
     * router is a [n_experts, hidden] f32 tensor, so a per-row call would fork an
     * OpenMP team S times per layer to cover 0.5 M flops each. */
    matmul(b->rlog, &m->L[li].router, X, S, &b->act);
    for (int s = 0; s < S; s++)
        route_row(m, b->rlog + (size_t)s * E, b->eidx + s * K, b->ewt + s * K);
    b->moe_nu = 0;
    for (int s = 0; s < S; s++) for (int j = 0; j < K; j++) {
        int e = b->eidx[s * K + j], seen = 0;
        for (int u = 0; u < b->moe_nu; u++) if (b->uniq[u] == e) { seen = 1; break; }
        if (!seen) b->uniq[b->moe_nu++] = e;
    }
    memset(out, 0, sizeof(float) * (size_t)S * D);
    /* moe_finish submits chunk n+1 before applying chunk n, so two chunks are
     * resident at once and need disjoint unpinned slots. Below 2 free slots there is
     * no room for the second, so the overlap is dropped rather than evicting a slot
     * still in use.
     *
     * free_slots/2 bounds a chunk from above; it is not the target. Taking it
     * whenever the union is small makes the overlap disappear: at decode S is 1, so
     * moe_nu is at most topk against many more free slots, the whole union goes out
     * as one chunk, and the layer stalls on the read with nothing computing over it.
     * So aim for two chunks, and fall back to the bound only where it binds. */
    int free_slots = c->slots_per_layer - m->npin;
    b->moe_prefetch = free_slots >= 2;
    if (b->moe_prefetch) {
        int cap = free_slots / 2, half = (b->moe_nu + 1) / 2;
        b->moe_chunk_n = half < cap ? half : cap;
        if (b->moe_chunk_n < 1) b->moe_chunk_n = 1;
    } else {
        b->moe_chunk_n = 1;
    }
    if (b->moe_chunk_n > b->moe_nu) b->moe_chunk_n = b->moe_nu;
    if (b->moe_chunk_n) moe_submit_chunk(m, li, b, 0, b->moe_chunk_n);
}

static void moe_apply_chunk(M *m, int li, const float *X, float *out, int S,
                            Buf *b, int c0, int cn) {
    Cfg *c = &m->c; int K = c->topk, SL = c->slots_per_layer;
    for (int u = 0; u < cn; u++) {
        int e = b->uniq[c0 + u], n = 0;
        for (int s = 0; s < S; s++) for (int j = 0; j < K; j++) if (b->eidx[s*K+j] == e) {
            b->rows[n] = s; b->roww[n] = b->ewt[s*K+j]; n++;
        }
        expert_apply(m, m->slots[(size_t)li * SL + b->moe_slots[c0 + u]].buf,
                     X, out, b->rows, n, b->roww, b);
    }
    moe_release_chunk(m, li, b, c0, cn);
}

static void moe_finish(M *m, int li, const float *X, float *out, int S, Buf *b) {
    int c0 = 0, cn = b->moe_chunk_n;
    while (c0 < b->moe_nu) {
        /* Submit the next chunk before applying this one, so its I/O overlaps the
         * CPU expert_apply. */
        moe_wait(m);
        int n0 = c0 + cn;
        int nn = b->moe_nu - n0;
        if (nn > b->moe_chunk_n) nn = b->moe_chunk_n;
        if (nn > 0 && b->moe_prefetch) moe_submit_chunk(m, li, b, n0, nn);
        moe_apply_chunk(m, li, X, out, S, b, c0, cn);
        if (nn > 0 && !b->moe_prefetch) moe_submit_chunk(m, li, b, n0, nn);
        c0 = n0; cn = nn;
    }
}

/* KV
 * With --kv the f32 ring holds at least the most recent `rwin` positions and older
 * ones are KVarN-encoded on their way out, so recent tokens always attend at full
 * precision. Only full-attention layers are ever quantised: a sliding layer holds at
 * most `sliding_window` positions however long the context is, so there is nothing
 * there to save.
 *
 * KVarN quantises a whole tile of `tile` consecutive tokens at once, so eviction is
 * not per position. The ring is a whole number of tiles (see kvarn_window), tiles
 * are aligned to absolute position, and the tile whose first slot is about to be
 * overwritten is sealed just before that write: exactly once, at the last moment
 * all of its tokens are still resident. */
static void kv_write(M *m, int li, int pos, int nkv, int hd, const float *k,
                     const float *v) {
    int cap = m->kvcap[li];
    int quant = m->pk[li] != NULL;
    int W = quant ? (m->rwin < cap ? m->rwin : cap) : cap;
    int g = m->tile;
    size_t vec = (size_t)nkv * hd;
    int slot = pos % W;

    /* W == cap means the ring is the whole cache and nothing ever falls out of it.
     * That is the only case where W need not be a multiple of the tile, so it is
     * also the case where no sealing may be attempted. */
    if (quant && W < cap && pos >= W && pos % g == 0) {
        int base = pos - W;                  /* first position of the tile going out */
        int ready = 1;
        /* Seal only a tile that is complete and holds the positions it should. A
         * rewritten position would otherwise bake the wrong vector into the store,
         * unrecoverably. */
        for (int i = 0; i < g; i++)
            if (m->ring_pos[li][(base + i) % W] != base + i) { ready = 0; break; }
        if (ready) {
            /* base is a multiple of both W and the tile, so the tile's g positions
             * sit in g contiguous ring slots and each head is a strided walk. */
            int ts = (base / g) % kvarn_ntiles(cap, g);
            size_t off = (size_t)(base % W) * vec;
            for (int h = 0; h < nkv; h++) {
                size_t idx = (size_t)ts * nkv + h;
                kvarn_encode_tile(&m->qk[li], m->kv_k[li] + off + (size_t)h * hd,
                                  vec, g, m->pk[li] + idx * m->qk[li].bytes);
                kvarn_encode_tile(&m->qv[li], m->kv_v[li] + off + (size_t)h * hd,
                                  vec, g, m->pv[li] + idx * m->qv[li].bytes);
            }
        }
    }
    if (quant) m->ring_pos[li][slot] = pos;
    memcpy(m->kv_k[li] + (size_t)slot * vec, k, sizeof(float) * vec);
    memcpy(m->kv_v[li] + (size_t)slot * vec, v, sizeof(float) * vec);
}

static void attn_fwd(M *m, int li, const float *Xn, float *OUT, int S,
                     int pos_base, Buf *b) {
    Cfg *c = &m->c;
    Layer *L = &m->L[li];
    int hd = c->head_dim, nkv = c->n_kv_heads, nh = c->n_heads, rep = nh / nkv;
    int swa = c->layer_swa[li];
    int cap = m->kvcap[li];
    int quant = m->pk[li] != NULL;
    int Wr = quant ? (m->rwin < cap ? m->rwin : cap) : cap;
    int g = m->tile, ntile = kvarn_ntiles(cap, g);
    size_t vec = (size_t)nkv * hd;
    float scale = 1.0f / sqrtf((float)hd);

    if (hd > 512 || rep > 64) { fprintf(stderr, "attention head geometry too large\n"); exit(1); }

    matmul(b->q, &L->q_proj, Xn, S, &b->act);
    matmul(b->k, &L->k_proj, Xn, S, &b->act);
    matmul(b->v, &L->v_proj, Xn, S, &b->act);   /* V is neither normed nor roped */

    /* Publish and attend one position at a time rather than writing the whole batch
     * first. On a sliding layer the ring is exactly `sliding_window` long, so
     * publishing position p+127 of a batch would evict p-385 -- which row 0 of that
     * same batch still needs. Interleaving is correct on both kinds of layer, since
     * row s only ever reads t <= pos_s, and it costs nothing: the projections above
     * are already batched and this loop is the attention itself. */
    for (int s = 0; s < S; s++) {
        int pos = pos_base + s;
        float *q = b->q + (size_t)s * nh * hd;
        float *k = b->k + (size_t)s * vec;
        float *v = b->v + (size_t)s * vec;
        for (int i = 0; i < nh; i++)
            rmsnorm(q + (size_t)i * hd, q + (size_t)i * hd, L->q_norm.f, hd, c->eps);
        for (int i = 0; i < nkv; i++)
            rmsnorm(k + (size_t)i * hd, k + (size_t)i * hd, L->k_norm.f, hd, c->eps);
        /* RoPE on the sliding layers only; the full-attention layers are NoPE. */
        if (swa) {
            rope(q, nh, hd, c->rope_dim, pos, m->inv_freq);
            rope(k, nkv, hd, c->rope_dim, pos, m->inv_freq);
        }
        /* fold the attention scale into q once rather than into every score */
        for (int i = 0; i < nh * hd; i++) q[i] *= scale;
        /* Meet the cache in its own frame: on a compressed layer everything the
         * attention loop below reads is Hadamard-rotated, so rotating q once here
         * and the output once at the end replaces a transform per cached position.
         * K and V still go into the ring unrotated, which is the model's frame and
         * what the encoder expects. */
        if (quant)
            for (int i = 0; i < nh; i++) kvarn_rot(q + (size_t)i * hd, hd);
        kv_write(m, li, pos, nkv, hd, k, v);

        int t0 = 0;
        if (swa) {
            t0 = pos - c->sliding_window + 1;
            if (t0 < 0) t0 = 0;
        }
        const float *qrow = q;
        float *orow = b->o + (size_t)s * nh * hd;

        /* One parallel unit per q head when the layer is uncompressed (16-way, and
         * reading the cache in place is free), but per KV head when it is not: a kv
         * head serves `rep` q heads, so owning it lets a thread decode a compressed
         * key once and reuse it instead of `rep` times. */
        int rpu = quant ? rep : 1;
        int nunits = nh / rpu;
        #pragma omp parallel for schedule(static)
        for (int u = 0; u < nunits; u++) {
            int kh = (u * rpu) / rep, r0 = (u * rpu) % rep;
            float kbuf[512], vbuf[512];
            float mx[64], z[64];
            KvarnPlanes pk, pv;
            pk.rec = pv.rec = NULL;
            for (int r = 0; r < rpu; r++) {
                mx[r] = -INFINITY; z[r] = 0.0f;
                memset(orow + (size_t)(kh * rep + r0 + r) * hd, 0, sizeof(float) * hd);
            }
            for (int t = t0; t <= pos; t++) {
                const float *kk, *vv;
                if (!quant) {
                    kk = m->kv_k[li] + (size_t)(t % Wr) * vec + (size_t)kh * hd;
                    vv = m->kv_v[li] + (size_t)(t % Wr) * vec + (size_t)kh * hd;
                } else if (m->ring_pos[li][t % Wr] == t) {
                    /* the ring's few positions join the rotated frame */
                    memcpy(kbuf, m->kv_k[li] + (size_t)(t % Wr) * vec + (size_t)kh * hd,
                           sizeof(float) * hd);
                    memcpy(vbuf, m->kv_v[li] + (size_t)(t % Wr) * vec + (size_t)kh * hd,
                           sizeof(float) * hd);
                    kvarn_rot(kbuf, hd);
                    kvarn_rot(vbuf, hd);
                    kk = kbuf; vv = vbuf;
                } else {
                    size_t idx = (size_t)((t / g) % ntile) * nkv + kh;
                    kvarn_decode_raw(&m->qk[li], m->pk[li] + idx * m->qk[li].bytes,
                                     t % g, &pk, kbuf);
                    kvarn_decode_raw(&m->qv[li], m->pv[li] + idx * m->qv[li].bytes,
                                     t % g, &pv, vbuf);
                    kk = kbuf; vv = vbuf;
                }
                for (int r = 0; r < rpu; r++) {
                    const float *qq = qrow + (size_t)(kh * rep + r0 + r) * hd;
                    float *ov = orow + (size_t)(kh * rep + r0 + r) * hd;
                    float score = head_dot(qq, kk, hd);
                    float nm = score > mx[r] ? score : mx[r];
                    float a = expf(mx[r] - nm), w = expf(score - nm), nz = a * z[r] + w;
                    float old = z[r] ? a * z[r] / nz : 0.0f, add = w / nz;
                    for (int d = 0; d < hd; d++) ov[d] = old * ov[d] + add * vv[d];
                    mx[r] = nm; z[r] = nz;
                }
            }
            /* one transform per head brings the whole weighted sum back */
            if (quant)
                for (int r = 0; r < rpu; r++)
                    kvarn_rot(orow + (size_t)(kh * rep + r0 + r) * hd, hd);
        }
    }
    matmul(OUT, &L->o_proj, b->o, S, &b->act);
}

/* layer
 *
 *     h = h + attention(input_layernorm(h))
 *     h = h + moe(post_attention_layernorm(h))
 *
 * Every layer is MoE (first_k_dense_replace is 0 and there are no shared experts), so
 * as in lfm25 there is no dense-MLP arithmetic beside the expert reads to hide them
 * behind. What remains is chunk n+1's I/O overlapping chunk n's compute, plus
 * batch-union. */
static void layer_fwd(M *m, int li, float *H, int S, int pos_base, Buf *b) {
    Cfg *c = &m->c;
    Layer *L = &m->L[li];
    int D = c->hidden;

    float *Xn = b->xn;
    for (int s = 0; s < S; s++)
        rmsnorm(Xn + (size_t)s * D, H + (size_t)s * D, L->in_norm.f, D, c->eps);
    attn_fwd(m, li, Xn, b->tmp, S, pos_base, b);
    for (int i = 0; i < S * D; i++) H[i] += b->tmp[i];

    float *X = b->eout;
    for (int s = 0; s < S; s++)
        rmsnorm(X + (size_t)s * D, H + (size_t)s * D, L->post_norm.f, D, c->eps);

    moe_start(m, li, X, b->h2, S, b);
    moe_finish(m, li, X, b->h2, S, b);
    for (int i = 0; i < S * D; i++) H[i] += b->h2[i];
}

/* embed one token row. Maple uses a plain nn.Embedding: no embed_scale. */
static void embed_row(M *m, int tok, float *h) {
    int D = m->c.hidden;
    const W *e = &m->embed;
    if (tok < 0 || tok >= e->O) { memset(h, 0, sizeof(float) * D); return; }
    q4a_dequant_row(e->q + (size_t)tok * q4a_row_bytes(D), h, D);
}

/* FlashHead
 *
 * Two phases. Score the `n_clusters` quantised centroids against the final hidden,
 * take the best `n_probes`, and compute exact lm_head logits only for the tokens
 * those clusters contain (plus a fixed set of forced control tokens such as EOS).
 * Every other logit is -inf, so greedy decoding is exact whenever the true argmax
 * lies in a probed cluster, and sampling is exact on the same condition.
 *
 * The win is bandwidth, not flops: the lm_head is 151936 x 2048 at 4 bits = 175 MiB,
 * so a decode step that reads all of it moves more bytes for the head than for the
 * whole rest of the model. Probing 512 of 4748 clusters reads the 5.5 MiB of
 * centroids plus 16384 rows, about 24 MiB. */
static int flash_partition(float *v, int *idx, int n, int k);

static void flash_logits(M *m, const float *hn, float *logits, Buf *b) {
    Cfg *c = &m->c;
    int NC = c->flash_n_clusters, CS = c->flash_cluster_size, NP = c->flash_n_probes;
    int V = c->vocab;

    matvec(b->fsim, &m->flash_centroids, hn, &b->act);
    /* Newer checkpoints fold the per-cluster scale into the centroid rows. When they
     * have not, apply it: centroids are directions, and scaling by the largest member
     * norm upper-bounds a cluster's best logit, so high-frequency small-norm tokens
     * are still probed. */
    if (!c->flash_scaled_centroids) {
        const float *cs = m->flash_cluster_scale.f;
        for (int i = 0; i < NC; i++) b->fsim[i] *= cs[i];
    }

    for (int i = 0; i < NC; i++) b->fsel[i] = i;
    flash_partition(b->fsim, b->fsel, NC, NP);

    const int32_t *tm = (const int32_t *)m->flash_token_map.q;
    int n = 0;
    for (int i = 0; i < NP; i++) {
        const int32_t *row = tm + (size_t)b->fsel[i] * CS;
        for (int j = 0; j < CS; j++) b->fids[n++] = row[j];
    }
    if (m->c.flash_n_force) {
        const int32_t *fo = (const int32_t *)m->flash_force.q;
        for (int j = 0; j < m->c.flash_n_force; j++) b->fids[n++] = fo[j];
    }

    /* No, sorting these ids first does not help, and it was measured rather than
     * assumed. The candidates are scattered over the head's 175 MiB, so making the
     * gather monotonic looked like an obvious win; interleaved against an unsorted
     * build it was worth 30.4 vs 30.3 tok/s, i.e. nothing. A probed row is 1152
     * contiguous bytes, which is already 18 cache lines of sequential burst, so the
     * prefetcher had nothing left to gain from ordering the rows themselves. */

    for (int i = 0; i < V; i++) logits[i] = -INFINITY;
#ifdef COLI_F32ACT
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        int id = b->fids[i];
        if (id >= 0 && id < V) logits[id] = w_dot_f32(&m->lm_head, id, hn);
    }
#else
    act_quant(&b->act, hn, 1, c->hidden);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        int id = b->fids[i];
        if (id >= 0 && id < V)
            logits[id] = w_dot(&m->lm_head, id, b->act.q, b->act.s, b->act.tot[0]);
    }
#endif
}

/* Move the k largest of v[] to the front of idx[] (order within the k is not
 * defined). Quickselect on the index array: at 4748 clusters and k = 512 a full sort
 * would cost more than the centroid matvec it is selecting from. */
static int flash_partition(float *v, int *idx, int n, int k) {
    int lo = 0, hi = n - 1;
    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    while (lo < hi) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        int p = lo + (int)((rng >> 33) % (uint64_t)(hi - lo + 1));
        float pivot = v[idx[p]];
        int t = idx[p]; idx[p] = idx[hi]; idx[hi] = t;
        int store = lo;
        for (int i = lo; i < hi; i++)
            if (v[idx[i]] > pivot) { t = idx[i]; idx[i] = idx[store]; idx[store] = t; store++; }
        t = idx[store]; idx[store] = idx[hi]; idx[hi] = t;
        if (store == k) break;
        if (store < k) lo = store + 1; else hi = store - 1;
    }
    return k;
}

/* Run S tokens from pos_base. logits may be NULL (prefill) or [S,vocab]; the common
 * case, only the last row, goes through `last_only`. */
static void forward_chunk(M *m, const int *ids, int S, int pos_base,
                          float *logits, int last_only, Buf *b) {
    Cfg *c = &m->c;
    int D = c->hidden;

    float *H = b->x;
    for (int s = 0; s < S; s++) embed_row(m, ids[s], H + (size_t)s * D);
    for (int l = 0; l < c->n_layers; l++) layer_fwd(m, l, H, S, pos_base, b);

    if (!logits) return;
    int s0 = last_only ? S - 1 : 0;
    /* The lm_head is by far the widest tensor here, so batch it too when every row's
     * logits are wanted (--check). The FlashHead is single-row by construction and
     * never applies to a batch. */
    if (!last_only && S > 1) {
        for (int s = 0; s < S; s++)
            rmsnorm(b->cy + (size_t)s * D, H + (size_t)s * D, m->final_norm.f, D, c->eps);
        matmul(logits, &m->lm_head, b->cy, S, &b->act);
        return;
    }
    for (int s = s0; s < S; s++) {
        rmsnorm(b->cy, H + (size_t)s * D, m->final_norm.f, D, c->eps);
        float *out = logits + (size_t)(last_only ? 0 : s) * c->vocab;
        if (m->use_flash) flash_logits(m, b->cy, out, b);
        else              matvec(out, &m->lm_head, b->cy, &b->act);
    }
}

/* As forward_chunk, but for an input of any length. Long inputs are processed in
 * chunks of b->S, which is exactly equivalent to one big call -- positions still go
 * through every layer in order and the KV accumulates -- and it bounds the batched
 * projection scratch, which would otherwise scale with the full context. */
static void forward(M *m, const int *ids, int S, int pos_base,
                    float *logits, int last_only, Buf *b) {
    for (int c0 = 0; c0 < S; c0 += b->S) {
        int cn = S - c0 < b->S ? S - c0 : b->S;
        int last = (c0 + cn == S);
        float *lg = !logits ? NULL
                  : last_only ? (last ? logits : NULL)
                  : logits + (size_t)c0 * m->c.vocab;
        forward_chunk(m, ids + c0, cn, pos_base + c0, lg, last_only, b);
    }
}

/* pinning */
typedef struct { int64_t n; int e; } EC;
static int ec_desc(const void *a, const void *b) {
    int64_t x = ((const EC *)a)->n, y = ((const EC *)b)->n;
    return x < y ? 1 : x > y ? -1 : 0;
}

/* Load usage.bin (if any) and pin the top-`npin` experts of each layer. */
static void pin_load(M *m, int npin) {
    Cfg *c = &m->c;
    int S = c->slots_per_layer, E = c->n_experts;
    if (npin <= 0) return;
    if (npin >= S) npin = S - 1;             /* always leave room for a miss */
    m->npin = npin;

    FILE *f = fopen(m->usage_path, "rb");
    if (!f) { m->npin = 0; return; }
    int64_t *u = calloc((size_t)c->n_layers * E, 8);
    size_t want = (size_t)c->n_layers * E;
    if (fread(u, 8, want, f) != want) { free(u); fclose(f); m->npin = 0; return; }
    fclose(f);

    double t0 = now();
    int64_t pinned = 0, tot = 0, hot = 0;
    EC *ec = malloc(sizeof(EC) * E);
    for (int l = 0; l < c->n_layers; l++) {
        for (int e = 0; e < E; e++) { ec[e].n = u[(size_t)l * E + e]; ec[e].e = e; }
        qsort(ec, E, sizeof(EC), ec_desc);
        for (int e = 0; e < E; e++) tot += ec[e].n;
        for (int i = 0; i < npin; i++) {
            if (ec[i].n == 0) break;         /* never routed: pinning it is dead RAM */
            Slot *s = &m->slots[(size_t)l * S + i];
            slot_read(m, l, ec[i].e, s->buf);
            s->eid = ec[i].e;
            s->pinned = 1;
            s->used = ~0ULL;
            hot += ec[i].n;
            pinned++;
        }
    }
    free(ec); free(u);
    fprintf(stderr, "pin: %lld experts (%d/layer) in %.1fs; they took %.1f%% of "
            "past routing\n", (long long)pinned, npin, now() - t0,
            tot ? 100.0 * hot / tot : 0.0);
}

/* Merge this run's counts into usage.bin so the pin set improves over runs. */
static void pin_save(M *m) {
    Cfg *c = &m->c;
    size_t n = (size_t)c->n_layers * c->n_experts;
    int64_t *u = calloc(n, 8);
    FILE *f = fopen(m->usage_path, "rb");
    if (f) { if (fread(u, 8, n, f) != n) memset(u, 0, n * 8); fclose(f); }
    for (size_t i = 0; i < n; i++) u[i] += m->ucount[i];
    f = fopen(m->usage_path, "wb");
    if (f) { fwrite(u, 8, n, f); fclose(f); }
    free(u);
}

/* init */
static void init(M *m, const char *dir, int n_io, int ctx_override, double ram_gb,
                 int kb, int vb, int rwin, int tile) {
    m->tile = tile > 0 ? tile : 128;
    /* KVARN_RWIN is a floor, not the ring size: KVarN seals whole tiles, so the
     * ring is rounded up to a whole number of them (kvarn_window). */
    m->rwin = kvarn_window(rwin > 0 ? rwin : 128, m->tile);
    manifest(m, dir);
    Cfg *c = &m->c;

    /* --ctx overrides what the container was converted with, and may go up: the
     * container's ctx only fixed slots_per_layer, and the weights do not care. */
    int ctx_planned = c->ctx;
    if (ctx_override > 0) c->ctx = ctx_override;
    m->ucount = calloc((size_t)c->n_layers * c->n_experts, 8);
    snprintf(m->usage_path, sizeof m->usage_path, "%s/usage.bin", dir);

    int half = c->rope_dim / 2;
    m->inv_freq = xmalloc(sizeof(float) * (size_t)(half + 1));
    for (int i = 0; i < half; i++)
        m->inv_freq[i] = powf(c->rope_theta, -(float)i / (float)half);

    m->kv_k = calloc(c->n_layers, sizeof(float *));
    m->kv_v = calloc(c->n_layers, sizeof(float *));
    m->ring_pos = calloc(c->n_layers, sizeof(int *));
    m->pk = calloc(c->n_layers, sizeof(uint8_t *));
    m->pv = calloc(c->n_layers, sizeof(uint8_t *));
    m->qk = calloc(c->n_layers, sizeof(Kvarn));
    m->qv = calloc(c->n_layers, sizeof(Kvarn));

    size_t kvb = 0;
    int hd = c->head_dim, nkv = c->n_kv_heads;
    for (int l = 0; l < c->n_layers; l++) {
        /* A sliding layer never needs more than `sliding_window` positions, whatever
         * the context length: this is where three quarters of the KV goes away. */
        int cap = c->layer_swa[l] ? (c->ctx < c->sliding_window ? c->ctx : c->sliding_window)
                                  : c->ctx;
        m->kvcap[l] = cap;
        /* Compression is for the full-attention layers only. A sliding layer is
         * already bounded, so quantising it would buy nothing and cost accuracy. */
        int quant = kb > 0 && !c->layer_swa[l];

        int W = quant ? (m->rwin < cap ? m->rwin : cap) : cap;
        m->kv_k[l] = xmalloc(sizeof(float) * (size_t)W * nkv * hd);
        m->kv_v[l] = xmalloc(sizeof(float) * (size_t)W * nkv * hd);
        m->ring_pos[l] = xmalloc(sizeof(int) * (size_t)W);
        for (int i = 0; i < W; i++) m->ring_pos[l][i] = -1;
        kvb += 2 * sizeof(float) * (size_t)W * nkv * hd;

        if (quant) {
            /* One codec per (layer, K/V). The last argument is which way up the
             * tile is balanced, which kvarn.h's header explains; note that it is
             * not how the codes end up stored. Nothing here is seeded, so an
             * encoder and a decoder built from the same config agree without
             * sharing any state. */
            kvarn_init(&m->qk[l], hd, kb, m->tile, 1);
            kvarn_init(&m->qv[l], hd, vb, m->tile, 0);
            size_t nt = (size_t)kvarn_ntiles(cap, m->tile) * nkv;
            m->pk[l] = xmalloc(m->qk[l].bytes * nt);
            m->pv[l] = xmalloc(m->qv[l].bytes * nt);
            kvb += (m->qk[l].bytes + m->qv[l].bytes) * nt;
        }
    }
    int nswa = 0;
    for (int l = 0; l < c->n_layers; l++) nswa += c->layer_swa[l];
    fprintf(stderr, "kv: %.0f MiB for ctx %d (%d sliding layers capped at %d)",
            kvb / 1048576.0, c->ctx, nswa, c->sliding_window);
    if (kb > 0) fprintf(stderr, " (KVarN K%d/V%d, tile %d, f32 ring %d, "
                        "full-attention layers only)", kb, vb, m->tile, m->rwin);
    fprintf(stderr, "\n");

    /* expert-cache plan
     * slots_per_layer is the only thing the conversion's --ram fixed, and nothing in
     * the container depends on it, so a runtime --ram re-runs the planner of
     * tools/convert_maple.py against figures now known -- the resident dense blob and
     * the KV just allocated. */
    if (ram_gb > 0) {
        int64_t scratch = 192 << 20;
        int64_t avail = (int64_t)(ram_gb * (double)(1LL << 30))
                      - (int64_t)m->dense_len - (int64_t)kvb - scratch;
        int64_t per = avail > 0 ? (avail / m->esz) / c->n_layers : 0;
        if (per > c->n_experts) per = c->n_experts;
        if (per < c->topk) {
            double min_gb = ((double)m->dense_len + (double)kvb + (double)scratch
                             + (double)c->topk * c->n_layers * m->esz) / (double)(1LL << 30);
            fprintf(stderr, "--ram %g GB leaves room for %lld experts per layer, "
                    "below topk=%d: this model needs %.2f GB at this context\n",
                    ram_gb, (long long)per, c->topk, min_gb);
            exit(1);
        }
        c->slots_per_layer = (int)per;
        fprintf(stderr, "ram: %g GB budget -> %d slots/layer "
                "(dense %.0f MiB + kv %.0f MiB + cache %.0f MiB)\n",
                ram_gb, c->slots_per_layer, m->dense_len / 1048576.0,
                kvb / 1048576.0,
                (double)c->slots_per_layer * c->n_layers * m->esz / 1048576.0);
    } else if (ctx_override > 0 && c->ctx > ctx_planned) {
        fprintf(stderr, "note: --ctx %d exceeds the container's plan of %d; pass "
                "--ram to re-plan the expert cache against the larger KV\n",
                c->ctx, ctx_planned);
    }
    if (c->slots_per_layer < c->topk) {
        fprintf(stderr, "slots_per_layer=%d < topk=%d: raise --ram\n",
                c->slots_per_layer, c->topk);
        exit(1);
    }
    {
        size_t ns = (size_t)c->n_layers * c->slots_per_layer;
        m->slots = calloc(ns, sizeof(Slot));
        for (int l = 0; l < c->n_layers; l++)
            for (int i = 0; i < c->slots_per_layer; i++) {
                Slot *s = &m->slots[(size_t)l * c->slots_per_layer + i];
                s->eid = -1;
                s->buf = xmalloc(m->esz);
                /* Map each slot once, up front: the streaming layer overwrites the
                 * bytes but never the address, so the mapping stays valid all run. */
                if (g_use_gpu && !gpu_map(s->buf, (size_t)m->esz)) {
                    fprintf(stderr, "metal: could not map expert slots; using CPU\n");
                    g_use_gpu = 0;
                }
            }
    }

    m->qcap = 2 * c->slots_per_layer;
    if (m->qcap < 2) m->qcap = 2;
    m->q = calloc((size_t)m->qcap, sizeof *m->q);
    if (!m->q) { fprintf(stderr, "OOM expert I/O queue\n"); exit(1); }
    pthread_mutex_init(&m->mu, NULL);
    pthread_cond_init(&m->cv, NULL);
    pthread_cond_init(&m->done, NULL);
    m->n_io = n_io;
    m->io = calloc(n_io, sizeof(pthread_t));
    for (int i = 0; i < n_io; i++) pthread_create(&m->io[i], NULL, io_worker, m);
}

static void act_alloc(Act *a, int Smax, int I) {
    a->q = xmalloc((size_t)Smax * I + 64);
    a->s = xmalloc(sizeof(float) * ((size_t)Smax * (I / Q40_BLK) + 8));
    a->tot = xmalloc(sizeof(float) * ((size_t)Smax + 8));
}

static Buf *bufs(M *m, int Smax) {
    Cfg *c = &m->c;
    int D = c->hidden, K = c->topk, MI = c->moe_inter;
    int qmax = c->n_heads * c->head_dim;
    int kvmax = c->n_kv_heads * c->head_dim;
    /* widest contracted dimension any matmul sees (the lm_head contracts over D, not
     * vocab); the batched activation scratch is sized by it */
    int cmax = D > MI ? D : MI;
    if (qmax > cmax) cmax = qmax;

    Buf *b = calloc(1, sizeof *b);
    b->S = Smax;
    b->x    = xmalloc(sizeof(float) * (size_t)Smax * D);
    b->eout = xmalloc(sizeof(float) * (size_t)Smax * D);   /* post-norm MoE input */
    b->h2   = xmalloc(sizeof(float) * (size_t)Smax * D);   /* MoE output */
    b->xn   = xmalloc(sizeof(float) * ((size_t)Smax * D + 64));
    b->q    = xmalloc(sizeof(float) * ((size_t)Smax * qmax + 64));
    b->k    = xmalloc(sizeof(float) * ((size_t)Smax * kvmax + 64));
    b->v    = xmalloc(sizeof(float) * ((size_t)Smax * kvmax + 64));
    b->o    = xmalloc(sizeof(float) * ((size_t)Smax * qmax + 64));
    b->tmp  = xmalloc(sizeof(float) * ((size_t)Smax * D + 64));
    b->cy   = xmalloc(sizeof(float) * ((size_t)Smax * D + 64));
    act_alloc(&b->act, Smax, cmax);
    act_alloc(&b->eact, Smax * K, D);
    act_alloc(&b->hact, Smax * K, MI);
    b->egate = xmalloc(sizeof(float) * ((size_t)Smax * K * MI + 64));
    b->rlog = xmalloc(sizeof(float) * ((size_t)Smax * c->n_experts + 64));
    b->eidx = xmalloc(sizeof(int) * (size_t)Smax * K);
    b->ewt  = xmalloc(sizeof(float) * (size_t)Smax * K);
    b->rows = xmalloc(sizeof(int) * (size_t)Smax * K);
    b->roww = xmalloc(sizeof(float) * (size_t)Smax * K);
    b->uniq = xmalloc(sizeof(int) * c->n_experts);
    if (g_use_gpu) {
        /* at most every row of the batch routes to a given expert */
        b->gx = xmalloc(sizeof(float) * ((size_t)Smax * D + 64));
        b->gg = xmalloc(sizeof(float) * ((size_t)Smax * MI + 64));
        b->gu = xmalloc(sizeof(float) * ((size_t)Smax * MI + 64));
        b->gd = xmalloc(sizeof(float) * ((size_t)Smax * D + 64));
    }
    if (c->flash_n_clusters) {
        b->fsim = xmalloc(sizeof(float) * (c->flash_n_clusters + 64));
        b->fsel = xmalloc(sizeof(int) * (c->flash_n_clusters + 64));
        size_t nf = (size_t)c->flash_n_probes * c->flash_cluster_size
                  + c->flash_n_force + 64;
        b->fids = xmalloc(sizeof(int) * nf);
    }
    return b;
}

/* sampling */
typedef struct { float p; int i; } PI;
static int pi_desc(const void *a, const void *b) {
    float x = ((const PI *)a)->p, y = ((const PI *)b)->p;
    return x < y ? 1 : x > y ? -1 : 0;
}
/* HF-style repetition penalty over the whole context: every token already in the
 * sequence has its logit divided by pen when positive, multiplied when negative, and
 * only once no matter how often it occurs. -inf logits (the FlashHead's unprobed
 * tokens) stay -inf under either branch. */
static void repetition_penalty(float *logits, int V, const int *ids, int n,
                               float pen, unsigned char *seen) {
    if (pen == 1.0f || n <= 0) return;
    memset(seen, 0, (size_t)V);
    for (int i = 0; i < n; i++) {
        int t = ids[i];
        if ((unsigned)t >= (unsigned)V || seen[t]) continue;
        seen[t] = 1;
        logits[t] = logits[t] > 0 ? logits[t] / pen : logits[t] * pen;
    }
}

/* temperature + top-k + nucleus. Greedy when temp <= 0. top_k is applied before
 * top_p, which is the order HF uses. */
static int sample(const float *logits, int V, float temp, float topp, int topk,
                  PI *buf, uint64_t *rng) {
    if (temp <= 0) {
        int am = 0;
        for (int i = 1; i < V; i++) if (logits[i] > logits[am]) am = i;
        return am;
    }
    float mx = -1e30f;
    for (int i = 0; i < V; i++) if (logits[i] > mx) mx = logits[i];
    double sum = 0;
    for (int i = 0; i < V; i++) {
        /* expf(-inf) is 0, which is what an unprobed FlashHead token should get */
        buf[i].p = expf((logits[i] - mx) / temp);
        buf[i].i = i;
        sum += buf[i].p;
    }
    for (int i = 0; i < V; i++) buf[i].p /= (float)sum;
    qsort(buf, V, sizeof(PI), pi_desc);

    int n = V;
    if (topk > 0 && topk < n) n = topk;           /* top-k first ... */

    double cum = 0;                               /* ... then nucleus within it */
    int mm = n;
    for (int i = 0; i < n; i++) {
        cum += buf[i].p;
        if (cum >= topp) { mm = i + 1; break; }
    }
    n = mm;
    double tot = 0;
    for (int i = 0; i < n; i++) tot += buf[i].p;

    *rng = *rng * 6364136223846793005ULL + 1442695040888963407ULL;
    double r = ((*rng >> 11) / (double)(1ULL << 53)) * tot;
    double acc = 0;
    for (int i = 0; i < n; i++) { acc += buf[i].p; if (r <= acc) return buf[i].i; }
    return buf[n - 1].i;
}

/* ChatML, transcribed from Maple's chat_template.jinja. The generation prompt opens
 * a <think> block unconditionally -- this is a reasoning model and the template
 * gives no way to turn that off, so --nothink suppresses it rather than the reverse. */
static void chat_prompt(char *out, size_t cap, const char *sys,
                        const char *user, int think) {
    size_t n = 0;
    #define ADD(...) n += snprintf(out + n, n < cap ? cap - n : 0, __VA_ARGS__)
    if (sys && *sys) ADD("<|im_start|>system\n%s<|im_end|>\n", sys);
    ADD("<|im_start|>user\n%s<|im_end|>\n", user ? user : "");
    ADD("<|im_start|>assistant\n");
    if (think) ADD("<think>\n");
    #undef ADD
}

/* OpenAI-compatible local server */
typedef struct {
    M *model;
    Buf *buffers;
    LfmTok *tokenizer;
    pthread_mutex_t generation_mu;
    atomic_int cancel;
    int *cached_ids;
    int cached_len;
    int cached_cap;
    int eos, eot, vlimit;
    const char *model_id;
} MapleServerContext;

typedef struct { char *data; size_t len, cap; } MpString;

static int mp_string_append(MpString *s, const char *data, size_t len) {
    if (len > SIZE_MAX - s->len - 1) return 0;
    size_t need = s->len + len + 1;
    if (need > s->cap) {
        size_t cap = s->cap ? s->cap : 256;
        while (cap < need) {
            if (cap > SIZE_MAX / 2) return 0;
            cap *= 2;
        }
        char *p = realloc(s->data, cap);
        if (!p) return 0;
        s->data = p; s->cap = cap;
    }
    memcpy(s->data + s->len, data, len); s->len += len; s->data[s->len] = 0;
    return 1;
}

static int mp_json_escape(MpString *s, const char *text, size_t len) {
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == '"' || c == '\\') { char x[2] = {'\\', (char)c}; if (!mp_string_append(s,x,2)) return 0; }
        else if (c == '\n') { if (!mp_string_append(s,"\\n",2)) return 0; }
        else if (c == '\r') { if (!mp_string_append(s,"\\r",2)) return 0; }
        else if (c == '\t') { if (!mp_string_append(s,"\\t",2)) return 0; }
        else if (c < 0x20) { char x[6] = {'\\','u','0','0',hex[c>>4],hex[c&15]}; if (!mp_string_append(s,x,6)) return 0; }
        else if (!mp_string_append(s,(const char *)&text[i],1)) return 0;
    }
    return 1;
}

static jval *mp_json_field(jval *object, const char *key, jtype type) {
    jval *v = json_get(object, key); return v && v->t == type ? v : NULL;
}

static int mp_build_chat_prompt(jval *messages, MpString *prompt) {
    if (!messages || messages->t != J_ARR) return 0;
    #define PUT(s) do { const char *_s = (s); \
                        if (!mp_string_append(prompt, _s, strlen(_s))) return 0; } while (0)
    for (int i = 0; i < messages->len; i++) {
        jval *message = messages->kids[i];
        jval *role = mp_json_field(message, "role", J_STR);
        jval *content = mp_json_field(message, "content", J_STR);
        if (!role || !content) continue;
        if (strcmp(role->str, "system") && strcmp(role->str, "user") &&
            strcmp(role->str, "assistant")) continue;
        PUT("<|im_start|>");
        PUT(role->str);
        PUT("\n");
        PUT(content->str);
        PUT("<|im_end|>\n");
    }
    /* The template's generation prompt opens a reasoning block unconditionally, and
     * this is a reasoning model, so the server follows it rather than the CLI's
     * --nothink. The reasoning therefore arrives as `content`, with the model's own
     * </think> marking where the answer starts. */
    PUT("<|im_start|>assistant\n<think>\n");
    #undef PUT
    return 1;
}

static int mp_send_chunk(int fd, const char *id, const char *field, const char *text, size_t len) {
    MpString out = {0};
    const char *prefix = "data: {\"id\":\"";
    const char *middle = "\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"";
    const char *suffix = "\":\"";
    const char *end = "\"},\"finish_reason\":null}]}\n\n";
    int ok = mp_string_append(&out,prefix,strlen(prefix)) && mp_json_escape(&out,id,strlen(id)) &&
        mp_string_append(&out,middle,strlen(middle)) && mp_string_append(&out,field,strlen(field)) &&
        mp_string_append(&out,suffix,strlen(suffix)) && mp_json_escape(&out,text,len) &&
        mp_string_append(&out,end,strlen(end));
    if (ok) ok = samosa_send_all(fd, out.data, out.len);
    free(out.data); return ok;
}

static int mp_send_done(int fd, const char *id, int prompt_tokens, int completion_tokens,
                        const char *reason) {
    char event[512];
    int n = snprintf(event, sizeof event,
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}],"
        "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}\n\n"
        "data: [DONE]\n\n", id, reason, prompt_tokens, completion_tokens,
        prompt_tokens + completion_tokens);
    return n > 0 && (size_t)n < sizeof event && samosa_send_all(fd, event, (size_t)n);
}

static int mp_serve_chat(MapleServerContext *ctx, int fd, jval *root) {
    jval *messages = mp_json_field(root, "messages", J_ARR);
    int has_user = 0;
    if (!messages) return samosa_http_json_error(fd,400,"invalid_messages","messages must be an array.");
    for (int i = 0; i < messages->len; i++) {
        jval *msg = messages->kids[i];
        jval *role = mp_json_field(msg,"role",J_STR);
        jval *content = mp_json_field(msg,"content",J_STR);
        if (role && content && !strcmp(role->str,"user")) has_user = 1;
    }
    if (!has_user) return samosa_http_json_error(fd,400,"invalid_messages","A text user message is required.");

    int stream = 0, max_tokens = 2048, seed = 0;
    /* same defaults the CLI resolves: top-k 20 behind the FlashHead, off without */
    int topk = ctx->model->use_flash ? 20 : 0;
    float temperature = 1.0f, topp = 0.95f, penalty = 1.0f;
    jval *v = json_get(root,"stream"); if (v && v->t == J_BOOL) stream = v->boolean;
    v = json_get(root,"max_tokens"); if (!v) v = json_get(root,"max_completion_tokens");
    if (v) {
        if (v->t != J_NUM || v->num < 1 || v->num > 32768 || floor(v->num) != v->num)
            return samosa_http_json_error(fd,400,"invalid_max_tokens","max_tokens must be an integer in 1..32768.");
        max_tokens = (int)v->num;
    }
    v = json_get(root,"temperature");
    if (v) {
        if (v->t != J_NUM || v->num < 0 || v->num > 2)
            return samosa_http_json_error(fd,400,"invalid_temperature","temperature must be in 0..2.");
        temperature = (float)v->num;
    }
    v = json_get(root,"top_p");
    if (v) {
        if (v->t != J_NUM || v->num <= 0 || v->num > 1)
            return samosa_http_json_error(fd,400,"invalid_top_p","top_p must be in (0,1].");
        topp = (float)v->num;
    }
    v = json_get(root,"top_k");
    if (v) {
        if (v->t != J_NUM || v->num < 1 || v->num > 1000 || floor(v->num) != v->num)
            return samosa_http_json_error(fd,400,"invalid_top_k","top_k must be an integer in 1..1000.");
        topk = (int)v->num;
    }
    v = json_get(root,"repetition_penalty");
    if (v) {
        if (v->t != J_NUM || v->num < 0.5 || v->num > 2)
            return samosa_http_json_error(fd,400,"invalid_repetition_penalty","repetition_penalty must be in 0.5..2.");
        penalty = (float)v->num;
    }
    v = json_get(root,"seed");
    if (v) {
        if (v->t != J_NUM || v->num < 0 || floor(v->num) != v->num || v->num > UINT32_MAX)
            return samosa_http_json_error(fd,400,"invalid_seed","seed must be a non-negative integer.");
        seed = (int)v->num;
    }

    M *m = ctx->model; Cfg *c = &m->c;
    MpString prompt = {0};
    if (!mp_build_chat_prompt(messages, &prompt)) {
        free(prompt.data);
        return samosa_http_json_error(fd,400,"invalid_prompt","Unable to construct the chat prompt.");
    }
    int *ids = malloc((size_t)c->ctx * sizeof *ids);
    float *logits = malloc((size_t)c->vocab * sizeof *logits);
    PI *pbuf = malloc((size_t)c->vocab * sizeof *pbuf);
    unsigned char *seen = malloc((size_t)c->vocab);
    if (!ids || !logits || !pbuf || !seen) { free(prompt.data); free(ids); free(logits); free(pbuf); free(seen); return samosa_http_json_error(fd,500,"out_of_memory","Unable to allocate generation buffers."); }
    LfmTok *tok = ctx->tokenizer;
    int np = lfmtok_encode(tok, prompt.data, ids, c->ctx);
    free(prompt.data);
    if (np <= 0) { free(ids); free(logits); free(pbuf); free(seen); return samosa_http_json_error(fd,400,"invalid_prompt","The prompt produced no tokens."); }
    if (np >= c->ctx) {
        char msg[256];
        snprintf(msg, sizeof msg, "The prompt is %d tokens and the context window is "
                 "%d; it leaves no room for a completion.", np, c->ctx);
        free(ids); free(logits); free(pbuf); free(seen);
        return samosa_http_json_error(fd, 400, "context_limit", msg);
    }
    if (max_tokens > c->ctx - np) max_tokens = c->ctx - np;   /* clamp, do not fail */

    pthread_mutex_lock(&ctx->generation_mu);
    atomic_store(&ctx->cancel, 0);
    /* Prefix reuse. Every layer here is attention with a position-addressed KV, so
     * unlike lfm25 (whose conv recurrence cannot be rewound) any common prefix can be
     * kept -- but only up to the sliding window: a sliding layer has already
     * discarded everything older than 512 positions behind what it absorbed, so
     * resuming further back than the last absorbed position is not possible. */
    int common = 0;
    while (common < ctx->cached_len && common < np && ctx->cached_ids[common] == ids[common]) common++;
    if (common > 0 && common == ctx->cached_len && common < np)
        forward(m, ids + common, np - common, common, logits, 1, ctx->buffers);
    else
        forward(m, ids, np, 0, logits, 1, ctx->buffers);

    char id[64]; snprintf(id,sizeof id,"maple-%llu",(unsigned long long)time(NULL));
    if (stream && !samosa_http_stream_headers(fd)) { pthread_mutex_unlock(&ctx->generation_mu); free(ids); free(logits); free(pbuf); free(seen); return 1; }
    MpString answer = {0}; uint64_t rng = seed ? (uint64_t)seed : 0x853c49e6748fea9bULL;
    int generated = 0; const char *reason = "length";
    while (generated < max_tokens && !atomic_load(&ctx->cancel)) {
        repetition_penalty(logits, ctx->vlimit, ids, np + generated, penalty, seen);
        int token = sample(logits, ctx->vlimit, temperature, topp, topk, pbuf, &rng);
        if (token == ctx->eos || token == ctx->eot) { reason = "stop"; break; }
        char piece[4096]; int n = lfmtok_decode(tok, &token, 1, piece, sizeof piece - 1);
        if (n <= 0) { reason = "stop"; break; }
        if (!mp_string_append(&answer, piece, (size_t)n)) { atomic_store(&ctx->cancel,1); break; }
        if (stream && !mp_send_chunk(fd,id,"content",piece,(size_t)n)) { atomic_store(&ctx->cancel,1); break; }
        ids[np + generated++] = token;
        if (generated < max_tokens) forward(m, &token, 1, np + generated - 1, logits, 1, ctx->buffers);
    }
    if (atomic_load(&ctx->cancel)) reason = "cancelled";
    if (stream) mp_send_done(fd,id,np,generated,reason);
    else {
        MpString body={0}; char prefix[512], suffix[512];
        int n=snprintf(prefix,sizeof prefix,"{\"id\":\"%s\",\"object\":\"chat.completion\",\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"",id,ctx->model_id);
        int ok=n>0&&mp_string_append(&body,prefix,(size_t)n)&&mp_json_escape(&body,answer.data?answer.data:"",answer.len);
        n=snprintf(suffix,sizeof suffix,"\"},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",reason,np,generated,np+generated);
        ok=ok&&n>0&&mp_string_append(&body,suffix,(size_t)n)&&samosa_http_headers(fd,200,"application/json",body.len,NULL)&&samosa_send_all(fd,body.data,body.len);
        free(body.data); (void)ok;
    }
    int final_len = np + generated;
    if (final_len > ctx->cached_cap) {
        int cap = ctx->cached_cap ? ctx->cached_cap : 256;
        while (cap < final_len) cap *= 2;
        int *cached = realloc(ctx->cached_ids, (size_t)cap * sizeof *cached);
        if (cached) { ctx->cached_ids = cached; ctx->cached_cap = cap; }
    }
    if (ctx->cached_ids && final_len <= ctx->cached_cap) {
        memcpy(ctx->cached_ids, ids, (size_t)final_len * sizeof *ids);
        ctx->cached_len = final_len;
    }
    free(answer.data); pthread_mutex_unlock(&ctx->generation_mu); free(ids); free(logits); free(pbuf); free(seen); return 0;
}

static int mp_serve_handler(SamosaHttpServer *server, int fd, const SamosaHttpRequest *request, void *opaque) {
    MapleServerContext *ctx = opaque;
    if (!strcmp(request->method,"GET") && !strcmp(request->path,"/healthz"))
        return samosa_http_response(fd,200,"application/json","{\"status\":\"ok\"}",NULL);
    if (!strcmp(request->method,"GET") && !strcmp(request->path,"/v1/models")) {
        char body[512]; snprintf(body,sizeof body,
            "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\","
            "\"owned_by\":\"local\",\"context_length\":%d,\"max_model_len\":%d}]}",
            ctx->model_id, ctx->model->c.ctx, ctx->model->c.ctx);
        return samosa_http_response(fd,200,"application/json",body,NULL);
    }
    if (!strcmp(request->method,"POST") && !strcmp(request->path,"/v1/cancel")) {
        atomic_store(&ctx->cancel,1); samosa_http_response(fd,200,"application/json","{\"cancelled\":true}",NULL);
    }
    if (!strcmp(request->method,"POST") && !strcmp(request->path,"/v1/chat/completions")) {
        char *arena=NULL; jval *root=json_parse(request->body,&arena);
        if (!root || root->t != J_OBJ) { json_free(root); free(arena); return samosa_http_json_error(fd,400,"invalid_json","A JSON object is required."); }
        int result=mp_serve_chat(ctx,fd,root); json_free(root); free(arena); return result;
    }
    if (!strcmp(request->method,"POST") && !strcmp(request->path,"/v1/shutdown")) {
        atomic_store(&ctx->cancel,1); samosa_http_response(fd,200,"application/json","{\"shutting_down\":true}",NULL); samosa_http_server_stop(server); return 1;
    }
    return samosa_http_json_error(fd,404,"not_found","Endpoint not found.");
}

static int run_maple_server(M *m, Buf *buffers, LfmTok *tokenizer, int vlimit,
                            const char *model_id, int port) {
    MapleServerContext ctx={.model=m,.buffers=buffers,.tokenizer=tokenizer,
                            .model_id=model_id,.vlimit=vlimit};
    ctx.eos = lfmtok_id(tokenizer, "<|im_end|>");
    ctx.eot = lfmtok_id(tokenizer, "<|endoftext|>");
    pthread_mutex_init(&ctx.generation_mu,NULL); atomic_init(&ctx.cancel,0);
    SamosaHttpServer server;
    if (!samosa_http_server_init(&server,port,mp_serve_handler,&ctx)) { fprintf(stderr,"server: cannot bind port %d: %s\n",port,strerror(errno)); pthread_mutex_destroy(&ctx.generation_mu); return 1; }
    fprintf(stderr,"[server] OpenAI endpoint ready at http://127.0.0.1:%d\n",server.port); fflush(stderr);
    int ok=samosa_http_server_run(&server); samosa_http_server_destroy(&server);
    free(ctx.cached_ids); pthread_mutex_destroy(&ctx.generation_mu); return ok?0:1;
}

/* main */
/* Numerically diff the Metal kernels against the CPU reference on random data, in
 * both of maple's formats. Whether the kernels are right is a property of the
 * machine they run on, so the check has to travel with the binary.
 *
 * The random tensors are built format-aware rather than as random bytes: an fp16
 * field filled from a PRNG hits inf/NaN often enough that a whole row goes NaN, and
 * a NaN diff compares false against every threshold, so the test would pass without
 * noticing. Scales and biases are therefore drawn as finite floats; only the
 * code/nibble payload is random bytes. */
static int check_gpu(void) {
    if (!gpu_ready()) {
        printf("no Metal device (or built without COLI_METAL) -- nothing to check\n");
        return 0;
    }
    printf("metal device: %s\n\n", gpu_name());
    struct { int fmt, O, I; const char *what; } shp[] = {
        {FMT_TQ2, 2048, 2048, "square proj (tq2)"},
        {FMT_TQ2,  512, 2048, "narrow proj (tq2)"},
        {FMT_TQ2,  512, 2048, "expert gate/up (tq2)"},
        {FMT_TQ2, 2048,  512, "expert down (tq2)"},
        {FMT_Q4A, 4096, 2048, "lm_head slab (q4a)"},
        {FMT_Q4A, 4736, 2048, "flash centroids (q4a)"},
    };
    int fail = 0;
    for (unsigned t = 0; t < sizeof shp / sizeof *shp; t++) {
        int fmt = shp[t].fmt, O = shp[t].O, I = shp[t].I;
        size_t wb = (size_t)(fmt == FMT_TQ2 ? tq2_tensor_bytes(O, I)
                                            : q4a_tensor_bytes(O, I));
        uint8_t *Wt = xmalloc((wb + 4095) & ~(size_t)4095);
        float *x  = xmalloc(sizeof(float) * I);
        float *yg = xmalloc(sizeof(float) * O), *yc = xmalloc(sizeof(float) * O);

        uint64_t r = 0x243f6a8885a308d3ULL ^ t;
        #define NEXT() (r = r * 6364136223846793005ULL + 1442695040888963407ULL)
        #define UNIT() ((float)((int64_t)(NEXT() >> 40) - 8388608) / 8388608.0f)
        for (size_t i = 0; i < wb; i++) Wt[i] = (uint8_t)(NEXT() >> 40);
        if (fmt == FMT_TQ2) {
            /* row scales are f32 and live at the front of the tensor */
            float *al = (float *)Wt;
            for (int o = 0; o < O; o++) al[o] = 0.01f + 0.04f * fabsf(UNIT());
        } else {
            /* one fp16 scale and one fp16 bias per 64-weight group */
            for (int o = 0; o < O; o++) {
                uint8_t *row = Wt + (size_t)o * q4a_row_bytes(I);
                for (int g = 0; g < I / Q4A_GRP; g++) {
                    uint8_t *grp = row + (size_t)g * Q4A_GRP_BYTES;
                    uint16_t hd = q40_f32_to_fp16(0.002f + 0.01f * fabsf(UNIT()));
                    uint16_t hm = q40_f32_to_fp16(0.05f * UNIT());
                    memcpy(grp, &hd, 2);
                    memcpy(grp + 2, &hm, 2);
                }
            }
        }
        for (int i = 0; i < I; i++) x[i] = UNIT();
        #undef UNIT
        #undef NEXT

        W w = { fmt, O, I, Wt, (const float *)Wt };
        if (!gpu_map(Wt, wb)) {
            printf("  %-24s gpu_map FAILED\n", shp[t].what); fail = 1; goto next;
        }
        if (!gpu_matmul(fmt, yg, Wt, x, O, I, 1)) {
            printf("  %-24s gpu_matmul DECLINED\n", shp[t].what); fail = 1; goto next;
        }
        for (int o = 0; o < O; o++) yc[o] = w_dot_f32(&w, o, x);

        double worst = 0, mag = 0;
        for (int o = 0; o < O; o++) {
            double d = fabs((double)yg[o] - (double)yc[o]);
            if (!(d <= worst)) worst = d;         /* NaN-safe: a NaN lands here */
            if (fabs(yc[o]) > mag) mag = fabs(yc[o]);
        }
        double rel = worst / (mag + 1e-9);
        printf("  %-24s [%5d x %5d]  max rel err %.3e  %s\n",
               shp[t].what, O, I, rel, rel < 1e-4 ? "ok" : "MISMATCH");
        if (!(rel < 1e-4)) fail = 1;
    next:
        free(Wt); free(x); free(yg); free(yc);
    }
    printf("\n%s\n", fail ? "GPU CHECK FAILED -- do not pass --metal" : "GPU CHECK PASSED");
    return fail;
}

static void usage(const char *prog, FILE *out) {
    fprintf(out,
        "usage: %s <dir> [flags...] [prompt]\n"
        "         [--chat] [--system S] [--nothink] [--raw] [--max_tokens N]\n"
        "         [--temp F] [--topp F] [--topk N]   (default 1.0 / 0.95 / 20 with\n"
        "                                 the FlashHead, top-k off without it)\n"
        "         [--penalty F]           repetition penalty (default 1, = off)\n"
        "         [--ctx N]               override the container's context length\n"
        "         [--ram F]               re-plan the expert cache for an F GB budget\n"
        "         [--pin N] [--io N] [--threads N] [--batch N] [--nobatch]\n"
        "         [--serve] [--port N]    OpenAI-compatible local server (default 8484)\n"
        "         [--noflash]             exact lm_head instead of the FlashHead\n"
        "         [--probes N]            FlashHead clusters to probe (default: the\n"
        "                                 container's, higher = more exact + slower)\n"
        "         [--kv PRESET]           KVarN KV compression, full-attention layers\n"
        "                                 only; PRESET is one of off | kvarn_k4v2_g128 |\n"
        "                                 kvarn_k4v4_g128 | kvarn_k4v2_g64 |\n"
        "                                 kvarn_k4v4_g64 (default kvarn_k4v2_g128)\n"
        "         [--metal]               offload the matmuls to the GPU (off by\n"
        "                                 default: usually slower here, see matvec)\n"
        "         [--check]               diff against the numpy oracle's logits\n"
        "         [--check-gpu]           diff the Metal kernels against the CPU\n"
        "         [--help]\n",
        prog);
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0], stderr); return 1; }
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(argv[0], stdout); return 0; }

    const char *dir = argv[1];
    const char *prompt = NULL, *sys = NULL;
    int think = 1, raw = 0, chat_mode = 0;
    /* KVarN is ON by default, at upstream's shipped preset, and a preset is all
     * there is: no per-parameter overrides, because the bit widths and the tile are
     * one calibrated recipe upstream measured together. --kv off gives f32 KV. */
    const KvarnPreset *kvp = kvarn_preset(KVARN_DEFAULT);
    int kv_set = 0;
    int check = 0, n_io = 8, max_tokens = 0, nobatch = 0, npin = 0, nthreads = 2;
    int batch = 128, ctx_override = 0, noflash = 0, probes = 0;
    double ram_gb = 0;                   /* 0 = keep the container's own plan */
    int serve_mode = 0, serve_port = 8484;
    int want_metal = 0, chk_gpu = 0;
    float temp = 1.0f, topp = 0.95f, penalty = 1.0f;
    /* -1 = "not given on the command line", resolved against the FlashHead below */
    int topk = -1;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--serve")) serve_mode = 1;
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) serve_port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--check")) check = 1;
        else if (!strcmp(argv[i], "--metal")) want_metal = 1;
        else if (!strcmp(argv[i], "--check-gpu")) chk_gpu = 1;
        else if (!strcmp(argv[i], "--nobatch")) nobatch = 1;
        else if (!strcmp(argv[i], "--noflash")) noflash = 1;
        else if (!strcmp(argv[i], "--probes") && i + 1 < argc) probes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) nthreads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--io") && i + 1 < argc) n_io = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx_override = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ram") && i + 1 < argc) ram_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--max_tokens") && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--topp") && i + 1 < argc) topp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--topk") && i + 1 < argc) topk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--penalty") && i + 1 < argc) penalty = atof(argv[++i]);
        else if (!strcmp(argv[i], "--pin") && i + 1 < argc) npin = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--system") && i + 1 < argc) sys = argv[++i];
        else if (!strcmp(argv[i], "--chat")) chat_mode = 1;
        else if (!strcmp(argv[i], "--nothink")) think = 0;
        else if (!strcmp(argv[i], "--raw")) raw = 1;
        else if (!strcmp(argv[i], "--kv") && i + 1 < argc) {
            const char *v = argv[++i];
            kv_set = 1;
            if (!strcmp(v, "off")) kvp = NULL;
            else if (!(kvp = kvarn_preset(v))) {
                fprintf(stderr, "--kv: expected %s (got %s)\n", kvarn_preset_list(), v);
                return 1;
            }
        }
        /* end of flags: whatever follows is the prompt, even if it starts with '-' */
        else if (!strcmp(argv[i], "--")) { if (i + 1 < argc && !prompt) prompt = argv[++i]; }
        /* Any leading dash, not just a double one. A single-dash typo (-kv for
         * --kv) must not fall through to the positional branch: it would silently
         * become the prompt and change what the model generates. */
        else if (argv[i][0] == '-') {
            fprintf(stderr, "unknown flag: %s\n\n", argv[i]);
            usage(argv[0], stderr);
            return 1;
        }
        else if (!prompt) prompt = argv[i];   /* first non-flag positional is the prompt */
    }
    /* --check diffs the forward pass against a stored oracle to ~1e-4, which is
     * tighter than any KV quantiser reproduces, so it defaults to f32 KV whatever
     * the engine default is. Pass --kv explicitly to measure the codec instead. */
    if (check && !kv_set) kvp = NULL;
    /* kb == 0 is how the rest of the engine spells "f32 KV". */
    int kb = kvp ? kvp->kbits : 0, vb = kvp ? kvp->vbits : 0;
    if (penalty < 0.5f || penalty > 2.0f) { fprintf(stderr, "--penalty must be in 0.5..2\n\n"); usage(argv[0], stderr); return 1; }
    if (chat_mode && check) { fprintf(stderr, "--chat cannot be used with --check\n\n"); usage(argv[0], stderr); return 1; }
    if (chat_mode && serve_mode) { fprintf(stderr, "--chat cannot be used with --serve\n\n"); usage(argv[0], stderr); return 1; }
    if (!serve_mode && !check && max_tokens == 0) max_tokens = 2048;
#ifdef _OPENMP
    if (nthreads > 0) omp_set_num_threads(nthreads);
#else
    (void)nthreads;
#endif
    /* before init(): the dense blob and the expert slots are GPU-mapped as they are
     * allocated, and that only happens if the device came up */
    if (want_metal || chk_gpu) g_use_gpu = gpu_init();
    if (chk_gpu) return check_gpu();
    if (want_metal && !g_use_gpu)
        fprintf(stderr, "metal: no device available; using CPU\n");

    M m; memset(&m, 0, sizeof m);
    double t0 = now();
    init(&m, dir, n_io, ctx_override, ram_gb, kb, vb, KVARN_RWIN,
         kvp ? kvp->group : 128);
    Cfg *c = &m.c;
    if (probes > 0) {
        if (!c->flash_n_clusters) {
            fprintf(stderr, "--probes needs a container built with the FlashHead\n");
            return 1;
        }
        c->flash_n_probes = probes > c->flash_n_clusters ? c->flash_n_clusters : probes;
    }
    /* The FlashHead is an approximation of the lm_head, so --check (which diffs every
     * logit against the oracle) always uses the exact head. */
    m.use_flash = c->flash_n_clusters && !noflash && !check;
    /* Maple's recommended sampling is 1.0 / 0.95 / 20 with the FlashHead, 1.0 / 0.95
     * with top-k off without it. The head is why they differ: it already restricts
     * the candidates to the probed clusters, so top-k 20 is the tail cut on top of
     * that, whereas with the exact head all 151936 logits are real and top-p does
     * the cutting alone. An explicit --topk always wins. */
    if (topk < 0) topk = m.use_flash ? 20 : 0;
    pin_load(&m, npin);
    if (batch < 1) batch = 1;
    if (batch > c->ctx) batch = c->ctx;
    Buf *b = bufs(&m, batch);

    int nswa = 0;
    for (int l = 0; l < c->n_layers; l++) nswa += c->layer_swa[l];
    fprintf(stderr, "maple: %d layers (%d sliding, %d full), %d ternary experts/layer, "
            "top-%d, %d slots/layer, dense %.1f MiB, ready in %.2fs\n",
            c->n_layers, nswa, c->n_layers - nswa, c->n_experts, c->topk,
            c->slots_per_layer, m.dense_len / 1048576.0, now() - t0);
    if (g_use_gpu)
        fprintf(stderr, "metal: %s (tq2/q4a matmul offloaded)\n", gpu_name());
    if (m.use_flash)
        fprintf(stderr, "flash head: probing %d/%d clusters -> %d of %d vocab rows "
                "scored per token\n", c->flash_n_probes, c->flash_n_clusters,
                c->flash_n_probes * c->flash_cluster_size + c->flash_n_force, c->vocab);
    else if (c->flash_n_clusters)
        fprintf(stderr, "flash head: present but disabled; scoring all %d vocab rows\n",
                c->vocab);

    char tp[4096];
    snprintf(tp, sizeof tp, "%s/tok.bin", dir);
    LfmTok *T = lfmtok_load(tp);
    /* The lm_head has vocab_size rows but the tokenizer defines fewer ids; the
     * remainder are padding that can never be decoded, so sampling must not see
     * them. */
    int vlimit = c->vocab;
    if (T && T->n_vocab < vlimit) vlimit = T->n_vocab;

    if (serve_mode) {
        if (!T) { fprintf(stderr, "--serve needs %s\n", tp); return 1; }
        return run_maple_server(&m, b, T, vlimit, "maple-preview", serve_port);
    }

    if (check) {
        char p[4096];
        snprintf(p, sizeof p, "%s/ref.json", dir);
        FILE *f = fopen(p, "r");
        if (!f) { perror(p); return 1; }
        char buf[8192];
        size_t n = fread(buf, 1, sizeof buf - 1, f);
        buf[n] = 0;
        fclose(f);
        char *qs = strstr(buf, "\"prompt\"");
        if (!qs) { fprintf(stderr, "ref.json: no \"prompt\" field\n"); return 1; }
        int ids[512], np = 0;
        for (char *s = strchr(qs, '['); s && *s && *s != ']'; s++)
            if (*s >= '0' && *s <= '9') { ids[np++] = strtol(s, &s, 10); s--; }

        /* The oracle runs on the dequantised container weights: comparing against an
         * fp32 reference would conflate engine bugs with the checkpoint's own
         * quantisation. */
        snprintf(p, sizeof p, "%s/deq_logits.f32", dir);
        f = fopen(p, "rb");
        if (!f) { perror(p); fprintf(stderr, "run tools/maple_oracle.py first\n"); return 1; }
        float *ref = xmalloc(sizeof(float) * (size_t)np * c->vocab);
        if (fread(ref, sizeof(float), (size_t)np * c->vocab, f) != (size_t)np * c->vocab) {
            fprintf(stderr, "short reference logits\n"); return 1;
        }
        fclose(f);

        float *logits = xmalloc(sizeof(float) * (size_t)np * c->vocab);
        double t = now();
        if (nobatch) {
            /* token-at-a-time: the path batch-union has to agree with */
            for (int s = 0; s < np; s++)
                forward(&m, ids + s, 1, s, logits + (size_t)s * c->vocab, 0, b);
        } else {
            forward(&m, ids, np, 0, logits, 0, b);       /* one batched prefill */
        }
        double el = now() - t;

        double worst = 0, den = 0;
        int agree = 0;
        for (int s = 0; s < np; s++) {
            const float *g = logits + (size_t)s * c->vocab;
            const float *r = ref + (size_t)s * c->vocab;
            int am = 0, ar = 0;
            for (int i = 0; i < c->vocab; i++) {
                double d = fabs(g[i] - r[i]);
                if (d > worst) worst = d;
                if (fabs(r[i]) > den) den = fabs(r[i]);
                if (g[i] > g[am]) am = i;
                if (r[i] > r[ar]) ar = i;
            }
            agree += (am == ar);
        }
        long long tot = m.hit + m.miss;
        printf("%s prefill of %d tokens in %.3fs\n",
               nobatch ? "sequential" : "batch-union", np, el);
        printf("teacher forcing: %d/%d argmax agree\n", agree, np);
        printf("max |logit diff| = %.4g   (ref scale %.4g, rel %.3g)\n",
               worst, den, worst / den);
        printf("expert reads: %lld (%lld hits, %lld misses)\n",
               tot, (long long)m.hit, (long long)m.miss);
        /* With exact activations the engine must reproduce the oracle to float
         * precision. The default build quantises activations to int8, which costs
         * ~1e-2 on logits. Argmax must agree either way. */
#ifdef COLI_F32ACT
        double tol = 1e-4;
#else
        double tol = 3e-2;
#endif
        int ok = (agree == np) && (worst / den < tol);
        printf("%s\n", ok ? "PASS" : "FAIL");
        /* Persist the routing counts here too, so that a following --check --pin
         * has a usage.bin to pin from and actually exercises the pinned path
         * instead of silently doing nothing. */
        pin_save(&m);
        return !ok;
    }

    if (max_tokens > 0) {
        float *logits = xmalloc(sizeof(float) * c->vocab);
        PI *pbuf = xmalloc(sizeof(PI) * c->vocab);
        unsigned char *seen = xmalloc((size_t)c->vocab);
        uint64_t rng = 0x853c49e6748fea9bULL;

        int eos = T ? lfmtok_id(T, "<|im_end|>") : -1;
        int eot = T ? lfmtok_id(T, "<|endoftext|>") : -1;

        int *ids = xmalloc(sizeof(int) * c->ctx);
        int np = 0;
        if (prompt && !chat_mode) {
            if (!T) { fprintf(stderr, "prompt needs %s\n", tp); return 1; }
            const char *text = prompt;
            char *chat = NULL;
            if (!raw) {                       /* --raw skips the chat template */
                size_t need = strlen(prompt) + (sys ? strlen(sys) : 0) + 256;
                chat = xmalloc(need);
                chat_prompt(chat, need, sys, prompt, think);
                text = chat;
            }
            if (!lfmtok_nfc_safe(T, text))
                fprintf(stderr, "warning: the prompt is not in NFC; Maple's tokenizer "
                        "normalises first and this engine does not, so the ids may "
                        "differ from the reference implementation's\n");
            np = lfmtok_encode(T, text, ids, c->ctx - max_tokens);
            if (getenv("MAPLEDBG")) fprintf(stderr, "prompt: %s\n[%d tokens]\n", text, np);
            free(chat);
            if (np <= 0) { fprintf(stderr, "empty prompt\n"); return 1; }
        }

        /* interactive multi-turn chat (--chat, or no prompt for compatibility) */
        int interactive = (chat_mode || !prompt);
        int first_prompt = (chat_mode && prompt != NULL);
        int *cached_ids = NULL;
        int cached_len = 0, cached_cap = 0;
        char *history = NULL;                 /* accumulated chat template text */
        size_t hist_len = 0, hist_cap = 0;

        if (interactive) {
            if (!T) { fprintf(stderr, "interactive mode needs %s\n", tp); return 1; }
            fprintf(stderr, "[chat] interactive mode (Ctrl-D to exit)\n");
            fflush(stderr);
        }

        double tpre = 0;

        for (;;) {
            if (interactive) {
                char line[4096];
                const char *user_text;
                size_t len;
                if (first_prompt) {
                    user_text = prompt; len = strlen(user_text); first_prompt = 0;
                } else {
                    fprintf(stdout, "> ");
                    fflush(stdout);
                    if (!fgets(line, sizeof line, stdin)) break;  /* Ctrl-D / EOF */
                    len = strlen(line);
                    while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
                    if (!len) continue;
                    user_text = line;
                }

                size_t need = hist_len + len + 256;
                char *chat = xmalloc(need);
                size_t pos = 0;
                if (hist_len) { memcpy(chat, history, hist_len); pos = hist_len; }
                if (!pos && sys && *sys)
                    pos += snprintf(chat + pos, need - pos,
                                    "<|im_start|>system\n%s<|im_end|>\n", sys);
                pos += snprintf(chat + pos, need - pos, "<|im_start|>user\n%s<|im_end|>\n", user_text);
                pos += snprintf(chat + pos, need - pos, "<|im_start|>assistant\n");
                if (think) pos += snprintf(chat + pos, need - pos, "<think>\n");

                np = lfmtok_encode(T, chat, ids, c->ctx - max_tokens);
                if (getenv("MAPLEDBG")) fprintf(stderr, "prompt: %s\n[%d tokens]\n", chat, np);
                if (np <= 0) { free(chat); continue; }

                if (!hist_cap) { hist_cap = 4096; history = xmalloc(hist_cap); }
                while (hist_cap < pos + 256) { hist_cap *= 2; history = realloc(history, hist_cap); }
                memcpy(history, chat, pos);
                hist_len = pos;
                free(chat);

                int common = 0;
                while (common < cached_len && common < np && cached_ids[common] == ids[common])
                    common++;
                if (common > 0 && common == cached_len && common < np)
                    forward(&m, ids + common, np - common, common, logits, 1, b);
                else
                    forward(&m, ids, np, 0, logits, 1, b);
            } else {
                /* one-shot prefill (prompt given on the command line) */
                double t0p = now();
                forward(&m, ids, np, 0, logits, 1, b);
                tpre = now() - t0p;
            }

            char piece[512];
            int n = 0;
            double t = now();

            while (n < max_tokens) {
                repetition_penalty(logits, vlimit, ids, np + n, penalty, seen);
                int tok = sample(logits, vlimit, temp, topp, topk, pbuf, &rng);
                if (tok == eos || tok == eot) break;
                if (T) { lfmtok_decode(T, &tok, 1, piece, sizeof piece); fputs(piece, stdout); }
                else printf("%d ", tok);
                ids[np + n] = tok;
                n++;
                fflush(stdout);
                if (n < max_tokens && np + n < c->ctx)
                    forward(&m, &ids[np + n - 1], 1, np + n - 1, logits, 1, b);
                else if (np + n >= c->ctx) break;
            }
            double el = now() - t;

            if (interactive) {
                printf("\n");
                fflush(stdout);
                char *resp = xmalloc((size_t)n * 16 + 256);
                size_t rlen = 0;
                for (int i = 0; i < n; i++) {
                    char piece2[64];
                    int pn = lfmtok_decode(T, &ids[np + i], 1, piece2, sizeof piece2 - 1);
                    if (pn > 0 && rlen + (size_t)pn < (size_t)n * 16 + 256 - 16) {
                        memcpy(resp + rlen, piece2, (size_t)pn);
                        rlen += (size_t)pn;
                    }
                }
                resp[rlen] = 0;

                size_t add = rlen + strlen("<|im_end|>\n");
                while (hist_cap < hist_len + add + 1) { hist_cap *= 2; history = realloc(history, hist_cap); }
                memcpy(history + hist_len, resp, rlen);
                hist_len += rlen;
                memcpy(history + hist_len, "<|im_end|>\n", strlen("<|im_end|>\n"));
                hist_len += strlen("<|im_end|>\n");
                history[hist_len] = 0;
                free(resp);

                int final_len = np + n;
                if (final_len > cached_cap) {
                    int cap = cached_cap ? cached_cap : 256;
                    while (cap < final_len) cap *= 2;
                    int *cc = realloc(cached_ids, (size_t)cap * sizeof *cc);
                    if (cc) { cached_ids = cc; cached_cap = cap; }
                }
                if (cached_ids && final_len <= cached_cap) {
                    memcpy(cached_ids, ids, (size_t)final_len * sizeof *ids);
                    cached_len = final_len;
                }
            } else {
                long long tot = m.hit + m.miss;
                printf("\n\nprefill %d tok in %.2fs (%.1f tok/s)\n", np, tpre, np / tpre);
                printf("decode  %d tok in %.2fs (%.2f tok/s)\n", n, el, n / el);
                printf("expert cache: %.1f%% hit (%lld reads total, %d pinned/layer)\n",
                       100.0 * m.hit / (tot ? tot : 1), tot, m.npin);
                pin_save(&m);
            }

            if (!interactive) break;
        }
        free(cached_ids);
        free(history);
    }
    return 0;
}
