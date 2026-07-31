/* lfm25.c — LiquidAI LFM2.5-8B-A1B on a small-RAM machine, by streaming experts.
 *
 * The sibling of gemma4.c, sharing its streaming machinery: an expert-granular
 * per-layer LRU instead of the OS page cache, batch-union prefill, a learned pin
 * set in usage.bin, and I/O threads. What differs is the model.
 *
 * THE BET. 8.3 B params, 1.5 B active, and the experts are TINY -- 32 per layer at
 * moe_intermediate_size 1792, ~5.9 MiB each at q4_0, 4 firing per token. A miss is
 * cheap and a layer's whole expert set is only 32 of them, so a modest cache holds
 * a large fraction of the model. (Gemma-4: 25 GB, 3.19 MiB experts, 128 per layer.)
 *
 * HYBRID ARCHITECTURE. 24 layers of which only 6 are attention; the other 18 mix
 * along the sequence with a short causal convolution (LFM's double-gated LIV block):
 *
 *     B, C, x = chunk(in_proj(h), 3)
 *     y       = C * causal_depthwise_conv(B * x, kernel=conv_L)
 *     out     = out_proj(y)
 *
 * Two consequences for the engine. A conv layer has NO KV, so KV costs a quarter of
 * what the layer count suggests. And the conv carries a RECURRENT STATE (the last
 * conv_L-1 gated activations per channel) which, unlike a position-addressed KV
 * cache, can only be advanced forwards: prefix reuse is allowed only when strictly
 * EXTENDING what was already absorbed (see conv_fwd, m->conv_pos). Getting this
 * wrong does not crash, it silently corrupts the state.
 *
 * The router is SIGMOID, not softmax, and the expert bias participates in selection
 * only -- see route_row.
 *
 * MIXED PRECISION, after apex-quant: always-on tensors (attention, conv, dense MLP)
 * are q8_0, routed experts q4_0 except in the edge layers. Every tensor carries its
 * own format, so matvec dispatches per tensor; tools/convert_lfm25.py decides the
 * gradient.
 *
 * No MTP, no DFlash: this model ships neither.
 *
 * Build:  cc -O3 -march=native -fopenmp lfm25.c -lm -lpthread -o lfm25
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
#include "lfmtok.h"
#include "kvq.h"
#include "openai_json.h"
#include "openai_http.h"

#define MAXL 64
#define MAXTOPK 16
#define MAXEXPERTS 256

/* tensor formats, as written by tools/convert_lfm25.py */
#define FMT_F32 0
#define FMT_Q40 1
#define FMT_Q80 2

/* ------------------------------------------------------------------ config */
typedef struct {
    int hidden, n_layers, n_heads, head_dim, n_kv_heads;
    int n_experts, topk, moe_inter, dense_inter, n_dense_layers, conv_L;
    int vocab, ctx, slots_per_layer;
    int layer_types[MAXL];                 /* 1 = attention, 0 = short conv */
    int norm_topk_prob, use_expert_bias;
    float eps, rope_theta, routed_scaling;
} Cfg;

/* a weight: q4_0 / q8_0 blob or f32 vector, all views into the resident dense blob */
typedef struct { int fmt; int O, I; const uint8_t *q; const float *f; } W;

typedef struct {
    W operator_norm, ffn_norm;
    W q_proj, k_proj, v_proj, o_proj, q_norm, k_norm;   /* attention layers */
    W conv_in, conv_w, conv_out;                        /* conv layers */
    W mlp_gate, mlp_up, mlp_down;                       /* dense layers (0..ND-1) */
    W router, expert_bias;                              /* MoE layers */
} Layer;

/* one cached expert */
typedef struct { int eid; uint64_t used; uint8_t *buf; int pinned, busy; } Slot;

typedef struct {
    Cfg c;
    Layer L[MAXL];
    const uint8_t *dense;   size_t dense_len;
    int efd;                                /* experts.bin */
    /* Per layer, because the apex gradient gives different layers different expert
     * formats and therefore different expert SIZES. Zero on conv/dense layers. */
    int64_t esz[MAXL], gate_b[MAXL], down_b[MAXL];
    int expert_fmt[MAXL];
    int64_t *eoff;                          /* [layer*n_experts + eid] -> file offset */

    Slot *slots;                            /* [n_layers][slots_per_layer] */
    uint64_t tick;
    int64_t hit, miss;

    /* LEARNED HOT-EXPERT PIN SET. Expert usage is heavily skewed and the hot set is
     * stable across prompts, so routing is counted per (layer, expert), persisted to
     * usage.bin, and on the next run the top-N per layer are pinned into slots the
     * LRU may never evict -- a policy the OS page cache cannot express. */
    int64_t *ucount;                        /* [n_layers * n_experts] routing counts */
    int npin;                               /* pinned slots per layer */
    char usage_path[4096];

    /* KV, attention layers only (NULL on conv layers). Every attention layer is
     * full-context -- LFM2.5 has no sliding window -- so position t lives at index
     * t and there is no ring aliasing. With --kvq the older positions are
     * TurboQuant-compressed and the most recent `rwin` stay f32. */
    float **kv_k, **kv_v;
    int **ring_pos;
    uint8_t **pk, **pv;
    Kvq *qk, *qv;
    int rwin;

    /* CONV RECURRENT STATE, [n_layers][(conv_L-1) * hidden], oldest position first.
     * conv_pos is how many positions have been absorbed; it is why prefix reuse is
     * restricted (see forward()). */
    float **conv_state;
    int conv_pos;

    /* RoPE inverse frequencies, [head_dim/2]: they depend only on the dimension, so
     * computing them in the rotation loop cost a powf per (head, dim, pos, layer). */
    float *inv_freq;

    /* I/O threads */
    pthread_t *io;
    int n_io;
    pthread_mutex_t mu;
    pthread_cond_t cv, done;
    struct { int layer, eid, slot; } *q;
    int qcap, qhead, qtail, qcount, inflight, stop;
} M;

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static void *xmalloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 4096, n ? n : 1)) { fprintf(stderr, "OOM %zu\n", n); exit(1); }
    return p;
}

/* ---------------------------------------------------------- format dispatch */
static inline int64_t fmt_row_bytes(int fmt, int I) {
    return fmt == FMT_Q40 ? q40_row_bytes(I)
         : fmt == FMT_Q80 ? q80_row_bytes(I)
         : (int64_t)I * 4;
}
/* COLI_F32ACT picks one of the two wdots below; the other is legitimately unused. */
#define MAYBE_UNUSED __attribute__((unused))

/* one weight row . int8-quantised activations */
MAYBE_UNUSED static inline float wdot(int fmt, const uint8_t *w, const int8_t *xq,
                                      const float *sx, int I) {
    return fmt == FMT_Q40 ? q40_dot(w, xq, sx, I) : q80_dot(w, xq, sx, I);
}
/* one weight row . f32 activations (COLI_F32ACT, and f32 tensors in either build) */
MAYBE_UNUSED static inline float wdot_f32(int fmt, const uint8_t *w, const float *x, int I) {
    if (fmt == FMT_Q40) return q40_dot_f32(w, x, I);
    if (fmt == FMT_Q80) return q80_dot_f32(w, x, I);
    const float *f = (const float *)w;
    double s = 0;
    for (int i = 0; i < I; i++) s += (double)f[i] * x[i];
    return (float)s;
}

/* ------------------------------------------------------------------ manifest */
typedef struct { char name[96]; int64_t off, len; int fmt, O, I; } DEnt;

static W dense_bind(const DEnt *dd, int ndense, const uint8_t *blob, const char *want) {
    for (int i = 0; i < ndense; i++) {
        if (!strcmp(dd[i].name, want)) {
            W w;
            w.fmt = dd[i].fmt; w.O = dd[i].O; w.I = dd[i].I;
            w.q = blob + dd[i].off;
            w.f = (const float *)(blob + dd[i].off);
            return w;
        }
    }
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
            if (!strcmp(k, "layer_types")) {
                int n = 0;
                for (char *t = strtok(v, " \n"); t && n < MAXL; t = strtok(NULL, " \n"))
                    c->layer_types[n++] = atoi(t);
            }
            else if (!strcmp(k, "hidden"))          c->hidden = atoi(v);
            else if (!strcmp(k, "n_layers"))        c->n_layers = atoi(v);
            else if (!strcmp(k, "n_heads"))         c->n_heads = atoi(v);
            else if (!strcmp(k, "head_dim"))        c->head_dim = atoi(v);
            else if (!strcmp(k, "n_kv_heads"))      c->n_kv_heads = atoi(v);
            else if (!strcmp(k, "n_experts"))       c->n_experts = atoi(v);
            else if (!strcmp(k, "topk"))            c->topk = atoi(v);
            else if (!strcmp(k, "moe_inter"))       c->moe_inter = atoi(v);
            else if (!strcmp(k, "dense_inter"))     c->dense_inter = atoi(v);
            else if (!strcmp(k, "n_dense_layers"))  c->n_dense_layers = atoi(v);
            else if (!strcmp(k, "conv_L"))          c->conv_L = atoi(v);
            else if (!strcmp(k, "vocab"))           c->vocab = atoi(v);
            else if (!strcmp(k, "ctx"))             c->ctx = atoi(v);
            else if (!strcmp(k, "slots_per_layer")) c->slots_per_layer = atoi(v);
            else if (!strcmp(k, "norm_topk_prob"))  c->norm_topk_prob = atoi(v);
            else if (!strcmp(k, "use_expert_bias")) c->use_expert_bias = atoi(v);
            else if (!strcmp(k, "eps"))             c->eps = atof(v);
            else if (!strcmp(k, "rope_theta"))      c->rope_theta = atof(v);
            else if (!strcmp(k, "routed_scaling"))  c->routed_scaling = atof(v);
            continue;
        }
        int li, e, fm, nexp;
        long long a, bb, cc, eo;
        if (sscanf(line, "eszl %d %lld %lld %lld %d", &li, &a, &bb, &cc, &fm) == 5) {
            m->esz[li] = a; m->gate_b[li] = bb; m->down_b[li] = cc;
            m->expert_fmt[li] = fm;
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

    /* embed doubles as the lm_head (tie_word_embeddings) -- stashed in a spare slot */
    m->L[MAXL - 1].q_proj = dense_bind(dd, ndense, blob, "embed_tokens");
    m->L[MAXL - 1].o_proj = dense_bind(dd, ndense, blob, "embedding_norm");

    char nm[128];
    for (int l = 0; l < c->n_layers; l++) {
        Layer *L = &m->L[l];
        #define B(f, s) do { snprintf(nm, sizeof nm, "layers.%d." s, l); \
                             L->f = dense_bind(dd, ndense, blob, nm); } while (0)
        B(operator_norm, "operator_norm");
        B(ffn_norm, "ffn_norm");
        if (c->layer_types[l]) {
            B(q_proj, "q_proj"); B(k_proj, "k_proj"); B(v_proj, "v_proj");
            B(o_proj, "o_proj"); B(q_norm, "q_norm"); B(k_norm, "k_norm");
        } else {
            B(conv_in, "conv_in"); B(conv_w, "conv_w"); B(conv_out, "conv_out");
        }
        if (l < c->n_dense_layers) {
            B(mlp_gate, "mlp_gate"); B(mlp_up, "mlp_up"); B(mlp_down, "mlp_down");
        } else {
            B(router, "router"); B(expert_bias, "expert_bias");
        }
        #undef B
    }
    free(dd);

    snprintf(p, sizeof p, "%s/experts.bin", dir);
    m->efd = open(p, O_RDONLY);
    if (m->efd < 0) { perror(p); exit(1); }

    /* We keep our own expert cache, so page-caching the same bytes double-buffers
     * them. F_NOCACHE is macOS's O_DIRECT analogue; POSIX_FADV_RANDOM stops Linux
     * doing readahead around multi-MiB random reads. */
#if defined(__APPLE__)
    fcntl(m->efd, F_NOCACHE, 1);
    fcntl(m->efd, F_RDAHEAD, 0);
#elif defined(POSIX_FADV_RANDOM)
    posix_fadvise(m->efd, 0, 0, POSIX_FADV_RANDOM);
#endif
}

/* ------------------------------------------------------------------ kernels */
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

/* y = W x. COLI_F32ACT keeps activations in f32 (weights stay quantised) so --check
 * can separate the int8-activation approximation from an actual bug. */
static void matvec(float *y, const W *w, const float *x, int8_t *xq, float *sx) {
    int64_t rb = fmt_row_bytes(w->fmt, w->I);
    if (w->fmt == FMT_F32) {
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < w->O; o++)
            y[o] = wdot_f32(FMT_F32, w->q + (size_t)o * rb, x, w->I);
        return;
    }
#ifdef COLI_F32ACT
    (void)xq; (void)sx;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < w->O; o++)
        y[o] = wdot_f32(w->fmt, w->q + (size_t)o * rb, x, w->I);
#else
    q40_quant_act(x, xq, sx, w->I);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < w->O; o++)
        y[o] = wdot(w->fmt, w->q + (size_t)o * rb, xq, sx, w->I);
#endif
}

/* rotate_half RoPE, full rotation (LFM2.5 has no partial rotary factor).
 *
 * The dimension loop is outside the head loop on purpose: cos/sin do not depend on
 * the head, so this is head_dim/2 transcendentals per call, not n_heads times that. */
static void rope(float *x, int H, int D, int pos, const float *invf) {
    int half = D / 2;
    for (int i = 0; i < half; i++) {
        float a = pos * invf[i], co = cosf(a), si = sinf(a);
        for (int h = 0; h < H; h++) {
            float *v = x + (size_t)h * D;
            float x1 = v[i], x2 = v[i + half];
            v[i]        = x1 * co - x2 * si;
            v[i + half] = x2 * co + x1 * si;
        }
    }
}

/* Y[S,O] = X[S,I] * W^T.
 *
 * For weight reuse, not arithmetic: a matvec streams the whole [O,I] matrix per
 * output vector, and at ~453 M dense params per position the engine is squarely
 * bandwidth-bound. Here each row is loaded once and dotted against all S
 * activations while it is still in cache. */
static void matmul(float *Y, const W *w, const float *X, int S,
                   int8_t *xq, float *sx) {
    if (S == 1) { matvec(Y, w, X, xq, sx); return; }
    int I = w->I, O = w->O, nb = I / Q40_BLK;
    int64_t rb = fmt_row_bytes(w->fmt, I);
#ifdef COLI_F32ACT
    (void)xq; (void)sx; (void)nb;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *row = w->q + (size_t)o * rb;
        for (int s = 0; s < S; s++)
            Y[(size_t)s * O + o] = wdot_f32(w->fmt, row, X + (size_t)s * I, I);
    }
#else
    if (w->fmt == FMT_F32) {
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint8_t *row = w->q + (size_t)o * rb;
            for (int s = 0; s < S; s++)
                Y[(size_t)s * O + o] = wdot_f32(FMT_F32, row, X + (size_t)s * I, I);
        }
        return;
    }
    for (int s = 0; s < S; s++)
        q40_quant_act(X + (size_t)s * I, xq + (size_t)s * I, sx + (size_t)s * nb, I);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *row = w->q + (size_t)o * rb;
        for (int s = 0; s < S; s++)
            Y[(size_t)s * O + o] = wdot(w->fmt, row, xq + (size_t)s * I,
                                        sx + (size_t)s * nb, I);
    }
#endif
}

/* ------------------------------------------------------------------ expert I/O */
static void slot_read(M *m, int layer, int eid, uint8_t *buf) {
    int64_t off = m->eoff[(int64_t)layer * m->c.n_experts + eid];
    int64_t sz = m->esz[layer];
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
    /* Do NOT publish the eid yet: the buffer holds the evicted expert until the I/O
     * completes, and the worker sets s->eid when the bytes are actually there. */
    base[lru].eid = -1;
    base[lru].used = ++m->tick;
    *need_io = 1;
    m->miss++;
    return lru;
}

/* ------------------------------------------------------------------ forward */
typedef struct {
    float *x, *xn, *q, *k, *v, *o, *tmp, *mlp;
    float *gate, *up, *eout, *h2, *bcx, *cy;
    int8_t *mxq; float *msx;         /* batched-activation scratch for matmul */
    int moe_nu, moe_chunk_n, moe_prefetch, moe_slots[MAXEXPERTS];
    /* Per-row CPU prefill scratch: avoids OpenMP fork/join for every row. */
    float *egate, *esx, *ehs;
    int8_t *exq, *ehq;
    int8_t *xq, *hq;
    float *sx, *hs;
    int *eidx;        /* [S * topk] chosen expert per (row, slot) */
    float *ewt;       /* [S * topk] weight */
    float *rwt;       /* [n_experts] sigmoid scores of one row */
    int *rows;        /* scratch: rows routed to one expert */
    float *roww;      /* scratch: their weights */
    int *uniq;        /* distinct experts in the batch */
    int S;
} Buf;

/* Sigmoid router whose expert bias participates in SELECTION ONLY:
 *
 *     routing_weights    = sigmoid(logits)
 *     scores_for_routing = routing_weights + expert_bias      <- selection
 *     idx                = topk(scores_for_routing)
 *     w                  = routing_weights[idx]               <- NOT the biased score
 *     if norm_topk_prob: w /= (sum(w) + 1e-6)
 *     w                 *= routed_scaling_factor
 *
 * The bias is a load-balancing nudge; letting it leak into the weight would rescale
 * every expert's contribution by an amount unrelated to the input. */
static void route_row(M *m, int li, const float *xn, int *idx, float *wts, Buf *b) {
    Cfg *c = &m->c;
    Layer *L = &m->L[li];
    int D = c->hidden, E = c->n_experts, K = c->topk;
    if (K <= 0 || K > E || K > MAXTOPK) {
        fprintf(stderr, "invalid router topk=%d for %d experts (MAXTOPK=%d)\n",
                K, E, MAXTOPK);
        exit(1);
    }
    const float *RP = L->router.f;
    const float *BI = L->expert_bias.f;

    float topv[MAXTOPK];
    for (int j = 0; j < K; j++) { idx[j] = -1; topv[j] = -INFINITY; }
    for (int e = 0; e < E; e++) {
        const float *r = RP + (size_t)e * D;
        double v = 0;
        for (int i = 0; i < D; i++) v += (double)r[i] * xn[i];
        float s = 1.0f / (1.0f + expf(-(float)v));       /* sigmoid, not softmax */
        b->rwt[e] = s;
        float sel = c->use_expert_bias ? s + BI[e] : s;  /* selection score only */
        int j = K;
        while (j > 0 && sel > topv[j - 1]) j--;
        if (j == K) continue;
        for (int t = K - 1; t > j; t--) { topv[t] = topv[t - 1]; idx[t] = idx[t - 1]; }
        topv[j] = sel; idx[j] = e;
    }
    float sum = 0.0f;
    for (int j = 0; j < K; j++) { wts[j] = b->rwt[idx[j]]; sum += wts[j]; }
    if (c->norm_topk_prob)
        for (int j = 0; j < K; j++) wts[j] /= (sum + 1e-6f);
    for (int j = 0; j < K; j++) wts[j] *= c->routed_scaling;
}

/* Apply one loaded expert to every row that routed to it. SwiGLU, silu (not gelu):
 * w2( silu(w1 x) * w3 x ).
 *
 * The loop order matters. Parallelising over ROWS -- one thread per token -- walks
 * the whole expert per thread, streaming its ~5.9 MiB once per row (~16x redundant
 * traffic at a 128-token batch). Parallelising over OUTPUT rows instead loads a
 * weight row once and dots it against every activation while it is still in L1:
 * same flops, nrows-fold less bandwidth. */
static void expert_apply_batch_cpu(M *m, int fmt, const uint8_t *G, const uint8_t *U,
        const uint8_t *Dn, size_t grb, size_t drb, const float *X, float *OUT,
        const int *rows, int nrows, const float *w, Buf *b) {
    Cfg *c = &m->c; int D = c->hidden, MI = c->moe_inter;
    size_t gst = (size_t)MI + 64;                    /* per-row stride of egate/ehq */
#ifdef COLI_F32ACT
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < MI; o++) {
        const uint8_t *g = G + (size_t)o * grb, *u = U + (size_t)o * grb;
        for (int r = 0; r < nrows; r++) {
            const float *x = X + (size_t)rows[r] * D;
            b->egate[(size_t)r * gst + o] = silu(wdot_f32(fmt, g, x, D)) *
                                                 wdot_f32(fmt, u, x, D);
        }
    }
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < D; o++) {
        const uint8_t *d = Dn + (size_t)o * drb;
        for (int r = 0; r < nrows; r++)
            OUT[(size_t)rows[r] * D + o] +=
                w[r] * wdot_f32(fmt, d, b->egate + (size_t)r * gst, MI);
    }
#else
    /* quantise each participating activation row once, up front */
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < nrows; r++)
        q40_quant_act(X + (size_t)rows[r] * D, b->exq + (size_t)r * (D + 64),
                      b->esx + (size_t)r * (D / Q40_BLK + 8), D);

    #pragma omp parallel for schedule(static)
    for (int o = 0; o < MI; o++) {
        const uint8_t *g = G + (size_t)o * grb, *u = U + (size_t)o * grb;
        for (int r = 0; r < nrows; r++) {
            const int8_t *xq = b->exq + (size_t)r * (D + 64);
            const float *sx = b->esx + (size_t)r * (D / Q40_BLK + 8);
            b->egate[(size_t)r * gst + o] = silu(wdot(fmt, g, xq, sx, D)) *
                                                 wdot(fmt, u, xq, sx, D);
        }
    }
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < nrows; r++)
        q40_quant_act(b->egate + (size_t)r * gst, b->ehq + (size_t)r * gst,
                      b->ehs + (size_t)r * (MI / Q40_BLK + 8), MI);

    #pragma omp parallel for schedule(static)
    for (int o = 0; o < D; o++) {
        const uint8_t *d = Dn + (size_t)o * drb;
        for (int r = 0; r < nrows; r++)
            OUT[(size_t)rows[r] * D + o] +=
                w[r] * wdot(fmt, d, b->ehq + (size_t)r * gst,
                            b->ehs + (size_t)r * (MI / Q40_BLK + 8), MI);
    }
#endif
}

static void expert_apply(M *m, int li, const uint8_t *blob, const float *X, float *OUT,
                         const int *rows, int nrows, const float *w, Buf *b) {
    Cfg *c = &m->c;
    int D = c->hidden, MI = c->moe_inter, fmt = m->expert_fmt[li];
    const uint8_t *G = blob, *U = blob + m->gate_b[li], *Dn = blob + 2 * m->gate_b[li];
    size_t grb = fmt_row_bytes(fmt, D), drb = fmt_row_bytes(fmt, MI);
    if (nrows > 1) {
        expert_apply_batch_cpu(m, fmt, G, U, Dn, grb, drb, X, OUT, rows, nrows, w, b);
        return;
    }
    for (int r = 0; r < nrows; r++) {
        const float *x = X + (size_t)rows[r] * D;
        float *out = OUT + (size_t)rows[r] * D;
#ifndef COLI_F32ACT
        q40_quant_act(x, b->xq, b->sx, D);
#endif
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < MI; o++) {
#ifdef COLI_F32ACT
            float g = wdot_f32(fmt, G + (size_t)o * grb, x, D);
            float u = wdot_f32(fmt, U + (size_t)o * grb, x, D);
#else
            float g = wdot(fmt, G + (size_t)o * grb, b->xq, b->sx, D);
            float u = wdot(fmt, U + (size_t)o * grb, b->xq, b->sx, D);
#endif
            b->gate[o] = silu(g) * u;
        }
        float ww = w[r];
#ifdef COLI_F32ACT
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < D; o++)
            out[o] += ww * wdot_f32(fmt, Dn + (size_t)o * drb, b->gate, MI);
#else
        q40_quant_act(b->gate, b->hq, b->hs, MI);
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < D; o++)
            out[o] += ww * wdot(fmt, Dn + (size_t)o * drb, b->hq, b->hs, MI);
#endif
    }
}

/* BATCH-UNION MoE. Token-at-a-time prefill reads topk experts per layer per token,
 * but the S rows of a batch collectively route to at most min(n_experts, topk*S)
 * DISTINCT experts, so the loop is inverted: gather expert -> {rows that chose it},
 * read each once, apply it to all of them. A prompt of any length then reads at
 * most n_experts per layer.
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

/* The router reads the ffn-normed hidden, which is also the experts' input (unlike
 * Gemma-4, whose router reads the raw residual). */
static void moe_start(M *m, int li, const float *X, float *out, int S, Buf *b) {
    Cfg *c = &m->c;
    int D = c->hidden, K = c->topk;
    for (int s = 0; s < S; s++)
        route_row(m, li, X + (size_t)s * D, b->eidx + s * K, b->ewt + s * K, b);
    b->moe_nu = 0;
    for (int s = 0; s < S; s++) for (int j = 0; j < K; j++) {
        int e = b->eidx[s * K + j], seen = 0;
        for (int u = 0; u < b->moe_nu; u++) if (b->uniq[u] == e) { seen = 1; break; }
        if (!seen) b->uniq[b->moe_nu++] = e;
    }
    memset(out, 0, sizeof(float) * (size_t)S * D);
    /* moe_finish submits chunk n+1 before applying chunk n, so two chunks are
     * resident at once and need disjoint unpinned slots. Below 2 free slots there is
     * no room for the second chunk, so the overlap is dropped rather than evicting a
     * slot still in use -- slower, but --pin near the slot count is the user's call.
     *
     * free_slots/2 is only the UPPER bound on a chunk. Sizing the chunk at that
     * bound is wrong whenever the union is small: at decode S is 1, so moe_nu is at
     * most topk (4) against ~14 free slots, the whole union goes out as one chunk,
     * moe_finish finds nothing left to submit, and the layer stalls on the read with
     * no compute overlapping it. That is the worst case for this model, which -- no
     * early routing signal, no dense branch beside the MoE -- has chunk pipelining
     * as its ONLY latency hiding. So aim for two chunks and take the bound only when
     * it binds. */
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
        expert_apply(m, li, m->slots[(size_t)li * SL + b->moe_slots[c0 + u]].buf,
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

/* ------------------------------------------------------------------ KV
 * With --kvq the f32 ring holds the most recent `rwin` positions and the occupant
 * about to be overwritten is TurboQuant-encoded on its way out, so recent tokens
 * always attend at full precision -- 3-4 bit compression without a residual window
 * produces garbage. Without --kvq the ring IS the whole cache. */
static void kv_write(M *m, int li, int pos, int nkv, int hd, int cap,
                     const float *k, const float *v) {
    int quant = m->pk[li] != NULL;
    int W = quant ? (m->rwin < cap ? m->rwin : cap) : cap;
    size_t vec = (size_t)nkv * hd;
    int slot = pos % W;

    if (quant) {
        /* Evict whatever the slot ACTUALLY holds rather than assuming pos-W: on a
         * rewritten position that would encode the wrong vector into the packed
         * store, unrecoverably. */
        int old = m->ring_pos[li][slot];
        if (old >= 0 && old != pos) {
            const float *ok = m->kv_k[li] + (size_t)slot * vec;
            const float *ov = m->kv_v[li] + (size_t)slot * vec;
            for (int h = 0; h < nkv; h++) {
                size_t idx = (size_t)old * nkv + h;
                kvq_encode(&m->qk[li], ok + (size_t)h * hd, m->pk[li] + idx * m->qk[li].bytes);
                kvq_encode(&m->qv[li], ov + (size_t)h * hd, m->pv[li] + idx * m->qv[li].bytes);
            }
        }
        m->ring_pos[li][slot] = pos;
    }
    memcpy(m->kv_k[li] + (size_t)slot * vec, k, sizeof(float) * vec);
    memcpy(m->kv_v[li] + (size_t)slot * vec, v, sizeof(float) * vec);
}

/* There is deliberately no kv_read: attn_fwd reads the cache in place, decoding a
 * compressed key only when the residual ring does not hold the position. */

/* ------------------------------------------------------------- the two mixers */

/* SHORT CAUSAL CONVOLUTION (Lfm2MoeShortConv), one position at a time.
 *
 * Depthwise with kernel conv_L over the SEQUENCE, so per channel it is a dot of
 * conv_L taps against the last conv_L values of (B*x). `state` holds the previous
 * conv_L-1 of those, OLDEST FIRST, so tap k pairs with state[k] and the last tap
 * with the current value. Positions must be fed in order: this is a recurrence, not
 * a cache, and it cannot be rewound. */
static void conv_fwd(M *m, int li, const float *Xn, float *OUT, int S, Buf *b) {
    Cfg *c = &m->c;
    Layer *L = &m->L[li];
    int D = c->hidden, CL = c->conv_L;
    float *st = m->conv_state[li];                  /* [(CL-1) * D] */

    /* Both projections are batched; only the recurrence between them is sequential,
     * and that part is elementwise. */
    matmul(b->bcx, &L->conv_in, Xn, S, b->mxq, b->msx);      /* [S, 3D] = B | C | x */

    for (int s = 0; s < S; s++) {
        const float *row = b->bcx + (size_t)s * 3 * D;
        const float *Bg = row, *Cg = row + D, *xv = row + 2 * D;
        float *cy = b->cy + (size_t)s * D;
        for (int i = 0; i < D; i++) {
            float cur = Bg[i] * xv[i];
            const float *w = L->conv_w.f + (size_t)i * CL;
            float acc = w[CL - 1] * cur;
            for (int k = 0; k < CL - 1; k++) acc += w[k] * st[(size_t)k * D + i];
            cy[i] = Cg[i] * acc;
            for (int k = 0; k + 2 < CL; k++)
                st[(size_t)k * D + i] = st[(size_t)(k + 1) * D + i];
            if (CL > 1) st[(size_t)(CL - 2) * D + i] = cur;
        }
    }
    matmul(OUT, &L->conv_out, b->cy, S, b->mxq, b->msx);
}

/* GQA attention: q/k RMSNorm per head, full-rotation RoPE, causal over the whole
 * context (no sliding window), scaled by 1/sqrt(head_dim). */
static void attn_fwd(M *m, int li, const float *Xn, float *OUT, int S,
                     int pos_base, Buf *b) {
    Cfg *c = &m->c;
    Layer *L = &m->L[li];
    int hd = c->head_dim, nkv = c->n_kv_heads, nh = c->n_heads, rep = nh / nkv;
    int cap = c->ctx;
    int quant = m->pk[li] != NULL;
    int Wr = quant ? (m->rwin < cap ? m->rwin : cap) : cap;
    size_t vec = (size_t)nkv * hd;
    float scale = 1.0f / sqrtf((float)hd);

    if (hd > 512 || rep > 64) { fprintf(stderr, "attention head geometry too large\n"); exit(1); }

    matmul(b->q, &L->q_proj, Xn, S, b->mxq, b->msx);
    matmul(b->k, &L->k_proj, Xn, S, b->mxq, b->msx);
    matmul(b->v, &L->v_proj, Xn, S, b->mxq, b->msx);   /* V is neither normed nor roped */

    /* Norm + rope + publish KV for the whole batch first. Safe because the causal
     * loop below only ever reads t <= pos, so a position never sees a future key. */
    for (int s = 0; s < S; s++) {
        int pos = pos_base + s;
        float *q = b->q + (size_t)s * nh * hd;
        float *k = b->k + (size_t)s * vec;
        float *v = b->v + (size_t)s * vec;
        for (int i = 0; i < nh; i++)
            rmsnorm(q + (size_t)i * hd, q + (size_t)i * hd, L->q_norm.f, hd, c->eps);
        rope(q, nh, hd, pos, m->inv_freq);
        for (int i = 0; i < nkv; i++)
            rmsnorm(k + (size_t)i * hd, k + (size_t)i * hd, L->k_norm.f, hd, c->eps);
        rope(k, nkv, hd, pos, m->inv_freq);
        /* fold the attention scale into q once rather than into every score */
        for (int i = 0; i < nh * hd; i++) q[i] *= scale;
        kv_write(m, li, pos, nkv, hd, cap, k, v);
    }

    /* One parallel region per position, over KV heads -- not one per KV position
     * over q heads, which forks a team per t to cover ~4 K flops each. A kv head
     * serves `rep` q heads, so owning it lets a thread decode a compressed key once
     * and reuse it, or read the cache in place when uncompressed. */
    for (int s = 0; s < S; s++) {
        int pos = pos_base + s;
        const float *qrow = b->q + (size_t)s * nh * hd;
        float *orow = b->o + (size_t)s * nh * hd;
        #pragma omp parallel for schedule(static)
        for (int kh = 0; kh < nkv; kh++) {
            float kbuf[512], vbuf[512];
            float mx[64], z[64];
            for (int r = 0; r < rep; r++) {
                mx[r] = -INFINITY; z[r] = 0.0f;
                memset(orow + (size_t)(kh * rep + r) * hd, 0, sizeof(float) * hd);
            }
            for (int t = 0; t <= pos; t++) {
                const float *kk, *vv;
                if (!quant || m->ring_pos[li][t % Wr] == t) {
                    kk = m->kv_k[li] + (size_t)(t % Wr) * vec + (size_t)kh * hd;
                    vv = m->kv_v[li] + (size_t)(t % Wr) * vec + (size_t)kh * hd;
                } else {
                    size_t idx = (size_t)t * nkv + kh;
                    kvq_decode(&m->qk[li], m->pk[li] + idx * m->qk[li].bytes, kbuf);
                    kvq_decode(&m->qv[li], m->pv[li] + idx * m->qv[li].bytes, vbuf);
                    kk = kbuf; vv = vbuf;
                }
                for (int r = 0; r < rep; r++) {
                    const float *qq = qrow + (size_t)(kh * rep + r) * hd;
                    float *ov = orow + (size_t)(kh * rep + r) * hd;
                    float score = 0.0f;
                    for (int d = 0; d < hd; d++) score += qq[d] * kk[d];
                    float nm = score > mx[r] ? score : mx[r];
                    float a = expf(mx[r] - nm), w = expf(score - nm), nz = a * z[r] + w;
                    float old = z[r] ? a * z[r] / nz : 0.0f, add = w / nz;
                    for (int d = 0; d < hd; d++) ov[d] = old * ov[d] + add * vv[d];
                    mx[r] = nm; z[r] = nz;
                }
            }
        }
    }
    matmul(OUT, &L->o_proj, b->o, S, b->mxq, b->msx);
}

/* ------------------------------------------------------------------ layer
 *
 *     h = h + operator(operator_norm(h))      operator = attention | short conv
 *     h = h + feed_forward(ffn_norm(h))       ffn      = dense SwiGLU | MoE
 *
 * Note what is NOT here: no dense branch alongside the MoE, so unlike gemma4 there
 * is no dense-MLP arithmetic to hide the expert reads behind. What remains is chunk
 * n+1's I/O overlapping chunk n's compute, plus batch-union. */
static void layer_fwd(M *m, int li, float *H, int S, int pos_base, Buf *b) {
    Cfg *c = &m->c;
    Layer *L = &m->L[li];
    int D = c->hidden;

    float *Xn = b->xn;
    for (int s = 0; s < S; s++)
        rmsnorm(Xn + (size_t)s * D, H + (size_t)s * D, L->operator_norm.f, D, c->eps);
    if (c->layer_types[li]) attn_fwd(m, li, Xn, b->tmp, S, pos_base, b);
    else                    conv_fwd(m, li, Xn, b->tmp, S, b);
    for (int i = 0; i < S * D; i++) H[i] += b->tmp[i];

    float *X = b->eout;
    for (int s = 0; s < S; s++)
        rmsnorm(X + (size_t)s * D, H + (size_t)s * D, L->ffn_norm.f, D, c->eps);

    if (li < c->n_dense_layers) {
        int DI = c->dense_inter;
        matmul(b->gate, &L->mlp_gate, X, S, b->mxq, b->msx);
        matmul(b->up, &L->mlp_up, X, S, b->mxq, b->msx);
        for (int i = 0; i < S * DI; i++) b->mlp[i] = silu(b->gate[i]) * b->up[i];
        matmul(b->tmp, &L->mlp_down, b->mlp, S, b->mxq, b->msx);
        for (int i = 0; i < S * D; i++) H[i] += b->tmp[i];
    } else {
        moe_start(m, li, X, b->h2, S, b);
        moe_finish(m, li, X, b->h2, S, b);
        for (int s = 0; s < S; s++) {
            float *h = H + (size_t)s * D;
            const float *o = b->h2 + (size_t)s * D;
            for (int i = 0; i < D; i++) h[i] += o[i];
        }
    }
}

/* embed one token row (the quantised embedding table is also the tied lm_head).
 * LFM2.5 uses a plain nn.Embedding: no embed_scale, unlike Gemma-4. */
static void embed_row(M *m, int tok, float *h) {
    int D = m->c.hidden;
    const W *e = &m->L[MAXL - 1].q_proj;
    const uint8_t *row = e->q + (size_t)tok * fmt_row_bytes(e->fmt, D);
    if (e->fmt == FMT_Q40)      q40_dequant_row(row, h, D);
    else if (e->fmt == FMT_Q80) q80_dequant_row(row, h, D);
    else memcpy(h, (const float *)row, sizeof(float) * D);
}

static void conv_reset(M *m) {
    Cfg *c = &m->c;
    size_t n = (size_t)(c->conv_L - 1) * c->hidden;
    for (int l = 0; l < c->n_layers; l++)
        if (m->conv_state[l]) memset(m->conv_state[l], 0, sizeof(float) * (n ? n : 1));
    m->conv_pos = 0;
}

/* Run S tokens from pos_base. logits may be NULL (prefill), [S,vocab], or -- the
 * common case -- only the last row via `last_only`.
 *
 * pos_base == 0 means a NEW SEQUENCE and resets the conv recurrence. Any other
 * pos_base must equal conv_pos: the conv state cannot be rewound, so a caller
 * reusing a cached prefix has to be strictly extending it. */
static void forward_chunk(M *m, const int *ids, int S, int pos_base,
                          float *logits, int last_only, Buf *b) {
    Cfg *c = &m->c;
    int D = c->hidden;
    W *embed = &m->L[MAXL - 1].q_proj;
    W *fnorm = &m->L[MAXL - 1].o_proj;

    float *H = b->x;
    for (int s = 0; s < S; s++) embed_row(m, ids[s], H + (size_t)s * D);
    for (int l = 0; l < c->n_layers; l++) layer_fwd(m, l, H, S, pos_base, b);
    m->conv_pos = pos_base + S;

    if (!logits) return;
    int s0 = last_only ? S - 1 : 0;
    /* The lm_head is by far the widest tensor here, so batch it too when every
     * row's logits are wanted (--check). */
    if (!last_only && S > 1) {
        for (int s = 0; s < S; s++)
            rmsnorm(b->cy + (size_t)s * D, H + (size_t)s * D, fnorm->f, D, c->eps);
        matmul(logits, embed, b->cy, S, b->mxq, b->msx);
        return;
    }
    for (int s = s0; s < S; s++) {
        rmsnorm(b->cy, H + (size_t)s * D, fnorm->f, D, c->eps);
        float *out = logits + (size_t)(last_only ? 0 : s) * c->vocab;
        matvec(out, embed, b->cy, b->xq, b->sx);          /* tied lm_head */
    }
}

/* As forward_chunk, but for an input of any length.
 *
 * Long inputs are processed in chunks of b->S. That is exactly equivalent to one
 * big call -- positions still go through every layer in order, the conv recurrence
 * carries across the boundary and KV accumulates -- and it bounds the batched
 * projection scratch, which would otherwise scale with the full context. */
static void forward(M *m, const int *ids, int S, int pos_base,
                    float *logits, int last_only, Buf *b) {
    if (pos_base == 0) conv_reset(m);
    else if (pos_base != m->conv_pos) {
        fprintf(stderr, "internal error: forward at %d but the conv state is at %d\n",
                pos_base, m->conv_pos);
        exit(1);
    }
    for (int c0 = 0; c0 < S; c0 += b->S) {
        int cn = S - c0 < b->S ? S - c0 : b->S;
        int last = (c0 + cn == S);
        float *lg = !logits ? NULL
                  : last_only ? (last ? logits : NULL)
                  : logits + (size_t)c0 * m->c.vocab;
        forward_chunk(m, ids + c0, cn, pos_base + c0, lg, last_only, b);
    }
}

/* ------------------------------------------------------------------ pinning */
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
        if (!m->esz[l]) continue;            /* conv / dense layers hold no experts */
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

/* ------------------------------------------------------------------ init */
static void init(M *m, const char *dir, int n_io, int ctx_override, double ram_gb,
                 int kb, int vb, int kv_protect, int rwin, int kv_pbits) {
    m->rwin = rwin > 0 ? rwin : 128;
    manifest(m, dir);
    Cfg *c = &m->c;

    /* --ctx overrides what the container was converted with, and may go UP: the
     * container's ctx only fixed slots_per_layer, and the weights do not care.
     * Whether that overshoots the conversion's budget is a question about BYTES,
     * not the context number -- with --kvq a longer context can cost less -- so the
     * comparison happens below, once the real KV size is known. */
    int ctx_planned = c->ctx;
    if (ctx_override > 0) c->ctx = ctx_override;
    m->ucount = calloc((size_t)c->n_layers * c->n_experts, 8);
    snprintf(m->usage_path, sizeof m->usage_path, "%s/usage.bin", dir);

    m->inv_freq = xmalloc(sizeof(float) * (size_t)(c->head_dim / 2 + 1));
    for (int i = 0; i < c->head_dim / 2; i++)
        m->inv_freq[i] = powf(c->rope_theta, -(float)(2 * i) / (float)c->head_dim);

    m->kv_k = calloc(c->n_layers, sizeof(float *));
    m->kv_v = calloc(c->n_layers, sizeof(float *));
    m->ring_pos = calloc(c->n_layers, sizeof(int *));
    m->pk = calloc(c->n_layers, sizeof(uint8_t *));
    m->pv = calloc(c->n_layers, sizeof(uint8_t *));
    m->qk = calloc(c->n_layers, sizeof(Kvq));
    m->qv = calloc(c->n_layers, sizeof(Kvq));
    m->conv_state = calloc(c->n_layers, sizeof(float *));

    size_t kvb = 0, cvb = 0;
    int quant = kb > 0;
    for (int l = 0; l < c->n_layers; l++) {
        if (!c->layer_types[l]) {                    /* conv layer: state, no KV */
            size_t n = (size_t)(c->conv_L - 1) * c->hidden;
            if (!n) n = 1;
            m->conv_state[l] = xmalloc(sizeof(float) * n);
            memset(m->conv_state[l], 0, sizeof(float) * n);
            cvb += sizeof(float) * n;
            continue;
        }
        int hd = c->head_dim, nkv = c->n_kv_heads, cap = c->ctx;
        /* Protected layers get more bits, not f32: pinning a full-context layer to
         * f32 undoes most of the saving. */
        int prot = (l < kv_protect) || (l >= c->n_layers - kv_protect);
        int lkb = prot ? kv_pbits : kb;
        int lvb = prot ? kv_pbits : vb;

        int W = quant ? (m->rwin < cap ? m->rwin : cap) : cap;
        m->kv_k[l] = xmalloc(sizeof(float) * (size_t)W * nkv * hd);
        m->kv_v[l] = xmalloc(sizeof(float) * (size_t)W * nkv * hd);
        m->ring_pos[l] = xmalloc(sizeof(int) * (size_t)W);
        for (int i = 0; i < W; i++) m->ring_pos[l][i] = -1;
        kvb += 2 * sizeof(float) * (size_t)W * nkv * hd;

        if (quant) {
            /* one codec per (layer, K/V): the sign flips are regenerated from the
             * seed, never stored, so the decoder reproduces the rotation exactly. */
            kvq_init(&m->qk[l], hd, lkb, 0x9e3779b97f4a7c15ULL ^ (uint64_t)(l * 2 + 1));
            kvq_init(&m->qv[l], hd, lvb, 0x9e3779b97f4a7c15ULL ^ (uint64_t)(l * 2 + 2));
            m->pk[l] = xmalloc(m->qk[l].bytes * (size_t)cap * nkv);
            m->pv[l] = xmalloc(m->qv[l].bytes * (size_t)cap * nkv);
            kvb += (m->qk[l].bytes + m->qv[l].bytes) * (size_t)cap * nkv;
        }
    }
    fprintf(stderr, "kv: %.0f MiB for ctx %d", kvb / 1048576.0, c->ctx);
    if (quant) fprintf(stderr, " (K%d/V%d, rwin %d, %d protected layers at %d bits)",
                       kb, vb, m->rwin, kv_protect, kv_pbits);
    else       fprintf(stderr, " (f32; --kvq would cut this a lot)");
    fprintf(stderr, ";  conv state: %.0f KiB\n", cvb / 1024.0);

    /* Compare against what the conversion budgeted (an f32 KV at the container's own
     * ctx). Only an actual overshoot warrants a warning -- --kvq routinely buys a
     * longer context for fewer bytes than the plan assumed. */
    {
        size_t planned = 0;
        for (int l = 0; l < c->n_layers; l++)
            if (c->layer_types[l])
                planned += 2 * sizeof(float) * (size_t)ctx_planned
                         * c->n_kv_heads * c->head_dim;
        /* With an explicit --ram the cache is about to be re-planned against this
         * very figure, so the overshoot is already accounted for. */
        if (kvb > planned && ram_gb <= 0)
            fprintf(stderr, "warning: that is %.0f MiB more KV than the container's "
                    "plan budgeted (%.0f MiB for ctx %d, f32), so total RAM will "
                    "exceed the conversion's --ram%s\n",
                    (kvb - planned) / 1048576.0, planned / 1048576.0, ctx_planned,
                    quant ? "" : "; --kvq would cut it a lot");
    }

    /* ------------------------------------------------- expert-cache plan
     * slots_per_layer is the only thing the conversion's --ram fixed, and nothing in
     * the container depends on it: experts are read one at a time and the cache is
     * pure LRU. So a runtime --ram re-runs the planner of tools/convert_lfm25.py
     * against numbers now known exactly -- the resident dense blob and the KV just
     * allocated. Slots are budgeted against the LARGEST expert so the plan is safe
     * on every layer (mixed precision makes the edge layers bigger). */
    if (ram_gb > 0) {
        int64_t esz_max = 0, esz_sum = 0, nmoe = 0;
        for (int l = 0; l < c->n_layers; l++)
            if (m->esz[l]) {
                nmoe++; esz_sum += m->esz[l];
                if (m->esz[l] > esz_max) esz_max = m->esz[l];
            }
        if (esz_max && nmoe) {
            int64_t scratch = 192 << 20;
            int64_t avail = (int64_t)(ram_gb * (double)(1LL << 30))
                          - (int64_t)m->dense_len - (int64_t)kvb - (int64_t)cvb
                          - scratch;
            int64_t per = avail > 0 ? (avail / esz_max) / nmoe : 0;
            if (per > c->n_experts) per = c->n_experts;
            if (per < c->topk) {
                double min_gb = ((double)m->dense_len + (double)kvb + (double)cvb
                                 + (double)scratch
                                 + (double)c->topk * nmoe * esz_max) / (double)(1LL << 30);
                fprintf(stderr, "--ram %g GB leaves room for %lld experts per layer, "
                        "below topk=%d: this model needs %.2f GB at this context%s\n",
                        ram_gb, (long long)per, c->topk, min_gb,
                        quant ? "" : " (--kvq would lower it)");
                exit(1);
            }
            c->slots_per_layer = (int)per;
        }
        /* Reported at the true per-layer sizes, not the budgeting maximum -- which
         * is why the printed cache comes out under the budget. */
        fprintf(stderr, "ram: %g GB budget -> %d slots/layer "
                "(dense %.0f MiB + kv %.0f MiB + cache %.0f MiB)\n",
                ram_gb, c->slots_per_layer, m->dense_len / 1048576.0,
                kvb / 1048576.0, (double)c->slots_per_layer * esz_sum / 1048576.0);
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
                /* sized for THIS layer's format: edge-layer experts are bigger */
                s->buf = m->esz[l] ? xmalloc(m->esz[l]) : NULL;
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

static Buf *bufs(M *m, int Smax) {
    Cfg *c = &m->c;
    int D = c->hidden, K = c->topk;
    int qmax = c->n_heads * c->head_dim;
    int kvmax = c->n_kv_heads * c->head_dim;
    int imax = c->dense_inter > c->moe_inter ? c->dense_inter : c->moe_inter;
    int wide = qmax > c->vocab ? qmax : c->vocab;
    if (wide < D) wide = D;
    if (wide < imax) wide = imax;
    if (wide < 3 * D) wide = 3 * D;              /* conv in_proj output */
    /* widest CONTRACTED dimension any matmul sees (lm_head contracts over D, not
     * vocab); the batched activation scratch is sized by it */
    int cmax = D > imax ? D : imax;

    Buf *b = calloc(1, sizeof *b);
    b->S = Smax;
    b->x    = xmalloc(sizeof(float) * (size_t)Smax * D);
    b->eout = xmalloc(sizeof(float) * (size_t)Smax * D);   /* ffn-normed input */
    b->h2   = xmalloc(sizeof(float) * (size_t)Smax * D);   /* MoE output */
    b->xn   = xmalloc(sizeof(float) * ((size_t)Smax * D + wide));
    b->q    = xmalloc(sizeof(float) * ((size_t)Smax * qmax + 64));
    b->k    = xmalloc(sizeof(float) * ((size_t)Smax * kvmax + 64));
    b->v    = xmalloc(sizeof(float) * ((size_t)Smax * kvmax + 64));
    b->o    = xmalloc(sizeof(float) * ((size_t)Smax * qmax + 64));
    b->tmp  = xmalloc(sizeof(float) * ((size_t)Smax * D + wide));
    b->mlp  = xmalloc(sizeof(float) * ((size_t)Smax * imax + 64));
    b->bcx  = xmalloc(sizeof(float) * ((size_t)Smax * 3 * D + 64));
    b->cy   = xmalloc(sizeof(float) * ((size_t)Smax * D + 64));
    b->gate = xmalloc(sizeof(float) * ((size_t)Smax * imax + 64));
    b->up   = xmalloc(sizeof(float) * ((size_t)Smax * imax + 64));
    b->mxq  = xmalloc((size_t)Smax * cmax + 64);
    b->msx  = xmalloc(sizeof(float) * ((size_t)Smax * (cmax / Q40_BLK) + 64));
    b->rwt  = xmalloc(sizeof(float) * (c->n_experts + 64));
    b->xq   = xmalloc(wide + 64);
    b->hq   = xmalloc(wide + 64);
    b->sx   = xmalloc(sizeof(float) * (wide / Q40_BLK + 8));
    b->hs   = xmalloc(sizeof(float) * (wide / Q40_BLK + 8));
    b->eidx = xmalloc(sizeof(int) * (size_t)Smax * K);
    b->ewt  = xmalloc(sizeof(float) * (size_t)Smax * K);
    b->rows = xmalloc(sizeof(int) * (size_t)Smax * K);
    b->roww = xmalloc(sizeof(float) * (size_t)Smax * K);
    b->uniq = xmalloc(sizeof(int) * c->n_experts);
    b->egate = xmalloc(sizeof(float) * (size_t)Smax * (c->moe_inter + 64));
    b->exq  = xmalloc((size_t)Smax * (D + 64));
    b->ehq  = xmalloc((size_t)Smax * (c->moe_inter + 64));
    b->esx  = xmalloc(sizeof(float) * (size_t)Smax * (D / Q40_BLK + 8));
    b->ehs  = xmalloc(sizeof(float) * (size_t)Smax * (c->moe_inter / Q40_BLK + 8));
    return b;
}

/* ------------------------------------------------------------------ sampling */
typedef struct { float p; int i; } PI;
static int pi_desc(const void *a, const void *b) {
    float x = ((const PI *)a)->p, y = ((const PI *)b)->p;
    return x < y ? 1 : x > y ? -1 : 0;
}
/* HF-style repetition penalty over the whole context: every token already in the
 * sequence has its logit divided by pen when positive, multiplied when negative,
 * and only once no matter how often it occurs. Applied to the fresh logits before
 * temperature, so it also bites in greedy mode. */
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

/* temperature + top-k + nucleus. Greedy when temp <= 0. top_k is applied BEFORE
 * top_p, which is the order HF uses. LFM2.5's own generation defaults are
 * temp 0.2 / top_k 80 (generation_config.json). */
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

/* ------------------------------------------------------------ chat template
 *
 * Plain ChatML, transcribed from LFM2.5's chat_template.jinja:
 *
 *   <|startoftext|>
 *   [<|im_start|>system\n SYSTEM <|im_end|>\n]
 *   <|im_start|>user\n USER <|im_end|>\n
 *   <|im_start|>assistant\n
 *
 * There is no thinking toggle: LFM2.5 decides for itself, emitting <think>...</think>
 * inside the assistant turn, and --think forces it by pre-filling the opening tag.
 * Generation stops on <|im_end|>. */
static void chat_prompt(char *out, size_t cap, const char *sys,
                        const char *user, int think) {
    size_t n = 0;
    #define ADD(...) n += snprintf(out + n, n < cap ? cap - n : 0, __VA_ARGS__)
    ADD("<|startoftext|>");
    if (sys && *sys) ADD("<|im_start|>system\n%s<|im_end|>\n", sys);
    ADD("<|im_start|>user\n%s<|im_end|>\n", user ? user : "");
    ADD("<|im_start|>assistant\n");
    if (think) ADD("<think>");
    #undef ADD
}

/* ------------------------------------------- OpenAI-compatible local server */
typedef struct {
    M *model;
    Buf *buffers;
    LfmTok *tokenizer;
    pthread_mutex_t generation_mu;
    atomic_int cancel;
    int *cached_ids;
    int cached_len;
    int cached_cap;
    int eos, eot;
    const char *model_id;
} LfmServerContext;

typedef struct { char *data; size_t len, cap; } LfmString;

static int lfm_string_append(LfmString *s, const char *data, size_t len) {
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

static int lfm_json_escape(LfmString *s, const char *text, size_t len) {
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == '"' || c == '\\') { char x[2] = {'\\', (char)c}; if (!lfm_string_append(s,x,2)) return 0; }
        else if (c == '\n') { if (!lfm_string_append(s,"\\n",2)) return 0; }
        else if (c == '\r') { if (!lfm_string_append(s,"\\r",2)) return 0; }
        else if (c == '\t') { if (!lfm_string_append(s,"\\t",2)) return 0; }
        else if (c < 0x20) { char x[6] = {'\\','u','0','0',hex[c>>4],hex[c&15]}; if (!lfm_string_append(s,x,6)) return 0; }
        else if (!lfm_string_append(s,(const char *)&text[i],1)) return 0;
    }
    return 1;
}

static jval *lfm_json_field(jval *object, const char *key, jtype type) {
    jval *v = json_get(object, key); return v && v->t == type ? v : NULL;
}

static int lfm_build_chat_prompt(jval *messages, LfmString *prompt) {
    if (!messages || messages->t != J_ARR) return 0;
    #define PUT(s) do { const char *_s = (s); \
                        if (!lfm_string_append(prompt, _s, strlen(_s))) return 0; } while (0)
    PUT("<|startoftext|>");
    for (int i = 0; i < messages->len; i++) {
        jval *message = messages->kids[i];
        jval *role = lfm_json_field(message, "role", J_STR);
        jval *content = lfm_json_field(message, "content", J_STR);
        if (!role || !content) continue;
        if (strcmp(role->str, "system") && strcmp(role->str, "user") &&
            strcmp(role->str, "assistant")) continue;
        PUT("<|im_start|>");
        PUT(role->str);
        PUT("\n");
        PUT(content->str);
        PUT("<|im_end|>\n");
    }
    PUT("<|im_start|>assistant\n");
    #undef PUT
    return 1;
}

static int lfm_send_chunk(int fd, const char *id, const char *field, const char *text, size_t len) {
    LfmString out = {0};
    const char *prefix = "data: {\"id\":\"";
    const char *middle = "\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"";
    const char *suffix = "\":\"";
    const char *end = "\"},\"finish_reason\":null}]}\n\n";
    int ok = lfm_string_append(&out,prefix,strlen(prefix)) && lfm_json_escape(&out,id,strlen(id)) &&
        lfm_string_append(&out,middle,strlen(middle)) && lfm_string_append(&out,field,strlen(field)) &&
        lfm_string_append(&out,suffix,strlen(suffix)) && lfm_json_escape(&out,text,len) &&
        lfm_string_append(&out,end,strlen(end));
    if (ok) ok = samosa_send_all(fd, out.data, out.len);
    free(out.data); return ok;
}

static int lfm_send_done(int fd, const char *id, int prompt_tokens, int completion_tokens,
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

static int lfm_serve_chat(LfmServerContext *ctx, int fd, jval *root) {
    jval *messages = lfm_json_field(root, "messages", J_ARR);
    int has_user = 0;
    if (!messages) return samosa_http_json_error(fd,400,"invalid_messages","messages must be an array.");
    for (int i = 0; i < messages->len; i++) {
        jval *msg = messages->kids[i];
        jval *role = lfm_json_field(msg,"role",J_STR);
        jval *content = lfm_json_field(msg,"content",J_STR);
        if (role && content && !strcmp(role->str,"user")) has_user = 1;
    }
    if (!has_user) return samosa_http_json_error(fd,400,"invalid_messages","A text user message is required.");

    int stream = 0, max_tokens = 2048, topk = 80, seed = 0;
    float temperature = 0.2f, topp = 1.0f, penalty = 1.05f;   /* LFM2.5 generation defaults */
    jval *v = json_get(root,"stream"); if (v && v->t == J_BOOL) stream = v->boolean;
    v = json_get(root,"max_tokens"); if (!v) v = json_get(root,"max_completion_tokens");
    if (v) {
        if (v->t != J_NUM || v->num < 1 || v->num > 8192 || floor(v->num) != v->num)
            return samosa_http_json_error(fd,400,"invalid_max_tokens","max_tokens must be an integer in 1..8192.");
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
        if (v->t != J_NUM || v->num < 1 || v->num > 256 || floor(v->num) != v->num)
            return samosa_http_json_error(fd,400,"invalid_top_k","top_k must be an integer in 1..256.");
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
    LfmString prompt = {0};
    if (!lfm_build_chat_prompt(messages, &prompt)) {
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
    /* The window is whatever --ctx sized the KV for at startup (advertised as
     * context_length on /v1/models); report both figures, not just "context limit". */
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
    /* PREFIX REUSE IS RESTRICTED. The conv recurrence only moves forwards, so the
     * cached state is reusable only when this prompt STRICTLY EXTENDS what was
     * absorbed. A diverging prefix -- or even an identical prompt, which would need
     * position np-1 replayed -- is reprocessed from scratch. */
    int common = 0;
    while (common < ctx->cached_len && common < np && ctx->cached_ids[common] == ids[common]) common++;
    if (common > 0 && common == ctx->cached_len && common == m->conv_pos && common < np)
        forward(m, ids + common, np - common, common, logits, 1, ctx->buffers);
    else
        forward(m, ids, np, 0, logits, 1, ctx->buffers);

    char id[64]; snprintf(id,sizeof id,"lfm25-%llu",(unsigned long long)time(NULL));
    if (stream && !samosa_http_stream_headers(fd)) { pthread_mutex_unlock(&ctx->generation_mu); free(ids); free(logits); free(pbuf); free(seen); return 1; }
    LfmString answer = {0}; uint64_t rng = seed ? (uint64_t)seed : 0x853c49e6748fea9bULL;
    int generated = 0; const char *reason = "length";
    while (generated < max_tokens && !atomic_load(&ctx->cancel)) {
        repetition_penalty(logits, c->vocab, ids, np + generated, penalty, seen);
        int token = sample(logits, c->vocab, temperature, topp, topk, pbuf, &rng);
        if (token == ctx->eos || token == ctx->eot) { reason = "stop"; break; }
        char piece[4096]; int n = lfmtok_decode(tok, &token, 1, piece, sizeof piece - 1);
        if (n <= 0) { reason = "stop"; break; }
        if (!lfm_string_append(&answer, piece, (size_t)n)) { atomic_store(&ctx->cancel,1); break; }
        if (stream && !lfm_send_chunk(fd,id,"content",piece,(size_t)n)) { atomic_store(&ctx->cancel,1); break; }
        ids[np + generated++] = token;
        if (generated < max_tokens) forward(m, &token, 1, np + generated - 1, logits, 1, ctx->buffers);
    }
    if (atomic_load(&ctx->cancel)) reason = "cancelled";
    if (stream) lfm_send_done(fd,id,np,generated,reason);
    else {
        LfmString body={0}; char prefix[512], suffix[512];
        int n=snprintf(prefix,sizeof prefix,"{\"id\":\"%s\",\"object\":\"chat.completion\",\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"",id,ctx->model_id);
        int ok=n>0&&lfm_string_append(&body,prefix,(size_t)n)&&lfm_json_escape(&body,answer.data?answer.data:"",answer.len);
        n=snprintf(suffix,sizeof suffix,"\"},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",reason,np,generated,np+generated);
        ok=ok&&n>0&&lfm_string_append(&body,suffix,(size_t)n)&&samosa_http_headers(fd,200,"application/json",body.len,NULL)&&samosa_send_all(fd,body.data,body.len);
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

static int lfm_serve_handler(SamosaHttpServer *server, int fd, const SamosaHttpRequest *request, void *opaque) {
    LfmServerContext *ctx = opaque;
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
        int result=lfm_serve_chat(ctx,fd,root); json_free(root); free(arena); return result;
    }
    if (!strcmp(request->method,"POST") && !strcmp(request->path,"/v1/shutdown")) {
        atomic_store(&ctx->cancel,1); samosa_http_response(fd,200,"application/json","{\"shutting_down\":true}",NULL); samosa_http_server_stop(server); return 1;
    }
    return samosa_http_json_error(fd,404,"not_found","Endpoint not found.");
}

static int run_lfm_server(M *m, Buf *buffers, LfmTok *tokenizer, const char *model_id, int port) {
    LfmServerContext ctx={.model=m,.buffers=buffers,.tokenizer=tokenizer,.model_id=model_id};
    ctx.eos = lfmtok_id(tokenizer, "<|im_end|>");
    ctx.eot = lfmtok_id(tokenizer, "<|endoftext|>");
    pthread_mutex_init(&ctx.generation_mu,NULL); atomic_init(&ctx.cancel,0);
    SamosaHttpServer server;
    if (!samosa_http_server_init(&server,port,lfm_serve_handler,&ctx)) { fprintf(stderr,"server: cannot bind port %d: %s\n",port,strerror(errno)); pthread_mutex_destroy(&ctx.generation_mu); return 1; }
    fprintf(stderr,"[server] OpenAI endpoint ready at http://127.0.0.1:%d\n",server.port); fflush(stderr);
    int ok=samosa_http_server_run(&server); samosa_http_server_destroy(&server);
    free(ctx.cached_ids); pthread_mutex_destroy(&ctx.generation_mu); return ok?0:1;
}

/* ------------------------------------------------------------------ main */
static void usage(const char *prog, FILE *out) {
    fprintf(out,
        "usage: %s <dir> [flags...] [prompt]\n"
        "         [--chat] [--system S] [--think] [--raw] [--max_tokens N]\n"
        "         [--temp F] [--topp F] [--topk N]   (default 0.2 / 1.0 / 80)\n"
        "         [--penalty F]           repetition penalty (default 1.05, 1 = off)\n"
        "         [--ctx N]               override the container's context length\n"
        "         [--ram F]               re-plan the expert cache for an F GB budget\n"
        "         [--pin N] [--io N] [--threads N] [--batch N] [--nobatch]\n"
        "         [--serve] [--port N]    OpenAI-compatible local server (default 8484)\n"
        "         [--kv off|k6v4|k4v2]    KV-cache compression preset\n"
        "         [--kvq] [--kbits N] [--vbits N] [--rwin N] [--protect N] [--pbits N]\n"
        "                                 override individual TurboQuant settings\n"
        "         [--check]               diff against the numpy oracle's logits\n"
        "         [--help]\n",
        prog);
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0], stderr); return 1; }
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(argv[0], stdout); return 0; }

    const char *dir = argv[1];
    const char *prompt = NULL, *sys = NULL;
    int think = 0, raw = 0, chat_mode = 0;
    int kvq_on = 0, kb = 6, vb = 4, rwin = 128, kv_protect = 2, kv_pbits = 8;
    int check = 0, n_io = 8, max_tokens = 0, nobatch = 0, npin = 0, nthreads = 2;
    int batch = 128, ctx_override = 0;
    double ram_gb = 0;                   /* 0 = keep the container's own plan */
    int serve_mode = 0, serve_port = 8484;
    float temp = 0.2f, topp = 1.0f, penalty = 1.05f;   /* LFM2.5 generation defaults */
    int topk = 80;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--serve")) serve_mode = 1;
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) serve_port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--check")) check = 1;
        else if (!strcmp(argv[i], "--nobatch")) nobatch = 1;
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
        else if (!strcmp(argv[i], "--think")) think = 1;
        else if (!strcmp(argv[i], "--raw")) raw = 1;
        else if (!strcmp(argv[i], "--kvq")) kvq_on = 1;
        else if (!strcmp(argv[i], "--kv") && i + 1 < argc) {
            /* The bits alone do not define a config: the residual window and the
             * protected-layer count matter as much, and a half-set K4/V2 (rwin=0) is
             * exactly the configuration upstream measured as broken. */
            const char *v = argv[++i];
            if (!strcmp(v, "off")) { kvq_on = 0; }
            else if (!strcmp(v, "k6v4")) {
                kvq_on = 1; kb = 6; vb = 4; rwin = 128; kv_protect = 2; kv_pbits = 8;
            } else if (!strcmp(v, "k4v2")) {
                kvq_on = 1; kb = 4; vb = 2; rwin = 128; kv_protect = 4; kv_pbits = 8;
            } else {
                fprintf(stderr, "--kv: expected off | k6v4 | k4v2 (got %s)\n", v);
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--kbits") && i + 1 < argc) { kb = atoi(argv[++i]); kvq_on = 1; }
        else if (!strcmp(argv[i], "--vbits") && i + 1 < argc) { vb = atoi(argv[++i]); kvq_on = 1; }
        else if (!strcmp(argv[i], "--rwin") && i + 1 < argc) rwin = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--protect") && i + 1 < argc) kv_protect = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pbits") && i + 1 < argc) kv_pbits = atoi(argv[++i]);
        else if (argv[i][0] == '-' && argv[i][1] == '-') {
            fprintf(stderr, "unknown flag: %s\n\n", argv[i]);
            usage(argv[0], stderr);
            return 1;
        }
        else if (!prompt) prompt = argv[i];   /* first non-flag positional is the prompt */
    }
    if (!kvq_on) kb = vb = 0;
    if (penalty < 0.5f || penalty > 2.0f) { fprintf(stderr, "--penalty must be in 0.5..2\n\n"); usage(argv[0], stderr); return 1; }
    if (chat_mode && check) { fprintf(stderr, "--chat cannot be used with --check\n\n"); usage(argv[0], stderr); return 1; }
    if (chat_mode && serve_mode) { fprintf(stderr, "--chat cannot be used with --serve\n\n"); usage(argv[0], stderr); return 1; }
    if (!serve_mode && !check && max_tokens == 0) max_tokens = 2048;
#ifdef _OPENMP
    if (nthreads > 0) omp_set_num_threads(nthreads);
#else
    (void)nthreads;
#endif

    M m; memset(&m, 0, sizeof m);
    double t0 = now();
    init(&m, dir, n_io, ctx_override, ram_gb, kb, vb, kv_protect, rwin, kv_pbits);
    Cfg *c = &m.c;
    pin_load(&m, npin);
    /* Prefill batch size: bigger reuses each streamed weight row over more rows, but
     * the projection scratch grows with it, so past a point it evicts the weights it
     * is trying to reuse. */
    if (batch < 1) batch = 1;
    if (batch > c->ctx) batch = c->ctx;
    Buf *b = bufs(&m, batch);

    int nattn = 0;
    for (int l = 0; l < c->n_layers; l++) nattn += c->layer_types[l];
    fprintf(stderr, "lfm25: %d layers (%d attn, %d conv), %d experts, top-%d, "
            "%d slots/layer, dense %.1f MiB, ready in %.2fs\n",
            c->n_layers, nattn, c->n_layers - nattn, c->n_experts, c->topk,
            c->slots_per_layer, m.dense_len / 1048576.0, now() - t0);

    if (serve_mode) {
        char tp[4096]; snprintf(tp, sizeof tp, "%s/tok.bin", dir);
        LfmTok *server_tokenizer = lfmtok_load(tp);
        if (!server_tokenizer) { fprintf(stderr, "--serve needs %s\n", tp); return 1; }
        return run_lfm_server(&m, b, server_tokenizer, "lfm2.5-8b-a1b", serve_port);
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

        /* The oracle runs on the DEQUANTISED container weights: comparing against an
         * fp32 reference would conflate engine bugs with the quantiser's own error. */
        snprintf(p, sizeof p, "%s/deq_logits.f32", dir);
        f = fopen(p, "rb");
        if (!f) { perror(p); fprintf(stderr, "run tools/lfm25_oracle.py first\n"); return 1; }
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
        return !ok;
    }

    if (max_tokens > 0) {
        float *logits = xmalloc(sizeof(float) * c->vocab);
        PI *pbuf = xmalloc(sizeof(PI) * c->vocab);
        unsigned char *seen = xmalloc((size_t)c->vocab);
        uint64_t rng = 0x853c49e6748fea9bULL;

        char tp[4096];
        snprintf(tp, sizeof tp, "%s/tok.bin", dir);
        LfmTok *T = lfmtok_load(tp);
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
            np = lfmtok_encode(T, text, ids, c->ctx - max_tokens);
            if (getenv("LFMDBG")) fprintf(stderr, "prompt: %s\n[%d tokens]\n", text, np);
            free(chat);
            if (np <= 0) { fprintf(stderr, "empty prompt\n"); return 1; }
        }

        /* ---- interactive multi-turn chat (--chat, or no prompt for compatibility) ---- */
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
                if (!pos) {
                    pos += snprintf(chat + pos, need - pos, "<|startoftext|>");
                    if (sys && *sys)
                        pos += snprintf(chat + pos, need - pos,
                                        "<|im_start|>system\n%s<|im_end|>\n", sys);
                }
                pos += snprintf(chat + pos, need - pos, "<|im_start|>user\n%s<|im_end|>\n", user_text);
                pos += snprintf(chat + pos, need - pos, "<|im_start|>assistant\n");
                if (think) pos += snprintf(chat + pos, need - pos, "<think>");

                np = lfmtok_encode(T, chat, ids, c->ctx - max_tokens);
                if (getenv("LFMDBG")) fprintf(stderr, "prompt: %s\n[%d tokens]\n", chat, np);
                if (np <= 0) { free(chat); continue; }

                if (!hist_cap) { hist_cap = 4096; history = xmalloc(hist_cap); }
                while (hist_cap < pos + 256) { hist_cap *= 2; history = realloc(history, hist_cap); }
                memcpy(history, chat, pos);
                hist_len = pos;
                free(chat);

                /* Same restriction as the server: reuse the conv state only when
                 * this prompt strictly EXTENDS what has already been absorbed. */
                int common = 0;
                while (common < cached_len && common < np && cached_ids[common] == ids[common])
                    common++;
                if (common > 0 && common == cached_len && common == m.conv_pos && common < np)
                    forward(&m, ids + common, np - common, common, logits, 1, b);
                else
                    forward(&m, ids, np, 0, logits, 1, b);
            } else {
                /* ---- one-shot prefill (prompt given on the command line) ---- */
                double t0p = now();
                forward(&m, ids, np, 0, logits, 1, b);
                tpre = now() - t0p;
            }

            char piece[512];
            int n = 0;
            double t = now();

            while (n < max_tokens) {
                repetition_penalty(logits, c->vocab, ids, np + n, penalty, seen);
                int tok = sample(logits, c->vocab, temp, topp, topk, pbuf, &rng);
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
