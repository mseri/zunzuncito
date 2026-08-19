/* gemma4.c — Gemma-4 26B-A4B on a small-RAM machine, by streaming experts.
 *
 * Gemma-4 26B-A4B is 25 GB of weights of which ~3.8 B params activate per token. At
 * q4_0 that is 1.31 GB of dense weights (resident) + 12.9 GB of routed experts (3840
 * of them, 3.19 MiB each). On a 4-8 GB machine the expert set does not fit, and mmap
 * leaves eviction to the kernel's global 4 KB-page LRU, which knows nothing about
 * expert granularity, expert hotness, or what the next layer needs. We keep an
 * explicit expert-granular per-layer LRU instead, and we prefetch.
 *
 * The prefetch is exact rather than predicted, because of how the layer is wired:
 * the router reads the post-attention residual, before the dense MLP touches it.
 *
 *     residual = h_after_attn
 *     h1 = post_ffn_ln_1( mlp( pre_ffn_ln(residual) ) )      <- dense MLP branch
 *     idx, w = router(residual)                              <- needs only residual
 *     h2 = post_ffn_ln_2( experts( pre_ffn_ln_2(residual) ) )
 *     h  = residual + post_ffn_ln(h1 + h2);  h *= layer_scalar
 *
 * So the 8 expert ids are known before the dense MLP runs. We route first, fire the
 * expert reads at the I/O threads, then compute the MLP, hiding NVMe latency behind
 * arithmetic. colibri's GLM path needs PILOT to guess next-layer routing at 71.6%;
 * here it is free and exact. A synchronous mmap fault cannot overlap at all.
 *
 * Everything is q4_0 (see q40.h): weights carry their fp16 scales inline, so one
 * expert is a single contiguous 4096-aligned byte range -> one pread, no scale seek.
 * With 240 expert reads per token, the read count is what we minimise.
 *
 * Build:  cc -O3 -march=native -fopenmp gemma4.c -lm -lpthread -o gemma4
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
#include <stdatomic.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "q40.h"
#include "g4tok.h"
#include "kvarn.h"
#include "gpu.h"
#include "openai_json.h"
#include "openai_http.h"

#define MAXL 64
#define MAXTOPK 16
#define MAXDRAFT 16

/* config */
typedef struct {
    int hidden, n_layers, n_heads, head_dim, global_head_dim;
    int n_kv_heads, n_global_kv_heads, k_eq_v_global;
    int n_experts, topk, moe_inter, dense_inter, vocab;
    int sliding_window, ctx, slots_per_layer;
    int layer_types[MAXL];                 /* 1 = full/global, 0 = sliding */
    float eps, rope_theta_local, rope_theta_global, rope_partial_global;
    float final_logit_softcap, embed_scale;
} Cfg;

/* a weight: q4_0 blob or f32 vector, both just views into the mmap'd dense blob */
typedef struct { int fmt; int O, I; const uint8_t *q; const float *f; } W;

typedef struct {
    W in_ln, post_attn_ln, pre_ffn_ln, post_ffn_ln;
    W pre_ffn_ln2, post_ffn_ln1, post_ffn_ln2, layer_scalar;
    W q_proj, k_proj, v_proj, o_proj, q_norm, k_norm;
    W mlp_gate, mlp_up, mlp_down;
    W router_proj, router_scale, router_pes;
    int has_v;                              /* 0 on global layers: V = k_proj (k_eq_v) */
} Layer;

typedef struct { int eid; uint64_t used; uint8_t *buf; int pinned, busy; } Slot;

typedef struct {
    Cfg c;
    Layer L[MAXL];
    const uint8_t *dense;   size_t dense_len;
    int efd;                                /* experts.bin */
    int64_t esz, gate_b, down_b;
    int64_t *eoff;                          /* [layer*n_experts + eid] -> file offset */

    Slot *slots;                            /* [n_layers][slots_per_layer] */
    uint64_t tick;
    int64_t hit, miss;

    /* MTP draft head (Gemma4AssistantForCausalLM). Not a separate model: it runs
     * inside this forward, attending into this model's own KV. See mtp_forward. */
    int mtp;                                /* head loaded */
    int mtp_L, mtp_D, mtp_BB, mtp_nh, mtp_hd, mtp_ghd, mtp_inter, mtp_vocab;
    float mtp_eps, mtp_theta_l, mtp_theta_g, mtp_partial;
    int mtp_types[MAXL];
    const uint8_t *mtp_blob;
    Layer mtp_layers[MAXL];
    W mtp_embed, mtp_norm, mtp_pre, mtp_post;
    int kv_last_slide, kv_last_full;        /* the layers whose KV the head shares */

    /* Post-norm hidden of each row of the last forward (small batches only).
     * The MTP head needs the hidden of the row before the token it conditions on --
     * see mtp_step -- so a single "last row" is not enough. */
    float *hid_batch;                       /* [(MAXDRAFT+2) * hidden] */
    int hid_rows;

    /* DFlash block-parallel drafter (DFlashDraftModel). Drafts an entire block of
     * tokens at once using bidirectional attention conditioned on backbone hidden
     * states extracted from specific layers. */
    int dflash;                             /* head loaded */
    int dflash_L, dflash_D, dflash_nh, dflash_hd, dflash_nkv, dflash_inter, dflash_vocab;
    float dflash_eps, dflash_theta;
    int dflash_block_size, dflash_mask_token_id, dflash_sliding_window;
    int dflash_n_target_layers, dflash_target_ids[16];
    int dflash_types[MAXL];
    const uint8_t *dflash_blob;
    Layer dflash_layers[MAXL];
    W dflash_fc, dflash_hidden_norm, dflash_norm;
    /* Intermediate hidden states captured from the backbone at target layers.
     * Sized per-forward: [n_target_layers * rows * hidden], grown on demand so a
     * long prefill can be captured too. The drafter must condition on the whole
     * prompt, not just the last batch, or acceptance collapses. */
    float *dflash_target_hidden;
    int dflash_target_hidden_cap;           /* capacity, in rows */
    /* Persistent draft-side context KV cache, one entry per absolute position, per
     * draft layer: [ctx * nkv * hd]. This mirrors past_key_values_draft in the
     * reference implementation: every confirmed position's projected target hidden
     * (fc + hidden_norm + k/v_proj + k_norm + rope at its own position) is cached
     * once and reused by every subsequent draft block. */
    float *dflash_ctx_k[MAXL], *dflash_ctx_v[MAXL];
    int dflash_ctx_len;                     /* positions [0, len) are absorbed */
    /* absorb scratch */
    float *dfa_concat, *dfa_th, *dfa_kv, *dfa_sx;
    int8_t *dfa_xq;

    /* Learned hot-expert pin set. Expert usage is heavily skewed: a minority of
     * experts take most of the routing mass, and which ones is stable across
     * prompts. We count how often each (layer, expert) is routed to, persist the
     * counts to usage.bin, and on the next run pin the top-N per layer into slots
     * the LRU may never evict. On a 4 GB box only 21 of 128 slots per layer exist,
     * so a pure LRU keeps re-reading the same hot experts after a burst of cold
     * ones pushes them out. The OS page cache cannot express this policy: it has no
     * idea what an "expert" is, let alone which are hot. */
    int64_t *ucount;                        /* [n_layers * n_experts] routing counts */
    int npin;                               /* pinned slots per layer */
    char usage_path[4096];

    /* KV. Sliding layers keep a ring of `sliding_window` positions; global layers
     * keep the full context. Both store K and V: attention_k_eq_v shares the
     * projection but not the post-processing (K is k_norm'd and roped, V is v_norm'd
     * and neither), so the two tensors genuinely differ.
     *
     * With --kv the older positions are KVarN-compressed a tile at a time and only
     * the most recent `rwin` are kept in f32. All KV growth is in the 5 global
     * layers, since the sliding ones are permanently capped by the window, so this
     * buys context length rather than RAM in general. */
    float **kv_k, **kv_v;                   /* f32 residual window (or the whole cache) */
    int **ring_pos;                         /* which position occupies each ring slot */
    /* Highest confirmed position. Speculation writes KV for positions that may be
     * rejected; those must never reach the KVarN store, because a sealed tile is
     * write-once and a rejected draft baked into it cannot be taken back out.
     * INT_MAX = not speculating, everything confirmed. */
    int kv_conf;
    uint8_t **pk, **pv;                     /* packed store, NULL if this layer is f32 */
    Kvarn *qk, *qv;
    int rwin;                               /* f32 ring, in positions: a whole number of tiles */
    int tile;                               /* KVarN tile width, in tokens */
    int pos;

    /* I/O threads */
    pthread_t *io;
    int n_io;
    pthread_mutex_t mu;
    pthread_cond_t cv, done;
    struct { int layer, eid, slot; } *q;
    int qcap, qhead, qtail, qcount, inflight, stop;
} M;

/* Auto-enables when a Metal device is present. Set to 0 by --no-metal, or when
 * gpu_init() fails for any reason. Every GPU call can decline (returning 0), and the
 * CPU path runs instead, so correctness never depends on Metal. */
static int g_use_gpu = 0;

/* DFlash iterative refinement (block-diffusion denoising). The reference drafts a
 * block in a single greedy pass; because the drafter is ~1000x smaller than the
 * target, we can optionally run extra denoising passes: after a pass, positions
 * whose drafted token is confident (max softmax prob >= g_dflash_conf) are frozen
 * as real input embeddings, the rest re-masked, and the block re-drafted. This is
 * in-distribution for the model and raises the accepted prefix at near-zero cost
 * relative to the target verify. g_dflash_refine counts the extra passes (0 = off,
 * i.e. reference behaviour). */
static int g_dflash_refine = 0;
static float g_dflash_conf = 0.9f;

/* Ring capacity of a layer's KV. Sliding layers need more than `sliding_window`
 * slots, for two independent reasons:
 *
 *  +1  The MTP head's bidirectional sliding mask attends t >= pos - W (W+1
 *      positions), one further back than the backbone's causal SWA
 *      (t >= pos - W + 1, W positions). Verified against HF.
 *
 *  +MAXDRAFT  Speculation writes ahead and then rewinds. A verify forward writes
 *      positions P..P+d, but on rejection the next step restarts at P+1 and its
 *      attention reaches back to (P+1) - W. The span that must be simultaneously
 *      live is therefore W + d + 1. With a W+1 ring the lookahead evicts the oldest
 *      positions of the very window the next step needs; output degrades and draft
 *      acceptance collapses with nothing to indicate why.
 *
 * At W = 1024 this costs 18 extra positions. */
static inline int kv_cap(const Cfg *c, int li) {
    return c->layer_types[li] ? c->ctx : c->sliding_window + MAXDRAFT + 2;
}

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static void *xmalloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 4096, n ? n : 1)) { fprintf(stderr, "OOM %zu\n", n); exit(1); }
    return p;
}

/* manifest */
typedef struct { char name[96]; int64_t off, len; int fmt, O, I; } DEnt;  /* dense.idx row */

/* Resolve a tensor name to a view into the dense blob.
 * File scope, not nested: nested functions are a GCC extension and clang rejects
 * them outright. */
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
    int ndense = 0, nexp = 0, di = 0;
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
            else if (!strcmp(k, "hidden"))            c->hidden = atoi(v);
            else if (!strcmp(k, "n_layers"))          c->n_layers = atoi(v);
            else if (!strcmp(k, "n_heads"))           c->n_heads = atoi(v);
            else if (!strcmp(k, "head_dim"))          c->head_dim = atoi(v);
            else if (!strcmp(k, "global_head_dim"))   c->global_head_dim = atoi(v);
            else if (!strcmp(k, "n_kv_heads"))        c->n_kv_heads = atoi(v);
            else if (!strcmp(k, "n_global_kv_heads")) c->n_global_kv_heads = atoi(v);
            else if (!strcmp(k, "k_eq_v_global"))     c->k_eq_v_global = (*v=='T'||*v=='1');
            else if (!strcmp(k, "n_experts"))         c->n_experts = atoi(v);
            else if (!strcmp(k, "topk"))              c->topk = atoi(v);
            else if (!strcmp(k, "moe_inter"))         c->moe_inter = atoi(v);
            else if (!strcmp(k, "dense_inter"))       c->dense_inter = atoi(v);
            else if (!strcmp(k, "vocab"))             c->vocab = atoi(v);
            else if (!strcmp(k, "sliding_window"))    c->sliding_window = atoi(v);
            else if (!strcmp(k, "ctx"))               c->ctx = atoi(v);
            else if (!strcmp(k, "slots_per_layer"))   c->slots_per_layer = atoi(v);
            else if (!strcmp(k, "eps"))               c->eps = atof(v);
            else if (!strcmp(k, "rope_theta_local"))  c->rope_theta_local = atof(v);
            else if (!strcmp(k, "rope_theta_global")) c->rope_theta_global = atof(v);
            else if (!strcmp(k, "rope_partial_global")) c->rope_partial_global = atof(v);
            else if (!strcmp(k, "final_logit_softcap")) c->final_logit_softcap = atof(v);
            else if (!strcmp(k, "embed_scale"))       c->embed_scale = atof(v);
            continue;
        }
        long long a, b, cc;
        if (sscanf(line, "esz %lld %lld %lld", &a, &b, &cc) == 3) {
            m->esz = a; m->gate_b = b; m->down_b = cc; continue;
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
        int li, e; long long eo;
        if (sscanf(line, "expert %d %d %lld", &li, &e, &eo) == 3) {
            m->eoff[(int64_t)li * c->n_experts + e] = eo; continue;
        }
    }
    fclose(f);

    /* dense.bin: read whole (1.31 GB at the real dims) -- it is the resident set */
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

    /* embed doubles as the lm_head (tie_word_embeddings) -- stashed in a spare slot */
    m->L[MAXL - 1].q_proj = dense_bind(dd, ndense, blob, "embed_tokens");
    m->L[MAXL - 1].o_proj = dense_bind(dd, ndense, blob, "norm");

    char nm[128];
    for (int l = 0; l < c->n_layers; l++) {
        Layer *L = &m->L[l];
        #define B(f, s) do { snprintf(nm, sizeof nm, "layers.%d." s, l); \
                             L->f = dense_bind(dd, ndense, blob, nm); } while (0)
        B(in_ln, "input_layernorm");
        B(post_attn_ln, "post_attention_layernorm");
        B(pre_ffn_ln, "pre_feedforward_layernorm");
        B(post_ffn_ln, "post_feedforward_layernorm");
        B(pre_ffn_ln2, "pre_feedforward_layernorm_2");
        B(post_ffn_ln1, "post_feedforward_layernorm_1");
        B(post_ffn_ln2, "post_feedforward_layernorm_2");
        B(layer_scalar, "layer_scalar");
        B(q_proj, "q_proj");  B(k_proj, "k_proj");  B(o_proj, "o_proj");
        B(q_norm, "q_norm");  B(k_norm, "k_norm");
        B(mlp_gate, "mlp_gate"); B(mlp_up, "mlp_up"); B(mlp_down, "mlp_down");
        B(router_proj, "router_proj"); B(router_scale, "router_scale");
        B(router_pes, "router_pes");
        L->has_v = !(c->layer_types[l] && c->k_eq_v_global);
        if (L->has_v) B(v_proj, "v_proj");
        #undef B
    }
    free(dd);

    snprintf(p, sizeof p, "%s/experts.bin", dir);
    m->efd = open(p, O_RDONLY);
    if (m->efd < 0) { perror(p); exit(1); }

    /* We keep our own expert cache, so letting the OS page-cache the same bytes
     * double-buffers them: on a 4 GB box that duplication wastes memory and evicts
     * the dense weights we need resident. Tell the kernel not to.
     * F_NOCACHE is macOS's O_DIRECT analogue; POSIX_FADV_RANDOM stops Linux from
     * doing useless readahead around 3.19 MiB random reads. */
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
static inline float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}
static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}
/* y = W x, W q4_0 [O,I].
 * COLI_F32ACT keeps activations in f32 (weights still q4_0). Only for validation:
 * it lets --check separate the int8-activation approximation from an actual bug. */
static void matvec(float *y, const W *w, const float *x, int8_t *xq, float *sx) {
    /* GPU first when available. The Metal kernel consumes f32 activations, so it is
     * numerically the q40_dot_f32 path, more accurate than the int8 default rather
     * than less. It declines (returns 0) if the weights are not GPU-mapped. */
    if (g_use_gpu && gpu_q40_matmul(y, w->q, x, w->O, w->I, 1)) return;
#ifdef COLI_F32ACT
    (void)xq; (void)sx;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < w->O; o++)
        y[o] = q40_dot_f32(w->q + (size_t)o * q40_row_bytes(w->I), x, w->I);
#else
    q40_quant_act(x, xq, sx, w->I);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < w->O; o++)
        y[o] = q40_dot(w->q + (size_t)o * q40_row_bytes(w->I), xq, sx, w->I);
#endif
}

/* Batched Y = W X, W q4_0 [O,I], X is [S,I] row-major, Y is [S,O] row-major.
 * One GPU dispatch fills an O*S grid, amortising launch latency and keeping the
 * device busy, which is why DFlash runs a whole block (S>1) at once. On the CPU
 * path this falls back to S independent matvecs. */
static void matmul(float *Y, const W *w, const float *X, int S, int8_t *xq, float *sx) {
    if (g_use_gpu && gpu_q40_matmul(Y, w->q, X, w->O, w->I, S)) return;
    for (int s = 0; s < S; s++)
        matvec(Y + (size_t)s * w->O, w, X + (size_t)s * w->I, xq, sx);
}

/* rotate_half RoPE. inv_freq's tail is zero on p-RoPE global layers (freq 0 =>
 * cos 1, sin 0 => identity), so partial rotary needs no special case here. */
static void rope(float *x, int H, int D, int pos, float theta, float partial) {
    int half = D / 2;
    int k = (int)(partial * D / 2);          /* live frequencies; rest are NOPE */
    for (int h = 0; h < H; h++) {
        float *v = x + (size_t)h * D;
        for (int i = 0; i < half; i++) {
            float f = (i < k) ? powf(theta, -(float)(2 * i) / (float)D) : 0.0f;
            float a = pos * f, co = cosf(a), si = sinf(a);
            float x1 = v[i], x2 = v[i + half];
            v[i]        = x1 * co - x2 * si;
            v[i + half] = x2 * co + x1 * si;
        }
    }
}

/* expert I/O */
static void slot_read(M *m, int layer, int eid, uint8_t *buf) {
    int64_t off = m->eoff[(int64_t)layer * m->c.n_experts + eid];
    for (int64_t o = 0; o < m->esz;) {
        ssize_t r = pread(m->efd, buf + o, m->esz - o, off + o);
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
/* Find or evict a slot for (layer,eid). Returns the slot index; sets *need_io if the
 * caller must fetch it. The LRU is per layer, so an expert only competes with the
 * other experts of its own layer; under a global page LRU a hot layer-3 expert can
 * be evicted by a cold layer-27 one. */
static int slot_for(M *m, int layer, int eid, int *need_io) {
    int S = m->c.slots_per_layer;
    Slot *base = &m->slots[(size_t)layer * S];
    m->ucount[(size_t)layer * m->c.n_experts + eid]++;      /* learn the hot set */
    for (int i = 0; i < S; i++)
        if (base[i].eid == eid) { base[i].used = ++m->tick; *need_io = 0; m->hit++; return i; }
    /* evict the LRU among the unpinned slots. If every slot is pinned the caller
     * asked for more distinct experts than the cache holds, which cannot happen:
     * moe_batch chunks by slots_per_layer, and npin < slots_per_layer is enforced. */
    int lru = -1;
    for (int i = 0; i < S; i++)
        if (!base[i].pinned && (lru < 0 || base[i].used < base[lru].used)) lru = i;
    if (lru < 0) lru = 0;
    /* Do not publish the eid yet: the buffer holds the evicted expert until the I/O
     * completes, and the worker sets s->eid once the bytes are actually there. */
    base[lru].eid = -1;
    base[lru].used = ++m->tick;
    *need_io = 1;
    m->miss++;
    return lru;
}

/* forward
 *
 * One code path for prefill and decode -- decode is simply S = 1 -- and the reason to
 * unify is the MoE.
 *
 * Token-at-a-time prefill reads 8 experts per layer per token, so a 1000-token prompt
 * is 240,000 reads of 3.19 MiB. But the S rows of a batch collectively route to at
 * most min(128, 8*S) distinct experts per layer, so the loop is inverted: gather
 * expert -> {rows that chose it}, read each once, apply it to all of them. Prefill
 * I/O drops from O(S * topk) to O(unique experts), bounded by 128 per layer however
 * long the prompt is -- up to a 60x cut.
 *
 * Chunking bounds that against slots_per_layer. Acquiring a slot bumps it to
 * most-recently-used, so nothing acquired within a chunk can be evicted by a later
 * acquisition in the same chunk.
 */
typedef struct {
    float *x, *xn, *q, *k, *v, *o, *mlp, *h1, *h2, *tmp;
    float *rprob, *gate, *up, *eout;
    float *kvk, *kvv;
    KvarnPlanes *kvpl;              /* [2 * max kv heads]: K planes then V planes */
    int moe_nu, moe_next, moe_chunk_n, moe_slots[128];
    float *gpu_g, *gpu_u, *gpu_d;
    /* Per-row CPU prefill scratch: avoids OpenMP fork/join for every row. */
    float *egate, *esx, *ehs;
    int8_t *exq, *ehq;
    int8_t *xq, *hq;
    float *sx, *hs;
    int *eidx;        /* [S * topk] chosen expert per (row, slot) */
    float *ewt;       /* [S * topk] weight */
    int *rows;        /* scratch: rows routed to one expert */
    int *uniq;        /* distinct experts in the batch */
    int S;
} Buf;

static void route_row(M *m, int li, const float *residual, int *idx, float *wts, Buf *b) {
Cfg *c = &m->c;
Layer *L = &m->L[li];
int D = c->hidden, E = c->n_experts, K = c->topk;
if (K <= 0 || K > E || K > MAXTOPK) {
fprintf(stderr, "invalid router topk=%d for %d experts (MAXTOPK=%d)\n", K, E, MAXTOPK);
exit(1);
}

rmsnorm(b->xn, residual, NULL, D, c->eps);
float rs = 1.0f / sqrtf((float)D);
for (int i = 0; i < D; i++) b->xn[i] *= L->router_scale.f[i] * rs;

/* Keep the selected experts in descending logit order while each logit is
* computed exactly once. Full softmax would preserve this order; its discarded
* denominator cancels when the selected probabilities are renormalised. */
const float *RP = L->router_proj.f;
float topv[MAXTOPK];
for (int j = 0; j < K; j++) { idx[j] = -1; topv[j] = -INFINITY; }
for (int e = 0; e < E; e++) {
float v = 0.0f;
for (int i = 0; i < D; i++) v += RP[(size_t)e * D + i] * b->xn[i];
int j = K;
while (j > 0 && v > topv[j - 1]) j--;
if (j == K) continue;
for (int t = K - 1; t > j; t--) { topv[t] = topv[t - 1]; idx[t] = idx[t - 1]; }
topv[j] = v; idx[j] = e;
}
float sum = 0.0f;
float mx = topv[0];
for (int j = 0; j < K; j++) { wts[j] = expf(topv[j] - mx); sum += wts[j]; }
for (int j = 0; j < K; j++) wts[j] = (wts[j] / sum) * L->router_pes.f[idx[j]];
}
/* Apply one loaded expert to every row that routed to it. For CPU prefill, all the
 * rows an expert selected go through two OpenMP passes rather than two parallel
 * regions per row. Each row keeps its own q4 activation quantisation and gate
 * scratch, so this changes scheduling only. */
static void expert_apply_batch_cpu(M *m, const uint8_t *G, const uint8_t *U,
const uint8_t *Dn, size_t grb, size_t drb, const float *X, float *OUT,
const int *rows, int nrows, const float *w, Buf *b) {
    Cfg *c = &m->c; int D = c->hidden, MI = c->moe_inter;
#ifdef COLI_F32ACT
#pragma omp parallel for schedule(static)
    for (int r = 0; r < nrows; r++) {
        const float *x = X + (size_t)rows[r] * D;
        float *gate = b->egate + (size_t)r * (MI + 64);
        for (int o = 0; o < MI; o++) gate[o] = gelu_tanh(q40_dot_f32(G + (size_t)o * grb, x, D)) * q40_dot_f32(U + (size_t)o * grb, x, D);
    }
#pragma omp parallel for schedule(static)
    for (int r = 0; r < nrows; r++) {
        float *out = OUT + (size_t)rows[r] * D;
        const float *gate = b->egate + (size_t)r * (MI + 64); float ww = w[r];
        for (int o = 0; o < D; o++) out[o] += ww * q40_dot_f32(Dn + (size_t)o * drb, gate, MI);
    }
#else
#pragma omp parallel for schedule(static)
    for (int r = 0; r < nrows; r++) {
        const float *x = X + (size_t)rows[r] * D;
        int8_t *xq = b->exq + (size_t)r * (D + 64);
        float *sx = b->esx + (size_t)r * (D / Q40_BLK + 8);
        float *gate = b->egate + (size_t)r * (MI + 64);
        q40_quant_act(x, xq, sx, D);
        for (int o = 0; o < MI; o++) gate[o] = gelu_tanh(q40_dot(G + (size_t)o * grb, xq, sx, D)) * q40_dot(U + (size_t)o * grb, xq, sx, D);
        q40_quant_act(gate, b->ehq + (size_t)r * (MI + 64), b->ehs + (size_t)r * (MI / Q40_BLK + 8), MI);
    }
#pragma omp parallel for schedule(static)
    for (int r = 0; r < nrows; r++) {
        float *out = OUT + (size_t)rows[r] * D;
        const int8_t *hq = b->ehq + (size_t)r * (MI + 64);
        const float *hs = b->ehs + (size_t)r * (MI / Q40_BLK + 8); float ww = w[r];
        for (int o = 0; o < D; o++) out[o] += ww * q40_dot(Dn + (size_t)o * drb, hq, hs, MI);
    }
#endif
}

static void expert_apply(M *m, const uint8_t *blob, const float *X, float *OUT,
                         const int *rows, int nrows, const float *w, Buf *b) {
    Cfg *c = &m->c;
    int D = c->hidden, MI = c->moe_inter;
    const uint8_t *G = blob, *U = blob + m->gate_b, *Dn = blob + 2 * m->gate_b;
    size_t grb = q40_row_bytes(D), drb = q40_row_bytes(MI);
    if (!g_use_gpu && nrows > 1) {
        expert_apply_batch_cpu(m, G, U, Dn, grb, drb, X, OUT, rows, nrows, w, b);
        return;
    }
    for (int r = 0; r < nrows; r++) {
        const float *x = X + (size_t)rows[r] * D;
        float *out = OUT + (size_t)rows[r] * D;

        if (g_use_gpu) {
            /* The expert blob was gpu_map'd when its slot was allocated, so the GPU
             * reads the very bytes the streaming cache pread into it, with no copy. */
            if (gpu_q40_matmul(b->gpu_g, G, x, MI, D, 1) &&
                gpu_q40_matmul(b->gpu_u, U, x, MI, D, 1)) {
                for (int o = 0; o < MI; o++)
                    b->gate[o] = gelu_tanh(b->gpu_g[o]) * b->gpu_u[o];
                if (gpu_q40_matmul(b->gpu_d, Dn, b->gate, D, MI, 1)) {
                    float ww = w[r];
                    for (int o = 0; o < D; o++) out[o] += ww * b->gpu_d[o];
                    continue;
                }
            }
            /* any decline above: fall through to the CPU path for this row */
        }
#ifndef COLI_F32ACT
        q40_quant_act(x, b->xq, b->sx, D);
#endif
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < MI; o++) {
#ifdef COLI_F32ACT
            float g = q40_dot_f32(G + (size_t)o * grb, x, D);
            float u = q40_dot_f32(U + (size_t)o * grb, x, D);
#else
            float g = q40_dot(G + (size_t)o * grb, b->xq, b->sx, D);
            float u = q40_dot(U + (size_t)o * grb, b->xq, b->sx, D);
#endif
            b->gate[o] = gelu_tanh(g) * u;
        }
        float ww = w[r];
#ifdef COLI_F32ACT
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < D; o++)
            out[o] += ww * q40_dot_f32(Dn + (size_t)o * drb, b->gate, MI);
#else
        q40_quant_act(b->gate, b->hq, b->hs, MI);
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < D; o++)
            out[o] += ww * q40_dot(Dn + (size_t)o * drb, b->hq, b->hs, MI);
#endif
    }
}

/* MoE over the whole batch: route every row, then read each distinct expert once. */
static void moe_wait(M *m) {
pthread_mutex_lock(&m->mu);
while (m->inflight) pthread_cond_wait(&m->done, &m->mu);
pthread_mutex_unlock(&m->mu);
}

/* Submit one chunk to the ring queue. Every returned slot is reserved until
* expert_apply has consumed it, so a later chunk cannot evict its bytes. */
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

/* Route and submit the first expert chunk. The caller can now compute the dense
* branch while the NVMe reads proceed. */
static void moe_start(M *m, int li, const float *residual, float *out, int S, Buf *b) {
Cfg *c = &m->c;
Layer *L = &m->L[li];
int D = c->hidden, K = c->topk;
for (int s = 0; s < S; s++)
    route_row(m, li, residual + (size_t)s * D, b->eidx + s * K, b->ewt + s * K, b);
b->moe_nu = 0;
for (int s = 0; s < S; s++) for (int j = 0; j < K; j++) {
int e = b->eidx[s * K + j], seen = 0;
for (int u = 0; u < b->moe_nu; u++) if (b->uniq[u] == e) { seen = 1; break; }
if (!seen) b->uniq[b->moe_nu++] = e;
}
float *X = b->eout + (size_t)S * D;
for (int s = 0; s < S; s++)
rmsnorm(X + (size_t)s * D, residual + (size_t)s * D, L->pre_ffn_ln2.f, D, c->eps);
memset(out, 0, sizeof(float) * (size_t)S * D);
/* Two resident chunks need disjoint unpinned slots. Pinned hits consume no
* eviction capacity, but this conservative bound also covers all-miss chunks. */
int free_slots = c->slots_per_layer - m->npin;
b->moe_chunk_n = free_slots > 1 ? free_slots / 2 : 1;
if (b->moe_chunk_n > c->slots_per_layer) b->moe_chunk_n = c->slots_per_layer;
if (b->moe_chunk_n > b->moe_nu) b->moe_chunk_n = b->moe_nu;
b->moe_next = b->moe_chunk_n;
if (b->moe_chunk_n) moe_submit_chunk(m, li, b, 0, b->moe_chunk_n);
}

static void moe_apply_chunk(M *m, int li, float *out, int S, Buf *b, int c0, int cn) {
Cfg *c = &m->c; int D = c->hidden, K = c->topk, SL = c->slots_per_layer;
float *X = b->eout + (size_t)S * D;
for (int u = 0; u < cn; u++) {
int e = b->uniq[c0 + u], n = 0;
for (int s = 0; s < S; s++) for (int j = 0; j < K; j++) if (b->eidx[s*K+j] == e) {
b->rows[n] = s; b->gate[c->moe_inter + n++] = b->ewt[s*K+j];
}
float wbuf[512], *wp = n <= 512 ? wbuf : malloc(sizeof(float) * n);
for (int r = 0; r < n; r++) wp[r] = b->gate[c->moe_inter + r];
expert_apply(m, m->slots[(size_t)li * SL + b->moe_slots[c0 + u]].buf, X, out, b->rows, n, wp, b);
if (wp != wbuf) free(wp);
}
moe_release_chunk(m, li, b, c0, cn);
}

static void moe_finish(M *m, int li, float *out, int S, Buf *b) {
Cfg *c = &m->c; Layer *L = &m->L[li]; int D = c->hidden;
int c0 = 0, cn = b->moe_chunk_n;
while (c0 < b->moe_nu) {
/* The first wait joins the dense-MLP overlap. Thereafter submit the next
* chunk before applying this one: its I/O overlaps CPU expert_apply, with no
* more than two chunks reserved and no active slot eligible for eviction. */
moe_wait(m);
int n0 = c0 + cn;
int nn = b->moe_nu - n0;
if (nn > b->moe_chunk_n) nn = b->moe_chunk_n;
if (nn > 0) moe_submit_chunk(m, li, b, n0, nn);
moe_apply_chunk(m, li, out, S, b, c0, cn);
c0 = n0; cn = nn;
}
for (int s = 0; s < S; s++)
rmsnorm(out + (size_t)s * D, out + (size_t)s * D, L->post_ffn_ln2.f, D, c->eps);
}
/* Write K/V for `pos`. The f32 residual ring holds at least the most recent `rwin`
 * positions; older ones are KVarN-encoded into the packed store on their way out, so
 * recent tokens always attend at full precision. 3-4 bit compression without a
 * residual window produces garbage.
 *
 * KVarN quantises a whole tile of `tile` consecutive tokens at once, so eviction is
 * not per position. The ring is a whole number of tiles (see kvarn_window), tiles
 * are aligned to absolute position, and the tile whose first slot is about to be
 * overwritten is sealed just before that write: exactly once, at the last moment
 * all of its tokens are still resident. */
static void kv_write(M *m, int li, int pos, int nkv, int hd, int cap,
                     const float *k, const float *v) {
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
        /* Seal only a tile that is complete and entirely confirmed. Speculation
         * breaks the "each position is written exactly once, in order" invariant:
         * the verify forward writes np..np+d-1, and on a partial acceptance the next
         * forward rewrites np+acc, which held a rejected draft. Sealing a tile with a
         * rejected draft in it poisons the quantised history unrecoverably, so check
         * that every slot still holds the position it should and that all of them are
         * confirmed. main() widens the ring by a tile plus the draft length whenever
         * speculation is on, so the margin W-g+1 always clears the draft and this
         * bails only if something has gone genuinely wrong. */
        for (int i = 0; i < g; i++) {
            int p = base + i;
            if (m->ring_pos[li][p % W] != p || p > m->kv_conf) { ready = 0; break; }
        }
        if (ready) {
            /* base % W == base % g == 0 modulo the tile, so the tile's g positions
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

/* Read K and V for position t into kb/vb ([nkv*hd] each).
 *
 * On a quantised layer both come back in the Hadamard-rotated frame, since that is
 * the frame the store already holds them in and attention is content to stay there:
 * the caller rotates its query once and transforms the accumulated output once,
 * instead of transforming every position it reads. The positions still in the f32
 * ring are rotated here to join them, at most `rwin` of those against a whole
 * context of cached ones. */
static void kv_read(M *m, int li, int t, int pos, int nkv, int hd, int cap,
                    KvarnPlanes *pl, float *kb, float *vb) {
    int quant = m->pk[li] != NULL;
    int W = quant ? (m->rwin < cap ? m->rwin : cap) : cap;
    size_t vec = (size_t)nkv * hd;

    /* In the f32 ring iff the slot really holds t (do not infer it from t > pos - W:
     * speculation can leave a slot holding a different position than the arithmetic
     * suggests). Sealing a tile does not retire its f32 copies: they stay readable
     * until their slots are reused, so a position keeps full precision for the whole
     * W writes exactly as it did under a per-vector codec. */
    if (!quant || m->ring_pos[li][t % W] == t) {
        memcpy(kb, m->kv_k[li] + (size_t)(t % W) * vec, sizeof(float) * vec);
        memcpy(vb, m->kv_v[li] + (size_t)(t % W) * vec, sizeof(float) * vec);
        if (quant)
            for (int h = 0; h < nkv; h++) {
                kvarn_rot(kb + (size_t)h * hd, hd);
                kvarn_rot(vb + (size_t)h * hd, hd);
            }
        return;
    }
    int g = m->tile;
    int ts = (t / g) % kvarn_ntiles(cap, g), col = t % g;
    /* pl is indexed by head and by K/V because here the head loop sits inside the
     * position loop: consecutive positions of one head share a tile. */
    for (int h = 0; h < nkv; h++) {
        size_t idx = (size_t)ts * nkv + h;
        kvarn_decode_raw(&m->qk[li], m->pk[li] + idx * m->qk[li].bytes, col,
                         &pl[h], kb + (size_t)h * hd);
        kvarn_decode_raw(&m->qv[li], m->pv[li] + idx * m->qv[li].bytes, col,
                         &pl[nkv + h], vb + (size_t)h * hd);
    }
}

static void layer_fwd(M *m, int li, float *H, int S, int pos_base, Buf *b) {
    Cfg *c = &m->c;
    Layer *L = &m->L[li];
    int D = c->hidden, glob = c->layer_types[li];
    int hd  = glob ? c->global_head_dim : c->head_dim;
    int nkv = glob ? c->n_global_kv_heads : c->n_kv_heads;
    int nh  = c->n_heads, rep = nh / nkv;
    float theta   = glob ? c->rope_theta_global : c->rope_theta_local;
    float partial = glob ? c->rope_partial_global : 1.0f;
    int cap = kv_cap(c, li);
    int kvq = m->pk[li] != NULL;      /* this layer's KV is compressed */

    /* attention
     * Positions are walked in order because a sliding layer's KV is a ring of
     * `sliding_window`: writing the whole batch first would overwrite keys that
     * earlier positions still need. Write-then-attend per position is correct for
     * any S, and the projections below are still done per row against weights that
     * are resident, so nothing is lost. */
    for (int s = 0; s < S; s++) {
        int pos = pos_base + s;
        float *h = H + (size_t)s * D;

        rmsnorm(b->xn, h, L->in_ln.f, D, c->eps);

        matvec(b->q, &L->q_proj, b->xn, b->xq, b->sx);
        for (int i = 0; i < nh; i++)
            rmsnorm(b->q + (size_t)i * hd, b->q + (size_t)i * hd, L->q_norm.f, hd, c->eps);
        rope(b->q, nh, hd, pos, theta, partial);
        /* Meet the cache in its own frame; see kv_read. After everything else that
         * touches q, since H has to be the last map applied to it. */
        if (kvq)
            for (int i = 0; i < nh; i++) kvarn_rot(b->q + (size_t)i * hd, hd);

        /* K = rope(k_norm(raw));  V = v_norm(raw), with neither scale nor rope.
         * On global layers there is no v_proj (attention_k_eq_v), so V reuses this
         * same k projection -- the raw one, before k_norm and rope, not K itself. */
        matvec(b->tmp, &L->k_proj, b->xn, b->xq, b->sx);
        if (L->has_v) {
            for (int i = 0; i < nkv; i++)
                rmsnorm(b->k + (size_t)i * hd, b->tmp + (size_t)i * hd, L->k_norm.f, hd, c->eps);
            matvec(b->tmp, &L->v_proj, b->xn, b->xq, b->sx);
            for (int i = 0; i < nkv; i++)
                rmsnorm(b->v + (size_t)i * hd, b->tmp + (size_t)i * hd, NULL, hd, c->eps);
        } else {
            for (int i = 0; i < nkv; i++)
                rmsnorm(b->v + (size_t)i * hd, b->tmp + (size_t)i * hd, NULL, hd, c->eps);
            for (int i = 0; i < nkv; i++)
                rmsnorm(b->k + (size_t)i * hd, b->tmp + (size_t)i * hd, L->k_norm.f, hd, c->eps);
        }
        rope(b->k, nkv, hd, pos, theta, partial);

        kv_write(m, li, pos, nkv, hd, cap, b->k, b->v);

        /* the backbone's causal sliding window: W positions, t >= pos - W + 1.
         * (The MTP head uses a different, one-wider window; see mtp_forward.) */
        int lo = glob ? 0 : (pos - c->sliding_window + 1 < 0 ? 0 : pos - c->sliding_window + 1);
        /* Online softmax: decode every KV position once and retain no ctx-sized
         * score or V scratch. m/z are per-head running maximum/normaliser. */
        float mx[256], z[256];
        if (nh > 256) { fprintf(stderr, "too many attention heads\n"); exit(1); }
        memset(b->o, 0, sizeof(float) * nh * hd);
        for (int hh = 0; hh < nh; hh++) { mx[hh] = -INFINITY; z[hh] = 0.0f; }
        for (int t = lo; t <= pos; t++) {
            kv_read(m, li, t, pos, nkv, hd, cap, b->kvpl, b->kvk, b->kvv);
            /* Heads are independent given this t; the online-softmax recurrence is
             * carried per head across t, so parallelise the head loop, not t. */
            #pragma omp parallel for schedule(static)
            for (int hh = 0; hh < nh; hh++) {
                const float *qq = b->q + (size_t)hh * hd;
                const float *kk = b->kvk + (size_t)(hh / rep) * hd;
                const float *vv = b->kvv + (size_t)(hh / rep) * hd;
                float score = head_dot(qq, kk, hd);
                float nm = score > mx[hh] ? score : mx[hh];
                float a = expf(mx[hh] - nm), w = expf(score - nm), nz = a * z[hh] + w;
                float *ov = b->o + (size_t)hh * hd;
                float old = z[hh] ? a * z[hh] / nz : 0.0f, add = w / nz;
                for (int d = 0; d < hd; d++) ov[d] = old * ov[d] + add * vv[d];
                mx[hh] = nm; z[hh] = nz;
            }
        }
        /* The accumulator is a weighted sum of rotated values and H is linear, so
         * one transform per head brings the whole sum back rather than one per
         * position. */
        if (kvq)
            for (int hh = 0; hh < nh; hh++) kvarn_rot(b->o + (size_t)hh * hd, hd);
        matvec(b->tmp, &L->o_proj, b->o, b->xq, b->sx);
        rmsnorm(b->tmp, b->tmp, L->post_attn_ln.f, D, c->eps);
        for (int i = 0; i < D; i++) h[i] += b->tmp[i];
    }

    /* feed-forward: the dense MLP and the MoE are parallel branches over separate
     * normalisations, both reading the post-attention residual H.
     *
     * The MoE goes first on purpose: routing needs only H, so the expert reads are
     * issued and then the dense MLP is computed while they are in flight. The
     * overlap is exact rather than predicted, which a synchronous mmap fault --
     * llama.cpp's expert path -- structurally cannot do. */
    float *res = b->eout;
    memcpy(res, H, sizeof(float) * (size_t)S * D);

    moe_start(m, li, res, b->h2, S, b);

    /* Dense work now overlaps the first chunk's asynchronous expert reads. Its
     * post-norm result goes into H, which res has already copied out of. */
    for (int s = 0; s < S; s++) {
        const float *r = res + (size_t)s * D;
        rmsnorm(b->xn, r, L->pre_ffn_ln.f, D, c->eps);
        matvec(b->gate, &L->mlp_gate, b->xn, b->xq, b->sx);
        matvec(b->up, &L->mlp_up, b->xn, b->xq, b->sx);
        for (int i = 0; i < c->dense_inter; i++) b->mlp[i] = gelu_tanh(b->gate[i]) * b->up[i];
        float *h1 = H + (size_t)s * D;
        matvec(h1, &L->mlp_down, b->mlp, b->xq, b->sx);
        rmsnorm(h1, h1, L->post_ffn_ln1.f, D, c->eps);
    }
    moe_finish(m, li, b->h2, S, b);
    for (int s = 0; s < S; s++) {
        const float *r = res + (size_t)s * D;
        float *h1 = H + (size_t)s * D, *h2 = b->h2 + (size_t)s * D;
        for (int i = 0; i < D; i++) b->tmp[i] = h1[i] + h2[i];
        rmsnorm(b->tmp, b->tmp, L->post_ffn_ln.f, D, c->eps);
        float ls = L->layer_scalar.f[0];
        for (int i = 0; i < D; i++) h1[i] = (r[i] + b->tmp[i]) * ls;
    }
}

/* The q4_0 embedding table doubles as the tied lm_head. */
static void embed_row(M *m, int tok, float *h) {
    Cfg *c = &m->c;
    int D = c->hidden;
    const uint8_t *row = m->L[MAXL - 1].q_proj.q + (size_t)tok * q40_row_bytes(D);
    for (int blk = 0; blk < D / Q40_BLK; blk++) {
        const uint8_t *bp = row + (size_t)blk * Q40_BLK_BYTES;
        uint16_t hd; memcpy(&hd, bp, 2);
        float d = q40_fp16_to_f32(hd);
        for (int j = 0; j < 16; j++) {
            h[blk * Q40_BLK + j]      = d * (float)((bp[2 + j] & 0x0f) - 8);
            h[blk * Q40_BLK + j + 16] = d * (float)((bp[2 + j] >> 4) - 8);
        }
    }
    for (int i = 0; i < D; i++) h[i] *= c->embed_scale;
}

/* Absorb `rows` captured target-hidden rows (positions pos0..pos0+rows-1) into the
 * persistent draft context KV cache: concat the target layers, project through
 * fc + hidden_norm, then per draft layer compute K (k_norm + rope at the row's own
 * absolute position) and V, stored at the absolute position. Overwrites are fine:
 * a position rewritten by a later forward always carries the confirmed token, so
 * the last write is the correct one (same argument as kv_write for the target).
 * This is the C equivalent of past_key_values_draft + crop(start). */
static void dflash_absorb(M *m, int pos0, int rows) {
    int D = m->dflash_D, ntl = m->dflash_n_target_layers;
    int nkv = m->dflash_nkv, hd = m->dflash_hd;
    size_t kvdim = (size_t)nkv * hd;
    for (int r = 0; r < rows; r++) {
        int p = pos0 + r;
        if (p >= m->c.ctx) break;
        for (int ti = 0; ti < ntl; ti++)
            memcpy(m->dfa_concat + (size_t)ti * D,
                   m->dflash_target_hidden + ((size_t)ti * rows + r) * D,
                   sizeof(float) * D);
        matvec(m->dfa_th, &m->dflash_fc, m->dfa_concat, m->dfa_xq, m->dfa_sx);
        rmsnorm(m->dfa_th, m->dfa_th, m->dflash_hidden_norm.f, D, m->dflash_eps);
        for (int li = 0; li < m->dflash_L; li++) {
            Layer *L = &m->dflash_layers[li];
            float *k = m->dflash_ctx_k[li] + (size_t)p * kvdim;
            float *v = m->dflash_ctx_v[li] + (size_t)p * kvdim;
            matvec(k, &L->k_proj, m->dfa_th, m->dfa_xq, m->dfa_sx);
            for (int i = 0; i < nkv; i++)
                rmsnorm(k + (size_t)i * hd, k + (size_t)i * hd,
                        L->k_norm.f, hd, m->dflash_eps);
            rope(k, nkv, hd, p, m->dflash_theta, 1.0f);
            matvec(v, &L->v_proj, m->dfa_th, m->dfa_xq, m->dfa_sx);
        }
    }
    if (pos0 + rows > m->dflash_ctx_len) m->dflash_ctx_len = pos0 + rows;
}

/* Run S tokens. logits may be NULL (prefill) or [S, vocab]; `last_only` covers the
 * common case where only the final row is wanted. */
static void forward(M *m, const int *ids, int S, int pos_base,
                    float *logits, int last_only, Buf *b) {
    Cfg *c = &m->c;
    int D = c->hidden;
    W *embed = &m->L[MAXL - 1].q_proj;
    W *fnorm = &m->L[MAXL - 1].o_proj;

    float *H = b->x;
    for (int s = 0; s < S; s++) embed_row(m, ids[s], H + (size_t)s * D);

    /* DFlash: capture hidden states at specific backbone layers. We snapshot H
     * after each target layer, before the next layer overwrites it. The hidden
     * states are post-norm (HF's hidden_states[layer_idx]). */
    if (m->dflash) {
        int ntl = m->dflash_n_target_layers;
        if (S > m->dflash_target_hidden_cap) {
            m->dflash_target_hidden_cap = S;
            free(m->dflash_target_hidden);
            m->dflash_target_hidden = xmalloc(sizeof(float) * (size_t)ntl * S * D);
        }
        for (int l = 0; l < c->n_layers; l++) {
            layer_fwd(m, l, H, S, pos_base, b);
            for (int ti = 0; ti < ntl; ti++) {
                if (m->dflash_target_ids[ti] == l) {
                    /* HF's hidden_states[l+1] is the raw output of layer l, without
                     * the final norm. DFlash's fc + hidden_norm was trained on those. */
                    for (int s = 0; s < S; s++)
                        memcpy(m->dflash_target_hidden + ((size_t)ti * S + s) * D,
                               H + (size_t)s * D, sizeof(float) * D);
                    break;
                }
            }
        }
        dflash_absorb(m, pos_base, S);
    } else {
        for (int l = 0; l < c->n_layers; l++) layer_fwd(m, l, H, S, pos_base, b);
    }

    /* Stash the post-norm hidden (HF's last_hidden_state) of every row for a small
     * batch, or just the final row for a long prefill. The head needs one specific
     * row, the last accepted one, which is only known after verification. */
    if (m->mtp || m->dflash) {
        if (S <= MAXDRAFT + 1) {
            for (int s = 0; s < S; s++)
                rmsnorm(m->hid_batch + (size_t)s * D, H + (size_t)s * D, fnorm->f, D, c->eps);
            m->hid_rows = S;
        } else {
            rmsnorm(m->hid_batch, H + (size_t)(S - 1) * D, fnorm->f, D, c->eps);
            m->hid_rows = 1;
        }
    }

    if (!logits) return;
    int s0 = last_only ? S - 1 : 0;
    for (int s = s0; s < S; s++) {
        rmsnorm(b->xn, H + (size_t)s * D, fnorm->f, D, c->eps);
        float *out = logits + (size_t)(last_only ? 0 : s) * c->vocab;
        matvec(out, embed, b->xn, b->xq, b->sx);          /* tied lm_head */
        if (c->final_logit_softcap > 0) {
            float cap = c->final_logit_softcap;
            for (int i = 0; i < c->vocab; i++) out[i] = tanhf(out[i] / cap) * cap;
        }
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

/* MTP head
 *
 * Gemma4AssistantForCausalLM is a draft head rather than a model. It has no k_proj,
 * no v_proj and no k_norm; num_kv_shared_layers == num_hidden_layers, so every one
 * of its layers takes K and V from the backbone's shared_kv_states[layer_type]. The
 * backbone publishes the KV of the last layer of each type (store_full_length_kv):
 * here, layer 28 (sliding) and layer 29 (full). The head's 3 sliding layers attend
 * into the former, its full layer into the latter.
 *
 * At q_len == 1 the assistant's bidirectional mask degenerates to full attention
 * over whatever KV it is given (its own comment says so), so we attend over every
 * cached position: for the sliding source that is the backbone's 1024-window, for
 * the full source the whole context. That is exactly what the engine caches.
 *
 * The draft loop comes from transformers' SinglePositionMultiTokenCandidateGenerator
 * (generation/candidate_generator.py). Four things live there rather than in the
 * model, and each is easy to get backwards:
 *
 *   1. inputs_embeds = cat([last_token_embedding, last_hidden_state], dim=-1) --
 *      the embedding comes first, then the hidden state.
 *
 *   2. last_token_embedding = target_model_input_embeddings(tok), which is
 *      get_input_embeddings(), the target's ScaledWordEmbedding: the embedding is
 *      multiplied by embed_scale. And last_hidden_state is hidden_states[-1],
 *      which for Gemma4TextModel comes after the final norm (verified: it is
 *      bit-identical to outputs.last_hidden_state).
 *
 *   3. position_ids = [[input_ids.shape[1] - 1]] is computed once, before the draft
 *      loop, and never advanced. Every drafted token is produced from the position
 *      of the last real token -- hence the class name, SinglePosition. Incrementing
 *      the position per draft step silently degrades every draft after the first.
 *
 *   4. The drafter is greedy: last_token_id = outputs.logits.argmax(-1), regardless
 *      of the target's sampling temperature. */

typedef struct {
    float *e, *h, *xn, *q, *o, *tmp, *gate, *up, *mlp, *kbuf, *vbuf;
    KvarnPlanes *kvpl;
    int8_t *xq; float *sx;
} MBuf;

static MBuf *mtp_bufs(M *m) {
    int D = m->mtp_D, BB = m->mtp_BB;
    int hd = m->mtp_ghd > m->mtp_hd ? m->mtp_ghd : m->mtp_hd;
    int qmax = m->mtp_nh * hd;
    int wide = D;
    if (wide < 2 * BB) wide = 2 * BB;
    if (wide < qmax) wide = qmax;
    if (wide < m->mtp_inter) wide = m->mtp_inter;
    if (wide < m->mtp_vocab) wide = m->mtp_vocab;

    MBuf *b = calloc(1, sizeof *b);
    b->e = xmalloc(sizeof(float) * 2 * BB);
    b->h = xmalloc(sizeof(float) * wide);
    b->xn = xmalloc(sizeof(float) * wide);
    b->q = xmalloc(sizeof(float) * qmax);
    b->o = xmalloc(sizeof(float) * qmax);
    b->tmp = xmalloc(sizeof(float) * wide);
    b->gate = xmalloc(sizeof(float) * (m->mtp_inter + 64));
    b->up = xmalloc(sizeof(float) * (m->mtp_inter + 64));
    b->mlp = xmalloc(sizeof(float) * (m->mtp_inter + 64));
    int mh = m->c.n_kv_heads > m->c.n_global_kv_heads ? m->c.n_kv_heads
                                                      : m->c.n_global_kv_heads;
    int mkv = m->c.n_kv_heads * m->c.head_dim;
    int gkv = m->c.n_global_kv_heads * m->c.global_head_dim;
    if (gkv > mkv) mkv = gkv;
    b->kbuf = xmalloc(sizeof(float) * mkv);
    b->vbuf = xmalloc(sizeof(float) * mkv);
    b->kvpl = xmalloc(sizeof(KvarnPlanes) * 2 * (size_t)mh);
    for (int i = 0; i < 2 * mh; i++) b->kvpl[i].rec = NULL;
    	b->xq = xmalloc(wide + 64);
    	b->sx = xmalloc(sizeof(float) * (wide / Q40_BLK + 8));
    	return b;
    }

    /* DFlash
     *
     * DFlash is a block-parallel drafter: instead of drafting one token at a time, it
     * drafts an entire block of `block_size` tokens simultaneously using bidirectional
     * attention. It conditions on hidden states extracted from specific layers of the
     * target backbone (e.g. layers 1, 6, 11, 17, 22, 27 for Gemma-4 26B).
     *
     * The attention is unusual: for each position in the draft block, K and V come from
     * the target hidden context (cross-attention) and from the draft block's own tokens
     * (bidirectional self-attention), concatenated along the sequence dim.
     *
     * The model is Qwen3-based rather than Gemma4, and its sliding_window affects
     * only the mask, not the backbone sense of the word. dflash_step runs the loop.
     */

    typedef struct {
        /* Per-row hidden states for the draft block [block_size * D] */
        float *h;
        /* Scratch: norm, q (per-position), tmp, gate, up, mlp */
        float *xn, *q, *tmp, *gate, *up, *mlp;
        /* Per-(position, head) attention output [BS * nh * hd] */
        float *ao;
        /* K/V of the draft block's own tokens [BS * nkv * hd] */
        float *nk, *nv;
        int8_t *xq; float *sx;
        /* Persistent logit/draft-prob buffers (avoid alloc/free per step) */
        float *dlog, *dprob;
    } DBuf;

    static DBuf *dflash_bufs(M *m) {
        int D = m->dflash_D, BS = m->dflash_block_size;
        int nh = m->dflash_nh, hd = m->dflash_hd, nkv = m->dflash_nkv;
        int qmax = nh * hd;          /* one position's query vector */
        int kvmax = nkv * hd;        /* one position's K/V */
        int wide = D;
        if (wide < qmax) wide = qmax;
        if (wide < m->dflash_inter) wide = m->dflash_inter;
        if (wide < m->dflash_vocab) wide = m->dflash_vocab;

        DBuf *b = calloc(1, sizeof *b);
        b->h = xmalloc(sizeof(float) * (size_t)BS * D);
        /* xn holds all BS normed rows contiguously so projections batch into one
         * matmul; gate/up/mlp likewise span the whole block. */
        b->xn = xmalloc(sizeof(float) * (size_t)BS * (wide < D ? D : wide));
        b->q = xmalloc(sizeof(float) * (size_t)BS * qmax);
        b->ao = xmalloc(sizeof(float) * (size_t)BS * qmax);
        b->tmp = xmalloc(sizeof(float) * (size_t)BS * wide);
        b->gate = xmalloc(sizeof(float) * ((size_t)BS * m->dflash_inter + 64));
        b->up = xmalloc(sizeof(float) * ((size_t)BS * m->dflash_inter + 64));
        b->mlp = xmalloc(sizeof(float) * ((size_t)BS * m->dflash_inter + 64));
        b->nk = xmalloc(sizeof(float) * (size_t)BS * kvmax);
        b->nv = xmalloc(sizeof(float) * (size_t)BS * kvmax);
        b->xq = xmalloc(wide + 64);
        b->sx = xmalloc(sizeof(float) * (wide / Q40_BLK + 8));
        /* Persistent logit/draft-prob buffers: [BS * V] each */
        b->dlog = xmalloc(sizeof(float) * (size_t)BS * m->dflash_vocab);
        b->dprob = xmalloc(sizeof(float) * (size_t)BS * m->dflash_vocab);
        return b;
    }

    /* DFlash attention for one layer. `S` is the number of draft positions,
     * `pos_base` the absolute position of the first one. Context K/V for positions
     * [0, pos_base) come from the persistent cache (m->dflash_ctx_k/v, filled by
     * dflash_absorb); only the block's own K/V are computed here. Bidirectional
     * within the block; layers 0-3 apply the 2048 sliding window over the context. */
    static void dflash_attn(M *m, int li, float *H, int S, int pos_base, DBuf *b) {
        int D = m->dflash_D, nh = m->dflash_nh, hd = m->dflash_hd;
        int nkv = m->dflash_nkv, rep = nh / nkv;
        size_t kvdim = (size_t)nkv * hd;
        float theta = m->dflash_theta;
        float scale = 1.0f / sqrtf((float)hd);
        Layer *L = &m->dflash_layers[li];
        int ctx = pos_base < m->dflash_ctx_len ? pos_base : m->dflash_ctx_len;
        int win = m->dflash_types[li] ? 0 : m->dflash_sliding_window;
        const float *CK = m->dflash_ctx_k[li], *CV = m->dflash_ctx_v[li];

        /* Q and the block's own K/V (all from the input_layernorm'd hidden)
         * All S rows are normed into b->xn contiguously, then each projection is a
         * single batched matmul over the block (one GPU dispatch instead of S). */
        for (int s = 0; s < S; s++)
            rmsnorm(b->xn + (size_t)s * D, H + (size_t)s * D, L->in_ln.f, D, m->dflash_eps);
        matmul(b->q,  &L->q_proj, b->xn, S, b->xq, b->sx);
        matmul(b->nk, &L->k_proj, b->xn, S, b->xq, b->sx);
        matmul(b->nv, &L->v_proj, b->xn, S, b->xq, b->sx);
        for (int s = 0; s < S; s++) {
            float *q = b->q + (size_t)s * nh * hd;
            for (int i = 0; i < nh; i++)
                rmsnorm(q + (size_t)i * hd, q + (size_t)i * hd,
                        L->q_norm.f, hd, m->dflash_eps);
            rope(q, nh, hd, pos_base + s, theta, 1.0f);
            float *k = b->nk + (size_t)s * kvdim;
            for (int i = 0; i < nkv; i++)
                rmsnorm(k + (size_t)i * hd, k + (size_t)i * hd,
                        L->k_norm.f, hd, m->dflash_eps);
            rope(k, nkv, hd, pos_base + s, theta, 1.0f);
        }

        /* Attention: each block position attends to the whole cached context
         * plus every block position (bidirectional). Two-pass softmax per (s, head)
         * with a thread-private score buffer; parallel over the S*nh tasks. */
        #pragma omp parallel for collapse(2) schedule(static)
        for (int s = 0; s < S; s++) {
            for (int hh = 0; hh < nh; hh++) {
                int qpos = pos_base + s;
                int lo = 0;
                if (win && qpos - win + 1 > 0) lo = qpos - win + 1;
                if (lo > ctx) lo = ctx;
                int nctx = ctx - lo, total = nctx + S;
                float sc[total];   /* <= sliding_window + block on 4 of 5 layers */
                const float *qq = b->q + ((size_t)s * nh + hh) * hd;
                size_t koff = (size_t)(hh / rep) * hd;
                float mx = -INFINITY;
                for (int t = 0; t < nctx; t++) {
                    const float *kk = CK + (size_t)(lo + t) * kvdim + koff;
                    float sco = 0.0f;
                    for (int d = 0; d < hd; d++) sco += qq[d] * kk[d];
                    sc[t] = sco * scale;
                    if (sc[t] > mx) mx = sc[t];
                }
                for (int t = 0; t < S; t++) {
                    const float *kk = b->nk + (size_t)t * kvdim + koff;
                    float sco = 0.0f;
                    for (int d = 0; d < hd; d++) sco += qq[d] * kk[d];
                    sc[nctx + t] = sco * scale;
                    if (sc[nctx + t] > mx) mx = sc[nctx + t];
                }
                float *ov = b->ao + ((size_t)s * nh + hh) * hd;
                memset(ov, 0, sizeof(float) * hd);
                float z = 0.0f;
                for (int t = 0; t < total; t++) {
                    float w = expf(sc[t] - mx);
                    z += w;
                    const float *vv = (t < nctx)
                        ? CV + (size_t)(lo + t) * kvdim + koff
                        : b->nv + (size_t)(t - nctx) * kvdim + koff;
                    for (int d = 0; d < hd; d++) ov[d] += w * vv[d];
                }
                float inv = 1.0f / z;
                for (int d = 0; d < hd; d++) ov[d] *= inv;
            }
        }

        /* output projection + residual (batched over the block) */
        matmul(b->tmp, &L->o_proj, b->ao, S, b->xq, b->sx);
        for (int s = 0; s < S; s++) {
            float *h = H + (size_t)s * D, *o = b->tmp + (size_t)s * D;
            for (int i = 0; i < D; i++) h[i] += o[i];
        }
    }

    /* DFlash forward: run all layers over the draft block.
     * H: [S * D] draft hidden states (initialized with noise embeddings).
     * pos_base: absolute position of the first draft token; context K/V for
     * positions [0, pos_base) come from the persistent draft cache. */
    static void dflash_forward(M *m, float *H, int S, int pos_base, DBuf *b) {
        int D = m->dflash_D;

        for (int li = 0; li < m->dflash_L; li++) {
            /* Post-attention residual add happens inside dflash_attn */
            dflash_attn(m, li, H, S, pos_base, b);

            /* FFN: plain MLP with SiLU, Qwen3-style rather than Gemma4's gelu_tanh.
             * Batched over the block: gate/up/down are each one matmul. */
            Layer *L = &m->dflash_layers[li];
            int MI = m->dflash_inter;
            for (int s = 0; s < S; s++)
                rmsnorm(b->xn + (size_t)s * D, H + (size_t)s * D,
                        L->post_attn_ln.f, D, m->dflash_eps);
            matmul(b->gate, &L->mlp_gate, b->xn, S, b->xq, b->sx);
            matmul(b->up, &L->mlp_up, b->xn, S, b->xq, b->sx);
            for (size_t i = 0; i < (size_t)S * MI; i++)
                b->mlp[i] = silu(b->gate[i]) * b->up[i];
            matmul(b->tmp, &L->mlp_down, b->mlp, S, b->xq, b->sx);
            for (int s = 0; s < S; s++) {
                float *h = H + (size_t)s * D, *o = b->tmp + (size_t)s * D;
                for (int i = 0; i < D; i++) h[i] += o[i];
            }
        }

        for (int s = 0; s < S; s++)
            rmsnorm(H + (size_t)s * D, H + (size_t)s * D, m->dflash_norm.f, D, m->dflash_eps);
    }

    /* Load the DFlash head from dflash.manifest.txt / dflash.bin. */
    static int dflash_load(M *m, const char *dir) {
        char p[4096];
        snprintf(p, sizeof p, "%s/dflash.manifest.txt", dir);
        FILE *f = fopen(p, "r");
        if (!f) return 0;

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
                        m->dflash_types[n++] = atoi(t);
                }
                else if (!strcmp(k, "target_layer_ids")) {
                    int n = 0;
                    for (char *t = strtok(v, " \n"); t && n < 16; t = strtok(NULL, " \n"))
                        m->dflash_target_ids[n++] = atoi(t);
                    m->dflash_n_target_layers = n;
                }
                else if (!strcmp(k, "hidden"))            m->dflash_D = atoi(v);
                else if (!strcmp(k, "n_layers"))          m->dflash_L = atoi(v);
                else if (!strcmp(k, "n_heads"))           m->dflash_nh = atoi(v);
                else if (!strcmp(k, "head_dim"))          m->dflash_hd = atoi(v);
                else if (!strcmp(k, "n_kv_heads"))        m->dflash_nkv = atoi(v);
                else if (!strcmp(k, "intermediate_size")) m->dflash_inter = atoi(v);
                else if (!strcmp(k, "vocab_size"))        m->dflash_vocab = atoi(v);
                else if (!strcmp(k, "eps"))               m->dflash_eps = atof(v);
                else if (!strcmp(k, "rope_theta"))        m->dflash_theta = atof(v);
                else if (!strcmp(k, "block_size"))        m->dflash_block_size = atoi(v);
                else if (!strcmp(k, "mask_token_id"))     m->dflash_mask_token_id = atoi(v);
                else if (!strcmp(k, "sliding_window"))    m->dflash_sliding_window = atoi(v);
                continue;
            }
            if (sscanf(line, "ndense %d", &ndense) == 1) { dd = calloc(ndense, sizeof *dd); continue; }
            char nm[96]; long long off, len; int fmt, O, I;
            if (sscanf(line, "dense %95s %lld %lld %d %d %d", nm, &off, &len, &fmt, &O, &I) == 6) {
                snprintf(dd[di].name, sizeof dd[di].name, "%s", nm);
                dd[di].off = off; dd[di].len = len; dd[di].fmt = fmt;
                dd[di].O = O; dd[di].I = I; di++;
            }
        }
        fclose(f);

        snprintf(p, sizeof p, "%s/dflash.bin", dir);
        int fd = open(p, O_RDONLY);
        if (fd < 0) { free(dd); return 0; }
        off_t sz = lseek(fd, 0, SEEK_END);
        uint8_t *blob = xmalloc(sz);
        for (off_t o = 0; o < sz;) {
            ssize_t r = pread(fd, blob + o, sz - o, o);
            if (r <= 0) { perror("read dflash.bin"); exit(1); }
            o += r;
        }
        close(fd);
        m->dflash_blob = blob;
        if (g_use_gpu) gpu_map(blob, (size_t)sz);

        m->dflash_fc = dense_bind(dd, ndense, blob, "fc");
        m->dflash_hidden_norm = dense_bind(dd, ndense, blob, "hidden_norm");
        m->dflash_norm = dense_bind(dd, ndense, blob, "norm");

        char nm[128];
        for (int l = 0; l < m->dflash_L; l++) {
            Layer *L = &m->dflash_layers[l];
            #define DB(f, str) do { snprintf(nm, sizeof nm, "layers.%d." str, l); \
                                    L->f = dense_bind(dd, ndense, blob, nm); } while (0)
            DB(in_ln, "input_layernorm");
            DB(post_attn_ln, "post_attention_layernorm");
            DB(q_proj, "q_proj");
            DB(k_proj, "k_proj");
            DB(v_proj, "v_proj");
            DB(o_proj, "o_proj");
            DB(q_norm, "q_norm");
            DB(k_norm, "k_norm");
            DB(mlp_gate, "mlp_gate");
            DB(mlp_up, "mlp_up");
            DB(mlp_down, "mlp_down");
            #undef DB
        }
        free(dd);

        if (m->dflash_vocab != m->c.vocab) {
            fprintf(stderr, "dflash: vocab %d != target %d\n", m->dflash_vocab, m->c.vocab);
            return 0;
        }
        if (m->dflash_D != m->c.hidden) {
            fprintf(stderr, "dflash: hidden %d != target %d\n", m->dflash_D, m->c.hidden);
            return 0;
        }

        /* Capture buffer starts small and grows to the largest forward batch (prefill). */
        m->dflash_target_hidden_cap = MAXDRAFT + 2;
        m->dflash_target_hidden = xmalloc(sizeof(float) * (size_t)m->dflash_n_target_layers
                                           * m->dflash_target_hidden_cap * m->c.hidden);
        /* Persistent draft context KV cache: one K/V per absolute position per layer. */
        size_t kvdim = (size_t)m->dflash_nkv * m->dflash_hd;
        for (int l = 0; l < m->dflash_L; l++) {
            m->dflash_ctx_k[l] = xmalloc(sizeof(float) * (size_t)m->c.ctx * kvdim);
            m->dflash_ctx_v[l] = xmalloc(sizeof(float) * (size_t)m->c.ctx * kvdim);
        }
        m->dflash_ctx_len = 0;
        /* absorb scratch */
        int wideA = m->dflash_n_target_layers * m->dflash_D;
        m->dfa_concat = xmalloc(sizeof(float) * wideA);
        m->dfa_th = xmalloc(sizeof(float) * m->dflash_D);
        m->dfa_kv = xmalloc(sizeof(float) * kvdim);
        m->dfa_xq = xmalloc(wideA + 64);
        m->dfa_sx = xmalloc(sizeof(float) * (wideA / Q40_BLK + 8));

        m->dflash = 1;
        fprintf(stderr, "dflash: %d layers, hidden %d, block_size %d, %d target layers\n",
                m->dflash_L, m->dflash_D, m->dflash_block_size, m->dflash_n_target_layers);
        fprintf(stderr, "dflash: target layer ids: ");
        for (int i = 0; i < m->dflash_n_target_layers; i++)
            fprintf(stderr, "%d ", m->dflash_target_ids[i]);
        fprintf(stderr, "\n");
        return 1;
    }

    /* one draft step. `bh` is the backbone-space hidden [2816] (in/out), `tok` the token
     * to condition on, `pos` its absolute position. Writes vocab logits. */
    static void mtp_forward(M *m, float *bh, int tok, int pos, float *logits, MBuf *b) {
    Cfg *c = &m->c;
    int D = m->mtp_D, BB = m->mtp_BB, nh = m->mtp_nh;

    /* concat(embed(tok), backbone_hidden) */
    float *emb = b->tmp;
    embed_row(m, tok, emb);                 /* includes embed_scale */
    memcpy(b->e, emb, sizeof(float) * BB);
    memcpy(b->e + BB, bh, sizeof(float) * BB);
    matvec(b->h, &m->mtp_pre, b->e, b->xq, b->sx);         /* [D] */

    /* 4 plain decoder layers, attending into the target's KV */
    for (int li = 0; li < m->mtp_L; li++) {
        Layer *L = &m->mtp_layers[li];
        int glob = m->mtp_types[li];
        /* the backbone layer whose KV this one shares */
        int src = glob ? m->kv_last_full : m->kv_last_slide;
        int shd = c->layer_types[src] ? c->global_head_dim : c->head_dim;
        int nkv = c->layer_types[src] ? c->n_global_kv_heads : c->n_kv_heads;
        int cap = kv_cap(c, src);
        int hd = glob ? m->mtp_ghd : m->mtp_hd;
        if (hd != shd) {                    /* head dims must line up with the source */
            fprintf(stderr, "mtp: head_dim %d != backbone kv head_dim %d\n", hd, shd);
            exit(1);
        }
        int rep = nh / nkv;
        float theta = glob ? m->mtp_theta_g : m->mtp_theta_l;
        float partial = glob ? m->mtp_partial : 1.0f;

        rmsnorm(b->xn, b->h, L->in_ln.f, D, m->mtp_eps);
        matvec(b->q, &L->q_proj, b->xn, b->xq, b->sx);
        for (int i = 0; i < nh; i++)
            rmsnorm(b->q + (size_t)i * hd, b->q + (size_t)i * hd, L->q_norm.f, hd, m->mtp_eps);
        rope(b->q, nh, hd, pos, theta, partial);
        /* The head reads backbone layer `src`'s cache, so it meets it in the same
         * frame the backbone does. */
        int kvq = m->pk[src] != NULL;
        if (kvq)
            for (int i = 0; i < nh; i++) kvarn_rot(b->q + (size_t)i * hd, hd);

        /* The head's mask is bidirectional:
         *   full layers    -> mask is None, attend every cached position;
         *   sliding layers -> attend t >= pos - W, i.e. W + 1 positions, one more
         *                     than the backbone's causal SWA (t >= pos - W + 1).
         * Verified against HF: getting this wrong (e.g. assuming q_len==1 means
         * "attend everything") changes the attention output by ~30%. */
        int lo = c->layer_types[src] ? 0
               : (pos - c->sliding_window < 0 ? 0 : pos - c->sliding_window);
        /* One-pass online softmax: no ctx-sized MTP score scratch and no KV reread. */
        float mx[256], z[256];
        if (nh > 256) { fprintf(stderr, "too many MTP heads\n"); exit(1); }
        memset(b->o, 0, sizeof(float) * nh * hd);
        for (int hh = 0; hh < nh; hh++) { mx[hh] = -INFINITY; z[hh] = 0.0f; }
        for (int t = lo; t <= pos; t++) {
            kv_read(m, src, t, pos, nkv, hd, cap, b->kvpl, b->kbuf, b->vbuf);
            for (int hh = 0; hh < nh; hh++) {
                const float *qq = b->q + (size_t)hh * hd;
                const float *kk = b->kbuf + (size_t)(hh / rep) * hd;
                const float *vv = b->vbuf + (size_t)(hh / rep) * hd;
                float score = head_dot(qq, kk, hd);
                float nm = score > mx[hh] ? score : mx[hh];
                float a = expf(mx[hh] - nm), w = expf(score - nm), nz = a * z[hh] + w;
                float *ov = b->o + (size_t)hh * hd;
                float old = z[hh] ? a * z[hh] / nz : 0.0f, add = w / nz;
                for (int d = 0; d < hd; d++) ov[d] = old * ov[d] + add * vv[d];
                mx[hh] = nm; z[hh] = nz;
            }
        }
        if (kvq)
            for (int hh = 0; hh < nh; hh++) kvarn_rot(b->o + (size_t)hh * hd, hd);
        matvec(b->tmp, &L->o_proj, b->o, b->xq, b->sx);
        rmsnorm(b->tmp, b->tmp, L->post_attn_ln.f, D, m->mtp_eps);
        for (int i = 0; i < D; i++) b->h[i] += b->tmp[i];

        /* plain (non-MoE) FFN: no router, no experts, no extra norms */
        rmsnorm(b->xn, b->h, L->pre_ffn_ln.f, D, m->mtp_eps);
        matvec(b->gate, &L->mlp_gate, b->xn, b->xq, b->sx);
        matvec(b->up, &L->mlp_up, b->xn, b->xq, b->sx);
        for (int i = 0; i < m->mtp_inter; i++) b->mlp[i] = gelu_tanh(b->gate[i]) * b->up[i];
        matvec(b->tmp, &L->mlp_down, b->mlp, b->xq, b->sx);
        rmsnorm(b->tmp, b->tmp, L->post_ffn_ln.f, D, m->mtp_eps);
        float ls = L->layer_scalar.f[0];
        for (int i = 0; i < D; i++) b->h[i] = (b->h[i] + b->tmp[i]) * ls;
    }

    rmsnorm(b->xn, b->h, m->mtp_norm.f, D, m->mtp_eps);   /* last_hidden_state */
    matvec(logits, &m->mtp_embed, b->xn, b->xq, b->sx);   /* tied lm_head, no softcap */
    matvec(bh, &m->mtp_post, b->xn, b->xq, b->sx);        /* back to backbone space */
}

/* Load the MTP head from mtp.manifest.txt / mtp.bin, and work out which backbone
 * layers publish the KV it shares (the last layer of each layer_type). */
static int mtp_load(M *m, const char *dir) {
    char p[4096];
    snprintf(p, sizeof p, "%s/mtp.manifest.txt", dir);
    FILE *f = fopen(p, "r");
    if (!f) return 0;

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
                    m->mtp_types[n++] = atoi(t);
            }
            else if (!strcmp(k, "hidden"))            m->mtp_D = atoi(v);
            else if (!strcmp(k, "backbone_hidden"))   m->mtp_BB = atoi(v);
            else if (!strcmp(k, "n_layers"))          m->mtp_L = atoi(v);
            else if (!strcmp(k, "n_heads"))           m->mtp_nh = atoi(v);
            else if (!strcmp(k, "head_dim"))          m->mtp_hd = atoi(v);
            else if (!strcmp(k, "global_head_dim"))   m->mtp_ghd = atoi(v);
            else if (!strcmp(k, "dense_inter"))       m->mtp_inter = atoi(v);
            else if (!strcmp(k, "vocab"))             m->mtp_vocab = atoi(v);
            else if (!strcmp(k, "eps"))               m->mtp_eps = atof(v);
            else if (!strcmp(k, "rope_theta_local"))  m->mtp_theta_l = atof(v);
            else if (!strcmp(k, "rope_theta_global")) m->mtp_theta_g = atof(v);
            else if (!strcmp(k, "rope_partial_global")) m->mtp_partial = atof(v);
            continue;
        }
        if (sscanf(line, "ndense %d", &ndense) == 1) { dd = calloc(ndense, sizeof *dd); continue; }
        char nm[96]; long long off, len; int fmt, O, I;
        if (sscanf(line, "dense %95s %lld %lld %d %d %d", nm, &off, &len, &fmt, &O, &I) == 6) {
            snprintf(dd[di].name, sizeof dd[di].name, "%s", nm);
            dd[di].off = off; dd[di].len = len; dd[di].fmt = fmt;
            dd[di].O = O; dd[di].I = I; di++;
        }
    }
    fclose(f);

    snprintf(p, sizeof p, "%s/mtp.bin", dir);
    int fd = open(p, O_RDONLY);
    if (fd < 0) { free(dd); return 0; }
    off_t sz = lseek(fd, 0, SEEK_END);
    uint8_t *blob = xmalloc(sz);
    for (off_t o = 0; o < sz;) {
        ssize_t r = pread(fd, blob + o, sz - o, o);
        if (r <= 0) { perror("read mtp.bin"); exit(1); }
        o += r;
    }
    close(fd);
    m->mtp_blob = blob;
    if (g_use_gpu) gpu_map(blob, (size_t)sz);

    m->mtp_embed = dense_bind(dd, ndense, blob, "embed_tokens");
    m->mtp_norm  = dense_bind(dd, ndense, blob, "norm");
    m->mtp_pre   = dense_bind(dd, ndense, blob, "pre_projection");
    m->mtp_post  = dense_bind(dd, ndense, blob, "post_projection");

    char nm[128];
    for (int l = 0; l < m->mtp_L; l++) {
        Layer *L = &m->mtp_layers[l];
        #define MB(f, str) do { snprintf(nm, sizeof nm, "layers.%d." str, l); \
                                L->f = dense_bind(dd, ndense, blob, nm); } while (0)
        MB(in_ln, "input_layernorm");
        MB(post_attn_ln, "post_attention_layernorm");
        MB(pre_ffn_ln, "pre_feedforward_layernorm");
        MB(post_ffn_ln, "post_feedforward_layernorm");
        MB(layer_scalar, "layer_scalar");
        MB(q_proj, "q_proj");
        MB(o_proj, "o_proj");
        MB(q_norm, "q_norm");
        MB(mlp_gate, "mlp_gate");
        MB(mlp_up, "mlp_up");
        MB(mlp_down, "mlp_down");
        #undef MB
    }
    free(dd);

    if (m->mtp_BB != m->c.hidden) {
        fprintf(stderr, "mtp: backbone_hidden %d != target hidden %d\n",
                m->mtp_BB, m->c.hidden);
        return 0;
    }

    /* the backbone publishes the KV of the last layer of each type
     * (store_full_length_kv); that is what the head shares. */
    m->kv_last_slide = m->kv_last_full = -1;
    for (int l = 0; l < m->c.n_layers; l++) {
        if (m->c.layer_types[l]) m->kv_last_full = l;
        else                     m->kv_last_slide = l;
    }
    if (m->kv_last_slide < 0 || m->kv_last_full < 0) return 0;

    m->hid_batch = xmalloc(sizeof(float) * (size_t)(MAXDRAFT + 2) * m->c.hidden);
    m->mtp = 1;
    fprintf(stderr, "mtp: %d layers, hidden %d; shares KV of backbone layers "
            "%d (sliding) and %d (full)\n",
            m->mtp_L, m->mtp_D, m->kv_last_slide, m->kv_last_full);
    return 1;
}

/* setup */
static void init(M *m, const char *dir, int n_io, int ctx_override, double ram_gb,
                 int kb, int vb, int rwin, int tile) {
    m->tile = tile > 0 ? tile : 128;
    /* KVARN_RWIN is a floor, not the ring size: KVarN seals whole tiles, so the
     * ring is rounded up to a whole number of them (kvarn_window). */
    m->rwin = kvarn_window(rwin > 0 ? rwin : 128, m->tile);
    manifest(m, dir);
    Cfg *c = &m->c;

    /* --ctx overrides what the container was converted with, and may go up: the
     * container's ctx only fixed slots_per_layer (the expert-cache plan), and the
     * weights do not care. Only the global layers grow anyway; the sliding ones are
     * capped by sliding_window at any context.
     *
     * Whether that is a problem depends on bytes rather than on the context number:
     * with --kv a much longer context can fit in less than the plan assumed. The
     * comparison is made below, once the real figure is known. */
    int ctx_planned = c->ctx;
    if (ctx_override > 0) c->ctx = ctx_override;
    m->ucount = calloc((size_t)c->n_layers * c->n_experts, 8);
    snprintf(m->usage_path, sizeof m->usage_path, "%s/usage.bin", dir);

    m->kv_k = calloc(c->n_layers, sizeof(float *));
    m->kv_v = calloc(c->n_layers, sizeof(float *));
    m->ring_pos = calloc(c->n_layers, sizeof(int *));
    m->pk = calloc(c->n_layers, sizeof(uint8_t *));
    m->pv = calloc(c->n_layers, sizeof(uint8_t *));
    m->qk = calloc(c->n_layers, sizeof(Kvarn));
    m->qv = calloc(c->n_layers, sizeof(Kvarn));

    size_t kvb = 0;
    for (int l = 0; l < c->n_layers; l++) {
        int glob = c->layer_types[l];
        int hd = glob ? c->global_head_dim : c->head_dim;
        int nkv = glob ? c->n_global_kv_heads : c->n_kv_heads;
        int cap = kv_cap(c, l);          /* sliding ring is W+1: see kv_cap */

        /* Every layer gets the preset's own bit widths. The previous codec kept the
         * first and last few layers at 8 bits because it could not hold K4/V2
         * anywhere; KVarN's whole claim is that the calibrated recipe needs no such
         * hedge, and a per-layer override would mean --kv kvarn_k4v2_g128 did not
         * actually give you kvarn_k4v2_g128. */
        int quant = kb > 0;

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
    fprintf(stderr, "kv: %.0f MiB for ctx %d", kvb / 1048576.0, c->ctx);
    if (kb > 0) fprintf(stderr, " (KVarN K%d/V%d, tile %d, f32 ring %d)",
                        kb, vb, m->tile, m->rwin);
    else        fprintf(stderr, " (f32, by --kv off)");
    fprintf(stderr, "\n");

    /* Compare against what the conversion budgeted: an f32 KV at the container's own
     * ctx. Only a real overshoot is worth a warning, since --kv routinely buys a
     * longer context for fewer bytes than the plan assumed. */
    {
        size_t planned = 0;
        for (int l = 0; l < c->n_layers; l++) {
            int glob = c->layer_types[l];
            int hd  = glob ? c->global_head_dim : c->head_dim;
            int nkv = glob ? c->n_global_kv_heads : c->n_kv_heads;
            int cap = glob ? ctx_planned : c->sliding_window + MAXDRAFT + 2;
            planned += 2 * sizeof(float) * (size_t)cap * nkv * hd;
        }
        /* With an explicit --ram the cache is about to be re-planned against this
         * very figure, so the overshoot is already accounted for. */
        if (kvb > planned && ram_gb <= 0)
            fprintf(stderr, "warning: that is %.0f MiB more KV than the container's "
                    "plan budgeted (%.0f MiB for ctx %d, f32), so total RAM will "
                    "exceed the conversion's --ram%s\n",
                    (kvb - planned) / 1048576.0, planned / 1048576.0, ctx_planned,
                    kb > 0 ? "" : "; dropping --kv off would cut it a lot");
    }

    /* expert-cache plan
     * slots_per_layer is the only thing the conversion's --ram fixed, and nothing in
     * the container depends on it: experts are read from experts.bin one at a time by
     * offset and the cache is a plain LRU. So --ram re-runs the planner of
     * tools/convert_gemma4.py against the figures we now know -- the resident dense
     * blob and the KV just allocated -- instead of the ones the conversion had to
     * assume (an f32 KV at the container's own ctx). */
    if (ram_gb > 0) {
        int64_t scratch = 192 << 20;
        int64_t avail = (int64_t)(ram_gb * (double)(1LL << 30))
                      - (int64_t)m->dense_len - (int64_t)kvb - scratch;
        int64_t per = avail > 0 ? (avail / (int64_t)m->esz) / c->n_layers : 0;
        if (per > c->n_experts) per = c->n_experts;
        if (per < c->topk) {
            double min_gb = ((double)m->dense_len + (double)kvb + (double)scratch
                             + (double)c->topk * c->n_layers * (double)m->esz)
                            / (double)(1LL << 30);
            fprintf(stderr, "--ram %g GB leaves room for %lld experts per layer, below "
                    "topk=%d: this model needs %.2f GB at this context%s\n",
                    ram_gb, (long long)per, c->topk, min_gb,
                    kb > 0 ? "" : " (dropping --kv off would lower it)");
            exit(1);
        }
        c->slots_per_layer = (int)per;
        fprintf(stderr, "ram: %g GB budget -> %d slots/layer "
                "(dense %.0f MiB + kv %.0f MiB + cache %.0f MiB)\n",
                ram_gb, c->slots_per_layer, m->dense_len / 1048576.0, kvb / 1048576.0,
                (double)c->slots_per_layer * c->n_layers * (double)m->esz / 1048576.0);
    }
    if (c->slots_per_layer < c->topk) {
        fprintf(stderr, "slots_per_layer=%d < topk=%d: raise --ram\n",
                c->slots_per_layer, c->topk);
        exit(1);
    }
    {
        size_t ns = (size_t)c->n_layers * c->slots_per_layer;
        m->slots = calloc(ns, sizeof(Slot));
        for (size_t i = 0; i < ns; i++) {
            m->slots[i].eid = -1;
            m->slots[i].buf = xmalloc(m->esz);
            /* Map each slot once, up front. Slots are reused for different experts,
             * and the mapping stays valid for the whole run: the streaming layer
             * overwrites the bytes, never the address. */
            if (g_use_gpu && !gpu_map(m->slots[i].buf, m->esz)) {
                fprintf(stderr, "metal: could not map expert slots; using CPU\n");
                g_use_gpu = 0;
            }
        }
    }

    m->kv_conf = INT_MAX;                   /* nothing speculative yet */

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
    int hd = c->global_head_dim > c->head_dim ? c->global_head_dim : c->head_dim;
    int qmax = c->n_heads * hd;
    int imax = c->dense_inter > c->moe_inter ? c->dense_inter : c->moe_inter;
    int wide = qmax > c->vocab ? qmax : c->vocab;
    if (wide < D) wide = D;
    if (wide < imax) wide = imax;

    Buf *b = calloc(1, sizeof *b);
    b->S = Smax;
    b->x   = xmalloc(sizeof(float) * (size_t)Smax * D);      /* [S,D] hidden */
    b->eout= xmalloc(sizeof(float) * (size_t)2 * Smax * D);  /* residual + expert inputs */
    b->h2  = xmalloc(sizeof(float) * (size_t)Smax * D);
    b->xn  = xmalloc(sizeof(float) * wide);
    b->q   = xmalloc(sizeof(float) * qmax);
    b->k   = xmalloc(sizeof(float) * qmax);
    b->v   = xmalloc(sizeof(float) * qmax);
    b->o   = xmalloc(sizeof(float) * qmax);
    b->tmp = xmalloc(sizeof(float) * wide);
    b->mlp = xmalloc(sizeof(float) * (imax + 64));
    b->h1  = xmalloc(sizeof(float) * wide);
    /* gate doubles as a weight stash in moe_batch: needs moe_inter + Smax*topk */
    b->gate= xmalloc(sizeof(float) * (imax + (size_t)Smax * K + 64));
    b->up  = xmalloc(sizeof(float) * (imax + 64));
    b->rprob = xmalloc(sizeof(float) * (c->n_experts + 64));
    b->xq  = xmalloc(wide + 64);
    b->hq  = xmalloc(wide + 64);
    b->sx  = xmalloc(sizeof(float) * (wide / Q40_BLK + 8));
    b->hs  = xmalloc(sizeof(float) * (wide / Q40_BLK + 8));
    b->eidx = xmalloc(sizeof(int) * (size_t)Smax * K);
    b->ewt  = xmalloc(sizeof(float) * (size_t)Smax * K);
    b->rows = xmalloc(sizeof(int) * (size_t)Smax * K);
    b->uniq = xmalloc(sizeof(int) * c->n_experts);
    int mkv = c->n_kv_heads * c->head_dim;
    int gkv = c->n_global_kv_heads * c->global_head_dim;
    if (gkv > mkv) mkv = gkv;
    int mh = c->n_kv_heads > c->n_global_kv_heads ? c->n_kv_heads
                                                  : c->n_global_kv_heads;
    b->kvk = xmalloc(sizeof(float) * mkv);
    b->kvv = xmalloc(sizeof(float) * mkv);
    b->kvpl = xmalloc(sizeof(KvarnPlanes) * 2 * (size_t)mh);
    for (int i = 0; i < 2 * mh; i++) b->kvpl[i].rec = NULL;
    b->gpu_g = xmalloc(sizeof(float) * (c->moe_inter + 64));
    b->gpu_u = xmalloc(sizeof(float) * (c->moe_inter + 64));
    b->gpu_d = xmalloc(sizeof(float) * (D + 64));
    b->egate = xmalloc(sizeof(float) * (size_t)Smax * (c->moe_inter + 64));
    b->exq = xmalloc((size_t)Smax * (D + 64));
    b->ehq = xmalloc((size_t)Smax * (c->moe_inter + 64));
    b->esx = xmalloc(sizeof(float) * (size_t)Smax * (D / Q40_BLK + 8));
    b->ehs = xmalloc(sizeof(float) * (size_t)Smax * (c->moe_inter / Q40_BLK + 8));
    return b;
}

/* sampling */
typedef struct { float p; int i; } PI;
static int pi_desc(const void *a, const void *b) {
    float x = ((const PI *)a)->p, y = ((const PI *)b)->p;
    return x < y ? 1 : x > y ? -1 : 0;
}
/* temperature + top-k + nucleus. Greedy when temp <= 0.
 * Gemma-4's own generation defaults are temp 1.0, top_p 0.95, top_k 64; top_k is
 * applied BEFORE top_p, which is the order HF uses. */
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
    if (topk > 0 && topk < n) n = topk;          /* top-k first ... */

    double c = 0;                                /* ... then nucleus within it */
    int m = n;
    for (int i = 0; i < n; i++) {
        c += buf[i].p;
        if (c >= topp) { m = i + 1; break; }
    }
    n = m;
    double tot = 0;
    for (int i = 0; i < n; i++) tot += buf[i].p;

    *rng = *rng * 6364136223846793005ULL + 1442695040888963407ULL;
    double r = ((*rng >> 11) / (double)(1ULL << 53)) * tot;
    double acc = 0;
    for (int i = 0; i < n; i++) { acc += buf[i].p; if (r <= acc) return buf[i].i; }
    return buf[n - 1].i;
}

/* Transcribed from Gemma-4's chat_template.jinja. The non-obvious part is the
 * trailing empty thought channel: with thinking disabled the template pre-fills one,
 * and that is what suppresses reasoning. Omit it and the model starts thinking. */
static void chat_prompt(char *out, size_t cap, const char *sys,
                        const char *user, int think) {
    size_t n = 0;
    #define ADD(...) n += snprintf(out + n, n < cap ? cap - n : 0, __VA_ARGS__)
    ADD("<bos>");
    if (think || (sys && *sys)) {
        ADD("<|turn>system\n");
        if (think) ADD("<|think|>\n");
        if (sys && *sys) ADD("%s", sys);
        ADD("<turn|>\n");
    }
    ADD("<|turn>user\n%s<turn|>\n", user ? user : "");
    ADD("<|turn>model\n");
    if (!think) ADD("<|channel>thought\n<channel|>");   /* empty channel = no reasoning */
    #undef ADD
}

/* speculative decoding
 *
 * Gemma-4 ships a dedicated MTP drafter as a separate model, unlike GLM's in-model
 * layer-78 head. So speculation is: draft d tokens cheaply with the small model,
 * then verify all d with a single batched forward of the big one.
 *
 * The verification forward is where this pays off for a streaming engine. Those d
 * positions collectively route to at most min(128, 8*d) distinct experts per layer,
 * and batch-union reads each one once, so verifying d tokens costs barely more
 * expert I/O than decoding one: d tokens of disk latency become 1.
 *
 * It is lossless. At temp=0 acceptance is exact: keep a draft token iff it is the
 * target's argmax. At temp>0 we use rejection sampling against the draft's own
 * distribution, so the output distribution matches non-speculative sampling.
 * Speculation changes the speed, not the text.
 *
 * If we reject at position j, the KV written for positions >= pos+j is garbage. We
 * do not need to undo it: every KV slot is addressed by absolute position
 * (`pos % cap`), so the next forward simply overwrites it. */

/* `dpos` counts how many tokens of `ids` the drafter has actually consumed into its
 * KV. It has to be tracked: the drafter emits draft[d-1] but never feeds it back, so
 * on full acceptance that position joins the accepted prefix with stale KV behind it
 * and the drafter starts disagreeing with itself (observed: an identical
 * drafter/target pair accepting only 75%). We catch it up on the accepted prefix
 * and no further. */
static int spec_step(M *tgt, M *drf, Buf *bt, Buf *bd,
                     int *ids, int np, int *dpos, float *tlog, float *dlog,
                     int d, float temp, float topp, int topk, PI *pbuf, PI *dbuf,
                     uint64_t *rng, int *out, int *accepted) {
    Cfg *c = &tgt->c;
    int V = c->vocab;
    int draft[MAXDRAFT];
    float *dprob = xmalloc(sizeof(float) * (size_t)d * V);

    /* catch the drafter up on every accepted token it has not seen */
    if (*dpos < np) {
        forward(drf, ids + *dpos, np - *dpos, *dpos, dlog, 1, bd);
        *dpos = np;
    }

    /* draft d tokens */
    for (int i = 0; i < d; i++) {
        /* draft distribution at this step (needed for rejection sampling) */
        float mx = -1e30f;
        for (int j = 0; j < V; j++) if (dlog[j] > mx) mx = dlog[j];
        double sum = 0;
        float T = temp > 0 ? temp : 1.0f;
        float *dp = dprob + (size_t)i * V;
        for (int j = 0; j < V; j++) { dp[j] = expf((dlog[j] - mx) / T); sum += dp[j]; }
        for (int j = 0; j < V; j++) dp[j] /= (float)sum;

        draft[i] = sample(dlog, V, temp, topp, topk, dbuf, rng);
        /* feed it back so the next draft is conditioned on it */
        forward(drf, &draft[i], 1, np + i, dlog, 1, bd);
    }

    /* verify all d with a single batched target forward.
     * tlog already holds the target's distribution for position np-1 (i.e. for
     * draft[0]); this call returns the distributions after each drafted token. */
    float *vl = xmalloc(sizeof(float) * (size_t)d * V);
    forward(tgt, draft, d, np, vl, 0, bt);

    int n = 0;
    const float *q = tlog;                       /* target dist for draft[n] */
    for (n = 0; n < d; n++) {
        int tok = draft[n];
        int ok;
        if (temp <= 0) {
            int am = 0;
            for (int j = 1; j < V; j++) if (q[j] > q[am]) am = j;
            ok = (am == tok);
            if (ok) out[n] = tok;
        } else {
            /* rejection sampling: accept with prob min(1, q(tok)/p(tok)) */
            float mx = -1e30f;
            for (int j = 0; j < V; j++) if (q[j] > mx) mx = q[j];
            double sum = 0;
            for (int j = 0; j < V; j++) { pbuf[j].p = expf((q[j] - mx) / temp); sum += pbuf[j].p; }
            float qt = (float)(pbuf[tok].p / sum);
            float pt = dprob[(size_t)n * V + tok];
            *rng = *rng * 6364136223846793005ULL + 1442695040888963407ULL;
            double r = (*rng >> 11) / (double)(1ULL << 53);
            ok = (pt <= 0) || (r < (double)qt / pt);
            if (ok) out[n] = tok;
        }
        if (!ok) break;
        q = vl + (size_t)n * V;                  /* target dist after accepting draft[n] */
    }

    /* n tokens accepted. Emit one more from the target's distribution at the first
     * rejected position (or the bonus token if all d were accepted). */
    out[n] = sample(q, V, temp, topp, topk, pbuf, rng);
    *accepted = n;

    /* The drafter consumed draft[0..d-1] at positions np..np+d-1, but only the first
     * n of those are on the accepted path. Everything past that is a rejected branch
     * with invalid KV, so mark it unconsumed and the next step re-feeds it. */
    *dpos = np + n;

    free(dprob);
    free(vl);
    return n + 1;                                /* tokens produced this step */
}

/* Numerically diff the Metal kernel against the CPU reference on random q4_0 data.
 * The Metal path cannot be tested where it was written; the machine it runs on
 * decides whether it is right, so run this once before trusting any GPU output.
 * Shapes cover the real ones: attention projections, dense MLP, both expert
 * matmuls. */
static int check_gpu(void) {
    if (!gpu_ready()) {
        printf("no Metal device (or built without COLI_METAL) -- nothing to check\n");
        return 0;
    }
    printf("metal device: %s\n\n", gpu_name());
    struct { int O, I; const char *what; } shp[] = {
        {4096, 2816, "q_proj (sliding)"},
        {8192, 2816, "q_proj (global)"},
        {2112, 2816, "mlp gate/up"},
        {2816, 2112, "mlp down"},
        { 704, 2816, "expert gate/up"},
        {2816,  704, "expert down"},
    };
    int fail = 0;
    for (unsigned t = 0; t < sizeof shp / sizeof *shp; t++) {
        int O = shp[t].O, I = shp[t].I;
        size_t wb = (size_t)O * q40_row_bytes(I);
        uint8_t *W = xmalloc((wb + 4095) & ~(size_t)4095);
        float *x = xmalloc(sizeof(float) * I);
        float *yg = xmalloc(sizeof(float) * O), *yc = xmalloc(sizeof(float) * O);

        uint64_t r = 0x243f6a8885a308d3ULL ^ t;
        for (size_t i = 0; i < wb; i++) {
            r = r * 6364136223846793005ULL + 1442695040888963407ULL;
            W[i] = (uint8_t)(r >> 40);
        }
        for (int i = 0; i < I; i++) {
            r = r * 6364136223846793005ULL + 1442695040888963407ULL;
            x[i] = (float)((int64_t)(r >> 40) - 8388608) / 8388608.0f;
        }
        if (!gpu_map(W, wb)) { printf("  %-18s gpu_map FAILED\n", shp[t].what); fail = 1; goto next; }
        if (!gpu_q40_matmul(yg, W, x, O, I, 1)) {
            printf("  %-18s gpu_q40_matmul DECLINED\n", shp[t].what); fail = 1; goto next;
        }
        for (int o = 0; o < O; o++)
            yc[o] = q40_dot_f32(W + (size_t)o * q40_row_bytes(I), x, I);

        double worst = 0, mag = 0;
        for (int o = 0; o < O; o++) {
            double d = fabs(yg[o] - yc[o]);
            if (d > worst) worst = d;
            if (fabs(yc[o]) > mag) mag = fabs(yc[o]);
        }
        double rel = worst / (mag + 1e-9);
        printf("  %-18s [%5d x %5d]  max rel err %.3e  %s\n",
               shp[t].what, O, I, rel, rel < 1e-4 ? "ok" : "MISMATCH");
        if (rel >= 1e-4) fail = 1;
    next:
        free(W); free(x); free(yg); free(yc);
    }
    printf("\n%s\n", fail ? "GPU CHECK FAILED -- run with --no-metal" : "GPU CHECK PASSED");
    return fail;
}

/* lm_head applied to one already-post-normed hidden row. */
static void lm_head_row(M *m, const float *hn, float *logits, Buf *b) {
    Cfg *c = &m->c;
    W *embed = &m->L[MAXL - 1].q_proj;
    matvec(logits, embed, hn, b->xq, b->sx);
    if (c->final_logit_softcap > 0) {
        float cap = c->final_logit_softcap;
        for (int i = 0; i < c->vocab; i++) logits[i] = tanhf(logits[i] / cap) * cap;
    }
}

/* One speculation step with the MTP head.
 *
 * State carried between steps:
 *   P       -- index of the last token in `ids` (its KV is not yet in the target)
 *   hprev   -- the target's post-norm hidden at position P-1
 *
 * That pairing is the crux, and it is not the obvious one. HF does:
 *     last_hidden_state = hidden[n_last_matches]   -> position t
 *     last_token_id     = input_ids[-1]            -> position t+1
 * so the hidden comes from one position earlier than the token: the EAGLE
 * convention, concat(e(x_{t+1}), h_t). Pairing h_{t+1} with x_{t+1} instead collapses
 * acceptance to a few percent.
 *
 * A step costs one target forward rather than two: the batch is
 * [x_P, draft_0 .. draft_{d-1}] at positions P..P+d. Row 0 both puts x_P's KV into
 * the cache and yields the distribution that verifies draft_0; row i yields the one
 * that verifies draft_{i+1}. Hidden row `acc` is exactly h_{P+acc}, the hprev the
 * next step needs. An extra "resync" forward for the newly sampled token is waste,
 * and is also what forces the wrong pairing above.
 *
 * Batch-union makes the verify nearly free in I/O terms: the d+1 positions read each
 * distinct expert once. */
static int mtp_step(M *m, Buf *bt, MBuf *bd, int *ids, int P, float *hprev,
                    float *dlog, int d, float temp, float topp, int topk,
                    PI *pbuf, PI *dbuf, uint64_t *rng, int *out, int *accepted) {
    Cfg *c = &m->c;
    int V = c->vocab, BB = m->mtp_BB;
    int draft[MAXDRAFT], batch[MAXDRAFT + 1];
    float *dprob = xmalloc(sizeof(float) * (size_t)d * V);
    float *bh = xmalloc(sizeof(float) * BB);

    /* draft d tokens: (h_{P-1}, x_P), all at the fixed position P */
    memcpy(bh, hprev, sizeof(float) * BB);
    int tok = ids[P];
    for (int i = 0; i < d; i++) {
        mtp_forward(m, bh, tok, P, dlog, bd);      /* bh <- post_projection(h) */

        float mx = -1e30f;
        for (int j = 0; j < V; j++) if (dlog[j] > mx) mx = dlog[j];
        double sum = 0;
        float T = temp > 0 ? temp : 1.0f;
        float *dp = dprob + (size_t)i * V;
        for (int j = 0; j < V; j++) { dp[j] = expf((dlog[j] - mx) / T); sum += dp[j]; }
        for (int j = 0; j < V; j++) dp[j] /= (float)sum;

        int am = 0;                                /* the drafter is greedy */
        for (int j = 1; j < V; j++) if (dlog[j] > dlog[am]) am = j;
        tok = am;
        draft[i] = tok;
    }
    (void)dbuf;

    /* a single target forward over [x_P, draft...] at positions P..P+d */
    batch[0] = ids[P];
    for (int i = 0; i < d; i++) batch[i + 1] = draft[i];
    m->kv_conf = P;                 /* anything beyond P is speculative until verified */
    /* logits=NULL: the lm_head is 738 M params (1.5 GFLOP per row), and on a rejection
     * every row past the first is thrown away. Compute it lazily instead, row by row,
     * stopping as soon as a draft is rejected; at ~48% acceptance that skips roughly
     * two of five rows. hid_batch already carries the post-norm hidden for each row. */
    forward(m, batch, d + 1, P, NULL, 0, bt);

    float *q = xmalloc(sizeof(float) * V);
    int n;
    for (n = 0; n < d; n++) {
        lm_head_row(m, m->hid_batch + (size_t)n * c->hidden, q, bt);
        int t = draft[n], ok;
        if (temp <= 0) {
            int am = 0;
            for (int j = 1; j < V; j++) if (q[j] > q[am]) am = j;
            ok = (am == t);
        } else {
            float mx = -1e30f;
            for (int j = 0; j < V; j++) if (q[j] > mx) mx = q[j];
            double sum = 0;
            for (int j = 0; j < V; j++) { pbuf[j].p = expf((q[j] - mx) / temp); sum += pbuf[j].p; }
            float qt = (float)(pbuf[t].p / sum);
            float pt = dprob[(size_t)n * V + t];
            *rng = *rng * 6364136223846793005ULL + 1442695040888963407ULL;
            double r = (*rng >> 11) / (double)(1ULL << 53);
            ok = (pt <= 0) || (r < (double)qt / pt);
        }
        if (!ok) break;
        out[n] = t;
    }
    /* the bonus token, from the target's own distribution at the first rejected
     * (or the last accepted) position. `q` already holds that row iff we broke out of
     * the loop; if every draft was accepted we still need the final row. */
    if (n == d) lm_head_row(m, m->hid_batch + (size_t)n * c->hidden, q, bt);
    out[n] = sample(q, V, temp, topp, topk, pbuf, rng);

    /* rows 0..n of the batch are on the accepted path, so positions P..P+n are now
     * confirmed and may be compressed. Rows beyond that were rejected. */
    m->kv_conf = P + n;

    /* hprev for the next step = h_{P+acc} = hidden row `acc` of this forward. The new
     * last token is the bonus, at position P+acc+1: one position later. */
    memcpy(hprev, m->hid_batch + (size_t)n * c->hidden, sizeof(float) * BB);

    *accepted = n;
    free(dprob); free(bh); free(q);
    return n + 1;
}

/* One DFlash step, mirroring dflash_generate in the reference. `pos` is the absolute
 * position of first_tok, whose KV is written here too. Writes first_tok + the
 * accepted drafts to out[0..n], the bonus to *next_tok, and returns n+1.
 *
 * The bonus token needs no forward of its own: its KV and draft context fall out of
 * the next step's verify forward, which is why the block starts at position 0 with
 * the token the previous step sampled. */
static int dflash_step(M *m, Buf *bt, DBuf *bd, int first_tok, int pos,
                        int d, float temp, float topp, int topk,
                        PI *pbuf, uint64_t *rng, int *out, int *accepted,
                        int *next_tok) {
    Cfg *c = &m->c;
    int V = c->vocab, D = c->hidden;
    int BS = m->dflash_block_size;
    int mask_id = m->dflash_mask_token_id;

    if (d > BS) d = BS;
    if (d < 1) d = 1;

    float *H = bd->h;
    float *dlog = bd->dlog;
    float *dprob = bd->dprob;
    W *embed = &m->L[MAXL - 1].q_proj;
    int ndraft = d - 1;  /* number of tokens we actually draft */
    int draft[MAXDRAFT];
    /* frozen[i] (draft index) = position i+1 held a confident token last pass and is
     * fed as a real embedding this pass instead of the mask token. Once set it stays
     * set (monotone denoising). All zero when refinement is off, i.e. the reference
     * path. */
    int frozen[MAXDRAFT] = {0};
    float cap = c->final_logit_softcap;
    int npass = g_dflash_refine + 1;
    double t0 = now(), t1 = t0;

    /* 2/3. One or more denoising passes. Each pass: (re)embed the block with the
     * currently frozen tokens, run the DFlash forward, then take greedy drafts from
     * the logits. On non-final passes we also measure per-position confidence and
     * freeze the confident ones so the next pass conditions on them. */
    for (int pass = 0; pass < npass; pass++) {
        int last_pass = (pass == npass - 1);

        /* Embed the block using the target's embedding table: position 0 is the bonus
         * token, the rest are either a frozen draft or the mask token. */
        embed_row(m, first_tok, H);
        for (int i = 1; i < d; i++) {
            if (frozen[i - 1]) embed_row(m, draft[i - 1], H + (size_t)i * D);
            else               embed_row(m, mask_id,      H + (size_t)i * D);
        }

        /* Context K/V for positions [0, pos) come from the persistent draft cache,
         * filled by every target forward (prefill included). dflash_forward mutates
         * H in place but touches no persistent state, so re-running it is safe. */
        dflash_forward(m, H, d, pos, bd);
        if (pass == 0) t1 = now();

        /* Logits for positions 1..d-1 only (skip position 0). Python:
         * draft_logits = target.lm_head(model(...)[:, 1-B:, :]). */
        for (int i = 0; i < ndraft; i++) {
            float *dl = dlog + (size_t)i * V;
            matvec(dl, embed, H + (size_t)(i + 1) * D, bd->xq, bd->sx);
            int am = 0;
            for (int j = 1; j < V; j++) if (dl[j] > dl[am]) am = j;
            draft[i] = am;   /* the drafter is greedy, like the reference */

            if (!last_pass) {
                /* Confidence = max softmax prob (temperature-independent, softcapped).
                 * Freeze this position for the next pass if it clears the threshold.
                 * Positions already frozen have a fixed token, so skip them. */
                if (frozen[i]) continue;
                float mx = cap > 0 ? tanhf(dl[am] / cap) * cap : dl[am];
                double sum = 0;
                for (int j = 0; j < V; j++) {
                    float lg = cap > 0 ? tanhf(dl[j] / cap) * cap : dl[j];
                    sum += expf(lg - mx);
                }
                if (1.0f / (float)sum >= g_dflash_conf) frozen[i] = 1;
            } else if (temp > 0) {
                /* Final pass: the softcap + softmax over the 262k vocab are only needed
                 * for the temp > 0 rejection test (tanh is monotone, argmax unaffected). */
                float mx = cap > 0 ? tanhf(dl[am] / cap) * cap : dl[am];
                double sum = 0;
                float *dp = dprob + (size_t)i * V;
                for (int j = 0; j < V; j++) {
                    float lg = cap > 0 ? tanhf(dl[j] / cap) * cap : dl[j];
                    dp[j] = expf((lg - mx) / temp);
                    sum += dp[j];
                }
                for (int j = 0; j < V; j++) dp[j] /= (float)sum;
            }
        }
    }

    /* 4. Verify with a single batched target forward over exactly the reference's
     * block: [first_tok, draft[0], ..., draft[ndraft-1]] at positions pos..pos+ndraft.
     * Row i's lm_head predicts position pos+i+1, so it verifies draft[i]. (Position
     * pos-1's KV is already written by the previous forward, and first_tok is the
     * target's own sample, so it needs no verification.) */
    int batch[MAXDRAFT + 1];
    batch[0] = first_tok;
    for (int i = 0; i < ndraft; i++) batch[i + 1] = draft[i];
    m->kv_conf = pos - 1;
    double t2 = now();
    forward(m, batch, d, pos, NULL, 0, bt);
    double t3 = now();
    if (getenv("G4DBG"))
        fprintf(stderr, "[dflash] pos %d: draft-fwd %.0fms lm_head %.0fms verify-fwd %.0fms\n",
                pos, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3);

    float *q = xmalloc(sizeof(float) * V);
    int n;  /* draft tokens accepted, not counting first_tok */
    out[0] = first_tok;

    for (n = 0; n < ndraft; n++) {
        lm_head_row(m, m->hid_batch + (size_t)n * c->hidden, q, bt);
        int t = draft[n], ok;
        if (temp <= 0) {
            int am = 0;
            for (int j = 1; j < V; j++) if (q[j] > q[am]) am = j;
            ok = (am == t);
        } else {
            float mx = -1e30f;
            for (int j = 0; j < V; j++) if (q[j] > mx) mx = q[j];
            double sum = 0;
            for (int j = 0; j < V; j++) {
                pbuf[j].p = expf((q[j] - mx) / temp);
                sum += pbuf[j].p;
            }
            float qt = (float)(pbuf[t].p / sum);
            float pt = dprob[(size_t)n * V + t];
            *rng = *rng * 6364136223846793005ULL + 1442695040888963407ULL;
            double r = (*rng >> 11) / (double)(1ULL << 53);
            ok = (pt <= 0) || (r < (double)qt / pt);
        }
        if (!ok) break;
        out[n + 1] = t;
    }

    /* Next step's first token ("bonus") from the target's distribution at the
     * first unverified position: row n's lm_head predicts position pos+n+1. */
    lm_head_row(m, m->hid_batch + (size_t)n * c->hidden, q, bt);
    *next_tok = sample(q, V, temp, topp, topk, pbuf, rng);

    /* Positions pos..pos+n are confirmed (first_tok + n drafts). The bonus token's
     * KV and draft context are computed by the next step's verify forward. */
    m->kv_conf = pos + n;
    *accepted = n;

    free(q);
    return n + 1;  /* first_tok + n drafts */
}

/* OpenAI-compatible local server */
typedef struct {
    M *model;
    Buf *buffers;
    G4Tok *tokenizer;
    pthread_mutex_t generation_mu;
    atomic_int cancel;
    int *cached_ids;
    int cached_len;
    int cached_cap;
    const char *model_id;
} G4ServerContext;

typedef struct { char *data; size_t len, cap; } G4String;

static int g4_string_append(G4String *s, const char *data, size_t len) {
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

static int g4_json_escape(G4String *s, const char *text, size_t len) {
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == '"' || c == '\\') { char x[2] = {'\\', (char)c}; if (!g4_string_append(s,x,2)) return 0; }
        else if (c == '\n') { if (!g4_string_append(s,"\\n",2)) return 0; }
        else if (c == '\r') { if (!g4_string_append(s,"\\r",2)) return 0; }
        else if (c == '\t') { if (!g4_string_append(s,"\\t",2)) return 0; }
        else if (c < 0x20) { char x[6] = {'\\','u','0','0',hex[c>>4],hex[c&15]}; if (!g4_string_append(s,x,6)) return 0; }
        else if (!g4_string_append(s,(const char *)&text[i],1)) return 0;
    }
    return 1;
}

static jval *g4_json_field(jval *object, const char *key, jtype type) {
    jval *v = json_get(object, key); return v && v->t == type ? v : NULL;
}

static int g4_build_chat_prompt(jval *messages, G4String *prompt) {
    if (!messages || messages->t != J_ARR) return 0;
    if (!g4_string_append(prompt, "<bos>", 5)) return 0;
    for (int i = 0; i < messages->len; i++) {
        jval *message = messages->kids[i];
        jval *role = g4_json_field(message, "role", J_STR);
        jval *content = g4_json_field(message, "content", J_STR);
        if (!role || !content) continue;
        if (!strcmp(role->str, "system")) {
            if (!g4_string_append(prompt, "<|turn>system\\n", strlen("<|turn>system\\n")) ||
                            !g4_string_append(prompt, content->str, strlen(content->str)) ||
                            !g4_string_append(prompt, "<turn|>\\n", strlen("<turn|>\\n"))) return 0;
        } else if (!strcmp(role->str, "user")) {
            if (!g4_string_append(prompt, "<|turn>user\\n", strlen("<|turn>user\\n")) ||
                            !g4_string_append(prompt, content->str, strlen(content->str)) ||
                            !g4_string_append(prompt, "<turn|>\\n", strlen("<turn|>\\n"))) return 0;
        } else if (!strcmp(role->str, "assistant")) {
            if (!g4_string_append(prompt, "<|turn>model\\n", strlen("<|turn>model\\n")) ||
                            !g4_string_append(prompt, content->str, strlen(content->str)) ||
                            !g4_string_append(prompt, "<turn|>\\n", strlen("<turn|>\\n"))) return 0;
        }
    }
    return g4_string_append(prompt, "<|turn>model\\n<|channel>thought\\n<channel|>", strlen("<|turn>model\\n<|channel>thought\\n<channel|>"));
}

static int g4_send_chunk(int fd, const char *id, const char *field, const char *text, size_t len) {
    G4String out = {0};
    const char *prefix = "data: {\"id\":\"";
    const char *middle = "\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"";
    const char *suffix = "\":\"";
    const char *end = "\"},\"finish_reason\":null}]}\n\n";
    int ok = g4_string_append(&out,prefix,strlen(prefix)) && g4_json_escape(&out,id,strlen(id)) &&
        g4_string_append(&out,middle,strlen(middle)) && g4_string_append(&out,field,strlen(field)) &&
        g4_string_append(&out,suffix,strlen(suffix)) && g4_json_escape(&out,text,len) &&
        g4_string_append(&out,end,strlen(end));
    if (ok) ok = samosa_send_all(fd, out.data, out.len);
    free(out.data); return ok;
}

static int g4_send_done(int fd, const char *id, int prompt_tokens, int completion_tokens,
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

static int g4_serve_chat(G4ServerContext *ctx, int fd, jval *root) {
    jval *messages = g4_json_field(root, "messages", J_ARR);
    int has_user = 0;
    if (!messages) return samosa_http_json_error(fd,400,"invalid_messages","messages must be an array.");
    for (int i = 0; i < messages->len; i++) {
        jval *msg = messages->kids[i];
        jval *role = g4_json_field(msg,"role",J_STR);
        jval *content = g4_json_field(msg,"content",J_STR);
        if (role && content && !strcmp(role->str,"user")) has_user = 1;
    }
    if (!has_user) return samosa_http_json_error(fd,400,"invalid_messages","A text user message is required.");

    int stream = 0, max_tokens = 2048, topk = 64, seed = 0;
    float temperature = 1.0f, topp = 0.95f;
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
    v = json_get(root,"seed");
    if (v) {
        if (v->t != J_NUM || v->num < 0 || floor(v->num) != v->num || v->num > UINT32_MAX)
            return samosa_http_json_error(fd,400,"invalid_seed","seed must be a non-negative integer.");
        seed = (int)v->num;
    }

    M *m = ctx->model; Cfg *c = &m->c;
    G4String prompt = {0};
    if (!g4_build_chat_prompt(messages, &prompt)) {
        free(prompt.data);
        return samosa_http_json_error(fd,400,"invalid_prompt","Unable to construct the chat prompt.");
    }
    int *ids = malloc((size_t)c->ctx * sizeof *ids);
    float *logits = malloc((size_t)c->vocab * sizeof *logits);
    PI *pbuf = malloc((size_t)c->vocab * sizeof *pbuf);
    if (!ids || !logits || !pbuf) { free(prompt.data); free(ids); free(logits); free(pbuf); return samosa_http_json_error(fd,500,"out_of_memory","Unable to allocate generation buffers."); }
    G4Tok *tok = ctx->tokenizer;
    int np = g4tok_encode(tok, prompt.data, ids, c->ctx);
    free(prompt.data);
    if (np <= 0) { free(ids); free(logits); free(pbuf); return samosa_http_json_error(fd,400,"invalid_prompt","The prompt produced no tokens."); }
    if (np >= c->ctx) {
        char msg[256];
        snprintf(msg, sizeof msg, "The prompt is %d tokens and the context window is "
                 "%d; it leaves no room for a completion.", np, c->ctx);
        free(ids); free(logits); free(pbuf);
        return samosa_http_json_error(fd, 400, "context_limit", msg);
    }
    if (max_tokens > c->ctx - np) max_tokens = c->ctx - np;

    pthread_mutex_lock(&ctx->generation_mu);
    atomic_store(&ctx->cancel, 0);
    int common = 0;
    while (common < ctx->cached_len && common < np && ctx->cached_ids[common] == ids[common]) common++;
    if (common == np && np > 0) {
        /* The cached logits are not retained; replay the final token only. */
        forward(m, &ids[np - 1], 1, np - 1, logits, 1, ctx->buffers);
    } else {
        forward(m, ids + common, np - common, common, logits, 1, ctx->buffers);
    }
    char id[64]; snprintf(id,sizeof id,"gemma4-%llu",(unsigned long long)time(NULL));
    if (stream && !samosa_http_stream_headers(fd)) { pthread_mutex_unlock(&ctx->generation_mu); free(ids); free(logits); free(pbuf); return 1; }
    G4String answer = {0}; uint64_t rng = seed ? (uint64_t)seed : 0x853c49e6748fea9bULL;
    int generated = 0; const char *reason = "length";
    while (generated < max_tokens && !atomic_load(&ctx->cancel)) {
        int token = sample(logits, c->vocab, temperature, topp, topk, pbuf, &rng);
        if (token == 1 || token == 106) { reason = "stop"; break; }
        char piece[4096]; int n = g4tok_decode(tok, &token, 1, piece, sizeof piece - 1);
        if (n <= 0) { reason = "stop"; break; }
        if (!g4_string_append(&answer, piece, (size_t)n)) { atomic_store(&ctx->cancel,1); break; }
        if (stream && !g4_send_chunk(fd,id,"content",piece,(size_t)n)) { atomic_store(&ctx->cancel,1); break; }
        ids[np + generated++] = token;
        if (generated < max_tokens) forward(m, &token, 1, np + generated - 1, logits, 1, ctx->buffers);
    }
    if (atomic_load(&ctx->cancel)) reason = "cancelled";
    if (stream) g4_send_done(fd,id,np,generated,reason);
    else {
        G4String body={0}; char prefix[512], suffix[512];
        int n=snprintf(prefix,sizeof prefix,"{\"id\":\"%s\",\"object\":\"chat.completion\",\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"",id,ctx->model_id);
        int ok=n>0&&g4_string_append(&body,prefix,(size_t)n)&&g4_json_escape(&body,answer.data?answer.data:"",answer.len);
        n=snprintf(suffix,sizeof suffix,"\"},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",reason,np,generated,np+generated);
        ok=ok&&n>0&&g4_string_append(&body,suffix,(size_t)n)&&samosa_http_headers(fd,200,"application/json",body.len,NULL)&&samosa_send_all(fd,body.data,body.len);
        free(body.data); (void)ok;
    }
    int final_len = np + generated;
    if (generated > 0)
        forward(m, &ids[final_len - 1], 1, final_len - 1, logits, 1, ctx->buffers);
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
    free(answer.data); pthread_mutex_unlock(&ctx->generation_mu); free(ids); free(logits); free(pbuf); return 0;
}

static int g4_serve_handler(SamosaHttpServer *server, int fd, const SamosaHttpRequest *request, void *opaque) {
    G4ServerContext *ctx = opaque;
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
        atomic_store(&ctx->cancel,1); samosa_http_response(fd,200,"application/json","{\"shutting_down\":true}",NULL);
    }
    if (!strcmp(request->method,"POST") && !strcmp(request->path,"/v1/chat/completions")) {
        char *arena=NULL; jval *root=json_parse(request->body,&arena);
        if (!root || root->t != J_OBJ) { json_free(root); free(arena); return samosa_http_json_error(fd,400,"invalid_json","A JSON object is required."); }
        int result=g4_serve_chat(ctx,fd,root); json_free(root); free(arena); return result;
    }
    if (!strcmp(request->method,"POST") && !strcmp(request->path,"/v1/shutdown")) {
        atomic_store(&ctx->cancel,1); samosa_http_response(fd,200,"application/json","{\"shutting_down\":true}",NULL); samosa_http_server_stop(server); return 1;
    }
    return samosa_http_json_error(fd,404,"not_found","Endpoint not found.");
}

static int run_g4_server(M *m, Buf *buffers, G4Tok *tokenizer, const char *model_id, int port) {
    G4ServerContext ctx={.model=m,.buffers=buffers,.tokenizer=tokenizer,.model_id=model_id};
    pthread_mutex_init(&ctx.generation_mu,NULL); atomic_init(&ctx.cancel,0);
    SamosaHttpServer server;
    if (!samosa_http_server_init(&server,port,g4_serve_handler,&ctx)) { fprintf(stderr,"server: cannot bind port %d: %s\n",port,strerror(errno)); pthread_mutex_destroy(&ctx.generation_mu); return 1; }
    fprintf(stderr,"[server] OpenAI endpoint ready at http://127.0.0.1:%d\n",server.port); fflush(stderr);
    int ok=samosa_http_server_run(&server); samosa_http_server_destroy(&server);
    free(ctx.cached_ids); pthread_mutex_destroy(&ctx.generation_mu); return ok?0:1;
}

/* main */
static void usage(const char *prog, FILE *out) {
    fprintf(out,
        "usage: %s <dir> [flags...] [prompt]\n"
        "         [--chat] [--system S] [--think] [--raw] [--max_tokens N]\n"
        "         [--temp F] [--topp F] [--topk N]   (default 1.0 / 0.95 / 64)\n"
        "         [--pin N] [--draft DIR] [--ndraft N]\n"
        "         [--mtp] [--dflash] [--drefine N] [--dconf F]\n"
        "                                DFlash extra denoising passes (default 0) and\n"
        "                                per-token freeze confidence (default 0.9)\n"
        "         [--ctx N]              override the container's context length\n"
        "         [--ram F]              re-plan the expert cache for an F GB budget\n"
        "         [--io N] [--nobatch] [--threads N]\n"
        "         [--metal] Metal is OFF by default (it is slower if gemma is not fully in RAM)\n"
        "         [--serve] [--port N]    OpenAI-compatible local server (default 8484)\n"
        "         [--kv PRESET]          KVarN KV-cache compression; PRESET is one of\n"
        "                                off | kvarn_k4v2_g128 | kvarn_k4v4_g128 |\n"
        "                                kvarn_k4v2_g64 | kvarn_k4v4_g64\n"
        "                                (default kvarn_k4v2_g128)\n"
        "         [--help]\n",
        prog);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0], stderr);
        return 1;
    }
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0], stdout);
            return 0;
        }
    }
    const char *dir = argv[1];
    const char *prompt = NULL, *sys = NULL;
    int think = 0, raw = 0, chat_mode = 0;
    /* KVarN is ON by default, at upstream's shipped preset, and a preset is all
     * there is: no per-parameter overrides, because the bit widths and the tile are
     * one calibrated recipe upstream measured together. --kv off gives f32 KV. */
    const KvarnPreset *kvp = kvarn_preset(KVARN_DEFAULT);
    int kv_set = 0;
    int rwin = KVARN_RWIN;
    int no_metal = 0, chk_gpu = 0, use_mtp = 0, use_dflash = 0, use_metal = 0;
    int check = 0, n_io = 8, max_tokens = 0, nobatch = 0, npin = 0, draft = 0, nthreads = 2;
    int serve_mode = 0, serve_port = 8484, ctx_override = 0;
    double ram_gb = 0;                   /* 0 = keep the container's own plan */
    const char *dpath = NULL;
    float temp = 1.0f, topp = 0.95f;   /* Gemma-4 generation defaults */
    int topk = 64;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--serve")) serve_mode = 1;
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) serve_port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--check")) check = 1;
        else if (!strcmp(argv[i], "--nobatch")) nobatch = 1;
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) nthreads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--io") && i + 1 < argc) n_io = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx_override = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ram") && i + 1 < argc) ram_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--max_tokens") && i + 1 < argc) max_tokens = atoi(argv[++i]);

        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--topp") && i + 1 < argc) topp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--topk") && i + 1 < argc) topk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pin") && i + 1 < argc) npin = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--draft") && i + 1 < argc) dpath = argv[++i];
        else if (!strcmp(argv[i], "--ndraft") && i + 1 < argc) draft = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--drefine") && i + 1 < argc) g_dflash_refine = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dconf") && i + 1 < argc) g_dflash_conf = atof(argv[++i]);
        else if (!strcmp(argv[i], "--system") && i + 1 < argc) sys = argv[++i];
        else if (!strcmp(argv[i], "--chat")) chat_mode = 1;
        else if (!strcmp(argv[i], "--think")) think = 1;
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
        else if (!strcmp(argv[i], "--no-metal")) no_metal = 1;
        else if (!strcmp(argv[i], "--metal")) use_metal = 1;
        else if (!strcmp(argv[i], "--check-gpu")) chk_gpu = 1;
        else if (!strcmp(argv[i], "--mtp")) use_mtp = 1;
        else if (!strcmp(argv[i], "--dflash")) use_dflash = 1;
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
        else if (!prompt) prompt = argv[i];  /* first non-flag positional arg is the prompt */
    }
    /* --check diffs the forward pass against a stored oracle to ~1e-4, which is
     * tighter than any KV quantiser reproduces, so it defaults to f32 KV whatever
     * the engine default is. Pass --kv explicitly to measure the codec instead. */
    if (check && !kv_set) kvp = NULL;
    /* kb == 0 is how the rest of the engine spells "f32 KV". */
    int kb = kvp ? kvp->kbits : 0, vb = kvp ? kvp->vbits : 0;
    int kv_tile = kvp ? kvp->group : 128;
    if (chat_mode && check) {
        fprintf(stderr, "--chat cannot be used with --check\n\n");
        usage(argv[0], stderr);
        return 1;
    }
    if (chat_mode && serve_mode) {
        fprintf(stderr, "--chat cannot be used with --serve\n\n");
        usage(argv[0], stderr);
        return 1;
    }
    /* Default max_tokens when a prompt is given (or interactive mode) but --max_tokens was not set. */
    if (!serve_mode && !check && max_tokens == 0) max_tokens = 2048;
    if (draft > MAXDRAFT) draft = MAXDRAFT;
    if (use_dflash && draft <= 0) draft = 16;  /* DFlash default block size */
    if ((dpath || use_mtp) && draft <= 0) draft = 4;
    /* Speculation is the one thing tiling charges for. A tile is sealed once its
     * youngest position is ring - tile + 1 behind the write head, and a sealed tile
     * is final, so a draft that is still unconfirmed by then would either be baked in
     * or drop the whole tile. Widen the ring by a tile rather than making the user
     * discover this: without --mtp/--draft/--dflash it stays at KVARN_RWIN. */
    if (kvp && draft > 0) {
        int need = kv_tile + draft + 2;
        if (rwin < need) rwin = need;
    }
#ifdef _OPENMP
    if (nthreads > 0) omp_set_num_threads(nthreads);
#else
    (void)nthreads;
#endif

    /* Metal is opt-in, via --metal.
     *
     * Measured on real hardware it is ~5x slower than the CPU for decode, and the
     * reason is structural rather than a tuning miss: this kernel issues one dispatch
     * and one waitUntilCompleted per matvec, and a token needs thousands of them. At
     * batch size 1 that is pure dispatch latency (~0.5 ms each) against ~40 ms of
     * actual arithmetic. A GPU only wins here with far more work per dispatch --
     * whole layers fused into one command buffer, or large prefill batches -- and
     * even then the engine is disk-bound at a 4-8 GB budget. On by default would
     * make everyone slower, so it stays off unless asked for. */
    if (use_metal && !no_metal) g_use_gpu = gpu_init();
    if (chk_gpu) { if (!g_use_gpu) g_use_gpu = gpu_init(); return check_gpu(); }

    M m; memset(&m, 0, sizeof m);
    double t0 = now();
    init(&m, dir, n_io, ctx_override, ram_gb, kb, vb, rwin, kv_tile);
    if (g_use_gpu)
        fprintf(stderr, "metal: %s (q4_0 matmul offloaded; --no-metal to disable)\n",
                gpu_name());
    Cfg *c = &m.c;
    pin_load(&m, npin);
    Buf *b = bufs(&m, c->ctx);

    if (serve_mode) {
        fprintf(stderr, "gemma4: %d layers, %d experts, top-%d, %d slots/layer, "
                "expert %.2f MiB, dense %.1f MiB, ready in %.2fs\n",
                c->n_layers, c->n_experts, c->topk, c->slots_per_layer,
                m.esz / 1048576.0, m.dense_len / 1048576.0, now() - t0);
        char tp[4096]; snprintf(tp, sizeof tp, "%s/tok.bin", dir);
        G4Tok *server_tokenizer = g4tok_load(tp);
        if (!server_tokenizer) { fprintf(stderr, "--serve needs %s\n", tp); return 1; }
        int result = run_g4_server(&m, b, server_tokenizer, "gemma-4-26b-a4b", serve_port);
        return result;
    }

    MBuf *mb = NULL;
    /* Backstop for the widening above: the margin that has to outlast an
     * unconfirmed draft is the ring minus a tile, plus one. */
    if (use_mtp && kvp && m.rwin - kv_tile + 1 < draft + 2) {
        fprintf(stderr, "f32 ring %d (tile %d) is too small for --ndraft %d: a tile "
                "must stay unsealed until every position in it is confirmed\n",
                m.rwin, kv_tile, draft);
        return 1;
    }
    if (use_mtp) {
        if (!mtp_load(&m, dir)) {
            fprintf(stderr, "--mtp: no usable mtp.* in %s "
                    "(run tools/convert_gemma4_mtp.py)\n", dir);
            return 1;
        }
        mb = mtp_bufs(&m);
    }

    DBuf *dfb = NULL;
    if (use_dflash) {
        if (!dflash_load(&m, dir)) {
            fprintf(stderr, "--dflash: no usable dflash.* in %s "
                    "(run tools/convert_gemma4_dflash.py)\n", dir);
            return 1;
        }
        dfb = dflash_bufs(&m);
        /* DFlash needs the hid_batch for verification (same as MTP) */
        if (!m.hid_batch)
            m.hid_batch = xmalloc(sizeof(float) * (size_t)(MAXDRAFT + 2) * c->hidden);
    }

    M dm; memset(&dm, 0, sizeof dm);
    Buf *db = NULL;
    if (dpath) {
        /* Same ctx as the target: the drafter has to span the same positions. Its KV
         * stays f32, being tiny. ram_gb 0 because --ram budgets the target's expert
         * cache, and the drafter is a dense model whose whole footprint is fixed by
         * its own container. */
        init(&dm, dpath, n_io, c->ctx, 0, 0, 0, rwin, kv_tile);
        db = bufs(&dm, dm.c.ctx);
        if (dm.c.vocab != c->vocab) {
            fprintf(stderr, "drafter vocab %d != target %d\n", dm.c.vocab, c->vocab);
            return 1;
        }
        fprintf(stderr, "drafter: %d layers, speculating %d tokens/step\n",
                dm.c.n_layers, draft);
    }
    fprintf(stderr, "gemma4: %d layers, %d experts, top-%d, %d slots/layer, "
            "expert %.2f MiB, dense %.1f MiB, ready in %.2fs\n",
            c->n_layers, c->n_experts, c->topk, c->slots_per_layer,
            m.esz / 1048576.0, m.dense_len / 1048576.0, now() - t0);

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
        int ids[512], np = 0;
        for (char *s = strchr(qs, '['); s && *s && *s != ']'; s++)
            if (*s >= '0' && *s <= '9') {
                if (np == (int)(sizeof ids / sizeof *ids)) {
                    fprintf(stderr, "--check: fixture prompt longer than %zu tokens\n",
                            sizeof ids / sizeof *ids);
                    return 1;
                }
                ids[np++] = strtol(s, &s, 10); s--;
            }

        /* Prefer deq_logits.f32: the numpy oracle run on the dequantised container
         * weights. Comparing against the fp32 HF logits instead conflates engine
         * bugs with q4_0's own error, which on a random fixture is ~10% and would
         * drown any bug worth finding. */
        snprintf(p, sizeof p, "%s/deq_logits.f32", dir);
        f = fopen(p, "rb");
        if (!f) {
            snprintf(p, sizeof p, "%s/ref_logits.f32", dir);
            f = fopen(p, "rb");
            fprintf(stderr, "note: no deq_logits.f32, falling back to the raw fp32 "
                            "reference (includes quantisation error)\n");
        }
        if (!f) { perror(p); return 1; }
        float *ref = xmalloc(sizeof(float) * (size_t)np * c->vocab);
        if (fread(ref, sizeof(float), (size_t)np * c->vocab, f) != (size_t)np * c->vocab) {
            fprintf(stderr, "short reference logits\n");
            return 1;
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
        /* Tolerance depends on the build: with exact activations the engine must
         * reproduce the oracle to float precision, and any gap is a bug. The default
         * build quantises activations to int8 (Q8_0), which costs ~1e-2 on logits.
         * Argmax must agree either way. */
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
        uint64_t rng = 0x853c49e6748fea9bULL;

        char tp[4096];
        snprintf(tp, sizeof tp, "%s/tok.bin", dir);
        G4Tok *T = g4tok_load(tp);

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
            np = g4tok_encode(T, text, ids, c->ctx - max_tokens);
            if (getenv("G4DBG")) fprintf(stderr, "prompt: %s\n[%d tokens]\n", text, np);
            free(chat);
            if (np <= 0) { fprintf(stderr, "empty prompt\n"); return 1; }
        }

        /* interactive multi-turn chat (--chat, or no prompt for compatibility) */
        int interactive = (chat_mode || (!prompt && !check));
        int first_prompt = (chat_mode && prompt != NULL);
        int *cached_ids = NULL;
        int cached_len = 0, cached_cap = 0;
        char *history = NULL;   /* accumulated chat template text */
        size_t hist_len = 0, hist_cap = 0;

        if (interactive) {
            if (!T) { fprintf(stderr, "interactive mode needs %s\n", tp); return 1; }
            fprintf(stderr, "[chat] interactive mode (Ctrl-D to exit)\n");
            fflush(stderr);
        }

        double tpre = 0;
        long long pre_reads = 0;

        for (;;) {
            if (interactive) {
                /* Use a positional prompt as the first turn when --chat is explicit;
                 * subsequent turns are read from stdin. */
                char line[4096];
                const char *user_text;
                size_t len;
                if (first_prompt) {
                    user_text = prompt;
                    len = strlen(user_text);
                    first_prompt = 0;
                } else {
                    fprintf(stdout, "> ");
                    fflush(stdout);
                    if (!fgets(line, sizeof line, stdin)) break;  /* Ctrl-D / EOF */
                    len = strlen(line);
                    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
                        line[--len] = 0;
                    if (!len) continue;  /* empty line */
                    user_text = line;
                }

                /* Build the full chat template text from history + new turn. */
                size_t need = hist_len + len + 256;
                char *chat = xmalloc(need);
                size_t pos = 0;
                if (hist_len) {
                    memcpy(chat, history, hist_len);
                    pos = hist_len;
                }
                if (!pos) {
                    /* First turn: include system message if any. */
                    pos += snprintf(chat + pos, need - pos, "<bos>");
                    if (think || (sys && *sys)) {
                        pos += snprintf(chat + pos, need - pos, "<|turn>system\n");
                        if (think) pos += snprintf(chat + pos, need - pos, "<|think|>\n");
                        if (sys && *sys) pos += snprintf(chat + pos, need - pos, "%s", sys);
                        pos += snprintf(chat + pos, need - pos, "<turn|>\n");
                    }
                }
                pos += snprintf(chat + pos, need - pos, "<|turn>user\n%s<turn|>\n", user_text);
                pos += snprintf(chat + pos, need - pos, "<|turn>model\n");
                if (!think) pos += snprintf(chat + pos, need - pos, "<|channel>thought\n<channel|>");

                /* Encode the full prompt. */
                np = g4tok_encode(T, chat, ids, c->ctx - max_tokens);
                if (getenv("G4DBG")) fprintf(stderr, "prompt: %s\n[%d tokens]\n", chat, np);
                if (np <= 0) { free(chat); continue; }

                /* Update history: append the user turn + model prefix. */
                if (!hist_cap) { hist_cap = 4096; history = xmalloc(hist_cap); }
                while (hist_cap < pos + 256) { hist_cap *= 2; history = realloc(history, hist_cap); }
                memcpy(history, chat, pos);
                hist_len = pos;
                free(chat);

                /* Context caching: find common prefix with previous turn. */
                int common = 0;
                while (common < cached_len && common < np && cached_ids[common] == ids[common])
                    common++;
                if (common == np && np > 0) {
                    forward(&m, &ids[np - 1], 1, np - 1, logits, 1, b);
                } else {
                    forward(&m, ids + common, np - common, common, logits, 1, b);
                }
            } else {
                /* one-shot prefill (prompt given on command line) */
                double t0p = now();
                forward(&m, ids, np, 0, logits, 1, b);
                tpre = now() - t0p;
                pre_reads = m.hit + m.miss;
            }

            char piece[512];
            int n = 0, steps = 0, acc_tot = 0;
            float *dlog = dpath ? xmalloc(sizeof(float) * c->vocab) : NULL;
            PI *dbuf = dpath ? xmalloc(sizeof(PI) * c->vocab) : NULL;
            float *mlog = use_mtp ? xmalloc(sizeof(float) * c->vocab) : NULL;
            PI *mbuf = use_mtp ? xmalloc(sizeof(PI) * c->vocab) : NULL;
            int out[MAXDRAFT + 1];

            int dpos = 0;
            if (dpath) { forward(&dm, ids, np, 0, dlog, 1, db); dpos = np; }
            double t = now();

            if (use_mtp) {
                float *hprev = xmalloc(sizeof(float) * c->hidden);
                memcpy(hprev, m.hid_batch + (size_t)(m.hid_rows - 1) * c->hidden,
                       sizeof(float) * c->hidden);

                int tok0 = sample(logits, c->vocab, temp, topp, topk, pbuf, &rng);
                if (!(tok0 == 1 || tok0 == 106)) {
                    if (T) { g4tok_decode(T, &tok0, 1, piece, sizeof piece); fputs(piece, stdout); }
                    else printf("%d ", tok0);
                    fflush(stdout);
                    ids[np + n] = tok0;
                    n++;
                }

                while (n < max_tokens) {
                    int d = draft;
                    if (n + d > max_tokens) d = max_tokens - n;
                    if (d < 1) d = 1;
                    int acc = 0;
                    int P = np + n - 1;
                    int got = mtp_step(&m, b, mb, ids, P, hprev, mlog,
                                       d, temp, topp, topk, pbuf, mbuf, &rng, out, &acc);
                    steps++;
                    acc_tot += acc;

                    int stop = 0;
                    for (int i = 0; i < got && n < max_tokens; i++) {
                        int tk = out[i];
                        if (tk == 1 || tk == 106) { stop = 1; break; }
                        if (T) { g4tok_decode(T, &tk, 1, piece, sizeof piece); fputs(piece, stdout); }
                        else printf("%d ", tk);
                        ids[np + n] = tk;
                        n++;
                    }
                    fflush(stdout);
                    if (stop) break;
                }
                free(hprev);
                goto done_decode;
            }

            if (use_dflash) {
                /* DFlash block-parallel speculative decode. One target forward per
                 * step: it verifies the drafts and writes the KV / draft context of
                 * the previous step's bonus token (block position 0), exactly like
                 * the reference's rolling block. */
                int cur = sample(logits, c->vocab, temp, topp, topk, pbuf, &rng);
                while (n < max_tokens) {
                    int d = draft;
                    if (d > m.dflash_block_size) d = m.dflash_block_size;
                    if (n + d > max_tokens) d = max_tokens - n;
                    int pos = np + n;      /* absolute position of `cur` */
                    if (pos + d > c->ctx) d = c->ctx - pos;   /* KV ring bound */
                    if (d < 1) d = 1;
                    int acc = 0, nxt = -1;
                    int got = dflash_step(&m, b, dfb, cur, pos,
                                          d, temp, topp, topk,
                                          pbuf, &rng, out, &acc, &nxt);
                    steps++;
                    acc_tot += acc;

                    int stop = 0;
                    for (int i = 0; i < got && n < max_tokens; i++) {
                        int tk = out[i];
                        if (tk == 1 || tk == 106) { stop = 1; break; }
                        if (T) { g4tok_decode(T, &tk, 1, piece, sizeof piece); fputs(piece, stdout); }
                        else printf("%d ", tk);
                        ids[np + n] = tk;
                        n++;
                    }
                    fflush(stdout);
                    if (stop) break;
                    cur = nxt;
                }
                goto done_decode;
            }

            while (n < max_tokens) {
                int got, acc = 0;
                if (dpath) {
                    int d = draft;
                    if (n + d > max_tokens) d = max_tokens - n;
                    if (d < 1) d = 1;
                    got = spec_step(&m, &dm, b, db, ids, np + n, &dpos, logits, dlog,
                                    d, temp, topp, topk, pbuf, dbuf, &rng, out, &acc);
                    steps++;
                    acc_tot += acc;

                } else {
                    out[0] = sample(logits, c->vocab, temp, topp, topk, pbuf, &rng);
                    got = 1;
                }

                int stop = 0;
                for (int i = 0; i < got && n < max_tokens; i++) {
                    int tok = out[i];
                    if (tok == 1 || tok == 106) { stop = 1; break; }
                    if (T) {
                        g4tok_decode(T, &tok, 1, piece, sizeof piece);
                        fputs(piece, stdout);
                    } else printf("%d ", tok);
                    ids[np + n] = tok;
                    n++;
                }
                fflush(stdout);
                if (stop) break;

                if (n < max_tokens) {
                    int last = ids[np + n - 1];
                    forward(&m, &last, 1, np + n - 1, logits, 1, b);
                }
            }
        done_decode:;
            double el = now() - t;

            if (interactive) {
                /* Accumulate the assistant response into history. */
                printf("\n");
                fflush(stdout);
                /* Append the generated tokens (as text) to history. */
                char *resp = xmalloc((size_t)n * 16 + 256);
                size_t rlen = 0;
                for (int i = 0; i < n; i++) {
                    char piece2[64];
                    int pn = g4tok_decode(T, &ids[np + i], 1, piece2, sizeof piece2 - 1);
                    if (pn > 0 && rlen + (size_t)pn < (size_t)n * 16 + 256 - 16) {
                        memcpy(resp + rlen, piece2, (size_t)pn);
                        rlen += (size_t)pn;
                    }
                }
                resp[rlen] = 0;

                /* Extend history: append the assistant response + turn end. */
                size_t add = rlen + strlen("<turn|>\n");
                while (hist_cap < hist_len + add + 1) {
                    hist_cap *= 2;
                    history = realloc(history, hist_cap);
                }
                memcpy(history + hist_len, resp, rlen);
                hist_len += rlen;
                memcpy(history + hist_len, "<turn|>\n", strlen("<turn|>\n"));
                hist_len += strlen("<turn|>\n");
                history[hist_len] = 0;
                free(resp);

                /* Update the token cache for next turn. */
                int final_len = np + n;
                if (n > 0)
                    forward(&m, &ids[final_len - 1], 1, final_len - 1, logits, 1, b);
                if (final_len > cached_cap) {
                    int cap = cached_cap ? cached_cap : 256;
                    while (cap < final_len) cap *= 2;
                    int *c = realloc(cached_ids, (size_t)cap * sizeof *c);
                    if (c) { cached_ids = c; cached_cap = cap; }
                }
                if (cached_ids && final_len <= cached_cap) {
                    memcpy(cached_ids, ids, (size_t)final_len * sizeof *ids);
                    cached_len = final_len;
                }
            } else {
                /* One-shot: print stats. */
                long long tot = m.hit + m.miss;
                printf("\n\nprefill %d tok in %.2fs (%.1f tok/s, %lld expert reads)\n",
                       np, tpre, np / tpre, pre_reads);
                printf("decode  %d tok in %.2fs (%.2f tok/s)\n", n, el, n / el);
                if ((dpath || use_mtp || use_dflash) && steps)
                    printf("speculation: %.1f%% acceptance, %.2f tok/target-forward\n",
                           100.0 * acc_tot / (double)(steps * draft),
                           n / (double)steps);
                printf("expert cache: %.1f%% hit (%lld reads total, %d pinned/layer)\n",
                       100.0 * m.hit / (tot ? tot : 1), tot, m.npin);
                pin_save(&m);
            }

            free(dlog); free(dbuf); free(mlog); free(mbuf);

            if (!interactive) break;
        }
        free(cached_ids);
        free(history);
    }
    return 0;
}
