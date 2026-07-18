/* gemma4.c — Gemma-4 26B-A4B on a small-RAM machine, by streaming experts.
 *
 * THE BET. Gemma-4 26B-A4B is 25 GB of weights of which ~3.8 B params activate per
 * token. At q4_0 that is 1.31 GB of dense weights (resident) + 12.9 GB of routed
 * experts (3840 of them, 3.19 MiB each). On a 4-8 GB machine the expert set does
 * not fit, so llama.cpp mmaps it and lets the kernel's global 4 KB-page LRU decide
 * what to evict -- a policy that knows nothing about expert granularity, expert
 * hotness, or what the next layer will need. We keep an explicit expert-granular
 * per-layer LRU instead, and we prefetch.
 *
 * THE STRUCTURAL EDGE. In Gemma-4 the router reads the RAW post-attention residual:
 *
 *     residual = h_after_attn
 *     h1 = post_ffn_ln_1( mlp( pre_ffn_ln(residual) ) )      <- dense MLP branch
 *     idx, w = router(residual)                              <- !! needs only residual
 *     h2 = post_ffn_ln_2( experts( pre_ffn_ln_2(residual) ) )
 *     h  = residual + post_ffn_ln(h1 + h2);  h *= layer_scalar
 *
 * So the 8 expert ids are known BEFORE the dense MLP runs. We route first, fire the
 * expert reads at the I/O threads, then compute the MLP -- which hides real NVMe
 * latency behind real arithmetic, exactly, with no prediction. (colibri's GLM path
 * needs PILOT to *guess* next-layer routing at 71.6%; here it is free and exact.)
 * A synchronous mmap fault, which is what llama.cpp does, cannot overlap at all.
 *
 * Everything is q4_0 (see q40.h): weights carry their fp16 scales inline, so one
 * expert is ONE contiguous 4096-aligned byte range -> one pread, no scale seek.
 * With 240 expert reads per token, read COUNT is the thing to minimise.
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
#include "kvq.h"
#include "gpu.h"
#include "openai_json.h"
#include "openai_http.h"

#define MAXL 64
#define MAXTOPK 16
#define MAXDRAFT 16

/* ------------------------------------------------------------------ config */
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

/* one cached expert */
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

    /* LEARNED HOT-EXPERT PIN SET.
     * MoE expert usage is heavily skewed: a minority of experts take a large
     * majority of the routing mass, and which ones is stable across prompts. So we
     * count how often each (layer, expert) is routed to, persist those counts to
     * usage.bin, and on the next run PIN the top-N per layer into slots that the
     * LRU may never evict. On a 4 GB box only 21 of 128 slots per layer exist, so
     * a pure LRU keeps re-reading the same hot experts after they get pushed out by
     * a burst of cold ones; pinning makes the hot set immune to that.
     * This is precisely the policy the OS page cache cannot express -- it has no
     * idea what an "expert" is, let alone which are hot. */
    /* MTP draft head (Gemma4AssistantForCausalLM). Not a separate model: it runs
     * inside this forward, attending into THIS model's KV. See mtp_forward. */
    int mtp;                                /* head loaded */
    int mtp_L, mtp_D, mtp_BB, mtp_nh, mtp_hd, mtp_ghd, mtp_inter, mtp_vocab;
    float mtp_eps, mtp_theta_l, mtp_theta_g, mtp_partial;
    int mtp_types[MAXL];
    const uint8_t *mtp_blob;
    Layer mtp_layers[MAXL];
    W mtp_embed, mtp_norm, mtp_pre, mtp_post;
    int kv_last_slide, kv_last_full;        /* the layers whose KV the head shares */

    /* Post-norm hidden of each row of the last forward (small batches only).
     * The MTP head needs the hidden of the row BEFORE the token it conditions on --
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
     * [n_target_layers * (MAXDRAFT+2) * hidden] */
    float *dflash_target_hidden;
    int dflash_target_hidden_rows;

    int64_t *ucount;                        /* [n_layers * n_experts] routing counts */
    int npin;                               /* pinned slots per layer */
    char usage_path[4096];

    /* KV. Sliding layers keep a ring of `sliding_window` positions; global layers
     * keep the full context. Both store K and V: attention_k_eq_v shares the
     * PROJECTION, not the post-processing (K is k_norm'd and roped, V is v_norm'd
     * and neither), so the two tensors genuinely differ.
     *
     * With --kvq the older positions are TurboQuant-compressed and only the most
     * recent `rwin` are kept in f32. All KV growth is in the 5 global layers --
     * the sliding ones are permanently capped by the window -- so this is a
     * context-length feature, not a general RAM one. */
    float **kv_k, **kv_v;                   /* f32 residual window (or the whole cache) */
    int **ring_pos;                         /* which position occupies each ring slot */
    /* Highest CONFIRMED position. Speculation writes KV for positions that may be
     * rejected; those must never reach the TurboQuant store, because the packed
     * store is write-once-per-position in practice and a rejected draft baked into
     * it cannot be taken back out. INT_MAX = not speculating, everything confirmed. */
    int kv_conf;
    uint8_t **pk, **pv;                     /* packed store, NULL if this layer is f32 */
    Kvq *qk, *qv;
    int rwin;                               /* f32 residual window size */
    int pos;

    /* I/O threads */
    pthread_t *io;
    int n_io;
    pthread_mutex_t mu;
    pthread_cond_t cv, done;
    struct { int layer, eid, slot; } *q;
    int qcap, qhead, qtail, qcount, inflight, stop;
} M;

/* Metal is opt-out, not opt-in: it auto-enables when a device is present. Set to 0
 * by --no-metal, or when gpu_init() fails for any reason. Every GPU call can decline
 * (returning 0) and the CPU path runs instead, so this is never load-bearing. */
static int g_use_gpu = 0;

/* Ring capacity of a layer's KV. Sliding layers need MORE than `sliding_window`
 * slots, for two independent reasons -- and neither is slack:
 *
 *  +1  The MTP head's BIDIRECTIONAL sliding mask attends t >= pos - W (W+1
 *      positions), one further back than the backbone's causal SWA
 *      (t >= pos - W + 1, W positions). Verified against HF.
 *
 *  +MAXDRAFT  Speculation writes AHEAD and then REWINDS. A verify forward writes
 *      positions P..P+d, but on rejection the next step restarts at P+1 and its
 *      attention reaches back to (P+1) - W. The span that must be simultaneously
 *      live is therefore W + d + 1, not W + 1. With a W+1 ring the lookahead
 *      silently evicts the oldest positions of the very window the next step needs:
 *      output degrades and draft acceptance collapses, with nothing to indicate why.
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

/* ------------------------------------------------------------------ manifest */
/* one row of dense.idx */
typedef struct { char name[96]; int64_t off, len; int fmt, O, I; } DEnt;

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

    /* We keep our OWN expert cache, so letting the OS page-cache the same bytes
     * double-buffers them -- on a 4 GB box that duplication is pure waste and it
     * evicts the dense weights we actually need resident. Tell the kernel not to.
     * F_NOCACHE is macOS's O_DIRECT analogue; POSIX_FADV_RANDOM stops Linux from
     * doing useless readahead around 3.19 MiB random reads. */
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
     * numerically the q40_dot_f32 path -- i.e. MORE accurate than the int8 default,
     * not less. It declines (returns 0) if the weights are not GPU-mapped. */
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

/* rotate_half RoPE. inv_freq's tail is ZERO on p-RoPE global layers (freq 0 =>
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

/* ------------------------------------------------------------------ expert I/O */
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
/* Find or evict a slot for (layer,eid). Returns the slot index; sets *need_io if
 * the caller must fetch it. LRU is per layer: an expert only ever competes with
 * the other experts of its own layer, which is the whole point -- a global page
 * LRU (what mmap gives llama.cpp) lets a hot layer-3 expert be evicted by a
 * cold layer-27 one. */
static int slot_for(M *m, int layer, int eid, int *need_io) {
    int S = m->c.slots_per_layer;
    Slot *base = &m->slots[(size_t)layer * S];
    m->ucount[(size_t)layer * m->c.n_experts + eid]++;      /* learn the hot set */
    for (int i = 0; i < S; i++)
        if (base[i].eid == eid) { base[i].used = ++m->tick; *need_io = 0; m->hit++; return i; }
    /* evict the LRU among the UNPINNED slots. If every slot is pinned the caller
     * asked for more distinct experts than the cache holds, which cannot happen:
     * moe_batch chunks by slots_per_layer, and npin < slots_per_layer is enforced. */
    int lru = -1;
    for (int i = 0; i < S; i++)
        if (!base[i].pinned && (lru < 0 || base[i].used < base[lru].used)) lru = i;
    if (lru < 0) lru = 0;
    /* Do NOT publish the eid yet: the buffer still holds the evicted expert until
     * the I/O completes. The worker sets s->eid when the bytes are actually there. */
    base[lru].eid = -1;
    base[lru].used = ++m->tick;
    *need_io = 1;
    m->miss++;
    return lru;
}

/* ------------------------------------------------------------------ forward */
/* ------------------------------------------------------------------ forward
 *
 * ONE code path for prefill and decode: decode is simply S = 1. The reason to
 * unify is the MoE.
 *
 * BATCH-UNION MoE. Token-at-a-time prefill reads 8 experts per layer per token --
 * a 1000-token prompt is 240,000 expert reads of 3.19 MiB. But the S tokens of a
 * batch collectively route to at most min(128, 8*S) DISTINCT experts per layer, so
 * we invert the loop: gather expert -> {rows that chose it}, read each distinct
 * expert ONCE, and apply it to every row that wants it. Prefill I/O collapses from
 * O(S * topk) to O(unique experts) -- bounded by 128 per layer no matter how long
 * the prompt is. Up to a 60x cut.
 *
 * The chunking below exists because the cache has only `slots_per_layer` slots: we
 * process the distinct experts in chunks of that size, and since acquiring a slot
 * bumps it to most-recently-used, nothing acquired within a chunk can be evicted
 * by a later acquisition in the same chunk.
 */
typedef struct {
    float *x, *xn, *q, *k, *v, *o, *mlp, *h1, *h2, *tmp;
    float *rprob, *gate, *up, *eout;
    float *kvk, *kvv;
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

/* softmax router on one row. Fills idx/wts (length topk). */
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
/* apply one loaded expert to every row that routed to it */
/* For CPU prefill, process all rows selected by an expert in two OpenMP passes
 * rather than launching two parallel regions per row. Each row retains distinct
 * q4 activation quantisation and gate scratch, so this changes scheduling only. */
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
             * reads the very bytes the streaming cache pread into it -- no copy. */
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

/* MoE over the whole batch: route every row, then read each DISTINCT expert once. */
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

/* Route and submit the first expert chunk. The caller may now compute the dense
* branch while NVMe reads proceed. */
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
/* Write K/V for `pos`. The f32 residual ring holds the most recent `rwin` positions;
 * the occupant about to be overwritten (position pos - rwin) is TurboQuant-encoded
 * into the packed store on its way out. Recent tokens therefore always attend at full
 * precision -- which is the single thing the upstream results say you cannot skip
 * ("3-4 bit compression without a residual window produces garbage"). */
static void kv_write(M *m, int li, int pos, int nkv, int hd, int cap,
                     const float *k, const float *v) {
    int quant = m->pk[li] != NULL;
    int W = quant ? (m->rwin < cap ? m->rwin : cap) : cap;
    size_t vec = (size_t)nkv * hd;
    int slot = pos % W;

    /* Evict whatever the slot ACTUALLY holds -- do not assume it is position pos-W.
     *
     * Speculative decoding breaks the "each position is written exactly once, in
     * order" invariant: the verify forward writes positions np..np+d-1, and on a
     * partial acceptance the next forward RE-WRITES np+acc (which held a rejected
     * draft). On that second write the slot still holds position np+acc itself, not
     * np+acc-W, so evicting blindly encodes the wrong vector into the TurboQuant
     * store and silently poisons the quantised history. Symptom: output that is
     * perfect until the residual window first spills (rwin tokens in) and then
     * degrades into noise. Tracking the true occupant makes re-writes harmless. */
    if (quant) {
        int old = m->ring_pos[li][slot];
        /* Only compress a CONFIRMED position. An unconfirmed one is a speculative
         * draft that may be rejected and rewritten; dropping it here is safe because
         * it stays in the f32 ring until it is either confirmed (and compressed on a
         * later eviction) or overwritten. This is why --rwin must exceed --ndraft. */
        if (old >= 0 && old != pos && old <= m->kv_conf) {
            const float *ok = m->kv_k[li] + (size_t)slot * vec;
            const float *ov = m->kv_v[li] + (size_t)slot * vec;
            for (int h = 0; h < nkv; h++) {
                size_t idx = (size_t)(old % cap) * nkv + h;
                kvq_encode(&m->qk[li], ok + (size_t)h * hd, m->pk[li] + idx * m->qk[li].bytes);
                kvq_encode(&m->qv[li], ov + (size_t)h * hd, m->pv[li] + idx * m->qv[li].bytes);
            }
        }
        m->ring_pos[li][slot] = pos;
    }
    memcpy(m->kv_k[li] + (size_t)slot * vec, k, sizeof(float) * vec);
    memcpy(m->kv_v[li] + (size_t)slot * vec, v, sizeof(float) * vec);
}

/* Read K and V for position t into kb/vb ([nkv*hd] each). */
static void kv_read(M *m, int li, int t, int pos, int nkv, int hd, int cap,
                    float *kb, float *vb) {
    int quant = m->pk[li] != NULL;
    int W = quant ? (m->rwin < cap ? m->rwin : cap) : cap;
    size_t vec = (size_t)nkv * hd;

    /* In the f32 ring iff the slot really holds t (do not infer it from t > pos - W:
     * speculation can leave a slot holding a different position than the arithmetic
     * suggests). */
    if (!quant || m->ring_pos[li][t % W] == t) {
        memcpy(kb, m->kv_k[li] + (size_t)(t % W) * vec, sizeof(float) * vec);
        memcpy(vb, m->kv_v[li] + (size_t)(t % W) * vec, sizeof(float) * vec);
        return;
    }
    for (int h = 0; h < nkv; h++) {
        size_t idx = (size_t)(t % cap) * nkv + h;
        kvq_decode(&m->qk[li], m->pk[li] + idx * m->qk[li].bytes, kb + (size_t)h * hd);
        kvq_decode(&m->qv[li], m->pv[li] + idx * m->qv[li].bytes, vb + (size_t)h * hd);
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

    /* ---- attention ----
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

        /* K = rope(k_norm(raw));  V = v_norm(raw), NO scale, NO rope.
         * On global layers there is no v_proj (attention_k_eq_v) -- V reuses this
         * same RAW k projection. Note V derives from the raw projection, not K. */
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

        /* the BACKBONE's causal sliding window: W positions, t >= pos - W + 1.
         * (The MTP head uses a different, one-wider window -- see mtp_forward.) */
        int lo = glob ? 0 : (pos - c->sliding_window + 1 < 0 ? 0 : pos - c->sliding_window + 1);
        /* Online softmax: decode every KV position once and retain no ctx-sized
         * score or V scratch. m/z are per-head running maximum/normaliser. */
        float mx[256], z[256];
        if (nh > 256) { fprintf(stderr, "too many attention heads\n"); exit(1); }
        memset(b->o, 0, sizeof(float) * nh * hd);
        for (int hh = 0; hh < nh; hh++) { mx[hh] = -INFINITY; z[hh] = 0.0f; }
        for (int t = lo; t <= pos; t++) {
            kv_read(m, li, t, pos, nkv, hd, cap, b->kvk, b->kvv);
            for (int hh = 0; hh < nh; hh++) {
                const float *qq = b->q + (size_t)hh * hd;
                const float *kk = b->kvk + (size_t)(hh / rep) * hd;
                const float *vv = b->kvv + (size_t)(hh / rep) * hd;
                float score = 0.0f;
                for (int d = 0; d < hd; d++) score += qq[d] * kk[d];
                float nm = score > mx[hh] ? score : mx[hh];
                float a = expf(mx[hh] - nm), w = expf(score - nm), nz = a * z[hh] + w;
                float *ov = b->o + (size_t)hh * hd;
                float old = z[hh] ? a * z[hh] / nz : 0.0f, add = w / nz;
                for (int d = 0; d < hd; d++) ov[d] = old * ov[d] + add * vv[d];
                mx[hh] = nm; z[hh] = nz;
            }
        }
        matvec(b->tmp, &L->o_proj, b->o, b->xq, b->sx);
        rmsnorm(b->tmp, b->tmp, L->post_attn_ln.f, D, c->eps);
        for (int i = 0; i < D; i++) h[i] += b->tmp[i];
    }

    /* ---- feed-forward: dense MLP and MoE are PARALLEL branches on DIFFERENT
     * inputs, both reading the post-attention residual H.
     *
     * The MoE goes FIRST on purpose: routing needs only H, so the expert reads are
     * issued and then the dense MLP is computed while they are in flight. That
     * overlap is exact (not predicted), and it is the thing a synchronous mmap
     * fault -- llama.cpp's expert path -- structurally cannot do. */
    float *res = b->eout;
    memcpy(res, H, sizeof(float) * (size_t)S * D);

    moe_start(m, li, res, b->h2, S, b);

    /* Dense work now overlaps the first chunk's asynchronous expert reads. Store
     * its post-norm result in H; H is no longer needed as an input after res copied. */
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

/* embed one token row (the q4_0 embedding table is also the tied lm_head) */
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

/* Run S tokens. logits may be NULL (prefill), or [S, vocab], or -- the common
 * case -- only the LAST row is wanted, which `last_only` gives. */
static void forward(M *m, const int *ids, int S, int pos_base,
                    float *logits, int last_only, Buf *b) {
    Cfg *c = &m->c;
    int D = c->hidden;
    W *embed = &m->L[MAXL - 1].q_proj;
    W *fnorm = &m->L[MAXL - 1].o_proj;

    float *H = b->x;
    for (int s = 0; s < S; s++) embed_row(m, ids[s], H + (size_t)s * D);

    /* DFlash: capture hidden states at specific backbone layers.
     * We snapshot H after each target layer, BEFORE the next layer overwrites it.
     * The hidden states are post-norm (HF's hidden_states[layer_idx]). */
    if (m->dflash && S <= MAXDRAFT + 1) {
        int ntl = m->dflash_n_target_layers;
        for (int l = 0; l < c->n_layers; l++) {
            layer_fwd(m, l, H, S, pos_base, b);
            /* Check if this layer is a target */
            for (int ti = 0; ti < ntl; ti++) {
                if (m->dflash_target_ids[ti] == l) {
                    /* HF's hidden_states[l+1] = post-norm of layer l's output.
                     * The offset is 1 because hidden_states[0] is the embedding. */
                    for (int s = 0; s < S; s++)
                        rmsnorm(m->dflash_target_hidden + (size_t)ti * (MAXDRAFT + 2) * D + (size_t)s * D,
                                H + (size_t)s * D, fnorm->f, D, c->eps);
                    break;
                }
            }
        }
        m->dflash_target_hidden_rows = S;
    } else {
        for (int l = 0; l < c->n_layers; l++) layer_fwd(m, l, H, S, pos_base, b);
    }

    /* Stash the post-norm hidden (HF's last_hidden_state) of every row for a small
     * batch, or just the final row for a long prefill. The head needs a specific
     * row -- the last ACCEPTED one -- which is only known after verification. */
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

/* ---------------------------------------------------------------- MTP head
 *
 * Gemma4AssistantForCausalLM is a draft HEAD, not a model. It has no k_proj, no
 * v_proj and no k_norm; num_kv_shared_layers == num_hidden_layers, so every one of
 * its layers takes K and V from the BACKBONE's shared_kv_states[layer_type]. The
 * backbone publishes the KV of the LAST layer of each type (store_full_length_kv):
 * here, layer 28 (sliding) and layer 29 (full). The head's 3 sliding layers attend
 * into the former, its full layer into the latter.
 *
 * One draft step:
 *     e   = concat(backbone_hidden, backbone_embed(tok))     [2 * 2816]
 *     h   = pre_projection(e)                                [1024]
 *     h   = 4 plain (non-MoE) decoder layers, attending into the TARGET's KV
 *     h   = norm(h)
 *     logits          = lm_head(h)            -> the drafted token
 *     backbone_hidden'= post_projection(h)    -> feeds the NEXT draft step
 *
 * At q_len == 1 the assistant's bidirectional mask degenerates to full attention
 * over whatever KV it is given (its own comment says so), so we attend over every
 * cached position -- which for the sliding source is the backbone's 1024-window and
 * for the full source is the whole context. That is exactly what the engine caches.
 *
 * THE DRAFT LOOP, taken from transformers' SinglePositionMultiTokenCandidateGenerator
 * (generation/candidate_generator.py) -- NOT guessed. Three things live there rather
 * than in the model, and all three are counter-intuitive:
 *
 *   1. inputs_embeds = cat([last_token_embedding, last_hidden_state], dim=-1)
 *      The EMBEDDING COMES FIRST, then the hidden state. (The natural guess is the
 *      other way round, and it is wrong.)
 *
 *   2. last_token_embedding = target_model_input_embeddings(tok), which is
 *      get_input_embeddings() -- the target's ScaledWordEmbedding, so the embedding
 *      IS multiplied by embed_scale. And last_hidden_state is hidden_states[-1],
 *      which for Gemma4TextModel is POST final norm (verified: it is bit-identical
 *      to outputs.last_hidden_state).
 *
 *   3. position_ids = [[input_ids.shape[1] - 1]] is computed ONCE, BEFORE the draft
 *      loop, and never advanced. Every drafted token is produced from the position
 *      of the last REAL token -- the head does not walk forward. Hence the class
 *      name: SinglePosition. Incrementing the position per draft step (the obvious
 *      thing to do) silently degrades every draft after the first.
 *
 *   4. The drafter is GREEDY: last_token_id = outputs.logits.argmax(-1), regardless
 *      of the target's sampling temperature. */

typedef struct {
    float *e, *h, *xn, *q, *o, *tmp, *gate, *up, *mlp, *kbuf, *vbuf;
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
    int mkv = m->c.n_kv_heads * m->c.head_dim;
    int gkv = m->c.n_global_kv_heads * m->c.global_head_dim;
    if (gkv > mkv) mkv = gkv;
    b->kbuf = xmalloc(sizeof(float) * mkv);
    b->vbuf = xmalloc(sizeof(float) * mkv);
    	b->xq = xmalloc(wide + 64);
    	b->sx = xmalloc(sizeof(float) * (wide / Q40_BLK + 8));
    	return b;
    }

    /* ------------------------------------------------------------------ DFlash
     *
     * DFlash is a block-parallel drafter: instead of drafting one token at a time, it
     * drafts an entire block of `block_size` tokens simultaneously using bidirectional
     * attention. It conditions on hidden states extracted from specific layers of the
     * target backbone (e.g. layers 1, 6, 11, 17, 22, 27 for Gemma-4 26B).
     *
     * The attention is special: for each position in the draft block, K and V come from
     * BOTH the target hidden context (cross-attention) AND the draft block's own tokens
     * (bidirectional self-attention). The two are concatenated along the sequence dim.
     *
     * The model is Qwen3-based (not Gemma4): each layer has q_proj, k_proj, v_proj,
     * o_proj, q_norm, k_norm, and a plain MLP (gate/up/down). No MoE, no sliding window
     * in the backbone sense -- the sliding_window config only affects the attention mask.
     *
     * The draft loop:
     *   1. Extract target hidden states from backbone layers (done in forward()).
     *   2. Project through fc + hidden_norm to get target_hidden.
     *   3. Initialize block with mask tokens, first position = target's sampled token.
     *   4. Run 5 decoder layers with DFlash attention.
     *   5. Pass through target's lm_head to get logits.
     *   6. Sample draft tokens from logits.
     *   7. Verify with ONE batched target forward.
     *   8. Accept matching tokens + one bonus token from target.
     */

    typedef struct {
        /* Per-row hidden states for the draft block [block_size * D] */
        float *h;
        /* Scratch: norm, q (per-position), o, tmp, gate, up, mlp */
        float *xn, *q, *o, *tmp, *gate, *up, *mlp;
        /* Target hidden context: [ctx_len * D] */
        float *target_h;
        /* K/V for the concatenated context+draft sequence */
        float *ctx_k, *ctx_v;
        int8_t *xq; float *sx;
        /* Persistent logit/draft-prob buffers (avoid alloc/free per step) */
        float *dlog, *dprob;
    } DBuf;

    static DBuf *dflash_bufs(M *m) {
        int D = m->dflash_D, BS = m->dflash_block_size;
        int nh = m->dflash_nh, hd = m->dflash_hd, nkv = m->dflash_nkv;
        int qmax = nh * hd;          /* one head's query vector */
        int kvmax = nkv * hd;        /* one position's K/V */
        int wide = D;
        if (wide < qmax) wide = qmax;
        if (wide < m->dflash_inter) wide = m->dflash_inter;
        if (wide < m->dflash_vocab) wide = m->dflash_vocab;

        DBuf *b = calloc(1, sizeof *b);
        b->h = xmalloc(sizeof(float) * (size_t)BS * D);
        b->xn = xmalloc(sizeof(float) * wide);
        /* q: one query vector per draft position [BS * nh * hd] */
        b->q = xmalloc(sizeof(float) * (size_t)BS * qmax);
        b->o = xmalloc(sizeof(float) * qmax);
        b->tmp = xmalloc(sizeof(float) * wide);
        b->gate = xmalloc(sizeof(float) * (m->dflash_inter + 64));
        b->up = xmalloc(sizeof(float) * (m->dflash_inter + 64));
        b->mlp = xmalloc(sizeof(float) * (m->dflash_inter + 64));
        /* target_h: enough for the context (1 row) */
        b->target_h = xmalloc(sizeof(float) * (size_t)(1 + BS) * D);
        /* ctx_k/v: concatenated context + draft positions */
        b->ctx_k = xmalloc(sizeof(float) * (size_t)(1 + BS) * kvmax);
        b->ctx_v = xmalloc(sizeof(float) * (size_t)(1 + BS) * kvmax);
        b->xq = xmalloc(wide + 64);
        b->sx = xmalloc(sizeof(float) * (wide / Q40_BLK + 8));
        /* Persistent logit/draft-prob buffers: [BS * V] each */
        b->dlog = xmalloc(sizeof(float) * (size_t)BS * m->dflash_vocab);
        b->dprob = xmalloc(sizeof(float) * (size_t)BS * m->dflash_vocab);
        return b;
    }

    /* DFlash attention for one layer. `S` is the number of draft positions (block_size),
     * `ctx_len` is the number of target context positions (1 for decode).
     * The attention sees ctx_len + S total K/V positions, with bidirectional mask within
     * the draft block. */
    static void dflash_attn(M *m, int li, float *H, int S, int ctx_len,
                             int pos_base, DBuf *b) {
        int D = m->dflash_D, nh = m->dflash_nh, hd = m->dflash_hd;
        int nkv = m->dflash_nkv, rep = nh / nkv;
        float theta = m->dflash_theta;
        Layer *L = &m->dflash_layers[li];
        int glob = m->dflash_types[li];
        int total = ctx_len + S;  /* total K/V positions */

        /* ---- Q from draft hidden ---- */
        for (int s = 0; s < S; s++) {
            float *h = H + (size_t)s * D;
            rmsnorm(b->xn, h, L->in_ln.f, D, m->dflash_eps);
            matvec(b->q + (size_t)s * nh * hd, &L->q_proj, b->xn, b->xq, b->sx);
            for (int i = 0; i < nh; i++)
                rmsnorm(b->q + (size_t)s * nh * hd + (size_t)i * hd,
                        b->q + (size_t)s * nh * hd + (size_t)i * hd,
                        L->q_norm.f, hd, m->dflash_eps);
            rope(b->q + (size_t)s * nh * hd, nh, hd, pos_base + s, theta, 1.0f);
        }

        /* ---- K/V from target context ----
         * target_h is already projected through fc + hidden_norm, so it's ready to use. */
        for (int t = 0; t < ctx_len; t++) {
            float *th = b->target_h + (size_t)t * D;
            matvec(b->ctx_k + (size_t)t * nkv * hd, &L->k_proj, th, b->xq, b->sx);
            matvec(b->ctx_v + (size_t)t * nkv * hd, &L->v_proj, th, b->xq, b->sx);
            for (int i = 0; i < nkv; i++)
                rmsnorm(b->ctx_k + (size_t)t * nkv * hd + (size_t)i * hd,
                        b->ctx_k + (size_t)t * nkv * hd + (size_t)i * hd,
                        L->k_norm.f, hd, m->dflash_eps);
            rope(b->ctx_k + (size_t)t * nkv * hd, nkv, hd, pos_base, theta, 1.0f);
        }

        /* ---- K/V from draft hidden (self-attention) ----
         * Use the same input_layernorm as Q for consistency with the Qwen3 architecture. */
        for (int s = 0; s < S; s++) {
            float *h = H + (size_t)s * D;
            rmsnorm(b->xn, h, L->in_ln.f, D, m->dflash_eps);
            size_t off = (size_t)(ctx_len + s) * nkv * hd;
            matvec(b->ctx_k + off, &L->k_proj, b->xn, b->xq, b->sx);
            matvec(b->ctx_v + off, &L->v_proj, b->xn, b->xq, b->sx);
            for (int i = 0; i < nkv; i++)
                rmsnorm(b->ctx_k + off + (size_t)i * hd,
                        b->ctx_k + off + (size_t)i * hd,
                        L->k_norm.f, hd, m->dflash_eps);
            rope(b->ctx_k + off, nkv, hd, pos_base + s, theta, 1.0f);
        }

        /* ---- Attention: for each draft position, attend to all ctx + draft positions.
         * Bidirectional within the draft block. The sliding window (2048) is much larger
         * than the block size (16), so it never restricts attention during decode. */
        for (int s = 0; s < S; s++) {
            int lo = 0;  /* attend to all positions in the concatenated sequence */
            (void)glob;  /* sliding window is a no-op for block_size << window */

            float mx[256], z[256];
            if (nh > 256) { fprintf(stderr, "too many DFlash heads\n"); exit(1); }
            memset(b->o, 0, sizeof(float) * nh * hd);
            for (int hh = 0; hh < nh; hh++) { mx[hh] = -INFINITY; z[hh] = 0.0f; }

            for (int t = lo; t < total; t++) {
                for (int hh = 0; hh < nh; hh++) {
                    const float *qqh = b->q + (size_t)s * nh * hd + (size_t)hh * hd;
                    const float *kkh = b->ctx_k + (size_t)t * nkv * hd + (size_t)(hh / rep) * hd;
                    const float *vvh = b->ctx_v + (size_t)t * nkv * hd + (size_t)(hh / rep) * hd;
                    float score = 0.0f;
                    for (int d = 0; d < hd; d++) score += qqh[d] * kkh[d];
                    score /= sqrtf((float)hd);
                    float nm = score > mx[hh] ? score : mx[hh];
                    float a = expf(mx[hh] - nm), w = expf(score - nm), nz = a * z[hh] + w;
                    float *ov = b->o + (size_t)hh * hd;
                    float old = z[hh] ? a * z[hh] / nz : 0.0f, add = w / nz;
                    for (int d = 0; d < hd; d++) ov[d] = old * ov[d] + add * vvh[d];
                    mx[hh] = nm; z[hh] = nz;
                }
            }

            /* output projection + residual */
            matvec(b->tmp, &L->o_proj, b->o, b->xq, b->sx);
            float *h = H + (size_t)s * D;
            for (int i = 0; i < D; i++) h[i] += b->tmp[i];
        }
    }

    /* DFlash forward: run all layers over the draft block.
     * H: [S * D] draft hidden states (initialized with noise embeddings).
     * target_h: [ctx_len * D] target context hidden states (already projected through fc).
     * S: block_size, ctx_len: number of context positions (1 for decode).
     * pos_base: absolute position of the first draft token. */
    static void dflash_forward(M *m, float *H, const float *target_h,
                                int S, int ctx_len, int pos_base, DBuf *b) {
        int D = m->dflash_D;

        /* Copy target hidden into buffer */
        memcpy(b->target_h, target_h, sizeof(float) * (size_t)ctx_len * D);

        for (int li = 0; li < m->dflash_L; li++) {
            /* Post-attention residual add happens inside dflash_attn */
            dflash_attn(m, li, H, S, ctx_len, pos_base, b);

            /* FFN: plain MLP with SiLU activation (Qwen3-style, not Gemma4's gelu_tanh) */
            Layer *L = &m->dflash_layers[li];
            for (int s = 0; s < S; s++) {
                float *h = H + (size_t)s * D;
                rmsnorm(b->xn, h, L->post_attn_ln.f, D, m->dflash_eps);
                matvec(b->gate, &L->mlp_gate, b->xn, b->xq, b->sx);
                matvec(b->up, &L->mlp_up, b->xn, b->xq, b->sx);
                for (int i = 0; i < m->dflash_inter; i++)
                    b->mlp[i] = silu(b->gate[i]) * b->up[i];
                matvec(b->tmp, &L->mlp_down, b->mlp, b->xq, b->sx);
                for (int i = 0; i < D; i++) h[i] += b->tmp[i];
            }
        }

        /* Final norm */
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

        /* Allocate target hidden buffer */
        m->dflash_target_hidden = xmalloc(sizeof(float) * (size_t)m->dflash_n_target_layers
                                           * (MAXDRAFT + 2) * m->c.hidden);
        m->dflash_target_hidden_rows = 0;

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

    /* ---- concat(embed(tok), backbone_hidden) ---- */
    float *emb = b->tmp;
    embed_row(m, tok, emb);                 /* includes embed_scale */
    memcpy(b->e, emb, sizeof(float) * BB);
    memcpy(b->e + BB, bh, sizeof(float) * BB);
    matvec(b->h, &m->mtp_pre, b->e, b->xq, b->sx);         /* [D] */

    /* ---- 4 plain decoder layers, attending into the TARGET's KV ---- */
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

        /* The head's mask is BIDIRECTIONAL, and that is not cosmetic:
         *   full layers   -> mask is None, attend every cached position;
         *   sliding layers-> attend t >= pos - W, i.e. W + 1 positions -- ONE MORE
         *                    than the backbone's causal SWA (t >= pos - W + 1).
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
            kv_read(m, src, t, pos, nkv, hd, cap, b->kbuf, b->vbuf);
            for (int hh = 0; hh < nh; hh++) {
                const float *qq = b->q + (size_t)hh * hd;
                const float *kk = b->kbuf + (size_t)(hh / rep) * hd;
                const float *vv = b->vbuf + (size_t)(hh / rep) * hd;
                float score = 0.0f;
                for (int d = 0; d < hd; d++) score += qq[d] * kk[d];
                float nm = score > mx[hh] ? score : mx[hh];
                float a = expf(mx[hh] - nm), w = expf(score - nm), nz = a * z[hh] + w;
                float *ov = b->o + (size_t)hh * hd;
                float old = z[hh] ? a * z[hh] / nz : 0.0f, add = w / nz;
                for (int d = 0; d < hd; d++) ov[d] = old * ov[d] + add * vv[d];
                mx[hh] = nm; z[hh] = nz;
            }
        }
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
 * layers publish the KV it shares (the LAST layer of each layer_type). */
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

    /* the backbone publishes the KV of the LAST layer of each type
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

/* ------------------------------------------------------------------ setup */
static void init(M *m, const char *dir, int n_io,
                 int kb, int vb, int kv_protect, int rwin, int kv_pbits) {
    m->rwin = rwin > 0 ? rwin : 128;
    manifest(m, dir);
    Cfg *c = &m->c;
    if (c->slots_per_layer < c->topk) {
        fprintf(stderr, "slots_per_layer=%d < topk=%d: raise --ram\n",
                c->slots_per_layer, c->topk);
        exit(1);
    }
    size_t ns = (size_t)c->n_layers * c->slots_per_layer;
    m->slots = calloc(ns, sizeof(Slot));
    for (size_t i = 0; i < ns; i++) {
        m->slots[i].eid = -1;
        m->slots[i].buf = xmalloc(m->esz);
        /* Map each slot once, up front. Slots are reused for different experts, so
         * the mapping stays valid for the whole run -- the streaming layer only ever
         * overwrites the bytes, never the address. */
        if (g_use_gpu && !gpu_map(m->slots[i].buf, m->esz)) {
            fprintf(stderr, "metal: could not map expert slots; using CPU\n");
            g_use_gpu = 0;
        }
    }
    m->ucount = calloc((size_t)c->n_layers * c->n_experts, 8);
    snprintf(m->usage_path, sizeof m->usage_path, "%s/usage.bin", dir);

    m->kv_k = calloc(c->n_layers, sizeof(float *));
    m->kv_v = calloc(c->n_layers, sizeof(float *));
    m->ring_pos = calloc(c->n_layers, sizeof(int *));
    m->pk = calloc(c->n_layers, sizeof(uint8_t *));
    m->pv = calloc(c->n_layers, sizeof(uint8_t *));
    m->qk = calloc(c->n_layers, sizeof(Kvq));
    m->qv = calloc(c->n_layers, sizeof(Kvq));

    size_t kvb = 0;
    for (int l = 0; l < c->n_layers; l++) {
        int glob = c->layer_types[l];
        int hd = glob ? c->global_head_dim : c->head_dim;
        int nkv = glob ? c->n_global_kv_heads : c->n_kv_heads;
        int cap = kv_cap(c, l);          /* sliding ring is W+1: see kv_cap */

        /* Protected layers get MORE BITS, not f32 -- upstream protects with extra
         * precision, and here the distinction is expensive: Gemma-4's LAST layer is
         * a global (full-context) one, so pinning it to f32 would cost 1.07 GiB at
         * 128K ctx and undo most of the saving. High-bit protection costs a fraction
         * of that. */
        int prot = (l < kv_protect) || (l >= c->n_layers - kv_protect);
        int lkb = prot ? kv_pbits : kb;
        int lvb = prot ? kv_pbits : vb;
        int quant = kb > 0;

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
    fprintf(stderr, "kv: %.0f MiB", kvb / 1048576.0);
    if (kb > 0) fprintf(stderr, " (K%d/V%d, rwin %d, %d protected layers at %d bits)",
                        kb, vb, m->rwin, kv_protect, kv_pbits);
    fprintf(stderr, "\n");

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
    b->kvk = xmalloc(sizeof(float) * mkv);
    b->kvv = xmalloc(sizeof(float) * mkv);
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

/* ------------------------------------------------------------------ sampling */
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

/* ------------------------------------------------------------ chat template
 *
 * Transcribed from Gemma-4's chat_template.jinja. Structure:
 *
 *   <bos>
 *   [<|turn>system\n  (<|think|>\n if thinking)  SYSTEM  <turn|>\n]
 *   <|turn>user\n USER <turn|>\n
 *   <|turn>model\n
 *   [<|channel>thought\n<channel|>]      <-- ONLY when thinking is OFF
 *
 * That last line is the non-obvious part: with thinking disabled the template
 * pre-fills an EMPTY thought channel, which is what suppresses reasoning. Omit it
 * and the model will happily start thinking. Conversely `<|think|>` at the top of
 * the system turn is what ENABLES it. The system turn is emitted if there is a
 * system message OR thinking is on (or tools -- not supported here).
 *
 * Generation stops on <turn|> (eot) or <eos>, which are the config's eos ids. */
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

/* ------------------------------------------------------- speculative decoding
 *
 * Gemma-4 ships a dedicated MTP drafter as a SEPARATE model (unlike GLM's in-model
 * layer-78 head), so speculation is: draft d tokens cheaply with the small model,
 * then VERIFY all d with ONE batched forward of the big one.
 *
 * The verification forward is where this pays off for a streaming engine. Those d
 * positions collectively route to at most min(128, 8*d) distinct experts per layer,
 * and batch-union reads each ONCE -- so verifying d tokens costs barely more expert
 * I/O than decoding one. Speculation converts "d tokens of disk latency" into "1
 * token of disk latency", which on a disk-bound engine is the entire game.
 *
 * LOSSLESS. At temp=0 acceptance is exact: keep a draft token iff it is the target's
 * argmax. At temp>0 we use rejection sampling against the draft's own distribution,
 * so the output distribution is IDENTICAL to non-speculative sampling -- speculation
 * changes the speed, never the text.
 *
 * REJECTED KV. If we reject at position j, the KV written for positions >= pos+j is
 * garbage. We do not need to undo it: every KV slot is addressed by absolute
 * position (`pos % cap`), so the next forward simply overwrites it. */

/* `dpos` = how many tokens of `ids` the DRAFTER has actually consumed into its KV.
 * Tracking it is not bookkeeping pedantry: the drafter emits draft[d-1] but never
 * feeds it back, so on full acceptance that position joins the accepted prefix with
 * STALE KV behind it, and the drafter then disagrees with itself. (Observed: an
 * identical drafter/target pair accepting only 75%.) We catch the drafter up on
 * exactly the accepted prefix and no further. */
static int spec_step(M *tgt, M *drf, Buf *bt, Buf *bd,
                     int *ids, int np, int *dpos, float *tlog, float *dlog,
                     int d, float temp, float topp, int topk, PI *pbuf, PI *dbuf,
                     uint64_t *rng, int *out, int *accepted) {
    Cfg *c = &tgt->c;
    int V = c->vocab;
    int draft[MAXDRAFT];
    float *dprob = xmalloc(sizeof(float) * (size_t)d * V);

    /* ---- catch the drafter up on every accepted token it has not seen ---- */
    if (*dpos < np) {
        forward(drf, ids + *dpos, np - *dpos, *dpos, dlog, 1, bd);
        *dpos = np;
    }

    /* ---- draft d tokens ---- */
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
        /* feed it back so the NEXT draft is conditioned on it */
        forward(drf, &draft[i], 1, np + i, dlog, 1, bd);
    }

    /* ---- verify all d with ONE batched target forward ----
     * tlog already holds the target's distribution for position np-1 (i.e. for
     * draft[0]); this call returns the distributions AFTER each drafted token. */
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
     * and its KV is invalid -- mark it unconsumed so the next step re-feeds it. */
    *dpos = np + n;

    free(dprob);
    free(vl);
    return n + 1;                                /* tokens produced this step */
}

/* Numerically diff the Metal kernel against the CPU reference on random q4_0 data.
 * This exists because the Metal path cannot be tested where it was written -- it is
 * the user's machine that decides whether it is right. Run it once before trusting
 * any GPU output. Shapes cover the real ones: attention projections, dense MLP, and
 * both expert matmuls. */
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
 *   P       -- index of the last token in `ids` (its KV is NOT yet in the target)
 *   hprev   -- the target's post-norm hidden at position P-1
 *
 * That pairing is the whole thing, and it is NOT the obvious one. HF does:
 *     last_hidden_state = hidden[n_last_matches]   -> position t
 *     last_token_id     = input_ids[-1]            -> position t+1
 * i.e. the hidden comes from ONE POSITION EARLIER than the token -- the EAGLE
 * convention, concat(e(x_{t+1}), h_t). Pairing h_{t+1} with x_{t+1} (the intuitive
 * thing, and what I did first) collapses acceptance to a few percent.
 *
 * ONE target forward per step, not two: the batch is [x_P, draft_0 .. draft_{d-1}]
 * at positions P..P+d. Row 0 both (a) puts x_P's KV into the cache and (b) yields the
 * distribution that verifies draft_0; row i yields the one that verifies draft_{i+1}.
 * And hidden row `acc` is exactly h_{P+acc}, which is the hprev the NEXT step needs.
 * An extra "resync" forward for the newly sampled token is pure waste -- it was also
 * what forced the wrong pairing above.
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

    /* ---- draft d tokens: (h_{P-1}, x_P) at the FIXED position P ---- */
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

        int am = 0;                                /* the drafter is GREEDY */
        for (int j = 1; j < V; j++) if (dlog[j] > dlog[am]) am = j;
        tok = am;
        draft[i] = tok;
    }
    (void)dbuf;

    /* ---- ONE target forward over [x_P, draft...] at positions P..P+d ---- */
    batch[0] = ids[P];
    for (int i = 0; i < d; i++) batch[i + 1] = draft[i];
    m->kv_conf = P;                 /* anything beyond P is speculative until verified */
    /* logits=NULL: the lm_head is 738 M params (1.5 GFLOP per row), and on a rejection
     * every row past the first is thrown away. Compute it lazily instead, row by row,
     * and stop as soon as a draft is rejected -- at ~48% acceptance that skips roughly
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

/* DFlash block-parallel speculative step.
 *
 * Unlike MTP (which drafts one token at a time), DFlash drafts an entire block of
 * `block_size` tokens at once using bidirectional attention. The draft is conditioned
 * on hidden states extracted from specific backbone layers.
 *
 * Algorithm:
 *   1. Extract target hidden from backbone layers (already captured in forward()).
 *   2. Project through fc + hidden_norm to get target_hidden.
 *   3. Initialize block: first position = target's sampled token, rest = mask tokens.
 *   4. Run DFlash forward to get draft hidden states.
 *   5. Pass through target's lm_head to get logits for each position.
 *   6. Sample draft tokens (greedy for the drafter).
 *   7. Verify with ONE batched target forward.
 *   8. Accept matching tokens + one bonus token from target.
 *
 * `target_hidden_rows` tells us how many rows of target hidden we have (1 for decode,
 * more for prefill). `pos` is the absolute position of the first draft token. */
/* Returns tokens produced. Also writes the next-target logits into `next_logits`
 * (caller must provide a [V] buffer) so the caller does NOT need a separate
 * forward() call — the verify forward already computed everything. */
static int dflash_step(M *m, Buf *bt, DBuf *bd, int *ids, int pos,
                        int target_hidden_rows, float *tlog,
                        int d, float temp, float topp, int topk,
                        PI *pbuf, uint64_t *rng, int *out, int *accepted,
                        float *next_logits) {
    Cfg *c = &m->c;
    int V = c->vocab, D = c->hidden;
    int BS = m->dflash_block_size;
    int ntl = m->dflash_n_target_layers;
    int mask_id = m->dflash_mask_token_id;

    if (d > BS) d = BS;
    if (d < 1) d = 1;

    /* ---- 1. Project target hidden through fc + hidden_norm ---- */
    /* Concatenate hidden states from all target layers for each row.
     * target_hidden: [ntl * target_hidden_rows * D]
     * We need [target_hidden_rows * ntl * D] -> fc [D, ntl*D] -> [target_hidden_rows * D] */
    int ctx_len = target_hidden_rows;
    float *concat_h = xmalloc(sizeof(float) * (size_t)ctx_len * ntl * D);
    for (int r = 0; r < ctx_len; r++) {
        for (int ti = 0; ti < ntl; ti++) {
            memcpy(concat_h + (size_t)r * ntl * D + (size_t)ti * D,
                   m->dflash_target_hidden + (size_t)ti * (MAXDRAFT + 2) * D + (size_t)r * D,
                   sizeof(float) * D);
        }
    }

    /* fc: [D, ntl*D] applied to each row */
    float *target_h = xmalloc(sizeof(float) * (size_t)ctx_len * D);
    for (int r = 0; r < ctx_len; r++) {
        matvec(target_h + (size_t)r * D, &m->dflash_fc,
               concat_h + (size_t)r * ntl * D, bd->xq, bd->sx);
        rmsnorm(target_h + (size_t)r * D, target_h + (size_t)r * D,
                m->dflash_hidden_norm.f, D, m->dflash_eps);
    }
    free(concat_h);

    /* ---- 2. Initialize draft block ---- */
    /* First position: the target's sampled token. Rest: mask tokens. */
    int block_ids[MAXDRAFT];
    block_ids[0] = sample(tlog, V, temp, topp, topk, pbuf, rng);
    for (int i = 1; i < d; i++) block_ids[i] = mask_id;

    /* Embed the block tokens using the TARGET's embedding table */
    float *H = xmalloc(sizeof(float) * (size_t)d * D);
    for (int i = 0; i < d; i++)
        embed_row(m, block_ids[i], H + (size_t)i * D);

    /* ---- 3. Run DFlash forward ---- */
    dflash_forward(m, H, target_h, d, ctx_len, pos, bd);

    /* ---- 4. Compute logits through target's lm_head ---- */
    float *dlog = bd->dlog;
    W *embed = &m->L[MAXL - 1].q_proj;
    W *fnorm = &m->L[MAXL - 1].o_proj;
    for (int i = 0; i < d; i++) {
        /* H is already post-norm from dflash_forward, but we need to re-norm with
         * the target's final norm for the lm_head */
        rmsnorm(bd->xn, H + (size_t)i * D, fnorm->f, D, c->eps);
        matvec(dlog + (size_t)i * V, embed, bd->xn, bd->xq, bd->sx);
        if (c->final_logit_softcap > 0) {
            float cap = c->final_logit_softcap;
            for (int j = 0; j < V; j++)
                dlog[(size_t)i * V + j] = tanhf(dlog[(size_t)i * V + j] / cap) * cap;
        }
    }

    /* ---- 5. Sample draft tokens (greedy for the drafter) ---- */
    int draft[MAXDRAFT];
    float *dprob = bd->dprob;
    for (int i = 0; i < d; i++) {
        float mx = -1e30f;
        for (int j = 0; j < V; j++) if (dlog[(size_t)i * V + j] > mx) mx = dlog[(size_t)i * V + j];
        double sum = 0;
        float T = temp > 0 ? temp : 1.0f;
        float *dp = dprob + (size_t)i * V;
        for (int j = 0; j < V; j++) {
            dp[j] = expf((dlog[(size_t)i * V + j] - mx) / T);
            sum += dp[j];
        }
        for (int j = 0; j < V; j++) dp[j] /= (float)sum;

        int am = 0;
        for (int j = 1; j < V; j++)
            if (dlog[(size_t)i * V + j] > dlog[(size_t)i * V + am]) am = j;
        draft[i] = am;
    }

    /* ---- 6. Verify with ONE batched target forward ----
     * Feed [last_confirmed, draft[0], ..., draft[d-1]] at positions [pos-1, pos, ..., pos+d-1].
     * The token at pos-1 is already in KV; repeating it gives us the hidden state for
     * row 0 (which verifies draft[0]). */
    int batch[MAXDRAFT + 1];
    batch[0] = ids[pos - 1];  /* the last confirmed token */
    for (int i = 0; i < d; i++) batch[i + 1] = draft[i];
    m->kv_conf = pos - 1;  /* only pos-1 is confirmed; pos..pos+d-1 are speculative */
    forward(m, batch, d + 1, pos - 1, NULL, 0, bt);

    /* ---- 7. Verify each draft token ---- */
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
        out[n] = t;
    }

    /* Bonus token from target's distribution */
    if (n == d) lm_head_row(m, m->hid_batch + (size_t)n * c->hidden, q, bt);
    out[n] = sample(q, V, temp, topp, topk, pbuf, rng);

    /* rows 0..n of the batch are on the accepted path (row 0 = pos-1, row 1 = pos, ...).
     * So positions (pos-1)..(pos-1+n) are confirmed. */
    m->kv_conf = pos - 1 + n;
    *accepted = n;

    /* The verify forward already populated hid_batch and dflash_target_hidden.
     * Compute the next-target logits from the last accepted row so the caller
     * can skip the redundant forward() call. */
    if (next_logits) {
        W *fnorm = &m->L[MAXL - 1].o_proj;
        W *embed = &m->L[MAXL - 1].q_proj;
        rmsnorm(bd->xn, m->hid_batch + (size_t)n * c->hidden, fnorm->f, D, c->eps);
        matvec(next_logits, embed, bd->xn, bd->xq, bd->sx);
        if (c->final_logit_softcap > 0) {
            float cap = c->final_logit_softcap;
            for (int i = 0; i < V; i++)
                next_logits[i] = tanhf(next_logits[i] / cap) * cap;
        }
    }

    free(target_h); free(H); free(q);
    return n + 1;
}

/* ------------------------------------------------------------------ OpenAI-compatible local server */
typedef struct {
    M *model;
    Buf *buffers;
    G4Tok *tokenizer;
    pthread_mutex_t generation_mu;
    atomic_int cancel;
    int *cached_ids;
    int cached_len;
    int cached_cap;
    int port;
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
    if (np >= c->ctx) { free(ids); free(logits); free(pbuf); return samosa_http_json_error(fd,400,"context_limit","The prompt leaves no room for completion."); }
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
        char body[512]; snprintf(body,sizeof body,"{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\",\"owned_by\":\"local\"}]}",ctx->model_id);
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
    G4ServerContext ctx={.model=m,.buffers=buffers,.tokenizer=tokenizer,.port=port,.model_id=model_id};
    pthread_mutex_init(&ctx.generation_mu,NULL); atomic_init(&ctx.cancel,0);
    SamosaHttpServer server;
    if (!samosa_http_server_init(&server,port,g4_serve_handler,&ctx)) { fprintf(stderr,"server: cannot bind port %d: %s\n",port,strerror(errno)); pthread_mutex_destroy(&ctx.generation_mu); return 1; }
    fprintf(stderr,"[server] OpenAI endpoint ready at http://127.0.0.1:%d\n",server.port); fflush(stderr);
    int ok=samosa_http_server_run(&server); samosa_http_server_destroy(&server);
    free(ctx.cached_ids); pthread_mutex_destroy(&ctx.generation_mu); return ok?0:1;
}

/* ------------------------------------------------------------------ main */
static void usage(const char *prog, FILE *out) {
    fprintf(out,
        "usage: %s <dir> [flags...] [prompt]\n"
        "         [--chat] [--system S] [--think] [--raw] [--max_tokens N]\n"
        "         [--temp F] [--topp F] [--topk N]   (default 1.0 / 0.95 / 64)\n"
        "         [--pin N] [--draft DIR] [--ndraft N]\n"
        "         [--mtp] [--dflash]\n"
        "         [--io N] [--nobatch] [--threads N]\n"
        "         [--metal] Metal is OFF by default (it is slower if gemma is not fully in RAM)\n"
        "         [--serve] [--port N]    OpenAI-compatible local server (default 8484)\n"
        "         [--kv off|k6v4|k4v2]   KV-cache compression preset\n"
        "         [--kvq] [--kbits N] [--vbits N] [--rwin N] [--protect N] [--pbits N]\n"
        "                                override individual TurboQuant settings\n"
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
    int kvq_on = 0, kb = 6, vb = 4, rwin = 128, kv_protect = 2, kv_pbits = 8;
    int no_metal = 0, chk_gpu = 0, use_mtp = 0, use_dflash = 0, use_metal = 0;
    int check = 0, n_io = 8, max_tokens = 0, nobatch = 0, npin = 0, draft = 0, nthreads = 2;
    int serve_mode = 0, serve_port = 8484;
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
        else if (!strcmp(argv[i], "--max_tokens") && i + 1 < argc) max_tokens = atoi(argv[++i]);

        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--topp") && i + 1 < argc) topp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--topk") && i + 1 < argc) topk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pin") && i + 1 < argc) npin = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--draft") && i + 1 < argc) dpath = argv[++i];
        else if (!strcmp(argv[i], "--ndraft") && i + 1 < argc) draft = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--system") && i + 1 < argc) sys = argv[++i];
        else if (!strcmp(argv[i], "--chat")) chat_mode = 1;
        else if (!strcmp(argv[i], "--think")) think = 1;
        else if (!strcmp(argv[i], "--raw")) raw = 1;
        else if (!strcmp(argv[i], "--kvq")) kvq_on = 1;
        else if (!strcmp(argv[i], "--kv") && i + 1 < argc) {
            /* Named presets. The bits alone do not define a config -- the residual
             * window and the protected-layer count matter as much, and a half-set
             * K4/V2 (rwin=0) is the exact configuration upstream measured as broken.
             * So the presets carry all four numbers together. */
            const char *v = argv[++i];
            if (!strcmp(v, "off")) { kvq_on = 0; }
            else if (!strcmp(v, "k6v4")) {          /* upstream's only EXACT config */
                kvq_on = 1; kb = 6; vb = 4; rwin = 128; kv_protect = 2; kv_pbits = 8;
            } else if (!strcmp(v, "k4v2")) {        /* the only one that fits 256K in 4 GB */
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
        else if (!strcmp(argv[i], "--no-metal")) no_metal = 1;
        else if (!strcmp(argv[i], "--metal")) use_metal = 1;
        else if (!strcmp(argv[i], "--check-gpu")) chk_gpu = 1;
        else if (!strcmp(argv[i], "--mtp")) use_mtp = 1;
        else if (!strcmp(argv[i], "--dflash")) use_dflash = 1;
        else if (argv[i][0] == '-' && argv[i][1] == '-') {
            fprintf(stderr, "unknown flag: %s\n\n", argv[i]);
            usage(argv[0], stderr);
            return 1;
        }
        else if (!prompt) prompt = argv[i];  /* first non-flag positional arg is the prompt */
    }
    /* Default K6/V4 + a 128-token f32 window: the ONLY configuration upstream
     * measured as EXACT on generation. K4/V2 is reachable (--kbits 4 --vbits 2) but
     * upstream's own corrected table has it MISSING the needle at 2K and 4K. */
    if (!kvq_on) kb = vb = 0;
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
    if ((dpath || use_mtp || use_dflash) && draft <= 0) draft = 4;
    if (use_dflash && draft <= 0) draft = 16;  /* DFlash default block size */
#ifdef _OPENMP
    if (nthreads > 0) omp_set_num_threads(nthreads);
#else
    (void)nthreads;
#endif

    /* Metal is opt-IN (--metal), not opt-out.
     *
     * Measured on real hardware it is ~5x SLOWER than the CPU for decode, and the
     * reason is structural, not a tuning miss: this kernel issues one dispatch and
     * one waitUntilCompleted PER MATVEC, and a token needs thousands of them. At
     * batch size 1 that is pure dispatch latency (~0.5 ms each) against ~40 ms of
     * actual arithmetic. A GPU only wins here with far more work per dispatch --
     * whole layers fused into one command buffer, or large prefill batches -- and
     * even then the engine is disk-bound at a 4-8 GB budget. Shipping it on by
     * default would just make everyone slower, so: off unless asked. */
    if (use_metal && !no_metal) g_use_gpu = gpu_init();
    if (chk_gpu) { if (!g_use_gpu) g_use_gpu = gpu_init(); return check_gpu(); }

    M m; memset(&m, 0, sizeof m);
    double t0 = now();
    init(&m, dir, n_io, kb, vb, kv_protect, rwin, kv_pbits);
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
    if (use_mtp && kvq_on && rwin < draft + 2) {
        fprintf(stderr, "--rwin %d is too small for --ndraft %d: the f32 residual "
                "window must hold every unconfirmed speculative position "
                "(need >= %d)\n", rwin, draft, draft + 2);
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
        init(&dm, dpath, n_io, 0, 0, 0, rwin, 8);  /* drafter KV stays f32: it is tiny */
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
            if (*s >= '0' && *s <= '9') { ids[np++] = strtol(s, &s, 10); s--; }

        /* Prefer deq_logits.f32: the numpy oracle run on the DEQUANTISED container
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
         * reproduce the oracle to float precision (any gap is a bug). The default
         * build quantises activations to int8 (Q8_0), which costs ~1e-2 on logits
         * -- real, expected, and not a bug. Argmax must agree either way. */
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

        /* ---- interactive multi-turn chat (--chat, or no prompt for compatibility) ---- */
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
                /* ---- one-shot prefill (prompt given on command line) ---- */
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
                /* DFlash block-parallel speculative decode.
                 * dflash_step does the verify forward internally, which populates
                 * both hid_batch and dflash_target_hidden. It also writes the
                 * next-target logits directly — no separate forward() needed. */
                while (n < max_tokens) {
                    int d = draft;
                    if (d > m.dflash_block_size) d = m.dflash_block_size;
                    if (n + d > max_tokens) d = max_tokens - n;
                    if (d < 1) d = 1;
                    int acc = 0;
                    int pos = np + n;
                    int got = dflash_step(&m, b, dfb, ids, pos,
                                          m.dflash_target_hidden_rows,
                                          logits, d, temp, topp, topk,
                                          pbuf, &rng, out, &acc, logits);
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
