/* ling.c — inclusionAI Ling-3.0-tiny on a small-RAM machine, by streaming experts.
 *
 * The fourth engine in this repo, sharing the streaming machinery of gemma4.c,
 * lfm25.c and maple.c: an expert-granular per-layer LRU instead of the OS page cache,
 * batch-union prefill, a learned pin set in usage.bin, I/O threads, the KVarN KV
 * codec, the FlashHead and the OpenAI server. What differs is the model, and here it
 * differs in the part that decides what the context costs.
 *
 * 7.9 B params, 1.3 B active. 24 layers in a 3:1 stack:
 *
 *   * 18 KDA layers (Kimi Delta Attention). A linear-attention layer: no KV cache at
 *     all, just a 16 x 128 x 128 recurrent state per layer, 1 MiB, THE SAME SIZE AT
 *     POSITION 1 AND POSITION 131072. Like lfm25's short convolutions it is a
 *     recurrence rather than a cache, so it only moves forwards and prefix reuse is
 *     restricted to strict extensions (see forward(), m->rec_pos).
 *   * 6 MLA layers (Multi-head Latent Attention), and those are where the context
 *     lives. They are run ABSORBED: kv_b_proj is folded into the query and the output
 *     by the converter, so the cache holds the 512-wide latent and the 64-wide RoPE
 *     key -- 2.3 KiB per position per layer, against 20 KiB if the per-head keys and
 *     values were expanded. Over the whole model that is 13.8 KiB a token, so 8 K of
 *     context costs 113 MiB rather than the ~1 GiB a 24-layer softmax model of this
 *     shape would want.
 *
 * The FFN is the usual streamed MoE -- 128 routed experts of 1.27 MiB at q4_0, 8
 * firing, layer 0 dense -- with one thing the other three engines do not have: an
 * always-on shared expert beside the routed ones. It is 512-wide and resident, and it
 * runs BETWEEN submitting the expert reads and applying them, which is the only real
 * arithmetic this architecture offers to hide expert I/O behind.
 *
 * The router is grouped rather than plain top-k: sigmoid scores plus a load-balancing
 * bias, 128 experts in 8 groups, the best 4 groups by their top-2 sum, then top-8
 * inside those. See route_row -- the bias picks but never weights.
 *
 * Mixed precision, decided by tools/convert_ling.py: every tensor carries its own
 * format and matvec dispatches per tensor.
 *
 * No MTP: this checkpoint declares num_nextn_predict_layers 0 and ships no MTP tensors.
 *
 * Build:  cc -O3 -march=native -fopenmp ling.c -lm -lpthread -o ling
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
#include "kvarn.h"
#include "gpu.h"
#include "openai_json.h"
#include "openai_http.h"

#define MAXL 64
#define MAXTOPK 16
#define MAXEXPERTS 256
#define MAXGROUP 32
#define MAXPART 64                     /* attention partitions; see mla_attend */

/* tensor formats, as written by tools/convert_ling.py. The numbering is the repo's:
 * 1 = q4_0, 2 = q8_0, 5 = raw int32 (the FlashHead token map). */
#define FMT_F32 0
#define FMT_Q40 1
#define FMT_Q80 2
#define FMT_I32 5

/* config */
typedef struct {
    int hidden, n_layers, n_heads, head_dim, conv_L;
    int q_lora, kv_lora, qk_nope, qk_rope, v_head;
    int n_experts, topk, moe_inter, shared_inter, dense_inter, n_dense_layers;
    int n_group, topk_group, norm_topk_prob;
    int vocab, ctx, slots_per_layer;
    int layer_types[MAXL];                 /* 1 = MLA, 0 = KDA */
    /* FlashHead; n_clusters == 0 means the container was built without one */
    int flash_n_clusters, flash_cluster_size, flash_n_probes;
    int flash_scaled_centroids, flash_n_force;
    float eps, rope_theta, routed_scale, kda_lower_bound;
} Cfg;

/* a weight: q4_0 / q8_0 blob or f32 vector, all views into the resident dense blob */
typedef struct { int fmt; int O, I; const uint8_t *q; const float *f; } W;

typedef struct {
    W in_norm, post_norm;
    /* KDA layers */
    W kq_proj, kk_proj, kv_proj, q_conv, k_conv, v_conv;
    W f_proj, kg_proj, b_proj, A_log, dt_bias, o_norm, o_proj;
    /* MLA layers. kv_b_kt is W_k TRANSPOSED and kv_b_v is W_v, both split per head by
     * the converter: this engine never sees kv_b_proj in its checkpoint shape. */
    W q_a_proj, q_a_norm, q_b_proj, kv_a_proj, kv_a_norm, kv_b_kt, kv_b_v;
    W g_proj, dense;
    /* FFN */
    W mlp_gate, mlp_up, mlp_down;                       /* dense layers (0..ND-1) */
    W router, expert_bias;                              /* MoE layers */
    W shared_gate, shared_up, shared_down;              /* the always-on expert */
} Layer;

typedef struct { int eid; uint64_t used; uint8_t *buf; int pinned, busy; } Slot;

typedef struct {
    Cfg c;
    Layer L[MAXL];
    const uint8_t *dense;   size_t dense_len;
    int efd;                                /* experts.bin */
    /* Per layer: --expert-edge can give different layers different expert formats and
     * therefore different sizes. Zero on the dense layer. */
    int64_t esz[MAXL], gate_b[MAXL], down_b[MAXL];
    int expert_fmt[MAXL];
    int64_t *eoff;                          /* [layer*n_experts + eid] -> file offset */

    Slot *slots;                            /* [n_layers][slots_per_layer] */
    uint64_t tick;
    int64_t hit, miss;

    /* Learned hot-expert pin set. Expert usage is heavily skewed and the hot set is
     * stable across prompts, so routing is counted per (layer, expert), persisted to
     * usage.bin, and on the next run the top-N per layer are pinned into slots the
     * LRU may never evict -- a policy the OS page cache cannot express. */
    int64_t *ucount;
    int npin;
    char usage_path[4096];

    /* MLA cache, on the MLA layers only (NULL on KDA layers).
     *
     * Two pieces, because they are compressed differently. kv_c holds the normalised
     * kv_lora latent, which is the whole of the key AND the whole of the value in
     * absorbed form, so it gets KVarN's 4-bit key budget; older tiles move out of an
     * f32 ring of `rwin` positions into the code planes. kv_r holds the qk_rope key,
     * 64 wide and shared by every head, and stays f32 for all `ctx` positions: it is
     * a ninth of the bytes and compressing it would buy nothing worth the code. */
    float **kv_c, **kv_r;
    int **ring_pos;
    uint8_t **pc;
    Kvarn *qc;
    int rwin;                               /* f32 ring, in positions: whole tiles */
    int tile;                               /* KVarN tile width, in tokens */

    /* KDA recurrence, on the KDA layers only.
     *   kda_state [n_heads * head_dim * head_dim], S[h][v][k], v-major
     *   conv_state [3][conv_L-1][n_heads*head_dim], q|k|v, oldest position first
     * rec_pos is how many positions have been absorbed, and it is why prefix reuse is
     * restricted (see forward()). */
    float **kda_state, **conv_state;
    int rec_pos;

    /* --check only. chk_route records every routing decision this run made,
     * [n_layers][ctx][topk]; pin_route, when set, REPLACES them with the oracle's.
     *
     * Grouped top-k is a discrete function of the hidden state, so the int8 build can
     * legitimately route differently from the f32 oracle -- and when it does, the
     * logits move by far more than the int8 arithmetic explains. Rather than fold the
     * two effects into one tolerance, --check measures them separately: one pass with
     * free routing to count the disagreements, one with the oracle's picks pinned to
     * measure the arithmetic alone. */
    int *chk_route, *pin_route;
    int chk_stride;                         /* row stride of both, in picks */

    /* FlashHead: an approximate lm_head, see flash_logits(). Bound only when the
     * container carries one; use_flash is what the decode path tests. */
    W flash_centroids, flash_token_map, flash_cluster_scale, flash_force;
    int use_flash;
    int flash_check;
    long long flash_steps, flash_agree, flash_probed;
    double flash_worst;              /* max |probed logit - exact logit| */
    float *flash_exact;

    /* RoPE inverse frequencies, [qk_rope/2]. Only the rope half of the MLA key
     * rotates, and the KDA layers have no positional encoding at all -- position
     * enters them through the recurrence. */
    float *inv_freq;

    /* I/O threads */
    pthread_t *io;
    int n_io;
    pthread_mutex_t mu;
    pthread_cond_t cv, done;
    struct { int layer, eid, slot; } *q;
    int qcap, qhead, qtail, qcount, inflight, stop;
} M;

/* Metal is off by default here, as in the other three: 1.3 B active params over 1.27
 * MiB experts makes a decode step a stream of small matvecs, and per-dispatch latency
 * plus the activation copy usually costs more than the arithmetic saved. --metal turns
 * it on; it is worth trying on prefill, which is batched and compute-bound. Every GPU
 * call can decline (returning 0) and the CPU path runs instead. */
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

/* format dispatch */
static inline int64_t fmt_row_bytes(int fmt, int I) {
    return fmt == FMT_Q40 ? q40_row_bytes(I)
         : fmt == FMT_Q80 ? q80_row_bytes(I)
         : (int64_t)I * 4;
}
/* COLI_F32ACT picks one of the two wdots below; the other is legitimately unused. */
#define MAYBE_UNUSED __attribute__((unused))

/* one weight row . int8-quantised activations.
 *
 * There is deliberately no FMT_F32 case: an f32 weight has no block scales to pair the
 * activation's against, so the caller must take the f32 path instead. matvec/matmul do
 * that. The two hand-rolled folds in mla_fwd cannot -- they walk a sub-block of a
 * tensor rather than the whole of it -- so they test the format themselves, and this
 * asserts rather than silently reinterpreting the bytes as q8_0. */
MAYBE_UNUSED static inline float wdot(int fmt, const uint8_t *w, const int8_t *xq,
                                      const float *sx, int I) {
    return fmt == FMT_Q40 ? q40_dot(w, xq, sx, I) : q80_dot(w, xq, sx, I);
}
MAYBE_UNUSED static inline int quantisable(int fmt) {
    return fmt == FMT_Q40 || fmt == FMT_Q80;
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

/* manifest */
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
            else if (!strcmp(k, "conv_L"))          c->conv_L = atoi(v);
            else if (!strcmp(k, "q_lora"))          c->q_lora = atoi(v);
            else if (!strcmp(k, "kv_lora"))         c->kv_lora = atoi(v);
            else if (!strcmp(k, "qk_nope"))         c->qk_nope = atoi(v);
            else if (!strcmp(k, "qk_rope"))         c->qk_rope = atoi(v);
            else if (!strcmp(k, "v_head"))          c->v_head = atoi(v);
            else if (!strcmp(k, "n_experts"))       c->n_experts = atoi(v);
            else if (!strcmp(k, "topk"))            c->topk = atoi(v);
            else if (!strcmp(k, "moe_inter"))       c->moe_inter = atoi(v);
            else if (!strcmp(k, "shared_inter"))    c->shared_inter = atoi(v);
            else if (!strcmp(k, "dense_inter"))     c->dense_inter = atoi(v);
            else if (!strcmp(k, "n_dense_layers"))  c->n_dense_layers = atoi(v);
            else if (!strcmp(k, "n_group"))         c->n_group = atoi(v);
            else if (!strcmp(k, "topk_group"))      c->topk_group = atoi(v);
            else if (!strcmp(k, "vocab"))           c->vocab = atoi(v);
            else if (!strcmp(k, "ctx"))             c->ctx = atoi(v);
            else if (!strcmp(k, "slots_per_layer")) c->slots_per_layer = atoi(v);
            else if (!strcmp(k, "norm_topk_prob"))  c->norm_topk_prob = atoi(v);
            else if (!strcmp(k, "eps"))             c->eps = atof(v);
            else if (!strcmp(k, "rope_theta"))      c->rope_theta = atof(v);
            else if (!strcmp(k, "routed_scale"))    c->routed_scale = atof(v);
            else if (!strcmp(k, "kda_lower_bound")) c->kda_lower_bound = atof(v);
            else if (!strcmp(k, "flash_n_clusters"))   c->flash_n_clusters = atoi(v);
            else if (!strcmp(k, "flash_cluster_size")) c->flash_cluster_size = atoi(v);
            else if (!strcmp(k, "flash_n_probes"))     c->flash_n_probes = atoi(v);
            else if (!strcmp(k, "flash_scaled_centroids")) c->flash_scaled_centroids = atoi(v);
            else if (!strcmp(k, "flash_n_force"))      c->flash_n_force = atoi(v);
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
    if (c->topk < 1 || c->topk > MAXTOPK || c->topk > c->n_experts) {
        fprintf(stderr, "invalid router topk=%d for %d experts (MAXTOPK=%d)\n",
                c->topk, c->n_experts, MAXTOPK);
        exit(1);
    }
    if (c->n_group < 1 || c->n_group > MAXGROUP || c->n_experts % c->n_group ||
        c->topk_group < 1 || c->topk_group > c->n_group) {
        fprintf(stderr, "invalid router grouping: %d experts in %d groups, top %d\n",
                c->n_experts, c->n_group, c->topk_group);
        exit(1);
    }
    /* The fixed-size scratch rows in kda_fwd and rope_interleave; a container past
     * these would corrupt the stack rather than run slowly. */
    if (c->kv_lora > KVARN_MAXD || c->head_dim > 512 || c->qk_rope > 512) {
        fprintf(stderr, "kv_lora=%d / head_dim=%d / qk_rope=%d beyond what the "
                "buffers allow\n", c->kv_lora, c->head_dim, c->qk_rope);
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

    /* Embeddings are NOT tied here, so unlike lfm25 there really are two tables.
     * Both live in a spare Layer slot, with the final norm. */
    m->L[MAXL - 1].kq_proj = dense_bind(dd, ndense, blob, "embed");
    m->L[MAXL - 1].kk_proj = dense_bind(dd, ndense, blob, "lm_head");
    m->L[MAXL - 1].o_proj  = dense_bind(dd, ndense, blob, "final_norm");

    if (c->flash_n_clusters) {
        m->flash_centroids     = dense_bind(dd, ndense, blob, "flash_centroids");
        m->flash_token_map     = dense_bind(dd, ndense, blob, "flash_token_map");
        m->flash_cluster_scale = dense_bind(dd, ndense, blob, "flash_cluster_scale");
        m->flash_force         = dense_bind(dd, ndense, blob, "flash_force");
    }

    char nm[128];
    for (int l = 0; l < c->n_layers; l++) {
        Layer *L = &m->L[l];
        #define B(f, s) do { snprintf(nm, sizeof nm, "layers.%d." s, l); \
                             L->f = dense_bind(dd, ndense, blob, nm); } while (0)
        B(in_norm, "input_layernorm");
        B(post_norm, "post_attention_layernorm");
        if (c->layer_types[l]) {
            B(q_a_proj, "q_a_proj"); B(q_a_norm, "q_a_norm"); B(q_b_proj, "q_b_proj");
            B(kv_a_proj, "kv_a_proj"); B(kv_a_norm, "kv_a_norm");
            B(kv_b_kt, "kv_b_kt"); B(kv_b_v, "kv_b_v");
            B(g_proj, "g_proj"); B(dense, "dense");
        } else {
            B(kq_proj, "q_proj"); B(kk_proj, "k_proj"); B(kv_proj, "v_proj");
            B(q_conv, "q_conv"); B(k_conv, "k_conv"); B(v_conv, "v_conv");
            B(f_proj, "f_proj"); B(kg_proj, "kg_proj"); B(b_proj, "b_proj");
            B(A_log, "A_log"); B(dt_bias, "dt_bias");
            B(o_norm, "o_norm"); B(o_proj, "o_proj");
        }
        if (l < c->n_dense_layers) {
            B(mlp_gate, "mlp_gate"); B(mlp_up, "mlp_up"); B(mlp_down, "mlp_down");
        } else {
            B(router, "router"); B(expert_bias, "expert_bias");
            B(shared_gate, "shared_gate"); B(shared_up, "shared_up");
            B(shared_down, "shared_down");
        }
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
/* q . k over one vector.
 *
 * Eight accumulators rather than one. A single running sum makes this a serial chain
 * of float adds, four cycles of latency each, which bounds the loop at four cycles an
 * element however wide the vector unit is. There is one of these per cached position
 * per head, and at kv_lora 512 it is the largest single cost in a long-context decode.
 * Eight is an AVX vector's width and enough chains to cover the latency. Reordering
 * the sum moves the result by about 1e-6 relative, well under the KV quantiser's own
 * error. */
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
static inline float sigmoidf_(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* y = W x. COLI_F32ACT keeps activations in f32 (weights stay quantised) so --check
 * can separate the int8-activation approximation from an actual bug. */
static void matvec(float *y, const W *w, const float *x, int8_t *xq, float *sx) {
    int64_t rb = fmt_row_bytes(w->fmt, w->I);
    /* The Metal kernels consume f32 activations, so they reproduce the wdot_f32 path,
     * more accurate than the int8 default rather than less. They decline for f32
     * tensors and for weights that are not GPU-mapped. */
    if (g_use_gpu && gpu_matmul(w->fmt, y, w->q, x, w->O, w->I, 1)) return;
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

/* Y[S,O] = X[S,I] * W^T.
 *
 * For weight reuse, not arithmetic: a matvec streams the whole [O,I] matrix per output
 * vector and the engine is squarely bandwidth-bound. Here each row is loaded once and
 * dotted against all S activations while it is still in cache. */
static void matmul(float *Y, const W *w, const float *X, int S,
                   int8_t *xq, float *sx) {
    if (S == 1) { matvec(Y, w, X, xq, sx); return; }
    /* One dispatch fills the whole O*S grid, which is where Metal has a chance:
     * prefill is batched and genuinely compute-bound. */
    if (g_use_gpu && gpu_matmul(w->fmt, Y, w->q, X, w->O, w->I, S)) return;
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

/* RoPE, interleaved (config.rope_interleave). The checkpoint stores the rotary half
 * with the two members of each pair ADJACENT, so HF de-interleaves to the halves
 * layout and then applies the usual rotate-half. Composed, that is exactly:
 *
 *     out[i]        = x[2i] * cos - x[2i+1] * sin
 *     out[D/2 + i]  = x[2i+1] * cos + x[2i] * sin
 *
 * and the result STAYS in the halves layout, which is what the dot product below
 * consumes. Both the query's rotary part and the single shared rope key go through
 * this, so any consistent convention would agree -- but the value of each individual
 * component is only right with this one, and there is no qk-norm here to hide it.
 *
 * This CANNOT be done in place. The read at i is x[2i], x[2i+1] and the writes are at
 * i and half+i, so from i = half/2 onwards the sources are slots earlier iterations
 * already overwrote. Hence the scratch row.
 *
 * cos/sin are computed once for the whole call rather than per head: they depend only
 * on the dimension. */
static void rope_interleave(float *x, int H, int stride, int D, int pos,
                            const float *invf) {
    int half = D / 2;
    float co[256], si[256], t[512];
    for (int i = 0; i < half; i++) {
        float a = pos * invf[i];
        co[i] = cosf(a); si[i] = sinf(a);
    }
    for (int h = 0; h < H; h++) {
        float *v = x + (size_t)h * stride;
        for (int i = 0; i < half; i++) {
            float x1 = v[2 * i], x2 = v[2 * i + 1];
            t[i]        = x1 * co[i] - x2 * si[i];
            t[half + i] = x2 * co[i] + x1 * si[i];
        }
        memcpy(v, t, sizeof(float) * (size_t)D);
    }
}

/* expert I/O */
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
    float *x, *xn, *tmp, *eout, *h2;
    int8_t *mxq; float *msx;         /* batched-activation scratch for matmul */
    /* KDA scratch, all [S, P] with P = n_heads * head_dim */
    float *kq, *kk, *kv, *kf, *kg, *kbeta, *ko;
    /* MLA scratch */
    float *qa;        /* [S, q_lora] */
    float *qb;        /* [S, n_heads * (qk_nope + qk_rope)] */
    float *kva;       /* [S, kv_lora + qk_rope] */
    float *mgate;     /* [S, n_heads] head-wise gate logits */
    float *qc;        /* [n_heads * kv_lora] the absorbed query, one position */
    float *mqr;       /* [n_heads * qk_rope] the rope query, gathered per head */
    float *mo;        /* [S, n_heads * v_head] */
    float *apart;     /* [npart][n_heads*(kv_lora + 2)] partition accumulators */
    int npart;
    /* FFN */
    float *gate, *up, *mlp;
    int moe_nu, moe_chunk_n, moe_prefetch, moe_slots[MAXEXPERTS];
    float *egate, *esx, *ehs;
    int8_t *exq, *ehq;
    int8_t *xq, *hq;
    float *sx, *hs;
    int *eidx;        /* [S * topk] chosen expert per (row, slot) */
    float *ewt;       /* [S * topk] weight */
    float *rwt;       /* [n_experts] unbiased sigmoid scores of one row */
    float *rsel;      /* [n_experts] biased selection scores of one row */
    int *rows;        /* scratch: rows routed to one expert */
    float *roww;      /* scratch: their weights */
    int *uniq;        /* distinct experts in the batch */
    /* FlashHead scratch */
    float *fsim;      /* [n_clusters] centroid scores */
    int *fsel;        /* [n_clusters] index scratch for the top-probe selection */
    int *fids;        /* [n_probes*cluster_size + n_force] candidate token ids */
    /* GPU expert scratch: the rows routed to one expert are scattered through X, and
     * the kernel wants them contiguous, so they are gathered here and scattered back. */
    float *gx, *gg, *gu, *gd;
    int S;
} Buf;

/* Grouped top-k routing (topk_method noaux_tc).
 *
 * Three stages, and each of them changes which experts fire:
 *
 *   1. scores = sigmoid(router . x), in f32. The checkpoint sets router_dtype fp32
 *      because near-ties at top-8 flip a few percent of picks in bf16, and that
 *      compounds over 23 layers.
 *   2. the experts are cut into `n_group` contiguous groups; a group is ranked by the
 *      SUM OF ITS TOP TWO biased scores, and only the best `topk_group` groups survive.
 *      This is what makes the routing hardware-friendly upstream, and it is not
 *      equivalent to a plain top-k: an expert with the second highest score overall
 *      does not fire if it sits in a group whose other members are weak.
 *   3. top-k among the surviving experts, again on the biased score.
 *
 * expert_bias picks the experts but never weights them: the weight applied is the
 * UNBIASED sigmoid, renormalised over the k kept and scaled by routed_scale. The bias
 * is a load-balancing nudge, and letting it leak into the weight would rescale every
 * expert's contribution by an amount unrelated to the input.
 *
 * Note what cannot be done here, which maple's router does: there the weight is a
 * softmax of the same logits the selection uses, so the full softmax cancels against
 * the renormalisation and only the top k need exponentiating. Here selection and
 * weighting are different quantities, so every score is needed. */
static void route_row(M *m, int li, int pos, const float *xn, int *idx, float *wts,
                      Buf *b) {
    Cfg *c = &m->c;
    Layer *L = &m->L[li];
    int D = c->hidden, E = c->n_experts, K = c->topk;
    int NG = c->n_group, TG = c->topk_group, EG = E / NG;
    const float *RP = L->router.f;
    const float *BI = L->expert_bias.f;

    /* Eight accumulators, for the reason head_dot has them, but double rather than
     * float: the width is about the dependency chain, the type is about step 1 above.
     * Reassociating within double moves a score by ~1e-16 relative, five orders under
     * the f32 it is rounded to before anything compares it.
     *
     * Parallel over experts because this is n_experts x hidden of f32 per layer over
     * 23 layers, and because it sits in front of moe_submit_chunk: time spent here is
     * time the expert reads have not started. */
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < E; e++) {
        const float *r = RP + (size_t)e * D;
        double s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int i = 0;
        for (; i + 8 <= D; i += 8)
            for (int k = 0; k < 8; k++) s[k] += (double)r[i + k] * xn[i + k];
        for (; i < D; i++) s[0] += (double)r[i] * xn[i];
        double v = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
        b->rwt[e] = 1.0f / (1.0f + expf(-(float)v));
        b->rsel[e] = b->rwt[e] + BI[e];
    }

    /* group score = sum of the group's two best biased scores */
    float gs[MAXGROUP];
    for (int g = 0; g < NG; g++) {
        float a = -INFINITY, bb = -INFINITY;
        const float *s = b->rsel + (size_t)g * EG;
        for (int i = 0; i < EG; i++) {
            if (s[i] > a) { bb = a; a = s[i]; }
            else if (s[i] > bb) bb = s[i];
        }
        gs[g] = (EG > 1) ? a + bb : a;
    }
    /* keep the best topk_group groups. Strict `>` so ties keep the lower index, which
     * is what torch.topk does on this data. */
    int keep[MAXGROUP];
    float topg[MAXGROUP];
    for (int j = 0; j < TG; j++) { keep[j] = -1; topg[j] = -INFINITY; }
    for (int g = 0; g < NG; g++) {
        int j = TG;
        while (j > 0 && gs[g] > topg[j - 1]) j--;
        if (j == TG) continue;
        for (int t = TG - 1; t > j; t--) { topg[t] = topg[t - 1]; keep[t] = keep[t - 1]; }
        topg[j] = gs[g]; keep[j] = g;
    }
    unsigned char live[MAXGROUP] = {0};
    for (int j = 0; j < TG; j++) if (keep[j] >= 0) live[keep[j]] = 1;

    /* top-k among the surviving experts, in ascending expert order so ties again keep
     * the lower index */
    float topv[MAXTOPK];
    for (int j = 0; j < K; j++) { idx[j] = -1; topv[j] = -INFINITY; }
    for (int e = 0; e < E; e++) {
        if (!live[e / EG]) continue;
        float sel = b->rsel[e];
        int j = K;
        while (j > 0 && sel > topv[j - 1]) j--;
        if (j == K) continue;
        for (int t = K - 1; t > j; t--) { topv[t] = topv[t - 1]; idx[t] = idx[t - 1]; }
        topv[j] = sel; idx[j] = e;
    }

    /* --check with pinned routing: keep this row's own scores (the weight is a
     * continuous function of them and is exactly what is under test) but take the
     * SELECTION from the oracle, so the comparison downstream is not measuring a
     * different expert. */
    if (m->pin_route && pos < m->chk_stride) {
        const int *pr = m->pin_route + ((size_t)li * m->chk_stride + pos) * K;
        if (pr[0] >= 0) for (int j = 0; j < K; j++) idx[j] = pr[j];
    }

    float sum = 0.0f;
    for (int j = 0; j < K; j++) { wts[j] = b->rwt[idx[j]]; sum += wts[j]; }
    if (c->norm_topk_prob && K > 1)
        for (int j = 0; j < K; j++) wts[j] /= (sum + 1e-20f);
    for (int j = 0; j < K; j++) wts[j] *= c->routed_scale;
}

/* Apply one loaded expert to every row that routed to it.
 *
 * The loop order matters. Parallelising over token rows -- one thread per token --
 * walks the whole expert per thread, streaming its 1.27 MiB once per row.
 * Parallelising over output rows instead loads a weight row once and dots it against
 * every activation while it is still in L1: same flops, nrows-fold less bandwidth. */
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
    /* GPU path. The expert slot was gpu_map'd at allocation, so the weights are read
     * in place; only the participating activation rows move, and they are scattered
     * through X, hence the gather into b->gx and the scatter back out of b->gd. */
    if (g_use_gpu) {
        for (int r = 0; r < nrows; r++)
            memcpy(b->gx + (size_t)r * D, X + (size_t)rows[r] * D, sizeof(float) * D);
        if (gpu_matmul(fmt, b->gg, G, b->gx, MI, D, nrows) &&
            gpu_matmul(fmt, b->gu, U, b->gx, MI, D, nrows)) {
            for (size_t i = 0; i < (size_t)nrows * MI; i++)
                b->gg[i] = silu(b->gg[i]) * b->gu[i];
            if (gpu_matmul(fmt, b->gd, Dn, b->gg, D, MI, nrows)) {
                for (int r = 0; r < nrows; r++) {
                    float *out = OUT + (size_t)rows[r] * D;
                    const float *dr = b->gd + (size_t)r * D;
                    for (int o = 0; o < D; o++) out[o] += w[r] * dr[o];
                }
                return;
            }
        }
        /* any decline above and we simply fall through to the CPU */
    }
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

/* Batch-union MoE. Token-at-a-time prefill reads topk experts per layer per token, but
 * the S rows of a batch collectively route to at most min(n_experts, topk*S) distinct
 * experts, so the loop is inverted: gather expert -> {rows that chose it}, read each
 * once, apply it to all of them. A prompt of any length then reads at most n_experts
 * per layer.
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

/* The router reads the post-attention-normed hidden, which is also the experts' input,
 * so as in lfm25 and maple (and unlike gemma4, whose router reads the raw residual)
 * there is no earlier routing signal to prefetch from. What this model does have that
 * they do not is the shared expert: it consumes the same input, it is resident, and it
 * is real arithmetic that can run while the routed reads are in flight. See
 * layer_fwd. */
static void moe_start(M *m, int li, const float *X, float *out, int S, int pos_base,
                      Buf *b) {
    Cfg *c = &m->c;
    int D = c->hidden, K = c->topk;
    for (int s = 0; s < S; s++)
        route_row(m, li, pos_base + s, X + (size_t)s * D,
                  b->eidx + s * K, b->ewt + s * K, b);
    if (m->chk_route)
        for (int s = 0; s < S && pos_base + s < m->chk_stride; s++)
            memcpy(m->chk_route + ((size_t)li * m->chk_stride + pos_base + s) * K,
                   b->eidx + s * K, sizeof(int) * (size_t)K);
    b->moe_nu = 0;
    for (int s = 0; s < S; s++) for (int j = 0; j < K; j++) {
        int e = b->eidx[s * K + j], seen = 0;
        for (int u = 0; u < b->moe_nu; u++) if (b->uniq[u] == e) { seen = 1; break; }
        if (!seen) b->uniq[b->moe_nu++] = e;
    }
    memset(out, 0, sizeof(float) * (size_t)S * D);
    /* moe_finish submits chunk n+1 before applying chunk n, so two chunks are resident
     * at once and need disjoint unpinned slots. Below 2 free slots there is no room for
     * the second, so the overlap is dropped rather than evicting a slot still in use.
     *
     * free_slots/2 bounds a chunk from above; it is not the target. Taking it whenever
     * the union is small makes the overlap disappear: at decode S is 1, so moe_nu is at
     * most topk against many more free slots, the whole union goes out as one chunk, and
     * the layer stalls on the read. So aim for two chunks, and fall back to the bound
     * only where it binds. */
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
        /* Submit the next chunk before applying this one, so its I/O overlaps the CPU
         * expert_apply. */
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

/* The always-on expert, a plain SwiGLU MLP on the same input the router read. Adds
 * into `out` rather than overwriting it, because the routed experts have already
 * accumulated there. */
static void shared_fwd(M *m, int li, const float *X, float *out, int S, Buf *b) {
    Cfg *c = &m->c;
    Layer *L = &m->L[li];
    int D = c->hidden, SI = c->shared_inter;
    matmul(b->gate, &L->shared_gate, X, S, b->mxq, b->msx);
    matmul(b->up, &L->shared_up, X, S, b->mxq, b->msx);
    for (int i = 0; i < S * SI; i++) b->mlp[i] = silu(b->gate[i]) * b->up[i];
    matmul(b->tmp, &L->shared_down, b->mlp, S, b->mxq, b->msx);
    for (int i = 0; i < S * D; i++) out[i] += b->tmp[i];
}

/* MLA cache
 *
 * The same ring-plus-tiles scheme the other engines use for K and V, over one vector
 * per position instead of two per head: in absorbed form the kv_lora latent IS both.
 * With --kv the f32 ring holds at least the most recent `rwin` positions and older
 * ones are KVarN-encoded on their way out; without it the ring is the whole cache.
 * The rope key is written straight to its own full-length f32 array.
 *
 * KVarN quantises a whole tile of `tile` consecutive tokens at once, so eviction is
 * not per position: the tile whose first slot is about to be overwritten is sealed
 * just before that write, exactly once, at the last moment all of its tokens are
 * still resident. */
static void kv_write(M *m, int li, int pos, const float *cvec, const float *rvec) {
    Cfg *c = &m->c;
    int cap = c->ctx, KL = c->kv_lora, RD = c->qk_rope;
    int quant = m->pc[li] != NULL;
    int Wr = quant ? (m->rwin < cap ? m->rwin : cap) : cap;
    int g = m->tile;
    int slot = pos % Wr;

    /* Wr == cap means the ring is the whole cache and nothing ever falls out of it.
     * That is the only case where Wr need not be a multiple of the tile, so it is also
     * the case where no sealing may be attempted. */
    if (quant && Wr < cap && pos >= Wr && pos % g == 0) {
        int base = pos - Wr;                 /* first position of the tile going out */
        int ready = 1;
        /* Seal only a tile that is complete and holds the positions it should. A
         * rewritten position would otherwise bake the wrong vector into the store,
         * unrecoverably. */
        for (int i = 0; i < g; i++)
            if (m->ring_pos[li][(base + i) % Wr] != base + i) { ready = 0; break; }
        if (ready) {
            /* base is a multiple of both Wr and the tile, so the tile's g positions sit
             * in g contiguous ring slots. */
            int ts = (base / g) % kvarn_ntiles(cap, g);
            kvarn_encode_tile(&m->qc[li], m->kv_c[li] + (size_t)(base % Wr) * KL,
                              KL, g, m->pc[li] + (size_t)ts * m->qc[li].bytes);
        }
    }
    if (quant) m->ring_pos[li][slot] = pos;
    memcpy(m->kv_c[li] + (size_t)slot * KL, cvec, sizeof(float) * KL);
    memcpy(m->kv_r[li] + (size_t)pos * RD, rvec, sizeof(float) * RD);
}

/* Attention over the cached latents for ONE position.
 *
 * qc is the absorbed query, [n_heads, kv_lora], already scaled (and, on a compressed
 * layer, already Hadamard-rotated); qr is the rope query, [n_heads, qk_rope], scaled.
 * The result is the latent-space weighted average, [n_heads, kv_lora] -- still in
 * whatever frame qc was in, so the caller unrotates before unfolding through W_v.
 *
 * Parallelised over CONTIGUOUS POSITION RANGES rather than over heads, which is the
 * opposite of what the softmax engines do, for two reasons. The latent is shared by
 * every head, so a head-parallel loop would decode each compressed position n_heads
 * times over. And there is only one of it, so a "one unit per KV head" split would be
 * one unit. Each partition runs its own online softmax and the results are combined
 * exactly: with per-partition (max, weight sum, unnormalised accumulator), the merge
 * is a second online softmax over the partitions. */
static void mla_attend(M *m, int li, const float *qc, const float *qr, float *out,
                       int pos, Buf *b) {
    Cfg *c = &m->c;
    int H = c->n_heads, KL = c->kv_lora, RD = c->qk_rope, cap = c->ctx;
    int quant = m->pc[li] != NULL;
    int Wr = quant ? (m->rwin < cap ? m->rwin : cap) : cap;
    int g = m->tile, ntile = kvarn_ntiles(cap, g);
    int n = pos + 1;
    int NP = b->npart;
    if (NP > n) NP = n;
    if (NP < 1) NP = 1;
    size_t PS = (size_t)H * (KL + 2);            /* acc[H*KL] | mx[H] | z[H] */

    #pragma omp parallel for schedule(static)
    for (int p = 0; p < NP; p++) {
        int t0 = (int)((int64_t)n * p / NP), t1 = (int)((int64_t)n * (p + 1) / NP);
        float *acc = b->apart + (size_t)p * PS;
        float *mx = acc + (size_t)H * KL, *z = mx + H;
        memset(acc, 0, sizeof(float) * (size_t)H * KL);
        for (int h = 0; h < H; h++) { mx[h] = -INFINITY; z[h] = 0.0f; }

        float cbuf[KVARN_MAXD];
        KvarnPlanes pl;
        pl.rec = NULL;
        for (int t = t0; t < t1; t++) {
            const float *cc;
            if (!quant) {
                cc = m->kv_c[li] + (size_t)(t % Wr) * KL;
            } else if (m->ring_pos[li][t % Wr] == t) {
                /* the ring's few positions join the rotated frame */
                memcpy(cbuf, m->kv_c[li] + (size_t)(t % Wr) * KL, sizeof(float) * KL);
                kvarn_rot(cbuf, KL);
                cc = cbuf;
            } else {
                kvarn_decode_raw(&m->qc[li],
                                 m->pc[li] + (size_t)((t / g) % ntile) * m->qc[li].bytes,
                                 t % g, &pl, cbuf);
                cc = cbuf;
            }
            /* The rope key is f32 and unrotated in both cases; it contributes to the
             * score only, never to the accumulator, so it never needs to share the
             * latent's frame. */
            const float *rr = m->kv_r[li] + (size_t)t * RD;
            for (int h = 0; h < H; h++) {
                float score = head_dot(qc + (size_t)h * KL, cc, KL)
                            + head_dot(qr + (size_t)h * RD, rr, RD);
                float *a = acc + (size_t)h * KL;
                if (score > mx[h]) {
                    float r = (mx[h] == -INFINITY) ? 0.0f : expf(mx[h] - score);
                    z[h] *= r;
                    for (int d = 0; d < KL; d++) a[d] *= r;
                    mx[h] = score;
                }
                float w = expf(score - mx[h]);
                z[h] += w;
                for (int d = 0; d < KL; d++) a[d] += w * cc[d];
            }
        }
    }

    /* merge the partitions: a second online softmax, over NP terms */
    for (int h = 0; h < H; h++) {
        float M = -INFINITY;
        for (int p = 0; p < NP; p++) {
            float v = b->apart[(size_t)p * PS + (size_t)H * KL + h];
            if (v > M) M = v;
        }
        float *o = out + (size_t)h * KL;
        memset(o, 0, sizeof(float) * KL);
        double Z = 0;
        if (M == -INFINITY) continue;            /* cannot happen: n >= 1 */
        for (int p = 0; p < NP; p++) {
            const float *acc = b->apart + (size_t)p * PS;
            float mxp = acc[(size_t)H * KL + h], zp = acc[(size_t)H * KL + H + h];
            if (zp == 0.0f) continue;
            float w = expf(mxp - M);
            Z += (double)zp * w;
            const float *a = acc + (size_t)h * KL;
            for (int d = 0; d < KL; d++) o[d] += w * a[d];
        }
        float inv = Z > 0 ? (float)(1.0 / Z) : 0.0f;
        for (int d = 0; d < KL; d++) o[d] *= inv;
    }
}

/* Multi-head Latent Attention, absorbed.
 *
 *   q  = q_b_proj(rmsnorm(q_a_proj(x)))            [H, qk_nope + qk_rope]
 *   c  = rmsnorm(kv_a_proj(x)[:kv_lora])           [kv_lora]     cached
 *   kr = rope(kv_a_proj(x)[kv_lora:])              [qk_rope]     cached, shared
 *
 * The textbook form would expand c through kv_b_proj into per-head keys and values and
 * attend on those. Instead the converter split kv_b_proj into W_k and W_v per head and
 * transposed W_k, and the two halves fold into the ends of the attention:
 *
 *   score_t = (W_k[h]^T q_nope[h]) . c_t  +  q_rot[h] . kr_t
 *   out[h]  = W_v[h] (sum_t a_t c_t)
 *
 * Both folds cost H*qk_nope*kv_lora, i.e. one kv_b_proj's worth, per token. What they
 * buy is that the CACHE is the latent: 576 floats a position instead of 5120, and the
 * per-cached-position work is a 576-wide dot instead of a 512x256 expansion.
 *
 * The head-wise gate is a single sigmoid per head applied to the attention output,
 * which is why g_proj is [n_heads, hidden] and stays f32. */
static void mla_fwd(M *m, int li, const float *Xn, float *OUT, int S,
                    int pos_base, Buf *b) {
    Cfg *c = &m->c;
    Layer *L = &m->L[li];
    int H = c->n_heads;
    int KL = c->kv_lora, RD = c->qk_rope, NOPE = c->qk_nope, VH = c->v_head;
    int QH = NOPE + RD;
    int quant = m->pc[li] != NULL;
    float scale = 1.0f / sqrtf((float)QH);

    /* the two LoRA legs and the gate, batched over the window */
    matmul(b->qa, &L->q_a_proj, Xn, S, b->mxq, b->msx);
    for (int s = 0; s < S; s++)
        rmsnorm(b->qa + (size_t)s * c->q_lora, b->qa + (size_t)s * c->q_lora,
                L->q_a_norm.f, c->q_lora, c->eps);
    matmul(b->qb, &L->q_b_proj, b->qa, S, b->mxq, b->msx);
    matmul(b->kva, &L->kv_a_proj, Xn, S, b->mxq, b->msx);
    matmul(b->mgate, &L->g_proj, Xn, S, b->mxq, b->msx);

    int64_t ktrb = fmt_row_bytes(L->kv_b_kt.fmt, NOPE);
    int64_t vrb  = fmt_row_bytes(L->kv_b_v.fmt, KL);

    for (int s = 0; s < S; s++) {
        int pos = pos_base + s;
        float *q = b->qb + (size_t)s * H * QH;
        float *kva = b->kva + (size_t)s * (KL + RD);

        /* normalise the latent in place, then rope the shared key */
        rmsnorm(kva, kva, L->kv_a_norm.f, KL, c->eps);
        rope_interleave(kva + KL, 1, RD, RD, pos, m->inv_freq);
        /* rope the rotary half of every query head, in place */
        rope_interleave(q + NOPE, H, QH, RD, pos, m->inv_freq);

        kv_write(m, li, pos, kva, kva + KL);

        /* Fold W_k^T into the query. kv_b_kt is [H*kv_lora, qk_nope] with the head's
         * kv_lora rows contiguous, so this is one flat loop over H*kv_lora output rows,
         * each reading its own head's q_nope -- one OpenMP region rather than H. */
#ifndef COLI_F32ACT
        int ktq = quantisable(L->kv_b_kt.fmt);
        if (ktq)
            for (int h = 0; h < H; h++)
                q40_quant_act(q + (size_t)h * QH, b->xq + (size_t)h * NOPE,
                              b->sx + (size_t)h * (NOPE / Q40_BLK), NOPE);
#endif
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < H * KL; i++) {
            int h = i / KL;
#ifdef COLI_F32ACT
            b->qc[i] = wdot_f32(L->kv_b_kt.fmt, L->kv_b_kt.q + (size_t)i * ktrb,
                                q + (size_t)h * QH, NOPE);
#else
            b->qc[i] = ktq
                ? wdot(L->kv_b_kt.fmt, L->kv_b_kt.q + (size_t)i * ktrb,
                       b->xq + (size_t)h * NOPE,
                       b->sx + (size_t)h * (NOPE / Q40_BLK), NOPE)
                : wdot_f32(L->kv_b_kt.fmt, L->kv_b_kt.q + (size_t)i * ktrb,
                           q + (size_t)h * QH, NOPE);
#endif
        }
        /* fold the attention scale into the query once rather than into every score */
        for (int i = 0; i < H * KL; i++) b->qc[i] *= scale;
        for (int h = 0; h < H; h++)
            for (int d = 0; d < RD; d++) q[(size_t)h * QH + NOPE + d] *= scale;
        /* Meet the cache in its own frame: on a compressed layer every latent the
         * attention reads is Hadamard-rotated, so rotating the query once here and the
         * output once below replaces a transform per cached position. */
        if (quant)
            for (int h = 0; h < H; h++) kvarn_rot(b->qc + (size_t)h * KL, KL);

        /* The rope query is the tail of each QH-wide head, so it is not contiguous
         * per head; mla_attend wants [n_heads, qk_rope]. */
        float *qr = b->mqr;
        for (int h = 0; h < H; h++)
            memcpy(qr + (size_t)h * RD, q + (size_t)h * QH + NOPE, sizeof(float) * RD);
        float *acc = b->qc + (size_t)H * KL;    /* the second half of the qc buffer */
        mla_attend(m, li, b->qc, qr, acc, pos, b);
        if (quant)
            for (int h = 0; h < H; h++) kvarn_rot(acc + (size_t)h * KL, KL);

        /* Unfold through W_v, then gate. Same flat-loop trick: kv_b_v is [H*v_head,
         * kv_lora] with the head's rows contiguous. */
        float *o = b->mo + (size_t)s * H * VH;
#ifndef COLI_F32ACT
        int vq = quantisable(L->kv_b_v.fmt);
        if (vq)
            for (int h = 0; h < H; h++)
                q40_quant_act(acc + (size_t)h * KL, b->hq + (size_t)h * KL,
                              b->hs + (size_t)h * (KL / Q40_BLK), KL);
#endif
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < H * VH; i++) {
            int h = i / VH;
#ifdef COLI_F32ACT
            o[i] = wdot_f32(L->kv_b_v.fmt, L->kv_b_v.q + (size_t)i * vrb,
                            acc + (size_t)h * KL, KL);
#else
            o[i] = vq
                ? wdot(L->kv_b_v.fmt, L->kv_b_v.q + (size_t)i * vrb,
                       b->hq + (size_t)h * KL, b->hs + (size_t)h * (KL / Q40_BLK), KL)
                : wdot_f32(L->kv_b_v.fmt, L->kv_b_v.q + (size_t)i * vrb,
                           acc + (size_t)h * KL, KL);
#endif
        }
        const float *gl = b->mgate + (size_t)s * H;
        for (int h = 0; h < H; h++) {
            float gv = sigmoidf_(gl[h]);
            for (int d = 0; d < VH; d++) o[(size_t)h * VH + d] *= gv;
        }
    }
    matmul(OUT, &L->dense, b->mo, S, b->mxq, b->msx);
}

/* Kimi Delta Attention.
 *
 * A gated delta rule with a DIAGONAL decay -- one decay per key channel rather than
 * the single scalar of Gated DeltaNet -- carried in a per-head state S[v][k]. For each
 * position, with q,k L2-normalised and beta = sigmoid(b_proj(x)):
 *
 *     S  <- S * diag(exp(g))
 *     u   = beta * (v - S k)
 *     S  <- S + u k^T
 *     o   = S q
 *
 * and the decay itself is the bounded form fla uses when a lower_bound is supplied:
 *
 *     g = lower_bound * sigmoid(exp(A_log[h]) * (f_proj(x) + dt_bias))
 *
 * so g lies in (lower_bound, 0) by construction. This is NOT the -exp(A)*softplus(f)
 * that the same kernel uses without a bound, and the checkpoint sets kda_lower_bound
 * -5, so getting the branch wrong gives a plausible-looking model that decays wrong.
 *
 * Everything up to the recurrence is batched; the recurrence is sequential in position
 * and parallel over heads, which is the right way round because the heads own disjoint
 * 128x128 states and the positions do not. Two passes over each state per position:
 * one to decay and form S k, one to update and read out S q. */
static void kda_fwd(M *m, int li, const float *Xn, float *OUT, int S, Buf *b) {
    Cfg *c = &m->c;
    Layer *L = &m->L[li];
    int H = c->n_heads, HD = c->head_dim, P = H * HD, CL = c->conv_L;
    float LB = c->kda_lower_bound;
    float qscale = 1.0f / sqrtf((float)HD);
    float *st = m->kda_state[li];
    float *cs = m->conv_state[li];                 /* [3][(CL-1)*P] */
    size_t cstride = (size_t)(CL - 1) * P;

    matmul(b->kq, &L->kq_proj, Xn, S, b->mxq, b->msx);
    matmul(b->kk, &L->kk_proj, Xn, S, b->mxq, b->msx);
    matmul(b->kv, &L->kv_proj, Xn, S, b->mxq, b->msx);
    matmul(b->kf, &L->f_proj, Xn, S, b->mxq, b->msx);
    matmul(b->kg, &L->kg_proj, Xn, S, b->mxq, b->msx);
    matmul(b->kbeta, &L->b_proj, Xn, S, b->mxq, b->msx);   /* [S, H] */

    /* Short causal convolutions, depthwise with kernel conv_L, then silu. `state`
     * holds the previous conv_L-1 inputs per channel, oldest first, so tap j pairs
     * with state[j] and the last tap with the current value. Positions must be fed in
     * order: this is a recurrence, not a cache, and it cannot be rewound. */
    {
        float *sig[3] = {b->kq, b->kk, b->kv};
        const W *cw[3] = {&L->q_conv, &L->k_conv, &L->v_conv};
        for (int u = 0; u < 3; u++) {
            float *stu = cs + (size_t)u * cstride;
            const float *w = cw[u]->f;
            for (int s = 0; s < S; s++) {
                float *row = sig[u] + (size_t)s * P;
                for (int i = 0; i < P; i++) {
                    float cur = row[i];
                    const float *wi = w + (size_t)i * CL;
                    float acc = wi[CL - 1] * cur;
                    for (int j = 0; j < CL - 1; j++) acc += wi[j] * stu[(size_t)j * P + i];
                    row[i] = silu(acc);
                    for (int j = 0; j + 2 < CL; j++)
                        stu[(size_t)j * P + i] = stu[(size_t)(j + 1) * P + i];
                    if (CL > 1) stu[(size_t)(CL - 2) * P + i] = cur;
                }
            }
        }
    }

    for (int s = 0; s < S; s++) {
        const float *qr = b->kq + (size_t)s * P, *kr = b->kk + (size_t)s * P;
        const float *vr = b->kv + (size_t)s * P, *fr = b->kf + (size_t)s * P;
        const float *gr = b->kg + (size_t)s * P;
        const float *br = b->kbeta + (size_t)s * H;
        float *orow = b->ko + (size_t)s * P;

        #pragma omp parallel for schedule(static)
        for (int h = 0; h < H; h++) {
            float qh[512], kh[512], eg[512], oh[512];
            float *Sh = st + (size_t)h * HD * HD;
            const float *dtb = L->dt_bias.f + (size_t)h * HD;
            float A = expf(L->A_log.f[h]);
            float beta = sigmoidf_(br[h]);

            /* L2-normalise q and k, with the kernel's own epsilon INSIDE the root */
            double nq = 0, nk = 0;
            for (int d = 0; d < HD; d++) {
                float a = qr[(size_t)h * HD + d], bb = kr[(size_t)h * HD + d];
                nq += (double)a * a; nk += (double)bb * bb;
            }
            float rq = 1.0f / sqrtf((float)nq + 1e-6f);
            float rk = 1.0f / sqrtf((float)nk + 1e-6f);
            for (int d = 0; d < HD; d++) {
                qh[d] = qr[(size_t)h * HD + d] * rq * qscale;
                kh[d] = kr[(size_t)h * HD + d] * rk;
                eg[d] = expf(LB * sigmoidf_(A * (fr[(size_t)h * HD + d] + dtb[d])));
            }

            /* Decay, form S k, apply the delta rule, read out S q -- one row at a time.
             * The textbook form is two sweeps of the state, the first producing all of
             * S k and the second consuming it, but nothing crosses rows: S k's entry
             * for row v uses only row v, u[v] only that entry, and the rank-1 update
             * to row v only u[v]. Fused, the 512-byte row is read and written once
             * instead of twice, and the state is 1 MiB a layer across the 18 KDA
             * layers -- the largest resident read in a decode step after the experts.
             * Same operations in the same order, so --check does not move. */
            for (int vdim = 0; vdim < HD; vdim++) {
                float *row = Sh + (size_t)vdim * HD;
                float sk = 0;
                for (int d = 0; d < HD; d++) { row[d] *= eg[d]; sk += row[d] * kh[d]; }
                float uu = beta * (vr[(size_t)h * HD + vdim] - sk), acc = 0;
                for (int d = 0; d < HD; d++) { row[d] += uu * kh[d]; acc += row[d] * qh[d]; }
                oh[vdim] = acc;
            }

            /* FusedRMSNormGated with activation sigmoid: normalise, scale by the
             * weight, THEN gate. Gating before the norm would be a different model. */
            rmsnorm(oh, oh, L->o_norm.f, HD, c->eps);
            for (int d = 0; d < HD; d++)
                orow[(size_t)h * HD + d] = oh[d] * sigmoidf_(gr[(size_t)h * HD + d]);
        }
    }
    matmul(OUT, &L->o_proj, b->ko, S, b->mxq, b->msx);
}

/* layer
 *
 *     h = h + attention(input_layernorm(h))          attention = MLA | KDA
 *     h = h + ffn(post_attention_layernorm(h))       ffn = dense SwiGLU | MoE + shared
 */
static void layer_fwd(M *m, int li, float *H, int S, int pos_base, Buf *b) {
    Cfg *c = &m->c;
    Layer *L = &m->L[li];
    int D = c->hidden;

    float *Xn = b->xn;
    for (int s = 0; s < S; s++)
        rmsnorm(Xn + (size_t)s * D, H + (size_t)s * D, L->in_norm.f, D, c->eps);
    if (c->layer_types[li]) mla_fwd(m, li, Xn, b->tmp, S, pos_base, b);
    else                    kda_fwd(m, li, Xn, b->tmp, S, b);
    for (int i = 0; i < S * D; i++) H[i] += b->tmp[i];

    float *X = b->eout;
    for (int s = 0; s < S; s++)
        rmsnorm(X + (size_t)s * D, H + (size_t)s * D, L->post_norm.f, D, c->eps);

    if (li < c->n_dense_layers) {
        int DI = c->dense_inter;
        matmul(b->gate, &L->mlp_gate, X, S, b->mxq, b->msx);
        matmul(b->up, &L->mlp_up, X, S, b->mxq, b->msx);
        for (int i = 0; i < S * DI; i++) b->mlp[i] = silu(b->gate[i]) * b->up[i];
        matmul(b->tmp, &L->mlp_down, b->mlp, S, b->mxq, b->msx);
        for (int i = 0; i < S * D; i++) H[i] += b->tmp[i];
    } else {
        /* The shared expert sits between the submit and the apply on purpose. It reads
         * the same X the router did, it is resident, and it is the only real arithmetic
         * this architecture offers to overlap the routed reads with -- neither lfm25
         * nor maple has a dense branch here to do the same job. */
        moe_start(m, li, X, b->h2, S, pos_base, b);
        shared_fwd(m, li, X, b->h2, S, b);
        moe_finish(m, li, X, b->h2, S, b);
        for (int i = 0; i < S * D; i++) H[i] += b->h2[i];
    }
}

/* embed one token row. Ling uses a plain nn.Embedding: no embed_scale, and the
 * embedding table is NOT the lm_head. */
static void embed_row(M *m, int tok, float *h) {
    int D = m->c.hidden;
    const W *e = &m->L[MAXL - 1].kq_proj;
    const uint8_t *row = e->q + (size_t)tok * fmt_row_bytes(e->fmt, D);
    if (e->fmt == FMT_Q40)      q40_dequant_row(row, h, D);
    else if (e->fmt == FMT_Q80) q80_dequant_row(row, h, D);
    else memcpy(h, (const float *)row, sizeof(float) * D);
}

/* FlashHead
 *
 * Two phases. Score the `n_clusters` quantised centroids against the final hidden,
 * take the best `n_probes`, and compute exact lm_head logits only for the tokens those
 * clusters contain (plus a fixed set of forced control tokens such as EOS). Every
 * other logit is -inf, so greedy decoding is exact whenever the true argmax lies in a
 * probed cluster, and sampling is exact on the same condition.
 *
 * The win is bandwidth, not flops. The head is 157184 x 1536, which at q8_0 is 245 MiB
 * read per decode step for one matvec -- more than the eight routed experts of that
 * step put together. Probing 530 of 4912 clusters reads the 12 MiB of centroids plus
 * 16960 rows, about 39 MiB.
 *
 * Unlike gemma4's and lfm25's, this clustering sees the real lm_head rather than a
 * tied embedding table: this checkpoint does not tie them. tools/flashhead.py builds
 * it at conversion time. */
static int flash_partition(float *v, int *idx, int n, int k);

static void flash_logits(M *m, const float *hn, float *logits, Buf *b) {
    Cfg *c = &m->c;
    int NC = c->flash_n_clusters, CS = c->flash_cluster_size, NP = c->flash_n_probes;
    int V = c->vocab, D = c->hidden;
    const W *e = &m->L[MAXL - 1].kk_proj;             /* the real lm_head */
    int64_t rb = fmt_row_bytes(e->fmt, D);

    matvec(b->fsim, &m->flash_centroids, hn, b->xq, b->sx);
    /* The converter folds the per-cluster scale into the centroid rows; the flag exists
     * because a container that has not can still be read. The scale matters: centroids
     * are directions, and scaling by the largest member norm keeps a cluster holding
     * one big row from being ranked on cosine alone. */
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
        for (int j = 0; j < CS; j++) b->fids[n++] = row[j];   /* -1 = padding slot */
    }
    if (c->flash_n_force) {
        const int32_t *fo = (const int32_t *)m->flash_force.q;
        for (int j = 0; j < c->flash_n_force; j++) b->fids[n++] = fo[j];
    }

    for (int i = 0; i < V; i++) logits[i] = -INFINITY;
#ifdef COLI_F32ACT
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        int id = b->fids[i];
        if (id >= 0 && id < V) logits[id] = wdot_f32(e->fmt, e->q + (size_t)id * rb, hn, D);
    }
#else
    q40_quant_act(hn, b->xq, b->sx, D);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        int id = b->fids[i];
        if (id >= 0 && id < V)
            logits[id] = wdot(e->fmt, e->q + (size_t)id * rb, b->xq, b->sx, D);
    }
#endif
}

/* Move the k largest of v[] to the front of idx[] (order within the k is not defined).
 * Quickselect on the index array: at ~5000 clusters and k ~ 530 a full sort would cost
 * more than the centroid matvec it is selecting from. */
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

/* Zero both recurrences. The MLA cache needs no reset: it is position-addressed and a
 * position is only ever read after it has been written for this sequence. */
static void rec_reset(M *m) {
    Cfg *c = &m->c;
    size_t ns = (size_t)c->n_heads * c->head_dim * c->head_dim;
    size_t nc = 3 * (size_t)(c->conv_L - 1) * c->n_heads * c->head_dim;
    for (int l = 0; l < c->n_layers; l++) {
        if (m->kda_state[l]) memset(m->kda_state[l], 0, sizeof(float) * ns);
        if (m->conv_state[l]) memset(m->conv_state[l], 0, sizeof(float) * (nc ? nc : 1));
    }
    m->rec_pos = 0;
}

/* Run S tokens from pos_base. logits may be NULL (prefill) or [S,vocab]; the common
 * case, only the last row, goes through `last_only`. */
static void forward_chunk(M *m, const int *ids, int S, int pos_base,
                          float *logits, int last_only, Buf *b) {
    Cfg *c = &m->c;
    int D = c->hidden;
    W *head = &m->L[MAXL - 1].kk_proj;
    W *fnorm = &m->L[MAXL - 1].o_proj;

    float *H = b->x;
    for (int s = 0; s < S; s++) embed_row(m, ids[s], H + (size_t)s * D);
    for (int l = 0; l < c->n_layers; l++) layer_fwd(m, l, H, S, pos_base, b);
    m->rec_pos = pos_base + S;

    if (!logits) return;
    int s0 = last_only ? S - 1 : 0;
    /* The lm_head is by far the widest tensor here, so batch it too when every row's
     * logits are wanted (--check). */
    if (!last_only && S > 1) {
        for (int s = 0; s < S; s++)
            rmsnorm(b->mlp + (size_t)s * D, H + (size_t)s * D, fnorm->f, D, c->eps);
        matmul(logits, head, b->mlp, S, b->mxq, b->msx);
        return;
    }
    for (int s = s0; s < S; s++) {
        rmsnorm(b->mlp, H + (size_t)s * D, fnorm->f, D, c->eps);
        float *out = logits + (size_t)(last_only ? 0 : s) * c->vocab;
        if (m->use_flash) {
            flash_logits(m, b->mlp, out, b);
            if (m->flash_check) {
                /* Same row, same hidden, both heads: the probed logits must be
                 * bit-identical (only the candidate SET is approximate, not the
                 * arithmetic), and the argmax agrees whenever the exact argmax fell in
                 * a probed cluster. */
                matvec(m->flash_exact, head, b->mlp, b->xq, b->sx);
                int ea = 0, fa = 0;
                for (int i = 1; i < c->vocab; i++) {
                    if (m->flash_exact[i] > m->flash_exact[ea]) ea = i;
                    if (out[i] > out[fa]) fa = i;
                }
                for (int i = 0; i < c->vocab; i++) {
                    if (!isfinite(out[i])) continue;
                    m->flash_probed++;
                    double d = fabs((double)out[i] - m->flash_exact[i]);
                    if (d > m->flash_worst) m->flash_worst = d;
                }
                m->flash_steps++;
                m->flash_agree += (ea == fa);
            }
        } else {
            matvec(out, head, b->mlp, b->xq, b->sx);
        }
    }
}

/* As forward_chunk, but for an input of any length.
 *
 * Long inputs are processed in chunks of b->S. That is exactly equivalent to one big
 * call -- positions still go through every layer in order, both recurrences carry
 * across the boundary and the MLA cache accumulates -- and it bounds the batched
 * projection scratch, which would otherwise scale with the full context.
 *
 * pos_base == 0 starts a new sequence and resets the recurrences. Any other pos_base
 * must equal rec_pos: the KDA state and the short-conv state cannot be rewound, so a
 * caller reusing a cached prefix has to be strictly extending it. Getting that wrong
 * does not crash, it silently corrupts the state. */
static void forward(M *m, const int *ids, int S, int pos_base,
                    float *logits, int last_only, Buf *b) {
    if (pos_base == 0) rec_reset(m);
    else if (pos_base != m->rec_pos) {
        fprintf(stderr, "internal error: forward at %d but the recurrence is at %d\n",
                pos_base, m->rec_pos);
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
        if (!m->esz[l]) continue;            /* the dense layer holds no experts */
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
                 int kb, int rwin, int tile) {
    m->tile = tile > 0 ? tile : 128;
    /* KVARN_RWIN is a floor, not the ring size: KVarN seals whole tiles, so the ring
     * is rounded up to a whole number of them (kvarn_window). */
    m->rwin = kvarn_window(rwin > 0 ? rwin : 128, m->tile);
    manifest(m, dir);
    Cfg *c = &m->c;

    /* --ctx overrides what the container was converted with, and may go up: the
     * container's ctx only fixed slots_per_layer, and the weights do not care. */
    int ctx_planned = c->ctx;
    if (ctx_override > 0) c->ctx = ctx_override;
    m->ucount = calloc((size_t)c->n_layers * c->n_experts, 8);
    snprintf(m->usage_path, sizeof m->usage_path, "%s/usage.bin", dir);

    m->inv_freq = xmalloc(sizeof(float) * (size_t)(c->qk_rope / 2 + 1));
    for (int i = 0; i < c->qk_rope / 2; i++)
        m->inv_freq[i] = powf(c->rope_theta, -(float)(2 * i) / (float)c->qk_rope);

    m->kv_c = calloc(c->n_layers, sizeof(float *));
    m->kv_r = calloc(c->n_layers, sizeof(float *));
    m->ring_pos = calloc(c->n_layers, sizeof(int *));
    m->pc = calloc(c->n_layers, sizeof(uint8_t *));
    m->qc = calloc(c->n_layers, sizeof(Kvarn));
    m->kda_state = calloc(c->n_layers, sizeof(float *));
    m->conv_state = calloc(c->n_layers, sizeof(float *));

    size_t kvb = 0, rcb = 0;
    int quant = kb > 0;
    int KL = c->kv_lora, RD = c->qk_rope, cap = c->ctx;
    for (int l = 0; l < c->n_layers; l++) {
        if (!c->layer_types[l]) {                    /* KDA layer: state, no KV */
            size_t ns = (size_t)c->n_heads * c->head_dim * c->head_dim;
            size_t nc = 3 * (size_t)(c->conv_L - 1) * c->n_heads * c->head_dim;
            m->kda_state[l] = xmalloc(sizeof(float) * ns);
            m->conv_state[l] = xmalloc(sizeof(float) * (nc ? nc : 1));
            memset(m->kda_state[l], 0, sizeof(float) * ns);
            memset(m->conv_state[l], 0, sizeof(float) * (nc ? nc : 1));
            rcb += sizeof(float) * (ns + (nc ? nc : 1));
            continue;
        }
        int Wr = quant ? (m->rwin < cap ? m->rwin : cap) : cap;
        m->kv_c[l] = xmalloc(sizeof(float) * (size_t)Wr * KL);
        m->kv_r[l] = xmalloc(sizeof(float) * (size_t)cap * RD);
        m->ring_pos[l] = xmalloc(sizeof(int) * (size_t)Wr);
        for (int i = 0; i < Wr; i++) m->ring_pos[l][i] = -1;
        kvb += sizeof(float) * ((size_t)Wr * KL + (size_t)cap * RD);

        if (quant) {
            /* One codec, not two. In absorbed form the latent is simultaneously the
             * key (it is what the folded query dots against) and the value (it is what
             * the attention weights average), so it gets the key budget -- the more
             * accurate of the two -- rather than being split down the middle. */
            kvarn_init(&m->qc[l], KL, kb, m->tile, 1);
            size_t nt = (size_t)kvarn_ntiles(cap, m->tile);
            m->pc[l] = xmalloc(m->qc[l].bytes * nt);
            kvb += m->qc[l].bytes * nt;
        }
    }
    int n_mla = 0;
    for (int l = 0; l < c->n_layers; l++) n_mla += c->layer_types[l];
    fprintf(stderr, "kv: %.0f MiB for ctx %d over %d MLA layers", kvb / 1048576.0,
            c->ctx, n_mla);
    if (quant) fprintf(stderr, " (KVarN K%d on the latent, tile %d, f32 ring %d)",
                       kb, m->tile, m->rwin);
    else       fprintf(stderr, " (f32, by --kv off)");
    fprintf(stderr, ";  kda state: %.1f MiB (context-independent)\n", rcb / 1048576.0);

    /* Compare against what the conversion budgeted (an f32 cache at the container's own
     * ctx). Only an actual overshoot warrants a warning -- --kv routinely buys a longer
     * context for fewer bytes than the plan assumed. */
    {
        size_t planned = (size_t)n_mla * (size_t)ctx_planned * (KL + RD) * sizeof(float);
        if (kvb > planned && ram_gb <= 0)
            fprintf(stderr, "warning: that is %.0f MiB more KV than the container's "
                    "plan budgeted (%.0f MiB for ctx %d, f32), so total RAM will "
                    "exceed the conversion's --ram%s\n",
                    (kvb - planned) / 1048576.0, planned / 1048576.0, ctx_planned,
                    quant ? "" : "; dropping --kv off would cut it a lot");
    }

    /* expert-cache plan
     * slots_per_layer is the only thing the conversion's --ram fixed, and nothing in
     * the container depends on it: experts are read one at a time and the cache is a
     * plain LRU. So a runtime --ram re-runs the planner of tools/convert_ling.py
     * against figures now known -- the resident dense blob and the cache just
     * allocated. */
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
                          - (int64_t)m->dense_len - (int64_t)kvb - (int64_t)rcb
                          - scratch;
            int64_t per = avail > 0 ? (avail / esz_max) / nmoe : 0;
            if (per > c->n_experts) per = c->n_experts;
            if (per < c->topk) {
                double min_gb = ((double)m->dense_len + (double)kvb + (double)rcb
                                 + (double)scratch
                                 + (double)c->topk * nmoe * esz_max) / (double)(1LL << 30);
                fprintf(stderr, "--ram %g GB leaves room for %lld experts per layer, "
                        "below topk=%d: this model needs %.2f GB at this context%s\n",
                        ram_gb, (long long)per, c->topk, min_gb,
                        quant ? "" : " (dropping --kv off would lower it)");
                exit(1);
            }
            c->slots_per_layer = (int)per;
        }
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
                s->buf = m->esz[l] ? xmalloc(m->esz[l]) : NULL;
                /* Map each slot once, up front: the streaming layer overwrites the
                 * bytes but never the address, so the mapping stays valid all run. */
                if (g_use_gpu && s->buf && !gpu_map(s->buf, (size_t)m->esz[l])) {
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

static Buf *bufs(M *m, int Smax) {
    Cfg *c = &m->c;
    int D = c->hidden, K = c->topk, H = c->n_heads;
    int P = H * c->head_dim;
    int KL = c->kv_lora, RD = c->qk_rope, QH = c->qk_nope + RD, VH = c->v_head;
    int imax = c->dense_inter;
    if (imax < c->moe_inter) imax = c->moe_inter;
    if (imax < c->shared_inter) imax = c->shared_inter;
    /* widest output any matvec writes, and widest dimension any of them contracts
     * over; xq/sx and mxq/msx are sized by them */
    int wide = c->vocab;
    int dims[] = {D, P, imax, H * QH, H * KL, KL + RD, c->q_lora, H * VH};
    for (unsigned i = 0; i < sizeof dims / sizeof *dims; i++)
        if (dims[i] > wide) wide = dims[i];
    int cmax = D;
    int cdims[] = {imax, P, KL, c->q_lora, c->qk_nope, H * VH};
    for (unsigned i = 0; i < sizeof cdims / sizeof *cdims; i++)
        if (cdims[i] > cmax) cmax = cdims[i];

    Buf *b = calloc(1, sizeof *b);
    b->S = Smax;
    b->x    = xmalloc(sizeof(float) * (size_t)Smax * D);
    b->eout = xmalloc(sizeof(float) * (size_t)Smax * D);   /* ffn-normed input */
    b->h2   = xmalloc(sizeof(float) * (size_t)Smax * D);   /* MoE + shared output */
    b->xn   = xmalloc(sizeof(float) * ((size_t)Smax * D + wide));
    b->tmp  = xmalloc(sizeof(float) * ((size_t)Smax * D + wide));
    b->mxq  = xmalloc((size_t)Smax * cmax + 64);
    b->msx  = xmalloc(sizeof(float) * ((size_t)Smax * (cmax / Q40_BLK) + 64));

    b->kq    = xmalloc(sizeof(float) * ((size_t)Smax * P + 64));
    b->kk    = xmalloc(sizeof(float) * ((size_t)Smax * P + 64));
    b->kv    = xmalloc(sizeof(float) * ((size_t)Smax * P + 64));
    b->kf    = xmalloc(sizeof(float) * ((size_t)Smax * P + 64));
    b->kg    = xmalloc(sizeof(float) * ((size_t)Smax * P + 64));
    b->ko    = xmalloc(sizeof(float) * ((size_t)Smax * P + 64));
    b->kbeta = xmalloc(sizeof(float) * ((size_t)Smax * H + 64));

    b->qa    = xmalloc(sizeof(float) * ((size_t)Smax * c->q_lora + 64));
    b->qb    = xmalloc(sizeof(float) * ((size_t)Smax * H * QH + 64));
    b->kva   = xmalloc(sizeof(float) * ((size_t)Smax * (KL + RD) + 64));
    b->mgate = xmalloc(sizeof(float) * ((size_t)Smax * H + 64));
    b->mo    = xmalloc(sizeof(float) * ((size_t)Smax * H * VH + 64));
    /* qc holds the folded query AND, in its second half, the attention output: both
     * are [n_heads, kv_lora] and their lifetimes do not overlap past mla_attend. */
    b->qc    = xmalloc(sizeof(float) * (2 * (size_t)H * KL + 64));
    b->mqr   = xmalloc(sizeof(float) * ((size_t)H * RD + 64));
#ifdef _OPENMP
    b->npart = omp_get_max_threads();
#else
    b->npart = 1;
#endif
    if (b->npart < 1) b->npart = 1;
    if (b->npart > MAXPART) b->npart = MAXPART;
    b->apart = xmalloc(sizeof(float) * (size_t)b->npart * H * (KL + 2));

    b->gate = xmalloc(sizeof(float) * ((size_t)Smax * imax + 64));
    b->up   = xmalloc(sizeof(float) * ((size_t)Smax * imax + 64));
    b->mlp  = xmalloc(sizeof(float) * ((size_t)Smax * imax + wide + 64));
    b->rwt  = xmalloc(sizeof(float) * (c->n_experts + 64));
    b->rsel = xmalloc(sizeof(float) * (c->n_experts + 64));
    /* xq/hq also hold the per-head quantised q_nope and latent of mla_fwd, so they are
     * sized for H copies of the widest of those alongside the plain matvec use */
    size_t qact = (size_t)H * (KL > c->qk_nope ? KL : c->qk_nope);
    if (qact < (size_t)wide) qact = (size_t)wide;
    b->xq   = xmalloc(qact + 64);
    b->hq   = xmalloc(qact + 64);
    b->sx   = xmalloc(sizeof(float) * (qact / Q40_BLK + 64));
    b->hs   = xmalloc(sizeof(float) * (qact / Q40_BLK + 64));
    b->eidx = xmalloc(sizeof(int) * (size_t)Smax * K);
    b->ewt  = xmalloc(sizeof(float) * (size_t)Smax * K);
    b->rows = xmalloc(sizeof(int) * (size_t)Smax * K);
    b->roww = xmalloc(sizeof(float) * (size_t)Smax * K);
    b->uniq = xmalloc(sizeof(int) * c->n_experts);
    if (g_use_gpu) {
        b->gx = xmalloc(sizeof(float) * ((size_t)Smax * D + 64));
        b->gg = xmalloc(sizeof(float) * ((size_t)Smax * c->moe_inter + 64));
        b->gu = xmalloc(sizeof(float) * ((size_t)Smax * c->moe_inter + 64));
        b->gd = xmalloc(sizeof(float) * ((size_t)Smax * D + 64));
    }
    b->egate = xmalloc(sizeof(float) * (size_t)Smax * (c->moe_inter + 64));
    b->exq  = xmalloc((size_t)Smax * (D + 64));
    b->ehq  = xmalloc((size_t)Smax * (c->moe_inter + 64));
    b->esx  = xmalloc(sizeof(float) * (size_t)Smax * (D / Q40_BLK + 8));
    b->ehs  = xmalloc(sizeof(float) * (size_t)Smax * (c->moe_inter / Q40_BLK + 8));
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
 * only once no matter how often it occurs. Applied to the fresh logits before
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

/* temperature + top-k + nucleus. Greedy when temp <= 0. top_k is applied before top_p,
 * which is the order HF uses. Ling's own generation defaults are temp 1.0 / top_p 0.95
 * / top_k 20 (generation_config.json). */
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

/* Bailing V3's chat format, transcribed from chat_template.jinja. Roles are spelt as
 * literal <role>NAME</role> text -- they are not special tokens -- and every turn ends
 * with <|role_end|>, which is also the EOS.
 *
 * The system turn always exists, because the thinking switch lives in it: the template
 * appends "detailed thinking on" or "off" unless the caller's own system message
 * already says one of them. --nothink also pre-closes the think block in the
 * generation prompt, which is what actually stops the model reasoning; the system line
 * alone only tells it which mode it is in. */
static void chat_system(char *out, size_t cap, size_t *n, const char *sys, int think) {
    #define ADD(...) *n += snprintf(out + *n, *n < cap ? cap - *n : 0, __VA_ARGS__)
    ADD("<role>SYSTEM</role>");
    if (sys && *sys) {
        if (strstr(sys, "detailed thinking on") || strstr(sys, "detailed thinking off"))
            ADD("%s<|role_end|>", sys);
        else
            ADD("%s\ndetailed thinking %s<|role_end|>", sys, think ? "on" : "off");
    } else {
        ADD("detailed thinking %s<|role_end|>", think ? "on" : "off");
    }
    #undef ADD
}

static void chat_prompt(char *out, size_t cap, const char *sys,
                        const char *user, int think) {
    size_t n = 0;
    chat_system(out, cap, &n, sys, think);
    #define ADD(...) n += snprintf(out + n, n < cap ? cap - n : 0, __VA_ARGS__)
    ADD("<role>HUMAN</role>%s<|role_end|>", user ? user : "");
    ADD("<role>ASSISTANT</role>\n<think>");
    if (!think) ADD("</think>");
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
    int eos, eot;
    int think;
    const char *model_id;
} LingServerContext;

typedef struct { char *data; size_t len, cap; } LingString;

static int ling_string_append(LingString *s, const char *data, size_t len) {
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

static int ling_json_escape(LingString *s, const char *text, size_t len) {
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == '"' || c == '\\') { char x[2] = {'\\', (char)c}; if (!ling_string_append(s,x,2)) return 0; }
        else if (c == '\n') { if (!ling_string_append(s,"\\n",2)) return 0; }
        else if (c == '\r') { if (!ling_string_append(s,"\\r",2)) return 0; }
        else if (c == '\t') { if (!ling_string_append(s,"\\t",2)) return 0; }
        else if (c < 0x20) { char x[6] = {'\\','u','0','0',hex[c>>4],hex[c&15]}; if (!ling_string_append(s,x,6)) return 0; }
        else if (!ling_string_append(s,(const char *)&text[i],1)) return 0;
    }
    return 1;
}

/* The generation prompt leaves <think> open, so a thinking completion begins *inside*
 * the reasoning block and only ever emits the closing tag. Split there: what precedes
 * </think> is the turn's reasoning_content, what follows is the content. The tag is
 * several tokens long and can straddle a decode boundary, so any tail of the stream
 * that is still a proper prefix of it is held back until the next piece decides. */
#define LING_THINK_CLOSE "</think>"
typedef struct { int in_think; char hold[sizeof LING_THINK_CLOSE - 1]; size_t nhold; } LingThink;

static int ling_think_feed(LingThink *s, const char *piece, size_t len,
                           LingString *reasoning, LingString *content) {
    if (!s->in_think) return ling_string_append(content, piece, len);
    static const size_t tag = sizeof LING_THINK_CLOSE - 1;
    char buf[4096 + sizeof LING_THINK_CLOSE];
    if (len > sizeof buf - s->nhold - 1) len = sizeof buf - s->nhold - 1;  /* cannot happen */
    memcpy(buf, s->hold, s->nhold);
    memcpy(buf + s->nhold, piece, len);
    size_t m = s->nhold + len;
    buf[m] = 0;
    s->nhold = 0;
    char *cut = memmem(buf, m, LING_THINK_CLOSE, tag);
    if (cut) {
        s->in_think = 0;
        return ling_string_append(reasoning, buf, (size_t)(cut - buf)) &&
               ling_string_append(content, cut + tag, m - (size_t)(cut - buf) - tag);
    }
    size_t h = m < tag - 1 ? m : tag - 1;            /* longest held-back tag prefix */
    while (h > 0 && memcmp(buf + m - h, LING_THINK_CLOSE, h) != 0) h--;
    memcpy(s->hold, buf + m - h, h); s->nhold = h;
    return ling_string_append(reasoning, buf, m - h);
}

/* End of generation: whatever is still held back was never a tag. */
static int ling_think_flush(LingThink *s, LingString *reasoning, LingString *content) {
    size_t h = s->nhold; s->nhold = 0;
    return ling_string_append(s->in_think ? reasoning : content, s->hold, h);
}

static jval *ling_json_field(jval *object, const char *key, jtype type) {
    jval *v = json_get(object, key); return v && v->t == type ? v : NULL;
}

static int ling_build_chat_prompt(jval *messages, LingString *prompt, int think) {
    #define PUT(s) do { const char *_s = (s); \
                        if (!ling_string_append(prompt, _s, strlen(_s))) return 0; } while (0)
    if (!messages || messages->t != J_ARR) return 0;
    /* The system turn is emitted first and once, from messages[0] if it is a system
     * message, because that is where the thinking switch has to go. */
    const char *sys = NULL;
    int first_is_system = 0;
    if (messages->len > 0) {
        jval *role = ling_json_field(messages->kids[0], "role", J_STR);
        jval *content = ling_json_field(messages->kids[0], "content", J_STR);
        if (role && content && !strcmp(role->str, "system")) {
            sys = content->str; first_is_system = 1;
        }
    }
    {
        char head[8192]; size_t n = 0;
        chat_system(head, sizeof head, &n, sys, think);
        PUT(head);
    }
    for (int i = first_is_system ? 1 : 0; i < messages->len; i++) {
        jval *message = messages->kids[i];
        jval *role = ling_json_field(message, "role", J_STR);
        jval *content = ling_json_field(message, "content", J_STR);
        if (!role || !content) continue;
        if (!strcmp(role->str, "user"))        PUT("<role>HUMAN</role>");
        else if (!strcmp(role->str, "system")) PUT("<role>SYSTEM</role>");
        else if (!strcmp(role->str, "assistant")) {
            /* preserved_thinking: an earlier turn's reasoning is replayed if the client
             * sends it back, either as reasoning_content or still inline in content. */
            jval *rc = ling_json_field(message, "reasoning_content", J_STR);
            const char *text = content->str;
            PUT("<role>ASSISTANT</role>\n<think>");
            if (rc && *rc->str) PUT(rc->str);
            else {
                const char *close = strstr(text, LING_THINK_CLOSE);
                if (close) {
                    const char *open = strstr(text, "<think>");
                    const char *begin = open && open < close ? open + 7 : text;
                    if (!ling_string_append(prompt, begin, (size_t)(close - begin))) return 0;
                    text = close + sizeof LING_THINK_CLOSE - 1;
                }
            }
            PUT("</think>");
            PUT(text);
            PUT("<|role_end|>");
            continue;
        }
        else continue;
        PUT(content->str);
        PUT("<|role_end|>");
    }
    PUT("<role>ASSISTANT</role>\n<think>");
    if (!think) PUT("</think>");
    #undef PUT
    return 1;
}

static int ling_send_chunk(int fd, const char *id, const char *field, const char *text, size_t len) {
    LingString out = {0};
    const char *prefix = "data: {\"id\":\"";
    const char *middle = "\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"";
    const char *suffix = "\":\"";
    const char *end = "\"},\"finish_reason\":null}]}\n\n";
    int ok = ling_string_append(&out,prefix,strlen(prefix)) && ling_json_escape(&out,id,strlen(id)) &&
        ling_string_append(&out,middle,strlen(middle)) && ling_string_append(&out,field,strlen(field)) &&
        ling_string_append(&out,suffix,strlen(suffix)) && ling_json_escape(&out,text,len) &&
        ling_string_append(&out,end,strlen(end));
    if (ok) ok = samosa_send_all(fd, out.data, out.len);
    free(out.data); return ok;
}

static int ling_send_done(int fd, const char *id, int prompt_tokens, int completion_tokens,
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

static int ling_serve_chat(LingServerContext *ctx, int fd, jval *root) {
    jval *messages = ling_json_field(root, "messages", J_ARR);
    int has_user = 0;
    if (!messages) return samosa_http_json_error(fd,400,"invalid_messages","messages must be an array.");
    for (int i = 0; i < messages->len; i++) {
        jval *msg = messages->kids[i];
        jval *role = ling_json_field(msg,"role",J_STR);
        jval *content = ling_json_field(msg,"content",J_STR);
        if (role && content && !strcmp(role->str,"user")) has_user = 1;
    }
    if (!has_user) return samosa_http_json_error(fd,400,"invalid_messages","A text user message is required.");

    int stream = 0, max_tokens = 2048, topk = 20, seed = 0, think = ctx->think;
    float temperature = 1.0f, topp = 0.95f, penalty = 1.0f;   /* Ling generation defaults */
    jval *v = json_get(root,"stream"); if (v && v->t == J_BOOL) stream = v->boolean;
    /* the switch the template calls enable_thinking, spelt the way the API does */
    v = json_get(root,"enable_thinking"); if (v && v->t == J_BOOL) think = v->boolean;
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
    LingString prompt = {0};
    if (!ling_build_chat_prompt(messages, &prompt, think)) {
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
    /* Prefix reuse is restricted here. The KDA state and the short-conv state only move
     * forwards, so the cached state is reusable only when this prompt strictly extends
     * what was absorbed. A diverging prefix -- or even an identical prompt, which would
     * need position np-1 replayed -- is reprocessed from scratch. */
    int common = 0;
    while (common < ctx->cached_len && common < np && ctx->cached_ids[common] == ids[common]) common++;
    if (common > 0 && common == ctx->cached_len && common == m->rec_pos && common < np)
        forward(m, ids + common, np - common, common, logits, 1, ctx->buffers);
    else
        forward(m, ids, np, 0, logits, 1, ctx->buffers);

    char id[64]; snprintf(id,sizeof id,"ling-%llu",(unsigned long long)time(NULL));
    if (stream && !samosa_http_stream_headers(fd)) { pthread_mutex_unlock(&ctx->generation_mu); free(ids); free(logits); free(pbuf); free(seen); return 1; }
    LingString answer = {0}, reasoning = {0};
    /* think means the prompt ended on an open <think>, so generation starts in it. */
    LingThink split = { .in_think = think };
    uint64_t rng = seed ? (uint64_t)seed : 0x853c49e6748fea9bULL;
    int generated = 0; const char *reason = "length";
    while (generated < max_tokens && !atomic_load(&ctx->cancel)) {
        repetition_penalty(logits, c->vocab, ids, np + generated, penalty, seen);
        int token = sample(logits, c->vocab, temperature, topp, topk, pbuf, &rng);
        if (token == ctx->eos || token == ctx->eot) { reason = "stop"; break; }
        char piece[4096]; int n = lfmtok_decode(tok, &token, 1, piece, sizeof piece - 1);
        if (n <= 0) { reason = "stop"; break; }
        size_t was_r = reasoning.len, was_c = answer.len;
        if (!ling_think_feed(&split, piece, (size_t)n, &reasoning, &answer)) { atomic_store(&ctx->cancel,1); break; }
        if (stream &&
            ((reasoning.len > was_r && !ling_send_chunk(fd,id,"reasoning_content",reasoning.data+was_r,reasoning.len-was_r)) ||
             (answer.len > was_c && !ling_send_chunk(fd,id,"content",answer.data+was_c,answer.len-was_c))))
            { atomic_store(&ctx->cancel,1); break; }
        ids[np + generated++] = token;
        if (generated < max_tokens) forward(m, &token, 1, np + generated - 1, logits, 1, ctx->buffers);
    }
    {   /* a held-back tag prefix at the end of the stream was never a tag */
        size_t was_r = reasoning.len, was_c = answer.len;
        ling_think_flush(&split, &reasoning, &answer);
        if (stream) {
            if (reasoning.len > was_r) ling_send_chunk(fd,id,"reasoning_content",reasoning.data+was_r,reasoning.len-was_r);
            if (answer.len > was_c) ling_send_chunk(fd,id,"content",answer.data+was_c,answer.len-was_c);
        }
    }
    if (atomic_load(&ctx->cancel)) reason = "cancelled";
    if (stream) ling_send_done(fd,id,np,generated,reason);
    else {
        LingString body={0}; char prefix[512], middle[64], suffix[512];
        int n=snprintf(prefix,sizeof prefix,"{\"id\":\"%s\",\"object\":\"chat.completion\",\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"reasoning_content\":\"",id,ctx->model_id);
        int ok=n>0&&ling_string_append(&body,prefix,(size_t)n)&&ling_json_escape(&body,reasoning.data?reasoning.data:"",reasoning.len);
        n=snprintf(middle,sizeof middle,"\",\"content\":\"");
        ok=ok&&n>0&&ling_string_append(&body,middle,(size_t)n)&&ling_json_escape(&body,answer.data?answer.data:"",answer.len);
        n=snprintf(suffix,sizeof suffix,"\"},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",reason,np,generated,np+generated);
        ok=ok&&n>0&&ling_string_append(&body,suffix,(size_t)n)&&samosa_http_headers(fd,200,"application/json",body.len,NULL)&&samosa_send_all(fd,body.data,body.len);
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
    free(answer.data); free(reasoning.data); pthread_mutex_unlock(&ctx->generation_mu); free(ids); free(logits); free(pbuf); free(seen); return 0;
}

static int ling_serve_handler(SamosaHttpServer *server, int fd, const SamosaHttpRequest *request, void *opaque) {
    LingServerContext *ctx = opaque;
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
        int result=ling_serve_chat(ctx,fd,root); json_free(root); free(arena); return result;
    }
    if (!strcmp(request->method,"POST") && !strcmp(request->path,"/v1/shutdown")) {
        atomic_store(&ctx->cancel,1); samosa_http_response(fd,200,"application/json","{\"shutting_down\":true}",NULL); samosa_http_server_stop(server); return 1;
    }
    return samosa_http_json_error(fd,404,"not_found","Endpoint not found.");
}

static int run_ling_server(M *m, Buf *buffers, LfmTok *tokenizer, const char *model_id,
                           int port, int think) {
    LingServerContext ctx={.model=m,.buffers=buffers,.tokenizer=tokenizer,
                           .model_id=model_id,.think=think};
    ctx.eos = lfmtok_id(tokenizer, "<|role_end|>");
    ctx.eot = lfmtok_id(tokenizer, "<|endoftext|>");
    pthread_mutex_init(&ctx.generation_mu,NULL); atomic_init(&ctx.cancel,0);
    SamosaHttpServer server;
    if (!samosa_http_server_init(&server,port,ling_serve_handler,&ctx)) { fprintf(stderr,"server: cannot bind port %d: %s\n",port,strerror(errno)); pthread_mutex_destroy(&ctx.generation_mu); return 1; }
    fprintf(stderr,"[server] OpenAI endpoint ready at http://127.0.0.1:%d\n",server.port); fflush(stderr);
    int ok=samosa_http_server_run(&server); samosa_http_server_destroy(&server);
    free(ctx.cached_ids); pthread_mutex_destroy(&ctx.generation_mu); return ok?0:1;
}

/* main */
/* Numerically diff the Metal kernels against the CPU reference on random data, in both
 * formats. The Metal path cannot be tested where it was written -- it is the user's
 * machine that decides whether it is right. Shapes cover the real ones. */
static int check_gpu(void) {
    if (!gpu_ready()) {
        printf("no Metal device (or built without COLI_METAL) -- nothing to check\n");
        return 0;
    }
    printf("metal device: %s\n\n", gpu_name());
    struct { int fmt, O, I; const char *what; } shp[] = {
        {FMT_Q80, 2048, 1536, "kda q/k/v/f/g_proj (q8_0)"},
        {FMT_Q80, 1536, 2048, "kda o_proj (q8_0)"},
        {FMT_Q80, 3072,  256, "mla q_b_proj (q8_0)"},
        {FMT_Q80,  576, 1536, "mla kv_a_proj (q8_0)"},
        {FMT_Q80, 8192,  128, "mla kv_b_kt (q8_0)"},
        {FMT_Q80, 2048,  512, "mla kv_b_v (q8_0)"},
        {FMT_Q80, 4608, 1536, "dense mlp gate/up (q8_0)"},
        {FMT_Q80, 1536, 4608, "dense mlp down (q8_0)"},
        {FMT_Q80,  512, 1536, "shared expert gate/up (q8_0)"},
        {FMT_Q40,  512, 1536, "expert gate/up (q4_0)"},
        {FMT_Q40, 1536,  512, "expert down (q4_0)"},
    };
    int fail = 0;
    for (unsigned t = 0; t < sizeof shp / sizeof *shp; t++) {
        int fmt = shp[t].fmt, O = shp[t].O, I = shp[t].I;
        size_t rb = (size_t)fmt_row_bytes(fmt, I), wb = (size_t)O * rb;
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
        if (!gpu_map(W, wb)) { printf("  %-28s gpu_map FAILED\n", shp[t].what); fail = 1; goto next; }
        if (!gpu_matmul(fmt, yg, W, x, O, I, 1)) {
            printf("  %-28s gpu_matmul DECLINED\n", shp[t].what); fail = 1; goto next;
        }
        for (int o = 0; o < O; o++) yc[o] = wdot_f32(fmt, W + (size_t)o * rb, x, I);

        double worst = 0, mag = 0;
        for (int o = 0; o < O; o++) {
            double d = fabs(yg[o] - yc[o]);
            if (d > worst) worst = d;
            if (fabs(yc[o]) > mag) mag = fabs(yc[o]);
        }
        double rel = worst / (mag + 1e-9);
        printf("  %-28s [%5d x %5d]  max rel err %.3e  %s\n",
               shp[t].what, O, I, rel, rel < 1e-4 ? "ok" : "MISMATCH");
        if (rel >= 1e-4) fail = 1;
    next:
        free(W); free(x); free(yg); free(yc);
    }
    printf("\n%s\n", fail ? "GPU CHECK FAILED -- do not pass --metal" : "GPU CHECK PASSED");
    return fail;
}

static void usage(const char *prog, FILE *out) {
    fprintf(out,
        "usage: %s <dir> [flags...] [prompt]\n"
        "         [--chat] [--system S] [--nothink] [--raw] [--max_tokens N]\n"
        "         [--temp F] [--topp F] [--topk N]   (default 1.0 / 0.95 / 20)\n"
        "         [--penalty F]           repetition penalty (default 1, = off)\n"
        "         [--ctx N]               override the container's context length\n"
        "         [--ram F]               re-plan the expert cache for an F GB budget\n"
        "         [--pin N] [--io N] [--threads N] [--batch N] [--nobatch]\n"
        "         [--serve] [--port N]    OpenAI-compatible local server (default 8484)\n"
        "         [--kv PRESET]           KVarN compression of the MLA latent cache;\n"
        "                                 PRESET is one of off | kvarn_k4v2_g128 |\n"
        "                                 kvarn_k4v4_g128 | kvarn_k4v2_g64 |\n"
        "                                 kvarn_k4v4_g64 (default kvarn_k4v2_g128;\n"
        "                                 only the K width is used, see init)\n"
        "         [--metal]               offload the matmuls to the GPU (off by\n"
        "                                 default: usually slower here, see matvec)\n"
        "         [--flash]               approximate lm_head: score clustered\n"
        "                                 centroids, compute only the top clusters\n"
        "         [--probes N]            FlashHead clusters probed per token\n"
        "         [--flash-check]         also run the exact head, report agreement\n"
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
    /* Thinking is ON by default: that is what the template does when the caller says
     * nothing (thinking_option defaults to 'on'). */
    int think = 1, raw = 0, chat_mode = 0;
    /* KVarN is ON by default, at upstream's shipped preset, and a preset is all there
     * is: no per-parameter overrides, because the bit widths and the tile are one
     * calibrated recipe upstream measured together. --kv off gives an f32 cache. */
    const KvarnPreset *kvp = kvarn_preset(KVARN_DEFAULT);
    int kv_set = 0;
    int check = 0, n_io = 8, max_tokens = 0, nobatch = 0, npin = 0, nthreads = 2;
    int batch = 128, ctx_override = 0, want_flash = 0, probes = 0, flash_check = 0;
    double ram_gb = 0;                   /* 0 = keep the container's own plan */
    int serve_mode = 0, serve_port = 8484;
    int want_metal = 0, chk_gpu = 0;
    float temp = 1.0f, topp = 0.95f, penalty = 1.0f;   /* Ling generation defaults */
    int topk = 20;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--serve")) serve_mode = 1;
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) serve_port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--check")) check = 1;
        else if (!strcmp(argv[i], "--flash")) want_flash = 1;
        else if (!strcmp(argv[i], "--flash-check")) { flash_check = 1; want_flash = 1; }
        else if (!strcmp(argv[i], "--probes") && i + 1 < argc) probes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--metal")) want_metal = 1;
        else if (!strcmp(argv[i], "--check-gpu")) chk_gpu = 1;
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
        /* Any leading dash, not just a double one. A single-dash typo (-kv for --kv)
         * must not fall through to the positional branch: it would silently become the
         * prompt and change what the model generates. */
        else if (argv[i][0] == '-') {
            fprintf(stderr, "unknown flag: %s\n\n", argv[i]);
            usage(argv[0], stderr);
            return 1;
        }
        else if (!prompt) prompt = argv[i];   /* first non-flag positional is the prompt */
    }
    /* --check diffs the forward pass against a stored oracle to ~1e-4, which is tighter
     * than any KV quantiser reproduces, so it defaults to an f32 cache whatever the
     * engine default is. Pass --kv explicitly to measure the codec instead. */
    if (check && !kv_set) kvp = NULL;
    /* kb == 0 is how the rest of the engine spells "f32 cache". */
    int kb = kvp ? kvp->kbits : 0;
    if (penalty < 0.5f || penalty > 2.0f) { fprintf(stderr, "--penalty must be in 0.5..2\n\n"); usage(argv[0], stderr); return 1; }
    if (chat_mode && check) { fprintf(stderr, "--chat cannot be used with --check\n\n"); usage(argv[0], stderr); return 1; }
    if (chat_mode && serve_mode) { fprintf(stderr, "--chat cannot be used with --serve\n\n"); usage(argv[0], stderr); return 1; }
    if (!serve_mode && !check && max_tokens == 0) max_tokens = 2048;
#ifdef _OPENMP
    if (nthreads > 0) omp_set_num_threads(nthreads);
#else
    (void)nthreads;
#endif

    if (want_metal || chk_gpu) g_use_gpu = gpu_init();
    if (chk_gpu) return check_gpu();
    if (want_metal && !g_use_gpu)
        fprintf(stderr, "metal: no device available; using CPU\n");

    M m; memset(&m, 0, sizeof m);
    double t0 = now();
    init(&m, dir, n_io, ctx_override, ram_gb, kb, KVARN_RWIN, kvp ? kvp->group : 128);
    Cfg *c = &m.c;
    pin_load(&m, npin);
    /* Prefill batch size: bigger reuses each streamed weight row over more rows, but
     * the projection scratch grows with it, so past a point it evicts the weights it is
     * trying to reuse. */
    if (batch < 1) batch = 1;
    if (batch > c->ctx) batch = c->ctx;
    if (probes > 0 && c->flash_n_clusters) {
        if (probes > c->flash_n_clusters) probes = c->flash_n_clusters;
        c->flash_n_probes = probes;
    }
    /* Opt-in, and off under --check, which diffs against the oracle's exact logits. */
    m.use_flash = c->flash_n_clusters && want_flash && !check;
    if (want_flash && !c->flash_n_clusters)
        fprintf(stderr, "--flash: this container has no FlashHead (rebuild it "
                "without --no-flash); using the exact head\n");
    if (m.use_flash && flash_check) {
        m.flash_check = 1;
        m.flash_exact = xmalloc(sizeof(float) * (size_t)c->vocab);
    }
    Buf *b = bufs(&m, batch);

    int n_mla = 0;
    for (int l = 0; l < c->n_layers; l++) n_mla += c->layer_types[l];
    fprintf(stderr, "ling: %d layers (%d MLA, %d KDA), %d experts + 1 shared, top-%d "
            "of %d groups, %d slots/layer, dense %.1f MiB, ready in %.2fs\n",
            c->n_layers, n_mla, c->n_layers - n_mla, c->n_experts, c->topk,
            c->topk_group, c->slots_per_layer, m.dense_len / 1048576.0, now() - t0);
    if (g_use_gpu)
        fprintf(stderr, "metal: %s (q4_0/q8_0 matmul offloaded)\n", gpu_name());
    if (m.use_flash)
        fprintf(stderr, "flash head: %d/%d clusters x %d probed (%.1f%% of the "
                "vocabulary scored)\n", c->flash_n_probes, c->flash_n_clusters,
                c->flash_cluster_size,
                100.0 * c->flash_n_probes * c->flash_cluster_size / c->vocab);

    if (serve_mode) {
        char tp[4096]; snprintf(tp, sizeof tp, "%s/tok.bin", dir);
        LfmTok *server_tokenizer = lfmtok_load(tp);
        if (!server_tokenizer) { fprintf(stderr, "--serve needs %s\n", tp); return 1; }
        return run_ling_server(&m, b, server_tokenizer, "ling-3.0-tiny", serve_port,
                               think);
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
         * fp32 reference would conflate engine bugs with the quantiser's own error. */
        snprintf(p, sizeof p, "%s/deq_logits.f32", dir);
        f = fopen(p, "rb");
        if (!f) { perror(p); fprintf(stderr, "run tools/ling_oracle.py first\n"); return 1; }
        float *ref = xmalloc(sizeof(float) * (size_t)np * c->vocab);
        if (fread(ref, sizeof(float), (size_t)np * c->vocab, f) != (size_t)np * c->vocab) {
            fprintf(stderr, "short reference logits\n"); return 1;
        }
        fclose(f);

        /* The oracle's routing decisions, if it wrote them. Optional, so a container
         * converted before deq_route.i32 existed still checks -- just without the
         * routing diagnostic and with the arithmetic and the routing conflated. */
        int K = c->topk;
        int *refroute = NULL;
        size_t nroute = (size_t)c->n_layers * np * K;
        snprintf(p, sizeof p, "%s/deq_route.i32", dir);
        f = fopen(p, "rb");
        if (f) {
            refroute = xmalloc(sizeof(int) * nroute);
            if (fread(refroute, sizeof(int), nroute, f) != nroute) {
                free(refroute); refroute = NULL;
            }
            fclose(f);
        }
        m.chk_stride = np;                       /* the oracle's row stride, not ctx */
        if (refroute) m.chk_route = xmalloc(sizeof(int) * nroute);

        float *logits = xmalloc(sizeof(float) * (size_t)np * c->vocab);
        double t = now(), el = 0;

        #define RUN() do {                                                            \
            if (m.chk_route)                                                          \
                for (size_t _i = 0; _i < nroute; _i++) m.chk_route[_i] = -1;           \
            if (nobatch)                                                              \
                for (int _s = 0; _s < np; _s++)                                       \
                    forward(&m, ids + _s, 1, _s, logits + (size_t)_s * c->vocab, 0, b); \
            else                                                                      \
                forward(&m, ids, np, 0, logits, 0, b);                                \
        } while (0)

        /* Pass 1: the engine routes for itself. This is the one that is timed, and the
         * one whose picks are compared. */
        RUN();
        el = now() - t;

        int rdiff = 0, rtot = 0;
        if (refroute)
            for (size_t i = 0; i < nroute; i++) {
                if (refroute[i] < 0 && m.chk_route[i] < 0) continue;   /* dense layer */
                rtot++;
                if (refroute[i] != m.chk_route[i]) rdiff++;
            }

        /* Pass 2, only when they disagreed: rerun with the oracle's picks pinned, so
         * the logit distance below measures the arithmetic and nothing else. The
         * routing rule itself is not let off -- rdiff is asserted on the exact build,
         * where the engine sees the very numbers the oracle did. */
        int pinned = 0;
        if (rdiff) {
            m.pin_route = refroute;
            RUN();
            m.pin_route = NULL;
            pinned = 1;
        }
        #undef RUN

        double worst = 0, den = 0;
        int agree = 0;
        int defensible MAYBE_UNUSED = 0;    /* only the int8 build asserts on this */
        for (int s = 0; s < np; s++) {
            const float *g = logits + (size_t)s * c->vocab;
            const float *r = ref + (size_t)s * c->vocab;
            int am = 0, ar = 0;
            double rw = 0;
            for (int i = 0; i < c->vocab; i++) {
                double d = fabs(g[i] - r[i]);
                if (d > rw) rw = d;
                if (fabs(r[i]) > den) den = fabs(r[i]);
                if (g[i] > g[am]) am = i;
                if (r[i] > r[ar]) ar = i;
            }
            if (rw > worst) worst = rw;
            agree += (am == ar);
            /* Exact argmax agreement is not something the int8 build can promise on a
             * fixture whose 256 random logits sit within a band narrower than the
             * quantisation error. What it CAN promise, and what actually matters, is
             * that it never picks a token the oracle ranked meaningfully lower: the
             * chosen token's reference logit must be within the row's own error of the
             * reference maximum. On a decided row that forces am == ar; on an
             * undecided one it accepts either answer and nothing else. */
            defensible += (r[am] >= r[ar] - 2.0 * rw);
        }

        long long tot = m.hit + m.miss;
        printf("%s prefill of %d tokens in %.3fs\n",
               nobatch ? "sequential" : "batch-union", np, el);
        if (refroute)
            printf("routing: %d/%d expert picks match the oracle\n", rtot - rdiff, rtot);
        printf("teacher forcing: %d/%d argmax agree%s\n", agree, np,
               pinned ? " (routing pinned to the oracle's)" : "");
        printf("max |logit diff| = %.4g   (ref scale %.4g, rel %.3g)%s\n",
               worst, den, worst / den, pinned ? "   [routing pinned]" : "");
        printf("expert reads: %lld (%lld hits, %lld misses)\n",
               tot, (long long)m.hit, (long long)m.miss);

        /* With exact activations the engine must reproduce the oracle to float
         * precision, INCLUDING every routing decision: it is looking at the same
         * numbers, so anything else is a bug.
         *
         * The int8 tolerance is looser here than in the other three engines, and the
         * reason is depth times chain length rather than anything this engine does
         * badly. Every layer runs five or six chained int8 matmuls (KDA: q, k, v, f, g,
         * o; MLA: q_a, q_b, kv_a, the two absorbed folds, dense), against two or four
         * in lfm25, and the fixture is 64 wide, so one block of 32 carries a whole
         * half of every row and amax/rms is ~3 on random data. Measured on the fixture,
         * the relative logit error grows smoothly with depth:
         *
         *     layers   1     2     3     4     5     6     7     8
         *     rel    .019  .026  .035  .041  .052  .057  .060  .082
         *
         * -- no step anywhere, which is what says it is accumulation and not a bug.
         * 1.2e-1 leaves room for that curve and still catches anything that doubles
         * it. The claim that stays tight is the exact build's 1e-4. */
#ifdef COLI_F32ACT
        double tol = 1e-4;
        int ok = (agree == np) && (worst / den < tol);
#else
        double tol = 1.2e-1;
        int ok = (defensible == np) && (worst / den < tol);
        if (agree != np)
            printf("argmax: %d/%d exact, %d/%d within the row's own error\n",
                   agree, np, defensible, np);
#endif
        if (rdiff) {
#ifdef COLI_F32ACT
            printf("  FAIL: the routing must be identical on exact activations\n");
            ok = 0;
#else
            printf("  note: %d pick(s) differ under int8 activations; grouped top-k is "
                   "discrete, so the figures above were remeasured with the oracle's "
                   "picks pinned\n", rdiff);
#endif
        }



        /* FlashHead, on the same rows. Two different claims: a probed logit must be
         * bit-identical to the exact head (the head itself is not approximated, only
         * the candidate set is), which is a hard failure; and the argmax agrees
         * whenever the exact argmax landed in a probed cluster, which is the
         * approximation and is reported rather than enforced -- on a random fixture the
         * clusters carry no structure, so any threshold here would be a threshold on
         * noise. */
        if (c->flash_n_clusters) {
            float *fl = xmalloc(sizeof(float) * c->vocab);
            m.flash_exact = xmalloc(sizeof(float) * (size_t)c->vocab);
            m.use_flash = m.flash_check = 1;
            for (int s = 0; s < np; s++)
                forward(&m, ids + s, 1, s, fl, 0, b);
            m.use_flash = m.flash_check = 0;
            printf("flash head: %lld/%lld argmax agree with the exact head on the "
                   "same rows, %.1f%% of the vocabulary probed, max |probed - exact| "
                   "= %.3g\n", m.flash_agree, m.flash_steps,
                   100.0 * m.flash_probed / ((double)m.flash_steps * c->vocab),
                   m.flash_worst);
            if (m.flash_worst != 0.0) { printf("  probed logits are not exact\n"); ok = 0; }
            free(fl); free(m.flash_exact); m.flash_exact = NULL;
        }
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
        int eos = T ? lfmtok_id(T, "<|role_end|>") : -1;
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
            if (getenv("LINGDBG")) fprintf(stderr, "prompt: %s\n[%d tokens]\n", text, np);
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

                size_t need = hist_len + len + 512;
                char *chat = xmalloc(need);
                size_t pos = 0;
                if (hist_len) { memcpy(chat, history, hist_len); pos = hist_len; }
                if (!pos) chat_system(chat, need, &pos, sys, think);
                pos += snprintf(chat + pos, need - pos,
                                "<role>HUMAN</role>%s<|role_end|>", user_text);
                pos += snprintf(chat + pos, need - pos, "<role>ASSISTANT</role>\n<think>");
                if (!think) pos += snprintf(chat + pos, need - pos, "</think>");

                np = lfmtok_encode(T, chat, ids, c->ctx - max_tokens);
                if (getenv("LINGDBG")) fprintf(stderr, "prompt: %s\n[%d tokens]\n", chat, np);
                if (np <= 0) { free(chat); continue; }

                if (!hist_cap) { hist_cap = 4096; history = xmalloc(hist_cap); }
                while (hist_cap < pos + 512) { hist_cap *= 2; history = realloc(history, hist_cap); }
                memcpy(history, chat, pos);
                hist_len = pos;
                free(chat);

                /* Same restriction as the server: reuse the recurrences only when this
                 * prompt strictly extends what has already been absorbed. */
                int common = 0;
                while (common < cached_len && common < np && cached_ids[common] == ids[common])
                    common++;
                if (common > 0 && common == cached_len && common == m.rec_pos && common < np)
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

            /* The template's generation prompt ends on an open <think>, so the model
             * only ever emits the closing tag; echo the opener so the transcript is
             * balanced. --raw is the caller's own prompt, verbatim: nothing added. */
            if (T && think && (interactive || !raw)) fputs("<think>", stdout);

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

                const char *close = "<|role_end|>";
                size_t add = rlen + strlen(close);
                while (hist_cap < hist_len + add + 1) { hist_cap *= 2; history = realloc(history, hist_cap); }
                memcpy(history + hist_len, resp, rlen);
                hist_len += rlen;
                memcpy(history + hist_len, close, strlen(close));
                hist_len += strlen(close);
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
                if (m.flash_check && m.flash_steps)
                    printf("flash head  : %lld/%lld argmax agree with the exact head "
                           "(%.1f%%)\n", m.flash_agree, m.flash_steps,
                           100.0 * m.flash_agree / m.flash_steps);
                pin_save(&m);
            }

            if (!interactive) break;
        }
        free(cached_ids);
        free(history);
    }
    return 0;
}
