/* OLMoE inference engine in pure C, with EXPERT-STREAMING from disk.
 * Port of the Python engine (engine.py). Stage A goal: produce the SAME
 * token ids as the reference (ref.json) -> validate the core before scaling to GLM-5.2.
 *
 * Dense part (embed, attn, router, norms, lm_head) resident in RAM (float32).
 * Experts read from disk on-demand via pread+fadvise(DONTNEED), per-layer LRU cache.
 * Multi-threaded matmul with OpenMP (no BLAS).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <unistd.h>                    /* sysconf: physical RAM on Linux/BSD */
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>                /* sysctlbyname("hw.memsize") */
#endif
#include "st.h"

/* ---------- config ---------- */
typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, head_dim;
    int n_experts, topk, inter, vocab;
    float theta, eps; int norm_topk;
} Cfg;

/* ---------- per-layer dense weights ---------- */
typedef struct {
    float *in_ln, *post_ln, *q, *k, *v, *o, *qn, *kn, *gate;
} Layer;

/* ---------- expert LRU cache (QUANTIZED weights) ----------
 * Each weight [out,in] kept as int8 (per-row) + float scale per row.
 * This way the RAM cache drops from 4 bytes/param (f32) to 1 byte/param: it is the
 * mechanism that makes GLM-5.2 fit in 15 GB. dequant-on-use in the matmul. */
typedef struct { int eid; int8_t *g, *u, *d; float *gs, *us, *ds; uint64_t used; } Slot;
typedef struct { Slot *slots; int n, cap; } LCache;

typedef struct {
    Cfg c;
    shards S;
    int quant_bits;        /* expert quantization bits (2..8); int8 storage, no f32 (#134) */
    float *embed, *lm_head, *final_norm;
    Layer *L;
    LCache *cache;          /* [n_layers] */
    uint64_t clock, hits, miss;
    /* per-layer kv-cache: K,V as [H * maxT * head_dim] */
    float **K, **V; int kv_len, max_t;
    double dense_load_s, warm_s;
    int cap, resident;                 /* experts/layer kept in RAM; resident=1 -> all fit (no streaming) */
} Model;

/* ---------- utility ---------- */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
#if defined(__APPLE__)
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0*1024.0); }  /* macOS: byte */
#else
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0); }        /* Linux: KB */
#endif
static float *falloc(int64_t n) { float *p = malloc(n*sizeof(float)); if(!p){fprintf(stderr,"OOM %ld\n",(long)n);exit(1);} return p; }

/* y[S,O] = x[S,I] @ W^T,  W is [O,I] row-major */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float acc = 0.f;
            for (int i = 0; i < I; i++) acc += xs[i] * w[i];
            y[(int64_t)s * O + o] = acc;
        }
    }
}

/* y[1,O] = x[1,I] @ W^T with quantized W: q[O,I] int8 + scale per row.
 * W[o,i] ~= q[o,i]*scale[o]  ->  y[o] = scale[o] * sum_i x[i]*q[o,i].
 * On ARM: Q8_0-quantized activation (scale per block of 16) + int8 dot
 * NEON (sdot where dotprod is available) — same IDOT family as glm.c, IDOT=0 for
 * the byte-exact scalar path. Measured 2.7x end-to-end on M5. */
#if defined(__ARM_NEON)
#include <arm_neon.h>
static inline int32_t dot_i8_16(const int8_t *a, const int8_t *b) {
    int32x4_t acc = vdupq_n_s32(0);
    int8x16_t va = vld1q_s8(a), vb = vld1q_s8(b);
#if defined(__ARM_FEATURE_DOTPROD)
    acc = vdotq_s32(acc, va, vb);
#else
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va),  vget_low_s8(vb)));
    acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
#endif
    return vaddvq_s32(acc);
}
#endif
static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
#if defined(__ARM_NEON)
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = !(e && *e == '0'); }
    if (idot && I % 16 == 0 && I <= 4096) {
        int nb = I / 16; int8_t xi[4096]; float xs[256];
        for (int b = 0; b < nb; b++) {
            const float *xb = x + b*16;
            float am = 0.f; for (int i = 0; i < 16; i++) { float a = fabsf(xb[i]); if (a > am) am = a; }
            float s = am/127.f; if (s < 1e-12f) s = 1e-12f;
            xs[b] = s; float inv = 1.f/s;
            for (int i = 0; i < 16; i++) xi[b*16+i] = (int8_t)lrintf(xb[i]*inv);
        }
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            float acc = 0.f;
            for (int b = 0; b < nb; b++) acc += xs[b]*(float)dot_i8_16(xi+b*16, w+b*16);
            y[o] = acc * scale[o];
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
}

/* quantizes an f32 weight [O,I] -> int8 q[O,I] + scale[O], symmetric per row.
 * Replicates the Python quant_dequant(): scale = amax(|w|, row)/qmax, q = round(w/scale). */
static void quantize_rows(const float *w, int8_t *q, float *scale, int O, int I, int bits) {
    int qmax = (1 << (bits - 1)) - 1;     /* 8->127, 4->7, 2->1 */
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *wr = w + (int64_t)o * I;
        float amax = 0.f; for (int i = 0; i < I; i++) { float a = fabsf(wr[i]); if (a > amax) amax = a; }
        float s = amax / qmax; if (s < 1e-8f) s = 1e-8f;
        scale[o] = s;
        int8_t *qr = q + (int64_t)o * I;
        for (int i = 0; i < I; i++) {
            int v = (int)lrintf(wr[i] / s);
            if (v >  qmax) v =  qmax;
            if (v < -qmax-1) v = -qmax-1;
            qr[i] = (int8_t)v;
        }
    }
}

/* rmsnorm over a row of length D, in-place on out (out may be == x) */
static void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i]*x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * r * w[i];
}

static void softmax_row(float *x, int n) {
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ---------- loading ---------- */
static void load_cfg(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *r = json_parse(buf, &arena);
    c->hidden    = (int)json_get(r,"hidden_size")->num;
    c->n_layers  = (int)json_get(r,"num_hidden_layers")->num;
    c->n_heads   = (int)json_get(r,"num_attention_heads")->num;
    c->n_kv_heads= (int)json_get(r,"num_key_value_heads")->num;
    c->n_experts = (int)json_get(r,"num_experts")->num;
    c->topk      = (int)json_get(r,"num_experts_per_tok")->num;
    c->inter     = (int)json_get(r,"intermediate_size")->num;
    c->vocab     = (int)json_get(r,"vocab_size")->num;
    c->head_dim  = c->hidden / c->n_heads;
    jval *th = json_get(r,"rope_theta");  c->theta = th ? (float)th->num : 10000.f;
    jval *ep = json_get(r,"rms_norm_eps"); c->eps   = ep ? (float)ep->num : 1e-5f;
    jval *nt = json_get(r,"norm_topk_prob"); c->norm_topk = (nt && nt->t==J_BOOL) ? nt->boolean : 0;
    free(buf); free(arena);
}

static float *load_t(Model *m, const char *name) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing %s\n", name); exit(1); }
    float *p = falloc(n);
    st_read_f32(&m->S, name, p, 0);   /* dense: no DONTNEED, stays resident */
    return p;
}

/* physical RAM in bytes, 0 if it can't be determined. */
static uint64_t phys_ram_bytes(void) {
#if defined(__APPLE__)
    uint64_t mem = 0; size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, NULL, 0) == 0) return mem;
    return 0;
#elif defined(_SC_PHYS_PAGES)
    long pages = sysconf(_SC_PHYS_PAGES), psz = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && psz > 0) return (uint64_t)pages * (uint64_t)psz;
    return 0;
#else
    return 0;
#endif
}

/* RAM held by one cached expert: int8 g,u,d (1 byte/elem) + f32 per-row scales.
 * Must match the allocations in expert_get(). */
static int64_t expert_cache_bytes(const Cfg *c) {
    int64_t ng = (int64_t)c->inter * c->hidden, nd = (int64_t)c->hidden * c->inter;
    return ng + ng + nd + (int64_t)(c->inter + c->inter + c->hidden) * (int64_t)sizeof(float);
}

/* Residency planner: pick experts/layer to keep in RAM. If ALL experts fit the RAM
 * budget we keep them all (cap = n_experts) -> the whole small MoE stays resident and
 * NOTHING streams from disk, which is what makes it fast. Otherwise the largest cap
 * that fits (the tail still streams via LRU). COLI_RAM_GB overrides the auto budget;
 * default budget = 65% of physical RAM (leaves headroom for dense weights, KV, OS). */
static int plan_cap(const Cfg *c) {
    int64_t bpe = expert_cache_bytes(c);
    double budget_gb;
    const char *env = getenv("COLI_RAM_GB");
    if (env && atof(env) > 0) budget_gb = atof(env);
    else { uint64_t ph = phys_ram_bytes(); budget_gb = ph ? (double)ph / 1e9 * 0.65 : 24.0; }
    int64_t per_layer = (int64_t)c->n_layers * bpe;
    int cap = per_layer > 0 ? (int)((int64_t)(budget_gb * 1e9) / per_layer) : c->n_experts;
    if (cap > c->n_experts) cap = c->n_experts;
    if (cap < 1) cap = 1;
    return cap;
}

static void expert_get(Model *m, int layer, int eid, Slot **out);  /* fwd: used by prewarm */

/* Pre-load every expert into the resident cache so decode never pays a cold miss.
 * Only meaningful (and only called) when the plan is fully resident (cap == n_experts). */
static void prewarm_experts(Model *m) {
    Cfg *c = &m->c;
    double t0 = now_s();
    for (int l = 0; l < c->n_layers; l++)
        for (int e = 0; e < c->n_experts; e++) { Slot *s; expert_get(m, l, e, &s); }
    m->hits = m->miss = m->clock = 0;   /* warmup loads are not decode traffic */
    m->warm_s = now_s() - t0;
}

static void model_init(Model *m, const char *snap, int cap, int bits) {
    memset(m, 0, sizeof(*m));
    m->quant_bits = bits;
    load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    Cfg *c = &m->c;
    if (cap <= 0) cap = plan_cap(c);    /* cap<=0 -> auto-plan from RAM budget */
    m->cap = cap; m->resident = (cap >= c->n_experts);
    double t0 = now_s();
    m->embed      = load_t(m, "model.embed_tokens.weight");
    m->lm_head    = load_t(m, "lm_head.weight");
    m->final_norm = load_t(m, "model.norm.weight");
    m->L = calloc(c->n_layers, sizeof(Layer));
    char nm[256];
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        #define LD(field, suffix) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_t(m,nm)
        LD(in_ln,  "input_layernorm.weight");
        LD(post_ln,"post_attention_layernorm.weight");
        LD(q, "self_attn.q_proj.weight"); LD(k, "self_attn.k_proj.weight");
        LD(v, "self_attn.v_proj.weight"); LD(o, "self_attn.o_proj.weight");
        LD(qn,"self_attn.q_norm.weight"); LD(kn,"self_attn.k_norm.weight");
        LD(gate, "mlp.gate.weight");
        #undef LD
    }
    m->cache = calloc(c->n_layers, sizeof(LCache));
    for (int i = 0; i < c->n_layers; i++) { m->cache[i].cap = cap; m->cache[i].slots = calloc(cap, sizeof(Slot)); }
    m->dense_load_s = now_s() - t0;
    if (m->resident && !(getenv("COLI_NOWARM") && atoi(getenv("COLI_NOWARM"))))
        prewarm_experts(m);             /* all experts fit -> load them once, decode stays 100% hits */
}

/* reads a weight from disk (streaming) and quantizes it into q[O,I]+scale[O].
 * Pre-quantized container (convert_olmoe.py: int8 + scale f32 in "name.qs"):
 * direct raw read — half the I/O and zero quantize_rows at runtime. Before
 * this patch the int8 container caused SIGBUS (st_read_f32 on I8 tensors). */
static void load_expert_w(Model *m, const char *name, int8_t *q, float *scale, int O, int I, float *tmp) {
    st_tensor *t = st_find(&m->S, name);
    if (t && t->dtype == 3) {                    /* I8/U8: colibri container */
        char qs[300]; snprintf(qs, sizeof(qs), "%s.qs", name);
        st_read_raw(&m->S, name, q, 1);
        st_read_f32(&m->S, qs, scale, 1);
        return;
    }
    st_read_f32(&m->S, name, tmp, 1);            /* pread + fadvise DONTNEED */
    quantize_rows(tmp, q, scale, O, I, m->quant_bits);
}

/* ---------- expert cache: returns the quantized weights (q+scale) from cache or disk ---------- */
static void expert_get(Model *m, int layer, int eid, Slot **out) {
    LCache *lc = &m->cache[layer];
    for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) {
        m->hits++; lc->slots[i].used = ++m->clock; *out = &lc->slots[i]; return;
    }
    m->miss++;
    Cfg *c = &m->c;
    int64_t ng = (int64_t)c->inter * c->hidden, nd = (int64_t)c->hidden * c->inter;
    Slot *s;
    if (lc->n < lc->cap) {
        s = &lc->slots[lc->n++];
        s->g = malloc(ng); s->u = malloc(ng); s->d = malloc(nd);
        s->gs = falloc(c->inter); s->us = falloc(c->inter); s->ds = falloc(c->hidden);
    } else { int lru = 0; for (int i = 1; i < lc->n; i++) if (lc->slots[i].used < lc->slots[lru].used) lru = i; s = &lc->slots[lru]; }
    float *tmp = falloc(ng > nd ? ng : nd);
    char nm[256];
    snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.gate_proj.weight",layer,eid); load_expert_w(m,nm,s->g,s->gs,c->inter,c->hidden,tmp);
    snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.up_proj.weight",  layer,eid); load_expert_w(m,nm,s->u,s->us,c->inter,c->hidden,tmp);
    snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.down_proj.weight",layer,eid); load_expert_w(m,nm,s->d,s->ds,c->hidden,c->inter,tmp);
    free(tmp);
    s->eid = eid; s->used = ++m->clock;
    *out = s;
}

/* ---------- RoPE on a single head's vector (head_dim) at absolute position pos ---------- */
static void rope_head(float *x, int pos, const Cfg *c) {
    int h = c->head_dim / 2;
    for (int j = 0; j < h; j++) {
        float inv = powf(c->theta, -2.0f * j / c->head_dim);
        float ang = pos * inv, cs = cosf(ang), sn = sinf(ang);
        float a = x[j], b = x[j+h];
        x[j]   = a*cs - b*sn;
        x[j+h] = b*cs + a*sn;
    }
}

/* attention over the new tokens x[S,hidden]; pos_base = absolute position of the first new token */
static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out) {
    Cfg *c = &m->c; int H = c->n_heads, hd = c->head_dim, D = c->hidden;
    float *q = falloc((int64_t)S*D), *k = falloc((int64_t)S*D), *vv = falloc((int64_t)S*D);
    matmul(q, x, l->q, S, D, D);
    matmul(k, x, l->k, S, D, D);
    matmul(vv, x, l->v, S, D, D);
    /* qk-norm over the whole hidden vector, then RoPE per head */
    for (int s = 0; s < S; s++) {
        rmsnorm_row(q + (int64_t)s*D, q + (int64_t)s*D, l->qn, D, c->eps);
        rmsnorm_row(k + (int64_t)s*D, k + (int64_t)s*D, l->kn, D, c->eps);
        int pos = pos_base + s;
        for (int hh = 0; hh < H; hh++) { rope_head(q + (int64_t)s*D + hh*hd, pos, c); rope_head(k + (int64_t)s*D + hh*hd, pos, c); }
    }
    /* writes k,v into the kv-cache at positions pos_base..pos_base+S-1 */
    for (int s = 0; s < S; s++) for (int hh = 0; hh < H; hh++) {
        int t = pos_base + s;
        memcpy(m->K[layer] + ((int64_t)hh*m->max_t + t)*hd, k + (int64_t)s*D + hh*hd, hd*sizeof(float));
        memcpy(m->V[layer] + ((int64_t)hh*m->max_t + t)*hd, vv + (int64_t)s*D + hh*hd, hd*sizeof(float));
    }
    int Tk = pos_base + S;             /* total number of keys available */
    float scale = 1.f / sqrtf((float)hd);
    float *ctx = falloc((int64_t)S*D);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int hh = 0; hh < H; hh++) {
        for (int s = 0; s < S; s++) {
            int qpos = pos_base + s;
            const float *qv = q + (int64_t)s*D + hh*hd;
            float sc[4096];
            for (int t = 0; t <= qpos; t++) {          /* causal: t <= qpos */
                const float *kv = m->K[layer] + ((int64_t)hh*m->max_t + t)*hd;
                float acc = 0; for (int dd = 0; dd < hd; dd++) acc += qv[dd]*kv[dd];
                sc[t] = acc * scale;
            }
            softmax_row(sc, qpos+1);
            float *cx = ctx + (int64_t)s*D + hh*hd;
            for (int dd = 0; dd < hd; dd++) cx[dd] = 0;
            for (int t = 0; t <= qpos; t++) {
                const float *vrow = m->V[layer] + ((int64_t)hh*m->max_t + t)*hd;
                float a = sc[t];
                for (int dd = 0; dd < hd; dd++) cx[dd] += a * vrow[dd];
            }
        }
    }
    (void)Tk;
    matmul(out, ctx, l->o, S, D, D);
    free(q); free(k); free(vv); free(ctx);
}

/* MoE over the tokens x[S,hidden] -> out[S,hidden] */
static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, K = c->topk, I = c->inter;
    float *logits = falloc((int64_t)S*E);
    matmul(logits, x, l->gate, S, D, E);
    memset(out, 0, (int64_t)S*D*sizeof(float));
    float *g = falloc(I), *u = falloc(I), *hh = falloc(D);
    for (int s = 0; s < S; s++) {
        float *pr = logits + (int64_t)s*E;
        softmax_row(pr, E);
        /* top-K indices (partial selection) */
        int idx[64]; float val[64];
        for (int kk = 0; kk < K; kk++) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < E; e++) {
                int taken = 0; for (int j = 0; j < kk; j++) if (idx[j]==e){taken=1;break;}
                if (!taken && pr[e] > bv) { bv = pr[e]; best = e; }
            }
            idx[kk] = best; val[kk] = bv;
        }
        if (c->norm_topk) { float sm=0; for(int kk=0;kk<K;kk++) sm+=val[kk]; for(int kk=0;kk<K;kk++) val[kk]/=sm; }
        const float *xs = x + (int64_t)s*D;
        for (int kk = 0; kk < K; kk++) {
            Slot *e; expert_get(m, layer, idx[kk], &e);
            matmul_q(g, xs, e->g, e->gs, D, I);     /* gate_proj [I,D] */
            matmul_q(u, xs, e->u, e->us, D, I);     /* up_proj   [I,D] */
            for (int i = 0; i < I; i++) { float gv = g[i]; g[i] = (gv / (1.f + expf(-gv))) * u[i]; }
            matmul_q(hh, g, e->d, e->ds, I, D);     /* down_proj [D,I] */
            float w = val[kk];
            float *os = out + (int64_t)s*D;
            for (int d = 0; d < D; d++) os[d] += w * hh[d];
        }
    }
    free(logits); free(g); free(u); free(hh);
}

/* one step: new tokens ids[S] at position pos_base. Returns logits of the last token (malloc'd). */
static float *step(Model *m, const int *ids, int S, int pos_base) {
    Cfg *c = &m->c; int D = c->hidden;
    float *x = falloc((int64_t)S*D);
    for (int s = 0; s < S; s++) memcpy(x + (int64_t)s*D, m->embed + (int64_t)ids[s]*D, D*sizeof(float));
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->in_ln, D, c->eps);
        attention(m, l, i, nrm, S, pos_base, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
        moe(m, l, i, nrm, S, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
    }
    m->kv_len = pos_base + S;
    /* only the last token -> logits */
    float *last = falloc(D);
    rmsnorm_row(last, x + (int64_t)(S-1)*D, m->final_norm, D, c->eps);
    float *logit = falloc(c->vocab);
    matmul(logit, last, m->lm_head, 1, D, c->vocab);
    free(x); free(nrm); free(tmp); free(last);
    return logit;
}

/* greedy generation. prompt[np] -> fills out[np+n_new] */
static void generate(Model *m, const int *prompt, int np, int n_new, int *out) {
    Cfg *c = &m->c;
    m->max_t = np + n_new;
    m->K = calloc(c->n_layers, sizeof(float*)); m->V = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++) {
        m->K[i] = falloc((int64_t)c->n_heads * m->max_t * c->head_dim);
        m->V[i] = falloc((int64_t)c->n_heads * m->max_t * c->head_dim);
    }
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    float *logit = step(m, prompt, np, 0);          /* PREFILL */
    int len = np;
    for (int s = 0; s < n_new; s++) {
        int best = 0; float bv = logit[0];
        for (int i = 1; i < c->vocab; i++) if (logit[i] > bv) { bv = logit[i]; best = i; }
        free(logit);
        out[len++] = best;
        if (s == n_new - 1) break;
        int one = best;
        logit = step(m, &one, 1, len - 1);          /* DECODE */
    }
}

/* ---------- ref.json reading ---------- */
static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    int *r = malloc(a->len * sizeof(int));
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len; return r;
}

int main(int argc, char **argv) {
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }
    int cap  = argc > 1 ? atoi(argv[1]) : 0;   /* 0 (default) -> auto-plan cache size from RAM */
    int bits = argc > 2 ? atoi(argv[2]) : 8;
    if (bits < 2 || bits > 8) {   /* expert storage is int8_t: bits>8 truncates in quantize_rows (#134). f32 mode is not implemented here — int8 is already token-exact vs the oracle. */
        fprintf(stderr, "quant_bits must be 2..8 (got %d); OLMoE experts are int8-backed, no f32 mode\n", bits);
        return 1;
    }
    const char *refpath = argc > 3 ? argv[3] : "ref.json";

    FILE *f = fopen(refpath, "rb"); if(!f){perror(refpath);return 1;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *ref = json_parse(buf, &arena);
    int np, nfull; int *prompt = read_int_array(ref,"prompt_ids",&np); int *full = read_int_array(ref,"full_ids",&nfull);
    int n_new = nfull - np;

    Model m; model_init(&m, snap, cap, bits);
    int64_t expert_gb_x10 = m.c.n_experts * (int64_t)m.c.n_layers * expert_cache_bytes(&m.c) / (int64_t)1e8;
    if (m.resident)
        printf("== C engine: FULLY RESIDENT — all %d experts/layer in RAM, ZERO disk streaming, experts @ %d-bit ==\n",
               m.c.n_experts, bits);
    else
        printf("== C engine: STREAMING — cache %d/%d experts/layer (model %.1f GB > RAM budget), experts @ %d-bit ==\n",
               m.cap, m.c.n_experts, expert_gb_x10 / 10.0, bits);
    printf("dense loaded in %.1fs | RSS after load: %.2f GB\n", m.dense_load_s, rss_gb());
    if (m.resident && m.warm_s > 0)
        printf("prewarmed %d experts in %.1fs (decode will be 100%% cache hits)\n",
               m.c.n_experts * m.c.n_layers, m.warm_s);

    int *out = malloc((np + n_new) * sizeof(int));
    double t = now_s();
    generate(&m, prompt, np, n_new, out);
    double dt = now_s() - t;

    int match = 0;
    printf("\nReference: ");  for (int i=np;i<nfull;i++) printf("%d ", full[i]);
    printf("\nC engine : ");  for (int i=np;i<nfull;i++) { printf("%d ", out[i]); if (out[i]==full[i]) match++; }
    printf("\nMatching tokens: %d/%d\n", match, n_new);
    double tot = m.hits + m.miss;
    printf("\nPEAK RSS: %.2f GB\n", rss_gb());
    printf("Expert cache hit rate: %.1f%%  (hit=%llu miss=%llu)\n", tot?100.0*m.hits/tot:0.0,
           (unsigned long long)m.hits, (unsigned long long)m.miss);
    printf("Speed: %.2f tok/s (%.1fs for %d tokens)\n", n_new/dt, dt, n_new);
    free(buf); free(arena);
    return 0;
}
