/* On-demand indexing and reading of tensors from multiple safetensors files.
 * Equivalent to Shards in engine.py, but:
 *   - reads with pread (no mmap) + posix_fadvise(DONTNEED) -> the pages do NOT
 *     stay resident in the process. This is the RSS bug fix: this way the
 *     peak RAM stays dense+cache, not the whole model. (see memory mmap-rss-bug)
 *   - always converts to float32 on output (BF16/F16/F32 supported). */
#ifndef ST_H
#define ST_H
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include "json.h"
#include "compat.h"

/* tetto sulla dimensione dell'header safetensors: gli header reali sono piccoli
 * (KB..pochi MB). Un file crafted che dichiara un hlen enorme causerebbe una
 * malloc gigante prima ancora di leggere: lo respingiamo. */
#define ST_MAX_HEADER (512ll << 20)

typedef struct {
    char   *name;
    int     fd;
    int64_t off;       /* absolute offset of the data within the file */
    int64_t nbytes;
    int     dtype;     /* 0=BF16 1=F16 2=F32 */
    int64_t numel;
} st_tensor;

typedef struct {
    st_tensor *t;
    int        n, cap;
    int        fds[512];
    int        dfds[512];  /* O_DIRECT twins (opened lazily): -2 = not yet attempted */
    char      *paths[512];
    int        nfd;
    int       *hidx;      /* hash map name->index (open addressing): with ~120k tensors
                           * (GLM: 256 experts x 78 layers x 3 x 2) the linear scan
                           * cost tens of seconds/token (measured on the first real run) */
    int        hcap;
} shards;
#define ST_MAX_SHARDS 512

static uint64_t st_hash(const char *s){
    uint64_t h=1469598103934665603ULL;
    while(*s){ h^=(unsigned char)*s++; h*=1099511628211ULL; }
    return h;
}

static int st_dtype_code(const char *s) {
    if (!strcmp(s, "BF16")) return 0;
    if (!strcmp(s, "F16"))  return 1;
    if (!strcmp(s, "F32"))  return 2;
    if (!strcmp(s, "U8"))   return 3;   /* quantized data (int4 packed / int8) */
    if (!strcmp(s, "I8"))   return 3;
    fprintf(stderr, "unsupported dtype: %s\n", s); exit(1);
}

static inline float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t u;
    if (exp == 0) {            /* subnormal or zero */
        if (man == 0) u = sign;
        else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; exp--; } man &= 0x3FF; u = sign | (exp << 23) | (man << 13); }
    } else if (exp == 0x1F) {  /* inf/nan */
        u = sign | 0x7F800000 | (man << 13);
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f; memcpy(&f, &u, 4); return f;
}

static int st_open_fd(shards *S, const char *path) {
    for (int i = 0; i < S->nfd; i++) if (!strcmp(S->paths[i], path)) return S->fds[i];
    int fd = open(path, COMPAT_O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    S->paths[S->nfd] = strdup(path); S->fds[S->nfd] = fd;
#ifdef O_DIRECT
    S->dfds[S->nfd] = open(path, COMPAT_O_RDONLY | O_DIRECT);   /* eager: lookup is then thread-safe */
#elif defined(__APPLE__) || defined(_WIN32)
    S->dfds[S->nfd] = compat_open_direct(path);          /* macOS: F_NOCACHE; Windows: NO_BUFFERING */
#else
    S->dfds[S->nfd] = -1;                                /* no equivalent: buffered only */
#endif
    S->nfd++;
    return fd;
}

/* O_DIRECT twin fd of the same file (bypasses the page cache: buffered read on
 * ext4-in-VHDX chokes at ~0.8 GB/s, O_DIRECT reaches 2.3+; measured). -1 if unavailable. */
static int st_direct_fd(shards *S, int fd) {
    for (int i = 0; i < S->nfd; i++) if (S->fds[i] == fd) return S->dfds[i];
    return -1;
}

/* index all model-*.safetensors in snap_dir */
static void st_init(shards *S, const char *snap_dir) {
    memset(S, 0, sizeof(*S));
    S->cap = 4096; S->t = calloc(S->cap, sizeof(st_tensor));
    /* collect the shard file names in order */
    static char files[ST_MAX_SHARDS][1024]; int nf = 0;
    DIR *d = opendir(snap_dir); struct dirent *e;
    if (!d) { perror(snap_dir); exit(1); }
    while ((e = readdir(d))) {
        const char *dot = strrchr(e->d_name, '.');
        if (dot && !strcmp(dot, ".safetensors")) {  /* model.safetensors or model-0000N-of-... */
            if (nf >= ST_MAX_SHARDS) { fprintf(stderr, "too many shards (>%d): raise ST_MAX_SHARDS\n", ST_MAX_SHARDS); exit(1); }
            snprintf(files[nf++], 1024, "%s/%s", snap_dir, e->d_name);
        }
    }
    closedir(d);
    for (int a = 0; a < nf; a++) for (int b = a+1; b < nf; b++)
        if (strcmp(files[a], files[b]) > 0) { char tmp[1024]; strcpy(tmp, files[a]); strcpy(files[a], files[b]); strcpy(files[b], tmp); }

    for (int fi = 0; fi < nf; fi++) {
        int fd = st_open_fd(S, files[fi]);
        struct stat sst;
        if (fstat(fd, &sst) != 0) { perror("fstat shard"); exit(1); }
        int64_t fsz = (int64_t)sst.st_size;
        uint64_t hlen;
        if (pread(fd, &hlen, 8, 0) != 8) { perror("pread hlen"); exit(1); }
        /* file malevolo/troncato: hlen deve stare nel file dopo gli 8 byte di
         * prefisso e sotto il tetto. Senza questo bound hlen+1 puo' andare in
         * overflow (malloc(0) e poi hdr[hlen]=0 fuori limiti) o forzare una
         * malloc gigante. */
        if (fsz < 8 || hlen > (uint64_t)(fsz - 8) || hlen > (uint64_t)ST_MAX_HEADER) {
            fprintf(stderr, "%s: bad safetensors header length %llu (file %lld bytes)\n",
                    files[fi], (unsigned long long)hlen, (long long)fsz); exit(1); }
        char *hdr = malloc(hlen + 1);
        if (!hdr) { perror("malloc safetensors header"); exit(1); }
        if (pread(fd, hdr, hlen, 8) != (ssize_t)hlen) { perror("pread hdr"); exit(1); }
        hdr[hlen] = 0;
        int64_t data_start = 8 + (int64_t)hlen;
        char *arena = NULL;
        jval *root = json_parse(hdr, &arena);
        if (!root || root->t != J_OBJ) {
            fprintf(stderr, "%s: safetensors header is not a JSON object\n", files[fi]); exit(1); }
        for (int i = 0; i < root->len; i++) {
            const char *name = root->keys[i];
            if (!strcmp(name, "__metadata__")) continue;
            jval *m = root->kids[i];
            jval *dt = json_get(m, "dtype");
            jval *off = json_get(m, "data_offsets");
            jval *shp = json_get(m, "shape");
            /* un header crafted puo' omettere i campi o dare tipi sbagliati:
             * senza questi guard si dereferenzia NULL (json_get) o si legge
             * off->kids[0/1] oltre i limiti dell'array. */
            if (!dt || dt->t != J_STR || !off || off->t != J_ARR || off->len < 2 ||
                !shp || shp->t != J_ARR) {
                fprintf(stderr, "%s: tensor '%s' has malformed dtype/data_offsets/shape\n",
                        files[fi], name); exit(1); }
            int64_t a0 = (int64_t)off->kids[0]->num, b0 = (int64_t)off->kids[1]->num;
            /* offset dichiarati dal file: non-negativi, ordinati e dentro al
             * file. Altrimenti nbytes=b0-a0 diventa negativo -> malloc((size_t))
             * gigante e la memcpy in st_read_f32 sfora il buffer del chiamante;
             * oppure off punta fuori dal file. */
            if (a0 < 0 || b0 < a0 || data_start + b0 > fsz) {
                fprintf(stderr, "%s: tensor '%s' data_offsets [%lld,%lld] out of file bounds (%lld)\n",
                        files[fi], name, (long long)a0, (long long)b0, (long long)fsz); exit(1); }
            int64_t numel = 1; for (int k = 0; k < shp->len; k++) numel *= (int64_t)shp->kids[k]->num;
            if (S->n == S->cap) { S->cap *= 2; S->t = realloc(S->t, S->cap*sizeof(st_tensor)); }
            st_tensor *t = &S->t[S->n++];
            t->name = strdup(name); t->fd = fd; t->off = data_start + a0;
            t->nbytes = b0 - a0; t->dtype = st_dtype_code(dt->str); t->numel = numel;
        }
        free(arena); /* the jvals stay leaked: fine, one-time at startup */
        free(hdr);
    }
    /* hash index built at the end of indexing (the indices stay valid after the reallocs) */
    S->hcap = 1; while (S->hcap < S->n * 2) S->hcap <<= 1;
    S->hidx = malloc(S->hcap * sizeof(int));
    for (int i = 0; i < S->hcap; i++) S->hidx[i] = -1;
    for (int i = 0; i < S->n; i++) {
        uint64_t h = st_hash(S->t[i].name) & (S->hcap - 1);
        while (S->hidx[h] >= 0) h = (h + 1) & (S->hcap - 1);
        S->hidx[h] = i;
    }
}

static st_tensor *st_find(shards *S, const char *name) {
    if (S->hidx) {
        uint64_t h = st_hash(name) & (S->hcap - 1);
        while (S->hidx[h] >= 0) {
            st_tensor *t = &S->t[S->hidx[h]];
            if (!strcmp(t->name, name)) return t;
            h = (h + 1) & (S->hcap - 1);
        }
        return NULL;
    }
    for (int i = 0; i < S->n; i++) if (!strcmp(S->t[i].name, name)) return &S->t[i];
    return NULL;
}
static int st_has(shards *S, const char *name) { return st_find(S, name) != NULL; }

/* ASYNCHRONOUS prefetch: tells the kernel to start reading the tensor's pages in
 * the background (readahead). Serves to overlap expert I/O with computation: prefetch
 * the whole expert set of a layer, then the synchronous preads find the cache
 * already warm. No-op if the tensor does not exist (e.g. the first .qs before the read). */
static void st_prefetch(shards *S, const char *name) {
    st_tensor *t = st_find(S, name);
    if (t) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_WILLNEED);
}

/* reads a tensor into a float32 buffer provided by the caller (numel floats).
 * drop=1 -> advises the kernel to discard the pages (for streaming experts). */
static int64_t st_read_f32(shards *S, const char *name, float *out, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    void *raw = malloc(t->nbytes);
    if (!raw) { fprintf(stderr, "malloc %lld bytes for tensor %s failed\n", (long long)t->nbytes, name); exit(1); }
    if (pread(t->fd, raw, t->nbytes, t->off) != t->nbytes) { perror("pread data"); exit(1); }
    if (t->dtype == 2) {
        memcpy(out, raw, t->nbytes);
    } else if (t->dtype == 0) {
        uint16_t *p = (uint16_t *)raw; for (int64_t i = 0; i < t->numel; i++) out[i] = bf16_to_f32(p[i]);
    } else {
        uint16_t *p = (uint16_t *)raw; for (int64_t i = 0; i < t->numel; i++) out[i] = f16_to_f32(p[i]);
    }
    free(raw);
    if (drop) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_DONTNEED);
    return t->numel;
}

static int64_t st_numel(shards *S, const char *name) {
    st_tensor *t = st_find(S, name); return t ? t->numel : -1;
}
static int64_t st_nbytes(shards *S, const char *name) {
    st_tensor *t = st_find(S, name); return t ? t->nbytes : -1;
}

/* reads the RAW bytes of a tensor (no dtype conversion): for the already
 * int4/int8-quantized weights of our container (dtype U8). drop=1 -> fadvise DONTNEED. */
static void st_read_raw(shards *S, const char *name, void *out, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    if (pread(t->fd, out, t->nbytes, t->off) != t->nbytes) { perror("pread raw"); exit(1); }
    if (drop) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_DONTNEED);
}

/* reads a SLICE of a tensor: n_elems starting from element elem_off.
 * Serves the fused experts of GLM (one tensor = block [E, ...]): read
 * only the requested expert via pread of the sub-range, no reading of the whole block. */
static void st_read_slice_f32(shards *S, const char *name, int64_t elem_off, int64_t n_elems, float *out, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    int esz = (t->dtype == 2) ? 4 : 2;
    int64_t boff = t->off + elem_off * esz, nb = n_elems * esz;
    void *raw = malloc(nb);
    if (pread(t->fd, raw, nb, boff) != nb) { perror("pread slice"); exit(1); }
    if (t->dtype == 2) memcpy(out, raw, nb);
    else if (t->dtype == 0) { uint16_t *p = raw; for (int64_t i = 0; i < n_elems; i++) out[i] = bf16_to_f32(p[i]); }
    else { uint16_t *p = raw; for (int64_t i = 0; i < n_elems; i++) out[i] = f16_to_f32(p[i]); }
    free(raw);
    if (drop) posix_fadvise(t->fd, boff, nb, POSIX_FADV_DONTNEED);
}

#endif
