/* Parser JSON minimale, header-only. Serve per:
 *  - l'header dei file safetensors (un grande oggetto nome->{dtype,shape,data_offsets})
 *  - ref.json (per leggere prompt_ids / full_ids)
 * Non e' completo (niente unicode \uXXXX, niente notazione esotica) ma copre cio' che serve. */
#ifndef JSON_H
#define JSON_H
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <stdint.h>

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } jtype;

typedef struct jval {
    jtype t;
    double num;            /* J_NUM */
    int    boolean;        /* J_BOOL */
    char  *str;            /* J_STR (NUL-terminata, dentro l'arena) */
    /* array: figli in [0..len); oggetto: chiavi[] e figli[] in parallelo */
    struct jval **kids;
    char        **keys;    /* solo per J_OBJ */
    int           len;
} jval;

typedef struct {
    const char *s;
    char       *arena;     /* buffer per le stringhe smontate */
    size_t      acap, aoff;
} jparser;

static char *j_dup(jparser *p, const char *b, int n) {
    /* ogni stringa ha la sua allocazione: un'arena con realloc sposterebbe il
     * buffer invalidando i puntatori gia' emessi (use-after-free). */
    (void)p;
    char *d = (char *)malloc(n + 1);
    memcpy(d, b, n); d[n] = 0;
    return d;
}

static void j_ws(jparser *p) { while (*p->s && isspace((unsigned char)*p->s)) p->s++; }

static jval *j_new(jtype t) {
    jval *v = (jval *)calloc(1, sizeof(jval));
    v->t = t; return v;
}

static jval *j_parse_val(jparser *p);

static char *j_parse_str_raw(jparser *p) {
    /* assume *p->s == '"' */
    p->s++;
    const char *start = p->s;
    /* trova la fine gestendo gli escape, poi copia decodificando i casi base */
    char tmp[1 << 16]; int n = 0;
    #define J_PUT(ch) do{ if (n < (int)sizeof(tmp)-1) tmp[n++] = (char)(ch); }while(0)
    while (*p->s && *p->s != '"') {
        char c = *p->s++;
        if (c == '\\' && *p->s) {
            char e = *p->s++;
            switch (e) {
                case 'n': c = '\n'; break; case 't': c = '\t'; break;
                case 'r': c = '\r'; break; case 'b': c = '\b'; break;
                case 'f': c = '\f'; break; case '/': c = '/'; break;
                case '\\': c = '\\'; break; case '"': c = '"'; break;
                case 'u': {  /* \uXXXX -> codepoint UTF-8 (con coppie surrogate) */
                    unsigned cp = (unsigned)strtoul((char[]){p->s[0],p->s[1],p->s[2],p->s[3],0}, NULL, 16);
                    p->s += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && p->s[0]=='\\' && p->s[1]=='u') {
                        unsigned lo = (unsigned)strtoul((char[]){p->s[2],p->s[3],p->s[4],p->s[5],0}, NULL, 16);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) { cp = 0x10000 + ((cp-0xD800)<<10) + (lo-0xDC00); p->s += 6; }
                    }
                    if (cp < 0x80) { J_PUT(cp); }
                    else if (cp < 0x800) { J_PUT(0xC0|(cp>>6)); J_PUT(0x80|(cp&0x3F)); }
                    else if (cp < 0x10000) { J_PUT(0xE0|(cp>>12)); J_PUT(0x80|((cp>>6)&0x3F)); J_PUT(0x80|(cp&0x3F)); }
                    else { J_PUT(0xF0|(cp>>18)); J_PUT(0x80|((cp>>12)&0x3F)); J_PUT(0x80|((cp>>6)&0x3F)); J_PUT(0x80|(cp&0x3F)); }
                    continue;
                }
                default: c = e; break;
            }
        }
        J_PUT(c);
    }
    #undef J_PUT
    if (*p->s == '"') p->s++;
    (void)start;
    return j_dup(p, tmp, n);
}

static jval *j_parse_val(jparser *p) {
    j_ws(p);
    char c = *p->s;
    if (c == '"') { jval *v = j_new(J_STR); v->str = j_parse_str_raw(p); return v; }
    if (c == '{') {
        p->s++; jval *v = j_new(J_OBJ);
        int cap = 8; v->keys = malloc(cap * sizeof(char*)); v->kids = malloc(cap * sizeof(jval*));
        j_ws(p);
        if (*p->s == '}') { p->s++; return v; }
        for (;;) {
            j_ws(p);
            char *key = j_parse_str_raw(p);
            j_ws(p); if (*p->s == ':') p->s++;
            jval *val = j_parse_val(p);
            if (v->len == cap) { cap *= 2; v->keys = realloc(v->keys, cap*sizeof(char*)); v->kids = realloc(v->kids, cap*sizeof(jval*)); }
            v->keys[v->len] = key; v->kids[v->len] = val; v->len++;
            j_ws(p);
            if (*p->s == ',') { p->s++; continue; }
            if (*p->s == '}') { p->s++; break; }
            break;
        }
        return v;
    }
    if (c == '[') {
        p->s++; jval *v = j_new(J_ARR);
        int cap = 8; v->kids = malloc(cap * sizeof(jval*));
        j_ws(p);
        if (*p->s == ']') { p->s++; return v; }
        for (;;) {
            jval *val = j_parse_val(p);
            if (v->len == cap) { cap *= 2; v->kids = realloc(v->kids, cap*sizeof(jval*)); }
            v->kids[v->len++] = val;
            j_ws(p);
            if (*p->s == ',') { p->s++; continue; }
            if (*p->s == ']') { p->s++; break; }
            break;
        }
        return v;
    }
    if (c == 't') { p->s += 4; jval *v = j_new(J_BOOL); v->boolean = 1; return v; }
    if (c == 'f') { p->s += 5; jval *v = j_new(J_BOOL); v->boolean = 0; return v; }
    if (c == 'n') { p->s += 4; return j_new(J_NULL); }
    /* numero */
    { char *end; double d = strtod(p->s, &end); p->s = end; jval *v = j_new(J_NUM); v->num = d; return v; }
}

/* API */
static jval *json_parse(const char *text, char **arena_out) {
    jparser p = { text, NULL, 0, 0 };
    jval *v = j_parse_val(&p);
    if (arena_out) *arena_out = p.arena; else free(p.arena);
    return v;
}

static jval *json_get(jval *o, const char *key) {
    if (!o || o->t != J_OBJ) return NULL;
    for (int i = 0; i < o->len; i++) if (strcmp(o->keys[i], key) == 0) return o->kids[i];
    return NULL;
}

static void json_free(jval *value) {
    if (!value) return;
    for (int i=0;i<value->len;i++) {
        if (value->t==J_OBJ) free(value->keys[i]);
        json_free(value->kids[i]);
    }
    if (value->t==J_STR) free(value->str);
    free(value->keys); free(value->kids); free(value);
}

/* Re-serializer: the inverse of json_parse, used to re-embed a parsed value (e.g. a tool
 * signature echoed back into a prompt, or a decoded arguments object) as compact JSON
 * text. Growable buffer, independent of any single model server's own string type. */
typedef struct { char *data; size_t len, cap; } jbuf;

static int jbuf_append(jbuf *b, const char *s, size_t n) {
    if (n > SIZE_MAX - b->len - 1) return 0;
    size_t need = b->len + n + 1;
    if (need > b->cap) {
        size_t cap = b->cap ? b->cap : 256;
        while (cap < need) { if (cap > SIZE_MAX / 2) return 0; cap *= 2; }
        char *p = (char *)realloc(b->data, cap);
        if (!p) return 0;
        b->data = p; b->cap = cap;
    }
    memcpy(b->data + b->len, s, n); b->len += n; b->data[b->len] = 0;
    return 1;
}

static int json_encode_str(jbuf *b, const char *s, size_t n) {
    static const char hex[] = "0123456789abcdef";
    if (!jbuf_append(b, "\"", 1)) return 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') { char x[2] = {'\\', (char)c}; if (!jbuf_append(b,x,2)) return 0; }
        else if (c == '\n') { if (!jbuf_append(b,"\\n",2)) return 0; }
        else if (c == '\r') { if (!jbuf_append(b,"\\r",2)) return 0; }
        else if (c == '\t') { if (!jbuf_append(b,"\\t",2)) return 0; }
        else if (c < 0x20) { char x[6] = {'\\','u','0','0',hex[c>>4],hex[c&15]}; if (!jbuf_append(b,x,6)) return 0; }
        else if (!jbuf_append(b,s+i,1)) return 0;
    }
    return jbuf_append(b, "\"", 1);
}

static inline int json_encode(jbuf *b, jval *v) {
    if (!v) return jbuf_append(b, "null", 4);
    switch (v->t) {
        case J_NULL: return jbuf_append(b, "null", 4);
        case J_BOOL: return v->boolean ? jbuf_append(b,"true",4) : jbuf_append(b,"false",5);
        case J_NUM: { char buf[64]; int n = snprintf(buf,sizeof buf,"%.17g",v->num);
                      return n > 0 && jbuf_append(b, buf, (size_t)n); }
        case J_STR: return json_encode_str(b, v->str, strlen(v->str));
        case J_ARR:
            if (!jbuf_append(b,"[",1)) return 0;
            for (int i = 0; i < v->len; i++) {
                if (i && !jbuf_append(b,",",1)) return 0;
                if (!json_encode(b, v->kids[i])) return 0;
            }
            return jbuf_append(b,"]",1);
        case J_OBJ:
            if (!jbuf_append(b,"{",1)) return 0;
            for (int i = 0; i < v->len; i++) {
                if (i && !jbuf_append(b,",",1)) return 0;
                if (!json_encode_str(b, v->keys[i], strlen(v->keys[i]))) return 0;
                if (!jbuf_append(b,":",1)) return 0;
                if (!json_encode(b, v->kids[i])) return 0;
            }
            return jbuf_append(b,"}",1);
    }
    return 0;
}

#endif
