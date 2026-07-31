/* test_lfmtok — encode/decode driver for lfmtok.h.
 *
 * Reads one test string per line from stdin (with \n, \t, \\ escapes so a case can
 * contain newlines) and prints its token ids, space-separated, one line per case.
 * tools/lfmtok_check.py diffs that against HF's own tokenizer. Keeping the C side
 * a dumb pipe is deliberate: the comparison logic lives where the reference is. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lfmtok.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s tok.bin [--decode]\n", argv[0]); return 1; }
    LfmTok *t = lfmtok_load(argv[1]);
    if (!t) return 1;
    int do_decode = (argc > 2 && !strcmp(argv[2], "--decode"));

    char line[65536];
    while (fgets(line, sizeof line, stdin)) {
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;

        char *text = malloc(n + 1);
        size_t m = 0;
        for (size_t i = 0; i < n; i++) {
            if (line[i] == '\\' && i + 1 < n) {
                char c = line[++i];
                text[m++] = c == 'n' ? '\n' : c == 't' ? '\t' : c == 'r' ? '\r' : c;
            } else text[m++] = line[i];
        }
        text[m] = 0;

        int ids[32768];
        int k = lfmtok_encode(t, text, ids, 32768);
        for (int i = 0; i < k; i++) printf(i ? " %d" : "%d", ids[i]);
        if (do_decode) {
            char back[65536];
            lfmtok_decode(t, ids, k, back, sizeof back);
            printf("\t%s", strcmp(back, text) ? "ROUNDTRIP-FAIL" : "ok");
        }
        printf("\n");
        free(text);
    }
    return 0;
}
