#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#define JEMALLOC_NO_DEMANGLE
#define JEMALLOC_MANGLE
#define JEMALLOC_EXPORT
#include "redis/deps/jemalloc/include/jemalloc/jemalloc.h"

int main() {
    bool prof_enabled = false;
    size_t sz = sizeof(prof_enabled);

    if (je_mallctl("opt.prof", &prof_enabled, &sz, NULL, 0) != 0) {
        printf("🚨 je_mallctl(\"opt.prof\") failed\n");
    } else {
        printf("🔍 opt.prof = %s\n", prof_enabled ? "true" : "false");
    }

    for (int i = 0; i < 100000; i++) {
        (void)malloc(1024);
    }

    const char *dump = "manual-profile.heap";
    if (je_mallctl("prof.dump", NULL, NULL, (void *)&dump, sizeof(const char *)) != 0) {
        printf("❌ je_mallctl(\"prof.dump\") failed\n");
    } else {
        printf("✅ Profile dumped: %s\n", dump);
    }

    return 0;
}
