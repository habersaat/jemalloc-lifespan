#include <jemalloc/jemalloc.h>

int main() {
    void *p = je_malloc(64);
    je_free(p);
    return 0;
}
