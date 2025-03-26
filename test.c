#include <jemalloc/jemalloc.h>

int main() {
    // Small allocations (64-byte mallocs) served from slabs
    void *a = je_malloc(64);
    void *b = je_malloc(64);
    void *c = je_malloc(64);

    je_free(a);
    je_free(b);
    je_free(c);

    // Large allocations (2MB mallocs) served from the page allocator
    void *d = je_malloc(2 * 1024 * 1024);
    je_free(d);
    
    return 0;
}
