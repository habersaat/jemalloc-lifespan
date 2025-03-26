#include <jemalloc/jemalloc.h>

int main() {
    void *a = je_malloc(64);
    void *b = je_malloc(64);
    void *c = je_malloc(64);

    je_free(a);
    je_free(b);
    je_free(c);
    
    return 0;
}
