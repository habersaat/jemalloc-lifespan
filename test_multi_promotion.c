#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <sys/mman.h>

// These should match your jemalloc build
#include <jemalloc/jemalloc.h>

#define ALLOC_SIZE (64 * 1024) // 64KB

int main() {
    printf("🔬 [test] Starting multi-allocation promotion test...\n\n");

    // Force early allocation to trigger arena/shard setup
    void *warmup = je_malloc(ALLOC_SIZE);
    if (!warmup) {
        fprintf(stderr, "❌ [test] Warmup allocation failed\n");
        return 1;
    }
    printf("[test] Deallocating extent for warmup alloc of size %d\n", ALLOC_SIZE);
    je_free(warmup);

    //  Multiple Allocations
    void *ptrs[3];
    for (int i = 0; i < 3; ++i) {
        ptrs[i] = je_malloc(ALLOC_SIZE);
        if (!ptrs[i]) {
            fprintf(stderr, "❌ [test] Allocation %d failed\n", i);
            return 1;
        }
        printf("🔧 [test] Allocated ptr[%d] = %p\n", i, ptrs[i]);
    }

    // Sleep to allow promotion opportunity
    printf("⏳ [test] Sleeping for 150ms to exceed class 0 (10ms) and class 1 (100ms)...\n");
    usleep(150000); // 150 milliseconds

    // Free and observe promotion/misclassification
    for (int i = 0; i < 3; ++i) {
        printf(" [test] Freeing ptr[%d] = %p\n", i, ptrs[i]);
        je_free(ptrs[i]);
    }

    // Wait long enough for final class 2 (1s) block to expire, if eligible
    printf("⏳ [test] Waiting 3s to observe reclamation of promoted blocks...\n");
    sleep(3);

    printf("✅ [test] Multi-allocation promotion test complete.\n");
    return 0;
}
