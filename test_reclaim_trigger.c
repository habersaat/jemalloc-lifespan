#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <jemalloc/jemalloc.h>

#define NUM_ALLOCS 128
#define ALLOC_SIZE 65536  // 64KB — one slice per alloc (assuming 256KB max slice)

int main() {
    printf("🔬 [test] Starting aggressive block reclamation test...\n");

    void *ptrs[NUM_ALLOCS] = {0};

    // Phase 1: Allocate slices rapidly
    for (int i = 0; i < NUM_ALLOCS; i++) {
        ptrs[i] = je_malloc(ALLOC_SIZE);
        if (!ptrs[i]) {
            fprintf(stderr, "❌ Allocation %d failed\n", i);
            continue;
        }
        memset(ptrs[i], 0xAB, ALLOC_SIZE);
        printf("✅ Alloc[%3d] = %p\n", i, ptrs[i]);

        // Optional: Space out timestamps slightly
        usleep(1000);  // 1ms between each — can vary this
    }

    // Phase 2: Sleep to exceed short-lived lifespan class deadline (e.g., 10ms)
    printf("⏳ Sleeping to exceed lifespan deadline...\n");
    usleep(20000);  // 20ms

    // Phase 3: Free all allocations — this should leave many 2MB blocks empty
    for (int i = 0; i < NUM_ALLOCS; i++) {
        if (ptrs[i]) {
            je_free(ptrs[i]);
            printf("🧹 Freed [%3d] = %p\n", i, ptrs[i]);
        }
    }

    // Phase 4: Let reclaimer detect and expire blocks
    printf("⏳ Waiting for reclaimer to run and reclaim...\n");
    sleep(3);

    printf("✅ [test] Done.\n");
    return 0;
}
