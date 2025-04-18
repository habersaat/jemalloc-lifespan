#include <jemalloc/jemalloc.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_LIFESPAN_CLASSES 7
#define ALLOC_SIZE (64 * 1024)

uint64_t lifespan_class_deadlines_ns[NUM_LIFESPAN_CLASSES] = {
    1ULL * 1000 * 1000,        // 1ms
    10ULL * 1000 * 1000,       // 10ms
    100ULL * 1000 * 1000,      // 100ms
    500ULL * 1000 * 1000,      // 500ms
    2000ULL * 1000 * 1000,     // 2s
    5000ULL * 1000 * 1000,     // 5s
    10000ULL * 1000 * 1000     // 10s
};

int main() {
    printf("🔬 [test] Starting structured reclaimer test...\n");

    void *allocs[NUM_LIFESPAN_CLASSES];

    // Allocate short-lived block (class 0)
    allocs[0] = je_malloc(ALLOC_SIZE);
    printf("🔧 [test] Allocated class 0 (short-lived) at %p\n", allocs[0]);
    usleep(50 * 1000); // 50ms

    // Allocate medium-lived block (class 1)
    allocs[1] = je_malloc(ALLOC_SIZE);
    printf("🔧 [test] Allocated class 1 (medium-lived) at %p\n", allocs[1]);
    usleep(100 * 1000); // 100ms

    // Allocate long-lived block (class 2)
    allocs[2] = je_malloc(ALLOC_SIZE);
    printf("🔧 [test] Allocated class 2 (long-lived) at %p\n", allocs[2]);

    // Sleep to allow blocks to start aging
    printf("⏳ [test] Sleeping 0.5s to let reclaimer begin...\n");
    usleep(500 * 1000); // 500ms

    // Free class 0 (should be expired already)
    printf("🗑️ [test] Freeing class 0 (short-lived)\n");
    je_free(allocs[0]);

    // Sleep and free class 1 (should be expired by now too)
    usleep(200 * 1000); // 200ms
    printf("🗑️ [test] Freeing class 1 (medium-lived)\n");
    je_free(allocs[1]);

    // Sleep and free class 2 (should just be hitting deadline)
    sleep(1);
    printf("🗑️ [test] Freeing class 2 (long-lived)\n");
    je_free(allocs[2]);

    // Trigger reuse
    void *new_alloc = je_malloc(ALLOC_SIZE);
    printf("🔁 [test] Allocated post-expiry block: %p\n", new_alloc);
    je_free(new_alloc);

    // Let reclaimer finalize
    printf("⏳ [test] Sleeping again to allow post-free expiration...\n");
    sleep(2);

    printf("✅ [test] Done.\n");
    return 0;
}
