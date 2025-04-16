#include <jemalloc/jemalloc.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_LIFESPAN_CLASSES 3
#define ALLOC_SIZE (64 * 1024)
#define CLASS_SHORT 0
#define CLASS_MEDIUM 1
#define CLASS_LONG 2

// These should match your actual runtime definitions
uint64_t lifespan_class_deadlines_ns[NUM_LIFESPAN_CLASSES] = {
    10ULL * 1000 * 1000,    // 10ms for short-lived class (class 0)
    100ULL * 1000 * 1000,   // 100ms for medium-lived class (class 1)
    1000ULL * 1000 * 1000   // 1s for long-lived class (class 2)
};

int main() {
    printf("🔬 [test_misclassification] Starting misclassification test...\n");

    // Force an allocation into class 1
    void *block = je_malloc(ALLOC_SIZE);
    printf("🧠 [test] Allocated block at %p (expected lifespan class 1)\n", block);

    // Sleep for 1 second (>> 100ms deadline)
    printf("⏳ [test] Simulating long-lived allocation...\n");
    sleep(1); // COMMENT THIS OUT TO SEE BEHAVIOR

    // Free the allocation — this should be detected as a misclassification
    printf("🧹 [test] Freeing block...\n");
    je_free(block);

    // Give reclaimer thread time to run
    printf("⏳ [test] Sleeping to allow reclaimer to finish...\n");
    sleep(2);

    printf("✅ [test_misclassification] Done.\n");
    return 0;
}
