#include <jemalloc/jemalloc.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#define NUM_LIFESPAN_CLASSES 3
#define ALLOC_SIZE (64 * 1024)
#define SHORT_CLASS 0
#define MID_CLASS 1
#define LONG_CLASS 2

void sleep_ms(int ms) {
    struct timespec req = {
        .tv_sec = ms / 1000,
        .tv_nsec = (ms % 1000) * 1000000
    };
    nanosleep(&req, NULL);
}

// External linkage to modify class deadlines at runtime
uint64_t lifespan_class_deadlines_ns[NUM_LIFESPAN_CLASSES] = {
    10ULL * 1000 * 1000,    // 10ms for short-lived
    100ULL * 1000 * 1000,   // 100ms for medium
    1000ULL * 1000 * 1000   // 1s for long-lived
};

int main() {
    printf("🔬 [test] Starting promotion test for lifespan class 0...\n");

    // class 1, 100ms
    void *alloc = je_malloc(ALLOC_SIZE);
    printf("🔧 [test] Allocated object at %p (intended for short-lived class)\n", alloc);

    // Sleep past the deadline to simulate long-lived allocation
    printf("⏳ [test] Sleeping for 1 second (longer than class 0 deadline)...\n");
    sleep(1); // Simulate long-lived allocation

    je_free(alloc);
    printf("🧹 [test] Freed object after sleeping\n");

    // Allow background reclaimer to run and possibly promote block
    printf("⏳ [test] Waiting 3 more seconds to observe block promotion...\n");
    sleep(3);

    printf("✅ [test] Done.\n");
    return 0;
}
