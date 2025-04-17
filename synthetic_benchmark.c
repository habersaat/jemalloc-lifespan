#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <jemalloc/jemalloc.h>

#define ALLOC_COUNT 1000
#define MIN_ALLOC 65536
#define MAX_ALLOC 262144

typedef struct {
    void *ptr;
    size_t size;
    int delay_ms;
    int freed;
} alloc_record;

size_t random_size() {
    return (rand() % (MAX_ALLOC - MIN_ALLOC + 1)) + MIN_ALLOC;
}

int random_delay() {
    return (rand() % 200);  // 0–200ms
}

void stress_sleep(int ms) {
    usleep(ms * 1000);
}

int main() {
    printf("🔬 [stress] Starting extended lifespan stress test...\n");
    srand(time(NULL));

    alloc_record allocs[ALLOC_COUNT] = {0};

    // Phase 1: Interleaved allocations + opportunistic frees
    for (int i = 0; i < ALLOC_COUNT; i++) {
        allocs[i].size = random_size();
        allocs[i].delay_ms = random_delay();
        allocs[i].freed = 0;

        allocs[i].ptr = je_malloc(allocs[i].size);
        if (!allocs[i].ptr) {
            fprintf(stderr, "❌ Failed to allocate index %d\n", i);
            continue;
        }

        memset(allocs[i].ptr, 0xAB, allocs[i].size);
        printf("✅ Alloc[%d] = %p (size = %zu, delay = %dms)\n",
               i, allocs[i].ptr, allocs[i].size, allocs[i].delay_ms);

        // Occasionally free earlier allocs
        if (i > 20 && (rand() % 4 == 0)) {
            int j = rand() % i;
            if (!allocs[j].freed) {
                printf("🧹 Freeing alloc[%d] = %p early\n", j, allocs[j].ptr);
                je_free(allocs[j].ptr);
                allocs[j].freed = 1;
            }
        }

        stress_sleep(10);  // faster than original
    }

    // Phase 2: Final frees
    for (int i = 0; i < ALLOC_COUNT; i++) {
        if (!allocs[i].freed && allocs[i].ptr != NULL) {
            printf("🧹 Final free of alloc[%d] = %p\n", i, allocs[i].ptr);
            je_free(allocs[i].ptr);
            allocs[i].freed = 1;
            stress_sleep(1);  // minimal delay
        }
    }

    // Phase 3: Wait to observe reclamation + reuse
    printf("⏳ Sleeping 5s to observe promotions & reclamation...\n");
    sleep(5);

    // Phase 4: Reuse-phase
    printf("♻️ [reuse] Allocating again to test reuse cache...\n");
    for (int i = 0; i < 100; i++) {
        size_t size = random_size();
        void *ptr = je_malloc(size);
        if (ptr) {
            memset(ptr, 0xCD, size);
            printf("🔁 Alloc[%d] = %p (size = %zu)\n", i, ptr, size);
            je_free(ptr);
        }
        stress_sleep(5);
    }

    printf("✅ [stress] Extended lifespan stress test complete.\n");
    return 0;
}
