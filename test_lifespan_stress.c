#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <jemalloc/jemalloc.h>

#define ALLOC_COUNT 50
#define MIN_ALLOC 65536
#define MAX_ALLOC 81920

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
    return (rand() % 400);  // 0–400ms
}

void stress_sleep(int ms) {
    usleep(ms * 1000);
}

int main() {
    printf("🔬 [stress] Starting lifespan stress test...\n");
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

        // After some delay, randomly free a previous one
        if (i > 5 && (rand() % 3 == 0)) {
            int j = rand() % i;
            if (!allocs[j].freed) {
                printf("🧹 Freeing alloc[%d] = %p early\n", j, allocs[j].ptr);
                je_free(allocs[j].ptr);
                allocs[j].freed = 1;
            }
        }

        stress_sleep(50);
    }

    // Phase 2: Free remaining allocations
    for (int i = 0; i < ALLOC_COUNT; i++) {
        if (!allocs[i].freed && allocs[i].ptr != NULL) {
            printf("🧹 Final free of alloc[%d] = %p\n", i, allocs[i].ptr);
            je_free(allocs[i].ptr);
            allocs[i].freed = 1;
            stress_sleep(10);
        }
    }

    // Wait to allow promotion/expiration logic to kick in
    printf("⏳ Sleeping 3s to observe promotions...\n");
    sleep(3);

    // Phase 3: Allocate again to test reuse + correctness
    printf("♻️ [reuse] Allocating again to test reuse cache...\n");
    for (int i = 0; i < 10; i++) {
        size_t size = random_size();
        void *ptr = je_malloc(size);
        if (ptr) {
            memset(ptr, 0xCD, size);
            printf("🔁 Alloc[%d] = %p (size = %zu)\n", i, ptr, size);
            je_free(ptr);
        }
        stress_sleep(30);
    }

    printf("✅ [stress] Lifespan stress test complete.\n");
    return 0;
}
