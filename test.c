#include <jemalloc/jemalloc.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NUM_LIFESPAN_CLASSES 3
#define NUM_ALLOCS_PER_CLASS 10
#define ALLOC_LOG_PATH "tmp/alloc_classes.log"

#define HUGEPAGE_SIZE (2 * 1024 * 1024)

int main() {
    // Clear log
    FILE *clear = fopen(ALLOC_LOG_PATH, "w");
    if (clear) fclose(clear);

    // Trigger allocations
    void *ptrs[NUM_LIFESPAN_CLASSES * NUM_ALLOCS_PER_CLASS];
    size_t alloc_size = 64 * 1024;

    for (int i = 0; i < NUM_LIFESPAN_CLASSES * NUM_ALLOCS_PER_CLASS; ++i) {
        ptrs[i] = je_malloc(alloc_size);
    }

    // Parse log file (written to only temporarily)
    FILE *log = fopen(ALLOC_LOG_PATH, "r");
    if (!log) {
        perror("Failed to open log file");
        return 1;
    }

    uintptr_t allocs[NUM_LIFESPAN_CLASSES][NUM_ALLOCS_PER_CLASS] = {0};
    int counts[NUM_LIFESPAN_CLASSES] = {0};

    uintptr_t addr;
    unsigned cls;
    while (fscanf(log, "%p %u\n", (void **)&addr, &cls) == 2) {
        if (cls < NUM_LIFESPAN_CLASSES && counts[cls] < NUM_ALLOCS_PER_CLASS) {
            allocs[cls][counts[cls]++] = addr;
        }
    }

    fclose(log);

    printf("\n ====== Lifespan Class Allocation Summary ======\n");
    for (int i = 0; i < NUM_LIFESPAN_CLASSES; ++i) {
        printf("  • Class %d: %d allocations\n", i, counts[i]);
    }
    printf("================================================\n");

    printf("\n✅ Verifying if allocations within each class fall in same hugepage:\n");
    uintptr_t base_addrs[NUM_LIFESPAN_CLASSES] = {0};

    for (int cls = 0; cls < NUM_LIFESPAN_CLASSES; ++cls) {
        if (counts[cls] == 0) {
            printf("❌ No allocations recorded for class %d\n", cls);
            continue;
        }

        uintptr_t base = allocs[cls][0] & ~(HUGEPAGE_SIZE - 1);
        base_addrs[cls] = base;

        int all_same = 1;
        for (int i = 1; i < counts[cls]; ++i) {
            if ((allocs[cls][i] & ~(HUGEPAGE_SIZE - 1)) != base) {
                all_same = 0;
                break;
            }
        }

        if (all_same) {
            printf("✅ Class %d allocations are within same 2MB block: %p\n", cls, (void *)base);
        } else {
            printf("❌ Class %d allocations span multiple huge pages\n", cls);
        }
    }

    printf("\n Verifying that each lifespan class uses a disjoint 2MB block...\n");
    int disjoint = 1;
    for (int i = 0; i < NUM_LIFESPAN_CLASSES; ++i) {
        for (int j = i + 1; j < NUM_LIFESPAN_CLASSES; ++j) {
            if (base_addrs[i] != 0 && base_addrs[i] == base_addrs[j]) {
                disjoint = 0;
                printf("❌ Class %d and Class %d share the same 2MB block: %p\n",
                       i, j, (void *)base_addrs[i]);
            }
        }
    }

    if (disjoint) {
        printf("✅ Lifespan classes use disjoint 2MB blocks\n\n");
    }

    // Free allocations
    for (int i = 0; i < NUM_LIFESPAN_CLASSES * NUM_ALLOCS_PER_CLASS; ++i) {
        je_free(ptrs[i]);
    }

    return 0;
}
