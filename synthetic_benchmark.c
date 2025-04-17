/* mlsim_benchmark_v2.c –– extended ML‑like lifetime generator */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include <jemalloc/jemalloc.h>

/* ---------- tunables ---------- */
#define EPOCHS         20     /* how many forward / backward cycles */
#define SHORT_PER_EP   800    /* per‑epoch short‑lived bufs   (64–128 KB) */
#define MID_PER_EP     400    /* per‑epoch mid‑lived bufs     (128–192 KB) */
#define LONG_PER_EP    120    /* per‑epoch long‑lived bufs    (192–256 KB) */

#define BURST_THREADS  4      /* small burst to mix threads   */
#define BURST_ALLOCS   3000   /* per thread, 64–96 KB each    */

#define MIN_ALLOC 65536       /* 64 KB  */
#define MAX_ALLOC 262144      /* 256 KB */

typedef struct { void *ptr; size_t sz; int freed; } rec_t;

/* ---------- helpers ---------- */
static inline size_t rnd_sz(size_t lo, size_t hi) {
    return (rand() % (hi - lo + 1)) + lo;
}
static inline void msleep(int ms) { usleep(ms * 1000); }

/* ---------- burst thread ---------- */
void* burst_thread(void *arg) {
    (void)arg;
    for (int i = 0; i < BURST_ALLOCS; i++) {
        size_t sz = rnd_sz(65536, 98304);    /* 64–96 KB */
        void *p   = je_malloc(sz);
        if (p) {
            memset(p, 0xEE, sz);
            je_free(p);
        }
    }
    return NULL;
}

/* ---------- main ---------- */
int main(void) {
    puts("🧠  ML‑like lifetime stress v2 starting …");
    srand((unsigned)time(NULL));

    /* long‑lived parameters survive across epochs */
    rec_t long_live_pool[LONG_PER_EP * EPOCHS];
    int   long_pool_len = 0;

    for (int ep = 0; ep < EPOCHS; ep++) {
        printf("\n🔄 Epoch %d / %d\n", ep + 1, EPOCHS);

        /* ---------- Phase 1 : forward (short) ---------- */
        rec_t short_bufs[SHORT_PER_EP] = {0};
        for (int i = 0; i < SHORT_PER_EP; i++) {
            size_t sz = rnd_sz(65536, 131072);          /* 64–128 KB */
            void *p   = je_malloc(sz);
            if (p) { short_bufs[i].ptr = p; short_bufs[i].sz = sz; }
            /* chance to free an older short early */
            if (i > 30 && (rand() & 7) == 0) {
                int j = rand() % i;
                if (short_bufs[j].ptr && !short_bufs[j].freed) {
                    je_free(short_bufs[j].ptr);
                    short_bufs[j].freed = 1;
                }
            }
            msleep(2);
        }

        /* ---------- Phase 2 : mid‑lived tensors ---------- */
        rec_t mid_bufs[MID_PER_EP] = {0};
        for (int i = 0; i < MID_PER_EP; i++) {
            size_t sz = rnd_sz(131072, 196608);         /* 128–192 KB */
            void *p   = je_malloc(sz);
            if (p) { mid_bufs[i].ptr = p; mid_bufs[i].sz = sz; }
            msleep(1);
        }

        /* ---------- Phase 3 : new long‑lived params ---------- */
        for (int i = 0; i < LONG_PER_EP; i++) {
            size_t sz = rnd_sz(196608, 262144);         /* 192–256 KB */
            void *p   = je_malloc(sz);
            if (p) {
                memset(p, 0xCC, sz);
                long_live_pool[long_pool_len].ptr = p;
                long_live_pool[long_pool_len].sz  = sz;
                long_pool_len++;
            }
        }

        /* ---------- Phase 4 : free short + mid ---------- */
        for (int i = 0; i < SHORT_PER_EP; i++)
            if (short_bufs[i].ptr && !short_bufs[i].freed) je_free(short_bufs[i].ptr);
        for (int i = 0; i < MID_PER_EP;   i++)
            if (mid_bufs[i].ptr) je_free(mid_bufs[i].ptr);

        /* give allocator time to react */
        msleep(10);
    }

    /* ---------- small multi‑threaded reuse burst ---------- */
    puts("\n⚡  Threaded reuse burst");
    pthread_t th[BURST_THREADS];
    for (int t = 0; t < BURST_THREADS; t++) pthread_create(&th[t], NULL, burst_thread, NULL);
    for (int t = 0; t < BURST_THREADS; t++) pthread_join(th[t], NULL);

    /* ---------- final cleanup ---------- */
    puts("\n📤  Freeing long‑lived params");
    for (int i = 0; i < long_pool_len; i++)
        if (long_live_pool[i].ptr) je_free(long_live_pool[i].ptr);

    puts("\n✅  Benchmark complete – happy training!");
    return 0;
}
