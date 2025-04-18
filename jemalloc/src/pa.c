#include "jemalloc/internal/jemalloc_preamble.h"
#include "jemalloc/internal/jemalloc_internal_includes.h"

#include "jemalloc/internal/san.h"
#include "jemalloc/internal/hpa.h"

#include "jemalloc/internal/background_thread_externs.h"
#include "jemalloc/internal/emap.h"
#include "jemalloc/internal/rtree.h"

#include <execinfo.h>   // for backtrace
#include <unistd.h>     // for getpid()
#include <stdint.h>     // for uint64_t

bool lifetime_ml_enabled = true;
bool generate_training_data = false;

size_t ml_total_frees = 0;
size_t ml_misclassified_frees = 0;


static uint64_t hash_stack_trace(void **buffer, int depth) {
    uint64_t hash = 5381;
    for (int i = 0; i < depth; ++i) {
        uintptr_t addr = (uintptr_t)buffer[i];
        hash = ((hash << 5) + hash) + addr; // djb2 hash
    }
    return hash;
}

static void
pa_nactive_add(pa_shard_t *shard, size_t add_pages) {
	atomic_fetch_add_zu(&shard->nactive, add_pages, ATOMIC_RELAXED);
}

static void
pa_nactive_sub(pa_shard_t *shard, size_t sub_pages) {
	assert(atomic_load_zu(&shard->nactive, ATOMIC_RELAXED) >= sub_pages);
	atomic_fetch_sub_zu(&shard->nactive, sub_pages, ATOMIC_RELAXED);
}

bool
pa_central_init(pa_central_t *central, base_t *base, bool hpa,
    hpa_hooks_t *hpa_hooks) {
	bool err;
	if (hpa) {
		err = hpa_central_init(&central->hpa, base, hpa_hooks);
		if (err) {
			return true;
		}
	}
	return false;
}

bool
pa_shard_init(tsdn_t *tsdn, pa_shard_t *shard, pa_central_t *central,
    emap_t *emap, base_t *base, unsigned ind, pa_shard_stats_t *stats,
    malloc_mutex_t *stats_mtx, nstime_t *cur_time,
    size_t pac_oversize_threshold, ssize_t dirty_decay_ms,
    ssize_t muzzy_decay_ms) {
	/* This will change eventually, but for now it should hold. */
	assert(base_ind_get(base) == ind);
	if (edata_cache_init(&shard->edata_cache, base)) {
		return true;
	}

	if (pac_init(tsdn, &shard->pac, base, emap, &shard->edata_cache,
	    cur_time, pac_oversize_threshold, dirty_decay_ms, muzzy_decay_ms,
	    &stats->pac_stats, stats_mtx)) {
		return true;
	}

	shard->ind = ind;

	shard->ever_used_hpa = false;
	atomic_store_b(&shard->use_hpa, false, ATOMIC_RELAXED);

	atomic_store_zu(&shard->nactive, 0, ATOMIC_RELAXED);

	shard->stats_mtx = stats_mtx;
	shard->stats = stats;
	memset(shard->stats, 0, sizeof(*shard->stats));

	shard->central = central;
	shard->emap = emap;
	shard->base = base;

	/* Initialize the lifespan reuse caches. */
	for (int i = 0; i < NUM_LIFESPAN_CLASSES; i++) {
		ecache_init(tsdn,
					&shard->lifespan_reuse[i],
					extent_state_retained,
					ind,    // arena index
					false); // delay_coalesce 
	}

	memset(shard->lifespan_blocks, 0, sizeof(shard->lifespan_blocks));

	return false;
}

bool
pa_shard_enable_hpa(tsdn_t *tsdn, pa_shard_t *shard,
    const hpa_shard_opts_t *hpa_opts, const sec_opts_t *hpa_sec_opts) {
	if (hpa_shard_init(&shard->hpa_shard, &shard->central->hpa, shard->emap,
	    shard->base, &shard->edata_cache, shard->ind, hpa_opts)) {
		return true;
	}
	if (sec_init(tsdn, &shard->hpa_sec, shard->base, &shard->hpa_shard.pai,
	    hpa_sec_opts)) {
		return true;
	}
	shard->ever_used_hpa = true;
	atomic_store_b(&shard->use_hpa, true, ATOMIC_RELAXED);

	return false;
}

void
pa_shard_disable_hpa(tsdn_t *tsdn, pa_shard_t *shard) {
	atomic_store_b(&shard->use_hpa, false, ATOMIC_RELAXED);
	if (shard->ever_used_hpa) {
		sec_disable(tsdn, &shard->hpa_sec);
		hpa_shard_disable(tsdn, &shard->hpa_shard);
	}
}

void
pa_shard_reset(tsdn_t *tsdn, pa_shard_t *shard) {
	atomic_store_zu(&shard->nactive, 0, ATOMIC_RELAXED);
	if (shard->ever_used_hpa) {
		sec_flush(tsdn, &shard->hpa_sec);
	}
}

static bool
pa_shard_uses_hpa(pa_shard_t *shard) {
	return atomic_load_b(&shard->use_hpa, ATOMIC_RELAXED);
}

void
pa_shard_destroy(tsdn_t *tsdn, pa_shard_t *shard) {
	pac_destroy(tsdn, &shard->pac);
	if (shard->ever_used_hpa) {
		sec_flush(tsdn, &shard->hpa_sec);
		hpa_shard_disable(tsdn, &shard->hpa_shard);
	}
}

static pai_t *
pa_get_pai(pa_shard_t *shard, edata_t *edata) {
	return (edata_pai_get(edata) == EXTENT_PAI_PAC
	    ? &shard->pac.pai : &shard->hpa_sec.pai);
}



static edata_t *
try_lifespan_block_alloc(tsdn_t *tsdn, pa_shard_t *shard,
                         uint8_t lifespan_class, size_t size,
                         size_t alignment, bool zero) {
	assert(lifespan_class < NUM_LIFESPAN_CLASSES);

	if (size > LIFESPAN_SLICE_SIZE) {
		printf("[jemalloc] ❌ Allocation size %zu exceeds fixed slice size %d\n",
		       size, LIFESPAN_SLICE_SIZE);
		return NULL;
	}

	lifespan_block_allocator_t *allocator = &shard->lifespan_blocks[lifespan_class];

	// First, try to find space in an existing block
	for (size_t b = 0; b < allocator->count; ++b) {
		lifespan_block_t *block = &allocator->blocks[b];

		edata_t *edata = block->block;
		if (edata == NULL) continue;

		size_t base = (size_t)edata_base_get(edata);
		size_t aligned_offset = ALIGNMENT_CEILING(block->offset, alignment);

		// Avoid emap collision
		if (aligned_offset == 0) {
			block->offset = aligned_offset + size;
			continue;
		}

		if (aligned_offset + size <= edata_size_get(edata)) {
			void *slice_addr = (void *)(base + aligned_offset);
			edata_t *slice = edata_cache_get(tsdn, &shard->edata_cache);
			if (slice == NULL) return NULL;

			edata_init(slice, shard->ind, slice_addr, size, /* paddings */ false,
			           EXTENT_PAI_PAC, extent_state_active,
			           zero, false, false, false, shard->ind);

			edata_lifespan_set(slice, lifespan_class);
			edata_set_initial_class(slice, lifespan_class);
			edata_mark_as_slice(slice);
			edata_slice_owner_set(slice, block);

			// Timestamp for slice
			struct timespec ts;
			clock_gettime(CLOCK_MONOTONIC, &ts);
			uint64_t slice_ts = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
			edata_lifespan_timestamp_set(slice, slice_ts);
			edata_state_set(slice, extent_state_active);

			assert(edata_state_get(slice) == extent_state_active);

			// emap_register_boundary(tsdn, shard->emap, slice, false, SC_NSIZES);

			block->offset = aligned_offset + size;
			block->live_slices++;

			for (size_t i = 0; i < MAX_SLICES_PER_BLOCK; ++i) {
				if (block->slices[i] == NULL) {
					block->slices[i] = slice;
					break;
				}
			}

			return slice;
		}
	}

	// No space in existing blocks — allocate new block
	if (allocator->count >= MAX_BLOCKS_PER_CLASS) {
		// printf("[jemalloc] Too many blocks in lifespan class %d\n", lifespan_class);
		return NULL;
	}

	edata_t *new_edata = pai_alloc(tsdn, &shard->pac.pai,
	                               HUGEPAGE_SIZE, HUGEPAGE_SIZE,
	                               zero, false, false, NULL);
	if (new_edata == NULL) return NULL;

	assert(((uintptr_t)edata_base_get(new_edata) & (HUGEPAGE_SIZE - 1)) == 0);
	edata_lifespan_set(new_edata, lifespan_class);
	edata_set_initial_class(new_edata, lifespan_class);
	edata_state_set(new_edata, extent_state_active);

	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	uint64_t ts_ns = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
	edata_lifespan_timestamp_set(new_edata, ts_ns);

	// Register new block
	lifespan_block_t *new_block = &allocator->blocks[allocator->count++];
	new_block->block = new_edata;
	new_block->offset = 0;
	nstime_init(&new_block->block_ts, ts_ns);
	new_block->live_slices = 0;
	memset(new_block->slices, 0, sizeof(new_block->slices));

	printf("[jemalloc] 🆕 Allocated new 2MB block for lifespan class %u at %p (ts = %lu ns)\n",
	       lifespan_class, edata_base_get(new_edata), ts_ns);

	// allocate from the new block directly
	size_t base = (size_t)edata_base_get(new_edata);
	size_t aligned_offset = ALIGNMENT_CEILING(0, alignment);  // offset is 0

	// Avoid emap collision at offset 0
	if (aligned_offset == 0) {
		aligned_offset += size;
	}

	if (aligned_offset + size > edata_size_get(new_edata)) {
		printf("[jemalloc] ❌ New block too small for requested allocation\n");
		return NULL;
	}

	void *slice_addr = (void *)(base + aligned_offset);
	edata_t *slice = edata_cache_get(tsdn, &shard->edata_cache);
	if (slice == NULL) return NULL;

	edata_init(slice, shard->ind, slice_addr, size, /* paddings */ false,
			EXTENT_PAI_PAC, extent_state_active,
			zero, false, false, false, shard->ind);

	edata_lifespan_set(slice, lifespan_class);
	edata_set_initial_class(slice, lifespan_class);
	edata_mark_as_slice(slice);
	edata_slice_owner_set(slice, new_block);
	edata_lifespan_timestamp_set(slice, ts_ns);
	edata_state_set(slice, extent_state_active);

	assert(edata_state_get(slice) == extent_state_active);

	emap_register_boundary(tsdn, shard->emap, slice, false, SC_NSIZES);

	new_block->offset = aligned_offset + size;
	new_block->live_slices++;
	new_block->slices[0] = slice;

	return slice;
}





// Periodically expire lifespan blocks that have exceeded their deadlines
void pa_expire_lifespan_blocks(tsdn_t *tsdn, pa_shard_t *shard) {
    if (shard == NULL) {
        printf("[jemalloc] [reclaimer] Skipping: shard is NULL\n");
        return;
    }

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t now_ns = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;

    // printf("\n================================\n");
    // printf("[jemalloc] Reclaimer running at %lu ns for shard = %p\n", now_ns, (void*)shard); 

    for (int class_id = NUM_LIFESPAN_CLASSES - 1; class_id >= 0; --class_id) {
        lifespan_block_allocator_t *allocator = &shard->lifespan_blocks[class_id];
        // printf("[jemalloc] [shard %p] Lifespan class %d has %d active blocks\n",
		// 	(void*)shard, class_id, allocator->count);

        for (ssize_t i = allocator->count - 1; i >= 0; --i) {
            lifespan_block_t *block = &allocator->blocks[i];
            edata_t *edata = block->block;
            if (edata == NULL) {
                continue;
            }

            uint64_t elapsed_ns = now_ns - nstime_ns(&block->block_ts);
            uint64_t deadline_ns = lifespan_class_deadlines_ns[class_id];

            // printf("[jemalloc] class %d block %zd age = %lu ns (deadline = %lu ns)\n",
            //        class_id, i, elapsed_ns, deadline_ns);

            if (elapsed_ns <= deadline_ns) {
                continue;
            }

            if (block->live_slices == 0) {
                // Reclaim block
                printf("[jemalloc] 🔥 Reclaiming expired block for class %d (block %zd)\n", class_id, i);

				// Invalidate slice list before freeing block
				for (size_t j = 0; j < MAX_SLICES_PER_BLOCK; ++j) {
					edata_t *slice = block->slices[j];
					if (slice != NULL && edata_is_marked_as_slice(slice)) {
						edata_slice_owner_set(slice, NULL);
						block->slices[j] = NULL;
					}
				}

				// Deregister from emap
				emap_deregister_boundary(tsdn, shard->emap, edata);

				// Return edata back to edata cache
				edata_cache_put(tsdn, &shard->edata_cache, edata);
				
				// extent_dalloc_wrapper(tsdn,
				// 	&shard->pac,
				// 	pa_shard_ehooks_get(shard),
				// 	edata);

                // Compact: move last block into current slot
                allocator->count--;
                if ((size_t)i != allocator->count) {
                    allocator->blocks[i] = allocator->blocks[allocator->count];
                }
                continue;
            }

            // Promote to next longer class if needed
            uint8_t block_lc = edata_lifespan_get(edata);
            if (block_lc + 1 < NUM_LIFESPAN_CLASSES) {
                uint8_t promoted_class = block_lc + 1;
                lifespan_block_allocator_t *dest_alloc = &shard->lifespan_blocks[promoted_class];

                if (dest_alloc->count >= MAX_BLOCKS_PER_CLASS) {
                    // printf("[jemalloc] Can't promote — class %u full\n", promoted_class);
                    continue;
                }

                // Move block
                size_t new_index = dest_alloc->count++;
                lifespan_block_t *new_block = &dest_alloc->blocks[new_index];
                new_block->block = edata;
                new_block->offset = block->offset;
                new_block->live_slices = block->live_slices;
                memcpy(new_block->slices, block->slices, sizeof(block->slices));

                // Promote slice metadata
                for (size_t j = 0; j < MAX_SLICES_PER_BLOCK; ++j) {
                    edata_t *slice = block->slices[j];
                    if (slice != NULL && edata_is_marked_as_slice(slice)) {
                        edata_lifespan_set(slice, promoted_class);
                        edata_slice_owner_set(slice, new_block);
                    }
                }

                // Update lifespan & timestamp
                edata_lifespan_set(edata, promoted_class);
                clock_gettime(CLOCK_MONOTONIC, &ts);
                uint64_t new_ts = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
                nstime_init(&new_block->block_ts, new_ts);

                // Compact: replace current block with last one
                allocator->count--;
                if ((size_t)i != allocator->count) {
                    allocator->blocks[i] = allocator->blocks[allocator->count];
                }

                // printf("[jemalloc] Promoted block from class %d → %d and set new deadline = %lu ns\n",
                    //    class_id, promoted_class, lifespan_class_deadlines_ns[promoted_class]);
                continue;
            } else {
                // printf("[jemalloc] Block already in longest LC (%d). Skipping.\n", class_id);
            }
        }
    }

    // printf("================================\n\n");
}



edata_t *
pa_alloc(tsdn_t *tsdn, pa_shard_t *shard, size_t size, size_t alignment,
    bool slab, szind_t szind, bool zero, bool guarded,
uint8_t lifespan_class,
    bool *deferred_work_generated) {
	witness_assert_depth_to_rank(tsdn_witness_tsdp_get(tsdn),
	    WITNESS_RANK_CORE, 0);
	assert(!guarded || alignment <= PAGE);

	edata_t *edata = NULL;

	/* Try slicing from lifespan block if not slab and lifespan is set */
	if (!slab && lifespan_class != EDATA_LIFETIME_DEFAULT) {
		// printf("[jemalloc] Trying lifespan block slice for class %u, size: %zu\n",
			// lifespan_class, size);

		edata = try_lifespan_block_alloc(tsdn, shard, lifespan_class, size, alignment, zero);

		if (edata != NULL) {
			// printf("[jemalloc] Reused slice from lifespan block class %u at %p\n",
				// lifespan_class, edata_base_get(edata));
			fflush(stdout);
		} else {
			// printf("[jemalloc] Slicing failed for class %u — falling back to reuse/ecache\n",
				// lifespan_class);
			fflush(stdout);
		}
	}

	if (!guarded && pa_shard_uses_hpa(shard)) {
		edata = pai_alloc(tsdn, &shard->hpa_sec.pai, size, alignment,
		    zero, /* guarded */ false, slab, deferred_work_generated);
	}
	/*
	 * Fall back to the PAC if the HPA is off or couldn't serve the given
	 * allocation request.
	 */
	if (edata == NULL) {
		edata = pai_alloc(tsdn, &shard->pac.pai, size, alignment, zero,
		    guarded, slab, deferred_work_generated);
	}
	if (edata != NULL) {
		assert(edata_size_get(edata) == size);
		pa_nactive_add(shard, size >> LG_PAGE);
		emap_remap(tsdn, shard->emap, edata, szind, slab);
		edata_szind_set(edata, szind);
		edata_slab_set(edata, slab);
		if (slab && (size > 2 * PAGE)) {
			emap_register_interior(tsdn, shard->emap, edata, szind);
		}
		assert(edata_arena_ind_get(edata) == shard->ind);

		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		uint64_t now_ns = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
		nstime_init(&edata->alloc_ts, now_ns);

		// Capture stack trace
		void *trace_buffer[32];
		int trace_depth = backtrace(trace_buffer, 32);
		uint64_t trace_hash = hash_stack_trace(trace_buffer, trace_depth);

		/* Logging for ML lifetime prediction */
		if (generate_training_data) {
		FILE *f = fopen("./tmp/alloc_metadata.log", "a");
			if (f != NULL) {
				fprintf(f, "%p %zu %u %lu %lu\n",
					edata_addr_get(edata),
					edata_size_get(edata),
					lifespan_class,
					(unsigned long)nstime_ns(&edata->alloc_ts),
					trace_hash);
				fclose(f);
			}
		}
	}
	return edata;
}

bool
pa_expand(tsdn_t *tsdn, pa_shard_t *shard, edata_t *edata, size_t old_size,
    size_t new_size, szind_t szind, bool zero, bool *deferred_work_generated) {
	assert(new_size > old_size);
	assert(edata_size_get(edata) == old_size);
	assert((new_size & PAGE_MASK) == 0);
	if (edata_guarded_get(edata)) {
		return true;
	}
	size_t expand_amount = new_size - old_size;

	pai_t *pai = pa_get_pai(shard, edata);

	bool error = pai_expand(tsdn, pai, edata, old_size, new_size, zero,
	    deferred_work_generated);
	if (error) {
		return true;
	}

	pa_nactive_add(shard, expand_amount >> LG_PAGE);
	edata_szind_set(edata, szind);
	emap_remap(tsdn, shard->emap, edata, szind, /* slab */ false);
	return false;
}

bool
pa_shrink(tsdn_t *tsdn, pa_shard_t *shard, edata_t *edata, size_t old_size,
    size_t new_size, szind_t szind, bool *deferred_work_generated) {
	assert(new_size < old_size);
	assert(edata_size_get(edata) == old_size);
	assert((new_size & PAGE_MASK) == 0);
	if (edata_guarded_get(edata)) {
		return true;
	}
	size_t shrink_amount = old_size - new_size;

	pai_t *pai = pa_get_pai(shard, edata);
	bool error = pai_shrink(tsdn, pai, edata, old_size, new_size,
	    deferred_work_generated);
	if (error) {
		return true;
	}
	pa_nactive_sub(shard, shrink_amount >> LG_PAGE);

	edata_szind_set(edata, szind);
	emap_remap(tsdn, shard->emap, edata, szind, /* slab */ false);
	return false;
}

void
pa_dalloc(tsdn_t *tsdn, pa_shard_t *shard, edata_t *edata,
    bool *deferred_work_generated) {
	emap_remap(tsdn, shard->emap, edata, SC_NSIZES, /* slab */ false);

	// Log deallocation event for ML training
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	uint64_t dealloc_ts = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;

	// Capture stack trace at deallocation
	void *trace_buffer[32];
	int trace_depth = backtrace(trace_buffer, 32);
	uint64_t trace_hash = hash_stack_trace(trace_buffer, trace_depth);

	// Log to dealloc metadata
	if (generate_training_data) {
	FILE *f = fopen("./tmp/dealloc_metadata.log", "a");
		if (f != NULL) {
			fprintf(f, "%p %zu %u %lu %lu %lu\n",
				edata_addr_get(edata),
				edata_size_get(edata),
				edata_lifespan_get(edata),
				(unsigned long)nstime_ns(&edata->alloc_ts),
				dealloc_ts,
				trace_hash);
			fclose(f);
		}
	}

	if (edata_slab_get(edata)) {
		emap_deregister_interior(tsdn, shard->emap, edata);
		/*
		 * The slab state of the extent isn't cleared.  It may be used
		 * by the pai implementation, e.g. to make caching decisions.
		 */
	}
	edata_addr_set(edata, edata_base_get(edata));
	edata_szind_set(edata, SC_NSIZES);
	pa_nactive_sub(shard, edata_size_get(edata) >> LG_PAGE);
	pai_t *pai = pa_get_pai(shard, edata);
	pai_dalloc(tsdn, pai, edata, deferred_work_generated);
}

bool
pa_shard_retain_grow_limit_get_set(tsdn_t *tsdn, pa_shard_t *shard,
    size_t *old_limit, size_t *new_limit) {
	return pac_retain_grow_limit_get_set(tsdn, &shard->pac, old_limit,
	    new_limit);
}

bool
pa_decay_ms_set(tsdn_t *tsdn, pa_shard_t *shard, extent_state_t state,
    ssize_t decay_ms, pac_purge_eagerness_t eagerness) {
	return pac_decay_ms_set(tsdn, &shard->pac, state, decay_ms, eagerness);
}

ssize_t
pa_decay_ms_get(pa_shard_t *shard, extent_state_t state) {
	return pac_decay_ms_get(&shard->pac, state);
}

void
pa_shard_set_deferral_allowed(tsdn_t *tsdn, pa_shard_t *shard,
    bool deferral_allowed) {
	if (pa_shard_uses_hpa(shard)) {
		hpa_shard_set_deferral_allowed(tsdn, &shard->hpa_shard,
		    deferral_allowed);
	}
}

void
pa_shard_do_deferred_work(tsdn_t *tsdn, pa_shard_t *shard) {
	if (pa_shard_uses_hpa(shard)) {
		hpa_shard_do_deferred_work(tsdn, &shard->hpa_shard);
	}
}

/*
 * Get time until next deferred work ought to happen. If there are multiple
 * things that have been deferred, this function calculates the time until
 * the soonest of those things.
 */
uint64_t
pa_shard_time_until_deferred_work(tsdn_t *tsdn, pa_shard_t *shard) {
	uint64_t time = pai_time_until_deferred_work(tsdn, &shard->pac.pai);
	if (time == BACKGROUND_THREAD_DEFERRED_MIN) {
		return time;
	}

	if (pa_shard_uses_hpa(shard)) {
		uint64_t hpa =
		    pai_time_until_deferred_work(tsdn, &shard->hpa_shard.pai);
		if (hpa < time) {
			time = hpa;
		}
	}
	return time;
}
