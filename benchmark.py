import ctypes
import time
import random
import sys

# ============================
# 🔧 jemalloc stat utilities
# ============================

try:
    jemalloc = ctypes.CDLL("./jemalloc/lib/libjemalloc.so", use_errno=True)
except OSError as e:
    print(f"[ERROR] Failed to load jemalloc: {e}")
    sys.exit(1)

def get_stat(stat_name):
    value = ctypes.c_size_t()
    size = ctypes.c_size_t(ctypes.sizeof(value))
    ret = jemalloc.mallctl(stat_name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0)
    if ret != 0:
        raise RuntimeError(f"mallctl failed for {stat_name}, ret={ret}")
    return value.value

def print_all_stats(label):
    print(f"\n🧠 jemalloc stats at: {label}")
    for stat in ["stats.allocated", "stats.active", "stats.resident"]:
        try:
            val = get_stat(stat)
            print(f"  {stat:<18} = {val:>10} bytes")
        except Exception as e:
            print(f"  {stat:<18} = [error: {e}]")

# ============================
# 🚀 Allocation benchmark
# ============================

NUM_ALLOC = 1000
MIN_ALLOC = 64 * 1024
MAX_ALLOC = 256 * 1024
MAX_LIFETIME_MS = 2000

print_all_stats("Start")

allocs = []

print(f"\n🚀 Starting {NUM_ALLOC} randomized allocations")
for i in range(NUM_ALLOC):
    size = random.randint(MIN_ALLOC, MAX_ALLOC)
    lifetime_ms = random.choice([10, 50, 100, 250, 500, 1000, 2000])
    ptr = ctypes.cast(jemalloc.je_malloc(size), ctypes.c_void_p)
    if not ptr:
        print(f"[{i}] ❌ Allocation failed")
        continue
    allocs.append((ptr, size))
    print(f"[{i}] ✅ Allocated {size} bytes for {lifetime_ms}ms at {hex(ptr.value)}")
    time.sleep(lifetime_ms / 1000.0)
    jemalloc.je_free(ptr)
    print(f"[{i}] 🧹 Freed {hex(ptr.value)}")

print_all_stats("End")
