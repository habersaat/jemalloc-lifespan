import torch
import ctypes
import time
import random
import os

# ====================================
# 🔧 jemalloc stat utilities
# ====================================

try:
    jemalloc = ctypes.CDLL("libjemalloc.so", use_errno=True)

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
            print(f"  {stat:<18} = {get_stat(stat):>10} bytes")

except OSError:
    jemalloc = None
    def print_all_stats(label):
        print(f"\n(jemalloc stats unavailable at: {label})")

# ====================================
# 📦 Allocation Helper
# ====================================

allocated = []

def allocate_tensor(min_kb=64, max_kb=256):
    size_kb = random.randint(min_kb, max_kb)
    elements = (size_kb * 1024) // 4  # float32
    t = torch.zeros(elements, dtype=torch.float32)
    allocated.append(t)
    return t

# ====================================
# 🧪 Main Simulation
# ====================================

print_all_stats("Start")

start_time = time.time()

# Phase 1: Burst of short-lived tensors (simulate forward pass)
for _ in range(1000):
    t = allocate_tensor(64, 128)
    if random.random() < 0.5:
        allocated.pop()  # free early
    time.sleep(random.uniform(0.0005, 0.001))  # 0.5–1ms delay

# Phase 2: Medium-lived tensors (simulate intermediate features)
mid_tensors = [allocate_tensor(128, 192) for _ in range(500)]
time.sleep(0.3)
del mid_tensors

# Phase 3: Long-lived tensors (simulate parameters or gradients)
long_tensors = [allocate_tensor(192, 256) for _ in range(100)]
time.sleep(1.0)
del long_tensors

# Phase 4: Repeated reuse-like burst
for _ in range(1000):
    t = allocate_tensor(64, 128)
    time.sleep(random.uniform(0.0005, 0.001))

end_time = time.time()

print_all_stats("End")

print(f"\n✅ Simulated workload completed in {end_time - start_time:.2f} seconds.")
