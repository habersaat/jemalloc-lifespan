import torch
import time
import random

def allocate_and_free_tensor(size_kb, hold_time_s):
    size_bytes = size_kb * 1024
    num_floats = size_bytes // 4  # float32 = 4 bytes
    tensor = torch.empty(num_floats, dtype=torch.float32)
    tensor.uniform_()  # Touch memory
    time.sleep(hold_time_s)
    return None  # Let GC free it

def run_lifespan_allocation_stress():
    print("🧪 Starting lifespan-aware stress benchmark")

    # These durations match lifespan class deadlines (all < 10s)
    durations = [0.002, 0.008, 0.05, 0.4, 1.5, 3, 6]

    for i in range(200):
        size_kb = random.choice([64, 128, 192, 256])
        duration = random.choice(durations)
        print(f"[{i:03}] Allocating {size_kb}KB for {duration:.3f}s")
        allocate_and_free_tensor(size_kb, duration)

    print("✅ Done with stress run — sleeping to allow reclamation...")
    time.sleep(5)

if __name__ == "__main__":
    run_lifespan_allocation_stress()
