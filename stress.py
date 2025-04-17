import torch
import time

# Matches your class deadlines
lifespan_classes_ns = [
    1e6,       # 1ms
    10e6,      # 10ms
    100e6,     # 100ms
    500e6,     # 500ms
    2e9,       # 2s
    5e9,       # 5s
    10e9       # 10s
]

# Time to hold allocations, just over the class deadlines
hold_times = [d / 1e9 + 0.1 for d in lifespan_classes_ns]  # seconds

SLICE_KB = 128
NUM_ALLOCS_PER_CLASS = 12  # More than MAX_BLOCKS_PER_CLASS * slices/block
NUM_CLASSES = len(hold_times)

print("🚀 Starting full-class expiration benchmark...")

for cycle in range(3):  # Repeat to test reuse and cleanup
    print(f"\n=== Cycle {cycle + 1} ===")
    for class_id, hold in enumerate(hold_times):
        print(f"📦 Allocating class {class_id} slices for {hold:.2f}s")

        allocs = []
        for i in range(NUM_ALLOCS_PER_CLASS):
            # 128KB per tensor → ~16 slices per block (for 256KB slice limit)
            tensor = torch.empty((SLICE_KB * 1024) // 4, dtype=torch.float32)
            tensor.uniform_()
            allocs.append(tensor)

        time.sleep(hold)
        del allocs  # Trigger deallocation for all at once

        print(f"🧹 Freed class {class_id} allocs — waiting 0.5s for reclaimer\n")
        time.sleep(0.5)

print("✅ Done! Sleeping to observe lingering reclaims...")
time.sleep(5)
