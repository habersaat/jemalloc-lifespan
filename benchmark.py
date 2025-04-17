import torch
import torch.nn as nn
import torch.optim as optim
import ctypes

# ============================
# 🔧 jemalloc stat utilities
# ============================

# Load jemalloc shared library
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

# ============================
# 📦 Define simple CNN
# ============================

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 64 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # (32, 16, 64, 64)
        x = torch.relu(self.conv2(x))  # (32, 32, 64, 64)
        x = x.view(x.size(0), -1)      # flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ============================
# 🧪 Setup training
# ============================

model = CustomCNN()
dummy_input = torch.randn(32, 3, 64, 64)  # ~1.5MB per batch

optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# ============================
# 🚀 Benchmark
# ============================

print_all_stats("Before training")

for step in range(500):
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = criterion(output, torch.randint(0, 10, (32,)))
    loss.backward()
    optimizer.step()

print_all_stats("After training")
