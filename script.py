import torch
import torch.nn as nn
import torch.optim as optim

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Each conv output is in the 100–200KB range
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)  # large input, few channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 64 * 64, 1024)  # intermediate tensor ~128KB+
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # 16x64x64
        x = torch.relu(self.conv2(x))  # 32x64x64
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Create dummy input of shape (32, 3, 64, 64) ~1.5MB per batch
model = CustomCNN()
dummy_input = torch.randn(32, 3, 64, 64)

optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for step in range(500):  # Small training loop
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = criterion(output, torch.randint(0, 10, (32,)))
    loss.backward()
    optimizer.step()
