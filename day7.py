import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ======================
# 1. Data（關鍵：標準化）
# ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

testset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# ======================
# 2. CNN（穩定98%架構🔥）
# ======================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),   # 28→26
            nn.ReLU(),
            nn.MaxPool2d(2),       # 26→13

            nn.Conv2d(32, 64, 3),  # 13→11
            nn.ReLU(),
            nn.MaxPool2d(2)        # 11→5
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*5*5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = CNN()

# ======================
# 3. Loss + Optimizer（標準）
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ======================
# 4. Accuracy function🔥
# ======================
def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc

# ======================
# 5. Training
# ======================
for epoch in range(30):  # 🔥 7 epoch 最穩

    model.train()
    running_loss = 0

    for images, labels in trainloader:

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    acc = evaluate()

    print(f"epoch {epoch}")
    print("loss:", running_loss / len(trainloader))
    print("accuracy:", acc, "%")
    print("------")

# ======================
# 6. Save model
# ======================
torch.save(model.state_dict(), "mnist_cnn.pth")