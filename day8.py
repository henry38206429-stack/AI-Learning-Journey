import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*5*5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

img = Image.open("test.png")
img = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img)
    pred = torch.argmax(output, dim=1)

print("AI 預測:", pred.item())