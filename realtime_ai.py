import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageOps
import tkinter as tk

# ======================
# 1. CNN（必須一致）
# ======================
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

# ======================
# 2. 載入模型
# ======================
model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location="cpu"))
model.eval()

# ======================
# 3. transform
# ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ======================
# 4. 畫板
# ======================
size = 200

window = tk.Tk()
window.title("Real-time AI Handwriting")

canvas = tk.Canvas(window, width=size, height=size, bg="white")
canvas.pack()

img = Image.new("L", (size, size), "white")
draw = ImageDraw.Draw(img)

# ======================
# 5. 畫筆
# ======================
def paint(event):
    x, y = event.x, event.y
    r = 10

    canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
    draw.ellipse([x-r, y-r, x+r, y+r], fill="black")

    predict()  # 🔥 每畫一筆就預測

canvas.bind("<B1-Motion>", paint)

# ======================
# 6. 預測（即時🔥）
# ======================
def predict():
    global img

    tmp = img.copy()

    # 🔥 修正（超重要）
    tmp = ImageOps.invert(tmp)

    tmp = tmp.resize((28, 28))

    x = transform(tmp).unsqueeze(0)

    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1).item()

    label.config(text=f"AI：{pred}")

# ======================
# 7. 清除
# ======================
def clear():
    global img, draw

    canvas.delete("all")
    img = Image.new("L", (size, size), "white")
    draw = ImageDraw.Draw(img)
    label.config(text="AI：")

# ======================
# 8. UI
# ======================
btn = tk.Button(window, text="清除", command=clear)
btn.pack()

label = tk.Label(window, text="AI：")
label.pack()

window.mainloop()