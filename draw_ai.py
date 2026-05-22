import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import tkinter as tk

# ======================
# 1. CNN（必須跟你訓練的一樣）
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
# 3. 圖片轉換（超重要🔥）
# ======================
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ======================
# 4. 畫板設定
# ======================
canvas_size = 200

window = tk.Tk()
window.title("AI 手寫辨識")

canvas = tk.Canvas(window, width=canvas_size, height=canvas_size, bg="white")
canvas.pack()

img = Image.new("L", (canvas_size, canvas_size), "white")
draw = ImageDraw.Draw(img)

# ======================
# 5. 畫筆功能
# ======================
def paint(event):
    x, y = event.x, event.y
    r = 8

    canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
    draw.ellipse([x-r, y-r, x+r, y+r], fill="black")

canvas.bind("<B1-Motion>", paint)
img = img.point(lambda x: 255 - x)
from PIL import ImageOps

img = ImageOps.invert(img)
img = ImageOps.autocontrast(img)
img = img.crop(img.getbbox())
img = img.resize((28, 28))
# ======================
# 6. 預測
# ======================
def predict():
    global img

    tmp = img.copy()

    # 🔥 1. 反色（最重要）
    tmp = ImageOps.invert(tmp)

    # 🔥 2. 自動對比
    tmp = ImageOps.autocontrast(tmp)

    # 🔥 3. 轉 tensor
    img_tensor = transform(tmp).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()

    result_label.config(text=f"AI 預測：{pred}")

# ======================
# 7. 清除畫布
# ======================
def clear():
    global img, draw

    canvas.delete("all")
    img = Image.new("L", (canvas_size, canvas_size), "white")
    draw = ImageDraw.Draw(img)

# ======================
# 8. UI 按鈕
# ======================
btn_predict = tk.Button(window, text="預測", command=predict)
btn_predict.pack()

btn_clear = tk.Button(window, text="清除", command=clear)
btn_clear.pack()

result_label = tk.Label(window, text="AI 預測：")
result_label.pack()

window.mainloop()