from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import base64
import io
import os

app = Flask(__name__)

# CNN（和訓練時完全一致）
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv=nn.Sequential(
            nn.Conv2d(1,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*5*5,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        return self.fc(self.conv(x))

model=CNN()

model.load_state_dict(
    torch.load(
        "mnist_cnn.pth",
        map_location="cpu"
    )
)

model.eval()

transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:
        print("🔥 REQUEST OK")

        data = request.json["image"]

        print("🔥 got image")

        data = data.split(",")[1]

        image = Image.open(io.BytesIO(base64.b64decode(data))).convert("L")

        print("🔥 image decoded")

        image = transform(image)
        image = image.unsqueeze(0)

        print("🔥 tensor ready:", image.shape)

        with torch.no_grad():
            output = model(image)
            pred = torch.argmax(output, dim=1).item()

        print("🔥 prediction:", pred)

        return jsonify({"prediction": pred})

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)