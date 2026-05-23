from flask import Flask, render_template, request
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)

# CNN（跟訓練時完全一致）
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
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

model= CNN()
model.load_state_dict(
    torch.load("mnist_cnn.pth",map_location="cpu")
)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route("/", methods=["GET","POST"])
def home():

    result=""

    if request.method=="POST":

        file=request.files["image"]

        img=Image.open(file)
        img=transform(img)
        img=img.unsqueeze(0)

        with torch.no_grad():

            output=model(img)
            pred=torch.argmax(output,dim=1)

        result=pred.item()

    return render_template(
        "index.html",
        result=result
    )

app.run(debug=True)