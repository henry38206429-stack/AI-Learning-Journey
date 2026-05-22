import torch
import torch.nn as nn
import torch.optim as optim

# ======================
# 1. 資料（跟 Day4 一樣）
# ======================
x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[0.3], [0.5], [0.2]])

# ======================
# 2. 神經網路模型
# ======================
model = nn.Linear(1, 1)

# ======================
# 3. loss function
# ======================
criterion = nn.MSELoss()

# ======================
# 4. optimizer（自動更新 w）
# ======================
optimizer = optim.SGD(model.parameters(), lr=0.1)

# ======================
# 5. 訓練
# ======================
for epoch in range(20):

    # 預測
    y_pred = model(x)

    # loss
    loss = criterion(y_pred, y)

    # 清空梯度
    optimizer.zero_grad()

    # 反向傳播（AI學習核心）
    loss.backward()

    # 更新權重
    optimizer.step()

    print(f"epoch {epoch}")
    print("loss:", loss.item())
    print("w:", model.weight.item())
    print("b:", model.bias.item())
    print("------")