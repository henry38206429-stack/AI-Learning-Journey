import torch
import torch.nn as nn
import torch.optim as optim

# ======================
# 1. 資料（改成分類問題）
# ======================
x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

y = torch.tensor([
    [0.0],
    [0.0],
    [1.0],
    [1.0]
])

# ======================
# 2. 多層神經網路（重點）
# ======================
model = nn.Sequential(
    nn.Linear(1, 8),   # 隱藏層
    nn.ReLU(),         # 非線性
    nn.Linear(8, 1)    # 輸出層
)

# ======================
# 3. loss
# ======================
criterion = nn.MSELoss()

# ======================
# 4. optimizer
# ======================
optimizer = optim.SGD(model.parameters(), lr=0.1)

# ======================
# 5. 訓練
# ======================
for epoch in range(50):

    # forward
    y_pred = model(x)

    # loss
    loss = criterion(y_pred, y)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"epoch {epoch}")
    print("loss:", loss.item())