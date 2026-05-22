import numpy as np

# 真實資料
x = np.array([1, 2, 3])
true_w = np.array([0.3, 0.5, 0.2])

# AI 初始亂猜
w = np.array([0.1, 0.1, 0.1])

# 學習率（AI 每次修正的步伐）
lr = 0.1

for i in range(10):

    # 預測
    y_pred = np.dot(x, w)

    # 真實答案
    y_true = np.dot(x, true_w)

    # 誤差
    error = y_true - y_pred

    # loss（觀察用）
    loss = error ** 2

    # 🔥 核心：更新 w
    w = w + lr * error * x

    print(f"step {i}")
    print("w:", w)
    print("loss:", loss)
    print("------")