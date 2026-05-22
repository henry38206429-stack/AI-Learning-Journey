import numpy as np

# 真實規則（我們不知道）
x = np.array([1, 2, 3])
true_w = np.array([0.3, 0.5, 0.2])

# AI 一開始亂猜
w = np.array([0.1, 0.1, 0.1])

# 真實答案
y_true = np.dot(x, true_w)

# AI 預測
y_pred = np.dot(x, w)

# 誤差（loss）
loss = (y_true - y_pred) ** 2

print("true:", y_true)
print("pred:", y_pred)
print("loss:", loss)