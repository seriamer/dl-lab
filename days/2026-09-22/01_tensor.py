"""9-22 demo 1：张量。跑完看打印，对照 d2l「数据操作」。"""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

# 创建
x = torch.arange(16, dtype=torch.float32)
print("arange 16:", x, "shape", tuple(x.shape))

X = x.reshape(4, 4)
print("reshape (4,4):\n", X)

# 切片：Python 一样，左闭右开
print("第 0 行:", X[0])
print("前两行、后3列:\n", X[:2, 1:])

# 逐元素 vs 矩阵乘
A = torch.ones(6, 8)
B = torch.arange(48, dtype=torch.float32).reshape(6, -1)
print("A * B 逐元素:\n", A * B)
print("A.T @ B  矩阵乘 (6,8)@(6,8) -> (8,8):\n", A.T @ B)

# 广播：(3,1) 和 (1,4) -> (3,4)
a = torch.arange(3).reshape(3, 1)
b = torch.arange(4).reshape(1, 4)
print("broadcast (3,1)+(1,4):\n", a + b)

# 放到 GPU。训练时数据和模型必须同一设备
x_gpu = torch.randn(1024, 1024, device=DEVICE)
y_gpu = x_gpu @ x_gpu
print("cuda matmul ok, y[0,0] =", float(y_gpu[0, 0]))
