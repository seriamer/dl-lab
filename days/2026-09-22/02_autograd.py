"""9-22 demo 2：自动求导。今天最重要的文件。对照 notes.md 第 3–5 节。"""

import torch
import torch.nn as nn

print("=== 手工能算的小例子 ===")
# y = 2 * x^T x ， dy/dx = 4x
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = 2 * (x * x).sum()
y.backward()
print("x =", x.tolist(), "y =", y.item())
print("x.grad =", x.grad.tolist(), "  期望 [4, 8, 12]")

print("\n=== backward 只填 .grad，不改参数 ===")
w = torch.nn.Parameter(torch.tensor([0.5]))
opt = torch.optim.SGD([w], lr=0.1)
loss = (w - 2.0) ** 2
print("backward 之前  w =", w.item(), "w.grad =", w.grad)
loss.backward()
print("backward 之后  w =", w.item(), "w.grad =", w.grad.item(), "  期望 2*(0.5-2)=-3")
opt.step()
print("step 之后      w =", w.item(), "  期望 0.5 - 0.1*(-3) = 0.8")

print("\n=== 不清零会把梯度累加 ===")
w2 = torch.nn.Parameter(torch.tensor([1.0]))
opt2 = torch.optim.SGD([w2], lr=0.1)
for i in range(2):
    loss2 = (w2 - 3.0) ** 2
    # 故意不清零
    loss2.backward()
    print(f"iter {i}: grad={w2.grad.item():.1f}  (每次应是 2*(1-3)=-4；第二次会变成 -8)")

print("\n=== 正确的一步：zero_grad → backward → step ===")
model = nn.Linear(1, 1, bias=False)
# 强行设成 0，目标是拟合 y=2x，所以 weight 应靠近 2
with torch.no_grad():
    model.weight.fill_(0.0)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
x = torch.tensor([[1.0], [2.0], [3.0]])
y = 2 * x
for step in range(8):
    pred = model(x)
    loss = nn.functional.mse_loss(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(
        f"step {step:02d}  loss={loss.item():.4f}  w={model.weight.item():.3f}  "
        f"grad={model.weight.grad.item():.3f}"
    )
print("w 应逐渐靠近 2.0")
