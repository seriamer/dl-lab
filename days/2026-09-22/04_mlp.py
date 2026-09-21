"""9-22 demo 4：线性模型过不了 XOR，一层隐藏层的 MLP 可以。"""

import torch
import torch.nn as nn

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=DEVICE)
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=DEVICE)


def train(model, steps=400, lr=0.1):
    model.to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(steps):
        pred = model(X)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(X)
    return loss.item(), pred.cpu().squeeze()


linear = nn.Linear(2, 1)
mlp = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))

loss_lin, pred_lin = train(linear)
loss_mlp, pred_mlp = train(mlp)

print("XOR 真值:            ", y.cpu().squeeze().tolist())
print(f"Linear  pred={pred_lin.tolist()}  loss={loss_lin:.4f}")
print(f"MLP     pred={pred_mlp.tolist()}  loss={loss_mlp:.4f}")
print("线性层四个点挤在 0.5 附近；MLP 应靠近 [0, 1, 1, 0]。")
print("ReLU 把空间折弯，这才叫多层感知机。明天把这个骨架接到真实图片上。")
