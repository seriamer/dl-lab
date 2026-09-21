"""9-22 demo 3：线性回归。先从零写梯度，再交给 autograd。"""

import torch

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRUE_W, TRUE_B, N = 2.0, -1.0, 256
X = torch.randn(N, 1, device=DEVICE)
noise = 0.1 * torch.randn(N, 1, device=DEVICE)
y = TRUE_W * X + TRUE_B + noise


def fit_from_scratch(steps=80, lr=0.1):
    w = torch.zeros(1, 1, device=DEVICE)
    b = torch.zeros(1, device=DEVICE)
    history = []
    for t in range(steps):
        pred = X @ w + b
        err = pred - y
        loss = (err * err).mean()
        # d(loss)/dw = 2/n * X^T err ，d(loss)/db = 2/n * sum(err)
        grad_w = (2 / N) * (X.T @ err)
        grad_b = (2 / N) * err.sum()
        w -= lr * grad_w
        b -= lr * grad_b
        history.append(float(loss))
        if t % 20 == 0 or t == steps - 1:
            print(f"[scratch] step {t:02d}  loss={float(loss):.4f}  w={float(w):.3f}  b={float(b):.3f}")
    return history, float(w), float(b)


def fit_with_nn(steps=80, lr=0.1):
    model = torch.nn.Linear(1, 1).to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    history = []
    for t in range(steps):
        pred = model(X)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
        if t % 20 == 0 or t == steps - 1:
            w = model.weight.item()
            b = model.bias.item()
            print(f"[nn]      step {t:02d}  loss={loss.item():.4f}  w={w:.3f}  b={b:.3f}")
    return history, model.weight.item(), model.bias.item()


print("真值 w,b =", TRUE_W, TRUE_B, "device =", DEVICE)
h1, w1, b1 = fit_from_scratch()
h2, w2, b2 = fit_with_nn()
print(f"scratch 收敛到 w={w1:.3f} b={b1:.3f}  末 loss={h1[-1]:.4f}")
print(f"nn      收敛到 w={w2:.3f} b={b2:.3f}  末 loss={h2[-1]:.4f}")
assert h1[-1] < h1[0] and h2[-1] < h2[0], "loss 应该下降"
print("两条路径都在下降，目标都是同一个 MSE。")
