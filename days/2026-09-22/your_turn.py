"""自己写。把四个函数补完后运行：python your_turn.py"""

import torch

# ---------- 1. 张量 ----------
def broadcast_sum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a 形状 (3,1)，b 形状 (1,4)，返回 (3,4) 的和。不要用 Python 循环。"""
    return a + b


def matmul_col_vector(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """A:(3,4)  v:(4,)  返回 (3,) 的矩阵乘。"""
    return A @ v


# ---------- 2. 手动梯度（对应从零线性回归） ----------
def mse_grad_w(X: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    pred = X @ w ，loss = mean((pred-y)^2)，X:(n,d) y:(n,1) w:(d,1)
    返回 d(loss)/dw，形状 (d,1)。公式：2/n * X^T (Xw - y)
    """
    n = X.shape[0]
    pred = X @ w
    grad = (2 / n) * (X.T @ (pred - y))
    return grad


# ---------- 3. 用 autograd 走一步 SGD ----------
def sgd_one_step(w: torch.Tensor, X: torch.Tensor, y: torch.Tensor, lr: float) -> torch.Tensor:
    """
    w 是 requires_grad 的 (d,1)。
    计算 MSE，backward，然后用 w.data -= lr * w.grad 更新。
    返回更新后的 w（还是同一个张量）。
    不要用 torch.optim，手写这一步是为了讲清 step()。
    """
    pred = X @ w
    loss = torch.mean((pred - y) ** 2)
    loss.backward()
    return w.data.sub_(lr * w.grad)


# ---------- 4. 口头检查写成代码：顺序必须对 ----------
def train_step(model, opt, loss_fn, x, y):
    """
    完成一次：forward → loss → zero_grad → backward → step。
    返回标量 loss（float）。
    """
    pred = model(x)
    loss = loss_fn(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def _check():
    a = torch.arange(3).reshape(3, 1)
    b = torch.arange(4).reshape(1, 4)
    s = broadcast_sum(a, b)
    assert s.shape == (3, 4)
    assert torch.equal(s, a + b)

    A = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    v = torch.tensor([1.0, 0.0, 0.0, 2.0])
    out = matmul_col_vector(A, v)
    assert out.shape == (3,)
    assert torch.allclose(out, A @ v)

    torch.manual_seed(0)
    X = torch.randn(20, 2)
    true_w = torch.tensor([[1.5], [-0.5]])
    y = X @ true_w
    w = torch.zeros(2, 1)
    g = mse_grad_w(X, y, w)
    manual = (2 / X.shape[0]) * (X.T @ (X @ w - y))
    assert g.shape == (2, 1)
    assert torch.allclose(g, manual, atol=1e-5)

    w = torch.zeros(2, 1, requires_grad=True)
    w_before = w.detach().clone()
    sgd_one_step(w, X, y, lr=0.1)
    assert w.grad is not None
    expected = w_before - 0.1 * w.grad
    assert torch.allclose(w.detach(), expected, atol=1e-5)

    model = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    x = torch.randn(8, 2)
    yb = torch.randn(8, 1)
    w0 = model.weight.detach().clone()
    loss = train_step(model, opt, torch.nn.MSELoss(), x, yb)
    assert isinstance(loss, float)
    assert not torch.equal(model.weight.detach(), w0), "step 之后参数应该变"

    print("your_turn: 全部通过")


if __name__ == "__main__":
    _check()
