# 2026-09-22 我的笔记

阶段 A · 张量 / 自动求导 / 线性回归 / MLP

---

## 1. 张量

数据和参数都是 `torch.Tensor`。先看三个属性：`shape`、`dtype`、`device`。

二维输入 `X` 的约定：

- **行 = 样本数 n**
- **列 = 特征数 d**（dimension）
- 每一行和 `w` 做点积（再加 `b`），得到这个样本的一个预测
- n 个样本排在一起，预测是形状 `(n, 1)` 的向量

常用运算：

| 写法 | 做什么 |
|---|---|
| `A * B` | 逐元素乘，形状靠广播对齐 |
| `A @ B` | 矩阵乘，左边列数 = 右边行数 |
| `x.reshape(...)` | 改形状，元素总数不变 |
| `x.to("cuda")` | 数据和模型放同一设备 |

**广播：** 任意维都能播，不限于向量。从右边对齐；某一维相等，或其中一边是 1。长度为 1 的维沿这个方向复用同一个数。`(3,1)+(1,4)→(3,4)` 是两个矩阵各扩各的。实现上经常是 stride=0 的视图，不真复制内存。

意义：同一条规则打在整批数据上，写成一行。例如 `(batch, d) + (d,)` 就是给每个样本加同一份 bias。

---

## 2. 自动求导（今天硬标准）

训练骨架：

```text
pred = model(x)
loss = criterion(pred, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

- `loss.backward()`：从标量 loss 沿计算图反走，把 `d(loss)/d(param)` **累加**进 `param.grad`。**不改参数的值。**
- `optimizer.step()`：用 `.grad` 改参数。SGD：`w ← w - lr * w.grad`。
- `optimizer.zero_grad()`：每次 backward 都是累加。不清零，下一步会叠在上一步的梯度上。

`w` 是权重，由调用方建好并设 `requires_grad=True`，形状常写成 `(d, 1)`（d 行 1 列）。函数里直接用传入的 `w`，不要在里面再 `torch.tensor(...)` 新建一份。

`sgd_one_step`：自己写前向、MSE、`backward`，再用 `w.data -= lr * w.grad`（不用 `torch.optim`）。  
`train_step`：同一套五步，改用现成的 `model` / `opt` / `loss_fn`，返回 `loss.item()`。

---

## 3. 线性回归

### 拟合目标

```text
pred = X @ w + b
y ≈ 2x - 1 + 噪声
```

demo 里 `X` 形状 `(256, 1)`，`w` 形状 `(1, 1)`，`b` 形状 `(1,)`。从 `w=0, b=0` 推到 `w≈2, b≈-1`。

练习题 `mse_grad_w` 没有 bias，就是 `pred = X @ w`。

### 损失

```text
err  = pred - y
loss = mean(err²) = (1/n) Σ errᵢ²
```

### 梯度为什么是 (2/n) · Xᵀ (Xw − y)

对单个样本：`predᵢ = xᵢ · w`（`xᵢ` 是第 i 行，长度 d）。

```text
∂loss / ∂predᵢ = (2/n) · errᵢ
predᵢ 对 w 的导数就是 xᵢ
```

链式法则后对所有样本求和：

```text
∂loss / ∂w = (2/n) · Xᵀ @ err = (2/n) · Xᵀ (Xw - y)
```

形状先核对：

```text
X     (n, d)
err   (n, 1)
Xᵀ    (d, n)
Xᵀ @ err → (d, 1)    和 w 同形状
```

用 n=2、d=2 展开，看 `Xᵀ` 从哪来。

$$
X = \begin{bmatrix} x_{11} & x_{12} \\ x_{21} & x_{22} \end{bmatrix},\quad
w = \begin{bmatrix} w_1 \\ w_2 \end{bmatrix},\quad
err = \begin{bmatrix} err_1 \\ err_2 \end{bmatrix}
$$

`w₁` 出现在两个样本的预测里：`∂err₁/∂w₁ = x₁₁`，`∂err₂/∂w₁ = x₂₁`。

$$
\frac{\partial \text{Loss}}{\partial w_1} = \frac{2}{n}(x_{11}\,err_1 + x_{21}\,err_2)
$$

括号里是 **X 的第 1 列** 和 **err** 的点积。把 `w₁、w₂` 的偏导叠成向量，就是 `Xᵀ @ err`：

$$
\frac{\partial \text{Loss}}{\partial w}
= \frac{2}{n}
\begin{bmatrix} x_{11} & x_{21} \\ x_{12} & x_{22} \end{bmatrix}
\begin{bmatrix} err_1 \\ err_2 \end{bmatrix}
= \frac{2}{n}
\begin{bmatrix} x_{11}err_1 + x_{21}err_2 \\ x_{12}err_1 + x_{22}err_2 \end{bmatrix}
$$

`Xᵀ` 把「每个样本的某一维特征」竖过来，和每个样本的误差一一相乘再求和。

对 bias：`∂loss/∂b = (2/n) · Σ errᵢ`。

### 手写梯度

```python
def mse_grad_w(X, y, w):
    n = X.shape[0]
    pred = X @ w
    grad = (2 / n) * (X.T @ (pred - y))
    return grad
```

这里的 `w` 没有 `requires_grad`，用矩阵乘法直接出梯度，不用 `backward()`。

### 两条训练路径（数学上同一件事）

| 环节 | 从零（手推） | `nn` + autograd |
|---|---|---|
| 参数 | `w = torch.zeros(d, 1)`，不开 grad | `nn.Linear(1, 1)` 自带 weight / bias |
| 梯度 | `(2/n) * X.T @ err` | `loss.backward()` |
| 更新 | `w -= lr * grad_w` | `opt.step()` |
| 清梯度 | 每次重算 `grad_w`，没有累加问题 | 必须 `opt.zero_grad()` |

```python
pred = model(X)
loss = loss_fn(pred, y)
opt.zero_grad()
loss.backward()
opt.step()
```

### 末损失停在约 0.01

数据是 `y = 2x - 1 + ε`，`ε` 标准差约 0.1。MSE 的下限大约是 `E[ε²] ≈ 0.01`。直线已经贴近真值，穿不过每一个被噪声挪过的点，loss 不必到 0。

---

## 4. MLP

线性层 `pred = x₀w₀ + x₁w₁ + b` 在平面上是一刀。XOR 同类在对角，一刀切不开，预测会挤在 0.5，MSE 停在 0.25。

### `nn.Linear(2, 8)` 两个数字

`nn.Linear(in_features, out_features)`：

- **2 = 输入特征**：XOR 每个点是 `(x₀, x₁)`
- **8 = 输出特征**：这一层把每个样本变成 8 个数（8 个隐藏神经元）

数学上：`h = x W₁ + b₁`。  
若把 `W₁` 写成右乘矩阵，形状是 `(2, 8)`，`b₁` 是 `(8,)`，输出 `(n, 8)`。  
PyTorch 里 `weight` 存成 `(out, in) = (8, 2)`，前向是 `x @ weight.T + bias`，和上面同一件事。

### 为什么先 (2, 8) 再 (8, 1)

前一层出口 = 后一层入口：

```text
输入 (n, 2)
  → Linear(2, 8)   → (n, 8)
  → ReLU           → (n, 8)    负数变 0
  → Linear(8, 1)   → (n, 1)
```

- 开头 2：由输入决定  
- 结尾 1：由要预测的一个数决定  
- 中间 8：隐藏层宽度，超参数；4 或 16 也能解 XOR。第二层必须写成 `Linear(8, 1)` 才能接上

直接 `Linear(2, 1)` 只有一刀，XOR 分不开。升到 8 维等于 8 个神经元各看一种中间特征（偏左、偏上等），ReLU 把负的折掉，最后一层看这 8 个特征再压成 1 个数。

两层线性中间若没有 ReLU：`y = (x W₁ + b₁) W₂ + b₂` 仍等价于一层线性。非线性在 ReLU 上。

训练仍是 03 那五步，只换 `model`：

```python
nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
```

---

## 口头检查

1. `backward()`：算梯度，写入 `.grad`，参数值不变。  
2. `step()`：按优化器公式用 `.grad` 改参数。  
3. `zero_grad()`：清掉上一步的 `.grad`，避免累加。
