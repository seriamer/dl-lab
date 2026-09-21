# 9-22 概念笔记（对着代码看）

资源只开 [d2l PyTorch 版](https://zh.d2l.ai/) 对应章节：数据操作、自动微分、线性回归、MLP。

## 1. 张量就是多维数组

训练里数据、权重、梯度全是 `torch.Tensor`。

| 操作 | 记住什么 |
|---|---|
| `x.shape` | 每一维多大 |
| `x.reshape` / `x.view` | 元素总数不变才能改形状 |
| `x * y` | **逐元素**乘，靠广播对齐形状 |
| `x @ y` 或 `x.matmul(y)` | **矩阵乘**，内维必须相等 |
| `x.to("cuda")` | 数据和模型要在同一设备 |

广播：从右边对齐维度，1 可以扩成对面的大小。`(3,1) * (1,4) → (3,4)`。

## 2. 训练循环的五件事

```text
pred = model(x)          # 前向：用当前参数算出预测
loss = criterion(pred, y)
optimizer.zero_grad()    # 清掉上一步留在 .grad 里的数
loss.backward()          # 反传：给每个参数填上 d(loss)/d(param)
optimizer.step()         # 更新：param ← param - lr * param.grad
```

线性回归、MLP、CNN、Transformer，骨架都是这一段。9/23 起每天都写它。

## 3. `loss.backward()` 在干什么

PyTorch 在 `requires_grad=True` 的张量参与运算时，会记下计算图。

`loss` 是一个标量。调用 `loss.backward()`：

1. 从 `loss` 出发，沿计算图反向走。
2. 用链式法则算每个叶子张量的梯度。
3. 把结果 **累加** 进 `param.grad`（是累加，不是覆盖）。

叶子张量：你创建的、需要求梯度的参数，例如 `nn.Linear` 的 `weight`，或 `torch.nn.Parameter`。

所以 backward **不改参数的值**，只往 `.grad` 里写数。

## 4. `optimizer.step()` 在干什么

以 SGD 为例，对每个参数：

```text
param.data = param.data - lr * param.grad
```

Adam 还会用动量和二阶估计，但接口一样：先 `backward` 填梯度，再 `step` 改参数。

## 5. 为什么必须 `zero_grad()`

`backward()` 是 **累加** 到 `.grad`。如果不清零，第 2 个 batch 的梯度会叠在第 1 个上面，更新方向就错了。

顺序固定写成：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 6. 线性回归在优化什么

数据：`y ≈ X w + b`。  
损失：均方误差 `mean((pred - y)^2)`。  
对 `w` 的梯度（一维时）：`2/n * X^T (Xw+b-y)`。  
从零实现就是自己算这个梯度再减。`nn` 版让 autograd 算同一件事。

## 7. MLP 多了什么

一层线性只能画一个超平面。XOR 四个点不是线性可分的。

```text
h = ReLU(x W1 + b1)
y = h W2 + b2
```

`ReLU(z) = max(z, 0)` 把空间折一下，两层就能表示 XOR。非线性是必须的，不是装饰。
