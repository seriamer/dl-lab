# 2026-09-22（阶段 A · 第 3 天）

对照 [`career-plan/03-phase-a.md`](../../career-plan/03-phase-a.md) 的 **9/21–9/22**。  
9/20 的环境已经亮了，今天把落下的底座一次补齐。

## 今天结束必须成立

1. 能自己用张量做：创建、切片、reshape、广播、矩阵乘，并放到 CUDA。
2. **口头检查（22 日硬标准）：** 能讲清 `loss.backward()` 和 `optimizer.step()` 各做了什么，以及为什么每次迭代要 `zero_grad()`。
3. 从零写出线性回归的一步更新；再用 `nn.Linear` + `MSELoss` + `SGD` 让 loss 下降。
4. 知道 MLP = 线性层 + 非线性，线性模型过不了 XOR。
5. LeetCode：**128. 最长连续序列**（哈希）。9/20–21 的两数之和、有效字母异位词没做就顺手补。

## 时间盒（上课日模板）

| 时段 | 做什么 | 文件 |
|---|---|---|
| 19:30–20:10 | 张量 + 自动求导，先读 notes 再跑 demo | `notes.md`、`01_tensor.py`、`02_autograd.py` |
| 20:10–21:00 | 线性回归从零开始，再对照 `nn` 版 | `03_linear_regression.py` |
| 21:00–21:20 | MLP / XOR；做 `your_turn.py` | `04_mlp.py`、`your_turn.py` |
| 21:20–22:10 | 刷题，计时 | `leetcode/128_longest_consecutive.py` |

自己敲关键几行。demo 用来对照，不要整段复制交作业。`your_turn.py` 里的函数必须你来写。

## 怎么跑

```bash
conda activate dl
cd ~/dl-lab/days/2026-09-22
python 01_tensor.py
python 02_autograd.py
python 03_linear_regression.py
python 04_mlp.py
python your_turn.py
python leetcode/128_longest_consecutive.py
```

## 不要做

ROS、Isaac、LeRobot、YOLO、再装一遍 PyTorch、同时打开吴恩达/花书。
