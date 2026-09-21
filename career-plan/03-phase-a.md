# 03 · 阶段 A 逐日（2026-09-20 → 10-07）

这是 **当前唯一要执行的表**。资源只用：

- [动手学深度学习 PyTorch 版](https://zh.d2l.ai/)  
- [PyTorch 60-minute blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)  
- LeetCode 热题，**Python**

不要同时开吴恩达 / CS231n / 花书。

## 10 月 7 日硬检查（四条全过）

1. 工位 5060 Ti：`torch.cuda.is_available() is True`，验证脚本里 GPU 矩阵乘成功  
2. 不看教程能写训练循环；CIFAR-10 或 Fashion-MNIST 上 loss 下降  
3. 自己写过 `scaled_dot_product_attention`、Multi-Head Attention、EncoderBlock  
4. LeetCode **累计 20 题**（数组、哈希、双指针、链表入门）

## 这 17 天明确不做

ROS、Isaac、LeRobot、YOLO 工业检测、买机械臂、为组里安全课题加班超过 15h、在笔记本 30Ti 上硬训。

---

## 9/20 日（会话当日）

- 两台机 `nvidia-smi`，记下型号和显存（工位已确认 5060 Ti 8GB）  
- 工位按 [05-hardware.md](05-hardware.md) 建 `dl` 环境、装 CUDA 13 轮子、跑验证脚本  
- GitHub 空仓库 `dl-lab`  
- LeetCode 第 1 题：**两数之和**  

环境没亮不要开始「看课」。CUDA 报错把 **完整终端输出** 发给助手。

## 9/21–9/22（一、二）

- d2l：张量、自动求导、线性回归、MLP  
- 自己敲，不整段粘贴  
- 每晚 1 题：有效字母异位词、最长连续序列（或同类哈希）  

**22 日检查：** 能口头解释 `loss.backward()` 和 `optimizer.step()`。

## 9/23–9/25（三–五）

- 手写完整循环：Dataset → DataLoader → model → loss → backward → step → 打印验证  
- 先 Fashion-MNIST 或随机数据，**必须在 50Ti 上跑**  
- 会保存 checkpoint；loss 记到文件或 TensorBoard  
- 每晚 1 题：无重复字符最长子串、盛水最多的容器（滑动窗口 / 双指针）  
- 组里活尽量下午连续干完  

**25 日检查：** `dl-lab/train_mlp.py` + README 三行怎么跑。

## 9/26–9/27（周末）

- d2l CNN 章：卷积和全连接差别，不追公式  
- CIFAR-10 或 Fashion-MNIST 训小 CNN，对比 MLP  
- 记录：准确率、时长、显存  
- 两天共 3 题：反转链表、合并两有序链表、环形链表  

**27 日检查：** `nvidia-smi` 训练时有占用，不是跑在 CPU。

## 9/28–9/30（一–三）

- torchvision 微调小 ResNet（CIFAR 即可）  
- 开始 d2l 注意力 / Transformer：QKV、mask、多头  
- 一页纸笔记：从 attention 到 encoder block  
- 每晚 1 题：栈或二叉树预热  

组里若加活：CNN 对比实验可少做，**Transformer 笔记不能停**。

## 10/01–10/07 国庆（每天 7–8h）

组里大概率放假。按表打穿 Transformer。

| 日期 | 任务 | 当天结束标准 |
|---|---|---|
| 10/01 | 手写 `scaled_dot_product_attention`，小随机 tensor 对形状 | `[B,H,T,D]` 正确且能跑 |
| 10/02 | MHA + 残差 + LayerNorm | `EncoderBlock.forward` 通 |
| 10/03 | 叠 2–4 层 + 一个 toy 任务（d2l 翻译或字符拷贝，越小越好） | loss 下降 |
| 10/04 | 对照 ACT/ViT 架构图，写半页：策略网络的 attention 在 attend 什么（只求看懂） | 能用自己的话讲 |
| 10/05 | 工位或笔记本装 WSL2 Ubuntu **或** 双系统；`python3` + `git` 可用 | `lsb_release -a` 有输出 |
| 10/06 | LeetCode 补到累计 20；Transformer 代码推进 GitHub | 仓库能看懂 |
| 10/07 | 上午休息。下午闭卷：重写训练循环 + MHA 形状；写不出的晚上补 | 阶段 A 收口 |

10/05 不要做成装机周。进得了 Ubuntu shell 即可。ROS 放阶段 B。

---

## 每天模板（上课日）

```
白天     课 / 组里（12–15h 顶格，尽量下午一块做完）
19:30–21:00   深度学习（最重要）
21:10–22:10   LeetCode 1–2 题，计时
周日晚 30min  对照本表打勾，git push
```

课程爆了的周：项目停，**每天仍 1 题**，训练循环至少提交一次，不断档。

## 阶段 A 结束后

进入 [04-to-winter-break.md](04-to-winter-break.md)：小视觉服务 → LeRobot，不再回头做工业检测。
