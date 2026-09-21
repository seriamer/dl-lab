# 阶段 A 进度

对照 [`03-phase-a.md`](03-phase-a.md)。做完打勾，周日晚更新一次。

## 10/07 硬检查

- [x] 工位 5060 Ti：`torch.cuda.is_available() is True`，GPU 矩阵乘成功（2026-09-21，WSL + conda `dl`，cu132）
- [ ] 不看教程能写训练循环；CIFAR-10 或 Fashion-MNIST 上 loss 下降
- [ ] 自己写过 `scaled_dot_product_attention`、Multi-Head Attention、EncoderBlock
- [ ] LeetCode 累计 20 题

## 逐日

| 日期 | 计划 | 状态 |
|---|---|---|
| 9/20 | nvidia-smi、建 `dl`、空仓库、两数之和 | 环境+仓库完成；两数之和未确认 |
| 9/21–22 | 张量、自动求导、线性回归、MLP；哈希题 | **今天执行 9-22** |
| 9/23–25 | 手写完整循环 + Fashion-MNIST/CIFAR | 未开始 |
| 9/26–27 | 小 CNN | 未开始 |
| 9/28–30 | 小 ResNet + 注意力笔记 | 未开始 |
| 10/01–07 | 手写 Transformer | 未开始 |

## LeetCode（Python，目标 20）

| # | 题 | 日期 |
|---|---|---|
| 1 | 1. 两数之和 | 9/20 计划，未确认 |
| 2 | 242. 有效字母异位词 | 9/21 计划，今天补做可选 |
| 3 | 128. 最长连续序列 | **9/22 必做** |
