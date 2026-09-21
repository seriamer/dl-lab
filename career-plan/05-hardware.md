# 05 · 硬件与环境

## 工位卡怎么理解

| 项 | 事实 | 用法 |
|---|---|---|
| RTX 5060 Ti | Blackwell，tensor 够用 | 主训练机 |
| **8GB**（8151 MiB） | 不是 16GB 版 | 真正瓶颈是显存不是算力 |
| WDDM + 显示器 On | 桌面约占 640MB | 可用约 7.5GB |
| 驱动 616.92 / CUDA 13.4 | 够新 | 必须装带 **sm_120** 的 PyTorch |
| Windows + Administrator | 能装 | conda 装用户目录，别装进 Program Files |
| 空闲时 QQ/Edge/NVIDIA App 占卡 | 正常 | **训练前关掉** |

8GB **够：** MLP、CIFAR CNN、小 ResNet、手写 Transformer、YOLO-n 推理、LeRobot ACT 小 batch。  
8GB **不够：** 7B VLA、Isaac 数千并行、大 ViT 全参大 batch。那些是明年的事。

不要因为「实验室没集群」去租云。阶段 A–D 这张卡够。只有以后训放不下的模型再租 AutoDL。

笔记本 30 系 Ti：写代码和刷题。训练默认工位。

## 安装（PowerShell，工位）

不要 `pip install torch` 不带 index（Windows 上容易装成 CPU 或旧 CUDA 轮子）。50 系旧轮子典型报错：`no kernel image is available`。

### 1. Miniconda

<https://docs.conda.io/en/latest/miniconda.html>  
装到 `C:\Users\Administrator\miniconda3`。装完 **关掉终端再开**。

```powershell
conda --version
```

### 2. 环境 + PyTorch（CUDA 13）

```powershell
conda create -n dl python=3.11 -y
conda activate dl
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

`cu130` 404 则改：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu132
```

**不要用** `cu118` / `cu124` / `cu126` 给这张 5060 Ti。

### 3. 验证（必须全过）

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('name', torch.cuda.get_device_name(0)); print('cap', torch.cuda.get_device_capability()); print('arch', torch.cuda.get_arch_list()); x=torch.randn(1024,1024,device='cuda'); y=x@x; print('ok', y[0,0].item())"
```

期望：

- `available True`  
- `name` 含 `RTX 5060 Ti`  
- `cap` 为 `(12, 0)`  
- `arch` 含 **`sm_120`**  
- 打印 `ok` 和一个数  

### 失败表

| 现象 | 原因 | 处理 |
|---|---|---|
| `available False` | CPU 版轮子 | `pip uninstall -y torch torchvision torchaudio` 后带 `--index-url` 重装 |
| `no kernel image is available` | 轮子无 sm_120 | 只用 cu130/cu132 |
| `conda` 不是命令 | PATH | 重开终端或 Anaconda Prompt |
| 一训练就 OOM | 阶段 A 不该发生 | 关 QQ/Edge；确认没开巨大 batch 的别人的脚本 |

## 8GB 习惯（后面一直用）

- 训练盯 `nvidia-smi`，不要长期 7.8GB+  
- 阶段 A：模型小，batch 可大  
- ACT / 检测：`batch=4 或 8`，`torch.amp.autocast('cuda')`  
- 显示器占显存正常，不为抠 600MB 关桌面  

## 实际安装记录（WSL Ubuntu，2026-09-21）

工位已经在 WSL 里用 conda 装好，**不要重装**：

```text
conda env: dl
python: 3.12
torch: 2.14.0+cu132
cuda runtime: 13.2
device: NVIDIA GeForce RTX 5060 Ti 8GB
capability: (12, 0)  → sm_120
```

每天开终端：

```bash
conda activate dl
cd ~/dl-lab
```

## Linux（阶段 A 的 10/05）

WSL Ubuntu 已经能进。10/05 不必再装系统，改成：确认 `git` / `python3` 日常能用，把这几天的练习 push 到 GitHub。

ROS / LeRobot 后期需要 Linux。阶段 A 只要求：

- 工位已是 Ubuntu → 最好  
- 工位 Win → **WSL2 Ubuntu** 或双系统；双系统若实验室管机器，先问能不能装  
- 工位不能乱动 → 笔记本 WSL2；训练仍用 Win 上的 50Ti  

10/05 验收：`lsb_release -a` 有输出，`python3` 和 `git` 能用。不要当天编译 ROS。

## 工位机注意

可能是实验室公用账号。conda 放用户目录；不要卸别人的驱动；长时间占卡先确认组里没人用这台做正事。
