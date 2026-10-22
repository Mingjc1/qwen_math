# Qwen Math

基于千问小模型，逐步实现 SFT、DPO、PPO、GRPO，比较数学任务上的效果与资源成本。

## 固定的模型与实验路线

- **主模型：`Qwen/Qwen2.5-Math-1.5B`，基座版本。**
- 目标硬件：单张标准 RTX 4090，24GB 显存；本地约 8GB 显存的电脑负责学习和小规模实现验证。
- 第一版采用普通 LoRA，BF16 作为服务器起步配置；实际支持情况与峰值显存需要实测。
- SFT 后保存一份不可变快照，DPO、PPO、GRPO 分别从它开始，不串联作为主对照实验。
- 第一版 PPO/GRPO 使用可验证的答案奖励；单独训练 reward model 留作扩展。

```text
Qwen2.5-Math-1.5B
       |
      SFT ──────────── 固定 SFT 参考快照
       |
       +── DPO
       +── PPO
       +── GRPO
```

选择理由、显存方案和模型输入约定见 [模型选择](docs/model_selection.md)。
模块职责与推进顺序见 [项目结构](docs/project_structure.md)，库函数学习从 [notes/README.md](notes/README.md) 开始。

## 当前完成状态

已提供目录、配置校验入口、数据格式示例和学习笔记。
算法、模型加载、训练与评测模块目前是有职责说明的骨架，**尚未实现训练或评测**。
配置中的超参数是待验证的起始值，不是训练结果。此阶段不下载权重，也不执行训练。

## 本地查看配置

在 qwen_math 根目录运行，Python 3.11 标准库即可，不需要 PyTorch：

```bash
uv run python scripts/train.py --config configs/sft.json --dry-run
uv run python scripts/train.py --config configs/dpo.json --dry-run
uv run python scripts/train.py --config configs/grpo.json --dry-run
uv run python scripts/train.py --config configs/ppo.json --dry-run
```

dry-run 校验配置并打印说明，不检查权重、数据或 CUDA 是否存在。
不加 --dry-run 时，当前入口明确报告训练器尚未实现，并以非零状态退出。
相对数据和输出路径统一相对于项目根目录解析，不依赖启动命令时的工作目录。

## 使用 uv 管理环境与依赖

不要继续向 MedicalGPT 环境添加本项目依赖。另开终端或先 deactivate，然后：

```bash
cd /home/mingjc/mjc_projects/qwen_math
uv --version
uv sync
uv run python -c "import sys; print(sys.version); print(sys.executable)"
```

项目已有 pyproject.toml，不需要再次 uv init。Python 版本由 .python-version 指定为 3.11。
uv sync 创建或同步本项目 .venv，并安装项目；默认不安装 train 等可选依赖，但解析锁文件时会考虑可选依赖的兼容性。
首次解析可能访问软件源；若缺少 Python 3.11，uv 也可能下载对应解释器。
使用 uv run 无须手动激活环境，也不要传 --active 去操作其他项目环境。

后续确定服务器驱动和 PyTorch CUDA 软件源后，再配置 uv 的 PyTorch source 并启用训练依赖：

```bash
uv sync --extra train
uv run --extra train python scripts/train.py --config configs/sft.json --dry-run
```

当前仅更新管理方式，尚未执行同步，也尚未生成 uv.lock。
生成的 uv.lock 应纳入版本管理；正式实验前验证依赖组合，并固定模型 revision 与数据版本。
保留可选依赖时运行命令也要带相同 extra，例如 --extra train，避免环境同步移除未选择的可选包。
TRL 仅作为 reference 可选依赖，第一版自主实现不需要安装。
不要在 uv 管理的项目里混用临时 pip 安装来维护正式依赖。

参考：[uv 项目管理](https://docs.astral.sh/uv/guides/projects/)、[锁定与同步](https://docs.astral.sh/uv/concepts/projects/sync/)。

## 数据和产物

`data/examples/` 仅有手写的格式示例，不能用来报告训练效果。
真实原始数据放 data/raw，清洗与划分后放 data/processed，偏好数据放 data/preferences。
权重缓存放 models，训练产物放 outputs，实验记录放 experiments；它们均已在 .gitignore 排除。

统一完整回答格式：

```text
解题过程……
Final answer: 5
```

基座第一版使用显式文本模板，不依赖 tokenizer 是否自带聊天模板。
标准答案只交给奖励与评测模块，不能泄漏到 RL 的输入 prompt。

## 下一步

先完成 `common/tensor_ops.py` 的因果对齐、回答 log probability 和 masked reduction，使用小张量验证；
再做 SFT 的 collator、模型加载与训练循环，然后按 DPO → GRPO → PPO 的顺序实现。

## 官方依据

- [模型卡](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)
- [RTX 4090 规格](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)
