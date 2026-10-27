# Qwen 数学微调：库与函数学习路线

本项目当前目标：先理解并实现 DPO、PPO、GRPO 的核心计算，正式训练与评测稍后在租用的服务器上完成。
笔记以初学者为对象，围绕项目会用到的接口展开，不是完整 API 手册。

主实验已选定 **Qwen/Qwen2.5-Math-1.5B 基座**，后续使用单张标准 24GB RTX 4090，采用 LoRA。
完整路线包含 SFT，再从同一 SFT 快照分别运行 DPO、PPO、GRPO。
见 [模型选择](../docs/model_selection.md) 与 [项目结构](../docs/project_structure.md)。
笔记中少数 Qwen3 聊天模板例子仅用于理解 API；主实验基座使用项目的显式文本模板。

## 学习顺序

| 顺序 | 笔记 | 学完要能做什么 |
| --- | --- | --- |
| 1 | [torch.md](torch.md) | 读懂张量形状、计算回答概率、反向传播 |
| 2 | [transformers.md](transformers.md) | 理解分词、聊天模板、模型前向与生成 |
| 3 | [datasets.md](datasets.md) | 将数学题处理成批量输入，构造回答 mask |
| 4 | [peft.md](peft.md) | 理解并添加 LoRA，保存与恢复适配器 |
| 5 | [algorithm_functions.md](algorithm_functions.md) | 用小张量串起 DPO、PPO、GRPO 核心计算 |
| 6 | [accelerate.md](accelerate.md) | 后续把单机训练循环迁移到服务器 |
| 7 | [trl.md](trl.md) | 用成熟实现核对算法定义与实验配置 |
| 查阅 | [python_tools.md](python_tools.md) | 管理配置、路径、日志和随机种子 |
| 项目实战 | [ppo.md](ppo.md) | 对照现有代码串起 PPO 采样、奖励、GAE、更新和断点恢复 |

第一轮只学习前五篇。Accelerate 和 TRL 可以等自己实现的损失函数能工作后再读。

## 统一符号

| 符号 | 含义 | 典型形状 |
| --- | --- | --- |
| B | batch size，一批序列数量 | — |
| T | 补齐后的序列长度，包含问题与回答 | — |
| V | 词表大小 | — |
| H | 模型隐藏维度 | — |
| G | 同一道题采样的回答数量 | — |
| input_ids | token 编号 | `[B, T]`，整数 |
| attention_mask | 有效输入位置为 1，补齐位置为 0 | `[B, T]` |
| response_mask | 回答位置为 1，其余为 0 | `[B, T]` |
| logits | 模型对每个词表 token 的未归一化分数 | `[B, T, V]` |

attention_mask 和 response_mask 用途不同：问题需要参加注意力计算，但通常不计入回答损失。

## 示例约定

- 标为“可直接运行”的代码块自带必要输入；安装对应库即可运行。
- 标为“片段”的代码需要前文提供的模型、tokenizer 或数据，不是完整训练脚本。
- 模型加载示例可能下载权重，不要把每个代码块都当作必须立即执行的练习。
- 本地先做 CPU 小张量练习；不要求立刻下载模型或开始训练。
- 不修改 MedicalGPT 的环境。本项目统一使用 uv，通过 uv sync 管理独立 .venv 和 uv.lock，通过 uv run 运行代码。

## 环境与版本

记录用户当前提供的环境：Python 3.11.16、PyTorch 2.14.0+cu132、CUDA 可用、RTX 4060 系列约 8GB 显存。
这是已有项目环境的信息，不代表 qwen_math 已安装相同依赖，也不代表这些版本已完成联合兼容验证。

官方文档核对日期：2026-10-21。文档中的主线接口可能变化，实际实现要固定并核对包版本。

可用 Python 标准库查看当前环境包版本，无须导入各个训练库：

```python
from importlib.metadata import PackageNotFoundError, version

for name in ["torch", "transformers", "peft", "datasets", "accelerate", "trl"]:
    try:
        print(name, version(name))
    except PackageNotFoundError:
        print(name, "未安装")
```

## 推荐的第一个练习

打开 torch.md，运行 `log_softmax + gather` 示例。确认你能解释：

1. 为什么 logits 最后一维是词表大小？
2. 为什么标签需要右移一位对齐？
3. 为什么 DPO 需要对回答 token 的 log probability 求和？

这三个问题是后续三种算法共同的基础。
