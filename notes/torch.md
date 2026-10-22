# PyTorch：算法实现的计算基础

`torch` 负责张量、自动求导和参数更新。DPO、PPO、GRPO 的核心损失最终都是 PyTorch 张量运算。

## 1. 创建张量、查看形状与设备

可直接运行（CPU）：

```python
import torch

ids = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
scores = torch.zeros(2, 3, dtype=torch.float32)
print(ids.shape, ids.dtype, ids.device)
print(torch.ones_like(scores))
print(torch.arange(3))
```

| 接口 | 作用 | 项目中的用途 |
| --- | --- | --- |
| `torch.tensor(data, dtype=...)` | 从数据创建张量 | token 编号、奖励 |
| `torch.zeros / ones / full` | 创建指定形状张量 | 初始化 mask、优势 |
| `torch.zeros_like / ones_like` | 复制形状与设备 | 初始化与输入匹配的缓冲区 |
| `x.shape / dtype / device` | 检查形状、类型、设备 | 排查维度错误 |
| `x.to(device)` | 移动设备，返回转换结果 | CPU/GPU 切换 |
| `x.float() / long() / bool()` | 转换类型 | 概率运算、索引、掩码 |

input_ids 通常为 long，mask 常为 bool 或整数；计算 loss 时转为合适的浮点类型。
调用 `x.to(...)` 后要接收返回值。不要对 token 编号使用浮点类型。

## 2. 维度、拼接和广播

可直接运行：

```python
import torch

x = torch.arange(6).reshape(2, 3)  # [2, 3]
print(x.unsqueeze(-1).shape)       # [2, 3, 1]
print(x.unsqueeze(-1).squeeze(-1).shape)
print(torch.stack([x, x], dim=0).shape)  # [2, 2, 3]，新增一维
print(torch.cat([x, x], dim=1).shape)    # [2, 6]，沿已有维度拼接
advantage = torch.tensor([1.0, -1.0])    # [2]
print(x * advantage.unsqueeze(-1))      # [2, 3]，每条序列共享一个优势
```

`reshape` 改形状；`transpose`、`permute` 调换轴。`view` 对内存布局有要求，初学时优先使用 reshape。
`squeeze()` 不指定维度可能在 B=1 时误删 batch 维，建议明确写 `squeeze(-1)`。

## 3. log_softmax + gather：取得指定 token 的概率

模型输出所有词表 token 的分数，算法只需要“实际回答 token”对应的 log probability。
log probability 是概率的自然对数，概率越大，对数越接近 0。

可直接运行：

```python
import torch
import torch.nn.functional as F

# B=1，两个预测位置，词表大小 V=3
logits = torch.tensor([[[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]]])
targets = torch.tensor([[1, 2]])
all_logps = F.log_softmax(logits, dim=-1)   # [1, 2, 3]
token_logps = all_logps.gather(
    dim=-1, index=targets.unsqueeze(-1)
).squeeze(-1)                            # [1, 2]
print(token_logps)
print(token_logps.exp())                 # 对应 token 的概率
print(token_logps.sum(-1))               # 整条序列的 log probability
```

`gather` 的 index 需要与输入维数相同，因而先 unsqueeze，再 squeeze。
使用 `F.log_softmax`，比先 softmax 再 log 更稳定。
大模型的完整 `[B,T,V]` 浮点中间张量占显存；这里以可读性优先，后续再优化。

## 4. 因果对齐与回答 mask

位置 t 的输出预测位置 t+1 的 token，所以需要同时移动 logits、targets 和 mask。
下面是项目自定义函数，不是 torch 自带 API。

可直接运行：

```python
import torch
import torch.nn.functional as F

def response_logps(logits, input_ids, response_mask):
    # 输入：[B,T,V]、[B,T]、[B,T]
    next_logits = logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    mask = response_mask[:, 1:].bool()
    all_logps = F.log_softmax(next_logits, dim=-1)
    token_logps = all_logps.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_logps, mask  # 均为 [B,T-1]

ids = torch.tensor([[1, 2, 3, 4, 0]])
response_mask = torch.tensor([[0, 0, 1, 1, 0]])
logits = torch.zeros(1, 5, 6, requires_grad=True)
token_logps, mask = response_logps(logits, ids, response_mask)
sequence_logps = (token_logps * mask).sum(-1)
loss = -(token_logps * mask).sum() / mask.sum().clamp_min(1)
loss.backward()
print(sequence_logps.detach(), loss.item())
```

首个回答 token 的概率由它前面的最后一个 prompt 位置预测。不要把首个回答 token 漏掉。
求序列 log probability 用 sum；SFT 的平均 token 损失用有效 token 总数归一化，二者不能随意互换。
`clamp_min(1)` 只是避免除零，实际训练仍要过滤没有有效回答 token 的样本。

## 5. 梯度相关函数

| 接口 | 含义 | 典型使用位置 |
| --- | --- | --- |
| `loss.backward()` | 将梯度累积到参数的 `.grad` | 当前策略更新 |
| `x.detach()` | 返回脱离计算图的张量，可能共享存储 | 缓存旧 logps、优势和奖励 |
| `with torch.no_grad():` | 不记录计算图 | 参考策略、旧策略计算 |
| `with torch.inference_mode():` | 更严格的推理优化模式 | 独立推理；训练数据流初学先用 no_grad |
| `p.requires_grad_(False)` | 冻结参数 | 独立参考模型 |
| `x.item()` | 单元素张量转 Python 数值 | 日志；不能继续用它反向传播 |
| `torch.isfinite(x)` | 检查 NaN/Inf | 排查数值不稳定 |

`model.eval()` 切换 dropout 等层的行为，但不会关闭求导；`no_grad()` 关闭求导，但不会切换 eval。
不要给当前策略需要求梯度的前向传播套 no_grad。
detach 不会复制数据；需要不共享存储时用 `x.detach().clone()`。

## 6. nn.Module、价值头和优化器

可直接运行：用一个隐藏向量预测标量价值。

```python
import torch
from torch import nn

class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        return self.proj(hidden_states).squeeze(-1)  # [B,T,H] -> [B,T]

head = ValueHead(8)
optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
hidden = torch.randn(2, 4, 8)
targets = torch.zeros(2, 4)
optimizer.zero_grad(set_to_none=True)
values = head(hidden)
loss = torch.nn.functional.mse_loss(values, targets)
loss.backward()
torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
optimizer.step()
print(loss.item())
```

`nn.Parameter` 用于直接声明可训练张量；子层赋值给 `self.xxx` 后会被 Module 注册。
优化器负责更新参数，backward 只计算梯度。梯度裁剪应在 backward 后、step 前进行。

梯度累积会跨多个微批次累加梯度；等大小微批次通常每次用 `loss / accumulation_steps`。
不同有效 token 数时，简单平均微批次损失不等于全局 token 平均，需要按目标定义加权。

## 7. 与算法直接对应的函数

| 函数 | 用途 |
| --- | --- |
| `F.logsigmoid(x)` | DPO 的稳定偏好损失 |
| `torch.exp(new_logps - old_logps)` | PPO/GRPO 的概率比 |
| `torch.clamp(ratio, 1-eps, 1+eps)` | 裁剪概率比 |
| `torch.minimum(a, b)` | 取两个 surrogate objective 的较小值 |
| `x.mean(dim, keepdim=True)` | GRPO 组内均值 |
| `x.std(dim, correction=0, keepdim=True)` | 组内标准差，此处采用总体标准差 |
| `torch.where(condition, a, b)` | 按条件选择值 |
| `torch.distributions.Categorical(logits=...)` | 玩具策略的采样、log_prob 和 entropy |

`std` 的 correction 影响算法定义，必须明确，不能依赖默认值。
Categorical 适合小张量学习；语言模型完整生成通常使用 Transformers 的 generate。

## 8. 保存与精度：后续再学

- `model.state_dict()`：模型参数与缓冲区。
- `torch.save(...)`：保存本项目可信 checkpoint；恢复训练还需优化器、调度器、步数和随机数状态。
- `torch.load(path, map_location="cpu", weights_only=True)`：加载适合受限加载器的数据；不要加载来源不明的 pickle。
- `model.load_state_dict(state)`：恢复参数。
- `torch.autocast(...)`：混合精度；FP16 常配合 GradScaler，BF16 是否使用需看硬件支持。
- `torch.cuda.max_memory_allocated()`：本进程张量的峰值显存，不等于 nvidia-smi 的总占用。

## 学习检查

能否在不看代码时写出 `[B,T,V] -> [B,T-1] -> [B]` 的回答概率计算？
能否解释为什么旧策略不保留梯度，而当前策略必须保留？

## 官方资料

- [Tensor API](https://docs.pytorch.org/docs/stable/tensors.html)
- [gather](https://docs.pytorch.org/docs/stable/generated/torch.gather.html)
- [自动求导](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [概率分布](https://docs.pytorch.org/docs/stable/distributions.html)
- [优化器](https://docs.pytorch.org/docs/stable/optim.html)
