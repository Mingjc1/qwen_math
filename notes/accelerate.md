# Accelerate：后续迁移训练循环

当前先学原生 torch。Accelerate 在之后负责设备、混合精度和分布式封装，不能自动替你实现 PPO/GRPO 的 rollout 或损失。

## 1. 常用接口

| 接口 | 用途 |
| --- | --- |
| `Accelerator(...)` | 创建训练辅助对象 |
| `accelerator.prepare(...)` | 包装模型、优化器、数据迭代器等 |
| `accelerator.device` | 当前进程使用的设备 |
| `accelerator.accumulate(model)` | 梯度累积上下文 |
| `accelerator.backward(loss)` | 配合精度与分布式设置反向传播 |
| `accelerator.sync_gradients` | 是否处在梯度同步与更新边界 |
| `accelerator.clip_grad_norm_(...)` | 与精度设置配合的梯度裁剪 |
| `accelerator.unwrap_model(model)` | 取得被包装的底层模型 |
| `accelerator.save_state(path)` | 保存已注册对象等训练状态 |
| `accelerator.load_state(path)` | 恢复相应训练状态 |
| `accelerator.gather_for_metrics(...)` | 收集跨进程评测结果 |

## 2. CPU 可运行的最小示例

需安装 torch 和 accelerate；这里训练的是一个小线性层，不会下载模型。

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator

accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
model = torch.nn.Linear(3, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loader = DataLoader(
    TensorDataset(torch.randn(8, 3), torch.zeros(8, 1)), batch_size=2
)
model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
optimizer.zero_grad(set_to_none=True)
for x, y in loader:
    with accelerator.accumulate(model):
        loss = (model(x)-y).square().mean()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
accelerator.print("完成 CPU 示例")
```

prepare 返回包装后的对象，必须接收并继续使用。
accumulate 配合包装后的优化器控制实际更新；不要照搬原生 torch 的累积代码再手动重复除以累积步数。
这个示例刻意指定 CPU；正式迁移时根据硬件设置 device 和 mixed_precision。

## 3. 服务器阶段再处理

- `accelerate config`：生成启动配置。
- `accelerate launch train.py`：按配置启动脚本；train.py 这里只是文件名示意。
- 多卡时每个进程通常各有模型副本，需考虑数据切分与奖励分组。
- GRPO 的 G 条回答若跨进程，组均值和标准差需要正确跨进程收集。
- 自定义 rollout 缓冲区、题目分组、额外随机状态不一定自动包含在 checkpoint 中，需要显式登记或保存。
- 保存适配器不等于保存可恢复训练的完整状态。

## 官方资料

- [Quicktour](https://huggingface.co/docs/accelerate/quicktour)
- [梯度累积](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation)
- [训练状态保存](https://huggingface.co/docs/accelerate/usage_guides/checkpoint)
