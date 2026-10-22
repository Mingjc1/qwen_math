# TRL：核对算法的参考库

TRL 提供语言模型后训练的训练器。对于本项目，它首先用于对照数据格式、损失定义与训练指标。
自己实现算法时，主训练循环仍应明确展示采样、概率计算、优势与更新过程。

## 1. 需要认识的类

| 类 | 职责 | 要核对的内容 |
| --- | --- | --- |
| `SFTConfig / SFTTrainer` | 监督微调 | 回答 mask、聊天模板、截断 |
| `DPOConfig / DPOTrainer` | 偏好优化 | beta、参考策略、序列概率计算 |
| `GRPOConfig / GRPOTrainer` | 组相对策略优化 | 分组生成、奖励标准化、KL、loss_type |
| `PPOConfig / PPOTrainer` | PPO 流程 | 版本支持、actor/critic、奖励接口 |

Config 表示配置对象，Trainer 执行数据处理与训练。类名不意味着它们在所有版本中都位于相同导入路径。
截至 2026-10-21，官方稳定文档的 PPO 入口与 main 文档有差异；实现时必须先固定版本，核对实际源码与示例。
本篇因此不提供一个未经本项目环境验证的 PPOTrainer 调用模板。

## 2. 读懂训练器调用的组成

以下是概念示意，不可直接运行，也不保证对应所有版本签名：

```text
trainer = 某个Trainer(
    model = 当前策略,
    args = 对应Config(...),
    train_dataset = 已适配该训练器格式的数据,
    processing_class = tokenizer,
    其他算法专用参数...
)
trainer.train()
```

以前版本可能使用 tokenizer 参数名，其他版本使用 processing_class；不要拼接多个年代的教程。
调用 train 能跑通，不代表自己已经实现或理解算法。

## 3. GRPO 自定义奖励函数

只演示可直接执行的纯 Python 函数，不实例化 Trainer：

```python
def exact_reward(completions, answer, **kwargs):
    # 本例只接受纯字符串回答；正式数学判分需提取、归一化与等价校验
    if len(completions) != len(answer):
        raise ValueError("回答与标准答案未对齐")
    return [float(text.strip() == gold.strip())
            for text, gold in zip(completions, answer)]

print(exact_reward(["5", "6"], ["5", "5"]))
```

接入实际 GRPOTrainer 时需核对 completions 是字符串还是对话消息列表、标准答案列如何扩展到 G 条回答。
奖励函数最好可独立验证，不能直接用完整数学解答字符串做简单相等判断。

## 4. 对照源码时按这个顺序阅读

1. 输入 schema：chosen/rejected 是字符串还是消息列表？
2. 分词与截断：prompt 和回答边界如何确定？
3. 概率：如何错位对齐、mask，sum 还是 mean？
4. 优势：组内如何归一化，零方差如何处理？
5. 目标：有无 KL、裁剪，按 token 还是序列平均？
6. 更新：旧策略在哪一步固定，rollout 被重用多少次？

GRPO 名称下可能存在多种 loss_type、归一化与偏差修正方式。
先选一个明确目标做基线，再添加变体，报告中记录定义。

## 官方资料

- [SFT](https://huggingface.co/docs/trl/sft_trainer)
- [DPO](https://huggingface.co/docs/trl/dpo_trainer)
- [GRPO](https://huggingface.co/docs/trl/grpo_trainer)
- [PPO main 文档](https://huggingface.co/docs/trl/main/ppo_trainer)
- [TRL 源码](https://github.com/huggingface/trl)
