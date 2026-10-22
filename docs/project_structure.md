# 项目结构与模块职责

```text
qwen_math/
├── README.md
├── pyproject.toml              # 基础包与可选训练依赖
├── configs/                   # 四种算法的独立起步配置
│   ├── sft.json
│   ├── dpo.json
│   ├── ppo.json
│   └── grpo.json
├── docs/                      # 选型、模块边界和路线
├── notes/                     # 库与函数学习笔记
├── scripts/
│   └── train.py               # 配置检查与未来训练入口
├── src/qwen_math/
│   ├── config.py              # 已实现：标准库配置加载与校验
│   ├── cli.py                 # 已实现：dry-run，训练入口暂未接通
│   ├── common/
│   │   ├── tensor_ops.py      # 待实现：mask、logps、归一化
│   │   ├── checkpoint.py      # 待实现：模型/优化器/随机状态
│   │   └── seed.py            # 待实现：种子与复现
│   ├── models/
│   │   ├── loader.py          # 待实现：基座、LoRA、SFT reference
│   │   └── value_head.py      # 待实现：PPO 价值头
│   ├── data/
│   │   ├── schemas.py         # 待实现：SFT/偏好/RL 样本校验
│   │   ├── templates.py       # 待实现：统一文本模板与 token 边界
│   │   ├── collators.py       # 待实现：补齐、labels、回答 mask
│   │   └── preferences.py     # 待实现：同题生成与偏好配对
│   ├── algorithms/
│   │   ├── sft.py             # 待实现：回答 token 的交叉熵
│   │   ├── dpo.py             # 待实现：标准 DPO 损失
│   │   ├── ppo.py             # 待实现：GAE、裁剪策略/价值损失
│   │   └── grpo.py            # 待实现：分组优势与策略目标
│   ├── rollout/
│   │   ├── sampler.py         # 待实现：分组/分批生成、终止信息
│   │   └── buffer.py          # 待实现：旧概率、价值、回报缓存
│   ├── rewards/
│   │   └── math_reward.py     # 待实现：答案提取、等价校验与奖励
│   ├── trainers/
│   │   ├── sft.py             # 待实现：监督训练循环
│   │   ├── dpo.py             # 待实现：偏好训练循环
│   │   ├── ppo.py             # 待实现：Actor-Critic 在线循环
│   │   └── grpo.py            # 待实现：组相对在线循环
│   └── evaluation/
│       ├── metrics.py         # 待实现：准确率、长度与解析失败率
│       └── runner.py          # 待实现：固定集合统一推理
├── data/
│   ├── examples/              # 已提供：少量手写格式示例
│   ├── raw/                   # 原始数据，忽略版本管理
│   ├── processed/             # 划分后的 SFT/RL 数据
│   └── preferences/           # chosen/rejected 偏好数据
├── tests/                     # 后续核心数学与梯度检查
├── models/                    # 权重缓存
├── outputs/                   # checkpoint、日志、评测结果
└── experiments/               # 实验配置快照与比较记录
```

## 模块边界

- algorithms 只做张量计算，不下载模型，不读文件，不记录训练日志。
- trainers 组织模型前向、loss、backward、optimizer 更新和 checkpoint。
- rollout 产生在线样本并保留原题分组、旧概率和终止/截断信息。
- rewards 与 evaluation 共享答案提取和等价校验，防止训练与评测使用不同判分口径。
- common 的 logps 函数由四个分支复用，避免各分支因对齐和 mask 差异形成不公平比较。

## 关键数据流

SFT：原始题解 → 文本模板 → 分词/回答 mask → 当前模型 → 回答交叉熵 → 参数更新。

DPO：问题/优选/劣选 → 当前与参考序列 logps → 偏好损失 → 当前策略更新。

PPO：问题 → 当前策略生成 → 规则奖励 + 参考 logps + 旧 values → GAE/returns → 多轮小批更新。

GRPO：问题 → 每题 G 个回答 → 规则奖励 → 组内优势 → 策略更新。

## 实现顺序与验收

1. common/tensor_ops：用 CPU 玩具张量核对 shift、mask、sum/mean 与梯度。
2. data + models：统一模板、真实 EOS 与 padding 边界，接入 LoRA。
3. SFT：先少量样本过拟合与保存/恢复验证。
4. DPO：核对优选/劣选梯度方向与冻结参考；可缓存参考 logps。
5. GRPO：核对 G 条同题回答分组与零方差处理，再接在线生成。
6. PPO：核对价值对齐、GAE、真实终止与截断，再实现 rollout 重用。
7. evaluation：统一预算和判分，服务器测量显存后开展正式对比。

此轮只建立结构；这些验收任务尚未完成。后续实现时再加入有针对性的测试，不用空测试冒充算法验证。
