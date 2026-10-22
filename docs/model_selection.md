# 模型选择与单卡方案

## 决定

主模型固定为 **Qwen/Qwen2.5-Math-1.5B**，不带 Instruct 后缀的基座版本。
官方模型卡将其定位为数学专用模型，参数规模适合本项目的学习与单卡实验。
模型权重首次下载时记录实际 commit revision；配置暂时为 null，表示尚未锁定权重版本。

| 选择 | 理由 |
| --- | --- |
| 数学专用基座 | 与数学应用题、基础代数的目标一致 |
| 约 1.5B 规模 | 为 PPO 的价值估计和参考策略留出单卡资源空间 |
| 从基座开始 SFT | 完整经历问题/解答监督学习，再开展偏好与强化学习 |
| 普通 LoRA | 先掌握算法与梯度，不在第一版增加量化依赖 |

Qwen3 的模型在已有库笔记中仍可作为 API 教学示例，不作为主实验模型。
主线代码不使用 Qwen3 的 enable_thinking 参数。此次选型重视可理解的实验流程，不以追求最新型号为目标。

## 输入约定

基座采用项目维护的显式文本格式：

```text
Question:
{question}

Solution:
```

SFT 在 Solution 后接完整解答与真实 EOS，训练损失只覆盖回答。
DPO 的 chosen/rejected 共用完全相同的问题前缀；PPO/GRPO 的生成也使用同一模板。
将 `Final answer: ...` 作为答案提取约定，并记录解析失败。
初期主实验可用 GSM8K 英文原题，避免翻译质量引入变量；中文案例单独记录。

## 标准 RTX 4090 的起步方案

这里按 NVIDIA 官方标准 4090 的 **24GB 显存**规划。租赁平台若标注其他显存容量，需要重新确认实际设备。

| 部分 | 起步约定 |
| --- | --- |
| 权重与计算 | BF16，运行前检查支持情况 |
| LoRA | r=8、alpha=16、dropout=0，先作用于 q_proj/v_proj |
| SFT/DPO 训练微批 | 1，通过梯度累积扩大有效批量 |
| 单序列长度 | prompt 最多 256、回答最多 256，合计最多 512 token |
| GRPO | 每题 G=4，但按 1 条回答的生成微批串行采样 |
| PPO 价值估计 | 当前策略与 value head 共享 transformer 骨干，联合优化 |
| old policy | 采样后缓存旧 logps/values，不常驻一份额外旧模型 |
| reference policy | 独立冻结 SFT 快照，先顺序计算参考 logps，必要时 CPU/GPU 分阶段切换 |
| 奖励 | 标准答案校验，不加载额外大型 reward model |
| 激活开销 | 使用梯度检查点，训练关闭 KV cache，按需分块重算概率 |

共享骨干是 PPO 的一个实现选择，应在报告中注明；若后来改成独立 critic，需重新估算显存和设计对照。
SFT adapter 与基座一起构成参考模型，不能简单禁用 SFT adapter 来冒充 SFT reference。
按本配置初步可行是工程判断，不是已完成的显存测试；长答案、大词表 logits、优化器与临时张量都会额外占用显存。
服务器先做短程测量，再逐步增加长度与批量，不承诺四种算法任意配置都能装入 24GB。

## 对照实验约定

1. 保存 base、SFT、SFT+DPO、SFT+PPO、SFT+GRPO 五类评测结果。
2. 三个后训练分支加载同一份 SFT adapter；参考快照始终冻结。
3. 固定 prompt 模板、测试集、判分器、生成预算。
4. 分别记录准确率、长度、峰值显存、GPU 时间与生成 token 数。
5. 在足够预算下对最终配置重复随机种子；不预设任何方法必然带来提升。

## 来源

- [Qwen2.5-Math-1.5B 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)
- [NVIDIA RTX 4090 官方规格](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)
