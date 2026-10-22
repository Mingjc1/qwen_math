# 起步配置

四个 JSON 文件共享同一份配置 schema，便于初学阶段直接比较，不需要学习 YAML 继承。
只适用于当前项目的骨架；训练器尚未实现，数值未经训练验证。

| 配置 | 数据路径 | 初始化 | 价值头 | 参考 |
| --- | --- | --- | --- | --- |
| sft.json | processed/sft_train.jsonl | 基座 | 无 | 无 |
| dpo.json | preferences/train.jsonl | SFT adapter | 无 | frozen_sft |
| ppo.json | processed/rl_train.jsonl | SFT adapter | 共享骨干 | frozen_sft |
| grpo.json | processed/rl_train.jsonl | SFT adapter | 无 | frozen_sft |

所有相对路径相对于 qwen_math 项目根目录，表中数据路径省略了 data/ 前缀。
真实数据尚不存在；dry-run 不检查数据或 checkpoint 存在性。

## 字段说明

- model_id/revision：统一基座与权重版本，revision=null 表示待服务器阶段锁定。
- dtype：目标计算精度；本地 CPU 小张量练习使用 float32。
- train_data/output_dir：数据输入与产物目录。
- sft_adapter_path：后训练起点。三个分支都指向相同且不可变的 SFT adapter。
- max_prompt_tokens/max_response_tokens：输入/回答预算上限；512 总长会限制部分长题，后续按截断率调整。
- micro_batch_size/gradient_accumulation_steps：训练微批与累积次数；在线 RL 的真实有效批量还取决于 rollout、分组和更新切分。
- learning_rate/max_updates：学习率和真实参数更新次数，不是微批计数。
- lora_* / target_modules：LoRA 设置，注入前检查模型实际层名。
- gradient_checkpointing：重算激活降低内存，使用时需验证适配器梯度。
- reference_policy/cache_reference_logps：参考为冻结 SFT；DPO 可预计算参考概率，RL 在每轮生成后计算。
- reward：PPO/GRPO 使用 answer_correctness；SFT/DPO 的 none 表示训练阶段不调用在线奖励器。
- group_size：GRPO 每题回答数 G，其他分支忽略。
- generation_micro_batch_size：每次实际生成的回答条数，不是每题总回答数；GRPO 先串行生成 G 条。
- rollout_prompts：PPO/GRPO 每轮采样题数；SFT/DPO 忽略。
- temperature/top_p/top_k：第一版固定原始策略分布采样，避免概率比定义混乱。
- dpo_beta：只用于 DPO。
- kl_coef/clip_epsilon/update_epochs：用于 PPO/GRPO，具体损失组合在算法实现时固定并记录。
- gamma/gae_lambda：只用于 PPO 的 GAE。
- value_head：PPO 使用 shared_backbone，其余为 none。

一份公共 schema 中会包含当前算法不用的字段，训练器应按上述约定读取，不能把同名系数跨算法混用。
后续若不同算法配置明显扩大，再拆分专用 dataclass。
