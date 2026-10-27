# PPO 训练入口

从项目根目录运行 `scripts/train_ppo.py`。入口复用已实现的 PPO 函数：
生成回答 → 规则评分 → 缓存旧概率与价值目标 → PPO 更新。

## 数据与实验解释

可以先复用 `data/processed/sft_train.jsonl`。SFT 学习其中的 `solution`；
PPO 只把 `question` 输入模型，用新生成的回答和 `answer` 计算奖励。
本入口的数据加载器复用 SFT 格式校验，所以输入 JSONL 仍需保留原来的三个字段。

这属于在相同训练题目上继续优化，不能用训练得分声称模型在新题上有提升。
保留原有验证集用于调参，官方测试集仅用于最终比较 SFT 与 PPO；两者使用相同
prompt、答案解析方法、生成长度及解码设置。若需要研究未见训练题目的 RL 效果，
应在 SFT 之前划出独立的 PPO 题目，而不是把已经使用的验证集拿来训练。

## 启动

默认参数在脚本的 `CONFIG`，命令行参数覆盖默认值。例如：

```bash
uv run --locked --extra train python scripts/train_ppo.py \
  --model-id /root/autodl-tmp/qwen_math/models/Qwen2.5-Math-1.5B \
  --sft-adapter-path outputs/sft/gsm8k_run01/best_adapter \
  --output-dir outputs/ppo/gsm8k_run01 \
  --max-iterations 100
```

完全从本地加载时，可在命令前加 `HF_HUB_OFFLINE=1`。
加 `--dry-run` 只校验配置、数据和 SFT 文件，不加载模型或执行训练。
`--config` 接受只包含本脚本 CONFIG 字段的 JSON；它不是旧的 `configs/ppo.json` CLI 骨架接口。

当前实现逐题生成，每次把 `rollout_prompts` 道题的回答拼成一批更新。
每轮独立随机抽题，轮内不重复；`max_iterations` 是采样迭代数，不是 epoch。
默认 100 次、每次 1 题，只是首轮实验规模，不会覆盖全部训练题目。
`update_epochs` 决定每批 rollout 更新多少次，期间旧概率和目标保持固定。
采用恒定学习率；当前入口尚未接入梯度累积、微批更新和自动验证集评估。

## 输出与续训

- `metrics.jsonl`：迭代、更新次数、损失、训练平均奖励、耗时。
- `run_config.json`：参数及训练文件、初始 SFT adapter 的 SHA256。
- `checkpoints/iteration-000010/`：每 `save_steps` 次迭代保存一次。
- `final-iteration-000100/`：训练结束时保存。

每个 checkpoint 包含 adapter、分词器、`value_head.pt`、`training_state.pt` 和配置。
不复制基座和固定参考模型；恢复时仍需要相同的基座、初始 SFT adapter 和训练文件。
`mean_score` 是本次训练采样的答案正确率，不是验证集准确率。
训练损失不是选择“最好模型”的可靠依据，本入口不自动输出 `best_adapter`。

使用与原运行一致的参数，加上：

```bash
--resume-from outputs/ppo/gsm8k_run01/checkpoints/iteration-000010
```

续训在完整迭代边界恢复策略、价值头、优化器、抽题和 token 采样随机状态。
参考模型仍使用最初的 SFT adapter。恒定学习率允许增加 `max_iterations`；
其他配置和数据哈希必须一致。不同硬件和软件环境不保证逐位复现。
中断发生在一次迭代内部时，从最近完整 checkpoint 重做后续迭代。
新实验使用空输出目录，已有 checkpoint 不会被覆盖。
