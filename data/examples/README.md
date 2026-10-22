# 手写格式示例

这三份 JSONL 使用同一道人造小题，仅演示三个数据 schema。
它们不是三套数据划分，不得用于训练/验证/测试对照。

- sft.jsonl：solution 是完整回答，answer 是最终答案。
- dpo.jsonl：chosen/rejected 是同题的完整优选/劣选回答。
- rl.jsonl：answer 只供奖励器使用，不传给策略模型。

正式数据先按题目 id 划分集合，再构造各类训练样本。
