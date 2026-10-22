# 算法实现后的检查位置

当前没有算法实现，因此尚未创建或宣称通过算法测试。
后续只加入能验证关键数学行为的检查：

- test_tensor_ops.py：因果对齐、首个回答 token、padding mask、梯度。
- test_dpo.py：优选/劣选梯度方向、参考冻结。
- test_grpo.py：同题分组、零方差组与损失归一化。
- test_ppo.py：GAE 手算、真实终止/截断、旧概率固定。
- test_collators.py：真实 EOS 与 padding 相同编号、截断无回答样本。

这些小张量测试以 CPU 执行为主，不依赖模型下载或昂贵训练。
