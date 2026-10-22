"""PPO 核心计算（待实现）。

实现 GAE、回报、裁剪策略目标与价值损失；显式处理 EOS、截断、padding。
区分缓存 old policy 与固定 SFT reference，约定 KL 只加入一次。"""
