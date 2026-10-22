"""Rollout 缓冲区（待实现）。

保存 tokens/masks、group id、old logps、ref logps、奖励，PPO 另存 old values。
张量 detach；多轮更新期间 old logps 不变。"""
