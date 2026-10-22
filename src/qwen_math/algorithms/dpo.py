"""标准 DPO 目标（待实现）。

以 policy/ref 的 chosen/rejected 序列 logps 构造偏好 margin，
使用 -logsigmoid(beta * margin)，reference 不参与求导。"""
