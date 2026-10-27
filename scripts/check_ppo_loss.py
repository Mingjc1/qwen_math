import torch
from qwen_math.algorithms.ppo import ppo_loss

new_logps = torch.tensor(
    [[-1.0, -2.0]], requires_grad=True
)
old_logps = new_logps.detach().clone()
advantages = torch.tensor([[1.0, -1.0]])
values = torch.tensor(
    [[0.0, 0.0]], requires_grad=True
)
returns = torch.tensor([[1.0, 2.0]])
mask = torch.tensor([[True, True]])

loss, policy_loss, value_loss = ppo_loss(
    new_logps, old_logps, advantages,
    values, returns, mask,
)
loss.backward()

print("总损失:", loss.item())
print("策略损失:", policy_loss.item())
print("价值损失:", value_loss.item())
print("策略梯度:", new_logps.grad)
print("价值梯度:", values.grad)