import torch
from qwen_math.algorithms.grpo import (
    group_advantages, expand_group_advantages, grpo_loss,
)

scores = torch.tensor([[1.0, 0.0]])
mask = torch.tensor([[True, True], [True, False]])

group_adv = group_advantages(scores)
advantages = expand_group_advantages(group_adv, mask)

new = torch.zeros(2, 2, requires_grad=True)
old = torch.zeros(2, 2, requires_grad=True)
ref = torch.zeros(2, 2, requires_grad=True)

total, policy, kl = grpo_loss(new, old, ref, advantages, mask)
total.backward()

assert torch.allclose(group_adv, torch.tensor([[1., -1.]]))
assert torch.allclose(total, torch.tensor(0.))
assert torch.allclose(
    new.grad,
    torch.tensor([[-0.25, -0.25], [0.5, 0.]]),
)
assert old.grad is None
assert ref.grad is None

print("总损失:", total.item())
print("策略梯度:", new.grad)
print("GRPO 基础损失检查通过")