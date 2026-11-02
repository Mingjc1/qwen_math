import torch
from qwen_math.algorithms.dpo import dpo_loss

chosen = torch.tensor([-2.0], requires_grad=True)
rejected = torch.tensor([-3.0], requires_grad=True)
ref_chosen = chosen.detach().clone().requires_grad_(True)
ref_rejected = rejected.detach().clone().requires_grad_(True)

loss = dpo_loss(
    chosen, rejected, ref_chosen, ref_rejected,
    beta=0.1,
)

assert loss.ndim == 0
assert torch.allclose(loss, torch.tensor(2.0).log())

loss.backward()

assert torch.allclose(chosen.grad, torch.tensor([-0.05]))
assert torch.allclose(rejected.grad, torch.tensor([0.05]))
assert ref_chosen.grad is None
assert ref_rejected.grad is None

print("loss:", loss.item())
print("chosen 梯度:", chosen.grad)
print("rejected 梯度:", rejected.grad)
print("DPO 损失检查通过")