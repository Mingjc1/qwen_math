# 用库函数串起 DPO、PPO、GRPO

本篇所有算法函数都是项目学习示例，不是库自带 API。代码只演示核心计算，不是完整训练器。
示例可在 CPU 运行，不需要 Qwen 权重。先读 torch.md 中的 log probability 与 mask。

## 1. DPO：让优选回答相对更可能

标准 DPO 使用回答序列的 log probability 总和，分别计算当前策略和冻结参考策略。
设 chosen 为优选，rejected 为劣选：

```text
margin = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
loss = -log_sigmoid(beta * margin)
```

可直接运行：

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    # 四个输入均为 [B]，单位是序列 log probability
    policy_margin = policy_chosen - policy_rejected
    ref_margin = (ref_chosen - ref_rejected).detach()
    return -F.logsigmoid(beta * (policy_margin-ref_margin)).mean()

chosen = torch.tensor([-2.0, -3.0], requires_grad=True)
rejected = torch.tensor([-4.0, -5.0], requires_grad=True)
ref_chosen = torch.tensor([-3.0, -3.0])
ref_rejected = torch.tensor([-4.0, -4.0])
loss = dpo_loss(chosen, rejected, ref_chosen, ref_rejected)
loss.backward()
print(loss.item(), chosen.grad, rejected.grad)
```

chosen.grad 应为负，梯度下降会增大 chosen 的 log probability；rejected 的方向相反。
beta 控制偏好 logit 的缩放，不能直接当成 PPO 中同名 KL 系数比较。
不要把总 log probability 改成平均值后仍称为完全相同的标准 DPO 目标。

## 2. PPO/GRPO 共享的裁剪策略目标

可直接运行：

```python
import torch

def clipped_policy_loss(new_logps, old_logps, advantages, mask, epsilon=0.2):
    # logps、mask: [B,L]，L 为回答 token 槽位数
    # advantages: [B,L] 或 [B,1]
    ratio = torch.exp(new_logps - old_logps.detach())
    objective = ratio * advantages.detach()
    clipped = ratio.clamp(1-epsilon, 1+epsilon) * advantages.detach()
    token_loss = -torch.minimum(objective, clipped)
    # 这里选择整批有效 token 平均；其他归一化方法会改变目标权重
    return (token_loss * mask).sum() / mask.sum().clamp_min(1)

old = torch.tensor([[-1.0, -2.0], [-2.0, -1.0]])
new = (old + 0.05).clone().requires_grad_(True)
adv = torch.tensor([[1.0], [-1.0]])
mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)
loss = clipped_policy_loss(new, old, adv, mask)
loss.backward()
print(loss.item(), new.grad)
```

old_logps 来自采样时的策略快照，在一轮 rollout 的多个更新步骤中保持不变。
reference_logps 来自固定参考策略，用于控制漂移；它与 old_logps 是两个概念。
不要在更新后重算 old_logps，或令它直接等于当前 new_logps。
实际模型若包含 dropout，应统一采样与重算的随机层行为；eval 模式仍可求梯度。

此函数仅为策略项，未包含 PPO 价值损失、熵项或 KL 项，也未指定完整 GRPO 的序列归一化约定。

## 3. GRPO：计算组内优势

对同一道题采样 G 个回答，按组计算相对奖励。下面固定采用总体标准差。

可直接运行：

```python
import torch

def group_advantages(rewards, epsilon=1e-6):
    # [题目数,G]，每行必须属于同一道题
    rewards = rewards.float()
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, correction=0, keepdim=True)
    return ((rewards-mean) / (std+epsilon)).detach()

rewards = torch.tensor([[1., 0., 1., 0.], [0., 0., 0., 0.]])
advantages = group_advantages(rewards)
print(advantages)
token_advantages = advantages.reshape(-1, 1)  # [题目数*G,1]
print(token_advantages.shape)
```

第二组优势全为 0，不能为该组提供相对奖励梯度；如果有 KL 项，KL 仍可能产生梯度。
展开回答时必须保持同一顺序，例如“题 1 的 G 条，题 2 的 G 条”。不能 shuffle 后仍按原行分组。
组大小为 1 时没有相对奖励信息。实际训练记录奖励标准差和零方差组比例。

## 4. KL 项：先分清三套概率

常见采样 KL 估计表达式如下，片段中每项均为有效 token 上的 log probability：

```python
# 片段：logp_ref 是冻结参考，logp_new 是当前可训练策略
delta = logp_ref.detach() - logp_new
kl_estimate = torch.exp(delta) - delta - 1
# 可将 beta * kl_estimate 加到对应 token 的策略损失，再按约定归一化
```

这是基于采样动作的估计项，不是枚举整个词表后计算的精确 KL。
其期望、梯度与采用的采样分布有关；脱策略更新需要结合明确的目标与重要性权重推导。
初版选定一份论文或框架定义后保持一致，不要把 PPO 奖励内的 KL 惩罚与额外 KL loss 无意重复加入。

## 5. PPO：用 GAE 连接奖励与价值

PPO 需要估计每个动作发生前状态的价值，并构造优势与回报目标。
下面是无 padding 的玩具轨迹：done=True 表示该动作后真实终止，不再 bootstrap。

可直接运行：

```python
import torch

@torch.no_grad()
def gae(rewards, values, dones, gamma=1.0, lam=0.95):
    # rewards,dones: [B,L]；values: [B,L+1]
    # values[:,t] 是生成动作 t 前的状态价值，最后一列是 bootstrap 值
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(rewards.shape[1])):
        alive = (~dones[:, t]).float()
        delta = rewards[:, t] + gamma*values[:, t+1]*alive - values[:, t]
        running = delta + gamma*lam*alive*running
        advantages[:, t] = running
    returns = advantages + values[:, :-1]
    return advantages, returns

rewards = torch.tensor([[0., 0., 1.]])
values = torch.zeros(1, 4)
dones = torch.tensor([[False, False, True]])
advantages, returns = gae(rewards, values, dones)
print(advantages, returns)
```

这个例子不支持变长 padding；正式实现必须按轨迹长度计算或显式加入有效位置 mask。
真实 EOS 终止与到长度上限截断要区分；是否 bootstrap 由任务终止定义决定。
transformer 价值头的输出位置也需和“动作之前的状态”正确对齐。

价值损失片段：

```python
# values_new、returns、mask 均为 [B,L]
value_loss = ((values_new - returns.detach()).square() * mask).sum() / mask.sum().clamp_min(1)
# total_loss = policy_loss + value_coef * value_loss
```

完整 PPO 还需设计 rollout 缓冲区、多轮小批次更新、奖励处理、价值裁剪等具体约定。

## 6. 正式模型之前的验收

1. DPO 优选/劣选梯度方向正确。
2. old_logps 和参考模型没有梯度，当前策略有梯度。
3. masked 位置不影响损失。
4. GRPO 零方差组产生有限值，组索引正确。
5. GAE 能手算核对终止轨迹，截断与补齐另有用例。

这些检查不需要大规模训练，是本地实现阶段的主要任务。

## 参考定义

- [DPO 论文](https://arxiv.org/abs/2305.18290)
- [PPO 论文](https://arxiv.org/abs/1707.06347)
- [GAE 论文](https://arxiv.org/abs/1506.02438)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [TRL GRPO 实现与目标变体](https://huggingface.co/docs/trl/grpo_trainer)
