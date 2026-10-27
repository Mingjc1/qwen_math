"""PPO 核心计算（待实现）。

实现 GAE、回报、裁剪策略目标与价值损失；显式处理 EOS、截断、padding。
区分缓存 old policy 与固定 SFT reference，约定 KL 只加入一次。"""

import torch
from qwen_math.common.tensor_ops import masked_mean
def policy_ratio(new_logps, old_logps):
    """
    new_logps：当前策略的 token 对数概率，[B, L]
    old_logps：采样时旧策略的 token 对数概率，[B, L]
    返回：概率比，[B, L]
    """
    # 实现 exp(new_logps - old_logps.detach())
    return torch.exp(new_logps - old_logps.detach())

def ppo_policy_loss(
    new_logps, old_logps, advantages, mask,
    clip_epsilon=0.2,
):
    """
    输入均为 [B, L]，mask 为有效回答位置。
    返回：标量损失。
    """
    # 1. 调用 policy_ratio 得到 ratio
    # 2. advantages.detach()，固定优势估计
    # 3. 原始目标：ratio * advantages
    # 4. 裁剪目标：
    #    ratio.clamp(1-clip_epsilon, 1+clip_epsilon)
    #    再乘 advantages
    # 5. 两个目标逐元素取 torch.minimum
    # 6. 返回负的 masked_mean(..., dim=None)
    ratio = policy_ratio(new_logps, old_logps)
    advantages = advantages.detach()
    uncliped_target = ratio * advantages
    cliped_target = ratio.clamp(1-clip_epsilon, 1+clip_epsilon) * advantages
    target = torch.minimum(uncliped_target, cliped_target)
    return -masked_mean(target ,mask ,dim=None)

def ppo_value_loss(values, returns, mask):
    """
    values：价值头的预测，[B, L]
    returns：目标回报，[B, L]
    mask：有效回答位置，[B, L]
    返回：标量损失。
    """
    # 1. returns.detach()，目标回报不参与梯度
    # 2. 计算逐 token 平方误差：
    #    (values - returns) ** 2
    # 3. 返回 0.5 * masked_mean(..., dim=None)
    returns = returns.detach()
    errors = (values - returns) ** 2
    return masked_mean(errors ,mask, dim=None) * 0.5

def compute_td_errors(
    rewards, values, next_values, terminated, gamma=1.0,
):
    """
    输入形状均为 [B, L]。
    rewards：当前 token 的奖励。
    values：生成当前 token 前的状态价值。
    next_values：生成当前 token 后的状态价值。
    terminated：是否真正终止，bool 张量。
    返回：TD 误差，[B, L]。
    """
    # 1. 真正终止后，不再使用下一状态价值
    #    continuation = (~terminated).float()
    # 2. 返回：
    #    rewards + gamma * continuation * next_values - values
    continuation = (~terminated).float()
    return rewards + gamma * continuation * next_values - values

@torch.no_grad()
def compute_gae(
    td_errors, terminated, mask,
    gamma=1.0, gae_lambda=0.95,
):
    """
    输入：[B, L]；terminated、mask 为 bool。
    返回 advantages：[B, L]。
    mask 为 True 的回答 token 必须连续。
    """
    advantages = torch.zeros_like(td_errors)
    running = torch.zeros_like(td_errors[:, 0])

    for t in reversed(range(td_errors.shape[1])):
        continuation = (~terminated[:, t]).float()
        # 2. running = 当前 TD 误差
        #      + gamma * gae_lambda * continuation * running
        running = td_errors[:,t] + gamma * gae_lambda * continuation * running
        # 3. 当前 mask 为 False 时，将 running 置零
        running = torch.where(mask[:,t] ,running ,torch.zeros_like(running))
        # 4. advantages[:, t] = running
        advantages[:,t] = running

    return advantages

@torch.no_grad()
def compute_returns(advantages, old_values, mask):
    """
    输入均为 [B, L]。
    old_values：采样时保存的状态价值。
    返回：目标回报，[B, L]，无效位置为 0。
    """
    # 1. returns = advantages + old_values
    # 2. 使用 masked_fill，将 ~mask 位置置为 0
    # 3. 返回 returns
    returns = advantages + old_values
    returns = torch.masked_fill(returns, ~mask, 0)
    return returns

@torch.no_grad()
def normalize_advantages(advantages, mask, eps=1e-8):
    """输入、输出均为 [B, L]，无效位置输出 0。"""
    # 1. 用 masked_mean(..., dim=None) 计算均值 mean
    # 2. centered = advantages - mean
    # 3. 用 masked_mean(centered.square(), mask, dim=None)
    #    计算方差 var
    # 4. normalized = centered / torch.sqrt(var + eps)
    # 5. 将无效位置置零并返回
    mean = masked_mean(advantages, mask, dim=None)
    centered = advantages - mean
    var = masked_mean(centered.square(), mask, dim=None)
    normlized = centered / torch.sqrt(var + eps)
    return normlized.masked_fill(~mask, 0)

@torch.no_grad()
def compute_kl_rewards(old_logps, ref_logps, mask, kl_coef=0.01):
    """
    输入均为 [B, L]。
    返回逐 token 的 KL 惩罚奖励，无效位置为 0。
    """
    # 1. log_ratio = old_logps - ref_logps
    # 2. rewards = -kl_coef * log_ratio
    # 3. 将 ~mask 位置置零并返回
    log_ratio = old_logps - ref_logps
    rewards = -kl_coef * log_ratio
    return rewards.masked_fill(~mask, 0)

@torch.no_grad()
def add_terminal_rewards(kl_rewards, scores, mask):
    """
    kl_rewards：[B, L]，逐 token 的 KL 奖励
    scores：[B]，每条回答的数学得分
    mask：[B, L]，有效回答位置
    返回：[B, L]，合并后的奖励
    """
    # 1. 检查每条回答至少有一个有效 token，否则抛出 ValueError
    # 2. rewards = kl_rewards.clone()，避免修改输入
    # 3. 构造位置编号：
    #    positions = torch.arange(mask.shape[1], device=mask.device)
    # 4. 将无效位置编号设为 -1，逐行取最大值：
    #    last_indices = positions.expand_as(mask).masked_fill(
    #        ~mask, -1
    #    ).max(dim=1).values
    # 5. 使用行索引，在 last_indices 位置加上 scores
    # 6. 返回 rewards
    if not mask.any(dim=-1).all(dim=-1):
        raise ValueError
    rewards = kl_rewards.clone()
    positions = torch.arange(mask.shape[1], device=mask.device)
    last_indices = positions.expand_as(mask).masked_fill(~mask, -1).max(dim=1).values
    rewards[torch.arange(mask.shape[0], device=mask.device), last_indices] += scores
    return rewards

@torch.no_grad()
def prepare_advantages(
    rewards, old_values, next_values, terminated, mask,
    gamma=1.0, gae_lambda=0.95,
):
    """
    输入均为 [B, L]。
    返回：
        advantages：[B, L]，标准化后供策略训练使用
        returns：[B, L]，供价值头训练使用
    """
    # 1. 调用 compute_td_errors，得到 td_errors
    # 2. 调用 compute_gae，得到原始 advantages
    # 3. 调用 compute_returns：
    #    使用原始 advantages 和 old_values
    # 4. 调用 normalize_advantages
    # 5. 返回 advantages, returns
    td_errors = compute_td_errors(rewards, old_values, next_values, terminated, gamma)
    advantages = compute_gae(td_errors, terminated, mask, gamma, gae_lambda)
    returns = compute_returns(advantages, old_values, mask)
    norm_advantages = normalize_advantages(advantages, mask)
    return norm_advantages, returns

def ppo_loss(
    new_logps, old_logps, advantages,
    values, returns, mask,
    clip_epsilon=0.2, value_coef=0.5,
):
    """返回 total_loss、policy_loss、value_loss，均为标量张量。"""
    # 1. 调用 ppo_policy_loss
    # 2. 调用 ppo_value_loss
    # 3. total_loss = policy_loss + value_coef * value_loss
    # 4. 返回三个 loss，暂时不要 .item() 或 detach()
    policy_loss = ppo_policy_loss(new_logps, old_logps, advantages, mask, clip_epsilon)
    value_loss = ppo_value_loss(values, returns, mask)
    total_loss = policy_loss + value_coef * value_loss
    return total_loss, policy_loss, value_loss