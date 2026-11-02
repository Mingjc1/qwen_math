"""GRPO 核心计算（待实现）。

按题目分组计算优势，明确标准差 correction、零方差处理与损失归一化。
实现裁剪目标与参考 KL；参考论文与具体变体须在实现时标注。"""
import torch
from qwen_math.common.tensor_ops import masked_mean
from qwen_math.algorithms.ppo import policy_ratio

@torch.no_grad()
def group_advantages(scores, eps=1e-8):
    """
    scores：浮点张量，[B, G]。
        B：题目数量
        G：每道题的回答数量，至少 2
    返回 advantages：[B, G]。
    """
    # 1. 检查 scores 为二维、B > 0、G >= 2
    #    检查 eps > 0，不满足则抛出 ValueError
    # 2. scores = scores.float()
    # 3. 沿最后一维计算 mean，keepdim=True
    # 4. 沿最后一维计算 std：
    #    correction=0，keepdim=True
    # 5. 返回 (scores - mean) / (std + eps)
    if not (
        scores.ndim == 2
        and scores.shape[0] > 0
        and scores.shape[1] >= 2
        and eps > 0
    ):
        raise ValueError("scores 必须为非空 [B, G]，G >= 2，eps > 0")

    scores = scores.float()
    mean_scores = scores.mean(dim=-1, keepdim=True)
    std_scores = scores.std(dim=-1, correction=0, keepdim=True)
    return (scores - mean_scores) / (std_scores + eps)

@torch.no_grad()
def expand_group_advantages(advantages, mask):
    """
    advantages：[B, G]，每条回答的优势。
    mask：[B*G, L]，对齐后的有效回答 token mask。
    行顺序：题目0的G条回答，再题目1的G条回答……

    返回：[B*G, L]，无效位置为0。
    """
    # 1. 检查 advantages、mask 都为二维
    # 2. 检查 advantages.numel() == mask.shape[0]
    #    不满足则抛出 ValueError
    # 3. advantages.reshape(-1, 1)，得到 [B*G, 1]
    # 4. expand_as(mask)，广播为 [B*G, L]
    # 5. masked_fill(~mask, 0)，返回结果
    if not(
        advantages.ndim == 2 and mask.ndim == 2
        and advantages.numel() == mask.shape[0]
    ):
        raise ValueError("advantages、mask 必须为二维,advantages.numel() 必须等于 mask.shape[0]")
    expanded_advantages = advantages.reshape(-1, 1).expand_as(mask)
    return expanded_advantages.masked_fill(~mask, 0)

def grpo_kl_penalty(new_logps, ref_logps, mask):
    """
    输入均为 [B*G, L]。
    返回逐 token KL 惩罚估计，无效位置为 0。
    """
    # 1. log_ratio = ref_logps.detach() - new_logps
    # 2. penalty = torch.exp(log_ratio) - log_ratio - 1
    # 3. 将 ~mask 位置置零并返回
    log_ratio = ref_logps.detach() - new_logps
    penalty = torch.exp(log_ratio) - log_ratio -1
    return penalty.masked_fill(~mask, 0)

def grpo_masked_mean(values, mask):
    """
    values、mask：[B*G, L]。
    返回标量，每条回答权重相同。
    """
    # 1. 检查每行至少有一个有效 token，否则抛出 ValueError
    # 2. masked_mean(values, mask, dim=-1)，得到 [B*G]
    # 3. 对结果调用 .mean() 并返回
    if not mask.any(dim=-1).all():
        raise ValueError("每行至少有一个有效 token")
    return masked_mean(values, mask, dim=-1).mean()

def grpo_policy_loss(
    new_logps, old_logps, advantages, mask,
    clip_epsilon=0.2,
):
    """
    输入均为 [B*G, L]。
    advantages 已广播到 token。
    返回标量损失。
    """
    # 1. ratio = policy_ratio(new_logps, old_logps)
    # 2. advantages = advantages.detach()
    # 3. 原始目标：ratio * advantages
    # 4. 裁剪目标：
    #    ratio.clamp(1-clip_epsilon, 1+clip_epsilon)
    #    再乘 advantages
    # 5. torch.minimum，取两个目标的较小值
    # 6. 返回 -grpo_masked_mean(target, mask)
    ratio = policy_ratio(new_logps, old_logps)
    advantages = advantages.detach()
    origin_target = ratio * advantages
    cliped_target = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages
    target = torch.minimum(origin_target, cliped_target)
    return -grpo_masked_mean(target, mask)

def grpo_loss(
    new_logps, old_logps, ref_logps, advantages, mask,
    clip_epsilon=0.2, kl_coef=0.01,
):
    """返回 total_loss、policy_loss、kl_loss，均为标量张量。"""
    # 1. 检查 kl_coef >= 0，否则抛出 ValueError
    # 2. 调用 grpo_policy_loss，传入 clip_epsilon
    # 3. 调用 grpo_kl_penalty，得到逐 token 惩罚
    # 4. 用 grpo_masked_mean 汇总，得到 kl_loss
    # 5. total_loss = policy_loss + kl_coef * kl_loss
    # 6. 按 total_loss、policy_loss、kl_loss 的顺序返回
    if kl_coef < 0:
        raise ValueError("需要kl_coef >= 0")
    policy_loss = grpo_policy_loss(new_logps, old_logps, advantages, mask, clip_epsilon)
    kl_tokens_loss = grpo_kl_penalty(new_logps, ref_logps, mask)
    kl_loss = grpo_masked_mean(kl_tokens_loss, mask)
    total_loss = policy_loss + kl_loss * kl_coef
    return total_loss, policy_loss, kl_loss