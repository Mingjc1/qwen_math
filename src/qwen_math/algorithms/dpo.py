"""标准 DPO 目标（待实现）。

以 policy/ref 的 chosen/rejected 序列 logps 构造偏好 margin，
使用 -logsigmoid(beta * margin)，reference 不参与求导。"""
from qwen_math.common.tensor_ops import response_logps, masked_sum
import torch.nn.functional as F

def sequence_logps(logits, input_ids, response_mask):
    """
    logits：[B, T, V]
    input_ids、response_mask：[B, T]
    返回：每条回答的对数概率之和，[B]
    """
    # 1. 调用 response_logps，获取对齐后的 token_logps、mask
    # 2. 使用 masked_sum，沿最后一维求和
    # 3. 返回结果
    token_logps, mask = response_logps(logits, input_ids, response_mask)
    return masked_sum(token_logps, mask, dim=-1)

def preference_margin(
    policy_chosen_logps, policy_rejected_logps,
    ref_chosen_logps, ref_rejected_logps,
):
    """
    输入均为整段回答的对数概率，[B]。
    返回 margin：[B]。
    """
    # 1. policy_gap = 策略 chosen logps - 策略 rejected logps
    # 2. ref_gap = 参考 chosen logps - 参考 rejected logps
    #    参考模型的结果使用 detach()
    # 3. 返回 policy_gap - ref_gap
    policy_gap = policy_chosen_logps - policy_rejected_logps
    ref_gap = ref_chosen_logps.detach() - ref_rejected_logps.detach()
    return policy_gap - ref_gap

def dpo_loss(
    policy_chosen_logps, policy_rejected_logps,
    ref_chosen_logps, ref_rejected_logps,
    beta=0.1,
):
    """输入均为 [B]，返回标量损失。"""
    # 1. 检查 beta > 0，否则抛出 ValueError
    # 2. 调用 preference_margin
    # 3. 计算 -F.logsigmoid(beta * margin)
    # 4. 对 B 对偏好样本取平均并返回
    if beta <= 0:
        raise ValueError("beta必须大于0")
    margin = preference_margin(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps)
    return -F.logsigmoid(beta * margin).mean()

def forward_preference_logps(model, batch):
    """
    batch：collate_preferences 返回的字典。
    返回 chosen_logps、rejected_logps，均为 [B]。
    """
    # 1. 检查 input_ids 的行数为正偶数，否则抛出 ValueError
    # 2. 模型前向，传入：
    #    input_ids=batch["input_ids"]
    #    attention_mask=batch["attention_mask"]
    #    use_cache=False
    # 3. 调用 sequence_logps，得到 [2B]
    # 4. 使用 logps.chunk(2, dim=0) 拆成两半并返回
    num_sequences = batch["input_ids"].shape[0]
    if num_sequences <= 0 or num_sequences % 2 != 0:
        raise ValueError("序列数量必须为正偶数")
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    logps = sequence_logps(outputs.logits, batch["input_ids"], batch["response_mask"])
    return logps.chunk(2, dim=0)