"""DPO 训练循环（待实现）。

组织数据/模型、调用 algorithms/dpo.py，执行参数更新、日志与 checkpoint。
尚未提供 train()；CLI 在训练模式显式退出，避免把骨架当作可运行训练器。"""
import torch
from qwen_math.algorithms.dpo import forward_preference_logps,dpo_loss
def dpo_backward(model, reference_model, batch, beta=0.1):
    """计算 DPO 损失并反向传播，返回日志用数值。"""
    # 1. 调用 forward_preference_logps，获取策略两侧 logps
    # 2. 在 with torch.no_grad(): 中：
    #    调用 forward_preference_logps，获取参考两侧 logps
    # 3. 将四个 logps 和 beta 传给 dpo_loss
    # 4. loss.backward()
    # 5. 返回 loss.item()
    policy_chosen_logps, policy_rejected_logps = forward_preference_logps(model, batch)
    with torch.no_grad():
        ref_chosen_logps, ref_rejected_logps = forward_preference_logps(reference_model, batch)
    loss = dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta)
    loss.backward()
    return loss.item()

def dpo_step(
    model, reference_model, batch, optimizer,
    beta=0.1, max_grad_norm=1.0,
):
    """执行一次参数更新，返回 Python loss 数值。"""
    # 1. model.eval()：关闭 dropout，仍然保留梯度
    # 2. reference_model.eval()
    # 3. optimizer.zero_grad(set_to_none=True)
    # 4. 调用 dpo_backward，传入 beta，取得 loss
    # 5. 收集 model 中 requires_grad=True 的参数
    # 6. 用 torch.nn.utils.clip_grad_norm_ 裁剪梯度
    #    传入 max_grad_norm、error_if_nonfinite=True
    # 7. optimizer.step()
    # 8. 返回 loss
    model.eval()
    reference_model.eval()
    optimizer.zero_grad(set_to_none=True)
    loss = dpo_backward(model, reference_model, batch, beta)
    params = [p for p in model.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(params, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return loss

def build_dpo_optimizer(model, learning_rate=1e-6):
    """返回仅优化策略可训练参数的 AdamW。"""
    # 1. 收集 model 中 requires_grad=True 的参数
    # 2. 参数列表为空时，抛出 ValueError
    # 3. 返回 torch.optim.AdamW：
    #    params=上述参数列表
    #    lr=learning_rate
    #    weight_decay=0.0
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError
    return torch.optim.AdamW(params, learning_rate, weight_decay=0.0)

def train_dpo_epoch(model, reference_model, dataloader, optimizer, beta=0.1):
    """返回按偏好对数量加权的平均训练损失。"""
    device = next(model.parameters()).device
    total_loss = 0.0
    total_pairs = 0

    # 1. 遍历 dataloader
    # 2. 将 batch 中所有张量移动到 device
    # 3. pair_count = batch["input_ids"].shape[0] // 2
    # 4. 调用 dpo_step，传入 beta，得到 loss
    # 5. 累加：
    #    total_loss += loss * pair_count
    #    total_pairs += pair_count
    # 6. total_pairs 为 0 时抛出 ValueError
    # 7. 返回 total_loss / total_pairs
    for batch in dataloader:
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)
        batch["response_mask"] = batch["response_mask"].to(device)
        pair_count = batch["input_ids"].shape[0] // 2
        loss = dpo_step(
            model=model,
            reference_model=reference_model,
            batch=batch,
            optimizer=optimizer,
            beta=beta,
        )
        total_loss += loss * pair_count
        total_pairs += pair_count
    if total_pairs == 0:
        raise ValueError
    return total_loss / total_pairs

@torch.no_grad()
def evaluate_dpo(model, reference_model, dataloader, beta=0.1):
    """返回按偏好对数量加权的验证损失。"""
    was_training = model.training
    model.eval()
    reference_model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_pairs = 0

    try:
        # 1. 遍历 dataloader，将 batch 各张量移动到 device
        # 2. 分别获取策略和参考模型的 chosen、rejected logps
        # 3. 调用 dpo_loss，传入四个 logps 和 beta
        # 4. 统计 pair_count = input_ids 行数 // 2
        # 5. 累加 loss.item() * pair_count 和 pair_count
        for batch in dataloader:
            batch["input_ids"] = batch["input_ids"].to(device)
            batch["attention_mask"] = batch["attention_mask"].to(device)
            batch["response_mask"] = batch["response_mask"].to(device)
            policy_chosen_logps, policy_rejected_logps = forward_preference_logps(model, batch)
            ref_chosen_logps, ref_rejected_logps =forward_preference_logps(reference_model, batch)
            loss = dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta)
            pair_count = batch["input_ids"].shape[0] // 2
            total_loss += pair_count * loss.item()
            total_pairs += pair_count
    finally:
        model.train(was_training)

    # 6. total_pairs 为 0 时抛出 ValueError
    # 7. 返回 total_loss / total_pairs
    if total_pairs ==0:
        raise ValueError
    return total_loss / total_pairs
