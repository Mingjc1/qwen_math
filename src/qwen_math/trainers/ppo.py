"""PPO 训练循环（待实现）。

组织数据/模型、调用 algorithms/ppo.py，执行参数更新、日志与 checkpoint。
尚未提供 train()；CLI 在训练模式显式退出，避免把骨架当作可运行训练器。"""
import torch
from qwen_math.models.value_head import forward_policy_value,rollout_values
from qwen_math.algorithms.ppo import ppo_loss,compute_kl_rewards,add_terminal_rewards,prepare_advantages
from qwen_math.rollout.sampler import collect_logps,sample_rollout_batch
from qwen_math.rewards.math_reward import score_responses
def build_ppo_optimizer(
    model, value_head, learning_rate=1e-6,
):
    """同时优化策略的可训练参数与独立价值头。"""
    # 1. 收集 model 中 requires_grad=True 的参数
    # 2. 收集 value_head 中 requires_grad=True 的参数
    # 3. 合并两个参数列表
    # 4. 如果列表为空，抛出 ValueError
    # 5. 返回 torch.optim.AdamW：
    #    params=合并后的列表
    #    lr=learning_rate
    #    weight_decay=0.0
    model_params = [p for p in model.parameters() if p.requires_grad]
    value_head_params = [p for p in value_head.parameters() if p.requires_grad]
    params = model_params + value_head_params
    if not params:
        raise ValueError
    return torch.optim.AdamW(params=params,lr=learning_rate,weight_decay=0.0)

def ppo_backward(
    model, value_head, batch,
    clip_epsilon=0.2, value_coef=0.5,
):
    """
    batch 包含模型输入，以及固定的：
    old_logps、advantages、returns，形状为 [B, T-1]。
    返回用于日志的三个 Python 数值。
    """
    # 1. 调用 forward_policy_value
    # 2. 调用 ppo_loss，传入：
    #    当前 logps、batch["old_logps"]、
    #    batch["advantages"]、当前 values、
    #    batch["returns"]、对齐后的 mask 和两个系数
    # 3. total_loss.backward()
    # 4. 返回三个 loss.item()，顺序与 ppo_loss 相同
    new_logps, values, mask =forward_policy_value(model,value_head,batch)
    total_loss, policy_loss, value_loss = ppo_loss(new_logps, batch['old_logps'], batch['advantages'], values, batch['returns'], mask, clip_epsilon,value_coef)
    if not torch.isfinite(total_loss):
        raise FloatingPointError("PPO 损失不是有限值，未执行反向传播和参数更新")
    total_loss.backward()
    return total_loss.item(), policy_loss.item(), value_loss.item()

def ppo_step(
    model, value_head, batch, optimizer,
    clip_epsilon=0.2, value_coef=0.5,
    max_grad_norm=1.0,
):
    """单批次更新，返回三个用于日志的 loss 数值。"""
    # 1. model.eval()：关闭 dropout，但保留梯度
    # 2. value_head.train()
    # 3. optimizer.zero_grad(set_to_none=True)
    # 4. 调用 ppo_backward，接收三个损失
    # 5. 收集 model 和 value_head 的可训练参数
    # 6. torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    # 7. optimizer.step()
    # 8. 返回三个损失
    model.eval()
    value_head.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, policy_loss, value_loss =ppo_backward(model,value_head,batch,clip_epsilon,value_coef)
    model_params = [p for p in model.parameters() if p.requires_grad]
    value_head_params = [p for p in value_head.parameters() if p.requires_grad]
    params = model_params + value_head_params
    torch.nn.utils.clip_grad_norm_(params,max_grad_norm,error_if_nonfinite=True)
    optimizer.step()
    return total_loss, policy_loss, value_loss

@torch.no_grad()
def prepare_rollout_batch(
    model, reference_model, value_head, batch, scores,
    kl_coef=0.01, gamma=1.0, gae_lambda=0.95,
):
    """
    batch：采样后的四个 [B, T] 张量。
    scores：[B]。
    返回新字典，增加 old_logps、advantages、returns。
    """
    # 1. collect_logps(model, batch) → old_logps、mask
    # 2. collect_logps(reference_model, batch) → ref_logps、_
    # 3. rollout_values(model, value_head, batch)
    #    → old_values、next_values
    # 4. compute_kl_rewards → kl_rewards
    # 5. add_terminal_rewards → rewards
    # 6. prepare_advantages → advantages、returns
    #    使用 batch["terminated"][:, 1:] 和 mask
    # 7. result = dict(batch)，避免修改原字典
    # 8. 将 old_logps、advantages、returns 写入 result 并返回
    old_logps, mask = collect_logps(model, batch)
    ref_logps, _ = collect_logps(reference_model, batch)
    old_values, next_values = rollout_values(model, value_head, batch)
    kl_rewards = compute_kl_rewards(old_logps, ref_logps, mask, kl_coef)
    rewards = add_terminal_rewards(kl_rewards, scores, mask)
    advantages, returns = prepare_advantages(rewards, old_values, next_values, batch["terminated"][:,1:], mask, gamma, gae_lambda)
    result =dict(batch)
    result["old_logps"] = old_logps
    result["advantages"] = advantages
    result["returns"] = returns 
    return result

def update_ppo(
    model, value_head, batch, optimizer,
    update_epochs=1,
    clip_epsilon=0.2, value_coef=0.5,
    max_grad_norm=1.0,
):
    """返回多次更新的平均 total、policy、value loss。"""
    # 1. update_epochs 必须是正整数，否则抛出 ValueError
    # 2. 初始化三个损失累计值
    # 3. 循环 update_epochs 次：
    #    调用 ppo_step，传入各超参数
    #    累加返回的三个损失
    # 4. 分别除以 update_epochs，返回三个平均值
    if (
        isinstance(update_epochs, bool)
        or not isinstance(update_epochs, int)
        or update_epochs <= 0
    ):
        raise ValueError("update_epochs 必须为正整数")
    total_total_loss, total_policy_loss, total_value_loss = 0, 0, 0
    for _ in range(update_epochs):
        total_loss, policy_loss, value_loss = ppo_step(
            model=model,
            value_head=value_head,
            batch=batch,
            optimizer=optimizer,
            clip_epsilon=clip_epsilon,
            value_coef=value_coef,
            max_grad_norm=max_grad_norm,
        )
        total_total_loss += total_loss
        total_policy_loss += policy_loss
        total_value_loss += value_loss
    return total_total_loss / update_epochs, total_policy_loss / update_epochs, total_value_loss / update_epochs

def train_ppo_iteration(
    model, reference_model, value_head, tokenizer,
    questions, expected_answers, optimizer,
    max_new_tokens=256, update_epochs=1,
    kl_coef=0.01, gamma=1.0, gae_lambda=0.95,
    clip_epsilon=0.2, value_coef=0.5,
    max_grad_norm=1.0,
):
    # 1. 检查 questions 与 expected_answers 等长且非空
    # 2. sample_rollout_batch → batch、response_texts
    # 3. score_responses → scores，放到 batch 所在设备
    # 4. prepare_rollout_batch → training_batch
    # 5. update_ppo → 三个平均损失
    # 6. 返回日志字典：
    #    loss、policy_loss、value_loss、
    #    mean_score（scores.mean().item()）
    if not questions or not expected_answers or len(questions) != len(expected_answers):
        raise ValueError
    batch, response_texts = sample_rollout_batch(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        max_new_tokens=max_new_tokens,
    )
    scores = score_responses(response_texts, expected_answers, device=batch["input_ids"].device)
    training_batch = prepare_rollout_batch(
        model=model,
        reference_model=reference_model,
        value_head=value_head,
        batch=batch,
        scores=scores,
        kl_coef=kl_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    loss, policy_loss, value_loss = update_ppo(
        model=model,
        value_head=value_head,
        batch=training_batch,
        optimizer=optimizer,
        update_epochs=update_epochs,
        clip_epsilon=clip_epsilon,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm,
    )
    return {
        "loss":loss,
        "policy_loss":policy_loss,
        "value_loss":value_loss,
        "mean_score":scores.mean().item()
    }
