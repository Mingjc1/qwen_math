"""GRPO 训练循环（待实现）。

组织数据/模型、调用 algorithms/grpo.py，执行参数更新、日志与 checkpoint。
尚未提供 train()；CLI 在训练模式显式退出，避免把骨架当作可运行训练器。"""
import torch

from qwen_math.rollout.sampler import collect_logps
from qwen_math.algorithms.grpo import (
    group_advantages,
    expand_group_advantages,
)
from qwen_math.common.tensor_ops import response_logps
from qwen_math.algorithms.grpo import grpo_loss
from qwen_math.rollout.sampler import sample_group_rollouts
from qwen_math.rewards.math_reward import score_responses

@torch.no_grad()
def prepare_grpo_batch(model, reference_model, batch, scores):
    """
    scores：[B, G]，数学得分。
    batch：按题目分组排列的 B*G 条序列。

    返回新字典，增加：
        old_logps、ref_logps、advantages
        三者形状均为 [B*G, T-1]。
    """
    # 1. collect_logps(model, batch) → old_logps、mask
    # 2. collect_logps(reference_model, batch) → ref_logps、_
    # 3. group_advantages(scores) → group_adv
    # 4. expand_group_advantages(group_adv, mask) → advantages
    # 5. result = dict(batch)，避免修改原字典
    # 6. 将上述三个字段加入 result 并返回
    old_logps, mask = collect_logps(model, batch)
    ref_logps, _ =collect_logps(reference_model, batch)
    group_adv = group_advantages(scores)
    advantages = expand_group_advantages(group_adv, mask)
    result = dict(batch)
    result["old_logps"] = old_logps
    result["ref_logps"] = ref_logps
    result["advantages"] = advantages
    return result

def grpo_backward(
    model, batch, clip_epsilon=0.2, kl_coef=0.01,
):
    """反向传播，返回 total、policy、kl 三个日志数值。"""
    # 1. 模型前向，传入：
    #    input_ids=batch["input_ids"]
    #    attention_mask=batch["attention_mask"]
    #    use_cache=False
    # 2. 调用 response_logps → new_logps、mask
    # 3. 调用 grpo_loss，传入：
    #    new_logps、batch["old_logps"]、batch["ref_logps"]、
    #    batch["advantages"]、mask，以及两个系数
    # 4. total_loss.backward()
    # 5. 按 total、policy、kl 顺序返回三个 .item()
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    new_logps, mask = response_logps(outputs.logits, batch["input_ids"], batch["response_mask"])
    total_loss, policy_loss, kl_loss = grpo_loss(
        new_logps=new_logps,
        old_logps=batch["old_logps"],
        ref_logps=batch["ref_logps"],
        advantages=batch["advantages"],
        mask=mask,
        clip_epsilon=clip_epsilon,
        kl_coef=kl_coef,
    )
    total_loss.backward()
    return total_loss.item(), policy_loss.item(), kl_loss.item()

def grpo_step(
    model, batch, optimizer,
    clip_epsilon=0.2, kl_coef=0.01,
    max_grad_norm=1.0,
):
    """更新一次策略，返回 total、policy、kl 三个日志数值。"""
    # 1. model.eval()，关闭 dropout，保留梯度
    # 2. optimizer.zero_grad(set_to_none=True)
    # 3. 调用 grpo_backward，传入两个损失系数
    # 4. 收集 model.parameters() 中 requires_grad=True 的参数
    # 5. torch.nn.utils.clip_grad_norm_：
    #    params、max_grad_norm、error_if_nonfinite=True
    # 6. optimizer.step()
    # 7. 返回三个损失数值
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    total_loss, policy_loss, kl_loss = grpo_backward(model, batch, clip_epsilon, kl_coef)
    params = [p for p in model.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(parameters=params, max_norm=max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return total_loss, policy_loss, kl_loss

def update_grpo(
    model, batch, optimizer, update_epochs=1,
    clip_epsilon=0.2, kl_coef=0.01,
    max_grad_norm=1.0,
):
    """返回多次更新的平均 total、policy、kl loss。"""
    # 1. 检查 update_epochs 为正整数，否则抛出 ValueError
    # 2. 初始化三个损失累计值
    # 3. 循环 update_epochs 次：
    #    调用 grpo_step，传入全部超参数
    #    累加三个损失
    # 4. 分别除以 update_epochs，按 total、policy、kl 返回
    if type(update_epochs) is not int or update_epochs <= 0:
        raise ValueError("update_epochs 必须为正整数")
    total_total_loss, total_policy_loss, total_kl_loss = 0, 0, 0
    for _ in range(update_epochs):
        total_loss, policy_loss, kl_loss = grpo_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            clip_epsilon=clip_epsilon,
            kl_coef=kl_coef,
            max_grad_norm=max_grad_norm,
        )
        total_total_loss += total_loss
        total_policy_loss += policy_loss
        total_kl_loss += kl_loss
    return total_total_loss / update_epochs, total_policy_loss / update_epochs, total_kl_loss / update_epochs

def collect_group_scores(
    model, tokenizer, questions, expected_answers,
    group_size=4, max_new_tokens=256,
):
    """
    返回：
        batch：B*G 条采样序列
        scores：[B, G]
    """
    # 1. 检查 questions、expected_answers 等长且非空
    # 2. 调用 sample_group_rollouts → batch、response_texts
    # 3. 每个 expected_answer 连续重复 group_size 次
    #    得到 expanded_answers，与回答顺序一致
    # 4. 调用 score_responses：
    #    response_texts、expanded_answers、
    #    device=batch["input_ids"].device
    # 5. scores.reshape(len(questions), group_size)
    # 6. 返回 batch、scores
    if not questions or not expected_answers or len(questions) != len(expected_answers):
        raise ValueError("需要questions、expected_answers 等长且非空")
    batch, response_texts =sample_group_rollouts(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        group_size=group_size,
        max_new_tokens=max_new_tokens,
    )
    expanded_answers = []
    for e in expected_answers:
        for _ in range(group_size):
            expanded_answers.append(e)
    scores = score_responses(response_texts=response_texts, expected_answers=expanded_answers, device=batch["input_ids"].device)
    scores = scores.reshape(len(questions), group_size)
    return batch, scores

def train_grpo_iteration(
    model, reference_model, tokenizer,
    questions, expected_answers, optimizer,
    group_size=4, max_new_tokens=256, update_epochs=1,
    clip_epsilon=0.2, kl_coef=0.01,
    max_grad_norm=1.0,
):
    """采样、评分、更新，返回日志字典。"""
    # 1. collect_group_scores → batch、scores
    #    传入 group_size、max_new_tokens
    # 2. prepare_grpo_batch → training_batch
    # 3. update_grpo → loss、policy_loss、kl_loss
    #    传入 update_epochs 和全部更新超参数
    # 4. 返回字典：
    #    loss、policy_loss、kl_loss、
    #    mean_score=scores.mean().item()
    batch, scores = collect_group_scores(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        expected_answers=expected_answers,
        group_size=group_size,
        max_new_tokens=max_new_tokens,
    )
    training_batch = prepare_grpo_batch(model, reference_model, batch, scores)
    loss, policy_loss, kl_loss = update_grpo(
        model=model,
        batch=training_batch,
        optimizer=optimizer,
        update_epochs=update_epochs,
        clip_epsilon=clip_epsilon,
        kl_coef=kl_coef,
        max_grad_norm=max_grad_norm,
    )
    return {
        "loss":loss,
        "policy_loss":policy_loss,
        "kl_loss":kl_loss,
        "mean_score":scores.mean().item(),
    }