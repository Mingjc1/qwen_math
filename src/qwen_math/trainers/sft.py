"""SFT 训练循环（待实现）。

组织数据/模型、调用 algorithms/sft.py，执行参数更新、日志与 checkpoint。
尚未提供 train()；CLI 在训练模式显式退出，避免把骨架当作可运行训练器。"""

from qwen_math.algorithms.sft import sft_step
import torch
from qwen_math.algorithms.sft import sft_loss
from itertools import islice
from qwen_math.algorithms.sft import sft_backward
import math

def iter_batch_windows(dataloader, accumulation_steps):
    """每次产出最多 accumulation_steps 个 batch。"""
    if accumulation_steps <= 0:
        raise ValueError("累积步数必须大于 0")

    iterator = iter(dataloader)

    while True:
        window = list(islice(iterator, accumulation_steps))

        if not window:
            return

        yield window

def train_sft_epoch(model, dataloader, optimizer,accumulation_steps=1,max_grad_norm=1.0,scheduler=None,on_update=None,skip_updates=0,
                    initial_loss_sum=0.0,initial_token_count=0,
                    ):
    """返回本轮按有效回答 token 加权的训练损失。"""
    device = next(model.parameters()).device
    total_loss = initial_loss_sum
    total_tokens = initial_token_count
    model.train()
    total_updates = math.ceil(
        len(dataloader) / accumulation_steps
    )
    if not 0 <= skip_updates <= total_updates:
        raise ValueError("skip_updates 超出本轮可恢复范围")
    
    if skip_updates > 0 and initial_token_count <= 0:
        raise ValueError("续训需要恢复已完成部分的 token 数")
    for update, window in enumerate(
        iter_batch_windows(dataloader, accumulation_steps),
        start=1,
    ):
        if update <= skip_updates:
            continue
        window_loss = 0.0
        window_tokens = sum(
            batch['response_mask'][:,1:].sum().item()
            for batch in window
        )
        optimizer.zero_grad(set_to_none=True)
        for batch in window:
            batch = {
                k:v.to(device) for k,v in batch.items()
            }
            loss ,count =sft_backward(model,batch,window_tokens)
            window_loss += loss * count
            total_loss +=loss*count
            total_tokens+=count

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=max_grad_norm,
            error_if_nonfinite=True,
        )
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if on_update is not None:
            on_update(update,total_loss, total_tokens)
        print(
            f"update={update}/{total_updates} "
            f"loss={window_loss / window_tokens:.6f} "
            f"lr={lr:.2e}",
            flush=True,
        )
    if total_tokens==0:
        raise ValueError("本轮没有有效回答 token")
    return total_loss / total_tokens

@torch.no_grad()
def evaluate_sft(model, dataloader):
    """返回按有效回答 token 加权的验证损失。"""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    try:
        # 遍历 batch：
        # 1. 将各张量移动到 device
        # 2. 模型前向，取得 logits
        # 3. 调用 sft_loss
        # 4. count = response_mask[:, 1:].sum().item()
        # 5. 累加 loss.item() * count 和 count
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            response_mask = batch['response_mask'].to(device)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            loss = sft_loss(logits,input_ids,response_mask)
            count = response_mask[...,1:].sum().item()
            if count == 0:
                raise ValueError("验证 batch 没有有效回答 token")
            total_loss += loss.item()*count
            total_tokens += count

    finally:
        model.train(was_training)
    # 检查 total_tokens > 0，再返回加权平均
    if total_tokens == 0:
        raise ValueError("验证集没有有效回答 token")
    return total_loss / total_tokens

