"""SFT 回答损失（待实现）。

对真实回答 token 计算交叉熵，使用有效 token 总数归一化。
若使用模型内置 labels loss，避免手工再次 shift。"""

from qwen_math.common.tensor_ops import (
    response_logps,
    masked_mean,
)


def sft_loss(logits, input_ids, response_mask):
    """
    logits: [B, T, V]
    input_ids、response_mask: [B, T]
    返回：标量损失
    """
    # 1. 调用 response_logps，取得对齐后的概率和 mask
    token_logps,mask =response_logps(logits,input_ids,response_mask)
    # 2. 取负号，使用 masked_mean 计算平均值
    return masked_mean(-token_logps,mask,dim=None)

def sft_step(model, batch, optimizer):
    """
    batch 包含 input_ids、attention_mask、response_mask。
    返回：用于日志的 Python 数值。
    """
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    response_mask = batch['response_mask']
    # 1. model.train()，清空旧梯度
    model.train()
    # 2. 将 input_ids、attention_mask 传给模型
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    #    使用 use_cache=False，不传 response_mask
    # 3. 将 outputs.logits 传入 sft_loss
    loss = sft_loss(outputs.logits,input_ids,response_mask)
    # 4. loss.backward()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # 5. optimizer.step()
    optimizer.step()
    # 6. 返回 loss.item()
    return loss.item()

def sft_backward(model, batch, window_token_count):
    """
    window_token_count：本组累积微批的有效回答 token 总数。
    返回：当前微批的原始平均 loss、有效 token 数。
    """
    # 1. 统计当前微批的有效 token 数 count
    # 2. 检查 count 和 window_token_count 均大于 0
    # 3. 模型前向，调用 sft_loss
    # 4. 按下面的权重反向传播：
    #    (loss * count / window_token_count).backward()
    # 5. 返回 loss.item(), count
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    response_mask = batch['response_mask']
    count = response_mask[:,1:].sum().item()
    if count <= 0 or window_token_count <= 0:
        raise ValueError("微批和累积窗口的有效 token 数必须大于 0")
    logits = model(input_ids=input_ids,
                   attention_mask=attention_mask,
                   use_cache=False
                   ).logits
    loss = sft_loss(logits,input_ids,response_mask)
    (loss*count/window_token_count).backward()
    return loss.item(),count
    
    
    
