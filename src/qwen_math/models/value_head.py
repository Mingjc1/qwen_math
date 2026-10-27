"""PPO 共享骨干价值头（待实现）。

将 [B,T,H] 隐藏状态映射为 [B,T]，对应动作发生前的状态。
价值头与策略联合训练，并显式保存额外参数。"""
import torch
import torch.nn as nn
from qwen_math.common.tensor_ops import response_logps

class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # 创建 nn.Linear：
        # 输入维度 hidden_size，输出维度 1
        self.proj = nn.Linear(hidden_size ,1)

    def forward(self, hidden_states):
        """
        hidden_states: [B, T, H]
        返回 values: [B, T]
        """
        # 1. 将 hidden_states 传给线性层
        # 2. squeeze(-1)，去掉最后大小为 1 的维度
        # 3. 返回 values
        values = self.proj(hidden_states).squeeze(-1)
        return values

def forward_policy_value(model, value_head, batch):
    """
    返回：
        token_logps、values、mask：均为 [B, T-1]
    """
    # 1. 模型前向，传入：
    #    input_ids=batch["input_ids"]
    #    attention_mask=batch["attention_mask"]
    #    output_hidden_states=True
    #    use_cache=False

    # 2. 调用 response_logps：
    #    使用 outputs.logits、input_ids、response_mask

    # 3. 取最后一层隐藏状态：
    #    hidden = outputs.hidden_states[-1]

    # 4. 将 hidden 转成价值头参数的 dtype：
    #    hidden = hidden.to(next(value_head.parameters()).dtype)

    # 5. values = value_head(hidden)[:, :-1]
    # 6. 返回 token_logps, values, mask
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
    )
    token_logps, mask = response_logps(outputs.logits, batch['input_ids'], batch['response_mask'])
    hidden = outputs.hidden_states[-1]
    hidden = hidden.to(next(value_head.parameters()).dtype)
    values = value_head(hidden)[:,:-1]
    return token_logps, values,  mask

@torch.no_grad()
def rollout_values(model, value_head, batch):
    """
    返回 old_values、next_values，均为 [B, T-1]。
    """
    # 1. model.eval()、value_head.eval()
    # 2. 模型前向，传入 input_ids、attention_mask，
    #    output_hidden_states=True、use_cache=False
    # 3. 取 outputs.hidden_states[-1]
    # 4. 转成价值头参数的 dtype
    # 5. full_values = value_head(hidden)，形状 [B, T]
    # 6. 返回 full_values[:, :-1], full_values[:, 1:]
    model.eval()
    value_head.eval()
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
    )
    hidden = outputs.hidden_states[-1].to(next(value_head.parameters()).dtype)
    full_values = value_head(hidden)
    return full_values[:,:-1], full_values[:,1:]