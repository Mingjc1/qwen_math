"""共享张量函数（待实现）。

计划接口：response_logps、masked_mean、masked_sum。
明确 logits/targets 的因果位移；DPO 使用序列求和；mask 排除 prompt 与 padding。"""
import torch
import torch.nn.functional as F


def response_logps(logits, input_ids, response_mask):
    """
    输入：
        logits: [B, T, V]
        input_ids: [B, T]
        response_mask: [B, T]

    返回：
        token_logps: [B, T-1]
        mask: [B, T-1]
    """
    # 1. logits 去掉最后一个位置
    logits = logits[:,:-1,:]
    # 2. input_ids 和 response_mask 去掉第一个位置
    input_ids = input_ids[:,1:]
    response_mask = response_mask[:,1:]
    # 3. log_softmax 得到整个词表的对数概率
    logits = F.log_softmax(logits,dim=-1)
    # 4. gather 取出目标 token 的对数概率
    token_logps = logits.gather(dim=-1,index=input_ids.unsqueeze(-1)).squeeze(-1)
    # 5. 返回 token_logps 和对齐后的 mask
    return token_logps, response_mask.bool()

def masked_sum(values, mask, dim=-1):
    """
    values: 浮点张量，例如 [B, L]
    mask: 同形状，True 表示参与计算
    dim: 沿哪个维度求和

    返回：沿指定维度求和后的张量
    """
    # 1. 将不参与计算的位置置为 0
    values = values.masked_fill(~mask.bool(),0)
    # 2. 沿 dim 求和
    return values.sum(dim=dim)

def masked_mean(values, mask, dim=-1):
    """
    values、mask：同形状
    返回：沿 dim 计算的有效位置平均值
    """
    # 1. 使用 masked_sum 计算有效值之和
    total = masked_sum(values,mask,dim)
    # 2. 计算有效位置数量
    count = mask.bool().sum(dim=dim)
    # 3. 将数量限制为至少 1，避免除零
    count = count.clamp_min(1)
    # 4. 返回平均值
    return total / count
    

    
    
