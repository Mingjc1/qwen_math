"""训练批次构造（待实现）。

生成 input_ids、attention_mask、response_mask、labels。
只训练回答，区分真实 EOS 与 padding，截断后无回答的样本必须剔除。"""

import torch


def collate_sft(samples, pad_token_id):
    """
    samples：encode_sft_example 返回的字典列表。
    返回三个形状为 [B, T] 的张量。
    """
    if not samples:
        raise ValueError("samples 不能为空")
    # 1. 找到本批最长 input_ids 的长度 T
    input_ids =[d['input_ids'] for d in samples]
    attention_masks = [d['attention_mask'] for d in samples]
    response_masks = [d['response_mask'] for d in samples]
    max_len = len(input_ids[0])
    for i in input_ids:
        max_len = max(max_len,len(i))
    # 2. 对每条样本进行右侧补齐：
    #    input_ids 补 pad_token_id
    #    attention_mask 补 0
    #    response_mask 补 False
    for i in range(len(input_ids)):
        pad_len = max_len-len(input_ids[i])
        input_ids[i] = input_ids[i]+pad_len*[pad_token_id]
        attention_masks[i] = attention_masks[i]+pad_len*[0]
        response_masks[i] = response_masks[i]+pad_len*[False]
    # 3. 将三个二维列表分别转成张量
    input_ids =torch.tensor(input_ids,dtype=torch.long)
    attention_masks = torch.tensor(attention_masks,dtype=torch.long)
    response_masks = torch.tensor(response_masks,dtype=torch.bool)
    # 4. 返回同名键的字典
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "response_mask": response_masks,
    }

from functools import partial
from torch.utils.data import DataLoader


def build_sft_dataloader(
    samples,
    pad_token_id,
    batch_size=1,
    shuffle=True,
    generator=None,
):
    """
    samples：encode_sft_example 返回的字典列表。
    返回：DataLoader。
    """
    # 1. 使用 partial，为 collate_sft 固定 pad_token_id
    collate_fn = partial(collate_sft,pad_token_id=pad_token_id)
    # 2. 创建并返回 DataLoader：
    #    dataset=samples
    #    batch_size=batch_size
    #    shuffle=shuffle
    #    collate_fn=第一步得到的函数
    #    num_workers=0
    return DataLoader(dataset=samples,batch_size=batch_size,shuffle=shuffle,collate_fn=collate_fn,num_workers=0,generator=generator,)