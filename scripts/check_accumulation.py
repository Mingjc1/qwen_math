from copy import deepcopy
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from qwen_math.data.collators import build_sft_dataloader
from qwen_math.trainers.sft import train_sft_epoch

config = Qwen2Config(
    vocab_size=16, hidden_size=8, intermediate_size=16,
    num_hidden_layers=1, num_attention_heads=2,
    num_key_value_heads=1, attention_dropout=0.0,
)
model_a = Qwen2ForCausalLM(config)
model_b = deepcopy(model_a)

samples = []
for ids in [[1, 3, 2], [1, 4, 5, 6, 2]]:
    samples.append({
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "response_mask": [False] + [True] * (len(ids)-1),
    })

for model, batch_size, steps in [
    (model_a, 1, 2),  # 两个微批累积
    (model_b, 2, 1),  # 整批更新
]:
    loader = build_sft_dataloader(
        samples, 0, batch_size=batch_size, shuffle=False
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_sft_epoch(model, loader, optimizer, steps)

difference = max(
    (a.detach()-b.detach()).abs().max().item()
    for a, b in zip(model_a.parameters(), model_b.parameters())
)
print("最大参数差异:", difference)
assert difference < 1e-6