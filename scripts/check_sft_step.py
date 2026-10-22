from qwen_math.models.loader import load_tokenizer
from qwen_math.data.templates import encode_sft_example
from qwen_math.data.collators import collate_sft

from qwen_math.common.checkpoint import save_adapter

tokenizer = load_tokenizer()

rows = [
    ("What is 2+3?", "2+3=5.\nFinal answer: 5"),
    ("What is 10 minus 4?", "Subtracting 4 from 10 gives 6.\nFinal answer: 6"),
]

samples = [
    encode_sft_example(q, s, tokenizer)
    for q, s in rows
]
batch = collate_sft(samples, tokenizer.pad_token_id)

import torch
from qwen_math.models.loader import load_base_model, add_lora
from qwen_math.algorithms.sft import sft_step

model = add_lora(load_base_model())
batch = {
    key: value.to(next(model.parameters()).device)
    for key, value in batch.items()
}

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=1e-4)

before = trainable[0].detach().clone()
loss = sft_step(model, batch, optimizer)

print("loss:", loss)
print("首个参数变化:",
      (trainable[0].detach() - before).abs().max().item())

save_adapter(model, tokenizer, "outputs/sft/adapter")
