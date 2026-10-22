import torch
from qwen_math.data.collators import build_sft_dataloader
from qwen_math.models.loader import load_base_model, add_lora
from qwen_math.trainers.sft import train_sft_epoch
from qwen_math.common.checkpoint import save_adapter
from qwen_math.models.loader import load_tokenizer
from qwen_math.data.templates import encode_sft_example
from qwen_math.data.collators import collate_sft

tokenizer = load_tokenizer()

rows = [
    ("What is 2+3?", "2+3=5.\nFinal answer: 5"),
    ("What is 10 minus 4?", "Subtracting 4 from 10 gives 6.\nFinal answer: 6"),
]

samples = [
    encode_sft_example(q, s, tokenizer)
    for q, s in rows
]

loader = build_sft_dataloader(
    samples, tokenizer.pad_token_id, batch_size=1
)
model = add_lora(load_base_model())

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4,
)

loss = train_sft_epoch(model, loader, optimizer)
print("epoch loss:", loss)

save_adapter(model, tokenizer, "outputs/sft/smoke_adapter")