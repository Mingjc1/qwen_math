import torch
from qwen_math.models.loader import (
    load_base_model, load_tokenizer, load_adapter,
)
from qwen_math.data.templates import format_prompt

#path = "outputs/sft/adapter"
path = "outputs/sft/first_run_adapter"
tokenizer = load_tokenizer(model_id=path)
model = load_adapter(load_base_model(), path)
model.eval()

batch = tokenizer(
    format_prompt("What is 2+3?"),
    add_special_tokens=False,
    return_tensors="pt",
)
device = next(model.parameters()).device
batch = {k: v.to(device) for k, v in batch.items()}

with torch.no_grad():
    logits = model(**batch).logits

print("输出形状:", logits.shape)
print("数值均有限:", torch.isfinite(logits).all().item())
print("可训练参数量:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))

with torch.no_grad():
    sequences = model.generate(
        **batch,
        max_new_tokens=256,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

prompt_length = batch["input_ids"].shape[1]
answer_ids = sequences[:, prompt_length:]

print("生成回答:")
print(tokenizer.decode(
    answer_ids[0],
    skip_special_tokens=True,
))