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
batch = collate_sft(samples, tokenizer.pad_token_id)

for name, tensor in batch.items():
    print(name, tensor.shape, tensor.dtype)

assert (
    batch["input_ids"].shape
    == batch["attention_mask"].shape
    == batch["response_mask"].shape
)
assert not (
    batch["response_mask"]
    & ~batch["attention_mask"].bool()
).any()

print(tokenizer.decode(samples[0]["input_ids"]))
print("数据检查通过")