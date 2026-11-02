from qwen_math.models.loader import load_tokenizer
from qwen_math.data.preferences import (
    load_preference_records,
    encode_preference_example,
)
from qwen_math.data.collators import build_preference_dataloader

def main():
    tokenizer = load_tokenizer()
    records = load_preference_records("data/examples/dpo.jsonl")

    # 1. 遍历 records，调用 encode_preference_example
    #    传入 question、chosen、rejected、tokenizer
    #    将结果收集为 samples
    samples = []
    for record in records:
        sample = encode_preference_example(record["question"], record["chosen"], record["rejected"], tokenizer)
        samples.append(sample)

    # 2. 调用 build_preference_dataloader
    #    batch_size=1，shuffle=False
    dataloader = build_preference_dataloader(samples, pad_token_id=tokenizer.pad_token_id, batch_size=1, shuffle=False)

    # 3. batch = next(iter(dataloader))
    # 4. 打印每个张量的名称、shape、dtype
    batch = next(iter(dataloader))
    for name, tensor in batch.items():
        print(f"{name},shape:{tensor.shape},dtype:{tensor.dtype}")

    # 5. 遍历 batch 的每一行：
    #    用 response_mask 取出回答 token
    #    tokenizer.decode(..., skip_special_tokens=True)
    #    打印回答文本
    for i in range(batch["input_ids"].shape[0]):
        response_ids = batch["input_ids"][i][batch["response_mask"][i]]
        text = tokenizer.decode(response_ids, skip_special_tokens=True)
        print(f"第 {i} 条回答：\n{text}")
if __name__ == "__main__":
    main()