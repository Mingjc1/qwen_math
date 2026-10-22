"""从 GSM8K 训练集准备 256 条 SFT 训练数据和 32 条验证数据。"""

import json
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VAL_SIZE = 500
SEED = 42


def convert_example(example):
    """把 GSM8K 记录转换为 question、solution、answer。"""
    question = example["question"].strip()
    raw_answer = example["answer"]
    if "####" not in raw_answer:
        raise ValueError("GSM8K 解答缺少最终答案分隔符 ####")

    solving_process, final_answer = raw_answer.rsplit("####", 1)
    solving_process = solving_process.strip()
    final_answer = final_answer.strip()
    if not question or not solving_process or not final_answer:
        raise ValueError("题目、解题过程或最终答案不能为空")

    return {
        "question": question,
        "solution": solving_process + "\nFinal answer: " + final_answer,
        "answer": final_answer,
    }


def save_jsonl(records, path):
    """创建父目录，将每条记录写成一行 JSON；覆盖已有文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    # 只使用官方训练集，官方测试集留给最终评测。
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    if len(dataset) <=  VAL_SIZE:
        raise ValueError("数据数量不足以划分训练集和验证集")

    # 打乱一次，再取互不重叠的区间；固定种子便于重复生成。
    dataset = dataset.shuffle(seed=SEED)
    train_size = len(dataset) - VAL_SIZE
    train_data = dataset.select(range(train_size))
    val_data = dataset.select(range(train_size, len(dataset)))
    
    train_records = [convert_example(row) for row in train_data]
    val_records = [convert_example(row) for row in val_data]

    # 路径相对于项目根目录，避免启动位置改变输出位置。
    train_path = PROJECT_ROOT / "data/processed/sft_train.jsonl"
    val_path = PROJECT_ROOT / "data/processed/sft_val.jsonl"
    save_jsonl(train_records, train_path)
    save_jsonl(val_records, val_path)

    print(f"训练数量: {len(train_records)}，保存到: {train_path}")
    print(f"验证数量: {len(val_records)}，保存到: {val_path}")
    print("首条训练样本:")
    print(json.dumps(train_records[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
