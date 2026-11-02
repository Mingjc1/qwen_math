from qwen_math.common.seed import set_seed
from qwen_math.data.io import load_sft_records
from qwen_math.data.preferences import (
    build_preference_records,
    save_preference_records,
)
from qwen_math.models.loader import (
    load_tokenizer,
    load_reference_model,
)
def main():
    set_seed(42)

    model_id = "models/Qwen2.5-Math-1.5B"
    adapter_path = "outputs/sft/gsm8k_run02/best_adapter"
    # 上述路径修改为服务器实际路径

    # 1. 用 model_id 加载 tokenizer
    # 2. 用 adapter_path、model_id 加载冻结的 SFT 模型
    # 3. 读取 data/processed/sft_train.jsonl，只取前 20 条
    # 4. 调用 build_preference_records：
    #    num_candidates=4、max_new_tokens=256
    # 5. 打印输入题数、成功配对数
    # 6. 如果结果非空，调用 save_preference_records
    #    保存到 data/preferences/dpo_train_smoke.jsonl
    #    否则打印“未生成有效偏好对”
    tokenizer = load_tokenizer(model_id)
    model = load_reference_model(adapter_path, model_id)
    question_answer_records = load_sft_records("data/processed/sft_train.jsonl")
    preferen_records = build_preference_records(
        model=model,
        tokenizer=tokenizer,
        records=question_answer_records,
        num_candidates=4,
        max_new_tokens=256,
    )
    print(f"输入题数：{len(question_answer_records)},成功配对数{len(preferen_records)}")
    if preferen_records:
        save_preference_records(preferen_records, "data/preferences/dpo_train.jsonl")
    else:
        print("未生成有效偏好对")

if __name__ == "__main__":
    main()