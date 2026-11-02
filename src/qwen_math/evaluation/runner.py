"""评测执行器（待实现）。

固定 prompt、数据划分、判分器和生成预算，对 base/SFT/三个分支统一评测。"""
import json
from pathlib import Path

from qwen_math.evaluation.metrics import evaluate_answer
from qwen_math.evaluation.metrics import compute_truncation_rate
from qwen_math.rollout.sampler import generate_response, trim_response


def evaluate_question(
    model, tokenizer, question, expected_answer,
    max_new_tokens=256,
):
    """生成并评估单题；标准答案仅用于判分，不传入生成模型。"""
    _, response_ids = generate_response(
        model=model,
        tokenizer=tokenizer,
        question=question,
        max_new_tokens=max_new_tokens,
    )
    trimmed_ids, terminated = trim_response(response_ids, tokenizer.eos_token_id)
    response_text = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
    return build_eval_record(
        question=question,
        expected_answer=expected_answer,
        response_text=response_text,
        terminated=terminated,
        response_token_count=trimmed_ids.numel(),
    )

def build_eval_record(
    question, expected_answer, response_text,
    terminated, response_token_count,
):
    """
    返回字典，字段：
        question、expected_answer、response_text、
        predicted_answer、format_valid、correct、
        terminated、response_token_count
    """
    # 1. 检查 terminated 为 Python bool
    # 2. 检查 response_token_count 为非负整数
    #    不满足时抛出 ValueError
    # 3. 调用 evaluate_answer，得到 result
    # 4. 将原始输入和 result 中的三个字段组成字典并返回
    if type(terminated) is not bool:
        raise ValueError("terminated 必须是 bool")

    if (
        type(response_token_count) is not int
        or response_token_count < 0
    ):
        raise ValueError("response_token_count 必须是非负整数")

    result = evaluate_answer(response_text, expected_answer)
    result["question"] = question
    result["expected_answer"] = expected_answer
    result["response_text"] = response_text
    result["terminated"] = terminated
    result["response_token_count"] = response_token_count
    return result

def summarize_eval_records(records):
    """
    records：build_eval_record 返回的字典列表。
    返回：
        count、accuracy、format_valid_rate、
        truncation_rate、mean_response_tokens
    """
    if not records:
        raise ValueError("评估记录不能为空")

    count = len(records)
    return {
        "count": count,
        "accuracy": sum(record["correct"] for record in records) / count,
        "format_valid_rate": sum(record["format_valid"] for record in records) / count,
        "truncation_rate": compute_truncation_rate(
            [record["terminated"] for record in records]
        ),
        "mean_response_tokens": sum(
            record["response_token_count"] for record in records
        ) / count,
    }


def save_eval_results(records, output_dir):
    """保存逐题 JSONL 与汇总 JSON，返回指标字典；覆盖同名文件。"""
    metrics = summarize_eval_records(records)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    return metrics


def evaluate_dataset(
    model, tokenizer, records, output_dir,
    max_new_tokens=256,
):
    """按数据顺序逐题生成，保存全部评估结果并返回汇总指标。"""
    if not records:
        raise ValueError("评估数据不能为空")

    predictions = []
    for i, record in enumerate(records, start=1):
        prediction = evaluate_question(
            model=model,
            tokenizer=tokenizer,
            question=record["question"],
            expected_answer=record["answer"],
            max_new_tokens=max_new_tokens,
        )
        predictions.append(prediction)
        print(f"评估进度：{i}/{len(records)}", flush=True)

    return save_eval_results(predictions, output_dir)
