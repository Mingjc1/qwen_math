"""偏好数据生成（待实现）。

从同一 SFT 策略对训练题采样，经答案校验选 chosen/rejected。
保留题目 id 与采样来源，不从测试集构造偏好对。"""

import json
from qwen_math.rollout.sampler import generate_response, trim_response
from qwen_math.rewards.math_reward import answer_correctness
from qwen_math.data.templates import encode_sft_example
from pathlib import Path

def load_preference_records(path):
    """读取偏好 JSONL，校验必需文本字段并保留其他字段。

    跳过空白行；无效记录的错误信息包含文件路径和原始行号。
    """
    records = []
    required_fields = ("question", "chosen", "rejected")

    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            location = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{location}: JSON 格式错误：{error.msg}") from error

            if not isinstance(record, dict):
                raise ValueError(f"{location}: 每行必须是一个 JSON 对象")

            for field in required_fields:
                if field not in record:
                    raise ValueError(f"{location}: 缺少字段 {field}")
                if not isinstance(record[field], str) or not record[field].strip():
                    raise ValueError(f"{location}: 字段 {field} 必须是非空字符串")

            records.append(record)

    if not records:
        raise ValueError(f"{path}: 数据文件没有有效样本")
    return records


def encode_preference_example(
    question, chosen, rejected, tokenizer,
):
    """
    返回：
        {
            "chosen": 编码后的字典,
            "rejected": 编码后的字典,
        }
    每个字典包含：
        input_ids、attention_mask、response_mask
    """
    # 1. 调用 encode_sft_example，编码 question + chosen
    # 2. 再调用一次，编码 question + rejected
    # 3. 按上述结构返回
    chosen_dict = encode_sft_example(question, chosen, tokenizer)
    rejected_dict =encode_sft_example(question, rejected, tokenizer)
    return {
        "chosen":chosen_dict,
        "rejected":rejected_dict,
    }

def filter_long_preferences(samples, max_length=512):
    if max_length <= 0:
        raise ValueError("max_length 必须大于 0")

    filtered = []
    for sample in samples:
        chosen_length = len(sample["chosen"]["input_ids"])
        rejected_length = len(sample["rejected"]["input_ids"])

        if chosen_length <= max_length and rejected_length <= max_length:
            filtered.append(sample)

    if not filtered:
        raise ValueError("过滤后没有有效偏好对")

    return filtered

def select_preference_pair(response_texts, scores):
    if len(response_texts) != len(scores):
        raise ValueError("回答与得分数量不一致")

    chosen, rejected = None, None

    for text, score in zip(response_texts, scores):
        if score == 1.0 and chosen is None:
            chosen = text
        elif score == 0.0 and rejected is None:
            rejected = text

    if chosen is not None and rejected is not None:
        return chosen, rejected
    return None

def generate_preference_pair(
    model, tokenizer, question, expected_answer,
    num_candidates=4, max_new_tokens=256,
):
    """返回 (chosen, rejected)，无法配对则返回 None。"""
    # 1. 检查 num_candidates >= 2，否则抛出 ValueError
    # 2. 创建 response_texts、scores 两个列表
    # 3. 循环 num_candidates 次：
    #    a. generate_response，传入 max_new_tokens
    #    b. trim_response，得到 trimmed_ids、terminated
    #    c. 若未正常生成 EOS，跳过该候选，避免采用截断回答
    #    d. decode(trimmed_ids, skip_special_tokens=True)
    #    e. 若文本去除首尾空白后为空，跳过
    #    f. answer_correctness(text, expected_answer) 得到分数
    #    g. 将文本、分数分别加入两个列表
    # 4. 调用 select_preference_pair 并返回
    if num_candidates < 2:
        raise ValueError("num_candidates必须大于等于2")
    response_texts, scores = [], []
    for _ in range(num_candidates):
        _, response_ids = generate_response(model, tokenizer, question, max_new_tokens)
        trimmed_ids, is_terminated = trim_response(response_ids, tokenizer.eos_token_id)
        if not is_terminated:
            continue
        text = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
        if not text.strip():
            continue
        score = answer_correctness(text, expected_answer)
        response_texts.append(text)
        scores.append(score)
    return select_preference_pair(response_texts, scores)

def build_preference_records(
    model, tokenizer, records,
    num_candidates=4, max_new_tokens=256,
):
    """
    records：包含 question、answer 的题目字典列表。
    返回偏好记录列表，每条包含：
        question、chosen、rejected、answer
    """
    preference_records = []

    # 1. 遍历 records
    # 2. 调用 generate_preference_pair，传入：
    #    model、tokenizer、
    #    record["question"]、record["answer"]、
    #    num_candidates、max_new_tokens
    # 3. 若返回 None，跳过这道题
    # 4. 否则解包 chosen、rejected
    # 5. 将 question、chosen、rejected、answer
    #    组成字典，加入 preference_records
    # 6. 返回 preference_records
    for record in records:
        pair = generate_preference_pair(
            model=model,
            tokenizer=tokenizer,
            question=record["question"],
            expected_answer=record["answer"],
            num_candidates=num_candidates,
            max_new_tokens=max_new_tokens,
        )
        if pair is None:
            continue
        chosen, rejected = pair
        preference_record = {
            "question":record["question"],
            "chosen":chosen,
            "rejected":rejected,
            "answer":record["answer"],
        }
        preference_records.append(preference_record)
    return preference_records

def save_preference_records(records, path):
    """每行保存一条偏好记录，使用 UTF-8 编码。"""
    if not records:
        raise ValueError("偏好记录不能为空")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # w 表示写入并覆盖已有文件；with 会在结束时自动关闭文件。
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            # 将 Python 字典转换成 JSON 字符串，每条记录占一行。
            line = json.dumps(record, ensure_ascii=False)
            file.write(line + "\n")
