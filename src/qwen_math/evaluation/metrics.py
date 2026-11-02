"""统一评测指标（待实现）。

准确率、可解析率、截断率、回答长度、生成预算；不以奖励增长代替独立评测。"""
from qwen_math.rewards.math_reward import (
    extract_final_answer,
    normalize_numeric_answer,
    answer_correctness,
)

def evaluate_answer(response_text, expected_answer):
    """
    返回字典：
        predicted_answer：提取的答案字符串，或 None
        format_valid：答案能否解析为支持的数值，bool
        correct：正确为 1.0，否则为 0.0
    """
    # 1. extract_final_answer，得到 predicted_answer
    # 2. normalize_numeric_answer：
    #    返回值不是 None，则 format_valid=True
    # 3. answer_correctness，得到 correct
    # 4. 返回包含上述三个字段的字典
    predicted_answer = extract_final_answer(response_text)

    normalized = normalize_numeric_answer(predicted_answer)
    format_valid = normalized is not None

    correct = answer_correctness(response_text, expected_answer)

    return {
        "predicted_answer": predicted_answer,
        "format_valid": format_valid,
        "correct": correct,
    }

def compute_answer_metrics(response_texts, expected_answers):
    """
    输入：两个等长、非空列表。
    返回：
        count：回答数量
        accuracy：正确率，范围 [0, 1]
        format_valid_rate：格式有效率，范围 [0, 1]
    """
    # 1. 检查两个列表等长且非空，否则抛出 ValueError
    # 2. 初始化 correct_count、valid_count 为 0
    # 3. 用 zip 配对回答和标准答案
    # 4. 每对调用 evaluate_answer
    # 5. 累加 result["correct"]、result["format_valid"]
    # 6. 两个累计值分别除以总数量，返回字典
    if not(len(response_texts) == len(expected_answers) and response_texts):
        raise ValueError("两个列表必须等长且非空")
    correct_count, valid_count, count = 0, 0, len(response_texts)
    for (response_text, expected_answer) in zip(response_texts, expected_answers):
        result = evaluate_answer(response_text, expected_answer)
        correct_count += result["correct"]
        valid_count += result["format_valid"]
    return {
        "count":count,
        "accuracy":correct_count / count,
        "format_valid_rate":valid_count / count,
    }

def compute_truncation_rate(terminated_flags):
    """
    terminated_flags：非空 Python bool 列表。
        True：生成时遇到 EOS，正常结束
        False：未遇到 EOS，因长度上限结束
    返回截断率，范围 [0, 1]。
    """
    # 1. 列表为空时抛出 ValueError
    # 2. 检查每个元素都是 bool，否则抛出 ValueError
    # 3. 统计 False 的数量
    # 4. 除以总数量并返回
    if not terminated_flags:
        raise ValueError("terminated_flags为空")
    truncated_count = 0
    for flag in terminated_flags:
        if type(flag) != bool:
            raise ValueError("每个元素都必须是 bool")
        if not flag:
            truncated_count += 1
    return truncated_count / len(terminated_flags)
    