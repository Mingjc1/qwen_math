"""数学规则奖励（待实现）。

提取 Final answer 后的答案，进行数值/符号归一化与等价校验。
返回正确性 0/1 并记录解析失败；标准答案不能进入模型输入。"""
import re
from decimal import Decimal, InvalidOperation
import torch

def extract_final_answer(text):
    parts = text.rsplit("Final answer:", 1)
    if len(parts) != 2:
        return None

    lines = parts[1].lstrip(" \t").splitlines()
    if not lines:
        return None

    answer = lines[0].strip()
    return answer if answer else None

def normalize_numeric_answer(answer):
    if answer is None:
        return None

    answer = answer.strip()

    pattern = (
        r"[+-]?"
        r"(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)"
        r"(?:\.[0-9]+)?"
    )
    if re.fullmatch(pattern, answer) is None:
        return None

    try:
        return Decimal(answer.replace(",", ""))
    except InvalidOperation:
        return None

def answer_correctness(response_text, expected_answer):
    """
    response_text：模型生成的回答文本。
    expected_answer：数据中的标准答案字符串。
    返回：正确为 1.0，否则为 0.0。
    """
    expected = normalize_numeric_answer(expected_answer)
    if expected is None:
        raise ValueError("标准答案不是合法数值")

    extracted = extract_final_answer(response_text)
    predicted = normalize_numeric_answer(extracted)
    if predicted is None:
        return 0.0

    return 1.0 if predicted == expected else 0.0


def score_responses(response_texts, expected_answers, device):
    """
    两个等长列表，长度为 B。
    返回 scores：[B]，dtype=torch.float32。
    """
    # 1. 检查两个列表长度相同，否则抛出 ValueError
    # 2. 检查列表不为空，否则抛出 ValueError
    # 3. 用 zip 配对回答与标准答案
    # 4. 每对调用 answer_correctness，收集得分
    # 5. 转成 torch.tensor：
    #    dtype=torch.float32，device=device
    # 6. 返回 scores
    if len(response_texts) != len(expected_answers):
        raise ValueError("列表长度不相等")
    if not response_texts or not expected_answers:
        raise ValueError("列表为空")
    scores = torch.zeros((len(response_texts)),dtype=torch.float32,device=device)
    for i, (r, e) in enumerate(zip(response_texts, expected_answers)):
        scores[i] = answer_correctness(r,e)
    return scores
        
    
