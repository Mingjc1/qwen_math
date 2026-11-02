from qwen_math.evaluation.metrics import compute_answer_metrics
from qwen_math.evaluation.metrics import compute_truncation_rate
responses = [
    "Final answer: 5",
    "Final answer: 6",
    "不知道",
]
expected = ["5", "5", "5"]

metrics = compute_answer_metrics(responses, expected)

assert metrics["count"] == 3
assert metrics["accuracy"] == 1 / 3
assert metrics["format_valid_rate"] == 2 / 3

# 确认答案为 0 时仍被视为有效格式。
zero = compute_answer_metrics(["Final answer: 0"], ["0"])
assert zero["accuracy"] == 1.0
assert zero["format_valid_rate"] == 1.0

print(metrics)
print("评估指标检查通过")

assert compute_truncation_rate([True, False, True, False]) == 0.5
assert compute_truncation_rate([True, True]) == 0.0
assert compute_truncation_rate([False, False]) == 1.0

# 数字 1 不能替代接口要求的 Python bool。
try:
    compute_truncation_rate([True, 1])
except ValueError:
    pass
else:
    raise AssertionError("非 bool 输入未被拒绝")

print("截断率检查通过")