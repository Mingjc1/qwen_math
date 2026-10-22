"""SFT 文本数据读写接口。"""

import json
from pathlib import Path


def load_sft_records(path):
    """读取 UTF-8 JSONL，校验必需文本字段，返回样本字典列表。

    跳过空白行；保留其他字段。相对路径以当前工作目录为准。
    非法记录的错误信息包含文件路径和原始行号。
    """
    path = Path(path)
    records = []
    required_fields = ("question", "solution", "answer")

    with path.open("r", encoding="utf-8") as file:
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
