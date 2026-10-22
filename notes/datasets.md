# Datasets 与 DataLoader：准备一批训练输入

Hugging Face `datasets` 管理、转换数据集；PyTorch `DataLoader` 把样本组成批次。
它们是两个库，不要把 `datasets.Dataset` 和 `torch.utils.data.Dataset` 当作同一个类。

## 1. 常用函数

| 接口 | 输入 -> 输出 | 用途 |
| --- | --- | --- |
| `Dataset.from_list(rows)` | 字典列表 -> Dataset | 本地小数据练习 |
| `load_dataset("json", data_files=...)` | JSON/JSONL -> DatasetDict | 读取数学题 |
| `dataset.train_test_split(...)` | Dataset -> 含 train/test 的字典 | 划分训练与验证数据 |
| `dataset.map(fn, batched=...)` | 样本 -> 新样本 | 格式转换、分词 |
| `dataset.filter(fn)` | 谓词 -> 筛选后数据集 | 去除空题、无答案记录 |
| `dataset.select(indices)` | 索引 -> 子集 | 小规模调试 |
| `dataset.shuffle(seed=...)` | 数据集 -> 打乱结果 | 可复现顺序 |
| `dataset.with_format("torch")` | 返回按 torch 格式取值的数据集 | 数字列转张量 |
| `DataLoader(..., collate_fn=...)` | 数据集 -> 批次迭代器 | 动态补齐 |

map/filter/shuffle 通常返回新对象，要接收返回值。
`batched=True` 时函数收到“列名 -> 值列表”，不是单条样本。

可直接运行，需安装 datasets：

```python
from datasets import Dataset

dataset = Dataset.from_list([
    {"id": str(i), "question": f"{i}+1=?", "answer": str(i+1)}
    for i in range(10)
])
dataset = dataset.filter(lambda row: bool(row["question"].strip()))
parts = dataset.train_test_split(test_size=0.2, seed=42)
train = parts["train"].map(lambda row: {"prompt": row["question"]})
print(train[0])
print(train.column_names)
```

项目中先按题目 id 划分，再生成 chosen/rejected 或 rollout，避免同题不同回答跨集合。
这里的 test 只是划分键名；正式项目应单独命名验证集与最终测试集。

## 2. 三类训练格式

```python
# 格式示意，不是完整数据集
sft = {"id": "1", "prompt": "2+3=?", "response": "2+3=5。答案：5", "answer": "5"}
dpo = {"id": "1", "prompt": "2+3=?", "chosen": "答案：5", "rejected": "答案：6"}
rl = {"id": "1", "prompt": "2+3=?", "answer": "5"}
```

RL 的 answer 供奖励器使用，不能拼进模型的问题输入。
这些是本项目建议的中间格式，不保证符合任意版本 TRL 的原生 schema。

## 3. 动态补齐与回答 mask

下面用已经分好词的玩具样本演示 collate_fn，可直接运行，只需要 torch。
约定：每条 input_ids 是“问题 token + 回答 token”，response_start 是首个回答 token 的索引。
实际分词应从模板或明确的 token 边界构造 response_start；不能默认分别分词后的长度总能直接相加。

```python
import torch
from torch.utils.data import DataLoader

rows = [
    {"input_ids": [10, 11, 20, 2], "response_start": 2},
    {"input_ids": [12, 21, 22, 23, 2], "response_start": 1},
]

def collate_fn(samples):
    pad_id = 0
    width = max(len(row["input_ids"]) for row in samples)
    ids, attention, response = [], [], []
    for row in samples:
        seq = row["input_ids"]
        length = len(seq)
        start = row["response_start"]
        if not 0 < start < length:
            raise ValueError("需要非空 prompt 与回答")
        padding = width - length
        ids.append(seq + [pad_id] * padding)
        attention.append([1] * length + [0] * padding)
        response.append([0] * start + [1] * (length-start) + [0] * padding)
    input_ids = torch.tensor(ids, dtype=torch.long)
    attention_mask = torch.tensor(attention, dtype=torch.long)
    response_mask = torch.tensor(response, dtype=torch.bool)
    labels = input_ids.clone()
    labels[~response_mask] = -100
    return dict(input_ids=input_ids, attention_mask=attention_mask,
                response_mask=response_mask, labels=labels)

loader = DataLoader(rows, batch_size=2, shuffle=False, collate_fn=collate_fn)
batch = next(iter(loader))
for name, value in batch.items():
    print(name, value.shape, value)
```

这里为训练演示使用右填充；批量生成通常使用左填充，mask 与边界也必须一起调整。
将 batch 传给模型时，显式选 input_ids、attention_mask、labels；response_mask 不是模型原生参数。

## 4. 最容易出现的问题

- 默认 collate 无法直接堆叠长度不同的 token 列表，需要补齐。
- `pad_token_id == eos_token_id` 时，不能用 token 编号相等判断全部 padding；真实 EOS 应保留。
- response_mask 应包含真实终止 token，排除后续 padding；截断回答需要单独记录。
- `DataCollatorForLanguageModeling(mlm=False)` 并不会自动理解“只训练回答”的项目约定；需核对其 labels 行为。
- map 中若保留旧列，可能误把字符串列交给默认 DataLoader；显式设置移除列或自定义 collator。

## 练习与资料

把第一条样本改成长于第二条，确认三个 mask 仍正确。指出首个回答 token 由哪个位置预测。

- [Datasets 数据处理](https://huggingface.co/docs/datasets/process)
- [PyTorch 数据接口](https://docs.pytorch.org/docs/stable/data.html)
