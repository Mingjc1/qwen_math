"""基座统一文本模板（待实现）。

使用 Question: + question + Solution:；不依赖聊天模板或 thinking 模式。
跟踪首个回答 token 边界，完整回答包含真实 EOS。"""

def format_prompt(question: str) -> str:
    """
    输入：数学题文本。
    输出：统一的问题前缀，不包含答案。
    """
    # 1. 去除 question 首尾空白
    question = question.strip()
    # 2. 若题目为空，抛出 ValueError
    if question=='':
        raise ValueError
    # 3. 按下面的格式返回字符串
    return f'Question:\n{question}\n\nSolution:\n'

def encode_sft_example(question, solution, tokenizer):
    """
    返回字典：
        input_ids: list[int]
        attention_mask: list[int]
        response_mask: list[bool]
    """
    # 1. 调用 format_prompt(question)
    question = format_prompt(question)
    # 2. 检查 solution 去除首尾空白后不为空
    solution =solution.strip()
    if not solution:
        raise ValueError
    # 3. 分别对 prompt、solution 分词
    q_ids = tokenizer.encode(question,add_special_tokens=False)
    s_ids = tokenizer.encode(solution,add_special_tokens=False)
    # 4. 回答 token 末尾追加 eos_token_id
    if tokenizer.eos_token_id==None:
        raise ValueError("tokenizer 未设置 EOS")
    s_ids.append(tokenizer.eos_token_id)
    # 5. 拼接并构造两个 mask
    ids =q_ids+s_ids
    attention_mask = [1]*len(ids)
    response_mask = [False]*len(q_ids)+[True]*len(s_ids)
    return {
    "input_ids": ids,
    "attention_mask": attention_mask,
    "response_mask": response_mask,
    }

def filter_long_samples(samples, max_length=512):
    """过滤超长样本，返回保留的样本列表。"""
    # 1. 保留 input_ids 长度不超过 max_length 的样本
    # 2. 如果全部被过滤，抛出 ValueError
    # 3. 返回保留的列表
    filtered_samples = [sample for sample in samples if len(sample['input_ids'])<=max_length]
    if filtered_samples==[]:
        raise ValueError
    return filtered_samples
    
