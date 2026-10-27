"""在线采样（待实现）。

按生成微批串行采样，保持题目 id/group id；记录有效回答 mask、EOS/截断。
第一版原始策略采样：temperature=1、top_p=1、top_k=0。"""
from qwen_math.data.templates import format_prompt
import torch
from qwen_math.common.tensor_ops import response_logps
from qwen_math.data.collators import collate_rollouts

def encode_rollout_prompt(question, tokenizer, device):
    """
    返回字典：
        input_ids：[1, P]
        attention_mask：[1, P]
    P 为 prompt 的 token 数。
    """
    prompt = format_prompt(question)
    result = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, padding=False, truncation=False).to(device)
    return {"input_ids":result["input_ids"], "attention_mask":result["attention_mask"] }

@torch.no_grad()
def generate_response(
    model, tokenizer, question, max_new_tokens=256,
):
    """返回 prompt_ids、response_ids，均为一维张量。"""
    if max_new_tokens <= 0:
        raise ValueError
    model.eval()
    device = next(model.parameters()).device
    inputs = encode_rollout_prompt(question, tokenizer, device)
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_length = inputs["input_ids"].shape[1]
    prompt_ids = inputs["input_ids"][0]
    response_ids = generated[0, prompt_length:]
    return prompt_ids, response_ids

def trim_response(response_ids, eos_token_id):
    """
    response_ids：一维张量。
    返回：
        trimmed_ids：保留首个 EOS（含）之前的回答
        terminated：是否出现 EOS，Python bool
    """
    if response_ids.numel() == 0 or eos_token_id is None:
        raise ValueError
    positions = (response_ids == eos_token_id).nonzero(as_tuple=True)[0]
    if positions.numel() > 0:
        first_eos = positions[0].item()
        return response_ids[:first_eos + 1], True
    else:
        return response_ids, False

def build_rollout_sample(prompt_ids, response_ids, eos_token_id):
    """
    输入：两个一维 token 张量。
    返回字典，张量均为 [T]：
        input_ids、attention_mask、response_mask、terminated
    """
    trimmed_ids ,is_terminal = trim_response(response_ids,eos_token_id)
    input_ids = torch.cat([prompt_ids, trimmed_ids])
    attention_mask = torch.ones_like(input_ids)
    response_mask = torch.cat([torch.zeros_like(prompt_ids, dtype=torch.bool),
    torch.ones_like(trimmed_ids, dtype=torch.bool),])
    terminated = torch.zeros_like(input_ids, dtype=torch.bool)
    if is_terminal:
        terminated[-1] = True
    return {
        "input_ids":input_ids,
        "attention_mask":attention_mask,
        "response_mask":response_mask,
        "terminated":terminated,
    }

@torch.no_grad()
def collect_logps(model, batch):
    """
    返回 token_logps、mask，均为 [B, T-1]。
    用于记录采样策略或固定参考模型的概率。
    """
    # 1. model.eval()
    # 2. 模型前向，传入：
    #    input_ids=batch["input_ids"]
    #    attention_mask=batch["attention_mask"]
    #    use_cache=False
    # 3. 调用 response_logps：
    #    outputs.logits、input_ids、response_mask
    # 4. 返回 token_logps、mask
    model.eval()
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    return response_logps(outputs.logits, batch["input_ids"], batch["response_mask"])

def sample_rollout_batch(
    model, tokenizer, questions, max_new_tokens=256,
):
    """
    questions：非空题目列表。
    返回：
        batch：四个 [B, T] 张量，位于模型设备
        response_texts：长度为 B 的回答文本列表
    """
    # 1. questions 为空时抛出 ValueError
    # 2. 创建 samples、response_texts 两个列表
    # 3. 遍历 questions：
    #    a. 调用 generate_response
    #    b. 调用 build_rollout_sample，加入 samples
    #    c. 取 sample["input_ids"][sample["response_mask"]]
    #    d. tokenizer.decode(..., skip_special_tokens=True)
    #       将回答文本加入 response_texts
    # 4. 调用 collate_rollouts
    # 5. 将 batch 中所有张量移动到模型设备
    # 6. 返回 batch, response_texts
    if not questions:
        raise ValueError("questions为空")
    samples, response_texts = [], []
    for q in questions:
        prompt_ids, response_ids =generate_response(model, tokenizer, q, max_new_tokens)
        sample = build_rollout_sample(prompt_ids, response_ids, tokenizer.eos_token_id)
        samples.append(sample)
        response_text = tokenizer.decode(sample["input_ids"][sample["response_mask"]], skip_special_tokens=True)
        response_texts.append(response_text)
    batch = collate_rollouts(samples, tokenizer.pad_token_id)
    device = next(model.parameters()).device
    batch = {
        key: tensor.to(device)
        for key, tensor in batch.items()
    }
    return batch, response_texts
