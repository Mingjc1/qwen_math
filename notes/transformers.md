# Transformers：从文本到模型输出

Transformers 提供模型结构、预训练权重、分词器与生成接口。本项目使用它取得 logits，自行用 torch 编写算法损失。

## 1. 必学接口

| 接口 | 输入与输出 | 项目用途 |
| --- | --- | --- |
| `AutoTokenizer.from_pretrained(path)` | 模型标识或目录 -> tokenizer | 载入配套词表与模板 |
| `AutoModelForCausalLM.from_pretrained(path)` | 权重目录 -> 因果语言模型 | 加载 Qwen |
| `AutoConfig.from_pretrained(path)` | 目录 -> 模型配置 | 查看 hidden_size 等信息 |
| `tokenizer(text, ...)` | 文本 -> 编号与 mask | 训练输入 |
| `tokenizer.apply_chat_template(...)` | 消息列表 -> 格式化文本或 token | 构造对话输入 |
| `model(...)` | token 与 mask -> 模型输出对象 | 获得 logits、loss、隐藏状态 |
| `model.generate(...)` | prompt -> prompt 加生成序列 | PPO/GRPO rollout |
| `tokenizer.decode / batch_decode` | token -> 文本 | 展示回答、计算奖励 |
| `model.save_pretrained(path)` | 模型 -> 本地文件 | 保存完整模型或 PEFT 适配器 |
| `tokenizer.save_pretrained(path)` | tokenizer -> 本地文件 | 保存分词配置 |

Auto 类根据配置选择具体模型实现。`CausalLM` 表示根据前文预测下一个 token。

## 2. 模型加载片段：需要模型权重

下例会在未缓存时访问 Hugging Face 并下载模型；现在可以只阅读。
使用 Qwen3-0.6B 演示聊天模板。主实验已固定为 Qwen2.5-Math-1.5B 基座，使用显式文本模板；本篇 Qwen3 示例仅用于学习聊天 API。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-0.6B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
model = model.to(device)
```

这里用较新版本的 `dtype=` 写法，旧版本示例可能使用 `torch_dtype=`；应以实际版本为准。
入门自定义训练使用明确的单设备放置。`device_map="auto"` 常用于推理分派，不是多卡训练方案。

## 3. 分词与聊天模板

`tokenizer(..., return_tensors="pt")` 返回可传给 PyTorch 的张量。
`padding=True` 补齐到该批最长长度；`truncation=True, max_length=...` 限制输入长度。

依赖上一节 tokenizer 的片段：

```python
messages = [{"role": "user", "content": "小明有 3 个苹果，又买了 2 个，一共有几个？"}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # Qwen3 模板参数，不是所有模型都支持
)
tokenizer.padding_side = "left"  # decoder-only 批量生成通常左填充
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
batch = tokenizer(
    [prompt], add_special_tokens=False, padding=True, return_tensors="pt"
).to(device)
print(batch["input_ids"].shape)
print(batch["attention_mask"].shape)
```

模板已经包含控制 token，再分词时使用 `add_special_tokens=False` 避免重复添加。
`add_generation_prompt=True` 表示加入“轮到 assistant 回答”的前缀；完整 SFT 对话一般不再追加这个生成前缀。
Base 模型未必有聊天模板，不能无条件套用这里的写法。

## 4. forward：训练的入口

依赖 model 和 batch 的片段：

```python
outputs = model(**batch, output_hidden_states=True, use_cache=False)
logits = outputs.logits                  # [B,T,V]
last_hidden = outputs.hidden_states[-1]  # [B,T,H]
```

`model(**batch)` 将字典按关键字展开，等价于传入 input_ids 和 attention_mask。
调用 model(...) 会经过 Module 的钩子，应优先于直接调用 model.forward(...)。
PPO 的价值头可以在隐藏状态上预测价值；这里还没有实现完整 critic。
`output_hidden_states=True` 返回多层状态，会增加内存；普通概率计算不需要打开它。
训练通常关闭 KV cache；生成时可使用 cache 加速。

SFT 片段：假设已经构造与 input_ids 对齐的 response_mask。

```python
labels = batch["input_ids"].clone()
labels[~response_mask.bool()] = -100
labels[batch["attention_mask"] == 0] = -100
outputs = model(**batch, labels=labels, use_cache=False)
loss = outputs.loss
loss.backward()
```

这里的 labels 与 input_ids 同形状；标准因果语言模型内部负责错位计算。不要提前再移动 labels。
`-100` 是损失忽略标记，不是合法 token 编号，不能传给 input_ids。
所有 labels 都是 -100 的样本应在数据阶段移除。
DPO/PPO/GRPO 通常自己从 logits 提取概率，不能直接用 outputs.loss 代替其目标函数。

## 5. generate：回答采样

依赖前文 model、batch 和 tokenizer 的片段：

```python
model.eval()
with torch.no_grad():
    sequences = model.generate(
        **batch,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
answer_ids = sequences[:, batch["input_ids"].shape[1]:]
answers = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)
print(answers)
```

| 参数 | 含义 |
| --- | --- |
| `max_new_tokens` | 最多生成多少个新 token，不含 prompt |
| `do_sample` | 是否随机采样；False 配合常规配置可用于贪心生成 |
| `temperature` | 采样分布温度 |
| `top_p / top_k` | 截断候选分布 |
| `num_return_sequences` | 每个输入返回多少条序列，需匹配生成策略 |
| `return_dict_in_generate` | 返回包含 sequences 等字段的对象 |
| `output_scores` | 保留每一步分数，长序列可能占用大量内存 |

批量左填充时，应按输入张量的统一宽度截掉 prompt，不能按每条有效 token 数直接切生成输出。
generate 的 scores 可能经过温度和采样处理，不要直接当作训练需要的原始策略 logps。
通常对生成序列再次 forward，取得指定策略定义下的 logps。
PPO/GRPO 中采样分布与概率比的定义必须一致，temperature/top_p/top_k 不能未经处理就混用；初版可从 temperature=1、top_p=1、top_k=0 开始。
本例的 0.8/0.95 仅演示推理参数。

## 6. 调度器与种子：稍后使用

```python
# 片段：optimizer 已创建，total_updates 为真实参数更新次数
from transformers import get_scheduler, set_seed

set_seed(42)
scheduler = get_scheduler(
    "linear", optimizer=optimizer,
    num_warmup_steps=10, num_training_steps=total_updates,
)
# 每次真实 optimizer.step() 之后，再 scheduler.step()
```

多次微批次梯度累积才更新一次参数时，调度器也只走一步。
`Trainer` 和 `TrainingArguments` 可在以后学习，第一版算法实现优先显式训练循环。

## 学习检查与官方资料

能否解释 forward 与 generate 的不同？能否区分 logits、token 编号和解码文本？

- [模型接口](https://huggingface.co/docs/transformers/main_classes/model)
- [聊天模板](https://huggingface.co/docs/transformers/chat_templating)
- [生成参数](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [Qwen3-0.6B 模型说明](https://huggingface.co/Qwen/Qwen3-0.6B)
