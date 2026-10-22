# PEFT：用 LoRA 减少可训练参数

PEFT 提供参数高效微调方法。LoRA 冻结原权重 W，仅训练低秩增量：`W' = W + (alpha/r) BA`（普通 LoRA 缩放约定）。
它减少需要训练的参数和优化器状态，但不会消除模型前向、激活和生成的显存开销。

## 1. 常用接口

| 接口 | 作用 | 当前学习优先级 |
| --- | --- | --- |
| `LoraConfig(...)` | 设置低秩适配器配置 | 必学 |
| `TaskType.CAUSAL_LM` | 指定因果语言模型任务 | 必学 |
| `get_peft_model(model, config)` | 给模型注入适配器 | 必学 |
| `model.print_trainable_parameters()` | 显示可训练参数比例 | 必学 |
| `model.save_pretrained(path)` | 保存适配器及配置 | 必学 |
| `PeftModel.from_pretrained(base, path, ...)` | 在基座上恢复适配器 | 必学 |
| `model.merge_and_unload()` | 合并适配器增量并返回普通模型 | 后续导出 |
| `model.disable_adapter()` | 临时禁用适配器的上下文管理器 | 参考策略扩展 |
| `prepare_model_for_kbit_training(model)` | 准备量化模型训练 | QLoRA 阶段 |

## 2. 添加 LoRA

片段：假设 base_model 已通过 Transformers 加载。

```python
from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
```

| 参数 | 含义 | 需要理解的地方 |
| --- | --- | --- |
| r | 低秩维度 | 越大可训练参数越多 |
| lora_alpha | 增量缩放系数 | 与 r 一起决定普通 LoRA 缩放 |
| lora_dropout | LoRA 分支 dropout | 初版 RL 调试使用 0 可减少概率重算差异 |
| target_modules | 注入层名称 | 需实际检查模型是否存在这些层 |
| bias | 是否训练相关 bias | 示例不训练 bias |
| modules_to_save | 适配器之外额外训练并保存的模块 | 自定义头需要单独设计保存策略 |

查看实际层名和梯度状态：

```python
# 片段：检查 base_model，再注入；避免重复包装
for name, module in base_model.named_modules():
    if name.endswith(("q_proj", "v_proj")):
        print(name, type(module).__name__)

# 片段：注入之后创建优化器
import torch
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=1e-4)
```

该学习示例的学习率不代表已验证的训练配置。
先注入再创建 optimizer，确保新适配器参数进入优化器。

## 3. 保存与加载

片段：前文模型与 tokenizer 已存在。

```python
model.save_pretrained("outputs/lora_adapter")
tokenizer.save_pretrained("outputs/lora_adapter")
```

通常保存的是 adapter 权重与配置，不是完整基座，也不是完整训练恢复状态。
必须记录基座的模型标识、revision 与 tokenizer 版本。

在新的模型对象上恢复：

```python
# 片段：加载权重，可能访问网络
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
restored = PeftModel.from_pretrained(
    base, "outputs/lora_adapter", is_trainable=True,
)
```

继续训练时注意 `is_trainable=True`；推理加载可使用默认冻结行为。
如需导出合并模型，可对适当精度、支持合并的模型使用 `merged = model.merge_and_unload()`，再保存 merged。
不要在训练过程中合并后继续复用旧的 optimizer。

## 4. 参考策略不能简单等同于“禁用 LoRA”

DPO/PPO/GRPO 可能需要固定的参考策略。若参考是 SFT 模型：

- 当 SFT 已合并到冻结基座，且当前阶段才新加 RL 适配器时，禁用当前适配器可能得到正确参考。
- 当 SFT 本身就是适配器，且正在继续更新它时，禁用适配器得到的是原基座，并不等于 SFT 参考。

初版优先明确保存 SFT 参考快照，并冻结其参数。后续再考虑独立适配器切换来节省显存。
仅写 `reference = model` 不会产生模型副本，两个名字指向同一对象。

## 5. LoRA 与 QLoRA

LoRA 是训练低秩增量；QLoRA 将它与量化冻结基座等技术组合。二者并非同义。
`prepare_model_for_kbit_training` 不负责下载量化权重，也不自动添加 LoRA。
本地先学普通 LoRA；等需要实际加载模型且显存不足时，再确定量化方案与依赖。

## 官方资料

- [PEFT 入门](https://huggingface.co/docs/peft/quicktour)
- [LoraConfig](https://huggingface.co/docs/peft/package_reference/lora)
- [PeftModel](https://huggingface.co/docs/peft/package_reference/peft_model)
- [量化训练](https://huggingface.co/docs/peft/developer_guides/quantization)
