"""模型与 LoRA 加载（待实现）。

固定 Qwen2.5-Math-1.5B；先创建/恢复 LoRA，再创建 optimizer。
后训练恢复同一 SFT adapter，参考策略为独立冻结的 SFT 快照。"""

from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM
from qwen_math.models.value_head import ValueHead

def load_tokenizer(
    model_id="Qwen/Qwen2.5-Math-1.5B",
    revision=None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer 未设置 EOS")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    return tokenizer
    


def load_base_model(
    model_id="Qwen/Qwen2.5-Math-1.5B",
    revision=None,
    device="cuda",
    dtype=torch.bfloat16,
):
    """加载基座模型，暂不添加 LoRA。"""
    # 1. 调用 AutoModelForCausalLM.from_pretrained
    #    传入 model_id、revision、dtype
    model = AutoModelForCausalLM.from_pretrained(model_id,revision=revision,dtype=dtype)
    # 2. 使用 model.to(device) 放到指定设备
    model.to(device)
    # 3. 设置 model.config.use_cache = False
    model.config.use_cache = False
    # 4. 返回 model
    return model

from peft import LoraConfig, TaskType, get_peft_model


def add_lora(model, r=8, alpha=16):
    """冻结基座，添加可训练的 LoRA 适配器。"""
    # 1. 创建 LoraConfig：
    #    task_type=TaskType.CAUSAL_LM
    #    r=r
    #    lora_alpha=alpha
    #    lora_dropout=0.0
    #    target_modules=["q_proj", "v_proj"]
    #    bias="none"
    config = LoraConfig(task_type=TaskType.CAUSAL_LM,r=r,lora_alpha=alpha,lora_dropout=0.0,target_modules=["q_proj", "v_proj"],bias="none")
    # 2. 用 get_peft_model(model, config) 包装模型
    lora_model = get_peft_model(model,config)
    # 3. 调用包装后模型的 print_trainable_parameters()
    lora_model.print_trainable_parameters()
    # 4. 返回包装后的模型
    return lora_model

from peft import PeftModel


def load_adapter(base_model, adapter_path, is_trainable=False):
    """在基座上恢复已保存的 LoRA 适配器。"""
    return PeftModel.from_pretrained(base_model,adapter_path,is_trainable=is_trainable)

def load_reference_model(
    adapter_path,
    model_id="Qwen/Qwen2.5-Math-1.5B",
    device="cuda",
    dtype=torch.bfloat16,
):
    """加载独立的 SFT 参考模型，冻结所有参数。"""
    base_model = load_base_model(model_id=model_id, device=device, dtype=dtype)
    model = load_adapter(base_model, adapter_path, is_trainable=False)
    model.requires_grad_(False)
    model.eval()
    return model
    
def load_ppo_policy(
    adapter_path,
    model_id="Qwen/Qwen2.5-Math-1.5B",
    device="cuda",
    dtype=torch.bfloat16,
):
    """返回可训练的 SFT 策略和独立价值头。"""
    # 1. load_base_model，传入 model_id、device、dtype
    # 2. load_adapter，is_trainable=True
    # 3. model.eval()，PPO 中关闭 dropout，但保留梯度
    # 4. 创建 ValueHead(model.config.hidden_size)
    # 5. 将价值头放到模型设备，dtype=torch.float32
    # 6. 返回 model, value_head
    base_model = load_base_model(model_id=model_id, device=device, dtype=dtype)
    model =load_adapter(base_model, adapter_path, is_trainable=True)
    model.eval()
    value_head = ValueHead(model.config.hidden_size).to(device=next(model.parameters()).device, dtype=torch.float32)
    return model, value_head
