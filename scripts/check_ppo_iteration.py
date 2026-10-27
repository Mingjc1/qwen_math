"""检查真实采样、规则奖励、优势计算和 PPO 更新；不保存更新后的模型。"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from qwen_math.common.seed import set_seed
from qwen_math.models.loader import (
    add_lora,
    load_ppo_policy,
    load_reference_model,
    load_tokenizer,
)
from qwen_math.trainers.ppo import build_ppo_optimizer, train_ppo_iteration


def save_tiny_fixture(directory):
    """创建随机小基座和初始 LoRA，仅用于离线检查，不代表已完成 SFT。"""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    base_dir = directory / "base"
    adapter_dir = directory / "adapter"
    words = [
        "[PAD]", "[UNK]", "[EOS]", "Question", ":", "Solution",
        "What", "is", "2", "+", "3", "?", "5", "Final", "answer", ".",
    ]
    backend = Tokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )
    tokenizer.save_pretrained(base_dir)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(words), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.0,
        pad_token_id=0, eos_token_id=2,
    ))
    model.save_pretrained(base_dir)
    add_lora(model).save_pretrained(adapter_dir)
    return str(base_dir), str(adapter_dir)


def trainable_snapshot(module):
    return {
        name: param.detach().cpu().clone()
        for name, param in module.named_parameters() if param.requires_grad
    }


def max_change(module, before):
    assert before, "没有可训练参数"
    largest = 0.0
    for name, param in module.named_parameters():
        if name in before:
            after = param.detach().cpu().float()
            assert torch.isfinite(after).all(), f"参数出现非有限值：{name}"
            largest = max(largest, (after - before[name].float()).abs().max().item())
    return largest


def parameter_digest(module):
    """逐块校验所有参数，避免复制整个参考模型占用大量内存或显存。"""
    digest = hashlib.sha256()
    for name, param in module.named_parameters():
        digest.update(name.encode())
        for chunk in param.detach().reshape(-1).split(1024 * 1024):
            raw = chunk.to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
            digest.update(raw)
    return digest.hexdigest()


def run_check(args, model_id, adapter_path):
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    tokenizer = load_tokenizer(model_id=model_id)
    model, value_head = load_ppo_policy(
        adapter_path, model_id=model_id, device=args.device, dtype=dtype
    )
    reference = load_reference_model(
        adapter_path, model_id=model_id, device=args.device, dtype=dtype
    )
    assert all(not p.requires_grad for p in reference.parameters()), "参考模型未完全冻结"
    policy_storage = {p.data_ptr() for p in model.parameters()}
    assert all(p.data_ptr() not in policy_storage for p in reference.parameters()), "策略与参考模型共享参数"
    assert model.peft_config["default"].inference_mode is False, "策略适配器不可训练"

    policy_before = trainable_snapshot(model)
    value_before = trainable_snapshot(value_head)
    print("校验参考模型全部参数……", flush=True)
    reference_before = parameter_digest(reference)
    # 较大的测试学习率便于观察变化，不作为正式训练配置。
    optimizer = build_ppo_optimizer(model, value_head, learning_rate=1e-4)
    print("开始采样、评分和 PPO 更新……", flush=True)
    metrics = train_ppo_iteration(
        model=model, reference_model=reference, value_head=value_head,
        tokenizer=tokenizer, questions=["What is 2+3?"], expected_answers=["5"],
        optimizer=optimizer, max_new_tokens=args.max_new_tokens,
        update_epochs=args.update_epochs,
    )
    assert set(metrics) == {"loss", "policy_loss", "value_loss", "mean_score"}
    assert all(math.isfinite(value) for value in metrics.values()), "日志出现非有限值"
    assert metrics["mean_score"] in (0.0, 1.0), "单题正确性奖励应为 0 或 1"
    policy_change = max_change(model, policy_before)
    value_change = max_change(value_head, value_before)
    assert reference_before == parameter_digest(reference), "参考模型参数发生变化"
    assert all(p.grad is None for p in reference.parameters()), "参考模型产生了梯度"
    assert policy_change > 0, "策略参数未更新；检查优势是否退化为全零"
    assert value_change > 0, "价值头未更新；检查回报是否恰好等于旧价值"

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("LoRA 最大参数变化:", policy_change)
    print("价值头最大参数变化:", value_change)
    print("参考模型全部参数 SHA256 保持不变，且没有梯度。")
    print("PPO 完整迭代检查通过；未保存更新后的权重。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--adapter-path", default="outputs/sft/first_run_adapter")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--tiny", action="store_true", help="临时创建小模型，在 CPU 离线检查完整流程")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.update_epochs <= 0:
        parser.error("生成长度和更新轮数必须大于零")
    args.device = args.device or ("cpu" if args.tiny else "cuda")
    set_seed(42)
    if args.tiny:
        with TemporaryDirectory(prefix="qwen_ppo_check_") as directory:
            model_id, adapter_path = save_tiny_fixture(Path(directory))
            run_check(args, model_id, adapter_path)
    else:
        if not (Path(args.adapter_path) / "adapter_config.json").is_file():
            parser.error("找不到适配器，请用 --adapter-path 指向 SFT adapter 目录")
        run_check(args, args.model_id, args.adapter_path)


if __name__ == "__main__":
    main()
