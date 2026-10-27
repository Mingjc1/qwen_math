"""检查一次 PPO 更新；使用人工目标，不执行真实 rollout 或保存权重。"""

import argparse
import math

import torch

from qwen_math.common.seed import set_seed
from qwen_math.data.collators import collate_sft
from qwen_math.data.templates import encode_sft_example
from qwen_math.models.loader import add_lora, load_base_model, load_tokenizer
from qwen_math.models.value_head import ValueHead, forward_policy_value
from qwen_math.trainers.ppo import build_ppo_optimizer, ppo_step


def build_inputs(args):
    if args.tiny:
        # 随机初始化的小模型：在 CPU 上检查接口，无须下载任何文件。
        from transformers import Qwen2Config, Qwen2ForCausalLM

        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_dropout=0.0,
            pad_token_id=0,
            eos_token_id=2,
        )).to(args.device)
        sample = {
            "input_ids": [3, 4, 5, 6, 2],
            "attention_mask": [1, 1, 1, 1, 1],
            "response_mask": [False, False, True, True, True],
        }
        batch = collate_sft([sample], pad_token_id=0)
    else:
        tokenizer = load_tokenizer(model_id=args.model_id)
        sample = encode_sft_example(
            "What is 2+3?", "2+3=5.\nFinal answer: 5", tokenizer
        )
        batch = collate_sft([sample], tokenizer.pad_token_id)
        dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
        model = load_base_model(
            model_id=args.model_id, device=args.device, dtype=dtype
        )

    model = add_lora(model)
    device = next(model.parameters()).device
    value_head = ValueHead(model.config.hidden_size).to(device)
    batch = {key: value.to(device) for key, value in batch.items()}
    return model, value_head, batch


def snapshot(module):
    return {
        name: param.detach().cpu().clone()
        for name, param in module.named_parameters()
        if param.requires_grad
    }


def max_change(module, before):
    changes = []
    for name, param in module.named_parameters():
        if name in before:
            after = param.detach().cpu().float()
            assert torch.isfinite(after).all(), f"参数出现非有限值：{name}"
            changes.append((after - before[name].float()).abs().max().item())
    assert changes, "没有找到可训练参数"
    return max(changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tiny", action="store_true", help="使用随机小模型在 CPU 上检查")
    args = parser.parse_args()
    if args.device is None:
        args.device = "cpu" if args.tiny else "cuda"
    set_seed(42)
    model, value_head, batch = build_inputs(args)

    model.eval()
    value_head.eval()
    with torch.no_grad():
        old_logps, old_values, mask = forward_policy_value(model, value_head, batch)

    expected_shape = (batch["input_ids"].shape[0], batch["input_ids"].shape[1] - 1)
    assert old_logps.shape == old_values.shape == mask.shape == expected_shape
    assert mask.any(), "样本没有有效回答 token"
    assert torch.isfinite(old_logps).all() and torch.isfinite(old_values).all()
    batch["old_logps"] = old_logps.detach().clone()
    batch["advantages"] = mask.float()
    batch["returns"] = old_values.detach() + mask.float()
    fixed_targets = {key: batch[key].clone() for key in ("old_logps", "advantages", "returns")}

    # 使用稍大的测试学习率，方便观察一次更新；不作为正式 PPO 超参数。
    optimizer = build_ppo_optimizer(model, value_head, learning_rate=1e-4)
    policy_before = snapshot(model)
    value_before = snapshot(value_head)
    losses = ppo_step(model, value_head, batch, optimizer)
    assert all(math.isfinite(loss) for loss in losses), "损失出现非有限值"
    for key, target in fixed_targets.items():
        assert not batch[key].requires_grad, f"{key} 不应参与求导"
        assert torch.equal(batch[key], target), f"更新过程中修改了 {key}"

    policy_change = max_change(model, policy_before)
    value_change = max_change(value_head, value_before)
    assert policy_change > 0, "策略参数未更新"
    assert value_change > 0, "价值头参数未更新"
    print("对齐后的形状:", expected_shape)
    for name, loss in zip(("总损失", "策略损失", "价值损失"), losses):
        print(f"{name}: {loss:.6f}")
    print("LoRA 最大参数变化:", policy_change)
    print("价值头最大参数变化:", value_change)
    print("PPO 单步检查通过；未保存权重。")


if __name__ == "__main__":
    main()
