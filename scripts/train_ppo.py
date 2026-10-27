"""PPO 训练入口：在线采样、参数更新、日志、保存及迭代边界续训。"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from qwen_math.common.checkpoint import (
    append_metrics, file_sha256, restore_rng_state, save_adapter, save_run_config,
)
from qwen_math.common.seed import set_seed
from qwen_math.data.io import load_sft_records
from qwen_math.data.templates import format_prompt
from qwen_math.models.loader import load_ppo_policy, load_reference_model, load_tokenizer
from qwen_math.rewards.math_reward import normalize_numeric_answer
from qwen_math.trainers.ppo import build_ppo_optimizer, train_ppo_iteration


CONFIG = {
    "seed": 42,
    "model_id": "Qwen/Qwen2.5-Math-1.5B",
    "sft_adapter_path": "outputs/sft/gsm8k_run01/best_adapter",
    "train_path": "data/processed/sft_train.jsonl",
    "output_dir": "outputs/ppo/gsm8k_run01",
    "device": "cuda",
    "dtype": "bfloat16",
    "max_prompt_tokens": 256,
    "rollout_prompts": 1,
    "max_new_tokens": 256,
    "max_iterations": 100,
    "update_epochs": 1,
    "learning_rate": 1e-6,
    "kl_coef": 0.01,
    "gamma": 1.0,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "value_coef": 0.5,
    "max_grad_norm": 1.0,
    "save_steps": 10,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="可选 JSON 配置，覆盖 CONFIG 默认值")
    parser.add_argument("--resume-from", help="PPO checkpoint 目录")
    parser.add_argument("--dry-run", action="store_true", help="仅校验配置、路径和数据，不加载模型")
    for name, default in CONFIG.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default))
    args = parser.parse_args()
    config = dict(CONFIG)
    if args.config:
        with Path(args.config).open(encoding="utf-8") as file:
            overrides = json.load(file)
        if not isinstance(overrides, dict) or set(overrides) - set(CONFIG):
            parser.error("配置必须为 JSON 对象，且只能包含本脚本 CONFIG 中的字段")
        config.update(overrides)
    for name in CONFIG:
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    return config, args


def validate_config(config):
    for name in (
        "rollout_prompts", "max_prompt_tokens", "max_new_tokens",
        "max_iterations", "update_epochs", "save_steps",
    ):
        value = config[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} 必须为正整数")
    if type(config["seed"]) is not int:
        raise ValueError("seed 必须为整数")
    for name in ("learning_rate", "kl_coef", "value_coef", "max_grad_norm", "clip_epsilon", "gamma", "gae_lambda"):
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} 必须是有限的非负数")
    if config["learning_rate"] == 0 or config["max_grad_norm"] == 0:
        raise ValueError("learning_rate 和 max_grad_norm 必须大于零")
    if not 0 < config["clip_epsilon"] < 1:
        raise ValueError("clip_epsilon 必须在 (0, 1) 内")
    if config["gamma"] > 1 or config["gae_lambda"] > 1:
        raise ValueError("gamma 和 gae_lambda 必须在 [0, 1] 内")
    if config["dtype"] not in ("float32", "bfloat16"):
        raise ValueError("dtype 只支持 float32 或 bfloat16")
    for name in ("model_id", "sft_adapter_path", "train_path", "output_dir", "device"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"{name} 必须是非空字符串")
    adapter = Path(config["sft_adapter_path"])
    if not (adapter / "adapter_config.json").is_file():
        raise ValueError(f"找不到 SFT adapter：{adapter}")
    if not any((adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError(f"SFT adapter 缺少权重：{adapter}")


def load_records(config, tokenizer=None):
    """复用 SFT JSONL，但仅把 question/answer 交给 PPO。超长问题整条过滤。"""
    source = load_sft_records(config["train_path"])
    records = []
    for record in source:
        if normalize_numeric_answer(record["answer"]) is None:
            raise ValueError(f"标准答案不是支持的数值格式：{record['answer']!r}")
        if tokenizer is not None:
            tokens = tokenizer.encode(format_prompt(record["question"]), add_special_tokens=False)
            if len(tokens) > config["max_prompt_tokens"]:
                continue
        records.append({"question": record["question"], "answer": record["answer"]})
    if len(records) < config["rollout_prompts"]:
        raise ValueError("有效题目数量少于 rollout_prompts")
    print(f"PPO 数据：原始 {len(source)} 条，保留 {len(records)} 条", flush=True)
    return records


def source_hashes(config):
    """锁定训练数据和固定 SFT 参考适配器的文件内容。"""
    adapter = Path(config["sft_adapter_path"])
    hashes = {"train_path": file_sha256(config["train_path"])}
    for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        if (adapter / name).is_file():
            hashes["sft/" + name] = file_sha256(adapter / name)
    return hashes


def load_resume_state(config, checkpoint, hashes):
    if checkpoint is None:
        return None
    checkpoint = Path(checkpoint)
    with (checkpoint / "run_config.json").open(encoding="utf-8") as file:
        saved = json.load(file)
    # 本版本使用恒定学习率，可增加总迭代数，但其余训练设置必须一致。
    old = {key: saved.get(key) for key in CONFIG if key != "max_iterations"}
    new = {key: value for key, value in config.items() if key != "max_iterations"}
    if old != new or saved.get("data_sha256") != hashes:
        raise ValueError("续训配置、训练数据或原始 SFT adapter 已改变")
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
    if state.get("format") != "qwen_math_ppo_v1":
        raise ValueError("该 checkpoint 不是本入口保存的 PPO 状态")
    if state["iteration"] >= config["max_iterations"]:
        raise ValueError("checkpoint 已达到 max_iterations；继续训练请增加总迭代数")
    return state


def save_checkpoint(model, value_head, tokenizer, optimizer, sampler, iteration, config, hashes, directory):
    """仅在完整迭代结束后保存；training_state.pt 是最后写入的状态文件。"""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"checkpoint 已存在，拒绝覆盖：{directory}")
    save_adapter(model, tokenizer, directory)
    torch.save(value_head.state_dict(), directory / "value_head.pt")
    save_run_config(config, directory, data_hashes=hashes)
    torch.save({
        "format": "qwen_math_ppo_v1",
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        "sampler_rng_state": sampler.getstate(),
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }, directory / "training_state.pt")
    print(f"已保存：{directory}", flush=True)


def run_training(config, resume_from=None, dry_run=False):
    validate_config(config)
    set_seed(config["seed"])
    hashes = source_hashes(config)
    state = load_resume_state(config, resume_from, hashes)
    if dry_run:
        load_records(config)
        print(json.dumps(config, ensure_ascii=False, indent=2))
        print("配置与数据检查通过；未下载模型，未执行分词长度过滤或训练。")
        return

    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；CPU 检查请指定 --device cpu --dtype float32")
    output = Path(config["output_dir"])
    if resume_from is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空，请更换 output_dir 或使用 resume_from：{output}")

    # 优先使用 SFT 保存的分词器；旧 adapter 没有分词器时回退到基座。
    adapter = Path(config["sft_adapter_path"])
    tokenizer_source = resume_from or (str(adapter) if (adapter / "tokenizer_config.json").is_file() else config["model_id"])
    tokenizer = load_tokenizer(model_id=tokenizer_source)
    records = load_records(config, tokenizer)
    dtype = getattr(torch, config["dtype"])
    print("加载 PPO 策略和价值头……", flush=True)
    model, value_head = load_ppo_policy(
        resume_from or config["sft_adapter_path"],
        model_id=config["model_id"], device=str(device), dtype=dtype,
    )
    if config["max_prompt_tokens"] + config["max_new_tokens"] > model.config.max_position_embeddings:
        raise ValueError("prompt 与回答的长度上限之和超过模型上下文长度")
    print("加载固定 SFT 参考模型……", flush=True)
    reference_model = load_reference_model(
        config["sft_adapter_path"], model_id=config["model_id"], device=str(device), dtype=dtype,
    )
    optimizer = build_ppo_optimizer(model, value_head, config["learning_rate"])
    sampler = random.Random(config["seed"])
    start_iteration = 0
    if state is not None:
        value_head.load_state_dict(torch.load(Path(resume_from) / "value_head.pt", map_location=device, weights_only=True))
        optimizer.load_state_dict(state["optimizer"])
        sampler.setstate(state["sampler_rng_state"])
        start_iteration = state["iteration"]
        # 模型、优化器等初始化完成后再恢复，保持下一次采样的随机状态。
        restore_rng_state(state)

    if state is None:
        save_run_config(config, output, data_hashes=hashes)
    print(f"开始 PPO：{start_iteration} → {config['max_iterations']} 次迭代；每次 {config['rollout_prompts']} 题", flush=True)
    print("mean_score 是训练采样得分，不是验证集准确率。", flush=True)
    for iteration in range(start_iteration + 1, config["max_iterations"] + 1):
        started = time.monotonic()
        # 每轮独立抽题，同一轮不重复；与 token 生成使用不同的随机数生成器。
        selected = sampler.sample(records, config["rollout_prompts"])
        metrics = train_ppo_iteration(
            model=model, reference_model=reference_model, value_head=value_head,
            tokenizer=tokenizer, questions=[record["question"] for record in selected],
            expected_answers=[record["answer"] for record in selected], optimizer=optimizer,
            **{name: config[name] for name in (
                "max_new_tokens", "update_epochs", "kl_coef", "gamma", "gae_lambda",
                "clip_epsilon", "value_coef", "max_grad_norm",
            )},
        )
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("PPO 日志含非有限数值，停止训练")
        metrics.update(iteration=iteration, updates=iteration * config["update_epochs"], seconds=time.monotonic() - started)
        append_metrics(metrics, output)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        if iteration % config["save_steps"] == 0:
            save_checkpoint(model, value_head, tokenizer, optimizer, sampler, iteration, config, hashes,
                            output / "checkpoints" / f"iteration-{iteration:06d}")

    save_checkpoint(model, value_head, tokenizer, optimizer, sampler, config["max_iterations"],
                    config, hashes, output / f"final-iteration-{config['max_iterations']:06d}")


def main():
    config, args = parse_args()
    run_training(config, resume_from=args.resume_from, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
