"""GRPO 训练入口：按题分组采样、规则评分、策略更新与适配器保存。"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from qwen_math.common.checkpoint import (
    append_metrics, file_sha256, save_adapter, save_run_config,
)
from qwen_math.common.seed import set_seed
from qwen_math.data.io import load_sft_records
from qwen_math.data.templates import format_prompt
from qwen_math.models.loader import (
    load_grpo_policy, load_reference_model, load_tokenizer,
)
from qwen_math.rewards.math_reward import normalize_numeric_answer
from qwen_math.trainers.dpo import build_dpo_optimizer
from qwen_math.trainers.grpo import train_grpo_iteration


CONFIG = {
    "seed": 42,
    "model_id": "Qwen/Qwen2.5-Math-1.5B",
    "sft_adapter_path": "outputs/sft/gsm8k_run02/best_adapter",
    "train_path": "data/processed/sft_train.jsonl",
    "output_dir": "outputs/grpo/gsm8k_run01",
    "device": "cuda",
    "dtype": "bfloat16",
    "group_size": 4,
    "rollout_prompts": 1,
    "max_prompt_tokens": 256,
    "max_new_tokens": 256,
    "max_iterations": 100,
    "update_epochs": 1,
    "learning_rate": 1e-6,
    "clip_epsilon": 0.2,
    "kl_coef": 0.01,
    "max_grad_norm": 1.0,
    "save_steps": 10,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON 配置，字段与本脚本 CONFIG 一致")
    parser.add_argument("--dry-run", action="store_true", help="仅检查配置和数据，不加载分词器或模型")
    for name, default in CONFIG.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default))
    args = parser.parse_args()
    config = dict(CONFIG)
    if args.config:
        with Path(args.config).open(encoding="utf-8") as file:
            overrides = json.load(file)
        if not isinstance(overrides, dict) or set(overrides) - set(CONFIG):
            parser.error("配置必须为 JSON 对象，且只能包含本脚本 CONFIG 的字段")
        config.update(overrides)
    for name in CONFIG:
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    return config, args


def validate_config(config):
    for name in (
        "group_size", "rollout_prompts", "max_prompt_tokens", "max_new_tokens",
        "max_iterations", "update_epochs", "save_steps",
    ):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} 必须为正整数")
    if config["group_size"] < 2:
        raise ValueError("group_size 必须至少为 2")
    if type(config["seed"]) is not int:
        raise ValueError("seed 必须为整数")
    for name in ("learning_rate", "clip_epsilon", "kl_coef", "max_grad_norm"):
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} 必须为有限非负数")
    if config["learning_rate"] == 0 or config["max_grad_norm"] == 0:
        raise ValueError("learning_rate 和 max_grad_norm 必须大于 0")
    if not 0 < config["clip_epsilon"] < 1:
        raise ValueError("clip_epsilon 必须在 (0, 1) 内")
    if config["dtype"] not in ("float32", "bfloat16"):
        raise ValueError("dtype 只支持 float32 或 bfloat16")
    for name in ("model_id", "sft_adapter_path", "train_path", "output_dir", "device"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"{name} 必须为非空字符串")


def load_records(config, tokenizer=None):
    """沿用 SFT 文件格式，仅将 question 和 answer 传给 GRPO。"""
    source = load_sft_records(config["train_path"])
    records = []
    for record in source:
        if normalize_numeric_answer(record["answer"]) is None:
            raise ValueError(f"标准答案不是支持的数值格式：{record['answer']!r}")
        if tokenizer is not None:
            prompt_ids = tokenizer.encode(format_prompt(record["question"]), add_special_tokens=False)
            if len(prompt_ids) > config["max_prompt_tokens"]:
                continue
        records.append({"question": record["question"], "answer": record["answer"]})
    if len(records) < config["rollout_prompts"]:
        raise ValueError("有效题目数量少于 rollout_prompts")
    print(f"GRPO 数据：原始 {len(source)} 题，保留 {len(records)} 题", flush=True)
    return records


def source_hashes(config):
    adapter = Path(config["sft_adapter_path"])
    hashes = {"train_path": file_sha256(config["train_path"])}
    for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        if (adapter / name).is_file():
            hashes["sft/" + name] = file_sha256(adapter / name)
    return hashes


def save_snapshot(model, tokenizer, directory, config, hashes):
    """保存推理用适配器和配置；不包含优化器或续训状态。"""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"适配器目录已存在，拒绝覆盖：{directory}")
    save_adapter(model, tokenizer, directory)
    save_run_config(config, directory, data_hashes=hashes)
    print(f"已保存：{directory}", flush=True)


def run_training(config, dry_run=False):
    validate_config(config)
    if dry_run:
        load_records(config)
        print(json.dumps(config, ensure_ascii=False, indent=2))
        print("配置与数据检查通过；未检查适配器、分词长度或 CUDA，未加载模型。")
        return

    adapter = Path(config["sft_adapter_path"])
    if not (adapter / "adapter_config.json").is_file() or not any(
        (adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise ValueError(f"找不到完整的 SFT 适配器：{adapter}")
    output = Path(config["output_dir"])
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"输出目录非空或不是目录，请更换 output_dir：{output}")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    set_seed(config["seed"])
    hashes = source_hashes(config)
    tokenizer_source = str(adapter) if (adapter / "tokenizer_config.json").is_file() else config["model_id"]
    tokenizer = load_tokenizer(model_id=tokenizer_source)
    records = load_records(config, tokenizer)
    model_kwargs = {
        "model_id": config["model_id"], "device": str(device), "dtype": getattr(torch, config["dtype"]),
    }
    print("加载可训练 GRPO 策略……", flush=True)
    model = load_grpo_policy(str(adapter), **model_kwargs)
    if config["max_prompt_tokens"] + config["max_new_tokens"] > model.config.max_position_embeddings:
        raise ValueError("prompt 与回答的长度上限之和超过模型上下文长度")
    print("加载固定 SFT 参考模型……", flush=True)
    reference_model = load_reference_model(str(adapter), **model_kwargs)
    # 与 DPO 相同，只优化策略的可训练参数；不创建价值头。
    optimizer = build_dpo_optimizer(model, learning_rate=config["learning_rate"])
    sampler = random.Random(config["seed"])
    save_run_config(config, output, data_hashes=hashes)
    response_count = config["rollout_prompts"] * config["group_size"]
    print(f"开始 GRPO：{config['max_iterations']} 次迭代，每次 {response_count} 条回答", flush=True)
    print("mean_score 是更新前采样得分，不是验证准确率；组内同分时奖励优势为零。", flush=True)

    for iteration in range(1, config["max_iterations"] + 1):
        started = time.monotonic()
        # 一次迭代内不重复抽取记录，各迭代间允许重复。
        selected = sampler.sample(records, config["rollout_prompts"])
        metrics = train_grpo_iteration(
            model=model, reference_model=reference_model, tokenizer=tokenizer,
            questions=[record["question"] for record in selected],
            expected_answers=[record["answer"] for record in selected], optimizer=optimizer,
            **{name: config[name] for name in (
                "group_size", "max_new_tokens", "update_epochs", "clip_epsilon", "kl_coef", "max_grad_norm",
            )},
        )
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("GRPO 日志包含非有限数值，停止训练")
        metrics.update(iteration=iteration, updates=iteration * config["update_epochs"],
                       seconds=time.monotonic() - started)
        append_metrics(metrics, output)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        if iteration % config["save_steps"] == 0:
            save_snapshot(model, tokenizer, output / "adapters" / f"iteration-{iteration:06d}", config, hashes)

    save_snapshot(model, tokenizer, output / "final_adapter", config, hashes)
    print("GRPO 训练完成。适配器已保存；本入口暂不支持断点续训。", flush=True)


def main():
    config, args = parse_args()
    run_training(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
