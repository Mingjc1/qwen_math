"""DPO 训练入口：独立偏好训练/验证集、日志与 LoRA 适配器保存。"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from qwen_math.common.checkpoint import (
    append_metrics, file_sha256, save_adapter, save_run_config,
)
from qwen_math.common.seed import set_seed
from qwen_math.data.collators import build_preference_dataloader
from qwen_math.data.preferences import (
    encode_preference_example, filter_long_preferences, load_preference_records,
)
from qwen_math.models.loader import (
    load_dpo_policy, load_reference_model, load_tokenizer,
)
from qwen_math.trainers.dpo import (
    build_dpo_optimizer, evaluate_dpo, train_dpo_epoch,
)


CONFIG = {
    "seed": 42,
    "model_id": "Qwen/Qwen2.5-Math-1.5B",
    "sft_adapter_path": "outputs/sft/gsm8k_run02/best_adapter",
    "train_path": "data/preferences/dpo_train.jsonl",
    "val_path": "data/preferences/dpo_val.jsonl",
    "output_dir": "outputs/dpo/gsm8k_run01",
    "device": "cuda",
    "dtype": "bfloat16",
    "max_length": 512,
    "batch_size": 1,
    "learning_rate": 1e-6,
    "num_epochs": 1,
    "beta": 0.1,
    "gradient_checkpointing": True,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON 配置，字段与本脚本 CONFIG 一致")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置和数据，不加载分词器或模型")
    for name, default in CONFIG.items():
        option = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(option, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(option, type=type(default))
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
    for name in ("max_length", "batch_size", "num_epochs"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} 必须为正整数")
    if type(config["seed"]) is not int:
        raise ValueError("seed 必须为整数")
    for name in ("learning_rate", "beta"):
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必须为有限正数")
    for name in ("model_id", "sft_adapter_path", "train_path", "val_path", "output_dir", "device"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"{name} 必须为非空字符串")
    if config["dtype"] not in ("float32", "bfloat16"):
        raise ValueError("dtype 只支持 float32 或 bfloat16")
    if type(config["gradient_checkpointing"]) is not bool:
        raise ValueError("gradient_checkpointing 必须为布尔值")


def load_records(config):
    train = load_preference_records(config["train_path"])
    val = load_preference_records(config["val_path"])
    # 同题的不同回答不能分散到训练和验证集。
    train_questions = {record["question"].strip() for record in train}
    val_questions = {record["question"].strip() for record in val}
    if train_questions & val_questions:
        raise ValueError("训练集和验证集存在重复题目，请按题目划分数据")
    for name, records in (("训练集", train), ("验证集", val)):
        if any(record["chosen"].strip() == record["rejected"].strip() for record in records):
            raise ValueError(f"{name}存在 chosen 与 rejected 相同的记录")
        print(f"{name}：{len(records)} 对", flush=True)
    return train, val


def prepare_samples(records, tokenizer, max_length, name):
    samples = [encode_preference_example(
        record["question"], record["chosen"], record["rejected"], tokenizer,
    ) for record in records]
    filtered = filter_long_preferences(samples, max_length=max_length)
    print(f"{name}：保留 {len(filtered)} 对，过滤 {len(samples) - len(filtered)} 对", flush=True)
    return filtered


def run_training(config, dry_run=False):
    validate_config(config)
    train_records, val_records = load_records(config)
    hashes = {name: file_sha256(config[name]) for name in ("train_path", "val_path")}
    if dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        print("配置与偏好数据检查通过；未检查适配器、分词长度或 CUDA，未加载模型。")
        return

    adapter = Path(config["sft_adapter_path"])
    if not (adapter / "adapter_config.json").is_file() or not any(
        (adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise ValueError(f"找不到完整的 SFT 适配器：{adapter}")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    output = Path(config["output_dir"])
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"输出目录非空或不是目录，请更换 output_dir：{output}")

    set_seed(config["seed"])
    tokenizer_source = str(adapter) if (adapter / "tokenizer_config.json").is_file() else config["model_id"]
    tokenizer = load_tokenizer(model_id=tokenizer_source)
    train_samples = prepare_samples(train_records, tokenizer, config["max_length"], "训练集")
    val_samples = prepare_samples(val_records, tokenizer, config["max_length"], "验证集")
    train_loader = build_preference_dataloader(
        train_samples, tokenizer.pad_token_id, batch_size=config["batch_size"], shuffle=True,
    )
    val_loader = build_preference_dataloader(
        val_samples, tokenizer.pad_token_id, batch_size=config["batch_size"], shuffle=False,
    )

    model_kwargs = {"model_id": config["model_id"], "device": str(device), "dtype": getattr(torch, config["dtype"])}
    print("加载可训练策略……", flush=True)
    model = load_dpo_policy(str(adapter), **model_kwargs)
    if config["max_length"] > model.config.max_position_embeddings:
        raise ValueError("max_length 超过模型上下文长度")
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("加载固定 SFT 参考模型……", flush=True)
    reference_model = load_reference_model(str(adapter), **model_kwargs)
    optimizer = build_dpo_optimizer(model, learning_rate=config["learning_rate"])
    for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        if (adapter / name).is_file():
            hashes["sft/" + name] = file_sha256(adapter / name)
    save_run_config(config, output, data_hashes=hashes)

    baseline = evaluate_dpo(model, reference_model, val_loader, beta=config["beta"])
    if not math.isfinite(baseline):
        raise FloatingPointError("初始验证损失不是有限数值")
    print(f"训练前验证 loss：{baseline:.6f}", flush=True)
    append_metrics({"epoch": 0, "val_loss": baseline}, output)
    best_val_loss = baseline
    # 初始 SFT 也是候选，避免所有 epoch 都退化时仍选择退化模型。
    save_adapter(model, tokenizer, output / "best_adapter")
    for epoch in range(1, config["num_epochs"] + 1):
        started = time.monotonic()
        train_loss = train_dpo_epoch(model, reference_model, train_loader, optimizer, beta=config["beta"])
        val_loss = evaluate_dpo(model, reference_model, val_loader, beta=config["beta"])
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise FloatingPointError("训练或验证损失不是有限数值")
        metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   "seconds": time.monotonic() - started}
        append_metrics(metrics, output)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        save_adapter(model, tokenizer, output / f"epoch-{epoch:03d}-adapter")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_adapter(model, tokenizer, output / "best_adapter")
    save_adapter(model, tokenizer, output / "final_adapter")
    print(f"训练完成，最佳验证 loss：{best_val_loss:.6f}；输出：{output}", flush=True)


def main():
    config, args = parse_args()
    run_training(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
