"""统一数学评估入口：支持基座与各算法保存的 LoRA 策略。"""

import argparse
import json
from pathlib import Path

import torch

from qwen_math.common.checkpoint import file_sha256
from qwen_math.common.seed import set_seed
from qwen_math.data.io import load_sft_records
from qwen_math.evaluation.runner import evaluate_dataset
from qwen_math.models.loader import load_adapter, load_base_model, load_tokenizer


CONFIG = {
    "seed": 42,
    "model_id": "Qwen/Qwen2.5-Math-1.5B",
    "adapter_path": None,  # None 表示评估基座；否则填写策略适配器目录。
    "data_path": "data/processed/sft_val.jsonl",
    "output_dir": "outputs/evaluation/base_val_smoke",
    "max_samples": 20,
    "max_new_tokens": 256,
}


def load_eval_model(
    model_id="Qwen/Qwen2.5-Math-1.5B",
    adapter_path=None,
    device="cuda",
    dtype=torch.bfloat16,
):
    """不传 adapter_path 评估基座，否则加载冻结的策略适配器。"""
    model = load_base_model(model_id=model_id, device=device, dtype=dtype)
    if adapter_path is not None:
        model = load_adapter(model, adapter_path, is_trainable=False)
    model.requires_grad_(False)
    model.eval()
    return model


def save_eval_config(config, sample_count):
    """保存配置、实际评估题数及完整数据文件的 SHA-256。"""
    snapshot = dict(config)
    snapshot["sample_count"] = sample_count
    snapshot["data_sha256"] = file_sha256(config["data_path"])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "eval_config.json").open("w", encoding="utf-8") as file:
        json.dump(snapshot, file, ensure_ascii=False, indent=2)


def main(dry_run=False):
    for name in ("max_samples", "max_new_tokens"):
        if type(CONFIG[name]) is not int or CONFIG[name] <= 0:
            raise ValueError(f"{name} 必须为正整数")

    set_seed(CONFIG["seed"])
    records = load_sft_records(CONFIG["data_path"])[:CONFIG["max_samples"]]
    print(f"评估题数：{len(records)}", flush=True)
    if dry_run:
        print(json.dumps(CONFIG, ensure_ascii=False, indent=2))
        print("配置与数据检查通过；未检查适配器、分词长度或 CUDA，未加载模型。")
        return

    output_dir = Path(CONFIG["output_dir"])
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"输出目录非空或不是目录，请更换 output_dir：{output_dir}")

    # 优先使用训练时保存的 tokenizer，适配器未包含 tokenizer 时使用基座。
    adapter_path = CONFIG["adapter_path"]
    if adapter_path is not None:
        adapter = Path(adapter_path)
        if not (adapter / "adapter_config.json").is_file() or not any(
            (adapter / name).is_file()
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        ):
            raise ValueError(f"找不到完整的策略适配器：{adapter}")
    tokenizer_source = CONFIG["model_id"]
    if adapter_path is not None and (Path(adapter_path) / "tokenizer_config.json").is_file():
        tokenizer_source = adapter_path
    tokenizer = load_tokenizer(model_id=tokenizer_source)
    model = load_eval_model(model_id=CONFIG["model_id"], adapter_path=adapter_path)
    save_eval_config(CONFIG, sample_count=len(records))
    metrics = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        records=records,
        output_dir=CONFIG["output_dir"],
        max_new_tokens=CONFIG["max_new_tokens"],
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="仅检查配置和数据，不加载模型")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
