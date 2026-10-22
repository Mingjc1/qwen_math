"""SFT 训练入口：准备数据、训练与验证、保存适配器及续训状态。"""

import math
from pathlib import Path

import torch
from transformers import get_linear_schedule_with_warmup

from qwen_math.common.checkpoint import (
    append_metrics,
    compute_data_hashes,
    load_training_state,
    restore_rng_state,
    save_adapter,
    save_run_config,
    save_training_state,
    validate_resume_config,
)
from qwen_math.common.seed import set_seed
from qwen_math.data.collators import build_sft_dataloader
from qwen_math.data.io import load_sft_records
from qwen_math.data.templates import encode_sft_example, filter_long_samples
from qwen_math.models.loader import (
    add_lora,
    load_adapter,
    load_base_model,
    load_tokenizer,
)
from qwen_math.trainers.sft import evaluate_sft, train_sft_epoch

CONFIG = {
    "seed": 42,
    "train_path": "data/processed/sft_train.jsonl",
    "val_path": "data/processed/sft_val.jsonl",
    "max_length": 512,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "num_epochs": 1,
    "output_dir": "outputs/sft/gsm8k_run01",
    "accumulation_steps": 8,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.03,
    "save_steps": 100,
    "resume_from": None,
}


def prepare_samples(path, tokenizer, max_length, name):
    """读取、分词、过滤，并输出一行数据统计。"""
    records = load_sft_records(path)
    samples = [
        encode_sft_example(record["question"], record["solution"], tokenizer)
        for record in records
    ]
    longest = max(len(sample["input_ids"]) for sample in samples)
    samples = filter_long_samples(samples, max_length=max_length)
    print(
        f"{name}：原始 {len(records)} 条，保留 {len(samples)} 条，"
        f"过滤 {len(records) - len(samples)} 条，过滤前最长 {longest} token"
    )
    return samples


def load_training_model(resume_from):
    """从基座新建 LoRA，或恢复可训练的适配器。"""
    base_model = load_base_model()
    if resume_from is None:
        model = add_lora(base_model)
    else:
        model = load_adapter(base_model, resume_from, is_trainable=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    return model


def main():
    resume_from = CONFIG["resume_from"]
    output_dir = Path(CONFIG["output_dir"])
    set_seed(CONFIG["seed"])
    if resume_from is not None:
        validate_resume_config(CONFIG, resume_from)

    # 数据指纹只在启动时计算，周期 checkpoint 复用它。
    data_hashes = compute_data_hashes(CONFIG)
    if resume_from is None:
        tokenizer = load_tokenizer()
    else:
        tokenizer = load_tokenizer(model_id=resume_from)

    train_samples = prepare_samples(
        CONFIG["train_path"], tokenizer, CONFIG["max_length"], "训练集"
    )
    val_samples = prepare_samples(
        CONFIG["val_path"], tokenizer, CONFIG["max_length"], "验证集"
    )
    return  # 临时停止，避免继续加载模型和启动训练

    # 数据加载器使用独立生成器，每轮按 seed + epoch 重建顺序。
    train_loader = build_sft_dataloader(
        train_samples,
        tokenizer.pad_token_id,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(CONFIG["seed"]),
    )
    val_loader = build_sft_dataloader(
        val_samples,
        tokenizer.pad_token_id,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        generator=torch.Generator().manual_seed(CONFIG["seed"]),
    )

    print("训练批次数:", len(train_loader))
    print("验证批次数:", len(val_loader))
    if resume_from is None:
        save_run_config(CONFIG, output_dir, data_hashes=data_hashes)

    model = load_training_model(resume_from)
    if resume_from is None:
        baseline_loss = evaluate_sft(model, val_loader)
        print(f"训练前验证 loss：{baseline_loss:.6f}")
        append_metrics(
            {"event": "baseline", "epoch": 0, "val_loss": baseline_loss},
            output_dir,
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CONFIG["learning_rate"],
    )

    # 调度器步数按参数更新次数计算，而非微批次数。
    num_epochs = CONFIG["num_epochs"]
    updates_per_epoch = math.ceil(
        len(train_loader) / CONFIG["accumulation_steps"]
    )
    total_updates = updates_per_epoch * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * CONFIG["warmup_ratio"]),
        num_training_steps=total_updates,
    )

    # 其他初始化完成后再恢复随机状态。
    state = None
    start_epoch = 0
    best_val_loss = float("inf")
    if resume_from is not None:
        state = load_training_state(optimizer, scheduler, resume_from)
        start_epoch = state["epoch"]
        best_val_loss = state["best_val_loss"]
        restore_rng_state(state)

    for epoch in range(start_epoch, num_epochs):
        resuming = state is not None and epoch == start_epoch

        train_loader.generator.manual_seed(CONFIG["seed"] + epoch)

        def save_periodic(update, loss_sum, token_count):
            if update % CONFIG["save_steps"] != 0:
                return
            checkpoint_dir = (
                output_dir / "checkpoints"
                / f"epoch-{epoch + 1}-update-{update:06d}"
            )
            save_adapter(model, tokenizer, checkpoint_dir)
            save_run_config(CONFIG, checkpoint_dir, data_hashes=data_hashes)
            save_training_state(
                optimizer, scheduler, epoch, update, checkpoint_dir,
                loss_sum=loss_sum,
                token_count=token_count,
                best_val_loss=best_val_loss,
            )

        train_loss = train_sft_epoch(
            model,
            train_loader,
            optimizer,
            accumulation_steps=CONFIG["accumulation_steps"],
            max_grad_norm=CONFIG["max_grad_norm"],
            scheduler=scheduler,
            on_update=save_periodic,
            skip_updates=state["update_in_epoch"] if resuming else 0,
            initial_loss_sum=state["loss_sum"] if resuming else 0.0,
            initial_token_count=state["token_count"] if resuming else 0,
        )
        val_loss = evaluate_sft(model, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_adapter(model, tokenizer, output_dir / "best_adapter")
            print(f"保存最佳适配器，val_loss={val_loss:.6f}")

        append_metrics(
            {
                "event": "epoch_end",
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            output_dir,
        )

        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

    save_adapter(model, tokenizer, output_dir / "final_adapter")


if __name__ == "__main__":
    main()
