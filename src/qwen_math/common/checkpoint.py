"""训练状态保存与恢复（待实现）。

保存 adapter/value_head、optimizer、scheduler、随机状态和配置。
SFT adapter 固定输出为 outputs/sft/adapter，后训练引用不可变快照。"""

from pathlib import Path
import hashlib
import json
import torch
import random

def restore_rng_state(state):
    random.setstate(state["python_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])

    cuda_states = state["cuda_rng_states"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("该 checkpoint 需要 CUDA")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("CUDA 设备数量与 checkpoint 不一致")

        torch.cuda.set_rng_state_all(cuda_states)

def load_training_state(
    optimizer, scheduler, checkpoint_dir
):
    path = Path(checkpoint_dir) / "training_state.pt"

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])

    return state

def save_training_state(
    optimizer, scheduler, epoch, update, output_dir,
    loss_sum, token_count, best_val_loss,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "update_in_epoch": update,
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else []
            ),
            "loss_sum": loss_sum,
            "token_count": token_count,
            "best_val_loss": best_val_loss,
        },
        output_dir / "training_state.pt",
    )


def file_sha256(path):
    """按块读取文件，计算其原始字节内容的 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def compute_data_hashes(config):
    """数据路径相对于当前工作目录，与现有数据读取函数一致。"""
    return {
        field: file_sha256(config[field])
        for field in ("train_path", "val_path")
    }


def save_run_config(config, output_dir, data_hashes=None):
    """保存配置及数据指纹，不修改传入的 CONFIG。

    训练入口传入启动时计算的 data_hashes，保证周期快照记录同一份数据指纹。
    单独调用时默认计算当前文件哈希。
    """
    snapshot = dict(config)
    snapshot["data_sha256"] = (
        compute_data_hashes(config) if data_hashes is None else dict(data_hashes)
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "run_config.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            snapshot, file,
            ensure_ascii=False,
            indent=2,
        )

def append_metrics(metrics, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "metrics.jsonl"
    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(metrics, ensure_ascii=False) + "\n"
        )

def save_adapter(model, tokenizer, output_dir):
    """保存 LoRA 适配器和配套 tokenizer。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def validate_resume_config(config, checkpoint_dir):
    path = Path(checkpoint_dir) / "run_config.json"
    with path.open("r", encoding="utf-8") as file:
        saved = json.load(file)

    # 哈希是快照元数据，需与当前文件内容比较，不参与普通配置字段比较。
    ignored = {"resume_from", "data_sha256"}
    old = {k: v for k, v in saved.items() if k not in ignored}
    new = {k: v for k, v in config.items() if k not in ignored}

    if old != new:
        raise ValueError("续训配置与 checkpoint 不一致")

    saved_hashes = saved.get("data_sha256")
    if not isinstance(saved_hashes, dict) or set(saved_hashes) != {"train_path", "val_path"}:
        raise ValueError("checkpoint 缺少完整数据哈希，无法确认数据一致性")
    current_hashes = compute_data_hashes(config)
    for field, current_hash in current_hashes.items():
        if current_hash != saved_hashes[field]:
            raise ValueError(
                f"续训数据内容已改变：{field}={config[field]}，SHA-256 与 checkpoint 不一致"
            )
