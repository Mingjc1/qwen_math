"""当前提供 dry-run；算法训练循环将逐步接入。"""

import argparse
import json
from pathlib import Path

from qwen_math.config import load_config


def main(project_root: Path) -> int:
    parser = argparse.ArgumentParser(description="Qwen Math 配置检查与训练入口")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="只校验并展示配置，不加载模型")
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else project_root / args.config
    try:
        config = load_config(path)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    if not args.dry_run:
        parser.exit(2, f"{config.algorithm} 训练器尚未实现；使用 --dry-run 检查配置。\n")
    print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
    print(f"项目根目录：{project_root}")
    print(f"训练数据：{project_root / config.train_data}")
    print(f"输出目录：{project_root / config.output_dir}")
    print("配置校验通过；未检查数据/权重/CUDA，未执行训练。")
    if config.model_revision is None:
        print("待办：正式实验前锁定模型 revision。")
    return 0
