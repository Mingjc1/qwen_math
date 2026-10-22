"""在未安装本项目的情况下也能从源码检查配置。"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qwen_math.cli import main


if __name__ == "__main__":
    raise SystemExit(main(PROJECT_ROOT))
