# Python 工具：配置、文件与记录

这一篇作为查阅表。大部分来自标准库，无须额外安装。

## 1. pathlib 与 JSON

可直接运行，会在当前目录创建一个临时演示目录，并在退出时自动删除：

```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory(prefix="qwen_math_demo_") as folder:
    path = Path(folder) / "config.json"
    config = {"algorithm": "dpo", "beta": 0.1, "seed": 42}
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    restored = json.loads(path.read_text(encoding="utf-8"))
    print(restored)
```

| 接口 | 用途 |
| --- | --- |
| `Path(...) / "file"` | 拼接跨平台路径 |
| `path.mkdir(parents=True, exist_ok=True)` | 创建输出目录 |
| `path.exists()` | 判断文件存在 |
| `path.read_text / write_text` | 文本读写，明确 UTF-8 |
| `json.loads / dumps` | 字符串与字典互转 |
| `json.load / dump` | 对文件对象进行 JSON 读写 |

JSONL 每行一个 JSON 对象，适合逐条读取题目。
PyTorch 张量不能直接 JSON 序列化；单个数用 item，数组用 detach().cpu().tolist()。

## 2. dataclasses 与 argparse

可直接运行：

```python
from dataclasses import dataclass, asdict
import argparse

@dataclass
class TrainConfig:
    algorithm: str = "dpo"
    beta: float = 0.1
    seed: int = 42

parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", choices=["dpo", "ppo", "grpo"], default="dpo")
args = parser.parse_args([])  # 示例固定为空；脚本中改成 parse_args()
config = TrainConfig(algorithm=args.algorithm)
print(asdict(config))
```

dataclass 描述配置字段，但默认不会严格校验运行时类型和取值范围；应主动检查 beta、组大小等参数。
第一版 JSON + dataclass 足够，暂时不必引入复杂配置框架。

## 3. logging 与随机种子

可直接运行，需要 torch：

```python
import logging
import random
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
random.seed(42)
torch.manual_seed(42)
logging.info("CPU toy tensor: %s", torch.randn(2).tolist())
```

如果使用 NumPy，还需 np.random.seed；DataLoader worker、分布式进程也要处理种子。
固定种子不保证跨硬件、跨软件版本逐位相同。

## 4. 稍后补充的第三方工具

| 工具 | 接口 | 何时使用 |
| --- | --- | --- |
| tqdm | `tqdm(loader)` | 给本地调试循环加进度显示 |
| TensorBoard | `SummaryWriter.add_scalar(...)` | 记录损失、KL、奖励曲线 |
| pytest | `assert`、`pytest.raises`、参数化测试 | 验证损失梯度、mask、终止轨迹 |
| Math-Verify | `parse(...)`、`verify(gold, prediction)` | 正式奖励器的数学答案校验 |

Math-Verify 的解析模式与容差应单独设计并人工抽查，不能等同于证明推理过程正确。
vLLM、DeepSpeed、分布式训练优化等留到正式训练阶段；当前不需要一次安装全部依赖。

## 官方资料

- [pathlib](https://docs.python.org/3/library/pathlib.html)
- [dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [argparse](https://docs.python.org/3/library/argparse.html)
- [logging](https://docs.python.org/3/library/logging.html)
- [Math-Verify](https://github.com/huggingface/Math-Verify)
