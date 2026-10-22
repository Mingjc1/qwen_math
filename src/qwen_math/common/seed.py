"""随机种子与复现（待实现）。

处理 random、torch、CUDA 与 DataLoader；记录环境版本，不承诺跨硬件逐位一致。"""

import random
import torch


def set_seed(seed):
    """设置当前使用的随机数生成器。"""
    # 1. 调用 random.seed(seed)
    # 2. 调用 torch.manual_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
