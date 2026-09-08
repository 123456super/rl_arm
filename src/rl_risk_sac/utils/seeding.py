from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 的随机种子。

    环境内部还会持有自己的 numpy Generator；这里主要保证网络初始化、
    replay buffer 采样和普通随机函数在同一 seed 下尽量可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
