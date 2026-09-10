from __future__ import annotations

import os
import random

import numpy as np
import torch


def derive_episode_seed(base_seed: int, episode: int) -> int:
    """Return a unique deterministic seed for an evaluation episode.

    Cantor pairing prevents the overlap produced by ``base_seed + episode``
    when consecutive base seeds each evaluate many episodes.
    """
    base_seed = int(base_seed)
    episode = int(episode)
    if base_seed < 0 or episode < 0:
        raise ValueError("base_seed and episode must be non-negative")
    pair_sum = base_seed + episode
    return pair_sum * (pair_sum + 1) // 2 + episode


def set_seed(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 的随机种子。

    环境内部还会持有自己的 numpy Generator；这里主要保证网络初始化、
    replay buffer 采样和普通随机函数在同一 seed 下尽量可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
