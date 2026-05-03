"""Deterministic seeding helper for training scripts."""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed Python ``random``, NumPy, and PyTorch (CPU + all CUDA devices).

    This does *not* enable ``torch.backends.cudnn.deterministic`` because the
    perf hit isn't worth it for our scale; runs may still differ slightly on
    GPU due to non-deterministic CuDNN kernels.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
