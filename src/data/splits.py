"""Game-level deterministic train/val/test splits."""
from __future__ import annotations

import random
from typing import Sequence


DEFAULT_RATIOS: tuple[float, float, float] = (0.85, 0.10, 0.05)


def split_games(
    games: Sequence[Sequence[str]],
    seed: int = 42,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """Split ``games`` into train/val/test by game (never by position).

    The split is deterministic given ``seed``. ``ratios`` must sum to 1.0
    (within float tolerance). Train gets ``round(n * ratios[0])`` games, val
    gets ``round(n * ratios[1])``, and test gets whatever remains so that the
    three lists exactly partition the input.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios} (sum={sum(ratios)})")
    if any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be non-negative, got {ratios}")

    n = len(games)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    if n_train + n_val > n:
        n_val = n - n_train

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train = [list(games[i]) for i in train_idx]
    val = [list(games[i]) for i in val_idx]
    test = [list(games[i]) for i in test_idx]
    return train, val, test
