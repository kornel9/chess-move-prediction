"""Tests for src.data.splits."""
from __future__ import annotations

import pytest

from src.data.splits import split_games


def _dummy_games(n: int) -> list[list[str]]:
    """Each game is a single-element list containing its index as a string —
    enough to identify the game across splits."""
    return [[str(i)] for i in range(n)]


def test_split_sizes_default_ratios():
    games = _dummy_games(100)
    train, val, test = split_games(games, seed=42)
    assert len(train) == 85
    assert len(val) == 10
    assert len(test) == 5


def test_split_is_deterministic_given_seed():
    games = _dummy_games(50)
    a = split_games(games, seed=7)
    b = split_games(games, seed=7)
    assert a == b


def test_different_seeds_produce_different_splits():
    games = _dummy_games(50)
    a = split_games(games, seed=1)
    b = split_games(games, seed=2)
    assert a != b


def test_no_leakage_between_splits():
    games = _dummy_games(73)
    train, val, test = split_games(games, seed=42)
    assert len(train) + len(val) + len(test) == 73

    flat = lambda chunks: {g[0] for g in chunks}
    s_train, s_val, s_test = flat(train), flat(val), flat(test)
    assert s_train | s_val | s_test == {str(i) for i in range(73)}
    assert s_train.isdisjoint(s_val)
    assert s_train.isdisjoint(s_test)
    assert s_val.isdisjoint(s_test)


def test_ratios_must_sum_to_one():
    games = _dummy_games(10)
    with pytest.raises(ValueError):
        split_games(games, ratios=(0.5, 0.4, 0.2))


def test_zero_test_split_allowed():
    games = _dummy_games(10)
    train, val, test = split_games(games, ratios=(0.5, 0.5, 0.0))
    assert len(train) == 5
    assert len(val) == 5
    assert test == []


def test_negative_ratio_rejected():
    games = _dummy_games(10)
    with pytest.raises(ValueError):
        split_games(games, ratios=(1.1, 0.0, -0.1))
