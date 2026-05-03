"""Tests for src.data.dataset (ChessMoveSequenceDataset, pad_collate)."""
from __future__ import annotations

from functools import partial

import torch
from torch.utils.data import DataLoader

from src.data.dataset import ChessMoveSequenceDataset, pad_collate
from src.data.vocab import Vocab


def _vocab(games: list[list[str]]) -> Vocab:
    return Vocab.build_from_games(games)


def test_sequence_sample_shapes_and_shift():
    games = [["e2e4", "e7e5", "g1f3"]]
    vocab = _vocab(games)
    ds = ChessMoveSequenceDataset(games, vocab)

    assert len(ds) == 1
    inp, tgt = ds[0]
    assert inp.shape == (4,)
    assert tgt.shape == (4,)
    assert inp[0].item() == vocab.start_id
    assert tgt[-1].item() == vocab.end_id
    # Shifted-by-one: target at position t equals input at position t+1, except
    # at the boundary where input ends with the last move and target ends with <END>.
    for t in range(len(inp) - 1):
        assert tgt[t].item() == inp[t + 1].item()


def test_sequence_targets_match_encoded_moves():
    games = [["e2e4", "c7c5", "g1f3"]]
    vocab = _vocab(games)
    ds = ChessMoveSequenceDataset(games, vocab)
    _, tgt = ds[0]
    assert tgt[0].item() == vocab.encode("e2e4")
    assert tgt[1].item() == vocab.encode("c7c5")
    assert tgt[2].item() == vocab.encode("g1f3")
    assert tgt[3].item() == vocab.end_id


def test_pad_collate_pads_to_longest():
    games = [["e2e4"], ["e2e4", "e7e5", "g1f3"]]
    vocab = _vocab(games)
    ds = ChessMoveSequenceDataset(games, vocab)
    collate = partial(pad_collate, pad_id=vocab.pad_id)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)
    inputs, targets = next(iter(loader))

    # Longest game has 3 plies → length 4 ([<START>, m0, m1, m2]).
    assert inputs.shape == (2, 4)
    assert targets.shape == (2, 4)
    # The shorter sample (1 ply, length 2) is right-padded.
    short_idx = 0  # games[0] is the 1-ply game and shuffle=False
    assert inputs[short_idx, 2:].tolist() == [vocab.pad_id, vocab.pad_id]
    assert targets[short_idx, 2:].tolist() == [vocab.pad_id, vocab.pad_id]


def test_pad_collate_preserves_dtype():
    games = [["e2e4"], ["e2e4", "e7e5"]]
    vocab = _vocab(games)
    ds = ChessMoveSequenceDataset(games, vocab)
    collate = partial(pad_collate, pad_id=vocab.pad_id)
    inputs, targets = collate([ds[0], ds[1]])
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long


def test_empty_game_yields_start_to_end():
    # Edge case: a zero-ply game should still produce a (1,) input/target pair
    # with input=[<START>] and target=[<END>], so the model can learn end-of-game
    # from an empty history.
    games = [[]]
    # Build vocab from a non-empty game so specials still resolve cleanly.
    vocab = _vocab([["e2e4"]])
    ds = ChessMoveSequenceDataset(games, vocab)
    inp, tgt = ds[0]
    assert inp.tolist() == [vocab.start_id]
    assert tgt.tolist() == [vocab.end_id]
