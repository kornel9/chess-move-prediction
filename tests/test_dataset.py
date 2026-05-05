"""Tests for src.data.dataset (ChessMoveSequenceDataset, pad_collate, ChessMovePerPlyIterable)."""
from __future__ import annotations

from functools import partial

import chess
import torch
from torch.utils.data import DataLoader

from src.data.board_encoder import BOARD_SIZE, NUM_CHANNELS, encode_board
from src.data.dataset import (
    ChessMoveSequenceDataset,
    ChessMovePerPlyIterable,
    pad_collate,
)
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


# --- ChessMovePerPlyIterable -------------------------------------------------


def test_iterable_yields_one_sample_per_ply():
    games = [["e2e4", "e7e5", "g1f3"], ["d2d4", "d7d5"]]
    vocab = _vocab(games)
    ds = ChessMovePerPlyIterable(games, vocab, history_len=8, shuffle=False)
    samples = list(ds)
    assert len(samples) == sum(len(g) for g in games) == 5
    assert len(ds) == 5

    # Every sample is a (history[8], board[18,8,8], target_id) triple with the
    # right shapes and dtypes.
    for h, b, t in samples:
        assert h.shape == (8,)
        assert h.dtype == torch.long
        assert b.shape == (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        assert b.dtype == torch.float32
        assert t.shape == ()
        assert t.dtype == torch.long


def test_iterable_history_truncation_and_padding():
    """history_len=4: a position with 0 prior moves gets [PAD, PAD, PAD, START];
    a position with 5+ prior moves keeps only the last 4 tokens."""
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]
    vocab = _vocab([moves])
    ds = ChessMovePerPlyIterable([moves], vocab, history_len=4, shuffle=False)
    samples = list(ds)

    # First ply: history=[<START>], padded to 4 → [PAD, PAD, PAD, START]
    h0, _, t0 = samples[0]
    assert h0.tolist() == [vocab.pad_id, vocab.pad_id, vocab.pad_id, vocab.start_id]
    assert t0.item() == vocab.encode("e2e4")

    # Fourth ply: history=[<START>, e2e4, e7e5, g1f3], length 4 already → no padding
    h3, _, t3 = samples[3]
    assert h3.tolist() == [
        vocab.start_id,
        vocab.encode("e2e4"),
        vocab.encode("e7e5"),
        vocab.encode("g1f3"),
    ]
    assert t3.item() == vocab.encode("b8c6")

    # Sixth ply (last): history grew to length 7, truncated to last 4
    h5, _, t5 = samples[5]
    assert h5.tolist() == [
        vocab.encode("e7e5"),
        vocab.encode("g1f3"),
        vocab.encode("b8c6"),
        vocab.encode("f1c4"),
    ]
    assert t5.item() == vocab.encode("g8f6")


def test_iterable_board_state_matches_python_chess():
    """The board tensor at each ply must match a board built independently
    by replaying the same moves with python-chess."""
    moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
    vocab = _vocab([moves])
    ds = ChessMovePerPlyIterable([moves], vocab, history_len=8, shuffle=False)
    samples = list(ds)

    expected_board = chess.Board()
    for ply, (uci, sample) in enumerate(zip(moves, samples)):
        _, b, _ = sample
        expected_tensor = torch.from_numpy(encode_board(expected_board))
        assert torch.equal(b, expected_tensor), f"board mismatch at ply {ply}"
        expected_board.push_uci(uci)


def test_iterable_works_with_dataloader_batching():
    games = [["e2e4", "e7e5", "g1f3"], ["d2d4", "d7d5"]]
    vocab = _vocab(games)
    ds = ChessMovePerPlyIterable(games, vocab, history_len=4, shuffle=False)
    loader = DataLoader(ds, batch_size=2)
    batches = list(loader)
    # 5 total samples, batch_size 2 → 3 batches (sizes 2, 2, 1)
    assert [b[0].shape[0] for b in batches] == [2, 2, 1]
    h, b, t = batches[0]
    assert h.shape == (2, 4)
    assert b.shape == (2, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert t.shape == (2,)


def test_iterable_shuffle_changes_game_order_but_not_per_game_ordering():
    """Within a game, plies must always come in chronological order. Shuffling
    only reorders games."""
    games = [["e2e4"], ["d2d4"], ["c2c4"]]
    vocab = _vocab(games)
    ds = ChessMovePerPlyIterable(games, vocab, history_len=4, shuffle=True, seed=0)
    targets_a = [t.item() for _, _, t in ds]
    ds.set_epoch(7)  # different seed offset
    targets_b = [t.item() for _, _, t in ds]
    # Same multiset of targets (one per game) regardless of shuffle.
    assert sorted(targets_a) == sorted(targets_b)
    # Each game has only 1 ply, so within-game order is trivial; assert that
    # the union of targets covers exactly the three games' first moves.
    expected = {vocab.encode("e2e4"), vocab.encode("d2d4"), vocab.encode("c2c4")}
    assert set(targets_a) == expected
