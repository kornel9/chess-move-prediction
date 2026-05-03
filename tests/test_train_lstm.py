"""Smoke tests for src.training.train_lstm._run_epoch wiring."""
from __future__ import annotations

from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import ChessMoveSequenceDataset, pad_collate
from src.data.vocab import Vocab
from src.models.lstm import MoveLSTM
from src.training.train_lstm import _run_epoch


def _setup(seed: int = 0):
    torch.manual_seed(seed)
    games = [
        ["e2e4", "e7e5", "g1f3", "b8c6"],
        ["d2d4", "d7d5", "c2c4"],
        ["e2e4"],
    ]
    vocab = Vocab.build_from_games(games)
    ds = ChessMoveSequenceDataset(games, vocab)
    loader = DataLoader(
        ds, batch_size=2, shuffle=False, collate_fn=partial(pad_collate, pad_id=vocab.pad_id)
    )
    model = MoveLSTM(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        start_id=vocab.start_id,
        end_id=vocab.end_id,
        embedding_dim=8,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    return vocab, loader, model, criterion


def test_run_epoch_train_step_decreases_loss():
    vocab, loader, model, criterion = _setup()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    device = torch.device("cpu")

    loss_sum_before, tokens_before = _run_epoch(model, loader, criterion, device, None, 0.0)
    loss_before = loss_sum_before / tokens_before

    # A few training epochs on the tiny set should drive train loss down.
    for _ in range(20):
        _run_epoch(model, loader, criterion, device, optimizer, grad_clip=1.0)

    loss_sum_after, tokens_after = _run_epoch(model, loader, criterion, device, None, 0.0)
    loss_after = loss_sum_after / tokens_after

    assert tokens_before == tokens_after  # non-pad token count is dataset-determined
    assert loss_after < loss_before


def test_run_epoch_eval_does_not_update_params():
    _, loader, model, criterion = _setup()
    device = torch.device("cpu")
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}

    _run_epoch(model, loader, criterion, device, optimizer=None, grad_clip=0.0)

    for k, v in model.state_dict().items():
        assert torch.equal(v, snapshot[k]), f"param {k} changed during eval"


def test_run_epoch_counts_only_non_pad_targets():
    vocab, loader, model, criterion = _setup()
    device = torch.device("cpu")
    _, n_tokens = _run_epoch(model, loader, criterion, device, optimizer=None, grad_clip=0.0)

    # Expected: each game contributes len(game)+1 supervised positions
    # (the trailing <END>), padding is masked out by ignore_index.
    games = [
        ["e2e4", "e7e5", "g1f3", "b8c6"],
        ["d2d4", "d7d5", "c2c4"],
        ["e2e4"],
    ]
    expected = sum(len(g) + 1 for g in games)
    assert n_tokens == expected
