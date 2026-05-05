"""Tests for src.models.hybrid (MoveBoardHybrid)."""
from __future__ import annotations

import torch

from src.data.board_encoder import BOARD_SIZE, NUM_CHANNELS
from src.models.hybrid import MoveBoardHybrid


# Tiny vocab keeps the tests fast. IDs follow the project convention.
PAD, START, END, UNK = 0, 1, 2, 3
A, B, C, D = 4, 5, 6, 7
VOCAB_SIZE = 8


def _build(seed: int = 0, **overrides) -> MoveBoardHybrid:
    torch.manual_seed(seed)
    kwargs = dict(
        vocab_size=VOCAB_SIZE,
        pad_id=PAD,
        start_id=START,
        end_id=END,
        embedding_dim=16,
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
        cnn_channels=(8, 16, 16),
        board_feature_dim=24,
    )
    kwargs.update(overrides)
    return MoveBoardHybrid(**kwargs)


def _dummy_board_batch(batch_size: int) -> torch.Tensor:
    return torch.zeros(batch_size, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)


def test_forward_shape():
    model = _build()
    history = torch.tensor([[START, A, B, C], [START, B, A, PAD]], dtype=torch.long)
    board = _dummy_board_batch(2)
    logits = model(history, board)
    assert logits.shape == (2, VOCAB_SIZE)
    assert logits.dtype == torch.float32


def test_forward_backward_smoke():
    model = _build()
    history = torch.tensor([[START, A, B, C], [START, B, A, PAD]], dtype=torch.long)
    board = _dummy_board_batch(2)
    target = torch.tensor([D, A], dtype=torch.long)
    logits = model(history, board)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()

    grads_seen = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad for {name}"
            grads_seen += 1
    assert grads_seen > 0


def test_padding_idx_embedding_row_is_zero():
    model = _build()
    pad_row = model.embedding.weight[PAD]
    assert torch.all(pad_row == 0)


def test_predict_topk_excludes_specials_by_default():
    model = _build()
    history = torch.tensor([START, A, B], dtype=torch.long)
    board = torch.zeros(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
    result = model.predict_topk(history, board, k=VOCAB_SIZE)
    returned_ids = {tok for tok, _ in result}
    assert PAD not in returned_ids
    assert START not in returned_ids
    assert END not in returned_ids
    # All non-special ids should be present when k >= |non-specials|.
    assert returned_ids == {UNK, A, B, C, D}


def test_predict_topk_respects_legal_mask():
    model = _build()
    history = torch.tensor([START, A, B], dtype=torch.long)
    board = torch.zeros(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
    legal = [A, B, C]
    result = model.predict_topk(history, board, k=2, legal_token_ids=legal)
    assert len(result) == 2
    assert all(tok in legal for tok, _ in result)


def test_predict_topk_returns_descending_log_probs():
    model = _build()
    history = torch.tensor([START, A, B], dtype=torch.long)
    board = torch.zeros(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
    result = model.predict_topk(history, board, k=5)
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


def test_predict_topk_does_not_leave_model_in_eval_mode():
    model = _build()
    model.train()
    history = torch.tensor([START, A], dtype=torch.long)
    board = torch.zeros(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
    model.predict_topk(history, board, k=3)
    assert model.training is True


def test_save_load_roundtrip(tmp_path):
    model = _build(seed=123)
    model.eval()
    history = torch.tensor([[START, A, B, C]], dtype=torch.long)
    board = _dummy_board_batch(1)
    expected = model(history, board)

    path = tmp_path / "hybrid.pt"
    model.save(path)
    loaded = MoveBoardHybrid.load(path)
    loaded.eval()
    actual = loaded(history, board)

    assert torch.allclose(actual, expected, atol=1e-6)
    assert loaded.config.to_dict() == model.config.to_dict()


def test_board_signal_changes_predictions():
    """Two different boards with the same history should produce different logits.

    Without this, the CNN branch would be functionally absent and the hybrid
    would degenerate to "LSTM with extra parameters."
    """
    model = _build(seed=7)
    model.eval()
    history = torch.tensor([[START, A, B, C]], dtype=torch.long)
    board_zero = torch.zeros(1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
    board_random = torch.randn(1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    logits_zero = model(history, board_zero)
    logits_random = model(history, board_random)
    assert not torch.allclose(logits_zero, logits_random, atol=1e-4)
