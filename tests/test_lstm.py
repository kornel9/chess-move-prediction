"""Tests for src.models.lstm (MoveLSTM)."""
from __future__ import annotations

import torch

from src.models.lstm import MoveLSTM


# Tiny vocab keeps the tests fast. IDs follow the project convention.
PAD, START, END, UNK = 0, 1, 2, 3
A, B, C, D = 4, 5, 6, 7
VOCAB_SIZE = 8


def _build(seed: int = 0, **overrides) -> MoveLSTM:
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
    )
    kwargs.update(overrides)
    return MoveLSTM(**kwargs)


def test_forward_shape():
    model = _build()
    history = torch.tensor([[START, A, B, C], [START, B, A, PAD]], dtype=torch.long)
    logits = model(history)
    assert logits.shape == (2, 4, VOCAB_SIZE)
    assert logits.dtype == torch.float32


def test_forward_backward_smoke():
    model = _build()
    history = torch.tensor([[START, A, B, C], [START, B, A, PAD]], dtype=torch.long)
    targets = torch.tensor([[A, B, C, END], [B, A, END, PAD]], dtype=torch.long)
    logits = model(history)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        targets.reshape(-1),
        ignore_index=PAD,
    )
    loss.backward()

    # All trainable params receive gradient (padding_idx row of the embedding
    # is intentionally zero, so we exclude it from the check).
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
    result = model.predict_topk(history, k=VOCAB_SIZE)
    returned_ids = {tok for tok, _ in result}
    assert PAD not in returned_ids
    assert START not in returned_ids
    assert END not in returned_ids
    # All non-special ids should be present when k >= |non-specials|.
    assert returned_ids == {UNK, A, B, C, D}


def test_predict_topk_respects_legal_mask():
    model = _build()
    history = torch.tensor([START, A, B], dtype=torch.long)
    legal = [A, B, C]
    result = model.predict_topk(history, k=2, legal_token_ids=legal)
    assert len(result) == 2
    assert all(tok in legal for tok, _ in result)


def test_predict_topk_returns_descending_log_probs():
    model = _build()
    history = torch.tensor([START, A, B], dtype=torch.long)
    result = model.predict_topk(history, k=5)
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


def test_predict_topk_does_not_leave_model_in_eval_mode():
    model = _build()
    model.train()
    history = torch.tensor([START, A], dtype=torch.long)
    model.predict_topk(history, k=3)
    assert model.training is True


def test_save_load_roundtrip(tmp_path):
    model = _build(seed=123)
    model.eval()
    history = torch.tensor([[START, A, B, C]], dtype=torch.long)
    expected = model(history)

    path = tmp_path / "lstm.pt"
    model.save(path)
    loaded = MoveLSTM.load(path)
    loaded.eval()
    actual = loaded(history)

    assert torch.allclose(actual, expected, atol=1e-6)
    assert loaded.config.to_dict() == model.config.to_dict()
