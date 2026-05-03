"""Tests for src.models.ngram (TrigramKatz)."""
from __future__ import annotations

import math

import pytest

from src.models.ngram import TrigramKatz


# Tiny vocab keeps probability-normalisation tests fast.
# IDs follow the project convention: PAD=0, START=1, END=2, UNK=3, then "real" tokens.
PAD, START, END, UNK = 0, 1, 2, 3
A, B, C, D = 4, 5, 6, 7
VOCAB_SIZE = 8


def _fit(games: list[list[int]], discount: float = 0.5) -> TrigramKatz:
    return TrigramKatz.fit(
        encoded_games=games,
        vocab_size=VOCAB_SIZE,
        pad_id=PAD,
        start_id=START,
        end_id=END,
        discount=discount,
    )


def test_fit_records_expected_counts():
    # Game: [A, B, C] →  triples (S,S,A), (S,A,B), (A,B,C), (B,C,END)
    model = _fit([[A, B, C]])

    assert model.unigram_counts == {A: 1, B: 1, C: 1, END: 1}
    assert model.unigram_total == 4
    assert model.bigram_counts[START] == {A: 1}
    assert model.bigram_counts[A] == {B: 1}
    assert model.bigram_counts[B] == {C: 1}
    assert model.bigram_counts[C] == {END: 1}
    assert model.trigram_counts[(START, START)] == {A: 1}
    assert model.trigram_counts[(START, A)] == {B: 1}
    assert model.trigram_counts[(A, B)] == {C: 1}
    assert model.trigram_counts[(B, C)] == {END: 1}


def test_probability_sums_to_one_for_seen_trigram_context():
    model = _fit([[A, B, C, A, B], [A, C, B]])
    context = (A, B)  # seen, with C as observed continuation
    total = sum(math.exp(model.log_prob(w, context)) for w in range(VOCAB_SIZE))
    assert total == pytest.approx(1.0, abs=1e-9)


def test_probability_sums_to_one_for_unseen_trigram_context():
    model = _fit([[A, B, C, A, B], [A, C, B]])
    context = (D, A)  # trigram context never seen → backs off to bigram on A
    total = sum(math.exp(model.log_prob(w, context)) for w in range(VOCAB_SIZE))
    assert total == pytest.approx(1.0, abs=1e-9)


def test_probability_sums_to_one_for_unseen_bigram_context():
    model = _fit([[A, B, C]])
    context = (D, D)  # both levels miss → falls through to Laplace unigram
    total = sum(math.exp(model.log_prob(w, context)) for w in range(VOCAB_SIZE))
    assert total == pytest.approx(1.0, abs=1e-9)


def test_backoff_to_bigram_when_trigram_context_unseen():
    model = _fit([[A, B, C, A, B, C]])
    # (D, B) was never seen as a trigram context, but B is a valid bigram context
    # (followed by C). So log_prob(C, (D, B)) should equal log_prob_bigram(C | B).
    log_p_via_trigram = model.log_prob(C, (D, B))
    log_p_via_bigram = math.log(model._prob_bigram(C, B))
    assert log_p_via_trigram == pytest.approx(log_p_via_bigram, abs=1e-12)


def test_backoff_to_unigram_when_bigram_context_unseen():
    model = _fit([[A, B, C]])
    # D was never observed as a bigram-left-context → falls to Laplace unigram.
    log_p = model.log_prob(C, (D, D))
    expected = math.log((1 + 1) / (4 + VOCAB_SIZE))  # C count = 1
    assert log_p == pytest.approx(expected, abs=1e-12)


def test_seen_trigram_uses_discounted_count():
    # Build a context with multiple distinct continuations to make the math meaningful.
    # Sequence: A,B,C,A,B,D → trigram (A,B) sees C once and D once. Total = 2.
    model = _fit([[A, B, C, A, B, D]], discount=0.5)
    p_c = math.exp(model.log_prob(C, (A, B)))
    expected = (1 - 0.5) / 2
    assert p_c == pytest.approx(expected, abs=1e-12)


def test_predict_topk_returns_legal_moves_only():
    model = _fit([[A, B, C, A, B, D]])
    legal = [A, B, C]
    result = model.predict_topk((A, B), k=2, legal_token_ids=legal)
    assert len(result) == 2
    assert all(tok in legal for tok, _ in result)


def test_predict_topk_excludes_specials_even_without_mask():
    model = _fit([[A, B, C]])
    result = model.predict_topk((A, B), k=VOCAB_SIZE)
    returned_ids = {tok for tok, _ in result}
    assert PAD not in returned_ids
    assert START not in returned_ids
    assert END not in returned_ids


def test_predict_topk_orders_by_descending_log_prob():
    model = _fit([[A, B, C, A, B, C, A, B, D]])  # C twice, D once after (A,B)
    result = model.predict_topk((A, B), k=4, legal_token_ids=[A, B, C, D])
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)
    # C should outrank D since it has more counts.
    rank = {tok: i for i, (tok, _) in enumerate(result)}
    assert rank[C] < rank[D]


def test_save_load_roundtrip(tmp_path):
    model = _fit([[A, B, C, A, B, D], [B, C, A]])
    path = tmp_path / "ngram.pkl.gz"
    model.save(path)
    loaded = TrigramKatz.load(path)

    assert loaded.vocab_size == model.vocab_size
    assert loaded.discount == model.discount
    assert loaded.unigram_counts == model.unigram_counts
    assert loaded.bigram_counts == model.bigram_counts
    assert loaded.trigram_counts == model.trigram_counts

    # Predictions identical for a sampled context.
    ctx = (A, B)
    for w in range(VOCAB_SIZE):
        assert loaded.log_prob(w, ctx) == pytest.approx(model.log_prob(w, ctx), abs=1e-12)


def test_perplexity_is_finite_and_positive():
    model = _fit([[A, B, C, A, B, D], [B, C, A]])
    pp = model.perplexity([[A, B, C], [B, A]])
    assert math.isfinite(pp)
    assert pp > 0


def test_invalid_discount_rejected():
    with pytest.raises(ValueError):
        _fit([[A, B, C]], discount=0.0)
    with pytest.raises(ValueError):
        _fit([[A, B, C]], discount=1.5)
