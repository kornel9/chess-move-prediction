"""Tests for src.training.evaluate (shared eval harness)."""
from __future__ import annotations

import math

import pytest
import torch

from src.data.vocab import Vocab
from src.models.lstm import MoveLSTM
from src.models.ngram import TrigramKatz
from src.training.evaluate import (
    LSTMPredictor,
    NgramPredictor,
    PositionScore,
    _aggregate,
    evaluate,
    phase_for_ply,
)


def test_phase_boundaries():
    # opening 0-19, early_mid 20-39, late_mid 40-79, endgame 80+.
    assert phase_for_ply(0) == "opening"
    assert phase_for_ply(19) == "opening"
    assert phase_for_ply(20) == "early_mid"
    assert phase_for_ply(39) == "early_mid"
    assert phase_for_ply(40) == "late_mid"
    assert phase_for_ply(79) == "late_mid"
    assert phase_for_ply(80) == "endgame"
    assert phase_for_ply(500) == "endgame"


def test_aggregate_topk_and_perplexity():
    # Four positions: target hits top-1, top-3, top-5, then a complete miss.
    positions = [
        PositionScore(0, 10, math.log(0.5),  [10, 20, 30, 40, 50]),
        PositionScore(1, 11, math.log(0.25), [20, 30, 11, 40, 50]),
        PositionScore(2, 12, math.log(0.10), [20, 30, 40, 50, 12]),
        PositionScore(3, 13, math.log(0.01), [20, 30, 40, 50, 60]),
    ]
    n, t1, t3, t5, pp = _aggregate(positions)
    assert n == 4
    assert t1 == pytest.approx(0.25)
    assert t3 == pytest.approx(0.50)
    assert t5 == pytest.approx(0.75)
    expected_pp = math.exp(
        -(math.log(0.5) + math.log(0.25) + math.log(0.10) + math.log(0.01)) / 4
    )
    assert pp == pytest.approx(expected_pp)


def test_aggregate_empty_returns_nan_perplexity():
    n, t1, t3, t5, pp = _aggregate([])
    assert (n, t1, t3, t5) == (0, 0.0, 0.0, 0.0)
    assert math.isnan(pp)


def test_aggregate_handles_empty_topk():
    # If a position has no legal candidates, top-K columns should ignore it
    # rather than counting a hit.
    ps = PositionScore(0, 10, math.log(0.5), [])
    n, t1, t3, t5, _ = _aggregate([ps])
    assert n == 1
    assert t1 == 0.0 and t3 == 0.0 and t5 == 0.0


def test_evaluate_buckets_into_phases_correctly():
    """A fake predictor places one position into each phase by ply index."""

    class FakePredictor:
        def score_game(self, moves, k):
            return [
                PositionScore(0,   42, math.log(0.5), [42]),  # opening
                PositionScore(25,  42, math.log(0.5), [42]),  # early_mid
                PositionScore(50,  42, math.log(0.5), [42]),  # late_mid
                PositionScore(100, 42, math.log(0.5), [42]),  # endgame
            ]

    result = evaluate(FakePredictor(), [["dummy"]], k=5)
    assert result.n_games == 1
    assert result.n_positions == 4
    assert result.top1 == 1.0
    for name in ("opening", "early_mid", "late_mid", "endgame"):
        assert result.by_phase[name].n_positions == 1
        assert result.by_phase[name].top1 == 1.0


def _tiny_ngram(games: list[list[str]]) -> tuple[Vocab, TrigramKatz]:
    vocab = Vocab.build_from_games(games)
    encoded = [[vocab.encode(m) for m in g] for g in games]
    model = TrigramKatz.fit(
        encoded_games=encoded,
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        start_id=vocab.start_id,
        end_id=vocab.end_id,
    )
    return vocab, model


def test_ngram_predictor_yields_one_score_per_ply():
    vocab, model = _tiny_ngram([["e2e4", "e7e5", "g1f3"]])
    pred = NgramPredictor(model, vocab)
    moves = ["e2e4", "e7e5", "g1f3"]
    scores = pred.score_game(moves, k=5)

    assert len(scores) == len(moves)
    for ply, ps in enumerate(scores):
        assert ps.ply_index == ply
        assert ps.target_token_id == vocab.encode(moves[ply])
        assert math.isfinite(ps.target_logprob) and ps.target_logprob <= 0.0
        # Specials must never appear in top-K (legal-move masking + special exclusion).
        assert vocab.pad_id not in ps.topk_token_ids
        assert vocab.start_id not in ps.topk_token_ids
        assert vocab.end_id not in ps.topk_token_ids


def test_evaluate_perplexity_matches_manual_ngram_logprobs():
    """evaluate() perplexity over plays should equal exp(- mean log P) using
    TrigramKatz.log_prob with the same per-ply contexts. This pins down the
    contract between the predictor and the harness."""
    vocab, model = _tiny_ngram([["e2e4", "e7e5", "g1f3"]])
    test_game = ["e2e4", "e7e5", "g1f3"]
    target_ids = [vocab.encode(m) for m in test_game]
    log_probs = [
        model.log_prob(target_ids[0], (model.start_id, model.start_id)),
        model.log_prob(target_ids[1], (model.start_id, target_ids[0])),
        model.log_prob(target_ids[2], (target_ids[0], target_ids[1])),
    ]
    expected_pp = math.exp(-sum(log_probs) / len(log_probs))

    result = evaluate(NgramPredictor(model, vocab), [test_game], k=5)
    assert result.perplexity == pytest.approx(expected_pp, rel=1e-9)


def test_lstm_predictor_smoke():
    torch.manual_seed(0)
    games = [["e2e4", "e7e5", "g1f3", "b8c6"]]
    vocab = Vocab.build_from_games(games)
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
    pred = LSTMPredictor(model, vocab, device=torch.device("cpu"))

    moves = games[0]
    scores = pred.score_game(moves, k=5)

    assert len(scores) == len(moves)
    for ply, ps in enumerate(scores):
        assert ps.ply_index == ply
        assert ps.target_token_id == vocab.encode(moves[ply])
        assert math.isfinite(ps.target_logprob) and ps.target_logprob <= 0.0
        assert vocab.pad_id not in ps.topk_token_ids
        assert vocab.start_id not in ps.topk_token_ids
        assert vocab.end_id not in ps.topk_token_ids


def test_lstm_predictor_target_logprob_matches_log_softmax():
    """The harness' target_logprob must equal the model's log_softmax at the
    target id, i.e. exactly the value that drives perplexity."""
    torch.manual_seed(0)
    games = [["e2e4", "e7e5", "g1f3"]]
    vocab = Vocab.build_from_games(games)
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
    model.eval()

    moves = games[0]
    target_ids = [vocab.encode(m) for m in moves]
    inputs = torch.tensor([vocab.start_id, *target_ids], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits = model(inputs)[0]
        log_probs = torch.log_softmax(logits[:-1], dim=-1)

    pred = LSTMPredictor(model, vocab, device=torch.device("cpu"))
    scores = pred.score_game(moves, k=5)
    for ply, ps in enumerate(scores):
        expected = float(log_probs[ply, target_ids[ply]].item())
        assert ps.target_logprob == pytest.approx(expected, abs=1e-6)


def test_lstm_predictor_does_not_leave_model_in_eval():
    torch.manual_seed(0)
    vocab = Vocab.build_from_games([["e2e4", "e7e5"]])
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
    model.train()
    LSTMPredictor(model, vocab, device=torch.device("cpu")).score_game(
        ["e2e4", "e7e5"], k=3
    )
    assert model.training is True


def test_evaluate_topk_monotonic_top1_le_top3_le_top5():
    vocab, model = _tiny_ngram([["e2e4", "e7e5", "g1f3", "b8c6"]])
    result = evaluate(
        NgramPredictor(model, vocab),
        [["e2e4", "e7e5", "g1f3", "b8c6"]],
        k=5,
    )
    assert result.top1 <= result.top3 <= result.top5
