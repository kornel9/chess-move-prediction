"""Shared evaluation harness for n-gram, LSTM, and hybrid models.

Each model is wrapped in a :class:`Predictor` adapter that exposes one method,
``score_game``, returning a per-ply :class:`PositionScore`. The harness then
aggregates those scores into top-1/3/5 accuracy, per-token perplexity, and a
phase-wise breakdown (opening / early mid / late mid / endgame) per CLAUDE.md.

Top-K is computed against legal moves only (legal-move masking). Perplexity is
computed against the model's raw distribution over the full vocab — masking
would inflate probabilities and stop the metric from comparing across models.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import chess
import torch

from src.data.pgn_parser import iter_games
from src.data.splits import split_games
from src.data.vocab import Vocab
from src.models.lstm import MoveLSTM
from src.models.ngram import TrigramKatz


# Ply indices are 0-based. CLAUDE.md gives phase boundaries in full-move
# numbers; each full move is two plies, so the bounds below are
# inclusive-lower / exclusive-upper ply indices.
PHASES: tuple[tuple[str, int, int], ...] = (
    ("opening", 0, 20),
    ("early_mid", 20, 40),
    ("late_mid", 40, 80),
    ("endgame", 80, 10**9),
)


def phase_for_ply(ply_index: int) -> str:
    """Return the phase name (``"opening"``, ``"early_mid"``, ``"late_mid"``,
    ``"endgame"``) for the given 0-based ply index."""
    for name, lo, hi in PHASES:
        if lo <= ply_index < hi:
            return name
    return "endgame"


@dataclass
class PositionScore:
    """Per-ply output of a :class:`Predictor`.

    - ``ply_index``: 0-based index of the played move within the game.
    - ``target_token_id``: vocab id of the move actually played.
    - ``target_logprob``: ``log P(target | history)`` under the model's full
      distribution (no legal-move masking).
    - ``topk_token_ids``: top-``k`` predicted token ids restricted to legal
      moves at this position, special tokens excluded, sorted by descending
      model score.
    """

    ply_index: int
    target_token_id: int
    target_logprob: float
    topk_token_ids: list[int]


class Predictor(Protocol):
    """Adapter that lets the eval harness treat each model class uniformly."""

    def score_game(self, moves: list[str], k: int) -> list[PositionScore]:
        """Return one :class:`PositionScore` per played move in ``moves``."""
        ...


class NgramPredictor:
    """Adapt :class:`TrigramKatz` to the :class:`Predictor` protocol."""

    def __init__(self, model: TrigramKatz, vocab: Vocab) -> None:
        self.model = model
        self.vocab = vocab

    def score_game(self, moves: list[str], k: int) -> list[PositionScore]:
        out: list[PositionScore] = []
        board = chess.Board()
        # Trigram contexts begin with two <START> tokens (matches TrigramKatz.fit).
        prev_a = self.model.start_id
        prev_b = self.model.start_id
        for ply, uci in enumerate(moves):
            target_id = self.vocab.encode(uci)
            legal_ids = sorted({self.vocab.encode(m.uci()) for m in board.legal_moves})
            context = (prev_a, prev_b)
            topk = self.model.predict_topk(context, k=k, legal_token_ids=legal_ids)
            target_logprob = self.model.log_prob(target_id, context)
            out.append(
                PositionScore(
                    ply_index=ply,
                    target_token_id=target_id,
                    target_logprob=target_logprob,
                    topk_token_ids=[tok for tok, _ in topk],
                )
            )
            board.push_uci(uci)
            prev_a, prev_b = prev_b, target_id
        return out


class LSTMPredictor:
    """Adapt :class:`MoveLSTM` to the :class:`Predictor` protocol.

    Runs one forward pass over the whole game (input
    ``[<START>, m_0, ..., m_{N-1}]``) and reuses the per-position logits for
    every score, so cost scales with sequence length rather than N forward
    passes. Legal moves are still recomputed per ply against a python-chess
    ``Board`` since the model itself has no legality knowledge.
    """

    def __init__(
        self,
        model: MoveLSTM,
        vocab: Vocab,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.device = device or next(model.parameters()).device
        self._specials = {model.pad_id, model.start_id, model.end_id}

    def score_game(self, moves: list[str], k: int) -> list[PositionScore]:
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                return self._score_game(moves, k)
        finally:
            if was_training:
                self.model.train()

    def _score_game(self, moves: list[str], k: int) -> list[PositionScore]:
        target_ids = [self.vocab.encode(m) for m in moves]
        if not target_ids:
            return []

        # Position t of the input predicts move t. The trailing <END> position
        # exists in training but isn't scored here — we only care about plays.
        input_ids = torch.tensor(
            [self.vocab.start_id, *target_ids],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        logits = self.model(input_ids)[0]                          # (N+1, V)
        log_probs = torch.log_softmax(logits[:-1], dim=-1).cpu()    # (N, V)

        out: list[PositionScore] = []
        board = chess.Board()
        for ply, (uci, target_id) in enumerate(zip(moves, target_ids)):
            row = log_probs[ply]
            target_logprob = float(row[target_id].item())

            legal_ids = sorted({self.vocab.encode(m.uci()) for m in board.legal_moves})
            candidates = [w for w in legal_ids if w not in self._specials]
            if candidates:
                cand = torch.tensor(candidates, dtype=torch.long)
                cand_scores = row.index_select(0, cand)
                top = min(k, cand.numel())
                _, indices = torch.topk(cand_scores, top)
                topk_ids = [int(cand[i].item()) for i in indices]
            else:
                topk_ids = []

            out.append(
                PositionScore(
                    ply_index=ply,
                    target_token_id=target_id,
                    target_logprob=target_logprob,
                    topk_token_ids=topk_ids,
                )
            )
            board.push_uci(uci)
        return out


@dataclass
class PhaseMetrics:
    """Aggregated metrics for one phase bucket."""

    n_positions: int
    top1: float
    top3: float
    top5: float
    perplexity: float


@dataclass
class EvalResult:
    """Overall + per-phase metrics for one model on one set of games."""

    n_games: int
    n_positions: int
    top1: float
    top3: float
    top5: float
    perplexity: float
    by_phase: dict[str, PhaseMetrics]


def _aggregate(positions: Sequence[PositionScore]) -> tuple[int, float, float, float, float]:
    """Return ``(n, top1, top3, top5, perplexity)`` for a flat list of positions.

    Empty input returns zero counts and ``nan`` perplexity. ``top-K`` is the
    fraction of positions where the played move is among the predictor's top
    ``K`` candidates (which were already restricted to legal moves).
    """
    n = len(positions)
    if n == 0:
        return 0, 0.0, 0.0, 0.0, float("nan")

    top1 = top3 = top5 = 0
    total_log_prob = 0.0
    for ps in positions:
        if ps.topk_token_ids:
            top1 += int(ps.target_token_id == ps.topk_token_ids[0])
            top3 += int(ps.target_token_id in ps.topk_token_ids[:3])
            top5 += int(ps.target_token_id in ps.topk_token_ids[:5])
        total_log_prob += ps.target_logprob

    if not math.isfinite(total_log_prob):
        perplexity = float("inf")
    else:
        perplexity = math.exp(-total_log_prob / n)
    return n, top1 / n, top3 / n, top5 / n, perplexity


def evaluate(
    predictor: Predictor,
    games: Sequence[Sequence[str]],
    k: int = 5,
) -> EvalResult:
    """Score ``games`` with ``predictor`` and aggregate into an :class:`EvalResult`.

    ``k`` must be ≥ 5 to populate the top-5 column (the predictor returns at
    most ``k`` candidates; lower ``k`` truncates everything).
    """
    all_positions: list[PositionScore] = []
    for moves in games:
        all_positions.extend(predictor.score_game(list(moves), k=k))

    n, top1, top3, top5, pp = _aggregate(all_positions)

    buckets: dict[str, list[PositionScore]] = {name: [] for name, _, _ in PHASES}
    for ps in all_positions:
        buckets[phase_for_ply(ps.ply_index)].append(ps)
    by_phase: dict[str, PhaseMetrics] = {}
    for name, _, _ in PHASES:
        pn, p1, p3, p5, ppx = _aggregate(buckets[name])
        by_phase[name] = PhaseMetrics(n_positions=pn, top1=p1, top3=p3, top5=p5, perplexity=ppx)

    return EvalResult(
        n_games=len(games),
        n_positions=n,
        top1=top1,
        top3=top3,
        top5=top5,
        perplexity=pp,
        by_phase=by_phase,
    )


def format_eval_result(result: EvalResult) -> str:
    """Render an :class:`EvalResult` as a fixed-width report."""
    lines = [
        f"Games: {result.n_games}  Positions: {result.n_positions}",
        (
            f"Overall: top1={result.top1:.4f} top3={result.top3:.4f} "
            f"top5={result.top5:.4f} perplexity={result.perplexity:.2f}"
        ),
        "By phase:",
    ]
    for name, _, _ in PHASES:
        pm = result.by_phase[name]
        lines.append(
            f"  {name:>10s}: n={pm.n_positions:>7d} "
            f"top1={pm.top1:.4f} top3={pm.top3:.4f} top5={pm.top5:.4f} "
            f"perplexity={pm.perplexity:.2f}"
        )
    return "\n".join(lines)


def _load_predictor(
    model_type: str,
    model_path: Path,
    vocab: Vocab,
    device: str | None = None,
) -> Predictor:
    if model_type == "ngram":
        return NgramPredictor(TrigramKatz.load(model_path), vocab)
    if model_type == "lstm":
        dev = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = MoveLSTM.load(model_path, map_location=dev).to(dev)
        return LSTMPredictor(model, vocab, device=dev)
    raise ValueError(f"unknown model_type: {model_type!r} (expected 'ngram' or 'lstm')")


def main(
    model_type: str,
    model_path: Path,
    pgn_path: Path,
    vocab_path: Path,
    split: str = "test",
    seed: int = 42,
    k: int = 5,
    device: str | None = None,
) -> EvalResult:
    """Load a model and evaluate it on the requested split of ``pgn_path``."""
    print(f"[evaluate] reading games from {pgn_path}")
    games = [moves for _, moves in iter_games(pgn_path)]
    print(f"[evaluate] kept {len(games)} games after filtering")

    train, val, test = split_games(games, seed=seed)
    splits = {"train": train, "val": val, "test": test}
    if split not in splits:
        raise ValueError(f"unknown split: {split!r}")
    eval_games = splits[split]
    print(f"[evaluate] split={split} size={len(eval_games)}")

    vocab = Vocab.load(vocab_path)
    print(f"[evaluate] loaded vocab from {vocab_path} (size={len(vocab)})")

    predictor = _load_predictor(model_type, model_path, vocab, device=device)
    print(f"[evaluate] loaded {model_type} from {model_path}")

    result = evaluate(predictor, eval_games, k=k)
    print(format_eval_result(result))
    return result


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a chess move-prediction model.")
    p.add_argument("--model-type", choices=["ngram", "lstm"], required=True)
    p.add_argument("--model", type=Path, required=True, help="Path to the trained model file.")
    p.add_argument("--pgn", type=Path, required=True, help="Path to .pgn or .pgn.zst dump.")
    p.add_argument("--vocab", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=5, help="Top-K cap (must be >=5 for top-5).")
    p.add_argument("--device", type=str, default=None)
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    if args.k < 5:
        raise SystemExit("--k must be >= 5 to compute top-5 accuracy")
    main(
        model_type=args.model_type,
        model_path=args.model,
        pgn_path=args.pgn,
        vocab_path=args.vocab,
        split=args.split,
        seed=args.seed,
        k=args.k,
        device=args.device,
    )
