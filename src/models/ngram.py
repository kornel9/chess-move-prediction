"""Trigram language model with Katz backoff (absolute discount) over UCI move ids."""
from __future__ import annotations

import gzip
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


_NEG_INF = float("-inf")


@dataclass
class TrigramKatz:
    """Trigram language model with Katz backoff and fixed absolute discount.

    Counts are stored as plain dicts so the model is pure Python and pickles
    cleanly. Backoff distributes the per-context reserved mass
    ``d * |seen|/c(context)`` over unseen continuations in proportion to the
    lower-order distribution. The recursion bottoms out at a Laplace-smoothed
    unigram so no probability is ever zero.
    """

    vocab_size: int
    discount: float
    pad_id: int
    start_id: int
    end_id: int
    unigram_counts: dict[int, int]
    unigram_total: int
    bigram_counts: dict[int, dict[int, int]]
    bigram_context_totals: dict[int, int]
    trigram_counts: dict[tuple[int, int], dict[int, int]]
    trigram_context_totals: dict[tuple[int, int], int]
    _alpha2_cache: dict[int, float] = field(default_factory=dict, repr=False)
    _alpha3_cache: dict[tuple[int, int], float] = field(default_factory=dict, repr=False)

    @classmethod
    def fit(
        cls,
        encoded_games: Iterable[list[int]],
        vocab_size: int,
        pad_id: int,
        start_id: int,
        end_id: int,
        discount: float = 0.5,
    ) -> "TrigramKatz":
        """Fit the trigram model on encoded games.

        ``encoded_games`` is an iterable of UCI-encoded move-id lists (the raw
        moves, without start/end markers). This method prepends two
        ``<START>`` tokens and appends one ``<END>`` token to each game so
        trigram contexts are well-defined for the first two predictions and
        the model learns to predict end-of-game.
        """
        if not (0.0 < discount < 1.0):
            raise ValueError(f"discount must be in (0, 1), got {discount}")

        unigram: dict[int, int] = defaultdict(int)
        bigram: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        trigram: dict[tuple[int, int], dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for moves in encoded_games:
            seq = [start_id, start_id, *moves, end_id]
            for i in range(2, len(seq)):
                a, b, w = seq[i - 2], seq[i - 1], seq[i]
                unigram[w] += 1
                bigram[b][w] += 1
                trigram[(a, b)][w] += 1

        bigram_plain = {b: dict(d) for b, d in bigram.items()}
        trigram_plain = {ctx: dict(d) for ctx, d in trigram.items()}
        bigram_totals = {b: sum(d.values()) for b, d in bigram_plain.items()}
        trigram_totals = {ctx: sum(d.values()) for ctx, d in trigram_plain.items()}
        unigram_plain = dict(unigram)
        unigram_total = sum(unigram_plain.values())

        return cls(
            vocab_size=vocab_size,
            discount=discount,
            pad_id=pad_id,
            start_id=start_id,
            end_id=end_id,
            unigram_counts=unigram_plain,
            unigram_total=unigram_total,
            bigram_counts=bigram_plain,
            bigram_context_totals=bigram_totals,
            trigram_counts=trigram_plain,
            trigram_context_totals=trigram_totals,
        )

    def _prob_unigram(self, w: int) -> float:
        # Laplace-smoothed unigram so the distribution covers every token id.
        return (self.unigram_counts.get(w, 0) + 1) / (self.unigram_total + self.vocab_size)

    def _prob_bigram(self, w: int, b: int) -> float:
        seen = self.bigram_counts.get(b)
        if not seen:
            return self._prob_unigram(w)
        total = self.bigram_context_totals[b]
        c = seen.get(w, 0)
        if c > 0:
            return (c - self.discount) / total
        alpha = self._alpha2_cache.get(b)
        if alpha is None:
            seen_unigram_mass = sum(self._prob_unigram(x) for x in seen)
            backoff_unseen = 1.0 - seen_unigram_mass
            reserved = self.discount * len(seen) / total
            alpha = reserved / backoff_unseen if backoff_unseen > 0 else 0.0
            self._alpha2_cache[b] = alpha
        return alpha * self._prob_unigram(w)

    def _prob_trigram(self, w: int, context: tuple[int, int]) -> float:
        seen = self.trigram_counts.get(context)
        if not seen:
            return self._prob_bigram(w, context[1])
        total = self.trigram_context_totals[context]
        c = seen.get(w, 0)
        if c > 0:
            return (c - self.discount) / total
        alpha = self._alpha3_cache.get(context)
        if alpha is None:
            seen_bigram_mass = sum(self._prob_bigram(x, context[1]) for x in seen)
            backoff_unseen = 1.0 - seen_bigram_mass
            reserved = self.discount * len(seen) / total
            alpha = reserved / backoff_unseen if backoff_unseen > 0 else 0.0
            self._alpha3_cache[context] = alpha
        return alpha * self._prob_bigram(w, context[1])

    def log_prob(self, token_id: int, context: tuple[int, int]) -> float:
        """Return ``log P(token_id | context)`` under Katz backoff."""
        p = self._prob_trigram(token_id, context)
        return math.log(p) if p > 0 else _NEG_INF

    def predict_topk(
        self,
        context: tuple[int, int],
        k: int = 5,
        legal_token_ids: Iterable[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Return the top-``k`` ``(token_id, log_prob)`` pairs for ``context``.

        If ``legal_token_ids`` is given, only those tokens are scored (use this
        to apply legal-move masking at inference). The special tokens
        (``<PAD>``, ``<START>``, ``<END>``) are always excluded from the
        candidate set.
        """
        excluded = {self.pad_id, self.start_id, self.end_id}
        if legal_token_ids is None:
            candidates = [w for w in range(self.vocab_size) if w not in excluded]
        else:
            candidates = [w for w in legal_token_ids if w not in excluded]

        scored = [(w, self.log_prob(w, context)) for w in candidates]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def perplexity(self, encoded_games: Iterable[list[int]]) -> float:
        """Return per-token perplexity on ``encoded_games``.

        Each game is wrapped with two start tokens and one end token to match
        the training contract. Perplexity = exp(- mean log P) over the
        predicted positions (the trailing ``<END>`` is included; the two
        leading ``<START>`` tokens are not predicted).
        """
        total_log_prob = 0.0
        n = 0
        for moves in encoded_games:
            seq = [self.start_id, self.start_id, *moves, self.end_id]
            for i in range(2, len(seq)):
                total_log_prob += self.log_prob(seq[i], (seq[i - 2], seq[i - 1]))
                n += 1
        if n == 0:
            return float("nan")
        return math.exp(-total_log_prob / n)

    def save(self, path: Path | str) -> None:
        """Persist the model to ``path`` as gzip-compressed pickle."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "vocab_size": self.vocab_size,
            "discount": self.discount,
            "pad_id": self.pad_id,
            "start_id": self.start_id,
            "end_id": self.end_id,
            "unigram_counts": self.unigram_counts,
            "unigram_total": self.unigram_total,
            "bigram_counts": self.bigram_counts,
            "bigram_context_totals": self.bigram_context_totals,
            "trigram_counts": self.trigram_counts,
            "trigram_context_totals": self.trigram_context_totals,
        }
        with gzip.open(p, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path | str) -> "TrigramKatz":
        """Load a model previously written by :meth:`save`."""
        with gzip.open(Path(path), "rb") as f:
            state = pickle.load(f)
        return cls(**state)
