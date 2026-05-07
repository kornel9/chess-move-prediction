"""LSTM language model over UCI move-id sequences."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


@dataclass
class MoveLSTMConfig:
    """Architecture + special-token wiring for :class:`MoveLSTM`.

    The defaults match the project spec (2 layers, hidden 512, embedding
    256). ``vocab_size`` and the special-token ids must be supplied by the
    caller from the project ``Vocab``.
    """

    vocab_size: int
    pad_id: int
    start_id: int
    end_id: int
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.2

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "pad_id": self.pad_id,
            "start_id": self.start_id,
            "end_id": self.end_id,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


class MoveLSTM(nn.Module):
    """Embedding → LSTM → linear head over the UCI vocabulary.

    Forward consumes a ``(B, T)`` long tensor of move ids and returns
    ``(B, T, V)`` logits — the prediction at position ``t`` is the model's
    distribution over the move played at position ``t+1`` of the input
    sequence (so loss is computed against ``input[:, 1:]``-style targets).
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        start_id: int,
        end_id: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.config = MoveLSTMConfig(
            vocab_size=vocab_size,
            pad_id=pad_id,
            start_id=start_id,
            end_id=end_id,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        # PyTorch ignores the dropout argument when num_layers == 1; that's fine.
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, vocab_size)

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def pad_id(self) -> int:
        return self.config.pad_id

    @property
    def start_id(self) -> int:
        return self.config.start_id

    @property
    def end_id(self) -> int:
        return self.config.end_id

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Compute logits over the vocab at every position of ``history``.

        ``history`` is a ``(B, T)`` ``LongTensor`` of token ids. Returns a
        ``(B, T, V)`` ``FloatTensor`` of unnormalised logits.
        """
        x = self.embedding(history)
        out, _ = self.lstm(x)
        return self.head(out)

    @torch.no_grad()
    def predict_topk(
        self,
        history: torch.Tensor,
        k: int = 5,
        legal_token_ids: Iterable[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Return the top-``k`` ``(token_id, log_prob)`` pairs for the next move.

        ``history`` is a 1D ``LongTensor`` of token ids representing the moves
        seen so far (typically ``[<START>, m_0, ..., m_{i-1}]``); the
        prediction is conditioned on the *last* time step. Special tokens
        (``<PAD>``, ``<START>``, ``<END>``) are always excluded from the
        candidate set, matching :meth:`TrigramKatz.predict_topk`. If
        ``legal_token_ids`` is given, candidates are restricted to that set.
        """
        if history.dim() != 1:
            raise ValueError(f"history must be 1D, got shape {tuple(history.shape)}")

        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device
            x = history.to(device).unsqueeze(0)
            logits = self.forward(x)[0, -1]
            log_probs = torch.log_softmax(logits, dim=-1)
        finally:
            if was_training:
                self.train()

        excluded = {self.pad_id, self.start_id, self.end_id}
        if legal_token_ids is None:
            candidates = [w for w in range(self.vocab_size) if w not in excluded]
        else:
            candidates = [int(w) for w in legal_token_ids if int(w) not in excluded]

        if not candidates:
            return []

        cand_tensor = torch.tensor(candidates, dtype=torch.long, device=log_probs.device)
        cand_scores = log_probs.index_select(0, cand_tensor)
        top = min(k, cand_tensor.numel())
        values, indices = torch.topk(cand_scores, top)
        return [(int(cand_tensor[i].item()), float(values[j].item())) for j, i in enumerate(indices)]

    def save(self, path: Path | str) -> None:
        """Persist ``state_dict`` and config to ``path`` via :func:`torch.save`."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"config": self.config.to_dict(), "state_dict": self.state_dict()},
            p,
        )

    @classmethod
    def load(cls, path: Path | str, map_location: str | torch.device | None = None) -> "MoveLSTM":
        """Load a model previously written by :meth:`save`."""
        blob = torch.load(Path(path), map_location=map_location, weights_only=False)
        model = cls(**blob["config"])
        model.load_state_dict(blob["state_dict"])
        return model
