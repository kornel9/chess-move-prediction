"""Hybrid CNN + LSTM next-move predictor.

Combines a CNN over the ``(18, 8, 8)`` board tensor with the existing LSTM
over move history. The CNN provides board awareness so the model has
predictive signal in positions where the move history alone is insufficient
(empty/short histories, custom positions, endgames). The two branches'
representations are concatenated and passed through a shared linear head.

Per CLAUDE.md, the input is the per-ply representation defined in
:class:`src.data.dataset.ChessMoveDataset` (and the streaming counterpart
:class:`ChessMovePerPlyIterable`): a ``(history_len,)`` ``LongTensor`` of
token ids (left-padded with ``<PAD>``) plus a ``(18, 8, 8)`` ``FloatTensor``
board tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from src.data.board_encoder import BOARD_SIZE, NUM_CHANNELS


@dataclass
class HybridConfig:
    """Architecture + special-token wiring for :class:`MoveBoardHybrid`."""

    vocab_size: int
    pad_id: int
    start_id: int
    end_id: int
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.2
    cnn_channels: tuple[int, int, int] = (64, 128, 128)
    board_feature_dim: int = 256

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
            "cnn_channels": list(self.cnn_channels),
            "board_feature_dim": self.board_feature_dim,
        }


class MoveBoardHybrid(nn.Module):
    """Hybrid CNN + LSTM next-move predictor.

    Forward signature is ``forward(history, board) -> logits``:
        - ``history``: ``(B, T)`` ``LongTensor`` of token ids (left-padded with
          ``<PAD>``); the prediction is conditioned on the *last* time step.
        - ``board``: ``(B, 18, 8, 8)`` ``FloatTensor``.
        - returns: ``(B, V)`` ``FloatTensor`` of unnormalised logits over the
          UCI vocab.

    The LSTM branch's last-timestep hidden state and the CNN branch's flattened
    + projected board feature vector are concatenated before the shared head
    so both signals contribute jointly to every prediction.
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
        cnn_channels: tuple[int, int, int] = (64, 128, 128),
        board_feature_dim: int = 256,
    ) -> None:
        super().__init__()
        cnn_channels_t = tuple(cnn_channels)
        self.config = HybridConfig(
            vocab_size=vocab_size,
            pad_id=pad_id,
            start_id=start_id,
            end_id=end_id,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            cnn_channels=cnn_channels_t,
            board_feature_dim=board_feature_dim,
        )

        # LSTM branch: same shape as MoveLSTM so weights and intuitions transfer.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # CNN branch over the (18, 8, 8) board tensor. Three same-padding 3x3
        # convs preserve the 8x8 spatial dim through the stack; a small linear
        # then projects the flattened activations down to a board feature vec.
        c1, c2, c3 = cnn_channels_t
        self.cnn = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.board_proj = nn.Linear(c3 * BOARD_SIZE * BOARD_SIZE, board_feature_dim)

        # Shared head over the concatenated representations.
        self.head = nn.Linear(hidden_dim + board_feature_dim, vocab_size)

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

    def forward(self, history: torch.Tensor, board: torch.Tensor) -> torch.Tensor:
        """Compute next-move logits given a batch of (history, board) pairs."""
        # LSTM branch: take the last timestep's hidden state.
        x = self.embedding(history)            # (B, T, E)
        out, _ = self.lstm(x)                  # (B, T, H)
        history_feat = out[:, -1, :]           # (B, H)

        # CNN branch: conv stack -> flatten -> linear projection (with ReLU).
        cnn_out = self.cnn(board)              # (B, c3, 8, 8)
        cnn_flat = cnn_out.flatten(start_dim=1)  # (B, c3 * 64)
        board_feat = torch.relu(self.board_proj(cnn_flat))  # (B, board_feature_dim)

        combined = torch.cat([history_feat, board_feat], dim=1)  # (B, H + board_feat)
        return self.head(combined)             # (B, V)

    @torch.no_grad()
    def predict_topk(
        self,
        history: torch.Tensor,
        board: torch.Tensor,
        k: int = 5,
        legal_token_ids: Iterable[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Return the top-``k`` ``(token_id, log_prob)`` pairs for the next move.

        ``history`` is a 1-D ``LongTensor`` of token ids (typically already
        left-padded to ``history_len``); ``board`` is a ``(18, 8, 8)``
        ``FloatTensor``. Special tokens (``<PAD>``, ``<START>``, ``<END>``)
        are always excluded from the candidate set, matching the n-gram and
        LSTM contracts. If ``legal_token_ids`` is given, candidates are
        restricted to that set.
        """
        if history.dim() != 1:
            raise ValueError(f"history must be 1D, got shape {tuple(history.shape)}")
        if board.dim() != 3:
            raise ValueError(f"board must be 3D, got shape {tuple(board.shape)}")

        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device
            x_h = history.to(device).unsqueeze(0)
            x_b = board.to(device).unsqueeze(0)
            logits = self.forward(x_h, x_b)[0]
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
    def load(
        cls,
        path: Path | str,
        map_location: str | torch.device | None = None,
    ) -> "MoveBoardHybrid":
        """Load a model previously written by :meth:`save`."""
        blob = torch.load(Path(path), map_location=map_location, weights_only=False)
        cfg = dict(blob["config"])
        # cnn_channels was serialised as a list; convert back to tuple for the dataclass.
        if isinstance(cfg.get("cnn_channels"), list):
            cfg["cnn_channels"] = tuple(cfg["cnn_channels"])
        model = cls(**cfg)
        model.load_state_dict(blob["state_dict"])
        return model
