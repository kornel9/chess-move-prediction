"""PyTorch ``Dataset`` classes for chess next-move prediction."""
from __future__ import annotations

import random
from typing import Iterator, Sequence

import chess
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from src.data.board_encoder import encode_board
from src.data.vocab import Vocab

DEFAULT_HISTORY_LEN: int = 64


class ChessMoveDataset(Dataset):
    """One sample per ply across all games.

    Each sample is a triple ``(move_history, board, target_move_id)``:
        - ``move_history``: ``LongTensor`` of shape ``(history_len,)`` containing
          the last ``history_len`` token ids of ``[<START>, move_0, ..., move_{i-1}]``,
          left-padded with ``<PAD>`` if shorter.
        - ``board``: ``FloatTensor`` of shape ``(18, 8, 8)`` from
          :func:`src.data.board_encoder.encode_board`, computed at the position
          before move ``i`` is played.
        - ``target_move_id``: scalar ``LongTensor`` with the vocab id of move ``i``.
    """

    def __init__(
        self,
        games: Sequence[Sequence[str]],
        vocab: Vocab,
        history_len: int = DEFAULT_HISTORY_LEN,
    ) -> None:
        if history_len <= 0:
            raise ValueError(f"history_len must be positive, got {history_len}")
        self.games: list[list[str]] = [list(g) for g in games]
        self.vocab = vocab
        self.history_len = history_len
        self.index: list[tuple[int, int]] = [
            (g_idx, p_idx)
            for g_idx, game in enumerate(self.games)
            for p_idx in range(len(game))
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        game_idx, ply_idx = self.index[idx]
        game = self.games[game_idx]

        history_tokens = [self.vocab.start_id]
        history_tokens.extend(self.vocab.encode(m) for m in game[:ply_idx])
        if len(history_tokens) >= self.history_len:
            history_tokens = history_tokens[-self.history_len:]
        else:
            pad = [self.vocab.pad_id] * (self.history_len - len(history_tokens))
            history_tokens = pad + history_tokens

        board = chess.Board()
        for move in game[:ply_idx]:
            board.push_uci(move)

        history_tensor = torch.tensor(history_tokens, dtype=torch.long)
        board_tensor = torch.from_numpy(encode_board(board))
        target_id = torch.tensor(self.vocab.encode(game[ply_idx]), dtype=torch.long)

        return history_tensor, board_tensor, target_id


class ChessMoveSequenceDataset(Dataset):
    """One sample per game for sequence-level LSTM training.

    Each sample is a pair ``(input_ids, target_ids)`` of equal length ``N+1``,
    where ``N`` is the number of plies in the game:

        - ``input_ids``  = ``[<START>, m_0, m_1, ..., m_{N-1}]``
        - ``target_ids`` = ``[m_0,    m_1, ..., m_{N-1}, <END>]``

    Sequences are *not* padded inside the dataset; use :func:`pad_collate` as
    the ``DataLoader`` ``collate_fn`` to right-pad a batch to its longest
    sequence with ``<PAD>``. The training loss should use
    ``CrossEntropyLoss(ignore_index=pad_id)`` so padded positions contribute
    nothing.
    """

    def __init__(self, games: Sequence[Sequence[str]], vocab: Vocab) -> None:
        self.games: list[list[str]] = [list(g) for g in games]
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.games)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        game = self.games[idx]
        encoded = [self.vocab.encode(m) for m in game]
        input_ids = torch.tensor([self.vocab.start_id, *encoded], dtype=torch.long)
        target_ids = torch.tensor([*encoded, self.vocab.end_id], dtype=torch.long)
        return input_ids, target_ids


def pad_collate(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a batch of ``(input_ids, target_ids)`` pairs to the batch's
    longest sequence using ``pad_id``."""
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(list(inputs), batch_first=True, padding_value=pad_id)
    targets_padded = pad_sequence(list(targets), batch_first=True, padding_value=pad_id)
    return inputs_padded, targets_padded


class ChessMovePerPlyIterable(IterableDataset):
    """Streaming per-ply dataset that builds boards incrementally per game.

    Yields the same ``(history, board, target_id)`` triples as
    :class:`ChessMoveDataset`, but processes each game *once* — pushing one
    move at a time and emitting a sample for every ply — instead of replaying
    the entire history from move 0 on every access. This collapses the total
    number of ``push_uci`` calls from ``O(plies^2)`` to ``O(plies)``, which
    matters at training scale (~25 M positions across 300 k games would
    otherwise take hours just to reconstruct boards).

    Within an epoch, games are shuffled (deterministically per ``seed`` plus
    worker id and epoch); plies inside a game are emitted in chronological
    order. That partial randomness is sufficient for SGD on this scale and
    is the standard tradeoff for streaming datasets. Set ``shuffle=False``
    for deterministic ordering (useful in tests).

    The ``__len__`` returns the total number of plies across all games so
    DataLoader progress and ``ReduceLROnPlateau``-style schedulers can size
    epochs correctly.
    """

    def __init__(
        self,
        games: Sequence[Sequence[str]],
        vocab: Vocab,
        history_len: int = DEFAULT_HISTORY_LEN,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if history_len <= 0:
            raise ValueError(f"history_len must be positive, got {history_len}")
        self.games: list[list[str]] = [list(g) for g in games]
        self.vocab = vocab
        self.history_len = history_len
        self.shuffle = shuffle
        self.seed = seed
        self._n_positions = sum(len(g) for g in self.games)
        # Bumped by the trainer between epochs so reshuffles differ run-to-run.
        self.epoch = 0

    def __len__(self) -> int:
        return self._n_positions

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch index used to seed per-epoch shuffling."""
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        if worker is None:
            game_indices = list(range(len(self.games)))
            worker_id = 0
        else:
            # Each worker gets a strided slice of the games list.
            game_indices = list(range(worker.id, len(self.games), worker.num_workers))
            worker_id = worker.id

        if self.shuffle:
            rng = random.Random(self.seed + 1009 * self.epoch + 31 * worker_id)
            rng.shuffle(game_indices)

        pad_id = self.vocab.pad_id
        start_id = self.vocab.start_id
        history_len = self.history_len

        for game_idx in game_indices:
            game = self.games[game_idx]
            board = chess.Board()
            history: list[int] = [start_id]
            for uci in game:
                # Pad/truncate the running history to history_len.
                if len(history) >= history_len:
                    h = history[-history_len:]
                else:
                    h = [pad_id] * (history_len - len(history)) + history

                target_id = self.vocab.encode(uci)

                yield (
                    torch.tensor(h, dtype=torch.long),
                    torch.from_numpy(encode_board(board)),
                    torch.tensor(target_id, dtype=torch.long),
                )

                # Advance the running state in place.
                board.push_uci(uci)
                history.append(target_id)
