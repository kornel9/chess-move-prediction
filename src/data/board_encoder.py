"""Encode a python-chess Board as an (18, 8, 8) float32 tensor."""
from __future__ import annotations

import chess
import numpy as np

NUM_CHANNELS: int = 18
BOARD_SIZE: int = 8


def encode_board(board: chess.Board) -> np.ndarray:
    """Return a ``(18, 8, 8)`` float32 tensor encoding ``board``.

    Channel layout:
        0-5   White pieces (P, N, B, R, Q, K) one-hot per square.
        6-11  Black pieces (p, n, b, r, q, k) one-hot per square.
        12    Side to move: filled with 1.0 if white to move, else 0.0.
        13-16 Castling rights K, Q, k, q: each plane uniformly 0 or 1.
        17    En-passant target square: single 1.0 at ``board.ep_square``,
              else all zeros.

    Rank index 0 corresponds to white's back rank (chess rank 1); file 0 = a-file.
    """
    tensor = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        colour_offset = 0 if piece.color == chess.WHITE else 6
        channel = colour_offset + (piece.piece_type - 1)
        tensor[channel, rank, file] = 1.0

    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        tensor[17, rank, file] = 1.0

    return np.ascontiguousarray(tensor)
