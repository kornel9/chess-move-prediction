import chess
import numpy as np

from src.data.board_encoder import BOARD_SIZE, NUM_CHANNELS, encode_board


def test_shape_and_dtype():
    t = encode_board(chess.Board())
    assert t.shape == (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert t.dtype == np.float32


def test_starting_position_piece_totals():
    t = encode_board(chess.Board())
    assert t[0:6].sum() == 16.0
    assert t[6:12].sum() == 16.0


def test_starting_position_pawn_ranks():
    t = encode_board(chess.Board())
    # White pawns occupy rank index 1; black pawns occupy rank index 6.
    assert t[0, 1, :].sum() == 8.0
    assert t[0, 0, :].sum() == 0.0
    assert t[6, 6, :].sum() == 8.0
    assert t[6, 7, :].sum() == 0.0


def test_starting_position_king_squares():
    t = encode_board(chess.Board())
    assert t[5, 0, 4] == 1.0   # white king on e1
    assert t[11, 7, 4] == 1.0  # black king on e8


def test_starting_position_turn_plane_is_full():
    t = encode_board(chess.Board())
    assert t[12].sum() == BOARD_SIZE * BOARD_SIZE


def test_starting_position_all_castling_rights_set():
    t = encode_board(chess.Board())
    for channel in (13, 14, 15, 16):
        assert t[channel].sum() == BOARD_SIZE * BOARD_SIZE


def test_starting_position_no_en_passant():
    t = encode_board(chess.Board())
    assert t[17].sum() == 0.0


def test_starting_position_total_sum():
    t = encode_board(chess.Board())
    # 32 pieces + 64 (turn=white) + 4*64 (all castling rights) + 0 EP
    expected = 32 + 64 + 4 * 64
    assert t.sum() == expected


def test_after_e4_pawn_moved_turn_flipped_ep_set():
    board = chess.Board()
    board.push_san("e4")
    t = encode_board(board)

    # White pawn moved e2 -> e4
    assert t[0, 1, 4] == 0.0  # e2 empty
    assert t[0, 3, 4] == 1.0  # e4 occupied

    # Turn now belongs to black
    assert t[12].sum() == 0.0

    # En-passant target on e3 (rank 2, file 4)
    assert t[17, 2, 4] == 1.0
    assert t[17].sum() == 1.0


def test_white_castling_rights_clear_after_king_move():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Ke2")
    t = encode_board(board)

    assert t[13].sum() == 0.0  # white kingside cleared
    assert t[14].sum() == 0.0  # white queenside cleared
    # Black still has full castling rights
    assert t[15].sum() == BOARD_SIZE * BOARD_SIZE
    assert t[16].sum() == BOARD_SIZE * BOARD_SIZE
