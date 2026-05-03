"""Tests for src.data.pgn_parser against synthetic PGN fixtures."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import chess
import chess.pgn
import zstandard as zstd

from src.data.pgn_parser import iter_games


def _knight_shuffle(plies: int) -> list[str]:
    """Generate ``plies`` always-legal SAN moves by shuffling both knights."""
    cycle = ["Nf3", "Nf6", "Ng1", "Ng8"]
    return [cycle[i % 4] for i in range(plies)]


def _build_game(moves_san: list[str], headers: dict[str, str]) -> chess.pgn.Game:
    game = chess.pgn.Game()
    for k, v in headers.items():
        game.headers[k] = v
    board = chess.Board()
    node: chess.pgn.GameNode = game
    for san in moves_san:
        move = board.parse_san(san)
        node = node.add_variation(move)
        board.push(move)
    return game


def _write_pgn(games: Iterable[chess.pgn.Game], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for g in games:
            print(g, file=f, end="\n\n")


def _write_pgn_zst(games: Iterable[chess.pgn.Game], path: Path) -> None:
    cctx = zstd.ZstdCompressor()
    with path.open("wb") as raw, cctx.stream_writer(raw) as compressor:
        for g in games:
            compressor.write((str(g) + "\n\n").encode("utf-8"))


def _sample_games() -> list[chess.pgn.Game]:
    long_moves = _knight_shuffle(50)
    short_moves = _knight_shuffle(30)
    return [
        _build_game(long_moves, {
            "Site": "https://lichess.org/passrapid",
            "TimeControl": "600+5", "WhiteElo": "1800", "BlackElo": "1700",
            "Termination": "Normal",
        }),
        _build_game(long_moves, {
            "Site": "https://lichess.org/passclass",
            "TimeControl": "1800+30", "WhiteElo": "2000", "BlackElo": "1900",
            "Termination": "Normal",
        }),
        _build_game(long_moves, {
            "Site": "https://lichess.org/failbullet",
            "TimeControl": "60+0", "WhiteElo": "1800", "BlackElo": "1700",
            "Termination": "Normal",
        }),
        _build_game(long_moves, {
            "Site": "https://lichess.org/failwelo",
            "TimeControl": "600+5", "WhiteElo": "1400", "BlackElo": "1700",
            "Termination": "Normal",
        }),
        _build_game(long_moves, {
            "Site": "https://lichess.org/failbelo",
            "TimeControl": "600+5", "WhiteElo": "1800", "BlackElo": "1499",
            "Termination": "Normal",
        }),
        _build_game(long_moves, {
            "Site": "https://lichess.org/failterm",
            "TimeControl": "600+5", "WhiteElo": "1800", "BlackElo": "1700",
            "Termination": "Time forfeit",
        }),
        _build_game(short_moves, {
            "Site": "https://lichess.org/failshort",
            "TimeControl": "600+5", "WhiteElo": "1800", "BlackElo": "1700",
            "Termination": "Normal",
        }),
        _build_game(long_moves, {
            "Site": "https://lichess.org/failnotc",
            "TimeControl": "-", "WhiteElo": "1800", "BlackElo": "1700",
            "Termination": "Normal",
        }),
    ]


PASSING_IDS = ["passrapid", "passclass"]


def test_iter_games_filters_plain_pgn(tmp_path):
    path = tmp_path / "sample.pgn"
    _write_pgn(_sample_games(), path)

    results = list(iter_games(path))
    assert [gid for gid, _ in results] == PASSING_IDS
    assert all(len(moves) == 50 for _, moves in results)


def test_iter_games_filters_zst_pgn(tmp_path):
    path = tmp_path / "sample.pgn.zst"
    _write_pgn_zst(_sample_games(), path)

    results = list(iter_games(path))
    assert [gid for gid, _ in results] == PASSING_IDS


def test_iter_games_yields_uci_strings(tmp_path):
    path = tmp_path / "sample.pgn"
    _write_pgn(_sample_games(), path)

    _, moves = next(iter(iter_games(path)))
    assert moves[0] == "g1f3"   # first knight shuffle
    for m in moves:
        chess.Move.from_uci(m)   # parses as UCI


def test_iter_games_drops_when_elo_missing(tmp_path):
    long_moves = _knight_shuffle(50)
    games = [
        _build_game(long_moves, {
            "Site": "https://lichess.org/keepme",
            "TimeControl": "600+5", "WhiteElo": "1800", "BlackElo": "1700",
            "Termination": "Normal",
        }),
        _build_game(long_moves, {
            "Site": "https://lichess.org/dropme",
            "TimeControl": "600+5", "WhiteElo": "?", "BlackElo": "1700",
            "Termination": "Normal",
        }),
    ]
    path = tmp_path / "sample.pgn"
    _write_pgn(games, path)
    assert [gid for gid, _ in iter_games(path)] == ["keepme"]


def test_iter_games_falls_back_to_counter_for_unknown_site(tmp_path):
    long_moves = _knight_shuffle(50)
    game = _build_game(long_moves, {
        "Site": "?",
        "TimeControl": "600+5", "WhiteElo": "1800", "BlackElo": "1700",
        "Termination": "Normal",
    })
    path = tmp_path / "sample.pgn"
    _write_pgn([game], path)

    results = list(iter_games(path))
    assert len(results) == 1
    assert results[0][0].startswith("game_")
