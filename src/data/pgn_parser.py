"""Stream games from Lichess PGN dumps with the project's filtering rules."""
from __future__ import annotations

import io
from pathlib import Path
from typing import IO, Iterator

import chess.pgn
import zstandard as zstd

# Lichess classifies time controls by estimated seconds = base + 40 * increment.
# Rapid = 480..1499s, Classical = 1500s+. We accept either.
RAPID_MIN_SECONDS = 480

MIN_ELO = 1500
MIN_PLIES = 40   # 20 full moves
MAX_PLIES = 240  # 120 full moves


def _open_pgn_stream(path: Path) -> IO[str]:
    """Return a UTF-8 text stream over a .pgn or .pgn.zst file."""
    if path.suffix == ".zst":
        raw = path.open("rb")
        try:
            decompressor = zstd.ZstdDecompressor().stream_reader(raw)
            return io.TextIOWrapper(decompressor, encoding="utf-8")
        except Exception:
            raw.close()
            raise
    return path.open("r", encoding="utf-8")


def _estimated_seconds(time_control: str) -> int | None:
    """Parse a PGN ``TimeControl`` tag (e.g. ``"600+5"``) into estimated seconds.

    Returns ``None`` for unlimited (``-``), unknown (``?``), or multi-stage controls.
    """
    if not time_control or time_control in {"-", "?"}:
        return None
    if "," in time_control:
        return None
    if "+" not in time_control:
        try:
            return int(time_control)
        except ValueError:
            return None
    base_s, inc_s = time_control.split("+", 1)
    try:
        return int(base_s) + 40 * int(inc_s)
    except ValueError:
        return None


def _passes_time_control(headers: chess.pgn.Headers) -> bool:
    estimated = _estimated_seconds(headers.get("TimeControl", ""))
    return estimated is not None and estimated >= RAPID_MIN_SECONDS


def _passes_elo(headers: chess.pgn.Headers) -> bool:
    try:
        white = int(headers.get("WhiteElo", "?"))
        black = int(headers.get("BlackElo", "?"))
    except ValueError:
        return False
    return white >= MIN_ELO and black >= MIN_ELO


def _passes_termination(headers: chess.pgn.Headers) -> bool:
    return headers.get("Termination", "") == "Normal"


def _game_id(headers: chess.pgn.Headers, fallback_index: int) -> str:
    """Extract the Lichess game id from the ``Site`` tag, with a counter fallback."""
    site = headers.get("Site", "")
    if site.startswith("https://lichess.org/"):
        return site.rsplit("/", 1)[-1]
    return f"game_{fallback_index}"


def iter_games(path: Path | str) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(game_id, uci_moves)`` tuples for games passing the project filters.

    Filters (per CLAUDE.md):
        - TimeControl is Rapid or Classical (estimated seconds >= 480).
        - Both ``WhiteElo`` and ``BlackElo`` >= 1500.
        - ``Termination`` tag equals ``"Normal"``.
        - 40 <= plies <= 240.

    ``path`` may point to a plain ``.pgn`` or a zstd-compressed ``.pgn.zst`` dump;
    files are decoded as UTF-8 and streamed game-by-game.
    """
    path = Path(path)
    counter = 0
    with _open_pgn_stream(path) as stream:
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            counter += 1

            headers = game.headers
            if not (
                _passes_time_control(headers)
                and _passes_elo(headers)
                and _passes_termination(headers)
            ):
                continue

            uci_moves = [move.uci() for move in game.mainline_moves()]
            if not (MIN_PLIES <= len(uci_moves) <= MAX_PLIES):
                continue

            yield _game_id(headers, counter), uci_moves


def iter_filtered_games(path: Path | str) -> Iterator[chess.pgn.Game]:
    """Yield :class:`chess.pgn.Game` objects passing the same project filters
    as :func:`iter_games`.

    Use this when you need the full game (headers + tree) so you can re-emit
    a valid PGN — e.g. building a smaller slice for smoke runs. For training
    pipelines, prefer :func:`iter_games` which is cheaper and yields just the
    UCI moves.
    """
    path = Path(path)
    with _open_pgn_stream(path) as stream:
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break

            headers = game.headers
            if not (
                _passes_time_control(headers)
                and _passes_elo(headers)
                and _passes_termination(headers)
            ):
                continue

            uci_moves = [move.uci() for move in game.mainline_moves()]
            if not (MIN_PLIES <= len(uci_moves) <= MAX_PLIES):
                continue

            yield game
