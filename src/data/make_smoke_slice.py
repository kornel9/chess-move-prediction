"""Extract the first N filter-passing games from a Lichess dump into a smaller PGN.

Used to build the smoke slice required by ``CLAUDE.md`` ("Before each full-data
training run, do a smoke run on a 5–10k-game slice first"). The output is a
plain ``.pgn`` that the existing training scripts can re-parse with
:func:`src.data.pgn_parser.iter_games`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.pgn_parser import iter_filtered_games


def make_smoke_slice(src: Path | str, dst: Path | str, n_games: int) -> int:
    """Write the first ``n_games`` filter-passing games from ``src`` to ``dst``.

    Returns the number of games actually written (may be less than ``n_games``
    if the source runs out). ``src`` may be ``.pgn`` or ``.pgn.zst``; ``dst``
    is written as plain UTF-8 ``.pgn``.
    """
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dst_path.open("w", encoding="utf-8") as out:
        for game in iter_filtered_games(Path(src)):
            print(game, file=out, end="\n\n")
            written += 1
            if written >= n_games:
                break
    return written


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a small PGN slice for smoke training.")
    p.add_argument("--src", type=Path, required=True, help=".pgn or .pgn.zst dump.")
    p.add_argument("--dst", type=Path, required=True, help="Output .pgn path.")
    p.add_argument("--n", type=int, default=5000, help="Target number of filter-passing games.")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    print(f"[smoke] reading from {args.src}, target {args.n} games")
    n = make_smoke_slice(args.src, args.dst, args.n)
    size_mb = args.dst.stat().st_size / 1e6
    print(f"[smoke] wrote {n} games to {args.dst} ({size_mb:.1f} MB)")
