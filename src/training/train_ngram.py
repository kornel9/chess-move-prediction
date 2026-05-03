"""Train the trigram baseline end-to-end from a Lichess PGN dump."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.pgn_parser import iter_games
from src.data.splits import split_games
from src.data.vocab import Vocab
from src.models.ngram import TrigramKatz


def _encode(games: list[list[str]], vocab: Vocab) -> list[list[int]]:
    return [[vocab.encode(m) for m in g] for g in games]


def main(
    pgn_path: Path,
    vocab_path: Path,
    out_path: Path,
    seed: int = 42,
    discount: float = 0.5,
) -> None:
    """Parse, split, fit, and save the trigram model.

    If ``vocab_path`` does not exist, a vocab is built from the train split
    only and written there. Validation perplexity is printed as a sanity check.
    """
    print(f"[train_ngram] reading games from {pgn_path}")
    games = [moves for _, moves in iter_games(pgn_path)]
    print(f"[train_ngram] kept {len(games)} games after filtering")

    train, val, test = split_games(games, seed=seed)
    print(f"[train_ngram] split sizes: train={len(train)}  val={len(val)}  test={len(test)}")

    if vocab_path.exists():
        vocab = Vocab.load(vocab_path)
        print(f"[train_ngram] loaded vocab from {vocab_path} (size={len(vocab)})")
    else:
        vocab = Vocab.build_from_games(train)
        vocab.save(vocab_path)
        print(f"[train_ngram] built vocab from train split (size={len(vocab)}) -> {vocab_path}")

    encoded_train = _encode(train, vocab)
    model = TrigramKatz.fit(
        encoded_games=encoded_train,
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        start_id=vocab.start_id,
        end_id=vocab.end_id,
        discount=discount,
    )
    print(
        f"[train_ngram] fit complete: "
        f"unigram_total={model.unigram_total}  "
        f"bigram_contexts={len(model.bigram_counts)}  "
        f"trigram_contexts={len(model.trigram_counts)}"
    )

    model.save(out_path)
    print(f"[train_ngram] saved model to {out_path}")

    if val:
        encoded_val = _encode(val, vocab)
        val_pp = model.perplexity(encoded_val)
        print(f"[train_ngram] validation perplexity: {val_pp:.2f}")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the trigram (Katz backoff) baseline.")
    parser.add_argument("--pgn", type=Path, required=True, help="Path to .pgn or .pgn.zst dump.")
    parser.add_argument("--vocab", type=Path, required=True, help="Vocab JSON path (built if missing).")
    parser.add_argument("--out", type=Path, required=True, help="Output path for the trained model (.pkl.gz).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--discount", type=float, default=0.5)
    return parser


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    main(
        pgn_path=args.pgn,
        vocab_path=args.vocab,
        out_path=args.out,
        seed=args.seed,
        discount=args.discount,
    )
