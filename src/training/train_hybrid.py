"""Train the hybrid CNN+LSTM next-move model on a Lichess PGN dump.

Mirrors the structure of :mod:`src.training.train_lstm` but uses the per-ply
streaming dataset (:class:`src.data.dataset.ChessMovePerPlyIterable`) so that
each training sample is a triple ``(history[64], board[18,8,8], target_id)``
matching the hybrid model's forward signature.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import (
    DEFAULT_HISTORY_LEN,
    ChessMovePerPlyIterable,
)
from src.data.pgn_parser import iter_games
from src.data.splits import split_games
from src.data.vocab import Vocab
from src.models.hybrid import MoveBoardHybrid
from src.training.seeding import seed_all


@dataclass
class TrainConfig:
    """Hyperparameters for a hybrid training run.

    Per-ply training has many more samples per epoch than the LSTM's per-game
    training (the same games × ~80 plies each), so fewer epochs are typically
    sufficient. Defaults below target a single Colab T4 session of ~60–90 min.
    """

    epochs: int = 4
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    num_workers: int = 0
    history_len: int = DEFAULT_HISTORY_LEN
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.2
    board_feature_dim: int = 256
    lr_factor: float = 0.5
    lr_patience: int = 1


def _resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
) -> tuple[float, int]:
    """Run one pass over ``loader``. Returns ``(sum_loss_weighted, n_examples)``.

    ``optimizer`` is ``None`` for evaluation. Loss is mean cross-entropy per
    example (one prediction per ply), so ``sum_loss_weighted / n`` gives the
    epoch's mean per-example loss and ``exp(mean)`` gives perplexity.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_examples = 0
    grad_context = torch.enable_grad if is_train else torch.no_grad

    with grad_context():
        for history, board, target in loader:
            history = history.to(device, non_blocking=True)
            board = board.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logits = model(history, board)
            loss = criterion(logits, target)
            n = target.size(0)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += float(loss.item()) * n
            total_examples += n

    return total_loss, total_examples


def main(
    pgn_path: Path,
    vocab_path: Path,
    out_path: Path,
    config: TrainConfig,
    seed: int = 42,
    device: str | None = None,
    use_wandb: bool = False,
    wandb_project: str = "chess-move-prediction",
) -> None:
    """End-to-end hybrid training: parse → split → vocab → train → checkpoint best."""
    seed_all(seed)
    dev = _resolve_device(device)
    print(f"[train_hybrid] device={dev}")

    print(f"[train_hybrid] reading games from {pgn_path}")
    games = [moves for _, moves in iter_games(pgn_path)]
    print(f"[train_hybrid] kept {len(games)} games after filtering")

    train, val, _test = split_games(games, seed=seed)
    print(f"[train_hybrid] split sizes: train={len(train)}  val={len(val)}")

    if vocab_path.exists():
        vocab = Vocab.load(vocab_path)
        print(f"[train_hybrid] loaded vocab from {vocab_path} (size={len(vocab)})")
    else:
        vocab = Vocab.build_from_games(train)
        vocab.save(vocab_path)
        print(f"[train_hybrid] built vocab from train split (size={len(vocab)}) -> {vocab_path}")

    train_ds = ChessMovePerPlyIterable(
        train, vocab, history_len=config.history_len, shuffle=True, seed=seed
    )
    val_ds = ChessMovePerPlyIterable(
        val, vocab, history_len=config.history_len, shuffle=False, seed=seed
    )
    print(
        f"[train_hybrid] positions: train={len(train_ds)}  val={len(val_ds)}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=(dev.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=(dev.type == "cuda"),
    )

    model = MoveBoardHybrid(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        start_id=vocab.start_id,
        end_id=vocab.end_id,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        board_feature_dim=config.board_feature_dim,
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_hybrid] model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience
    )
    criterion = nn.CrossEntropyLoss()

    wandb_run = None
    if use_wandb:
        import wandb  # local import so the dependency is optional at runtime

        wandb_run = wandb.init(
            project=wandb_project,
            config={
                **config.__dict__,
                "seed": seed,
                "vocab_size": len(vocab),
                "n_params": n_params,
                "model": "hybrid",
            },
        )

    best_val_loss = math.inf
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        train_ds.set_epoch(epoch)  # different shuffle per epoch
        train_loss_sum, train_examples = _run_epoch(
            model, train_loader, criterion, dev, optimizer, config.grad_clip
        )
        val_loss_sum, val_examples = _run_epoch(
            model, val_loader, criterion, dev, None, 0.0
        )

        train_loss = train_loss_sum / max(train_examples, 1)
        val_loss = val_loss_sum / max(val_examples, 1)
        train_pp = math.exp(train_loss)
        val_pp = math.exp(val_loss)
        dt = time.time() - t0

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            model.save(out_path)

        marker = "  (best)" if improved else ""
        print(
            f"[train_hybrid] epoch {epoch:>2}/{config.epochs} "
            f"train_loss={train_loss:.4f} train_pp={train_pp:.2f} "
            f"val_loss={val_loss:.4f} val_pp={val_pp:.2f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"time={dt:.1f}s{marker}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/perplexity": train_pp,
                    "val/loss": val_loss,
                    "val/perplexity": val_pp,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch_seconds": dt,
                }
            )

        scheduler.step(val_loss)

    print(
        f"[train_hybrid] best val_loss={best_val_loss:.4f}  "
        f"best val_pp={math.exp(best_val_loss):.2f}"
    )
    print(f"[train_hybrid] saved best checkpoint to {out_path}")
    if wandb_run is not None:
        wandb_run.finish()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the hybrid CNN+LSTM next-move model.")
    p.add_argument("--pgn", type=Path, required=True, help="Path to .pgn or .pgn.zst dump.")
    p.add_argument("--vocab", type=Path, required=True, help="Vocab JSON path (built if missing).")
    p.add_argument("--out", type=Path, required=True, help="Output path for the best checkpoint (.pt).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda).")
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--grad-clip", type=float, default=TrainConfig.grad_clip)
    p.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    p.add_argument("--history-len", type=int, default=TrainConfig.history_len)
    p.add_argument("--embedding-dim", type=int, default=TrainConfig.embedding_dim)
    p.add_argument("--hidden-dim", type=int, default=TrainConfig.hidden_dim)
    p.add_argument("--num-layers", type=int, default=TrainConfig.num_layers)
    p.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    p.add_argument("--board-feature-dim", type=int, default=TrainConfig.board_feature_dim)
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    p.add_argument("--wandb-project", type=str, default="chess-move-prediction")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        history_len=args.history_len,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        board_feature_dim=args.board_feature_dim,
    )
    main(
        pgn_path=args.pgn,
        vocab_path=args.vocab,
        out_path=args.out,
        config=cfg,
        seed=args.seed,
        device=args.device,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )
