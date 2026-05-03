"""Train the LSTM next-move model on a Lichess PGN dump."""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import ChessMoveSequenceDataset, pad_collate
from src.data.pgn_parser import iter_games
from src.data.splits import split_games
from src.data.vocab import Vocab
from src.models.lstm import MoveLSTM
from src.training.seeding import seed_all


@dataclass
class TrainConfig:
    """Hyperparameters for an LSTM training run."""

    epochs: int = 8
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    num_workers: int = 0
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.2
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
    """Run one pass over ``loader``. Returns ``(sum_loss_weighted, n_tokens)``.

    ``optimizer`` is ``None`` for evaluation. ``sum_loss_weighted`` is the sum
    of per-token cross-entropy losses (so dividing by ``n_tokens`` gives the
    epoch's mean per-token loss, and ``exp(mean)`` gives perplexity).
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_tokens = 0
    grad_context = torch.enable_grad if is_train else torch.no_grad

    with grad_context():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(inputs)
            n_tokens = int((targets != criterion.ignore_index).sum().item())
            if n_tokens == 0:
                continue

            loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += float(loss.item()) * n_tokens
            total_tokens += n_tokens

    return total_loss, total_tokens


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
    """End-to-end training: parse → split → vocab → train → checkpoint best."""
    seed_all(seed)
    dev = _resolve_device(device)
    print(f"[train_lstm] device={dev}")

    print(f"[train_lstm] reading games from {pgn_path}")
    games = [moves for _, moves in iter_games(pgn_path)]
    print(f"[train_lstm] kept {len(games)} games after filtering")

    train, val, _test = split_games(games, seed=seed)
    print(f"[train_lstm] split sizes: train={len(train)}  val={len(val)}")

    if vocab_path.exists():
        vocab = Vocab.load(vocab_path)
        print(f"[train_lstm] loaded vocab from {vocab_path} (size={len(vocab)})")
    else:
        vocab = Vocab.build_from_games(train)
        vocab.save(vocab_path)
        print(f"[train_lstm] built vocab from train split (size={len(vocab)}) -> {vocab_path}")

    train_ds = ChessMoveSequenceDataset(train, vocab)
    val_ds = ChessMoveSequenceDataset(val, vocab)
    collate = partial(pad_collate, pad_id=vocab.pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=config.num_workers,
        pin_memory=(dev.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=config.num_workers,
        pin_memory=(dev.type == "cuda"),
    )

    model = MoveLSTM(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        start_id=vocab.start_id,
        end_id=vocab.end_id,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_lstm] model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience
    )
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    wandb_run = None
    if use_wandb:
        import wandb  # local import so the dependency is optional at runtime

        wandb_run = wandb.init(
            project=wandb_project,
            config={**config.__dict__, "seed": seed, "vocab_size": len(vocab), "n_params": n_params},
        )

    best_val_loss = math.inf
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        train_loss_sum, train_tokens = _run_epoch(
            model, train_loader, criterion, dev, optimizer, config.grad_clip
        )
        val_loss_sum, val_tokens = _run_epoch(model, val_loader, criterion, dev, None, 0.0)

        train_loss = train_loss_sum / max(train_tokens, 1)
        val_loss = val_loss_sum / max(val_tokens, 1)
        train_pp = math.exp(train_loss)
        val_pp = math.exp(val_loss)
        dt = time.time() - t0

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            model.save(out_path)

        marker = "  (best)" if improved else ""
        print(
            f"[train_lstm] epoch {epoch:>2}/{config.epochs} "
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

    print(f"[train_lstm] best val_loss={best_val_loss:.4f}  best val_pp={math.exp(best_val_loss):.2f}")
    print(f"[train_lstm] saved best checkpoint to {out_path}")
    if wandb_run is not None:
        wandb_run.finish()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the LSTM next-move model.")
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
    p.add_argument("--embedding-dim", type=int, default=TrainConfig.embedding_dim)
    p.add_argument("--hidden-dim", type=int, default=TrainConfig.hidden_dim)
    p.add_argument("--num-layers", type=int, default=TrainConfig.num_layers)
    p.add_argument("--dropout", type=float, default=TrainConfig.dropout)
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
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
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
