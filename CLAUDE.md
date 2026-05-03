# Chess Next-Move Prediction — ML Course Project

## Project goal
Predict the next move of a chess player given a game history and board position.
Apply sequence modelling to behaviour prediction and compare to a simpler baseline.
Submission: May 10 2026. 40-point rubric.

## Rubric weights (what matters)
- Sequential scenario definition: 15%
- Data preparation: 20%
- Baseline model: 15%
- Sequence model: 25%  ← biggest category
- Performance analysis: 15%
- Report and demo: 10%

## Models we are building (in order)
1. N-gram baseline (trigram with Katz backoff, absolute discount d=0.5) — pure Python, no ML framework
2. LSTM on move sequences — PyTorch, 2-layer, hidden 512, embedding 256
3. Hybrid CNN + LSTM — CNN encodes 8×8×18 board tensor, LSTM encodes move history, outputs are concatenated

## Data
- Source: Lichess Open Database (PGN dumps), ~300k games after filtering
- Filtering: Rapid/Classical time controls (estimated seconds ≥ 480), both players Elo ≥ 1500, 40–240 plies (20–120 full moves), `Termination == "Normal"`
- Tokenisation: UCI notation, ~1972-token vocabulary (4 specials + ~1968 UCI moves), `data/vocab.json`
- Special token IDs (locked in): `<PAD>=0`, `<START>=1`, `<END>=2`, `<UNK>=3`
- Board tensor: 8×8×18 float32 — 12 piece planes (P,N,B,R,Q,K white then black) + 6 metadata planes (1 turn + 4 castling K,Q,k,q + 1 en-passant target). Rank 0 = white's back rank. No tactical channels.
- Move history: per-ply dataset granularity (one sample per position). History truncated to last 64 plies, left-padded with `<PAD>`, `<START>` prepended before truncation.
- Splits: 85% train / 10% val / 5% test, split by game not by position; deterministic via `src/data/splits.py` (seed=42).

## Evaluation metrics
- Top-1, Top-3, Top-5 accuracy
- Perplexity
- Phase-wise breakdown: opening (1–10), early mid (11–20), late mid (21–40), endgame (41+)
- Legal-move masking applied at inference for all models

## Tech stack
- Python 3.11, PyTorch 2.x, python-chess (installed as `chess` on PyPI — *not* `python-chess`), numpy, pandas, zstandard
- Streamlit for demo app
- W&B for experiment tracking
- pytest for unit tests; `conftest.py` at repo root injects the project root onto `sys.path`
- No Docker, no FAISS, no Stockfish integration

## Project structure (target)
chess-move-prediction/
├── CLAUDE.md
├── WORKFLOW.md                 # engineering journal — updated after each meaningful step
├── conftest.py                 # injects repo root onto sys.path for tests
├── requirements.txt
├── data/
│   ├── raw/                    # downloaded PGN files (gitignored)
│   ├── processed/              # tokenised sequences, board tensors (gitignored)
│   └── vocab.json
├── src/
│   ├── data/                   # pgn_parser, vocab, board_encoder, dataset, splits
│   ├── models/                 # ngram.py, lstm.py, hybrid.py
│   ├── training/               # train_ngram.py, train_lstm.py, train_hybrid.py, evaluate.py
│   └── demo/                   # streamlit_app.py
├── tests/                      # pytest suites mirroring src/ layout
├── notebooks/                  # exploration, figure generation
├── checkpoints/                # saved model weights (gitignored)
└── report/                     # LaTeX or markdown report

## Coding conventions
- Type hints on all functions
- Docstrings on all public functions
- Unit tests for data pipeline components (pytest)
- Config via dataclasses or simple JSON, not argparse soup
- Seed everything: torch, numpy, random

## Workflow expectations
- This file is the authority. Claude follows it by default and won't silently deviate.
- When Claude sees a meaningful improvement or gap — even one that contradicts something here — Claude flags it as a recommendation and asks before changing course.
- Decisions agreed mid-session get written back into this file so future sessions inherit them.
- After completing a meaningful step (new pipeline component, model, evaluation milestone, or design decision), append a short entry to `WORKFLOW.md` covering **what** was built, **why**, and any **problems** worth remembering. Update the "Last updated" date and tick the relevant item under "Up next." Keep entries concise — it's an engineering journal, not a duplicate of this file.
- Non-obvious one-off facts (environment quirks, external resources) go into the auto-memory system instead of bloating this file.
- Always run Python via `.venv/Scripts/python.exe` — the project venv has torch and the project deps; the system Anaconda does not.
- Before each full-data training run, do a smoke run on a 5–10k-game slice first.

## When to step outside this session
Some work is impractical inside a Claude Code session. Claude flags these explicitly whenever we hit them:
- **GPU training runs** — full-data LSTM and hybrid training. Use Google Colab, Kaggle, or Lightning AI free-tier T4/L4. CPU training on 300k games is too slow to iterate on.
- **Hyperparameter sweeps** — run in a notebook with W&B; bring summary metrics back here.
- **W&B dashboard inspection** — open the run page in a browser; paste the relevant numbers or charts back if you want them interpreted here.
- **Large PGN dump downloads** — Lichess monthlies are multi-GB. Download outside the session and point the training script at the resulting path.
For each of these Claude states up front what to take to the external tool (script + args + expected output) and what to bring back.

## What this project is NOT doing
- No transformer models (LSTM is sufficient for the rubric)
- No Elo-conditioned models
- No FAISS similarity retrieval
- No Stockfish evaluation pipeline
- No calibration analysis (ECE, reliability diagrams)
- No data augmentation (board mirroring)