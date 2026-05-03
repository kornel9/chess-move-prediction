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
1. N-gram baseline (trigram with Katz backoff) — pure Python, no ML framework
2. LSTM on move sequences — PyTorch, 2-layer, hidden 512, embedding 256
3. Hybrid CNN + LSTM — CNN encodes 8×8×18 board tensor, LSTM encodes move history, outputs are concatenated

## Data
- Source: Lichess Open Database (PGN dumps), ~300k games after filtering
- Filtering: Rapid/Classical time controls, both players Elo 1500+, 20–120 moves, normal termination
- Tokenisation: UCI notation, ~1968-token vocabulary, vocab.json
- Board tensor: 8×8×18 (12 piece channels + 6 metadata channels — no tactical channels)
- Splits: 85% train / 10% val / 5% test, split by game not by position

## Evaluation metrics
- Top-1, Top-3, Top-5 accuracy
- Perplexity
- Phase-wise breakdown: opening (1–10), early mid (11–20), late mid (21–40), endgame (41+)
- Legal-move masking applied at inference for all models

## Tech stack
- Python 3.11, PyTorch 2.x, python-chess, numpy, pandas
- Streamlit for demo app
- W&B for experiment tracking
- No Docker, no FAISS, no Stockfish integration

## Project structure (target)
chess-move-prediction/
├── CLAUDE.md
├── requirements.txt
├── data/
│   ├── raw/          # downloaded PGN files
│   ├── processed/    # tokenised sequences, board tensors
│   └── vocab.json
├── src/
│   ├── data/         # PGN parser, filtering, board encoder, dataset classes
│   ├── models/       # ngram.py, lstm.py, hybrid.py
│   ├── training/     # train.py, evaluate.py
│   └── demo/         # streamlit_app.py
├── notebooks/        # exploration, figure generation
├── checkpoints/      # saved model weights
└── report/           # LaTeX or markdown report

## Coding conventions
- Type hints on all functions
- Docstrings on all public functions
- Unit tests for data pipeline components (pytest)
- Config via dataclasses or simple JSON, not argparse soup
- Seed everything: torch, numpy, random

## What this project is NOT doing
- No transformer models (LSTM is sufficient for the rubric)
- No Elo-conditioned models
- No FAISS similarity retrieval
- No Stockfish evaluation pipeline
- No calibration analysis (ECE, reliability diagrams)
- No data augmentation (board mirroring)