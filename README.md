# Chess Next-Move Prediction

**Predicting the next chess move from a game's history and board position.**
Course final project for Machine Learning at Corvinus University of Budapest, comparing a statistical baseline against two neural sequence models.

The full write-up is in [`report/report.md`](report/report.md) (PDF: [`report/report.pdf`](report/report.pdf)).

---

## Models

Three next-move predictors trained and evaluated on the same 300 000-game slice of [Lichess Open Database](https://database.lichess.org/) (January 2017), filtered to Rapid/Classical games where both players are Elo ≥ 1500.

| Model | Description | Params |
| --- | --- | ---: |
| **Trigram (Katz back-off)** | Pure-Python statistical baseline over UCI move tokens. Absolute discount d = 0.5. | – |
| **LSTM** | 2-layer LSTM over UCI move sequences, embedding 256, hidden 512. | 5.17 M |
| **Hybrid CNN + LSTM** | Same LSTM branch over moves, plus a 3-layer CNN over an 8×8×18 board tensor. | 8.0 M |

All three apply **legal-move masking** at inference, so they cannot recommend illegal or invented moves.

## Headline results

Three-way comparison on a held-out 15 000-game test split (1.16 M position-prediction pairs).

| Metric | Trigram | LSTM | Hybrid |
| --- | ---: | ---: | ---: |
| Top-1 accuracy | 0.252 | 0.371 | **0.408** |
| Top-3 accuracy | 0.458 | 0.630 | **0.675** |
| Top-5 accuracy | 0.563 | 0.744 | **0.784** |
| Perplexity | 156.7 | 10.51 | **8.03** |

The most interesting finding is per-phase: adding board awareness (LSTM → hybrid) gives essentially **no gain in the opening** but improves perplexity by **47 % in the endgame** — board state matters most precisely where the move-sequence context is least informative. Full breakdown and discussion in [`report/report.md`](report/report.md), section 6.

---

## Repository layout

```
chess-move-prediction/
├── src/
│   ├── data/        # PGN parsing, tokenisation, board encoding, splits, datasets
│   ├── models/      # ngram.py · lstm.py · hybrid.py
│   ├── training/    # train_ngram.py · train_lstm.py · train_hybrid.py · evaluate.py
│   └── demo/        # streamlit_app.py · board_image.py
├── tests/           # pytest suite mirroring src/
├── notebooks/       # Colab notebooks for full-data training on a free T4
├── report/          # report.md + report.pdf
├── WORKFLOW.md      # Engineering journal — what was built, why, and gotchas
└── requirements.txt
```

## Quickstart: run the demo

The demo is a Streamlit app with two modes — *Free play* (click pieces on the board, see each model's top-5 predictions update in real time, with multi-coloured arrows showing each model's top-1) and *Game playback* (paste a PGN, step through it, and see whether each model would have predicted the actually-played move at every position).

```bash
git clone https://github.com/kornel9/chess-move-prediction
cd chess-move-prediction

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

You'll need three checkpoints in `checkpoints/full/` to run the demo:

- `ngram.pkl.gz` — trigram baseline
- `lstm.pt` — LSTM checkpoint
- `hybrid.pt` — hybrid CNN+LSTM checkpoint (optional; the demo silently hides that panel if missing)

…and `data/vocab.json` (vocab built from the train split). Either retrain (see below) or use the project's pre-trained checkpoints. Then:

```bash
python -m streamlit run src/demo/streamlit_app.py
```

Open <http://localhost:8501> and pick a mode at the top.
You can also use this Streamlit link to run it in the browser: https://chess-move-prediction-demo.streamlit.app/

## Reproducing the results

Two free-tier Colab notebooks under `notebooks/` train end-to-end on a single T4 GPU. Run the LSTM notebook first (it builds the vocab the hybrid reuses):

1. [`notebooks/train_full_lstm_colab.ipynb`](notebooks/train_full_lstm_colab.ipynb) — downloads a Lichess monthly, slices to 300 k filter-passing games, trains the LSTM (≈1–2 h), evaluates on the test split, optionally trains and evaluates the n-gram baseline on the same slice.
2. [`notebooks/train_full_hybrid_colab.ipynb`](notebooks/train_full_hybrid_colab.ipynb) — re-uses the LSTM run's vocab so all three models evaluate on identical splits with identical token IDs. Trains the hybrid (≈60–90 min) and evaluates on the test split.

Both checkpoints persist directly to Google Drive so a Colab runtime disconnect can't lose progress.

To retrain locally (CPU-only is feasible only on a small slice — full 300 k is GPU-bound):

```bash
# Smoke run on 5 000 games to validate the pipeline end-to-end
python -m src.data.make_smoke_slice --src data/raw/lichess_2017-01.pgn.zst --dst data/raw/smoke.pgn --n 5000

python -m src.training.train_ngram  --pgn data/raw/smoke.pgn --vocab data/vocab.json --out checkpoints/ngram.pkl.gz
python -m src.training.train_lstm   --pgn data/raw/smoke.pgn --vocab data/vocab.json --out checkpoints/lstm.pt    --epochs 2
python -m src.training.train_hybrid --pgn data/raw/smoke.pgn --vocab data/vocab.json --out checkpoints/hybrid.pt  --epochs 2

python -m src.training.evaluate --model-type lstm   --model checkpoints/lstm.pt    --pgn data/raw/smoke.pgn --vocab data/vocab.json --split test
python -m src.training.evaluate --model-type ngram  --model checkpoints/ngram.pkl.gz --pgn data/raw/smoke.pgn --vocab data/vocab.json --split test
python -m src.training.evaluate --model-type hybrid --model checkpoints/hybrid.pt  --pgn data/raw/smoke.pgn --vocab data/vocab.json --split test
```

## Tests

```bash
pytest -q
```

The suite covers the data pipeline (parser, vocab, splits, board encoder, datasets), the n-gram model, the LSTM forward/backward and predict, the hybrid forward/predict, and the evaluation harness. Tests run on CPU and complete in a few seconds.

## Tech stack

- Python 3.11
- PyTorch 2.5
- python-chess (installed as `chess` on PyPI — *not* `python-chess`)
- numpy, pandas, zstandard
- Streamlit + streamlit-image-coordinates (demo)
- W&B (optional, for experiment tracking during training)
- pytest (test suite)

Pinned versions in [`requirements.txt`](requirements.txt).

## License

Course-project code; no specific license terms beyond standard academic-use expectations. Contact the author for reuse.
