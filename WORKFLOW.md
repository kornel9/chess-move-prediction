# WORKFLOW.md — Process Log

Living development log for the Chess Move Prediction course project. Each meaningful step (new pipeline component, model, evaluation milestone, design decision) gets a short entry: **what** was built, **why**, and any **problems** worth remembering.

`CLAUDE.md` is the project authority (goals, rubric, conventions). This file is the engineering journal it draws from. The final, audience-facing writeup lives in `report/` and pulls from here.

**Last updated:** 2026-05-03 (Colab notebook for full run)

---

## 0. Setup

Layout: `src/{data,models,training,demo}/` mirrored by `tests/`; repo-root `conftest.py` puts the project root on `sys.path` so test imports like `from src.data.vocab import Vocab` just work. Python 3.11 in `.venv/`. Always run via `.venv/Scripts/python.exe` — the system Anaconda doesn't have torch.

## 1. Data pipeline ✅

**PGN parser (`src/data/pgn_parser.py`)** — streams `.pgn` / `.pgn.zst`, yields `(game_id, uci_moves)` for games passing the rubric filters: TimeControl estimated ≥ 480 s (Rapid + Classical), both sides Elo ≥ 1500, `Termination == "Normal"`, 40 ≤ plies ≤ 240. Why these filters: blitz/low-Elo/abnormal-termination games dilute the "skilled, complete play" signal we want the model to learn.

**Vocab (`src/data/vocab.py`)** — UCI tokens in sorted order with locked IDs `<PAD>=0, <START>=1, <END>=2, <UNK>=3`. Why locked: the LSTM and n-gram both bake special-token IDs into their state — reordering would silently corrupt every checkpoint.

**Board encoder (`src/data/board_encoder.py`)** — `(18, 8, 8) float32`: 12 piece planes (white P,N,B,R,Q,K then black) + 1 turn + 4 castling + 1 en-passant. Rank 0 = white's back rank. Why this layout (and only this): the hybrid CNN should learn geometry from raw piece positions, not from precomputed tactical channels (attack maps, threats) — that would conflate "the model knows chess" with "we precomputed it." Same logic ruled out board-mirroring augmentation.

**Datasets (`src/data/dataset.py`)** — `ChessMoveDataset` (per-ply, returns `(history[64], board[18,8,8], target_id)` for the hybrid) and `ChessMoveSequenceDataset` (per-game, `(input_ids, target_ids)` shifted by one for the LSTM, with `pad_collate` for batching). Why two: the hybrid needs a board at every step (only meaningful per-ply); the LSTM gets full-game BPTT at far lower cost from per-game batches.

**Splits (`src/data/splits.py`)** — 85/10/5 by game (never by position), deterministic via `seed=42`. Position-level splits would leak openings between train and test.

**Problems**: python-chess is on PyPI as `chess`, *not* `python-chess` (the latter is a different package). `requirements.txt` pins `chess==1.11.2`.

## 2. N-gram baseline ✅

`src/models/ngram.py` — `TrigramKatz`, pure Python, no framework. Trigram with Katz backoff, absolute discount `d=0.5`, Laplace-smoothed unigram floor so no probability is ever zero. Per-context backoff weight `α` cached lazily on first use. Counts in plain dicts; saved as gzip+pickle.

Trained via `src/training/train_ngram.py` (parse → split → fit → save → val perplexity sanity check).

Why this baseline: trigram is the smallest sequence context that captures opening-line shape without state explosion. Katz backoff with `d=0.5` is the textbook default and easy to defend in the report. Pure Python keeps the baseline architecturally distinct from the LSTM, which is the comparison the rubric rewards.

**Problems**: none recorded.

## 3. LSTM model + trainer ✅

`src/models/lstm.py` — `Embedding(V, 256, padding_idx=PAD) → LSTM(256→512, 2 layers, dropout 0.2) → Linear(512→V)`. Forward `(B, T) → (B, T, V)` logits. `predict_topk` applies legal-move masking via index_select on the candidate set.

`src/training/train_lstm.py` — sequence-level training with `CrossEntropyLoss(ignore_index=pad_id)`, AdamW + `ReduceLROnPlateau`, grad clip 1.0, best-by-val-loss checkpointing, optional W&B.

Why this size: 2 layers / hidden 512 / embedding 256 is the canonical "small but real" sequence baseline — big enough to learn the ~1968-token UCI distribution across 100+ ply contexts, small enough to train on a free-tier T4. Transformers were rejected: the rubric doesn't reward architectural novelty, and an LSTM is the textbook sequence-model baseline this report needs.

**Problems**: none recorded.

## 4. Eval harness ✅ (this session)

`src/training/evaluate.py` — shared evaluation for n-gram, LSTM, and the (future) hybrid.

Design: a `Predictor` `Protocol` with one method, `score_game(moves, k) -> list[PositionScore]`. Each model gets a thin adapter (`NgramPredictor`, `LSTMPredictor`); the harness only knows the protocol. The hybrid will plug in by adding a `HybridPredictor` — no change to the aggregator.
- `LSTMPredictor` runs **one forward pass per game** (logits at position `t` already predict move `t`), so cost scales with sequence length rather than N forward passes.
- `NgramPredictor` walks ply-by-ply maintaining the trigram `(a, b)` context.
- Both rebuild a `chess.Board` per ply to compute legal moves — the model itself has no legality knowledge.

Metrics: top-1 / top-3 / top-5 (legal-move-masked, specials excluded), per-token perplexity, and a phase-wise breakdown (opening / early-mid / late-mid / endgame from the CLAUDE.md ply boundaries).

Why no legal-mask on perplexity: masking inflates probabilities and breaks cross-model comparability — top-K is the legal-aware metric, perplexity is the raw-distribution one. Both metrics are needed; mixing them muddles both.

CLI: `python -m src.training.evaluate --model-type {ngram,lstm} --model … --pgn … --vocab … --split test`.

**Problems**: first pass at `evaluate()` left dead code (a stale `by_phase` dict comprehension that the populating loop overwrote); cleaned before tests. 11 new tests cover phase boundaries, aggregation math, both predictors' contracts, and that `LSTMPredictor.score_game` doesn't strand the model in eval mode. 68/68 tests pass.

## 5. Smoke run on 5k-game slice ✅

End-to-end pipeline validation per the CLAUDE.md "smoke first" rule. Source: Lichess 2013-01 monthly (`lichess_db_standard_rated_2013-01.pgn.zst`, 17 MB compressed) downloaded into `data/raw/`. Sliced to `data/raw/smoke.pgn` (5 000 filter-passing games, 4.5 MB) via the new `src/data/make_smoke_slice.py`, which streams the dump through the existing filters and re-emits valid PGN.

**N-gram** (`src/training/train_ngram.py`): 4 250-game train split → vocab size 1 870, 175 144 trigram contexts, val pp 403. Saved to `checkpoints/smoke/ngram.pkl.gz`.

**LSTM** (CPU smoke config: embedding 64 / hidden 128 / 1 layer / 2 epochs / batch 32, 460 k params): 2 epochs in ~150 s, val pp dropped 552 → 370. Saved to `checkpoints/smoke/lstm.pt`.

**Eval on test split (250 games, 20 203 positions)** via the new harness:

| Metric | N-gram | LSTM (smoke) |
|---|---|---|
| top-1 / top-3 / top-5 | **0.213 / 0.404 / 0.498** | 0.134 / 0.280 / 0.374 |
| perplexity | 386.8 | **370.1** |
| opening top-1 / pp | **0.336** / **28.1** | 0.243 / 51.7 |
| early-mid top-1 / pp | **0.178** / **409.5** | 0.078 / 433.9 |
| late-mid top-1 / pp | **0.155** / 1 260.7 | 0.080 / **866.8** |
| endgame top-1 / pp | **0.203** / 1 631.3 | 0.173 / **976.8** |

**Reading the numbers**: n-gram dominates top-K everywhere (memorizes opening lines on 4 250 games); LSTM already wins on perplexity overall and in 2 phases — its distribution is better calibrated even though its argmax is worse. Both models behave as expected by phase: openings easy (top-1 30 %+), midgame is the bottleneck (top-1 8–18 %), endgame perplexity is rarity-driven. **Pipeline is sound; the smoke LSTM is undertrained as designed.** The full run (embedding 256 / hidden 512 / 2 layers / 8 epochs / 300 k games) is where the LSTM should overtake on top-K too — and is the moment the hybrid becomes worth evaluating against.

**Problems / decisions**:
- Found a stale 8-token `data/vocab.json` left over from a fixture run; it silently encoded everything to `<UNK>` and made the n-gram report a fake pp=1.19. Deleted and rebuilt from the smoke train split. (Lesson: vocab.json should never be checked in if it can be derived from the train split — it's already gitignored, but worth flagging for the report.)
- Added `iter_filtered_games` to `src/data/pgn_parser.py` so the slicer can re-emit valid PGN (the existing `iter_games` loses headers). Existing 5 tests still pass.
- 2013-01 is early Lichess; the Elo ≥ 1500 filter survived fine (5 000 games extracted in seconds). For larger smoke slices a 2014+ monthly would be safer.

## 6. Colab notebook for the full run ✅

`notebooks/train_full_lstm_colab.ipynb` — staged the full-data run on free-tier T4. The notebook mounts Drive, clones the repo from GitHub, installs only the deps Colab doesn't already ship (`chess`, `zstandard`, `wandb`), downloads one Lichess monthly (default `2017-01`, 1.9 GB compressed — chosen so the post-filter yield comfortably exceeds 300 k games), slices to `data/raw/full.pgn` via the existing `make_smoke_slice.py`, runs `train_lstm.py` with the CLAUDE.md target config (embedding 256 / hidden 512 / 2 layers / dropout 0.2 / 8 epochs / batch 64) optionally with W&B, evaluates on the test split, and copies `lstm.pt` + `vocab.json` + `eval_test.txt` back to Drive. Cell 9 also trains and evaluates the n-gram on the same slice for an apples-to-apples baseline. Expected wall-clock: ≈1–2 h on T4.

What the notebook needs from the user before running: (1) push this repo to GitHub and edit `GITHUB_URL` in cell 5, or use the Drive-zip alt path documented above it; (2) Runtime → T4 GPU; (3) optional W&B API key. Bring back to the next session: contents of `eval_test_lstm.txt` / `eval_test_ngram.txt`, the W&B run URL, total wall-clock, and any surprises.

**Decisions / open thoughts**:
- 300 k games is the target from CLAUDE.md, not a hard floor. If 2017-01 yields more, the slicer just stops at 300 k. If a later monthly is needed for a sanity-check rerun, the notebook is parameterized via `MONTHLY` and `N_GAMES`.
- Reused `make_smoke_slice.py` directly rather than renaming it — the script is general-purpose; only the original use case was "smoke". A note in the notebook calls this out.

---

## 7. Up next

- [ ] **Run the Colab notebook**, paste results back here.
- [ ] **Hybrid CNN + LSTM** (`src/models/hybrid.py`) — CNN over `(18, 8, 8)` board, concat with LSTM hidden, shared head. Worth doing only after the full LSTM run gives a real comparison point.
- [ ] **Streamlit demo** (`src/demo/streamlit_app.py`).
- [ ] **Final report** (`report/`).
