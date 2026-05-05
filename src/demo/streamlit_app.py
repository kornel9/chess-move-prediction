"""Streamlit demo: interactive chess board with side-by-side model predictions.

Run from the project root with:

    .venv\\Scripts\\python.exe -m streamlit run src/demo/streamlit_app.py

Then open http://localhost:8501 in a browser.

The app loads:
- The trained LSTM       (`checkpoints/full/lstm.pt`)
- The trigram baseline   (`checkpoints/full/ngram.pkl.gz`)
- The hybrid CNN+LSTM    (`checkpoints/full/hybrid.pt`) — *optional*, used
  if the file exists; the panel is skipped otherwise.
- The vocab              (`data/vocab.json`)

All inference runs on CPU. Pick any legal move and watch the panels update
— predictions are restricted to legal moves only, so the demo can never
recommend an illegal or invented move.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit's `streamlit run` puts the script's directory on sys.path, not the
# project root. Inject the project root explicitly so `from src...` imports work.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import chess
import chess.svg
import numpy as np
import streamlit as st
import torch

from src.data.board_encoder import encode_board
from src.data.dataset import DEFAULT_HISTORY_LEN
from src.data.vocab import Vocab
from src.models.hybrid import MoveBoardHybrid
from src.models.lstm import MoveLSTM
from src.models.ngram import TrigramKatz
from src.training.evaluate import phase_for_ply


LSTM_PATH = _ROOT / "checkpoints" / "full" / "lstm.pt"
NGRAM_PATH = _ROOT / "checkpoints" / "full" / "ngram.pkl.gz"
HYBRID_PATH = _ROOT / "checkpoints" / "full" / "hybrid.pt"
VOCAB_PATH = _ROOT / "data" / "vocab.json"

K = 5
LSTM_ARROW_COLOR = "#22c55e"   # green-500
LSTM_BAR_COLOR = "#3b82f6"     # blue-500
NGRAM_BAR_COLOR = "#8b5cf6"    # violet-500
HYBRID_BAR_COLOR = "#f59e0b"   # amber-500


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models …")
def load_artifacts() -> tuple[Vocab, MoveLSTM, TrigramKatz, MoveBoardHybrid | None]:
    """Load vocab + LSTM + n-gram + (optional) hybrid once per process.

    The hybrid is loaded lazily: if ``checkpoints/full/hybrid.pt`` is missing
    locally (e.g. you haven't downloaded it from Drive yet) the demo silently
    skips that panel rather than failing at startup.
    """
    vocab = Vocab.load(VOCAB_PATH)

    lstm = MoveLSTM.load(LSTM_PATH, map_location="cpu")
    lstm.eval()
    with torch.no_grad():
        _ = lstm(torch.tensor([[vocab.start_id]], dtype=torch.long))

    ngram = TrigramKatz.load(NGRAM_PATH)

    hybrid: MoveBoardHybrid | None = None
    if HYBRID_PATH.exists():
        hybrid = MoveBoardHybrid.load(HYBRID_PATH, map_location="cpu")
        hybrid.eval()
        with torch.no_grad():
            dummy_h = torch.tensor([[vocab.pad_id] * (DEFAULT_HISTORY_LEN - 1) + [vocab.start_id]], dtype=torch.long)
            dummy_b = torch.zeros(1, 18, 8, 8, dtype=torch.float32)
            _ = hybrid(dummy_h, dummy_b)

    return vocab, lstm, ngram, hybrid


@st.cache_data(show_spinner=False)
def _lstm_predict_cached(
    history_tuple: tuple[int, ...],
    legal_ids_tuple: tuple[int, ...],
) -> list[tuple[int, float]]:
    """Top-K LSTM predictions for a given history + legal-move set."""
    _, lstm, _, _ = load_artifacts()
    with torch.no_grad():
        history = torch.tensor(history_tuple, dtype=torch.long)
        return lstm.predict_topk(history, k=K, legal_token_ids=legal_ids_tuple)


@st.cache_data(show_spinner=False)
def _ngram_predict_cached(
    context: tuple[int, int],
    legal_ids_tuple: tuple[int, ...],
) -> list[tuple[int, float]]:
    """Top-K n-gram predictions for a given trigram context + legal-move set."""
    _, _, ngram, _ = load_artifacts()
    return ngram.predict_topk(context, k=K, legal_token_ids=legal_ids_tuple)


@st.cache_data(show_spinner=False)
def _hybrid_predict_cached(
    history_tuple: tuple[int, ...],
    board_fen: str,
    legal_ids_tuple: tuple[int, ...],
) -> list[tuple[int, float]]:
    """Top-K hybrid predictions for a (history, board) pair + legal-move set.

    ``board_fen`` is used as a hashable cache key for the board state — the
    actual ``(18, 8, 8)`` tensor is reconstructed from it on cache miss.
    """
    _, _, _, hybrid = load_artifacts()
    if hybrid is None:
        return []
    board = chess.Board(board_fen)
    with torch.no_grad():
        history = torch.tensor(history_tuple, dtype=torch.long)
        board_tensor = torch.from_numpy(encode_board(board))
        return hybrid.predict_topk(history, board_tensor, k=K, legal_token_ids=legal_ids_tuple)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state(start_id: int) -> None:
    """Seed session-state on first run."""
    ss = st.session_state
    if "board" not in ss:
        ss.board = chess.Board()
        ss.history_ids = [start_id]
        ss.move_stack_uci = []
        ss.last_move = None


def _reset_state(start_id: int) -> None:
    ss = st.session_state
    ss.board = chess.Board()
    ss.history_ids = [start_id]
    ss.move_stack_uci = []
    ss.last_move = None


# ---------------------------------------------------------------------------
# Action handlers (Streamlit on_click callbacks)
# ---------------------------------------------------------------------------
def play_move(uci: str) -> None:
    """Push a UCI move onto the board and append to the history."""
    vocab, _, _, _ = load_artifacts()
    ss = st.session_state
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return
    if move in ss.board.legal_moves:
        ss.board.push(move)
        ss.history_ids.append(vocab.encode(uci))
        ss.move_stack_uci.append(uci)
        ss.last_move = move


def undo() -> None:
    ss = st.session_state
    if ss.move_stack_uci:
        ss.board.pop()
        ss.history_ids.pop()
        ss.move_stack_uci.pop()
        ss.last_move = ss.board.move_stack[-1] if ss.board.move_stack else None


def reset() -> None:
    vocab, _, _, _ = load_artifacts()
    _reset_state(vocab.start_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _legal_token_ids(board: chess.Board, vocab: Vocab) -> tuple[int, ...]:
    return tuple(sorted({vocab.encode(m.uci()) for m in board.legal_moves}))


def _renormalize(pairs: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Softmax-renormalize log-probs over the returned top-K so they sum to 1.

    The per-row probability shown in the UI is therefore "share among the
    shown top-5 legal moves". That keeps the bars visually meaningful without
    pretending the legal-mask doesn't drop probability mass.
    """
    if not pairs:
        return []
    log_probs = torch.tensor([lp for _, lp in pairs])
    probs = torch.softmax(log_probs, dim=0).tolist()
    return [(tok, p) for (tok, _), p in zip(pairs, probs)]


def _render_panel(
    title: str,
    pairs: list[tuple[int, float]],
    vocab: Vocab,
    bar_color: str,
    key_prefix: str,
) -> None:
    st.markdown(f"#### {title}")
    if not pairs:
        st.markdown("_no predictions (game over)_")
        return
    norm = _renormalize(pairs)
    for i, (token_id, prob) in enumerate(norm):
        uci = vocab.decode(token_id)
        cols = st.columns([1, 3, 1])
        cols[0].button(
            uci,
            key=f"{key_prefix}_{i}",
            on_click=play_move,
            args=(uci,),
            use_container_width=True,
        )
        cols[1].markdown(
            f'<div style="background:#e5e7eb;border-radius:4px;height:28px;'
            f'width:100%;overflow:hidden;">'
            f'<div style="background:{bar_color};height:100%;'
            f'width:{prob * 100:.1f}%;"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        cols[2].markdown(f"`{prob:.1%}`")


def _gameover_reason(board: chess.Board) -> str:
    if board.is_checkmate():
        winner = "Black" if board.turn else "White"
        return f"Checkmate — {winner} wins"
    if board.is_stalemate():
        return "Stalemate — draw"
    if board.is_insufficient_material():
        return "Draw — insufficient material"
    if board.is_seventyfive_moves():
        return "Draw — 75-move rule"
    if board.is_fivefold_repetition():
        return "Draw — fivefold repetition"
    return "Game over"


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
def render() -> None:
    st.set_page_config(
        page_title="Chess move prediction demo",
        layout="wide",
        page_icon="♞",
    )

    vocab, _, _, hybrid = load_artifacts()
    _init_state(vocab.start_id)
    ss = st.session_state
    board: chess.Board = ss.board

    legal_ids = _legal_token_ids(board, vocab)
    ply = board.ply()
    phase = phase_for_ply(ply)
    gameover = board.is_game_over()

    if gameover:
        lstm_top: list[tuple[int, float]] = []
        ngram_top: list[tuple[int, float]] = []
        hybrid_top: list[tuple[int, float]] = []
    else:
        lstm_top = _lstm_predict_cached(tuple(ss.history_ids), legal_ids)
        a = ss.history_ids[-2] if len(ss.history_ids) >= 2 else vocab.start_id
        b = ss.history_ids[-1] if len(ss.history_ids) >= 1 else vocab.start_id
        ngram_top = _ngram_predict_cached((a, b), legal_ids)
        if hybrid is not None:
            # Hybrid expects history left-padded to history_len (matches the
            # ChessMoveDataset / ChessMovePerPlyIterable contract used in training).
            h = ss.history_ids
            if len(h) >= DEFAULT_HISTORY_LEN:
                h_padded = tuple(h[-DEFAULT_HISTORY_LEN:])
            else:
                h_padded = tuple([vocab.pad_id] * (DEFAULT_HISTORY_LEN - len(h)) + h)
            hybrid_top = _hybrid_predict_cached(h_padded, board.fen(), legal_ids)
        else:
            hybrid_top = []

    arrows: list[chess.svg.Arrow] = []
    if lstm_top:
        top1_uci = vocab.decode(lstm_top[0][0])
        try:
            top1_move = chess.Move.from_uci(top1_uci)
            arrows.append(
                chess.svg.Arrow(
                    top1_move.from_square,
                    top1_move.to_square,
                    color=LSTM_ARROW_COLOR,
                )
            )
        except ValueError:
            pass

    left, right = st.columns([3, 2])

    with left:
        st.markdown("## ♞ Chess move prediction")
        side = "White" if board.turn else "Black"
        if gameover:
            badge = f":red[**Game over —**] {_gameover_reason(board)}"
        else:
            badge = (
                f"**Ply** `{ply}`  ·  **Phase** `{phase}`  ·  **{side}** to move"
            )
        st.markdown(badge)

        svg = chess.svg.board(
            board,
            lastmove=ss.last_move,
            arrows=arrows,
            size=480,
        )
        st.markdown(
            f'<div style="display:flex;justify-content:center;">{svg}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Green arrow = LSTM top-1 prediction.")

        if not gameover:
            legal_uci = sorted(m.uci() for m in board.legal_moves)
            pick = st.selectbox("Make a move", legal_uci, key="picker")
        else:
            pick = None

        c1, c2, c3 = st.columns(3)
        c1.button(
            "Play",
            on_click=play_move,
            args=(pick or "",),
            disabled=(pick is None or gameover),
            use_container_width=True,
            type="primary",
        )
        c2.button(
            "Undo",
            on_click=undo,
            disabled=(not ss.move_stack_uci),
            use_container_width=True,
        )
        c3.button("Reset", on_click=reset, use_container_width=True)

    with right:
        st.markdown("### Top-5 predictions")
        st.caption(
            "Probabilities renormalised within the shown top-5 (legal-mask applied). "
            "Click any row to play that move."
        )
        if hybrid is not None:
            _render_panel("Hybrid (CNN + LSTM)", hybrid_top, vocab, HYBRID_BAR_COLOR, "hybrid")
            st.divider()
        _render_panel("LSTM", lstm_top, vocab, LSTM_BAR_COLOR, "lstm")
        st.divider()
        _render_panel("Trigram (Katz)", ngram_top, vocab, NGRAM_BAR_COLOR, "ngram")


render()
