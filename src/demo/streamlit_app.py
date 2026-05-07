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

import io
import sys
from pathlib import Path

# Streamlit's `streamlit run` puts the script's directory on sys.path, not the
# project root. Inject the project root explicitly so `from src...` imports work.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import chess
import chess.pgn
import chess.svg
import numpy as np
import streamlit as st
import torch
from streamlit_image_coordinates import streamlit_image_coordinates

from src.data.board_encoder import encode_board
from src.data.dataset import DEFAULT_HISTORY_LEN
from src.data.vocab import Vocab
from src.demo.board_image import ArrowSpec, pixel_to_square, render_board_image
from src.models.hybrid import MoveBoardHybrid
from src.models.lstm import MoveLSTM
from src.models.ngram import TrigramKatz
from src.training.evaluate import phase_for_ply


LSTM_PATH = _ROOT / "checkpoints" / "full" / "lstm.pt"
NGRAM_PATH = _ROOT / "checkpoints" / "full" / "ngram.pkl.gz"
HYBRID_PATH = _ROOT / "checkpoints" / "full" / "hybrid.pt"
VOCAB_PATH = _ROOT / "data" / "vocab.json"

K = 5
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
    # Mode + playback state (initialised lazily so older sessions still work)
    if "mode" not in ss:
        ss.mode = "Free play"
    if "playback_moves" not in ss:
        ss.playback_moves = []
        ss.playback_index = 0
        ss.playback_pgn = ""
        ss.playback_error = ""
    # Click-on-board state for free play.
    # ``last_click_pos`` is the (x, y) of the last click we *processed*; the
    # streamlit_image_coordinates widget keeps returning the same coordinates
    # on every rerun until a new physical click happens, so we dedup against
    # this to avoid a select-deselect-select-… loop on every state change.
    if "selected_square" not in ss:
        ss.selected_square = None
    if "last_click_pos" not in ss:
        ss.last_click_pos = None


def _reset_state(start_id: int) -> None:
    ss = st.session_state
    ss.board = chess.Board()
    ss.history_ids = [start_id]
    ss.move_stack_uci = []
    ss.last_move = None
    ss.selected_square = None


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
        ss.selected_square = None


def undo() -> None:
    ss = st.session_state
    if ss.move_stack_uci:
        ss.board.pop()
        ss.history_ids.pop()
        ss.move_stack_uci.pop()
        ss.last_move = ss.board.move_stack[-1] if ss.board.move_stack else None
        ss.selected_square = None


def _legal_move_between(board: chess.Board, from_sq: int, to_sq: int) -> chess.Move | None:
    """Return the legal move from ``from_sq`` to ``to_sq`` if one exists.

    For pawn promotions, auto-promotes to queen (the demo's v1 limitation;
    the other three promotion pieces are extremely rare in real play).
    """
    queen_promo = None
    for mv in board.legal_moves:
        if mv.from_square == from_sq and mv.to_square == to_sq:
            if mv.promotion is None:
                return mv
            if mv.promotion == chess.QUEEN:
                queen_promo = mv
    return queen_promo


def square_clicked(square: int) -> None:
    """Click-on-board state machine: select source, then play to destination."""
    ss = st.session_state
    board: chess.Board = ss.board
    if board.is_game_over():
        return

    if ss.selected_square is None:
        # First click: select a square holding a piece of the side to move.
        piece = board.piece_at(square)
        if piece is not None and piece.color == board.turn:
            ss.selected_square = square
        return

    # Second click.
    if square == ss.selected_square:
        # Click the selected square again -> deselect.
        ss.selected_square = None
        return

    move = _legal_move_between(board, ss.selected_square, square)
    if move is not None:
        play_move(move.uci())
        return

    # Not a legal destination — if the new square holds another own-piece,
    # switch selection to it; otherwise clear selection.
    piece = board.piece_at(square)
    if piece is not None and piece.color == board.turn:
        ss.selected_square = square
    else:
        ss.selected_square = None


def reset() -> None:
    vocab, _, _, _ = load_artifacts()
    _reset_state(vocab.start_id)


def _parse_pgn_to_moves(pgn_text: str) -> list[str] | None:
    """Parse a PGN string into a list of UCI move strings.

    Returns ``None`` if the text isn't valid PGN or contains no mainline moves.
    """
    if not pgn_text.strip():
        return None
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
    except Exception:
        return None
    if game is None:
        return None
    moves = [m.uci() for m in game.mainline_moves()]
    if not moves:
        return None
    return moves


def load_pgn(pgn_text: str) -> None:
    """Parse the textarea contents and prime the playback state."""
    ss = st.session_state
    moves = _parse_pgn_to_moves(pgn_text)
    if moves is None:
        ss.playback_error = (
            "Could not parse PGN — check the format. A minimal example: "
            "`1. e4 e5 2. Nf3 Nc6 *`"
        )
        return
    vocab, _, _, _ = load_artifacts()
    _reset_state(vocab.start_id)
    ss.playback_moves = moves
    ss.playback_index = 0
    ss.playback_pgn = pgn_text
    ss.playback_error = ""


def playback_next() -> None:
    ss = st.session_state
    if 0 <= ss.playback_index < len(ss.playback_moves):
        play_move(ss.playback_moves[ss.playback_index])
        ss.playback_index += 1


def playback_prev() -> None:
    ss = st.session_state
    if ss.move_stack_uci and ss.playback_index > 0:
        undo()
        ss.playback_index -= 1


def playback_reset() -> None:
    """Rewind to position 0 of the loaded PGN without dropping it."""
    vocab, _, _, _ = load_artifacts()
    ss = st.session_state
    saved_moves = list(ss.playback_moves)
    saved_pgn = ss.playback_pgn
    _reset_state(vocab.start_id)
    ss.playback_moves = saved_moves
    ss.playback_index = 0
    ss.playback_pgn = saved_pgn
    ss.playback_error = ""


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
    *,
    actual_next_uci: str | None = None,
    clickable: bool = True,
) -> None:
    """Render one model's top-K panel.

    If ``actual_next_uci`` is given (playback mode), show a small badge after
    the title indicating whether the model's top-1 matched the actually-played
    move at this position. If ``clickable`` is False (playback mode), the move
    rows are read-only — clicking should not advance the game.
    """
    header = f"#### {title}"
    if actual_next_uci is not None and pairs:
        top1_uci = vocab.decode(pairs[0][0])
        if top1_uci == actual_next_uci:
            header += f" &nbsp; ✅ predicted (`{top1_uci}`)"
        else:
            # Did the played move at least appear lower in the top-K?
            played_rank: int | None = None
            for i, (tok, _) in enumerate(pairs):
                if vocab.decode(tok) == actual_next_uci:
                    played_rank = i + 1
                    break
            if played_rank is not None:
                header += (
                    f" &nbsp; 🟡 #{played_rank}/{K} — predicted `{top1_uci}` "
                    f"· played `{actual_next_uci}`"
                )
            else:
                header += (
                    f" &nbsp; ❌ predicted `{top1_uci}` · played `{actual_next_uci}`"
                )
    st.markdown(header, unsafe_allow_html=True)

    if not pairs:
        st.markdown("_no predictions (game over)_")
        return
    norm = _renormalize(pairs)
    for i, (token_id, prob) in enumerate(norm):
        uci = vocab.decode(token_id)
        is_played = (uci == actual_next_uci)
        cols = st.columns([1, 3, 1])
        if clickable:
            cols[0].button(
                uci,
                key=f"{key_prefix}_{i}",
                on_click=play_move,
                args=(uci,),
                use_container_width=True,
            )
        else:
            label = f"**{uci}**" if is_played else f"`{uci}`"
            cols[0].markdown(label)
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

    # Mode toggle at the very top — selecting the same value Streamlit just
    # rendered is a no-op; switching modes triggers a rerun and the relevant
    # branch below renders. We deliberately do *not* clear the board state on
    # mode toggle so a position you set up in one mode is still there in the
    # other (within reason — playback Reset does its own state reset).
    ss.mode = st.radio(
        "Mode",
        options=("Free play", "Game playback"),
        index=0 if ss.mode == "Free play" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )

    board: chess.Board = ss.board

    legal_ids = _legal_token_ids(board, vocab)
    ply = board.ply()
    phase = phase_for_ply(ply)
    gameover = board.is_game_over()

    # In playback mode, the "actually played" move is the one at the current
    # playback index — i.e., the move that comes next in the loaded game.
    # Used by the prediction panels to show ✅/❌ next to each model.
    actual_next_uci: str | None = None
    if ss.mode == "Game playback":
        if 0 <= ss.playback_index < len(ss.playback_moves):
            actual_next_uci = ss.playback_moves[ss.playback_index]

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

    # Multi-coloured top-1 arrows: one per model. Stacked in this order so
    # later (more "trusted") arrows render on top — n-gram first, then LSTM,
    # then hybrid — when models agree, you mostly see the hybrid colour.
    arrows: list[ArrowSpec] = []
    for top, color in (
        (ngram_top, NGRAM_BAR_COLOR),
        (lstm_top, LSTM_BAR_COLOR),
        (hybrid_top, HYBRID_BAR_COLOR),
    ):
        if not top:
            continue
        try:
            mv = chess.Move.from_uci(vocab.decode(top[0][0]))
            arrows.append(ArrowSpec(mv.from_square, mv.to_square, color))
        except ValueError:
            pass

    legal_dests: set[int] = set()
    if ss.selected_square is not None and ss.mode == "Free play":
        legal_dests = {
            mv.to_square for mv in board.legal_moves
            if mv.from_square == ss.selected_square
        }

    board_image = render_board_image(
        board,
        last_move=ss.last_move,
        arrows=arrows if not gameover else (),
        selected_square=ss.selected_square if ss.mode == "Free play" else None,
        legal_destinations=legal_dests,
        size=480,
    )

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

        # Free play: click capture. Playback: read-only image.
        if ss.mode == "Free play":
            # Stable key — remounting per ply caused a full board flash on
            # every move. Stale clicks are filtered by ``last_click_pos``
            # equality below instead.
            click = streamlit_image_coordinates(
                board_image,
                key="board_clicker",
            )
            if click is not None:
                pos = (click["x"], click["y"])
                if pos != ss.last_click_pos:
                    ss.last_click_pos = pos
                    square = pixel_to_square(click["x"], click["y"], size=480)
                    if square is not None:
                        square_clicked(square)
                        st.rerun()
        else:
            st.image(board_image, width=480)

        st.caption(
            "Arrows: amber = Hybrid top-1 · blue = LSTM top-1 · violet = Trigram top-1. "
            "When models agree the arrows overlap; when they disagree you see all of them."
        )

        if ss.mode == "Free play":
            if not gameover:
                if ss.selected_square is not None:
                    sel_name = chess.square_name(ss.selected_square)
                    st.caption(
                        f"**{sel_name}** selected — click a green-dotted destination to "
                        f"play, click {sel_name} again to deselect, or pick from a "
                        f"prediction panel on the right."
                    )
                else:
                    st.caption(
                        "Click a piece of the side to move to select it, then click a "
                        "destination — or click any row in the prediction panels on the "
                        "right to play that move directly."
                    )

            c1, c2 = st.columns(2)
            c1.button(
                "Undo",
                on_click=undo,
                disabled=(not ss.move_stack_uci),
                use_container_width=True,
            )
            c2.button("Reset", on_click=reset, use_container_width=True)
        else:
            # Game playback: paste a PGN, step through it, see model predictions
            # vs the move actually played at every position.
            pgn_input = st.text_area(
                "Paste a PGN game",
                value=ss.playback_pgn,
                height=140,
                key="pgn_input",
                placeholder=(
                    "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O *"
                ),
            )

            cload, cprev, cnext, creset = st.columns([2, 1, 1, 1])
            cload.button(
                "Load game",
                on_click=load_pgn,
                args=(pgn_input,),
                use_container_width=True,
                type="primary",
            )
            has_loaded = bool(ss.playback_moves)
            cprev.button(
                "← Prev",
                on_click=playback_prev,
                disabled=(not has_loaded or ss.playback_index <= 0),
                use_container_width=True,
            )
            cnext.button(
                "Next →",
                on_click=playback_next,
                disabled=(
                    not has_loaded
                    or ss.playback_index >= len(ss.playback_moves)
                    or gameover
                ),
                use_container_width=True,
            )
            creset.button(
                "Reset",
                on_click=playback_reset,
                disabled=(not has_loaded),
                use_container_width=True,
            )

            if ss.playback_error:
                st.error(ss.playback_error)
            elif has_loaded:
                total = len(ss.playback_moves)
                st.caption(
                    f"Move **{ss.playback_index}** of **{total}** "
                    f"({total - ss.playback_index} remaining)."
                )

    clickable_panels = (ss.mode == "Free play")
    with right:
        st.markdown("### Top-5 predictions")
        if ss.mode == "Free play":
            st.caption(
                "Probabilities renormalised within the shown top-5 (legal-mask applied). "
                "Click any row to play that move."
            )
        else:
            st.caption(
                "Probabilities renormalised within the shown top-5 (legal-mask applied). "
                "Next to each model name: ✅ = top-1 matched the played move · "
                "🟡 #r/5 = played move was rank r in the top-5 · "
                "❌ = played move was outside the top-5."
            )
        if hybrid is not None:
            _render_panel(
                "Hybrid (CNN + LSTM)", hybrid_top, vocab, HYBRID_BAR_COLOR, "hybrid",
                actual_next_uci=actual_next_uci, clickable=clickable_panels,
            )
            st.divider()
        _render_panel(
            "LSTM", lstm_top, vocab, LSTM_BAR_COLOR, "lstm",
            actual_next_uci=actual_next_uci, clickable=clickable_panels,
        )
        st.divider()
        _render_panel(
            "Trigram (Katz)", ngram_top, vocab, NGRAM_BAR_COLOR, "ngram",
            actual_next_uci=actual_next_uci, clickable=clickable_panels,
        )


render()
