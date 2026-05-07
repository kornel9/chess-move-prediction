"""Pure-PIL chessboard renderer for the Streamlit demo.

We render the board as a raster ``PIL.Image`` (rather than an SVG) so that
``streamlit_image_coordinates`` can capture click coordinates on it and feed
them back to the click-on-board state machine. The same image carries all
the visual cues: piece glyphs, last-move highlight, selected-square +
legal-destination indicators, and one multi-coloured arrow per top-1
model prediction.

A 480 px image with 8×8 = 60 px squares is the canonical size used by
the demo; click-to-square math depends on that grid spacing being exact,
so rendering with ``coordinates=False`` and ``borders=False`` (no padding
around the playable area) is the only correct mode.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import chess
from PIL import Image, ImageDraw, ImageFont


LIGHT_SQUARE = "#f0d9b5"
DARK_SQUARE = "#b58863"
LAST_MOVE_HIGHLIGHT = (247, 236, 91, 130)        # translucent yellow
SELECTED_HIGHLIGHT = (255, 165, 0, 150)          # translucent orange
LEGAL_DEST_DOT = (60, 130, 60, 170)              # translucent dark green

# Filled unicode chess glyphs for both colours so we can paint white pieces
# in white-with-black-outline and black pieces in black for crisp contrast on
# both light and dark squares (the outlined "white" glyphs U+2654-U+2659
# would render nearly invisibly against light squares).
_PIECE_GLYPH = "♚♛♜♝♞♟"   # K, Q, R, B, N, P (filled)
_PIECE_INDEX = {
    chess.KING: 0, chess.QUEEN: 1, chess.ROOK: 2,
    chess.BISHOP: 3, chess.KNIGHT: 4, chess.PAWN: 5,
}


@dataclass
class ArrowSpec:
    from_sq: int
    to_sq: int
    color: str
    """Hex string '#rrggbb'. Rendered with translucent fill."""


def _load_chess_font(size: int) -> ImageFont.ImageFont:
    """Try a list of system fonts that contain the chess glyphs.

    Falls back to PIL's default bitmap font (which on most platforms also
    has the U+265x range) so the demo never crashes for lack of a font.
    """
    candidates = [
        "C:/Windows/Fonts/seguisym.ttf",                              # Segoe UI Symbol — Windows
        "C:/Windows/Fonts/arial.ttf",                                 # Arial — wider Windows fallback
        "/System/Library/Fonts/Apple Symbols.ttf",                    # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",            # Linux
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _piece_glyph(piece_type: int) -> str:
    return _PIECE_GLYPH[_PIECE_INDEX[piece_type]]


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


def square_to_pixel_box(square: int, cell: int) -> tuple[int, int, int, int]:
    """Top-left + bottom-right pixel corners of a square, with white at bottom."""
    f = chess.square_file(square)
    r = chess.square_rank(square)
    x0 = f * cell
    y0 = (7 - r) * cell
    return x0, y0, x0 + cell, y0 + cell


def pixel_to_square(x: int, y: int, size: int = 480) -> int | None:
    """Translate a click on the rendered image into a chess square index.

    Returns ``None`` if the click is out of the playable area. Assumes white
    is at the bottom of the board (the rendering default).
    """
    cell = size // 8
    if x < 0 or y < 0 or x >= 8 * cell or y >= 8 * cell:
        return None
    f = x // cell
    r = 7 - (y // cell)
    return chess.square(f, r)


def _draw_arrow(
    img: Image.Image,
    from_sq: int,
    to_sq: int,
    color: str,
    cell: int,
    offset_units: float = 0.0,
) -> None:
    """Draw a single arrow from ``from_sq`` to ``to_sq`` in ``color``.

    ``offset_units`` shifts the arrow perpendicularly to its direction by
    ``offset_units * cell // 4`` pixels. We use this to fan out arrows that
    share the same (from, to) pair so all model colours stay visible when
    multiple models agree on the same top-1 move.
    """
    if from_sq == to_sq:
        return

    f0 = chess.square_file(from_sq)
    r0 = chess.square_rank(from_sq)
    f1 = chess.square_file(to_sq)
    r1 = chess.square_rank(to_sq)

    x0 = f0 * cell + cell // 2
    y0 = (7 - r0) * cell + cell // 2
    x1 = f1 * cell + cell // 2
    y1 = (7 - r1) * cell + cell // 2

    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length        # unit direction
    px, py = -uy, ux                          # perpendicular

    offset_px = offset_units * (cell // 4)
    x0 += px * offset_px
    y0 += py * offset_px
    x1 += px * offset_px
    y1 += py * offset_px

    head_len = max(cell // 3, 18)
    head_width = max(cell // 4, 14)
    shaft_width = max(cell // 8, 6)

    # Shorten the shaft so the arrowhead has somewhere to sit.
    sx, sy = x1 - ux * head_len * 0.7, y1 - uy * head_len * 0.7

    rgba = _hex_to_rgba(color, alpha=230)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    drw = ImageDraw.Draw(overlay)

    # Shaft
    drw.line([(x0, y0), (sx, sy)], fill=rgba, width=shaft_width)

    # Arrowhead triangle
    base_x, base_y = x1 - ux * head_len, y1 - uy * head_len
    head = [
        (x1, y1),
        (base_x + px * head_width, base_y + py * head_width),
        (base_x - px * head_width, base_y - py * head_width),
    ]
    drw.polygon(head, fill=rgba)

    img.alpha_composite(overlay)


def render_board_image(
    board: chess.Board,
    *,
    last_move: chess.Move | None = None,
    arrows: list[ArrowSpec] | tuple[ArrowSpec, ...] = (),
    selected_square: int | None = None,
    legal_destinations: set[int] | tuple[int, ...] = (),
    size: int = 480,
) -> Image.Image:
    """Render the board as an RGB PIL image of side ``size``.

    Pieces are filled glyphs in the style of standard chess fonts; white
    pieces are drawn in white with a thin black outline so they remain
    crisp on both light and dark squares.
    """
    cell = size // 8
    img = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    # Square colours
    for sq in chess.SQUARES:
        x0, y0, x1, y1 = square_to_pixel_box(sq, cell)
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        fill = LIGHT_SQUARE if (f + r) % 2 == 1 else DARK_SQUARE
        draw.rectangle([x0, y0, x1, y1], fill=fill)

    # Last-move highlight (both squares)
    if last_move is not None:
        for sq in (last_move.from_square, last_move.to_square):
            x0, y0, x1, y1 = square_to_pixel_box(sq, cell)
            draw.rectangle([x0, y0, x1, y1], fill=LAST_MOVE_HIGHLIGHT)

    # Selected source + legal destination dots
    if selected_square is not None:
        x0, y0, x1, y1 = square_to_pixel_box(selected_square, cell)
        draw.rectangle([x0, y0, x1, y1], fill=SELECTED_HIGHLIGHT)
        for dest in legal_destinations:
            cx_dest = chess.square_file(dest) * cell + cell // 2
            cy_dest = (7 - chess.square_rank(dest)) * cell + cell // 2
            radius = cell // 7
            draw.ellipse(
                [cx_dest - radius, cy_dest - radius, cx_dest + radius, cy_dest + radius],
                fill=LEGAL_DEST_DOT,
            )

    # Pieces
    font = _load_chess_font(int(cell * 0.78))
    for sq, piece in board.piece_map().items():
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        cx = f * cell + cell // 2
        cy = (7 - r) * cell + cell // 2
        glyph = _piece_glyph(piece.piece_type)
        if piece.color == chess.WHITE:
            draw.text(
                (cx, cy), glyph, font=font, fill="#ffffff",
                anchor="mm", stroke_width=2, stroke_fill="#000000",
            )
        else:
            draw.text((cx, cy), glyph, font=font, fill="#000000", anchor="mm")

    # Arrows go on top of everything — they're a visual layer about predictions.
    # Group by (from_sq, to_sq) so arrows that fully overlap (models agree on
    # the same move) get fanned out perpendicular to the move direction. This
    # is what keeps every model colour visible in the common case where the
    # n-gram, LSTM, and hybrid all pick the same top-1 move.
    groups: dict[tuple[int, int], list[ArrowSpec]] = {}
    for arrow in arrows:
        groups.setdefault((arrow.from_sq, arrow.to_sq), []).append(arrow)
    for group_arrows in groups.values():
        n = len(group_arrows)
        # Symmetric offsets centred on 0: 1 -> [0]; 2 -> [-0.5, 0.5]; 3 -> [-1, 0, 1].
        offsets = [i - (n - 1) / 2 for i in range(n)]
        for arrow, offset in zip(group_arrows, offsets):
            _draw_arrow(img, arrow.from_sq, arrow.to_sq, arrow.color, cell, offset_units=offset)

    return img.convert("RGB")
