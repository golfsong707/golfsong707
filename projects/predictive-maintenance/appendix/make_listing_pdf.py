"""Render a Python source file as a print-ready A4 code listing.

The layout is two columns per page and the type size is auto-fitted so the
listing fits in exactly ``--pages`` A4 pages (default 2). Syntax highlighting
is applied with the standard-library ``tokenize`` module, so no Pygments
dependency is needed.

Usage
-----
    pip install reportlab
    python make_listing_pdf.py --source ../stage1_audit_eda.py \
        --out stage1_audit_eda_A4.pdf --pages 2

Only the layout is computed here; the source file is never modified.
"""
from __future__ import annotations

import argparse
import io
import keyword
import tokenize
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# --------------------------------------------------------------------------
# Palette (print-safe: every colour survives a black-and-white laser printer)
# --------------------------------------------------------------------------
C_TEXT = "#111111"
C_COMMENT = "#6e7781"
C_STRING = "#0a7d32"
C_NUMBER = "#953800"
C_KEYWORD = "#0b4f9e"
C_BUILTIN = "#6f42c1"
C_PUNCT = "#444444"

BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
    "len", "list", "max", "min", "open", "print", "range", "round",
    "set", "sorted", "str", "sum", "tuple", "zip", "isinstance",
}

MM = 72.0 / 25.4  # millimetre -> point

# page geometry (points) - tight but printer-safe
MARGIN_L = 10 * MM
MARGIN_R = 10 * MM
MARGIN_T = 12 * MM
MARGIN_B = 12 * MM
GUTTER = 6 * MM
LEADING = 1.12          # line spacing as a multiple of the type size


# --------------------------------------------------------------------------
# 1. Colourise the source, character by character
# --------------------------------------------------------------------------
def colour_for(ttype, text):
    if ttype == tokenize.COMMENT:
        return C_COMMENT
    if ttype == tokenize.STRING:
        return C_STRING
    if ttype == tokenize.NUMBER:
        return C_NUMBER
    if ttype == tokenize.OP:
        return C_PUNCT
    if ttype == tokenize.NAME:
        if keyword.iskeyword(text) or keyword.issoftkeyword(text):
            return C_KEYWORD
        if text in BUILTINS:
            return C_BUILTIN
    return C_TEXT


def colourise(source: str):
    """Return one list of (char, colour) pairs per physical source line."""
    lines = source.split("\n")
    grid = [[C_TEXT] * len(line) for line in lines]

    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        ttype, text, (srow, scol), (erow, ecol), _ = tok
        if ttype in (tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE,
                     tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        colour = colour_for(ttype, text)
        for offset, segment in enumerate(text.split("\n")):
            row = srow - 1 + offset
            if not segment or row >= len(grid):
                continue
            start = scol if offset == 0 else 0
            for col in range(start, min(start + len(segment), len(grid[row]))):
                grid[row][col] = colour

    return [[(ch, grid[i][j]) for j, ch in enumerate(line)]
            for i, line in enumerate(lines)]


# --------------------------------------------------------------------------
# 2. Wrap to the column width, keeping the colour of every character
# --------------------------------------------------------------------------
def wrap_line(chars, width, cont_indent=4):
    """Split one logical line into (indent, chars) pieces."""
    if not chars:
        return [(0, [])]
    pieces, first, i = [], True, 0
    while True:
        room = width if first else width - cont_indent
        if i + room >= len(chars):
            pieces.append((0 if first else cont_indent, chars[i:]))
            break
        cut = i + room
        while cut > i and chars[cut - 1][0] not in " \t":
            cut -= 1
        if cut == i:                       # no break opportunity: hard break
            cut = i + room
        pieces.append((0 if first else cont_indent, chars[i:cut]))
        i = cut
        while i < len(chars) and chars[i][0] == " ":
            i += 1
        first = False
    return pieces


def layout(coloured_lines, width):
    """Wrap every line; each piece carries its originating line number."""
    pieces = []
    for number, chars in enumerate(coloured_lines, start=1):
        wrapped = wrap_line(chars, width)
        pieces.append((wrapped[0][0], wrapped[0][1], number))
        for indent, chunk in wrapped[1:]:
            pieces.append((indent, chunk, None))
    return pieces


def column_geometry(page_h):
    """Return (column width, text height) in points."""
    col_w = (A4[0] - MARGIN_L - MARGIN_R - GUTTER) / 2
    text_h = page_h - MARGIN_T - MARGIN_B
    return col_w, text_h


# --------------------------------------------------------------------------
# 3. Auto-fit the type size to the requested number of A4 pages
# --------------------------------------------------------------------------
def fit(coloured_lines, pages, line_numbers, leading_factor=LEADING):
    """Largest Courier size whose layout fits in ``pages`` A4 pages."""
    col_w, text_h = column_geometry(A4[1])
    sizes = [round(8.0 - 0.05 * i, 2) for i in range(56)]  # 8.00 -> 5.25 pt
    for size in sizes:
        cw = 0.6 * size
        reserve = (5 if line_numbers else 0) * cw + 2 * MM
        width = max(24, int((col_w - reserve) / cw))
        pieces = layout(coloured_lines, width)
        per_col = int(text_h // (size * leading_factor))
        if per_col * 2 * pages >= len(pieces):
            return size, pieces, per_col, col_w, width
    size = sizes[-1]
    cw = 0.6 * size
    width = max(24, int((col_w - (5 if line_numbers else 0) * cw - 2 * MM) / cw))
    return size, layout(coloured_lines, width), int(text_h // (size * leading_factor)), col_w, width


# --------------------------------------------------------------------------
# 4. Draw
# --------------------------------------------------------------------------
def draw(pdf_path, coloured_lines, title, subtitle, pages, line_numbers,
         leading_factor=LEADING):
    page_w, page_h = A4
    size, pieces, per_col, col_w, width = fit(coloured_lines, pages,
                                              line_numbers, leading_factor)
    leading = size * leading_factor
    cw = 0.6 * size
    n_cols = 2 * pages
    per_col = min(-(-len(pieces) // n_cols), int((page_h - MARGIN_T - MARGIN_B) // leading))

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    c.setTitle(title)

    idx = 0
    for page in range(pages):
        # ---- page furniture -------------------------------------------
        c.setFont("Helvetica-Bold", 7.5)
        c.setFillColor("#111111")
        c.drawString(MARGIN_L, page_h - MARGIN_T + 6 * MM, title)
        c.setFont("Helvetica", 6.5)
        c.setFillColor("#666666")
        c.drawRightString(page_w - MARGIN_R, page_h - MARGIN_T + 6 * MM, subtitle)
        c.setStrokeColor("#cccccc")
        c.setLineWidth(0.4)
        c.line(MARGIN_L, page_h - MARGIN_T + 3 * MM,
               page_w - MARGIN_R, page_h - MARGIN_T + 3 * MM)

        c.setFont("Helvetica", 6.5)
        c.drawCentredString(page_w / 2, MARGIN_B - 6 * MM,
                            f"Page {page + 1} of {pages}")
        c.line(MARGIN_L, MARGIN_B - 3 * MM, page_w - MARGIN_R, MARGIN_B - 3 * MM)

        mid = MARGIN_L + col_w + GUTTER / 2
        c.setStrokeColor("#e2e2e2")
        c.line(mid, page_h - MARGIN_T, mid, MARGIN_B)

        # ---- columns ----------------------------------------------------
        for col in range(2):
            x0 = MARGIN_L + col * (col_w + GUTTER)
            y = page_h - MARGIN_T - size
            for _ in range(per_col):
                if idx >= len(pieces) or y < MARGIN_B:
                    break
                indent, chars, number = pieces[idx]
                if line_numbers and number is not None:
                    c.setFillColor("#9aa0a6")
                    c.setFont("Courier", size)
                    c.drawRightString(x0 + 3.6 * cw, y, f"{number:>3}")
                dx = (5 * cw) if line_numbers else 0
                runs, pos = [], indent
                for ch, colour in chars:
                    if runs and runs[-1][2] == colour:
                        runs[-1][1] += ch
                    else:
                        runs.append([pos, ch, colour])
                    pos += 1
                for start, text, colour in runs:
                    c.setFillColor(colour)
                    c.setFont("Courier", size)
                    c.drawString(x0 + dx + start * cw, y, text)
                idx += 1
                y -= leading
        c.showPage()
    c.save()
    if idx < len(pieces):  # nothing may be dropped from the listing
        raise SystemExit(f"ERROR: {len(pieces) - idx} lines did not fit "
                         f"in {pages} page(s); try --pages {pages + 1}")
    return size, len(pieces), per_col, col_w, width


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(here.parent / "stage1_audit_eda.py"))
    ap.add_argument("--out", default=str(here / "stage1_audit_eda_A4.pdf"))
    ap.add_argument("--pages", type=int, default=2)
    ap.add_argument("--title",
                    default="Appendix — Corrected Pipeline 2.0 · Stage 1: "
                            "Raw Data Audit and EDA")
    ap.add_argument("--subtitle", default="stage1_audit_eda.py")
    ap.add_argument("--line-numbers", action="store_true",
                    help="print source line numbers (costs ~5 chars per column)")
    ap.add_argument("--leading", type=float, default=1.12,
                    help="line spacing as a multiple of the type size")
    args = ap.parse_args()

    source = Path(args.source).read_text(encoding="utf-8")
    coloured = colourise(source)
    size, n_pieces, per_col, col_w, width = draw(
        Path(args.out), coloured, args.title, args.subtitle,
        args.pages, args.line_numbers, args.leading)

    print(f"source       : {args.source}")
    print(f"logical lines: {len(coloured)}")
    print(f"rendered     : {n_pieces} lines across {args.pages * 2} columns "
          f"(<= {per_col} per column)")
    print(f"column       : {width} chars, {col_w / MM:.1f} mm wide")
    print(f"type size    : {size:.2f} pt Courier")
    print(f"written      : {args.out}")


if __name__ == "__main__":
    main()
