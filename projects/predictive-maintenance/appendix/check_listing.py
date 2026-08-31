"""Check a generated code listing: page count, margins and completeness.

The completeness test is the important one: it re-extracts the text from the
PDF (in column reading order) and compares it, character by character, with
the source file it was rendered from -- so a listing can never silently drop
part of the code.

Run:  python check_listing.py stage1_audit_eda_A4.pdf ../stage1_audit_eda.py
      (needs pymupdf: pip install pymupdf)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import pymupdf
except ImportError:  # pragma: no cover
    sys.exit("pymupdf is required for this check:  pip install pymupdf")

MM = 72.0 / 25.4
A4_W, A4_H = 595.28, 841.89
MARGIN_L = MARGIN_R = 10 * MM
MARGIN_T = MARGIN_B = 12 * MM


def extract(pdf_path):
    """Return (page_count, code text in reading order, out-of-margin items)."""
    doc = pymupdf.open(str(pdf_path))
    mid = (MARGIN_L + (A4_W - MARGIN_R)) / 2
    chunks, problems = [], []
    for page_no, page in enumerate(doc, start=1):
        words = page.get_text("words")  # x0, y0, x1, y1, word, block, line, word_no
        furniture, code = [], []
        for x0, y0, x1, y1, word, *_ in words:
            if y1 < MARGIN_T or y0 > A4_H - MARGIN_B:
                furniture.append(word)          # header / footer
                continue
            if x0 < MARGIN_L - 1 or x1 > A4_W - MARGIN_R + 1:
                problems.append((page_no, round(x0, 1), round(x1, 1), word))
            code.append((0 if x0 < mid else 1, round(y0, 1), x0, word))
        code.sort(key=lambda w: (w[0], w[1], w[2]))
        chunks.extend(w[3] for w in code)
    return doc.page_count, "".join(chunks), problems


def squash(text):
    """Drop all whitespace so only the character sequence matters."""
    return re.sub(r"\s+", "", text)


def main():
    pdf_path, source_path = Path(sys.argv[1]), Path(sys.argv[2])
    expected_pages = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    pages, got, problems = extract(pdf_path)
    want = squash(source_path.read_text(encoding="utf-8"))

    print(f"pdf          : {pdf_path}")
    print(f"source       : {source_path}")
    print(f"pages        : {pages} (expected {expected_pages})")
    print(f"characters   : pdf {len(got):,} vs source {len(want):,} "
          f"(whitespace removed)")
    print(f"outside margins: {len(problems)}"
          + (f" -> {problems[:3]}" if problems else ""))

    ok = pages == expected_pages and not problems and squash(got) == want
    if squash(got) != want:
        for i, (a, b) in enumerate(zip(squash(got), want)):
            if a != b:
                print(f"first divergence at character {i}: "
                      f"pdf {a!r} vs source {b!r}")
                break
    print("\nRESULT:", "LISTING COMPLETE" if ok else "CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
