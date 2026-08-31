# NLNG Predictive Analytics — Stage 1, print-ready listings

**Corrected Pipeline 2.0 · Stage 1 — Raw Data Audit and Exploratory Data
Analysis**, formatted to fit **exactly two A4 pages** as a code appendix.

Two print editions are provided. Both are the *same program* — see
[Verification](#verification) — they differ only in how densely the source is
laid out, which decides how large the printed type can be.

| Edition | Source | Printed listing | Type size | Notes |
| ------- | ------ | --------------- | --------- | ----- |
| Faithful | [`stage1_audit_eda.py`](stage1_audit_eda.py) | [`appendix/stage1_audit_eda_A4.pdf`](appendix/stage1_audit_eda_A4.pdf) | 6.85 pt | every original statement, line for line |
| Large type | [`stage1_audit_eda_refactored.py`](stage1_audit_eda_refactored.py) | [`appendix/stage1_audit_eda_A4_largetype.pdf`](appendix/stage1_audit_eda_A4_largetype.pdf) | 7.15 pt | repeated banner / save / plot boilerplate folded into helpers |

The original script as supplied is kept untouched in
[`stage1_audit_eda_original.py`](stage1_audit_eda_original.py) (517 lines) so
you can always diff against it.

## Layout

* A4 portrait, two columns per page, 2 pages
* 10 mm side margins, 12 mm top/bottom, 6 mm gutter — printer-safe
* Courier, auto-fitted: the renderer searches for the **largest** type size
  that still fits the requested page count, so nothing is ever silently
  dropped (it raises instead)
* Syntax highlighting via the standard-library `tokenize` module (no Pygments)
* Running header and "Page n of 2" footer

## Rebuilding the PDFs

```bash
pip install -r appendix/requirements.txt

python appendix/make_listing_pdf.py \
    --source stage1_audit_eda.py \
    --out appendix/stage1_audit_eda_A4.pdf \
    --pages 2 --leading 1.08 \
    --title "Appendix — Corrected Pipeline 2.0 · Stage 1: Raw Data Audit and EDA"

python appendix/make_listing_pdf.py \
    --source stage1_audit_eda_refactored.py \
    --out appendix/stage1_audit_eda_A4_largetype.pdf \
    --pages 2 --leading 1.10
```

Useful switches: `--pages 3` (more room, bigger type), `--line-numbers`
(adds source line numbers, costs ~5 characters per column), `--leading`
(line spacing, default 1.12).

## Verification

Two independent checks ship with the folder, so "it fits" never means
"it changed":

```bash
python appendix/verify_equivalence.py   # same program?
python appendix/check_listing.py appendix/stage1_audit_eda_A4.pdf \
                                stage1_audit_eda.py 2   # same characters?
```

`verify_equivalence.py`

1. compares the **AST** of the faithful edition with the original — identical,
   i.e. only whitespace, comments and line breaks changed;
2. runs all three editions against the same synthetic dirty CSV (missing
   values, duplicates, bad timestamps, out-of-range sensors) and compares
   their console output **and** every CSV/JSON/PNG artefact — identical.

`check_listing.py` re-extracts the text from the PDF in column reading order
and compares it character by character with the source file it was rendered
from, and confirms the page count and that nothing spills outside the margins.

Both currently report: **ALL EDITIONS EQUIVALENT** and **LISTING COMPLETE**
(2 pages, 0 characters lost, 0 items outside margins).

## Running the pipeline

Point `raw_data_path` / `corrected_root` at your own folders (section 1) and
run either source file. Stage 1 writes 13 audit tables, 9 EDA figures and
`stage1_summary.json` into `.../04_corrected_pipeline/stage_1_audit_eda`.
