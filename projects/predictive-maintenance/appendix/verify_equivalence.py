"""Prove that the three Stage 1 editions are the same program.

Checks performed
----------------
1. AST check   : the compact edition parses to exactly the same syntax tree
                 as the original pasted script (only formatting changed).
2. Runtime check: all three editions are executed against the same synthetic
                 "dirty" CSV and their console output, CSV/JSON artefacts and
                 figure files are compared.

Run:  python verify_equivalence.py        (needs pandas, numpy, matplotlib)
"""
from __future__ import annotations

import ast
import contextlib
import io
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EDITIONS = {
    "original": ROOT / "stage1_audit_eda_original.py",
    "compact": ROOT / "stage1_audit_eda.py",
    "refactored": ROOT / "stage1_audit_eda_refactored.py",
}


# --------------------------------------------------------------------------
# Synthetic "dirty" extract with the same schema as NLNG_Dirty_Data.csv
# --------------------------------------------------------------------------
def make_dirty_csv(path: Path, n=800, seed=7):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "Equipment_ID": rng.choice([f"EQ-{i:03d}" for i in range(1, 9)], n),
        "Train": rng.choice(["Train 1", "Train 2", "Train 3"], n),
        "Load_Factor": np.clip(rng.normal(0.7, 0.25, n), -0.2, 1.4),
        "Wear_Level": np.clip(rng.normal(0.4, 0.2, n), -0.1, 1.3),
        "Lubrication_Health_Index": rng.normal(70, 25, n),
        "Oil_Pressure": rng.normal(45, 12, n),
        "Oil_Particles_PPM": rng.gamma(2, 8, n),
        "Overall_Vibration": rng.gamma(3, 1.2, n),
        "Vibration": rng.gamma(2, 0.9, n),
        "RPM": rng.normal(3000, 400, n),
        "Hours_Since_Maint": rng.integers(0, 4000, n).astype(float),
        "Cumulative_OP_Hours": rng.integers(0, 90000, n).astype(float),
        "Failure_Within_24h": rng.integers(0, 2, n),
        "Failure_Within_72h": rng.integers(0, 2, n),
        "Failure_Within_7d": rng.integers(0, 2, n),
        "RUL_Days": rng.gamma(4, 12, n),
        "RUL_Censored": rng.integers(0, 2, n),
        "Failure_Rate": rng.random(n),
    })
    # realistic dirt: missing values, duplicates, bad timestamps, negatives
    for col in ["Oil_Pressure", "Wear_Level", "RUL_Days", "Vibration"]:
        df.loc[rng.choice(n, 40, replace=False), col] = np.nan
    df.loc[rng.choice(n, 15, replace=False), "Timestamp"] = "not-a-date"
    df = pd.concat([df, df.iloc[:25]], ignore_index=True)  # exact duplicates
    df.loc[rng.choice(len(df), 12, replace=False), "Oil_Pressure"] *= -1
    df.to_csv(path, index=False)
    return path


def run_edition(script: Path, raw_csv: Path, workdir: Path):
    """Execute one edition with its Windows paths redirected to ``workdir``."""
    src = script.read_text(encoding="utf-8")
    src = re.sub(r'raw_data_path = \(.*?\)',
                 f'raw_data_path = r"{raw_csv}"', src, flags=re.S)
    src = re.sub(r'corrected_root = \(.*?\)',
                 f'corrected_root = r"{workdir}"', src, flags=re.S)
    if "corrected_root = " not in src:
        raise SystemExit(f"could not patch paths in {script}")

    out = io.StringIO()
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            exec(compile(src, str(script), "exec"), {"__name__": "__main__"})
    finally:
        os.chdir(cwd)
    return out.getvalue(), sorted(
        p.name for p in (workdir / "stage_1_audit_eda").glob("*"))


def artefacts(workdir: Path):
    folder = workdir / "stage_1_audit_eda"
    files = {}
    for p in sorted(folder.glob("*")):
        files[p.name] = p.read_bytes() if p.suffix != ".png" else b"<png>"
    return files


def main():
    ok = True

    # -- 1. AST equivalence of the compact edition -----------------------
    original_ast = ast.dump(ast.parse(EDITIONS["original"].read_text()))
    compact_ast = ast.dump(ast.parse(EDITIONS["compact"].read_text()))
    print(f"[AST] compact == original : {original_ast == compact_ast}")
    ok &= original_ast == compact_ast

    # -- 2. Runtime equivalence of all three editions --------------------
    base = Path(tempfile.mkdtemp(prefix="stage1_"))
    csv = make_dirty_csv(base / "NLNG_Dirty_Data.csv")
    def normalise(text, workdir):
        # each edition runs in its own temp folder; blank the path out
        return text.replace(str(workdir), "<workdir>").strip()

    results = {}
    for name, script in EDITIONS.items():
        workdir = base / name
        (workdir / "stage_1_audit_eda").mkdir(parents=True, exist_ok=True)
        stdout, files = run_edition(script, csv, workdir)
        results[name] = (normalise(stdout, workdir), artefacts(workdir), files)
        print(f"[run] {name:<11}: {len(stdout.splitlines())} stdout lines, "
              f"{len(files)} artefacts")

    ref = results["original"]
    for name in ("compact", "refactored"):
        stdout, files, names = results[name]
        same_out = stdout == ref[0]
        same_files = files == ref[1]
        print(f"[cmp] {name:<11} stdout identical: {same_out} | "
              f"artefacts identical: {same_files}")
        if not same_out:
            import difflib
            diff = list(difflib.unified_diff(ref[0].splitlines(),
                                             stdout.splitlines(),
                                             "original", name, lineterm=""))[:20]
            print("\n".join(diff))
        if not same_files:
            print("  original :", sorted(ref[1]))
            print(f"  {name}:", sorted(files))
        ok &= same_out and same_files

    shutil.rmtree(base, ignore_errors=True)
    print("\nRESULT:", "ALL EDITIONS EQUIVALENT" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
