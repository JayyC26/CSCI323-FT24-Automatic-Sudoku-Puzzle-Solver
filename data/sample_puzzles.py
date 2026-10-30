"""Build the shared puzzle set from the Kaggle "3 million Sudoku puzzles with ratings".

Dataset: https://www.kaggle.com/datasets/radcliffe/3-million-sudoku-puzzles-with-ratings
Download sudoku-3m.csv from Kaggle and put it in data/raw/ (that folder is gitignored,
the full file is too big to commit). Then run, from the repo root:

    python data/sample_puzzles.py

This reads the big file in chunks, splits puzzles into difficulty levels using the
dataset's own "difficulty" rating, samples the same number from each level, and
writes data/puzzles.csv. Everyone benchmarks on that one file, so commit it.

Options:
    --per-level 150          puzzles per level (default 150)
    --cuts 1.0 3.0           difficulty cut points: easy < 1.0 <= medium < 3.0 <= hard
    --seed 323               random seed, so the sample is reproducible
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_IN = HERE / "raw" / "sudoku-3m.csv"
DEFAULT_OUT = HERE / "puzzles.csv"
REQUIRED = ["id", "puzzle", "solution", "clues", "difficulty"]
LEVELS = ["easy", "medium", "hard"]


def label(difficulty: pd.Series, cuts: tuple[float, float]) -> pd.Series:
    low, high = cuts
    return pd.cut(
        difficulty,
        bins=[float("-inf"), low, high, float("inf")],
        labels=LEVELS,
        right=False,
    ).astype(str)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--per-level", type=int, default=150)
    ap.add_argument("--cuts", type=float, nargs=2, default=(1.0, 3.0))
    ap.add_argument("--seed", type=int, default=323)
    ap.add_argument("--chunksize", type=int, default=500_000)
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Can't find {args.input}\n"
            "Download sudoku-3m.csv from Kaggle and place it in data/raw/."
        )

    header = pd.read_csv(args.input, nrows=0).columns.tolist()
    missing = [c for c in REQUIRED if c not in header]
    if missing:
        raise SystemExit(f"Unexpected CSV format. Missing columns {missing}; found {header}")

    # Pass 1: take a generous random slice of each level from every chunk,
    # so we never hold all 3M rows in memory at once.
    keep = []
    totals = {lvl: 0 for lvl in LEVELS}
    for i, chunk in enumerate(pd.read_csv(args.input, usecols=REQUIRED, chunksize=args.chunksize)):
        chunk["level"] = label(chunk["difficulty"], tuple(args.cuts))
        for lvl, group in chunk.groupby("level"):
            totals[lvl] += len(group)
            n = min(len(group), args.per_level)
            keep.append(group.sample(n=n, random_state=args.seed + i))
    pool = pd.concat(keep, ignore_index=True)

    # Pass 2: final sample of exactly --per-level from each level.
    parts = []
    for lvl in LEVELS:
        group = pool[pool["level"] == lvl]
        if len(group) < args.per_level:
            print(f"warning: only {len(group)} '{lvl}' puzzles available, using all of them")
        parts.append(group.sample(n=min(len(group), args.per_level), random_state=args.seed))
    sample = pd.concat(parts, ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample[["id", "puzzle", "solution", "clues", "difficulty", "level"]].to_csv(args.output, index=False)

    print("Puzzles in the full dataset per level:", totals)
    print(sample.groupby("level")["difficulty"].describe()[["count", "min", "mean", "max"]])
    print(f"\nWrote {len(sample)} puzzles to {args.output}")


if __name__ == "__main__":
    main()
