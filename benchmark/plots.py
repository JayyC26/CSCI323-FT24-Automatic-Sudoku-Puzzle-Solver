"""Turn results/results.csv into summary tables and charts for the slides and report.

From the repo root, after running the benchmark:

    python -m benchmark.plots

Writes to results/:
    summary.csv           solve rate, mean/median time, mean steps per solver and level
    solve_rate.png        % of puzzles solved correctly, per level
    time_by_level.png     median solve time per level (log scale)
    time_vs_difficulty.png  every attempt's time against the dataset's difficulty rating
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LEVEL_ORDER = ["easy", "medium", "hard"]
COLORS = {"backtracking": "#2a6f97", "aco": "#e0a526", "proposed": "#2e9e6b"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "results.csv")
    ap.add_argument("--outdir", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    df["correct"] = df["correct"].astype(str).str.lower() == "true"
    levels = [lvl for lvl in LEVEL_ORDER if lvl in set(df["level"].dropna())] or sorted(df["level"].dropna().unique())
    solvers = list(dict.fromkeys(df["solver"]))
    args.outdir.mkdir(parents=True, exist_ok=True)

    summary = (
        df.groupby(["solver", "level"])
        .agg(attempts=("correct", "size"),
             solve_rate=("correct", "mean"),
             mean_time_s=("time_s", "mean"),
             median_time_s=("time_s", "median"),
             mean_steps=("steps", "mean"))
        .reset_index()
    )
    summary["solve_rate"] = (summary["solve_rate"] * 100).round(1)
    summary["level"] = pd.Categorical(summary["level"], categories=levels, ordered=True)
    summary = summary.sort_values(["solver", "level"]).reset_index(drop=True)
    summary.to_csv(args.outdir / "summary.csv", index=False)
    print(summary.to_string(index=False))

    def grouped_bars(metric: str, ylabel: str, filename: str, log: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        width = 0.8 / max(len(solvers), 1)
        for j, s in enumerate(solvers):
            sub = summary[summary["solver"] == s].set_index("level").reindex(levels)
            xs = [i + (j - (len(solvers) - 1) / 2) * width for i in range(len(levels))]
            ax.bar(xs, sub[metric], width, label=s, color=COLORS.get(s))
        ax.set_xticks(range(len(levels)), levels)
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.outdir / filename, dpi=200)
        plt.close(fig)

    grouped_bars("solve_rate", "Solved correctly (%)", "solve_rate.png")
    grouped_bars("median_time_s", "Median solve time (s, log scale)", "time_by_level.png", log=True)

    if pd.to_numeric(df["difficulty"], errors="coerce").notna().any():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for s in solvers:
            sub = df[df["solver"] == s]
            ax.scatter(pd.to_numeric(sub["difficulty"], errors="coerce"), sub["time_s"],
                       s=10, alpha=0.5, label=s, color=COLORS.get(s))
        ax.set_xlabel("Dataset difficulty rating")
        ax.set_ylabel("Solve time (s, log scale)")
        ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.outdir / "time_vs_difficulty.png", dpi=200)
        plt.close(fig)

    print(f"\nCharts and summary written to {args.outdir}")


if __name__ == "__main__":
    main()
