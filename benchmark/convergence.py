"""ACO convergence: how many cells are still unsolved after each round.

ACO has no training loss; the closest equivalent is the number of cells the best
solution so far still leaves unsolved (81 - cells fixed). This script runs ACO on
a sample of puzzles per level, records that number after every round, and plots:

    convergence.png       mean cells left unsolved vs round, per level
    solved_by_round.png   % of puzzles solved by round N, per level
    convergence.csv       one row per (puzzle, round)

From the repo root:

    python -m benchmark.convergence
    python -m benchmark.convergence --per-level 50 --max-iters 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter
import pandas as pd

from solvers.aco import solve

ROOT = Path(__file__).resolve().parent.parent
LEVEL_ORDER = ["easy", "medium", "hard", "very_hard"]
LEVEL_COLORS = {"easy": "#2e9e6b", "medium": "#2a6f97", "hard": "#e0a526", "very_hard": "#c0392b"}


def parse(puzzle: str) -> list[int]:
    return [int(ch) if ch.isdigit() else 0 for ch in puzzle]


def round_axis(ax, max_round: int, start: int) -> None:
    ax.set_xscale("symlog", linthresh=1)
    ticks = [t for t in [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000] if start <= t <= max_round]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlim(start, max_round)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", type=Path, nargs="+",
                    default=[ROOT / "data" / "puzzles.csv", ROOT / "data" / "puzzles_very_hard.csv"])
    ap.add_argument("--per-level", type=int, default=50)
    ap.add_argument("--max-iters", type=int, default=1000)
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=323)
    ap.add_argument("--outdir", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    puzzles = pd.concat([pd.read_csv(p) for p in args.puzzles if p.exists()], ignore_index=True)
    levels = [lvl for lvl in LEVEL_ORDER if lvl in set(puzzles["level"])]

    rows = []
    for lvl in levels:
        sample = puzzles[puzzles["level"] == lvl].head(args.per_level)
        for k, p in enumerate(sample.itertuples()):
            out = solve(parse(p.puzzle), seed=args.seed + k, max_iters=args.max_iters,
                        time_limit=args.time_limit, track_history=True)
            for rnd, left in enumerate(out["history"]):
                rows.append({"id": p.id, "level": lvl, "round": rnd, "cells_left": left,
                             "solved": out["solved"]})
        print(f"{lvl}: {len(sample)} puzzles done")

    df = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.outdir / "convergence.csv", index=False)

    # pad every run to the same length (a solved run stays at 0, an unsolved run stays where it stopped)
    max_round = int(df["round"].max())
    curves = {}
    for lvl in levels:
        runs = [g.set_index("round")["cells_left"].reindex(range(max_round + 1)).ffill()
                for _, g in df[df["level"] == lvl].groupby("id")]
        curves[lvl] = pd.concat(runs, axis=1)

    # 1) mean cells left vs round (from round 1: round 0 is logic alone, shown in the title)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    start_means = []
    for lvl in levels:
        c = curves[lvl].loc[1:] if max_round >= 1 else curves[lvl]
        ax.plot(c.index, c.mean(axis=1), label=lvl.replace("_", " "),
                color=LEVEL_COLORS.get(lvl), linewidth=2)
        start_means.append(f"{lvl.replace('_', ' ')} {curves[lvl].loc[0].mean():.0f}")
    round_axis(ax, max_round, 1)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cells left unsolved (mean)")
    ax.set_title("ACO convergence (cells left before any ant: " + ", ".join(start_means) + ")",
                 fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.outdir / "convergence.png", dpi=200)
    plt.close(fig)

    # 2) % solved by round N
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for lvl in levels:
        ax.plot(curves[lvl].index, (curves[lvl] == 0).mean(axis=1) * 100, label=lvl.replace("_", " "),
                color=LEVEL_COLORS.get(lvl), linewidth=2)
    round_axis(ax, max_round, 0)
    ax.set_xlabel("Round (0 = solved by logic alone)")
    ax.set_ylabel("% of puzzles solved")
    ax.set_title("Puzzles solved by round N")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.outdir / "solved_by_round.png", dpi=200)
    plt.close(fig)

    summary = df.groupby(["level", "id"]).agg(rounds=("round", "max"), solved=("solved", "first"))
    print(summary.groupby("level").agg(puzzles=("solved", "size"), solved=("solved", "sum"),
                                        median_rounds=("rounds", "median"), max_rounds=("rounds", "max")))


if __name__ == "__main__":
    main()
