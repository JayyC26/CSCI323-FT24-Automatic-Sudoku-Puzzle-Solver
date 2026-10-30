"""Run every solver on the same puzzle set and save one row per attempt.

From the repo root:

    python -m benchmark.run_benchmark                          # all solvers, data/puzzles.csv
    python -m benchmark.run_benchmark --solvers backtracking   # just one
    python -m benchmark.run_benchmark --puzzles data/demo_puzzles.csv
    python -m benchmark.run_benchmark --runs 5                 # repeat stochastic solvers

Output: results/results.csv with columns
    solver, run, seed, id, level, difficulty, clues, solved, correct, time_s, steps, ...
"correct" is checked independently here (full grid, no conflicts, clues kept,
and matches the dataset's solution when one is given), so a solver can't mark
itself as solved by mistake.

Deterministic solvers (backtracking) only run once per puzzle, whatever --runs is.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from common.board import is_solved, parse, respects_clues
from solvers import SOLVERS

ROOT = Path(__file__).resolve().parent.parent
DETERMINISTIC = {"backtracking"}
BASE_COLUMNS = ["solver", "run", "seed", "id", "level", "difficulty", "clues",
                "solved", "correct", "time_s", "steps"]


def load_puzzles(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def check(puzzle: list[int], answer: list[int], expected: str) -> bool:
    if len(answer) != 81 or not is_solved(answer) or not respects_clues(puzzle, answer):
        return False
    if expected:
        return parse(expected) == answer
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", type=Path, default=ROOT / "data" / "puzzles.csv")
    ap.add_argument("--solvers", nargs="+", default=list(SOLVERS))
    ap.add_argument("--runs", type=int, default=1, help="repeats per puzzle for stochastic solvers")
    ap.add_argument("--seed", type=int, default=323)
    ap.add_argument("--limit", type=int, default=None, help="only use the first N puzzles (quick tests)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "results.csv")
    args = ap.parse_args()

    if not args.puzzles.exists():
        raise SystemExit(f"Can't find {args.puzzles}. Run data/sample_puzzles.py first, "
                         "or try --puzzles data/demo_puzzles.csv")

    puzzles = load_puzzles(args.puzzles)[: args.limit]
    rows: list[dict] = []

    for name in args.solvers:
        if name not in SOLVERS:
            raise SystemExit(f"Unknown solver '{name}'. Options: {list(SOLVERS)}")
        solve = SOLVERS[name]
        runs = 1 if name in DETERMINISTIC else args.runs
        print(f"\n== {name} ({len(puzzles)} puzzles x {runs} run(s)) ==")

        try:
            for run in range(runs):
                for k, p in enumerate(puzzles):
                    grid = parse(p["puzzle"])
                    seed = args.seed + run * 100_000 + k
                    start = time.perf_counter()
                    out = solve(list(grid), seed=seed)
                    elapsed = time.perf_counter() - start

                    answer = out.get("solution", [])
                    row = {
                        "solver": name, "run": run, "seed": seed,
                        "id": p.get("id", k), "level": p.get("level", ""),
                        "difficulty": p.get("difficulty", ""), "clues": p.get("clues", ""),
                        "solved": bool(out.get("solved", False)),
                        "correct": check(grid, answer, p.get("solution", "")),
                        "time_s": round(elapsed, 6),
                        "steps": out.get("steps", ""),
                    }
                    # keep any extra stats a solver reports (e.g. iterations, restarts)
                    for key, val in out.items():
                        if key not in ("solution", "solved", "steps"):
                            row[key] = val
                    rows.append(row)
        except NotImplementedError as e:
            print(f"  skipped: {e}")
            continue

        done = [r for r in rows if r["solver"] == name]
        ok = sum(r["correct"] for r in done)
        avg = sum(r["time_s"] for r in done) / max(len(done), 1)
        print(f"  correct {ok}/{len(done)}   mean time {avg:.4f}s")

    if not rows:
        raise SystemExit("No results: no solver ran.")

    columns = BASE_COLUMNS + sorted({k for r in rows for k in r} - set(BASE_COLUMNS))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
