# Automatic Sudoku Puzzle Solver (CSCI323)

Sudoku as a constraint satisfaction problem (CSP), solved and compared with three meaningfully different methods:

| # | Method | Type | File |
|---|--------|------|------|
| 1 | Recursive backtracking | Established, systematic, complete | `solvers/backtracking_solver.py` |
| 2 | Ant Colony Optimisation + constraint propagation | Research paper, stochastic, incomplete | `solvers/aco.py` |
| 3 | Image detection + digit recognition, solved as a CSP | Research paper | `solvers/image_csp.py` (to be added) |

Method 2 is based on: Lloyd, H., & Amos, M. (2019). Solving Sudoku with ant colony optimization. *IEEE Transactions on Games, 12*(3), 302–311. https://doi.org/10.1109/TG.2019.2942773

## Setup

Python 3.10+.

```bash
pip install -r requirements.txt
```

## Repo layout

```
common/board.py           shared helpers: parse, peers, candidates, conflict count, validity checks
solvers/                  one file per method, all with the same solve() interface
data/sample_puzzles.py    builds data/puzzles.csv from the Kaggle dataset
data/demo_puzzles.csv     3 puzzles for quick tests
benchmark/run_benchmark.py  runs solvers on the same puzzles -> results/<solver>/results.csv
benchmark/convergence.py  ACO cells left / % solved per round -> results/aco/
benchmark/plots.py        summary table + charts for slides/report
results/                  one folder per method: aco/, backtracking/, image_csp/ + comparison/ (all methods side by side)
tests/                    sanity tests (python -m pytest)
```

## Extra experiments (marginal cases)

Very hard puzzles (Kaggle rating 6 or above, about 1,300 of the 3M):

    python data/sample_puzzles.py --min-difficulty 6 --level-name very_hard --output data/puzzles_very_hard.csv
    python -m benchmark.run_benchmark --solvers aco --puzzles data/puzzles_very_hard.csv --runs 3 --out results/aco/results_very_hard.csv
    python -m benchmark.plots --results results/aco/results.csv results/aco/results_very_hard.csv

Iteration cap (how often ACO solves within N iterations):

    python -m benchmark.run_benchmark --solvers aco --puzzles data/puzzles_very_hard.csv --max-iters 1 --out results/aco/cap_1.csv

## The solver interface (everyone must follow this)

```python
def solve(grid: list[int], seed: int | None = None, **options) -> dict:
    # grid: 81 ints, row by row, 0 = empty. Don't modify it in place.
    return {
        "solution": [...],   # 81 ints, best grid found
        "solved": True,      # did you find a valid solution?
        "steps": 1234,       # your unit of work (backtracking: nodes, ACO: ant constructions)
        # any extra stats you want in the results CSV, e.g. "iterations": 42
    }
```

The benchmark times each call and checks correctness itself, so solvers don't need to.
Stochastic methods must use `seed` so runs are repeatable.

## Getting the puzzles

1. Download `sudoku-3m.csv` from [Kaggle: 3 million Sudoku puzzles with ratings](https://www.kaggle.com/datasets/radcliffe/3-million-sudoku-puzzles-with-ratings).
2. Put it in `data/raw/` (gitignored; don't commit it, it's huge).
3. Build the shared sample (150 each of easy/medium/hard by the dataset's difficulty rating):

```bash
python data/sample_puzzles.py
```

Commit the resulting `data/puzzles.csv` so everyone tests on exactly the same puzzles.

## Running

```bash
python -m pytest                                              # sanity tests
python -m benchmark.run_benchmark --puzzles data/demo_puzzles.csv   # quick check
python -m benchmark.run_benchmark --runs 5                    # all solvers -> results/results.csv
python -m benchmark.run_benchmark --solvers aco --runs 3      # one solver -> results/aco/results.csv
python -m benchmark.plots --results results/aco/results.csv   # summary.csv + charts next to the results file
```

Solvers that still raise `NotImplementedError` are skipped automatically.

## Workflow

- One branch per pair, e.g. `aco`, `image_csp`. Open a pull request into `main` when your solver passes `python -m pytest`.
- Only change `common/` and `benchmark/` after telling the group, since every solver depends on them.
