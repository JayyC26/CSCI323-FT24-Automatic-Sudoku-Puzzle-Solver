"""Sanity checks. Run from the repo root with:  python -m pytest

Each implemented solver must solve the demo puzzles correctly.
Solvers that still raise NotImplementedError are skipped, not failed.
"""

import csv
from pathlib import Path

import pytest

from common.board import is_solved, parse, respects_clues
from solvers import SOLVERS

DEMO = Path(__file__).resolve().parent.parent / "data" / "demo_puzzles.csv"
PUZZLES = list(csv.DictReader(open(DEMO, newline="")))


@pytest.mark.parametrize("name", list(SOLVERS))
@pytest.mark.parametrize("row", PUZZLES, ids=[p["id"] for p in PUZZLES])
def test_solver_on_demo(name, row):
    grid = parse(row["puzzle"])
    original = list(grid)
    try:
        out = SOLVERS[name](grid, seed=323)
    except NotImplementedError:
        pytest.skip(f"{name} not implemented yet")

    assert grid == original, "solver must not modify the input grid"
    assert out["solved"] is True
    assert is_solved(out["solution"])
    assert respects_clues(original, out["solution"])
    assert out["solution"] == parse(row["solution"])
    assert isinstance(out["steps"], int)
