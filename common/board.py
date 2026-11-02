"""Shared Sudoku helpers used by every solver and the benchmark.

A grid is a flat list of 81 ints, row by row. 0 means an empty cell.
Cell index i sits at row i // 9, column i % 9.
"""

from __future__ import annotations

N = 9
CELLS = N * N
DIGITS = range(1, N + 1)


def _build_units() -> tuple[list[list[int]], list[list[int]]]:
    rows = [[r * N + c for c in range(N)] for r in range(N)]
    cols = [[r * N + c for r in range(N)] for c in range(N)]
    boxes = [
        [(br + r) * N + (bc + c) for r in range(3) for c in range(3)]
        for br in (0, 3, 6)
        for bc in (0, 3, 6)
    ]
    units = rows + cols + boxes
    # PEERS[i] = every other cell sharing a row, column or box with cell i
    peers = []
    for i in range(CELLS):
        p = set()
        for unit in units:
            if i in unit:
                p.update(unit)
        p.discard(i)
        peers.append(sorted(p))
    return units, peers


UNITS, PEERS = _build_units()


def parse(puzzle: str) -> list[int]:
    """Turn an 81-character string ('.' or '0' for blanks) into a grid."""
    puzzle = puzzle.strip()
    if len(puzzle) != CELLS:
        raise ValueError(f"expected 81 characters, got {len(puzzle)}")
    return [0 if ch in ".0" else int(ch) for ch in puzzle]


def count_conflicts(grid: list[int]) -> int:
    """Number of duplicate digits across all rows, columns and boxes.

    0 on a full grid means the grid is a valid solution. Local-search
    methods (ACO, SA, min-conflicts) can use this as their cost function.
    """
    conflicts = 0
    for unit in UNITS:
        vals = [grid[i] for i in unit if grid[i]]
        conflicts += len(vals) - len(set(vals))
    return conflicts


def is_solved(grid: list[int]) -> bool:
    """True if the grid is full and breaks no constraint."""
    return all(grid) and count_conflicts(grid) == 0


def respects_clues(puzzle: list[int], grid: list[int]) -> bool:
    """True if every given clue in the puzzle is unchanged in the grid."""
    return all(p == 0 or p == g for p, g in zip(puzzle, grid))
