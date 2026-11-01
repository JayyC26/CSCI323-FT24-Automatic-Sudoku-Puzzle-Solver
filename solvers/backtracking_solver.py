"""
Method 1: Naive Recursive Depth-First Backtracking Solver (Baseline)

Reference:
    Bhattarai et al. (2025). A Study Of Sudoku Solving Algorithms: 
    Backtracking and Heuristic. ICAN-OPT 2025.
    https://doi.org/10.48550/arXiv.2507.09708

Algorithm Overview:
    1. Scan the 9x9 grid sequentially for an unassigned cell (represented by 0).
    2. Attempt to place digits 1 through 9.
    3. Check row, column, and 3x3 box validity constraints.
    4. Recursively attempt to solve the next cell.
    5. Backtrack (reset to 0) if a assignment leads to a dead end.
"""

import time
from typing import List, Dict, Any, Tuple


def is_valid(grid: List[List[int]], row: int, col: int, num: int) -> bool:
    """Check if placing `num` at grid[row][col] is valid under Sudoku constraints."""
    for i in range(9):
        # Check row and column constraints
        if grid[row][i] == num or grid[i][col] == num:
            return False

    # Check 3x3 box constraint
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if grid[start_row + i][start_col + j] == num:
                return False

    return True


def find_empty_cell(grid: List[List[int]]) -> Tuple[int, int] | None:
    """Finds the next empty cell (0). Returns (row, col) or None if full."""
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                return (r, c)
    return None


def solve(puzzle: List[int] | List[List[int]], **kwargs) -> Dict[str, Any]:
    """
    Standard solver entry point matching group interface.
    Accepts either a 1D flat list (length 81) or a 2D 9x9 list.
    """
    # Convert 1D input to 2D grid if necessary
    if len(puzzle) == 81 and isinstance(puzzle[0], int):
        grid = [puzzle[i : i + 9] for i in range(0, 81, 9)]
    else:
        grid = [row[:] for row in puzzle]

    backtrack_count = 0
    start_time = time.perf_counter()

    def _backtrack() -> bool:
        nonlocal backtrack_count
        empty = find_empty_cell(grid)
        
        # Base case: No empty cells left means puzzle is solved
        if not empty:
            return True

        row, col = empty
        for num in range(1, 10):
            if is_valid(grid, row, col, num):
                grid[row][col] = num

                if _backtrack():
                    return True

                # Undo assignment (backtrack)
                grid[row][col] = 0
                backtrack_count += 1

        return False

    is_solved = _backtrack()
    elapsed_time = time.perf_counter() - start_time

    # Flatten output back to 1D array to match group format
    flat_solution = [cell for row in grid for cell in row]

    return {
        "solution": flat_solution,
        "solved": is_solved,
        "backtracks": backtrack_count,
        "time_taken": elapsed_time,
        "algorithm": "Naive Recursive Backtracking"
    } 
