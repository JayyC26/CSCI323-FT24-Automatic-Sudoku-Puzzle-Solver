"""Solver registry.

Every solver module exposes the same function:

    def solve(grid: list[int], **options) -> dict

Input:  grid is 81 ints, row by row, 0 = empty. Do not modify it in place.
Output: a dict with at least
    "solution": list[int] of 81 ints (best grid found, even if unsolved)
    "solved":   bool, True only if the solver believes it found a valid solution
    "steps":    int, the method's own unit of work
                (backtracking = search nodes, ACO = ant constructions, etc.)
Optional extra keys (e.g. "iterations", "restarts") are kept in the results CSV.

The benchmark times each call itself, so solvers do not need to time themselves.
To plug in a new method, add it to SOLVERS below.
"""

from solvers import aco

SOLVERS = {
    # "backtracking": backtracking.solve,  # Method 1: established (to be added)
    "aco": aco.solve,                    # Method 2: research paper (Lloyd & Amos)
    # "image_csp": image_csp.solve,        # Method 3: image detection + CSP (to be added)
}
