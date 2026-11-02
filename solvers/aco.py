"""Method 2: Ant Colony System (ACS) with constraint propagation and Best Value Evaporation.

Reference:
    Lloyd, H., & Amos, M. (2019). Solving Sudoku with ant colony optimization.
    IEEE Transactions on Games, 12(3), 302-311.
    https://doi.org/10.1109/TG.2019.2942773
    Authors' C++ code: https://github.com/huwlloyd-mmu/sudoku_acs

Algorithm 1 from the paper:

    read puzzle; propagate constraints for every given clue
    initialise pheromone tau[i][k] = tau0 for every cell i and digit k
    while not solved:
        give each ant its own copy of the (propagated) puzzle
        start each ant on a different random cell
        repeat c times (c = 81 cells):
            for each ant:
                if its current cell is not fixed:
                    choose a digit from the cell's value set      (Eq. 1, 2)
                    fix it and propagate constraints
                    local pheromone update on that choice        (Eq. 3)
                move to the next cell (wrap around)
        find the best ant: f_best = most cells fixed               (Eq. 4)
        global pheromone update using the best solution            (Eq. 5, 6)
        best value evaporation                                     (Eq. 7)

Details:
    Value choice (Eq. 1, 2): with probability q0, take the digit in the cell's
        value set with the highest tau ("greedy"); otherwise roulette-wheel
        selection with probability tau[i][k] / sum of tau over the value set.
    Fail cells: if propagation empties a cell's value set, that cell can't be
        fixed. Count it as a failure; score f_n = number of cells fixed.
    Local update (Eq. 3), after each choice:
        tau[i][s] = (1 - xi) * tau[i][s] + xi * tau0
        This makes the value slightly less attractive to the next ant (exploration).
    Global update (Eq. 5, 6):
        delta = c / (c - f_best)
        if delta > delta_best: delta_best = delta, best_solution = that ant's grid
        for every (i, s) in best_solution:
            tau[i][s] = (1 - rho) * tau[i][s] + rho * delta_best
        Only the best solution's values are updated. There is no global
        evaporation of all pheromone (unlike the classic Ant System).
    Best Value Evaporation (Eq. 7), the paper's new operator:
        delta_best = delta_best * (1 - rho_bve)
        Stops the colony locking in to one old best solution (anti-stagnation).

Constraint propagation (Section III-A), applied whenever a value is fixed:
    1. remove a fixed value from all of its peers' value sets
    2. if a digit can only go in one cell of a unit, fix it there
    Repeat until nothing changes. The paper's backtracking baseline uses the
    same propagation, which is what makes the comparison fair.

Parameters used in the paper (Section IV-A):
    m (ants) = 10, q0 = 0.9, rho = 0.9, xi = 0.1, rho_bve = 0.005, tau0 = 1 / c

What the paper found (useful for our analysis):
    On 9x9 puzzles, backtracking and Dancing Links are usually faster than ACS.
    ACS wins on the hardest large instances (25x25 with 40-50% of cells given),
    and without BVE its success rate on hard puzzles drops sharply.

ACS is stochastic: use the `seed` option for repeatable runs.
"steps" = colony iterations (each iteration = `ants` full constructions).
"""

from __future__ import annotations

import random
import time

from common.board import CELLS, PEERS, UNITS, is_solved

ALL = (1 << 9) - 1                      # bitmask 1..9: bit (d-1) means digit d is possible
BIT = [1 << d for d in range(9)]


def _single(m: int) -> bool:
    return m != 0 and m & (m - 1) == 0


def _digit(m: int) -> int:
    return m.bit_length()               # single-bit mask -> digit 1..9


def _propagate(masks: list[int], queue: list[int]) -> None:
    """Constraint propagation, Section III-A. Works in place on the value sets.

    Rule 1: a fixed cell's value is removed from all its peers.
    Rule 2: if a digit fits only one cell of a unit, it is fixed there.
    Cells whose value set becomes empty are failures; they stay empty.
    """
    while True:
        while queue:
            i = queue.pop()
            b = masks[i]
            for p in PEERS[i]:
                m = masks[p]
                if m & b:
                    m &= ~b
                    masks[p] = m
                    if _single(m):
                        queue.append(p)
        for unit in UNITS:
            seen1 = seen2 = 0
            for c in unit:
                m = masks[c]
                seen2 |= seen1 & m
                seen1 |= m
            once = seen1 & ~seen2
            while once:
                b = once & -once
                once ^= b
                for c in unit:
                    if masks[c] & b:
                        if masks[c] != b:
                            masks[c] = b
                            queue.append(c)
                        break
        if not queue:
            return


def _fixed(masks: list[int]) -> int:
    return sum(1 for m in masks if _single(m))


def _to_grid(masks: list[int]) -> list[int]:
    return [_digit(m) if _single(m) else 0 for m in masks]


def solve(grid: list[int], seed: int | None = None, ants: int = 10, q0: float = 0.9,
          rho: float = 0.9, xi: float = 0.1, bve: float = 0.005,
          time_limit: float = 20.0, max_iters: int | None = None,
          track_history: bool = False, **options) -> dict:
    """Ant Colony System for Sudoku (Lloyd & Amos 2019, Algorithm 1).

    Defaults are the paper's parameters. Stops when solved, after time_limit
    seconds, or after max_iters iterations. "steps" = iterations of the colony.
    With track_history=True, also returns "history": cells left unsolved by the
    best solution so far, after propagation (round 0) and after every round.
    """
    rng = random.Random(seed)
    c = CELLS
    tau0 = 1.0 / c

    # read puzzle, propagate constraints for every given clue
    base = [BIT[v - 1] if v else ALL for v in grid]
    _propagate(base, [i for i, v in enumerate(grid) if v])

    tau = [[tau0] * 9 for _ in range(c)]
    best_masks = list(base)
    best_f = _fixed(base)
    delta_best = 0.0
    iters = 0
    history = [c - best_f]
    start = time.perf_counter()

    while best_f < c:
        if max_iters is not None and iters >= max_iters:
            break
        if time.perf_counter() - start > time_limit:
            break
        iters += 1

        # each ant gets its own copy of the puzzle and a random start cell
        boards = [list(base) for _ in range(ants)]
        pos = [rng.randrange(c) for _ in range(ants)]

        for _ in range(c):
            for a in range(ants):
                masks = boards[a]
                i = pos[a]
                m = masks[i]
                if m and not _single(m):
                    t = tau[i]
                    choices = [d for d in range(9) if m & BIT[d]]
                    if rng.random() < q0:                      # greedy (Eq. 1)
                        d = max(choices, key=t.__getitem__)
                    else:                                       # roulette (Eq. 2)
                        r = rng.random() * sum(t[k] for k in choices)
                        for d in choices:
                            r -= t[d]
                            if r <= 0:
                                break
                    masks[i] = BIT[d]
                    _propagate(masks, [i])
                    t[d] = (1 - xi) * t[d] + xi * tau0          # local update (Eq. 3)
                pos[a] = (i + 1) % c

        # best ant of this iteration: most cells fixed (Eq. 4)
        scores = [_fixed(b) for b in boards]
        a_best = max(range(ants), key=scores.__getitem__)
        f = scores[a_best]
        if f == c:
            best_masks, best_f = boards[a_best], f
            history.append(0)
            break
        delta = c / (c - f)                                     # Eq. 5
        if delta > delta_best:
            delta_best = delta
            best_masks, best_f = boards[a_best], f

        # global update on the best solution's values only (Eq. 6)
        for i, m in enumerate(best_masks):
            if _single(m):
                d = _digit(m) - 1
                tau[i][d] = (1 - rho) * tau[i][d] + rho * delta_best

        delta_best *= (1 - bve)                                 # Best Value Evaporation (Eq. 7)
        history.append(c - best_f)

    solution = _to_grid(best_masks)
    out = {
        "solution": solution,
        "solved": is_solved(solution),
        "steps": iters,
        "iterations": iters,
        "cells_fixed": best_f,
    }
    if track_history:
        out["history"] = history
    return out
