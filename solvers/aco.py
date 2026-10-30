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

Useful helpers: common.board.PEERS, common.board.UNITS, common.board.candidates.
ACS is stochastic: use the `seed` option for repeatable runs. Report the number
of iterations as "steps" (or ant constructions; say which in the report).
"""

from __future__ import annotations


def solve(grid: list[int], seed: int | None = None, **options) -> dict:
    raise NotImplementedError("ACO solver not implemented yet")
