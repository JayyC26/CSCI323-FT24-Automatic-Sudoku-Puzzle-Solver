from collections import deque
import copy

class SudokuCSP:
    def __init__(self, board):
        self.variables = [(r, c) for r in range(9) for c in range(9)]
        self.domains = {}
        for r, c in self.variables:
            if board[r][c] != 0:
                self.domains[(r, c)] = {int(board[r][c])}
            else:
                self.domains[(r, c)] = set(range(1, 10))

        self.neighbors = {v: set() for v in self.variables}
        self._build_constraints()
        self.revisions = 0  # Metric for your report evaluation

    def _build_constraints(self):
        for r, c in self.variables:
            row_peers = [(r, col) for col in range(9) if col != c]
            col_peers = [(row, c) for row in range(9) if row != r]
            br, bc = 3 * (r // 3), 3 * (c // 3)
            box_peers = [
                (row, col)
                for row in range(br, br + 3)
                for col in range(bc, bc + 3)
                if (row, col) != (r, c)
            ]
            self.neighbors[(r, c)] = set(row_peers + col_peers + box_peers)

    def ac3(self, domains):
        queue = deque(
            [(xi, xj) for xi in self.variables for xj in self.neighbors[xi]]
        )
        while queue:
            xi, xj = queue.popleft()
            if self.revise(domains, xi, xj):
                if len(domains[xi]) == 0:
                    return False  # Domain wiped out: unsolvable state
                for xk in self.neighbors[xi] - {xj}:
                    queue.append((xk, xi))
        return True

    def revise(self, domains, xi, xj):
        revised = False
        self.revisions += 1
        # In a binary inequality constraint (xi != xj), a value x in D(xi)
        # has no support in D(xj) ONLY if D(xj) is a singleton {x}.
        if len(domains[xj]) == 1:
            val_j = next(iter(domains[xj]))
            if val_j in domains[xi]:
                domains[xi].remove(val_j)
                revised = True
        return revised

    def solve(self):
        # Step 1: Initial AC-3 Constraint Propagation
        if not self.ac3(self.domains):
            return None
        # Step 2: MAC Search if any domain still has >1 candidate
        result_domains = self._mac_search(self.domains)
        if result_domains is None:
            return None
        return [[next(iter(result_domains[(r, c)])) for c in range(9)] for r in range(9)]

    def _mac_search(self, domains):
        if all(len(domains[v]) == 1 for v in self.variables):
            return domains

        # Minimum Remaining Values (MRV) heuristic to pick next unassigned cell
        unassigned = [v for v in self.variables if len(domains[v]) > 1]
        var = min(unassigned, key=lambda v: len(domains[v]))

        for value in sorted(domains[var]):
            local_domains = copy.deepcopy(domains)
            local_domains[var] = {value}
            if self.ac3(local_domains):
                result = self._mac_search(local_domains)
                if result is not None:
                    return result
        return None