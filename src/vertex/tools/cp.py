"""Constraint Programming tools for puzzles and scheduling."""

from ortools.sat.python import cp_model
from pydantic import BaseModel, Field

from vertex.config import SolverStatus


class SudokuResult(BaseModel):
    """Result of Sudoku solving."""

    status: SolverStatus
    grid: list[list[int]] = Field(default_factory=list)
    solve_time_ms: float | None = None


class NQueensResult(BaseModel):
    """Result of N-Queens problem."""

    status: SolverStatus
    queens: list[int] = Field(
        default_factory=list, description="Column position for each row"
    )
    solve_time_ms: float | None = None


def solve_sudoku(grid: list[list[int]]) -> SudokuResult:
    """
    Solve a Sudoku puzzle using constraint programming.

    Args:
        grid: 9x9 grid with 0 for empty cells, 1-9 for filled cells.

    Returns:
        SudokuResult with completed grid.
    """
    import time

    start_time = time.time()
    model = cp_model.CpModel()

    # Variables
    cells = {}
    for i in range(9):
        for j in range(9):
            if grid[i][j] != 0:
                cells[(i, j)] = model.new_int_var(grid[i][j], grid[i][j], f"c_{i}_{j}")
            else:
                cells[(i, j)] = model.new_int_var(1, 9, f"c_{i}_{j}")

    # Row constraints
    for i in range(9):
        model.add_all_different([cells[(i, j)] for j in range(9)])

    # Column constraints
    for j in range(9):
        model.add_all_different([cells[(i, j)] for i in range(9)])

    # 3x3 box constraints
    for box_i in range(3):
        for box_j in range(3):
            model.add_all_different(
                [
                    cells[(box_i * 3 + i, box_j * 3 + j)]
                    for i in range(3)
                    for j in range(3)
                ]
            )

    solver = cp_model.CpSolver()
    status = solver.solve(model)
    elapsed = (time.time() - start_time) * 1000

    if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
        return SudokuResult(status=SolverStatus.INFEASIBLE, solve_time_ms=elapsed)

    result_grid = [[solver.value(cells[(i, j)]) for j in range(9)] for i in range(9)]

    return SudokuResult(
        status=SolverStatus.OPTIMAL,
        grid=result_grid,
        solve_time_ms=elapsed,
    )


def solve_n_queens(n: int) -> NQueensResult:
    """
    Solve N-Queens problem - place N queens on NxN board with no attacks.

    Args:
        n: Board size and number of queens.

    Returns:
        NQueensResult with queen positions.
    """
    import time

    start_time = time.time()
    model = cp_model.CpModel()

    # queens[i] = column of queen in row i
    queens = [model.new_int_var(0, n - 1, f"q_{i}") for i in range(n)]

    # All different columns
    model.add_all_different(queens)

    # All different diagonals
    model.add_all_different([queens[i] + i for i in range(n)])
    model.add_all_different([queens[i] - i for i in range(n)])

    solver = cp_model.CpSolver()
    status = solver.solve(model)
    elapsed = (time.time() - start_time) * 1000

    if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
        return NQueensResult(status=SolverStatus.INFEASIBLE, solve_time_ms=elapsed)

    return NQueensResult(
        status=SolverStatus.OPTIMAL,
        queens=[solver.value(queens[i]) for i in range(n)],
        solve_time_ms=elapsed,
    )
