"""Tests for constraint programming tools."""

from vertex.tools.cp import solve_n_queens, solve_sudoku


def test_sudoku_solver():
    """Test Sudoku solver with simple puzzle."""
    puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]

    result = solve_sudoku(puzzle)

    assert result.status.value == "optimal"
    assert result.grid is not None
    assert len(result.grid) == 9


def test_n_queens():
    """Test N-Queens solver."""
    result = solve_n_queens(n=4)

    assert result.status.value == "optimal"
    assert result.queens is not None
    assert len(result.queens) == 4
