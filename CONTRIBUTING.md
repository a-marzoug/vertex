# Contributing to Vertex

Thank you for your interest in contributing to Vertex! We welcome contributions from the community to help make this the best Operations Research MCP server.

## Code of Conduct

This project adheres to the Contributor Covenant [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork the repository** and clone it locally
2. **Install `uv`**: We use `uv` for dependency management

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3. **Install dependencies**:

    ```bash
    uv sync
    ```

4. **Verify installation**:

    ```bash
    uv run vertex --help
    ```

## Development Workflow

1. **Create a branch** for your feature or fix:

    ```bash
    git checkout -b feature/your-feature-name
    ```

2. **Make your changes** following our coding standards

3. **Run tests** to ensure everything works:

    ```bash
    uv run pytest
    ```

4. **Lint and format** your code:

    ```bash
    uv run ruff check .
    uv run ruff format .
    ```

5. **Type check** to catch type errors:

    ```bash
    uv run mypy src/vertex
    ```

6. **Commit your changes** with clear, descriptive messages:

    ```bash
    git commit -m "feat: add new optimization tool for X"
    ```

## Coding Standards

### Python Style

- Follow PEP 8 guidelines (enforced by Ruff)
- Use type hints for all function signatures
- Maximum line length: 88 characters
- Use double quotes for strings
- Sort imports with isort (integrated in Ruff)

### Type Hints

All functions must have complete type annotations:

```python
def solve_problem(
    variables: list[str],
    constraints: dict[str, float],
    objective: dict[str, float]
) -> LPSolution:
    ...
```

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include examples for complex functions
- Update relevant documentation in `docs/` when adding features

### Testing

- Write tests for all new functionality
- Aim for >80% code coverage
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`
- Include edge cases and error conditions

Example:

```python
def test_solve_lp_with_infeasible_constraints_raises_error():
    with pytest.raises(ValidationError):
        solve_lp(...)
```

## Adding New Features

### Adding New Tools

1. Create your tool in the appropriate module under `src/vertex/tools/`
2. Define input/output models in `src/vertex/models/`
3. Implement the solver logic in `src/vertex/solvers/`
4. Add validation using decorators from `src/vertex/validation.py`
5. Register the tool in `src/vertex/server.py`
6. Add comprehensive tests in `tests/`
7. Update documentation in `docs/API_REFERENCE.md`

Example structure:

```python
from vertex.validation import validate_problem_size, validate_positive

@validate_problem_size(max_vars=10000)
@validate_positive("capacity")
async def solve_new_problem(
    items: list[str],
    capacity: float,
) -> NewProblemResult:
    """Solve a new optimization problem.
    
    Args:
        items: List of item names
        capacity: Maximum capacity constraint
        
    Returns:
        Solution with optimal selection
    """
    ...
```

### Adding New Solvers

1. Create solver class in `src/vertex/solvers/`
2. Inherit from `Solver` base class if applicable
3. Implement the `solve()` method
4. Add solver to `src/vertex/solvers/tuning.py` for automatic selection
5. Add integration tests

### Adding New Prompts

1. Create prompt function in `src/vertex/prompts/`
2. Use clear, structured guidance for LLMs
3. Include examples and templates
4. Register in `src/vertex/server.py`

## Testing Guidelines

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_linear.py

# Run with coverage report
uv run pytest --cov=src/vertex --cov-report=html

# Run specific test
uv run pytest tests/test_linear.py::test_simple_lp
```

### Test Categories

- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test tool workflows end-to-end
- **Validation tests**: Test error handling and edge cases

### Writing Good Tests

- Test one thing per test function
- Use fixtures for common setup
- Mock external dependencies
- Test both success and failure paths
- Use parametrize for multiple similar cases

## Pull Request Process

1. **Update documentation** for any user-facing changes
2. **Add tests** that prove your fix/feature works
3. **Ensure CI passes** - all tests, linting, and type checks must pass
4. **Update CHANGELOG** (if applicable) with your changes
5. **Request review** from maintainers
6. **Address feedback** promptly and professionally

### PR Title Format

Use conventional commit format:

- `feat: add new knapsack solver variant`
- `fix: correct shadow price calculation in sensitivity analysis`
- `docs: update API reference for routing tools`
- `test: add coverage for edge cases in MIP solver`
- `refactor: simplify constraint validation logic`
- `perf: optimize network flow algorithm`

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- List of specific changes made

## Testing
How was this tested?

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Type hints added
- [ ] CI passes
- [ ] No breaking changes (or documented)
```

## Performance Considerations

- Use numpy/scipy for numerical operations
- Avoid unnecessary copies of large arrays
- Profile code for bottlenecks before optimizing
- Consider memory usage for large-scale problems
- Use appropriate solver backends (GLOP, SCIP, OR-Tools)

## Security

- Never commit API keys, credentials, or sensitive data
- Validate all user inputs
- Use timeouts for long-running operations
- Report security issues privately (see SECURITY.md)

## Getting Help

- Check existing [documentation](docs/)
- Search [existing issues](https://github.com/yourusername/vertex/issues)
- Ask questions in discussions
- Join our community channels

## Recognition

Contributors will be recognized in:

- GitHub contributors page
- Release notes for significant contributions
- Project documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
