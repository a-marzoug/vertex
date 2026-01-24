# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release of Vertex MCP Server
- 60+ optimization tools covering LP, MIP, network flow, routing, scheduling, and stochastic programming
- 7 natural language prompts for problem formulation
- HTTP and stdio transport modes
- Docker support with compose profiles
- Comprehensive validation and error handling
- Prometheus metrics endpoint
- Structured logging with JSON output
- Sensitivity analysis and what-if scenarios
- Multi-objective optimization (Pareto frontier)
- Queueing theory analysis (M/M/1, M/M/c)
- Monte Carlo simulation tools
- MDP solver for sequential decision problems
- Automatic solver selection based on problem characteristics

### Changed

- Improved input validation for `solve_linear_program` to ensure at least one variable is defined.
- Enhanced MCP protocol test stability and teardown logic in `tests/test_mcp_protocol.py`.

### Removed

- Constraint programming tools (Sudoku, N-Queens) to focus the engine on core mathematical optimization.
- Diet optimization template (consolidated into general linear programming usage).

### Documentation

- Getting started guide
- Comprehensive API reference
- Business case examples
- Architecture documentation
- Development roadmap
