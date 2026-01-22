# Vertex Roadmap

This document outlines the development roadmap for the Vertex MCP Server. It combines immediate implementation goals with a long-term vision for a comprehensive Operations Research platform.

## Completed Features

### Core Optimization

- [x] **Linear Programming (LP)** (GLOP backend)
- [x] **Mixed-Integer Programming (MIP)** (SCIP backend)
- [x] **Quadratic Programming (QP)** (CVXPY/OSQP backend)
- [x] **Sensitivity Analysis** (Shadow prices, reduced costs)
- [x] **Multi-Objective Optimization** (Pareto frontiers)
- [x] **Constraint Programming** (Sudoku, N-Queens)

### Network & Graph

- [x] Max Flow & Min Cost Flow
- [x] Shortest Path & Minimum Spanning Tree
- [x] Multi-Commodity Flow
- [x] Transshipment Problems
- [x] Graph Coloring

### Scheduling & Routing

- [x] Traveling Salesman Problem (TSP)
- [x] Vehicle Routing Problem (VRP) with Capacity & Time Windows
- [x] Job Shop, Flexible Job Shop, & Flow Shop Scheduling
- [x] Parallel Machine Scheduling
- [x] Resource-Constrained Project Scheduling (RCPSP)

### Combinatorial & Discrete

- [x] Bin Packing (1D)
- [x] Set Covering / Partitioning
- [x] Cutting Stock Problem
- [x] Knapsack Problem
- [x] Assignment Problem (Workers to Tasks)

### Stochastic & Dynamic

- [x] Two-Stage Stochastic Programming
- [x] Robust Optimization (Worst-case bounds)
- [x] Newsvendor Model (Stochastic Inventory)
- [x] Dynamic Lot Sizing (Wagner-Whitin)
- [x] Queueing Models (M/M/1, M/M/c)
- [x] Monte Carlo Simulation (Risk Analysis)
- [x] Crew Scheduling / Shift Planning
- [x] Chance-Constrained Programming

### Combinatorial & Network

- [x] 2D Bin Packing / Rectangle Packing
- [x] Capacitated Network Design
- [x] Steiner Tree
- [x] Quadratic Assignment Problem (QAP)
- [x] Multi-echelon Inventory

### Domain Templates

- [x] Production Planning
- [x] Diet Problem
- [x] Portfolio Optimization
- [x] Supply Chain Network Design
- [x] Workforce Scheduling
- [x] Healthcare Resource Allocation
- [x] Inventory Optimization (EOQ)
- [x] Facility Location

### Analysis & Utilities

- [x] What-If Analysis
- [x] Infeasibility Diagnosis
- [x] Model Statistics (Size, Sparsity)
- [x] Solution Pool (Alternative optima)

## Short-Term Goals (Next 3 Months)

Focus: Robustness, User Experience, and Missing Core Models.

### Infrastructure & Usability

- [x] **Async Solving**: Support for long-running optimization jobs suitable for larger datasets.
- [x] **Solver Selection**: Allow users to explicitly choose between solvers (e.g., GLOP vs. SCIP vs. CP-SAT).
- [x] **Visualization Helpers**: Generate data structures specifically for plotting (e.g., Gantt chart JSON for scheduling).

### Advanced Modeling

- [x] **Maintenance Planning**: Condition-based maintenance models using equipment degradation curves.

## Medium-Term Goals (3-6 Months)

Focus: Stochasticity, Simulation, and specialized Industrial Verticals.

### Uncertainty & Risk

- [x] **Markov Decision Processes (MDP)**: Framework for sequential decision-making problems (e.g., equipment replacement).

### Network Design

- [x] **Large-Scale VRP**: Metaheuristic approaches enabled via configuration (Guided Local Search, Tabu Search).
- [x] **Multi-Depot VRP**: Routing from multiple depots.

## Long-Term Vision (6-12+ Months)

Focus: Advanced Algorithms, Scale, and Autonomous Features.

### Large-Scale Optimization

- [ ] **Decomposition Methods**: Implement Benders Decomposition and Column Generation for massive scale problems.
- [ ] **Distributed Solving**: Ability to offload solver computation to external clusters.

### Nonlinear & Hybrid

- [x] **Nonlinear Programming (NLP)**: Support for non-linear objective functions and constraints.
- [x] **Mixed-Integer Nonlinear Programming (MINLP)**: Handling complex process engineering problems.
- [x] **Simulation-Optimization**: Optimization where the objective function is evaluated via discrete-event simulation.

### AI Integration

- [ ] **Predict-then-Optimize**: Tighter integration with forecasting models to automatically populate optimization parameters.
- [x] **Self-Tuning**: Auto-selection of solver parameters and algorithms based on problem characteristics.
