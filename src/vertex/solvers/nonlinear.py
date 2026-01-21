"""Solver for Nonlinear Programming (NLP) and MINLP using SciPy."""

import time

import numpy as np
import scipy.optimize
from scipy.optimize import NonlinearConstraint
from sympy import lambdify, symbols, sympify

from vertex.config import SolverStatus, VariableType
from vertex.models.nonlinear import NLPProblem, NLPSolution


def solve_nlp(problem: NLPProblem, time_limit_seconds: int = 30) -> NLPSolution:
    """
    Solve NLP/MINLP using SciPy.

    Uses SLSQP for continuous problems and differential_evolution for mixed-integer.

    Args:
        problem: NLP problem definition.
        time_limit_seconds: Not directly supported by minimize, but we verify result.

    Returns:
        Solution.
    """
    start_time = time.time()

    # 1. Parse variables
    var_names = [v.name for v in problem.variables]
    # Bounds for DE/SLSQP: (min, max)
    # Ensure bounds are float (DE requires finite bounds usually, assume large if None)
    bounds = []
    x0 = []
    integrality = []  # True for integer, False for continuous
    has_integers = False

    for v in problem.variables:
        lb = v.lower_bound if v.lower_bound is not None else -1e10
        ub = v.upper_bound if v.upper_bound is not None else 1e10
        bounds.append((lb, ub))
        x0.append(v.initial_guess)

        is_int = v.var_type in (VariableType.INTEGER, VariableType.BINARY)
        integrality.append(is_int)
        if is_int:
            has_integers = True

    if not var_names:
        return NLPSolution(status=SolverStatus.ERROR, message="No variables defined")

    # Sympy symbols
    sym_vars = symbols(var_names)

    # 2. Parse objective
    try:
        obj_expr = sympify(problem.objective_expression)
        # Use modules="numpy" for vectorization support which DE exploits
        obj_func_sym = lambdify(sym_vars, obj_expr, modules="numpy")

        # DE passes (N, S) array for population? No, default is 1D array per call unless vectorized=True
        # SLSQP passes 1D array.

        def obj_wrapper(x):
            return obj_func_sym(*x)

        sign = 1.0 if problem.objective_sense == "minimize" else -1.0

        def objective(x):
            return sign * obj_wrapper(x)

    except Exception as e:
        return NLPSolution(status=SolverStatus.ERROR, message=f"Invalid objective: {e}")

    # 3. Parse constraints
    cons_slsqp = []
    cons_de = []

    for c in problem.constraints:
        try:
            expr = sympify(c.expression)
            expr_func_sym = lambdify(sym_vars, expr, modules="numpy")

            # For DE: NonlinearConstraint(fun, lb, ub)
            if c.sense == "=":
                lb, ub = c.rhs, c.rhs
            elif c.sense == ">=":
                lb, ub = c.rhs, np.inf
            elif c.sense == "<=":
                lb, ub = -np.inf, c.rhs

            # Wrapper for DE constraint (must accept x)
            def de_wrapper(x, func=expr_func_sym):
                return func(*x)

            cons_de.append(NonlinearConstraint(de_wrapper, lb, ub))

            # For SLSQP: dict
            def slsqp_wrapper(x, func=expr_func_sym, rhs=c.rhs, sense=c.sense):
                val = func(*x)
                if sense == "=":
                    return val - rhs
                elif sense == ">=":
                    return val - rhs
                elif sense == "<=":
                    return rhs - val
                return 0.0

            cons_slsqp.append(
                {"type": "eq" if c.sense == "=" else "ineq", "fun": slsqp_wrapper}
            )

        except Exception as e:
            return NLPSolution(
                status=SolverStatus.ERROR,
                message=f"Invalid constraint '{c.expression}': {e}",
            )

    # 4. Solve
    try:
        if has_integers:
            # Use Differential Evolution for MINLP
            # Requires bounds to be finite? DE usually does.
            # We set +/- 1e10 defaults which is fine.
            res = scipy.optimize.differential_evolution(
                objective,
                bounds=bounds,
                constraints=cons_de,
                integrality=integrality,
                seed=42,
                maxiter=100,  # Limit iterations for speed
                popsize=10,
                tol=1e-3,
            )
        else:
            # Use SLSQP for continuous NLP
            res = scipy.optimize.minimize(
                objective,
                x0,
                bounds=bounds,
                constraints=cons_slsqp,
                method="SLSQP",
                options={"maxiter": 1000, "ftol": 1e-6},
            )
    except Exception as e:
        return NLPSolution(status=SolverStatus.ERROR, message=f"Solver error: {e}")

    elapsed = (time.time() - start_time) * 1000

    # Determine status
    if res.success:
        status = SolverStatus.OPTIMAL
    else:
        status = SolverStatus.FEASIBLE  # Conservative

    return NLPSolution(
        status=status,
        objective_value=round(sign * res.fun, 6),
        variable_values={
            name: round(float(val), 6) for name, val in zip(var_names, res.x)
        },
        solve_time_ms=elapsed,
        message=str(res.message),
    )
