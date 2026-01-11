"""Network optimization solvers using Google OR-Tools."""

import time

from ortools.graph.python import max_flow, min_cost_flow

from vertex.config import SolverStatus
from vertex.models.network import MaxFlowResult, MinCostFlowResult, ShortestPathResult


def _build_node_index(nodes: list[str]) -> dict[str, int]:
    """Create mapping from node names to indices."""
    return {name: i for i, name in enumerate(nodes)}


def solve_max_flow(
    nodes: list[str],
    arcs: list[dict],
    source: str,
    sink: str,
) -> MaxFlowResult:
    """
    Find maximum flow from source to sink.

    Args:
        nodes: List of node names.
        arcs: List of arcs with 'source', 'target', 'capacity'.
        source: Source node name.
        sink: Sink node name.

    Returns:
        MaxFlowResult with max_flow and arc_flows.
    """
    start_time = time.time()
    node_idx = _build_node_index(nodes)

    smf = max_flow.SimpleMaxFlow()
    arc_keys = []

    for arc in arcs:
        smf.add_arc_with_capacity(
            node_idx[arc["source"]],
            node_idx[arc["target"]],
            int(arc["capacity"]),
        )
        arc_keys.append(f"{arc['source']}->{arc['target']}")

    status = smf.solve(node_idx[source], node_idx[sink])
    solve_time = (time.time() - start_time) * 1000

    if status != smf.OPTIMAL:
        return MaxFlowResult(status=SolverStatus.ERROR, solve_time_ms=solve_time)

    arc_flows = {}
    for i in range(smf.num_arcs()):
        if smf.flow(i) > 0:
            arc_flows[arc_keys[i]] = float(smf.flow(i))

    idx_to_node = {v: k for k, v in node_idx.items()}
    source_cut = [idx_to_node[i] for i in smf.get_source_side_min_cut()]
    sink_cut = [idx_to_node[i] for i in smf.get_sink_side_min_cut()]

    return MaxFlowResult(
        status=SolverStatus.OPTIMAL,
        max_flow=float(smf.optimal_flow()),
        arc_flows=arc_flows,
        source_side_cut=source_cut,
        sink_side_cut=sink_cut,
        solve_time_ms=solve_time,
    )


def solve_min_cost_flow(
    nodes: list[str],
    arcs: list[dict],
    supplies: dict[str, int],
) -> MinCostFlowResult:
    """
    Find minimum cost flow satisfying supplies/demands.

    Args:
        nodes: List of node names.
        arcs: List of arcs with 'source', 'target', 'capacity', 'cost'.
        supplies: Node supplies (positive) and demands (negative).

    Returns:
        MinCostFlowResult with total_cost and arc_flows.
    """
    start_time = time.time()
    node_idx = _build_node_index(nodes)

    smcf = min_cost_flow.SimpleMinCostFlow()
    arc_keys = []

    for arc in arcs:
        smcf.add_arc_with_capacity_and_unit_cost(
            node_idx[arc["source"]],
            node_idx[arc["target"]],
            int(arc["capacity"]),
            int(arc["cost"]),
        )
        arc_keys.append(f"{arc['source']}->{arc['target']}")

    for node, supply in supplies.items():
        smcf.set_node_supply(node_idx[node], supply)

    status = smcf.solve()
    solve_time = (time.time() - start_time) * 1000

    if status != smcf.OPTIMAL:
        return MinCostFlowResult(status=SolverStatus.ERROR, solve_time_ms=solve_time)

    arc_flows = {}
    total_flow = 0
    for i in range(smcf.num_arcs()):
        if smcf.flow(i) > 0:
            arc_flows[arc_keys[i]] = float(smcf.flow(i))
            total_flow += smcf.flow(i)

    return MinCostFlowResult(
        status=SolverStatus.OPTIMAL,
        total_cost=float(smcf.optimal_cost()),
        total_flow=float(sum(supplies.get(n, 0) for n in nodes if supplies.get(n, 0) > 0)),
        arc_flows=arc_flows,
        solve_time_ms=solve_time,
    )


def solve_shortest_path(
    nodes: list[str],
    arcs: list[dict],
    source: str,
    target: str,
) -> ShortestPathResult:
    """
    Find shortest path using min cost flow with unit flow.

    Args:
        nodes: List of node names.
        arcs: List of arcs with 'source', 'target', 'cost'.
        source: Start node.
        target: End node.

    Returns:
        ShortestPathResult with distance and path.
    """
    start_time = time.time()
    node_idx = _build_node_index(nodes)

    smcf = min_cost_flow.SimpleMinCostFlow()
    arc_info = []

    for arc in arcs:
        smcf.add_arc_with_capacity_and_unit_cost(
            node_idx[arc["source"]],
            node_idx[arc["target"]],
            1,
            int(arc.get("cost", 1)),
        )
        arc_info.append((arc["source"], arc["target"]))

    smcf.set_node_supply(node_idx[source], 1)
    smcf.set_node_supply(node_idx[target], -1)

    status = smcf.solve()
    solve_time = (time.time() - start_time) * 1000

    if status != smcf.OPTIMAL:
        return ShortestPathResult(status=SolverStatus.INFEASIBLE, solve_time_ms=solve_time)

    # Reconstruct path
    path_edges = []
    for i in range(smcf.num_arcs()):
        if smcf.flow(i) > 0:
            path_edges.append(arc_info[i])

    # Build path from edges
    path = [source]
    current = source
    while current != target and path_edges:
        for i, (s, t) in enumerate(path_edges):
            if s == current:
                path.append(t)
                current = t
                path_edges.pop(i)
                break
        else:
            break

    return ShortestPathResult(
        status=SolverStatus.OPTIMAL,
        distance=float(smcf.optimal_cost()),
        path=path,
        solve_time_ms=solve_time,
    )
