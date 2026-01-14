"""Network optimization solvers using Google OR-Tools."""

import time
from typing import Any

from ortools.graph.python import max_flow, min_cost_flow

from vertex.config import SolverStatus
from vertex.models.network import (
    MaxFlowResult,
    MinCostFlowResult,
    MSTResult,
    MultiCommodityFlowResult,
    ShortestPathResult,
)


def _build_node_index(nodes: list[str]) -> dict[str, int]:
    """Create mapping from node names to indices."""
    return {name: i for i, name in enumerate(nodes)}


def solve_max_flow(
    nodes: list[str],
    arcs: list[dict[str, Any]],
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
    arcs: list[dict[str, Any]],
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
        total_flow=float(
            sum(supplies.get(n, 0) for n in nodes if supplies.get(n, 0) > 0)
        ),
        arc_flows=arc_flows,
        solve_time_ms=solve_time,
    )


def solve_shortest_path(
    nodes: list[str],
    arcs: list[dict[str, Any]],
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
        return ShortestPathResult(
            status=SolverStatus.INFEASIBLE, solve_time_ms=solve_time
        )

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


def solve_mst(
    nodes: list[str],
    edges: list[dict[str, Any]],
) -> MSTResult:
    """
    Find Minimum Spanning Tree using Kruskal's algorithm.

    Args:
        nodes: List of node names.
        edges: List of edges with 'source', 'target', 'weight'.

    Returns:
        MSTResult with total_weight and edges.
    """
    from vertex.models.network import MSTEdge

    start_time = time.time()

    # Union-Find
    parent = {n: n for n in nodes}
    rank = {n: 0 for n in nodes}

    def find(x: str) -> str:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str) -> bool:
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    # Sort edges by weight
    sorted_edges = sorted(edges, key=lambda e: e["weight"])

    mst_edges = []
    total_weight = 0

    for edge in sorted_edges:
        if union(edge["source"], edge["target"]):
            mst_edges.append(
                MSTEdge(
                    source=edge["source"],
                    target=edge["target"],
                    weight=edge["weight"],
                )
            )
            total_weight += edge["weight"]
            if len(mst_edges) == len(nodes) - 1:
                break

    solve_time = (time.time() - start_time) * 1000

    if len(mst_edges) < len(nodes) - 1:
        return MSTResult(status=SolverStatus.INFEASIBLE, solve_time_ms=solve_time)

    return MSTResult(
        status=SolverStatus.OPTIMAL,
        total_weight=total_weight,
        edges=mst_edges,
        solve_time_ms=solve_time,
    )


def solve_multi_commodity_flow(
    nodes: list[str],
    arcs: list[dict[str, Any]],
    commodities: list[dict[str, Any]],
    time_limit_seconds: int = 30,
) -> MultiCommodityFlowResult:
    """
    Solve Multi-Commodity Flow Problem using LP.

    Args:
        nodes: List of node names.
        arcs: List of arcs with 'source', 'target', 'capacity', 'cost'.
        commodities: List of commodities with 'name', 'source', 'sink', 'demand'.
        time_limit_seconds: Solver time limit.

    Returns:
        MultiCommodityFlowResult with flows per commodity.
    """
    from ortools.linear_solver import pywraplp

    from vertex.models.network import MultiCommodityFlowResult

    start_time = time.time()

    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        return MultiCommodityFlowResult(status=SolverStatus.ERROR)

    node_idx = _build_node_index(nodes)
    arc_keys = [(a["source"], a["target"]) for a in arcs]

    # Variables: flow[k][a] = flow of commodity k on arc a
    flow = {}
    for k, comm in enumerate(commodities):
        for i, arc in enumerate(arcs):
            flow[(k, i)] = solver.NumVar(0, arc["capacity"], f"f_{k}_{i}")

    # Capacity constraints: sum of all commodities on arc <= capacity
    for i, arc in enumerate(arcs):
        solver.Add(
            sum(flow[(k, i)] for k in range(len(commodities))) <= arc["capacity"]
        )

    # Flow conservation for each commodity
    for k, comm in enumerate(commodities):
        for j, node in enumerate(nodes):
            inflow = sum(
                flow[(k, i)] for i, a in enumerate(arcs) if a["target"] == node
            )
            outflow = sum(
                flow[(k, i)] for i, a in enumerate(arcs) if a["source"] == node
            )

            if node == comm["source"]:
                solver.Add(outflow - inflow == comm["demand"])
            elif node == comm["sink"]:
                solver.Add(inflow - outflow == comm["demand"])
            else:
                solver.Add(inflow == outflow)

    # Minimize total cost
    solver.Minimize(
        sum(
            flow[(k, i)] * arcs[i].get("cost", 0)
            for k in range(len(commodities))
            for i in range(len(arcs))
        )
    )

    solver.set_time_limit(time_limit_seconds * 1000)
    status = solver.Solve()
    solve_time = (time.time() - start_time) * 1000

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return MultiCommodityFlowResult(
            status=SolverStatus.INFEASIBLE, solve_time_ms=solve_time
        )

    commodity_flows = {}
    for k, comm in enumerate(commodities):
        arc_flows = {}
        for i, (src, tgt) in enumerate(arc_keys):
            if flow[(k, i)].solution_value() > 1e-6:
                arc_flows[f"{src}->{tgt}"] = round(flow[(k, i)].solution_value(), 6)
        commodity_flows[comm["name"]] = arc_flows

    return MultiCommodityFlowResult(
        status=SolverStatus.OPTIMAL
        if status == pywraplp.Solver.OPTIMAL
        else SolverStatus.FEASIBLE,
        total_cost=round(solver.Objective().Value(), 6),
        commodity_flows=commodity_flows,
        solve_time_ms=solve_time,
    )


def solve_transshipment(
    sources: list[str],
    transshipment_nodes: list[str],
    destinations: list[str],
    supplies: dict[str, int],
    demands: dict[str, int],
    costs: dict[str, dict[str, float]],
    capacities: dict[str, dict[str, float]] | None = None,
) -> "MinCostFlowResult":
    """
    Solve Transshipment Problem - ship goods through intermediate nodes.

    Args:
        sources: Source node names.
        transshipment_nodes: Intermediate node names.
        destinations: Destination node names.
        supplies: Supply at each source.
        demands: Demand at each destination.
        costs: costs[from][to] = unit shipping cost.
        capacities: Optional capacities[from][to] = max flow.

    Returns:
        MinCostFlowResult with flows and total cost.
    """

    all_nodes = sources + transshipment_nodes + destinations

    # Build arcs from cost matrix
    arcs = []
    for src, dests in costs.items():
        for dst, cost in dests.items():
            cap = capacities.get(src, {}).get(dst, 10000) if capacities else 10000
            arcs.append(
                {"source": src, "target": dst, "capacity": int(cap), "cost": int(cost)}
            )

    # Build supplies dict (sources positive, destinations negative)
    node_supplies = {}
    for s in sources:
        node_supplies[s] = supplies.get(s, 0)
    for t in transshipment_nodes:
        node_supplies[t] = 0
    for d in destinations:
        node_supplies[d] = -demands.get(d, 0)

    return solve_min_cost_flow(all_nodes, arcs, node_supplies)
