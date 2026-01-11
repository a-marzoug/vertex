"""Network optimization tools."""

from vertex.models.network import MaxFlowResult, MinCostFlowResult, ShortestPathResult
from vertex.solvers.network import solve_max_flow, solve_min_cost_flow, solve_shortest_path


def compute_max_flow(
    nodes: list[str],
    arcs: list[dict],
    source: str,
    sink: str,
) -> MaxFlowResult:
    """
    Find maximum flow from source to sink in a network.

    Args:
        nodes: List of node names. Example: ["S", "A", "B", "T"]
        arcs: List of arcs with 'source', 'target', 'capacity'.
            Example: [{"source": "S", "target": "A", "capacity": 10}]
        source: Source node name.
        sink: Sink node name.

    Returns:
        MaxFlowResult with max_flow, arc_flows, and min-cut.
    """
    return solve_max_flow(nodes, arcs, source, sink)


def compute_min_cost_flow(
    nodes: list[str],
    arcs: list[dict],
    supplies: dict[str, int],
) -> MinCostFlowResult:
    """
    Find minimum cost flow satisfying node supplies and demands.

    Args:
        nodes: List of node names.
        arcs: List of arcs with 'source', 'target', 'capacity', 'cost'.
            Example: [{"source": "A", "target": "B", "capacity": 10, "cost": 5}]
        supplies: Node supplies (positive) and demands (negative).
            Example: {"source": 10, "sink": -10}

    Returns:
        MinCostFlowResult with total_cost and arc_flows.
    """
    return solve_min_cost_flow(nodes, arcs, supplies)


def compute_shortest_path(
    nodes: list[str],
    arcs: list[dict],
    source: str,
    target: str,
) -> ShortestPathResult:
    """
    Find shortest path between two nodes.

    Args:
        nodes: List of node names.
        arcs: List of arcs with 'source', 'target', 'cost'.
            Example: [{"source": "A", "target": "B", "cost": 5}]
        source: Start node.
        target: End node.

    Returns:
        ShortestPathResult with distance and path.
    """
    return solve_shortest_path(nodes, arcs, source, target)
