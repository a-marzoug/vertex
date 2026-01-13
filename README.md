# Vertex

Operations Research MCP Server for decision makers.

## Features

- **Linear Programming** - Solve optimization problems with linear objectives and constraints
- **Mixed-Integer Programming** - Handle discrete decisions with integer and binary variables
- **Domain Templates** - Pre-built tools for production, diet, portfolio, assignment, knapsack, and facility location
- **Natural Language Support** - Prompts to help formulate problems from descriptions
- **Scalable Deployment** - Stateless HTTP transport for Docker/Kubernetes

## Documentation

- [🚀 Getting Started](docs/GETTING_STARTED.md) - Installation and first steps
- [📚 API Reference](docs/API_REFERENCE.md) - Comprehensive tool guide
- [💼 Business Cases](docs/BUSINESS_CASES.md) - Industry applications and examples
- [🏗️ Architecture](docs/ARCHITECTURE.md) - System design and components
- [🗺️ Roadmap](docs/ROADMAP.md) - Future plans and vision

## Installation

```bash
uv sync
```

## Usage

### Run the Server

```bash
# Stdio mode (default, for Claude Desktop)
uv run vertex

# HTTP mode (for web clients and Docker)
uv run vertex --http

# Or directly
uv run python -m vertex.server
uv run python -m vertex.server --http
```

When running with `--http`, server starts at `http://localhost:8000/mcp`

### Docker Build & Run

You can build and run the server using either Docker Compose or plain Docker commands.

#### Option 1: Docker Compose (Recommended)

Compose profiles allow you to easily switch between `stdio` (for Claude) and `http` (for web/external) modes.

```bash
# Build the stdio image (for Claude Desktop)
docker compose build vertex-stdio

# Build and run the HTTP server (for web apps)
docker compose --profile http up --build -d
```

#### Option 2: Manual Docker Build

If you prefer building images manually:

```bash
# For Stdio (Claude Desktop)
docker build -f docker/Dockerfile.stdio -t vertex-mcp .

# For HTTP (Web Apps)
docker build -f docker/Dockerfile.http -t vertex-mcp-http .
docker run -p 8000:8000 vertex-mcp-http
```

### Connect from Claude Desktop

Add to your `claude_desktop_config.json`:

- **Linux**: `~/.config/Claude/claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "vertex-mcp": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "vertex-mcp"
      ]
    }
  }
}
```

*Note: Ensure you have built the image named `vertex-mcp` (or `vertex-stdio`) first.*

### Connect from Vercel AI SDK (v6)

To use use Vertex with the Vercel AI SDK, you can use the `@ai-sdk/mcp` package.

```bash
npm install @ai-sdk/mcp ai
```

```typescript
import { createMCPClient } from '@ai-sdk/mcp';
// import { Experimental_StdioMCPTransport } from '@ai-sdk/mcp/mcp-stdio'; // For Stdio

// Connect to HTTP Deployment
const client = await createMCPClient({
  transport: {
    type: 'http',
    url: 'http://localhost:8000/mcp',
  },
});

// Or connect via Stdio (requires local Docker or python setup)
/*
const client = await createMCPClient({
  transport: new Experimental_StdioMCPTransport({
    command: 'docker',
    args: ['run', '-i', '--rm', 'vertex-mcp'],
  }),
});
*/

const tools = await client.tools();
console.log(tools);
```

### Connect from LangChain / LangGraph (Python)

You can use the `langchain-mcp-adapters` package to integrate Vertex with LangChain.

```bash
pip install langchain-mcp-adapters
```

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

async def main():
    client = MultiServerMCPClient(
        {
            "vertex": {
                "transport": "http",
                "url": "http://localhost:8000/mcp", 
            }
        }
    )

    tools = await client.get_tools()
    
    # Use tools with any LangChain agent
    # agent = create_agent(model, tools)
```

### Connect from CrewAI

CrewAI supports MCP servers natively.

```bash
pip install crewai
```

```python
from crewai import Agent
from crewai.mcp import MCPServerHTTP

agent = Agent(
    role="Optimization Expert",
    goal="Solve complex scheduling and logistics problems",
    backstory="Expert mathematician with access to advanced solvers.",
    mcps=[
        MCPServerHTTP(
            url="http://localhost:8000/mcp"
        )
    ]
)
```

## Tools

| Tool | Description |
|------|-------------|
| `solve_linear_program` | Solve arbitrary LP problems |
| `analyze_lp_sensitivity` | Get shadow prices and reduced costs for LP solutions |
| `analyze_what_if_scenario` | Vary parameters and see impact on objective |
| `diagnose_infeasibility` | Find conflicting constraints in infeasible problems |
| `get_model_statistics` | Get model size, sparsity, and type breakdown |
| `solve_pareto_frontier` | Multi-objective optimization - find Pareto frontier |
| `solve_sudoku_puzzle` | Solve Sudoku using constraint programming |
| `solve_n_queens_puzzle` | Solve N-Queens problem |
| `find_alternative_optimal_solutions` | Find multiple near-optimal solutions |
| `optimize_production_plan` | Maximize profit given resource constraints |
| `optimize_diet_plan` | Minimize cost meeting nutritional requirements |
| `optimize_investment_portfolio` | Maximize returns with allocation constraints |
| `solve_mixed_integer_program` | Solve MIP with integer/binary variables |
| `optimize_worker_assignment` | Assign workers to tasks minimizing cost |
| `optimize_knapsack_selection` | Select items to maximize value within capacity |
| `optimize_facility_locations` | Decide which facilities to open |
| `optimize_inventory_eoq` | Economic Order Quantity optimization |
| `optimize_workforce` | Schedule workers to shifts |
| `optimize_healthcare_resources` | Allocate medical resources across locations |
| `optimize_supply_chain_network` | Design supply chain with facility location |
| `find_max_flow` | Find maximum flow from source to sink |
| `find_min_cost_flow` | Find minimum cost flow satisfying supplies/demands |
| `find_shortest_path` | Find shortest path between two nodes |
| `find_minimum_spanning_tree` | Find MST connecting all nodes |
| `find_multi_commodity_flow` | Route multiple commodities through network |
| `solve_transshipment` | Ship goods through intermediate nodes |
| `solve_tsp` | Traveling Salesman Problem - shortest tour |
| `solve_vrp` | Vehicle Routing Problem with capacity constraints |
| `solve_vrp_time_windows` | VRP with time window constraints |
| `solve_job_shop` | Job Shop Scheduling - minimize makespan |
| `solve_rcpsp` | Resource-Constrained Project Scheduling |
| `solve_flexible_job_shop` | Flexible Job Shop - tasks on alternative machines |
| `solve_bin_packing` | Bin Packing - minimize bins used |
| `solve_set_cover` | Set Covering - minimum cost coverage |
| `solve_graph_coloring` | Graph Coloring - minimize colors used |
| `solve_cutting_stock` | Cutting Stock - minimize material waste |
| `solve_two_stage_stochastic` | Two-stage stochastic programming with recourse |
| `solve_newsvendor` | Single-period stochastic inventory (newsvendor) |
| `solve_lot_sizing` | Dynamic lot sizing (Wagner-Whitin algorithm) |
| `solve_robust_production` | Robust optimization under demand uncertainty |
| `analyze_mm1_queue` | M/M/1 queue performance analysis |
| `analyze_mmc_queue` | M/M/c queue performance analysis |
| `solve_flow_shop` | Flow shop scheduling (same machine sequence) |
| `solve_parallel_machines` | Parallel machine scheduling |
| `simulate_newsvendor` | Monte Carlo simulation for newsvendor |
| `simulate_production` | Monte Carlo simulation for production |
| `solve_crew_schedule` | Crew/shift scheduling with constraints |
| `solve_chance_constrained` | Chance-constrained production planning |
| `solve_2d_bin_packing` | 2D rectangle bin packing |
| `solve_network_design` | Capacitated network design |
| `solve_quadratic_assignment_problem` | Facility layout (QAP) |
| `solve_steiner_tree` | Connect terminals with minimum cost |
| `optimize_multi_echelon` | Multi-echelon inventory optimization |
| `solve_qp` | Quadratic Programming (convex) |
| `optimize_portfolio_variance` | Markowitz mean-variance portfolio |

## Prompts

| Prompt | Description |
|--------|-------------|
| `formulate_lp_problem` | Guide for extracting LP components from natural language |
| `formulate_mip_problem` | Guide for formulating mixed-integer problems |
| `interpret_lp_solution` | Explain optimization results to decision makers |

## Example

```python
from vertex.tools.templates.production import optimize_production

result = optimize_production(
    products=["chairs", "tables"],
    resources=["wood", "labor"],
    profits={"chairs": 45, "tables": 80},
    requirements={
        "chairs": {"wood": 5, "labor": 2},
        "tables": {"wood": 20, "labor": 5},
    },
    availability={"wood": 400, "labor": 100},
)

print(f"Optimal profit: ${result.total_profit:.2f}")
print(f"Production plan: {result.production_plan}")
```

## License

MIT
