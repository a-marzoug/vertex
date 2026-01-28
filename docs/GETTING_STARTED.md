# Getting Started with Vertex

This guide will help you set up the Vertex MCP Server and run your first optimization problem.

## Prerequisites

- **python 3.13+** or **Docker**
- A Matrix Control Protocol (MCP) client, such as:
  - [Claude Desktop App](https://claude.ai/download)
  - [MCP Inspector](https://github.com/modelcontextprotocol/inspector)

## Installation

You can run Vertex either as a Docker container (recommended for production) or directly via Python (recommended for development).

### Option 1: Docker (Recommended)

1. **Pull the image** (or build locally):

    ```bash
    docker build -t vertex-mcp .
    ```

2. **Run the container**:

    ```bash
    docker run -p 8000:8000 vertex-mcp
    ```

The server will be available at `http://localhost:8000/sse`.

### Option 2: Python (uv)

We use `uv` for fast dependency management.

1. **Clone the repository**:

    ```bash
    git clone https://github.com/a-marzoug/vertex.git
    cd vertex
    ```

2. **Install dependencies**:

    ```bash
    uv sync
    ```

3. **Run the server**:

    ```bash
    uv run vertex
    ```

    or for development with auto-reload:

    ```bash
    uv run mcp-server-vertex --dev
    ```

## Connecting Vertex to Claude Desktop

To use Vertex with Claude, configure your `claude_desktop_config.json`:

### For Docker Users

```json
{
  "mcpServers": {
    "vertex": {
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

*Note: The above config uses stdio transport. For SSE (HTTP), you would need a bridging client.*

### For Local Python Users

```json
{
  "mcpServers": {
    "vertex": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/vertex",
        "run",
        "vertex"
      ]
    }
  }
}
```

## Verifying Installation

You can use the **MCP Inspector** to test tools without a full LLM.

1. Run the inspector:

    ```bash
    npx @modelcontextprotocol/inspector uv run vertex
    ```

2. Open the URL provided in the terminal (usually `http://localhost:5173`).
3. Look for "vertex" in the servers list.
4. Try listing tools. You should see `solve_linear_program`, `optimize_production_plan`, etc.

## Running Your First Optimization

Let's solve a simple **Diet Problem** using the text interface.

**Prompt equivalent:** (This is what you might ask Claude)
> "I need to plan a diet using Apples and Bananas. Apples cost $1 and provide 0.5g protein and 20g carbs. Bananas cost $0.5 and provide 1g protein and 25g carbs. I need at least 10g protein and 100g carbs. Minimize cost."

**Tool Call:** `optimize_diet_plan`

**Input:**

```json
{
  "foods": ["Apple", "Banana"],
  "nutrients": ["Protein", "Carbs"],
  "costs": {"Apple": 1.0, "Banana": 0.5},
  "nutrition_values": {
    "Apple": {"Protein": 0.5, "Carbs": 20},
    "Banana": {"Protein": 1.0, "Carbs": 25}
  },
  "min_requirements": {"Protein": 10, "Carbs": 100}
}
```

**Output:**
The server will return the optimal quantity of Apples and Bananas to buy!
