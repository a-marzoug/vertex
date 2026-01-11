#!/bin/bash
# Wrapper script to run Vertex MCP server via Docker with stdio transport
# For WSL2 + Claude Desktop on Windows

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Always use native WSL2 Docker (not docker.exe)
# This works better for stdio communication from Claude Desktop
DOCKER_CMD="docker"
COMPOSE_CMD="docker-compose"

# Build the image if it doesn't exist using docker-compose
if ! $DOCKER_CMD image inspect vertex:stdio >/dev/null 2>&1; then
    echo "Building vertex Docker image..." >&2
    cd "$SCRIPT_DIR" && $COMPOSE_CMD build vertex-stdio >&2
fi

# Run the container with stdio transport
# Use --rm to clean up, -i for interactive stdin
exec $DOCKER_CMD run --rm -i vertex:stdio python -m vertex.server
