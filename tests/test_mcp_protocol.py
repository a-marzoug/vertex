"""MCP protocol integration tests."""

import pytest
from mcp.client.session import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session

from vertex.server import mcp


@pytest.fixture
def anyio_backend():
    """Use asyncio backend for tests."""
    return "asyncio"


@pytest.fixture
async def mcp_session():
    """Create an in-memory MCP client-server session."""
    async with create_connected_server_and_client_session(
        mcp, raise_exceptions=True
    ) as session:
        yield session


@pytest.mark.anyio
async def test_mcp_initialization(mcp_session: ClientSession):
    """Test MCP protocol initialization handshake."""
    # Initialize is called automatically by the fixture
    # Verify server info
    assert mcp_session._server_info is not None
    assert mcp_session._server_info.name == "Vertex"
    
    # Verify capabilities
    assert mcp_session._server_capabilities is not None
    assert mcp_session._server_capabilities.tools is not None


@pytest.mark.anyio
async def test_list_tools(mcp_session: ClientSession):
    """Test listing available tools via MCP."""
    tools = await mcp_session.list_tools()
    
    # Verify we have tools
    assert len(tools.tools) > 0
    
    # Check for key tools
    tool_names = [t.name for t in tools.tools]
    assert "solve_linear_program" in tool_names
    assert "solve_mixed_integer_program" in tool_names
    assert "optimize_production_plan" in tool_names
    
    # Verify tool has proper schema
    lp_tool = next(t for t in tools.tools if t.name == "solve_linear_program")
    assert lp_tool.description is not None
    assert lp_tool.inputSchema is not None


@pytest.mark.anyio
async def test_list_prompts(mcp_session: ClientSession):
    """Test listing available prompts via MCP."""
    prompts = await mcp_session.list_prompts()
    
    # Verify we have prompts
    assert len(prompts.prompts) > 0
    
    # Check for key prompts
    prompt_names = [p.name for p in prompts.prompts]
    assert "select_optimization_approach" in prompt_names
    assert "formulate_lp_problem" in prompt_names


@pytest.mark.anyio
async def test_call_simple_tool(mcp_session: ClientSession):
    """Test calling a simple optimization tool via MCP."""
    result = await mcp_session.call_tool(
        "solve_linear_program",
        {
            "variables": ["x", "y"],
            "objective": {"x": 3, "y": 2},
            "objective_sense": "maximize",
            "constraints": [
                {"coefficients": {"x": 2, "y": 1}, "sense": "<=", "rhs": 20},
                {"coefficients": {"x": 1, "y": 1}, "sense": "<=", "rhs": 12},
            ],
        },
    )
    
    # Verify result structure
    assert len(result.content) > 0
    assert result.content[0].type == "text"
    
    # Verify solution is valid
    assert "optimal" in result.content[0].text.lower()


@pytest.mark.anyio
async def test_call_tool_with_validation_error(mcp_session: ClientSession):
    """Test that validation errors are properly returned via MCP."""
    with pytest.raises(Exception) as exc_info:
        await mcp_session.call_tool(
            "solve_linear_program",
            {
                "variables": [],  # Empty variables should fail validation
                "objective": {},
                "objective_sense": "maximize",
                "constraints": [],
            },
        )
    
    # Verify error is meaningful
    assert "variables" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower()


@pytest.mark.anyio
async def test_get_prompt(mcp_session: ClientSession):
    """Test getting a prompt via MCP."""
    result = await mcp_session.get_prompt(
        "select_optimization_approach",
        {"problem_description": "I need to schedule workers to shifts"},
    )
    
    # Verify prompt result
    assert len(result.messages) > 0
    assert result.messages[0].role == "user"
    assert "schedule" in result.messages[0].content.text.lower()


@pytest.mark.anyio
async def test_concurrent_tool_calls(mcp_session: ClientSession):
    """Test multiple concurrent tool calls via MCP."""
    import asyncio
    
    # Call multiple tools concurrently
    tasks = [
        mcp_session.call_tool(
            "solve_linear_program",
            {
                "variables": ["x"],
                "objective": {"x": 1},
                "objective_sense": "maximize",
                "constraints": [{"coefficients": {"x": 1}, "sense": "<=", "rhs": 10}],
            },
        )
        for _ in range(3)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # All should succeed
    assert len(results) == 3
    for result in results:
        assert len(result.content) > 0


@pytest.mark.anyio
async def test_tool_with_complex_output(mcp_session: ClientSession):
    """Test tool that returns structured data."""
    result = await mcp_session.call_tool(
        "get_model_statistics",
        {
            "variables": ["x", "y", "z"],
            "constraints": [
                {"coefficients": {"x": 1, "y": 1}, "sense": "<=", "rhs": 10},
                {"coefficients": {"y": 1, "z": 1}, "sense": ">=", "rhs": 5},
            ],
        },
    )
    
    # Verify structured output
    assert len(result.content) > 0
    text = result.content[0].text
    assert "variables" in text.lower()
    assert "constraints" in text.lower()
