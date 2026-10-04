"""ToolInfo wraps a function with its OpenAI tool spec."""

from arxiv_curator.mcp import ToolInfo


def test_tool_info_executes() -> None:
    tool = ToolInfo(
        name="add",
        spec={"type": "function", "function": {"name": "add"}},
        exec_fn=lambda a, b: a + b,
    )

    assert tool.exec_fn(a=1, b=2) == 3
