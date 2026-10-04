"""MCP tooling: tool description and wrappers around Databricks managed MCP servers."""

from collections.abc import Callable

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from pydantic import BaseModel


class ToolInfo(BaseModel):
    """A tool an agent can call."""

    name: str
    spec: dict
    exec_fn: Callable


def create_managed_exec_fn(server_url: str, tool_name: str, w: WorkspaceClient) -> Callable:
    """Build a function that calls one tool of a managed MCP server."""

    def exec_fn(**kwargs):
        client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
        response = client.call_tool(tool_name, kwargs)
        return "".join([c.text for c in response.content])

    return exec_fn


async def create_mcp_tools(w: WorkspaceClient, url_list: list[str]) -> list[ToolInfo]:
    """List the tools of each MCP server and return them as ToolInfo objects."""
    tools = []
    for server_url in url_list:
        mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
        for tool in mcp_client.list_tools():
            input_schema = tool.inputSchema.copy() if tool.inputSchema else {}
            tool_spec = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "parameters": input_schema,
                    "description": tool.description or f"Tool: {tool.name}",
                },
            }
            exec_fn = create_managed_exec_fn(server_url, tool.name, w)
            tools.append(ToolInfo(name=tool.name, spec=tool_spec, exec_fn=exec_fn))
    return tools
