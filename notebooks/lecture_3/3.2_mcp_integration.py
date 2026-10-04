# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.2: Model Context Protocol (MCP) Integration
# MAGIC Adapted: profile-based `WorkspaceClient`, names from `project_config.yml`.

# COMMAND ----------

import asyncio
import json

import nest_asyncio
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient

from arxiv_curator.config import load_config
from arxiv_curator.mcp import create_mcp_tools

nest_asyncio.apply()

cfg = load_config("project_config.yml", "dev")
w = WorkspaceClient(profile="student-bot-1")
host = w.config.host

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Search MCP: list and call

# COMMAND ----------

vector_search_mcp_url = f"{host}/api/2.0/mcp/vector-search/{cfg.catalog}/{cfg.schema}"
print(vector_search_mcp_url)

vs_mcp_client = DatabricksMCPClient(server_url=vector_search_mcp_url, workspace_client=w)
vs_tools = vs_mcp_client.list_tools()
print(f"Vector Search MCP Tools ({len(vs_tools)}):")
for tool in vs_tools:
    print("Tool:", tool.name)
    print("Description:", tool.description)
    if tool.inputSchema:
        print("Parameters:", list(tool.inputSchema.get("properties", {})))

# COMMAND ----------

tool_name = f"{cfg.catalog}__{cfg.schema}__arxiv_index"
search_result = vs_mcp_client.call_tool(
    tool_name, {"query": "machine learning and neural networks"}
)
for content in search_result.content:
    print(content.text[:1500])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Genie MCP (space created in 3.2b, id in `project_config.yml`)

# COMMAND ----------

mcp_urls = [vector_search_mcp_url]
if cfg.genie_space_id:
    genie_mcp_url = f"{host}/api/2.0/mcp/genie/{cfg.genie_space_id}"
    mcp_urls.append(genie_mcp_url)
    genie_tools = DatabricksMCPClient(server_url=genie_mcp_url, workspace_client=w).list_tools()
    print(f"Genie MCP Tools ({len(genie_tools)}):", [t.name for t in genie_tools])

# COMMAND ----------

mcp_tools = asyncio.run(create_mcp_tools(w, mcp_urls))
print(f"Loaded {len(mcp_tools)} tools:", [t.name for t in mcp_tools])

tools_dict = {tool.name: tool for tool in mcp_tools}
print(tools_dict[tool_name].exec_fn(query="deep learning architectures")[:1000])

# COMMAND ----------

print(json.dumps(mcp_tools[0].spec, indent=2))
