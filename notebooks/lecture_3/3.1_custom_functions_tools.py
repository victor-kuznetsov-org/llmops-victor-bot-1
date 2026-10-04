# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.1: Custom Functions & Tools for Agents
# MAGIC Adapted: profile-based clients, names from `project_config.yml`, Databricks Connect for Spark.

# COMMAND ----------

import json

import mlflow
from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from mlflow.entities import SpanType
from pyspark.sql import functions as F

from arxiv_curator.config import load_config
from arxiv_curator.mcp import ToolInfo

PROFILE = "student-bot-1"
cfg = load_config("project_config.yml", "dev")
w = WorkspaceClient(profile=PROFILE)
spark = DatabricksSession.builder.profile(PROFILE).serverless(True).getOrCreate()
vsc = VectorSearchClient(
    workspace_url=w.config.host,
    personal_access_token=w.config.authenticate()["Authorization"].split()[1],
    disable_notice=True,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculator tool and its specification


# COMMAND ----------


def calculator(operation: str, a: float, b: float) -> float:
    """Perform basic arithmetic operations."""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float("inf"),
    }
    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")
    return operations[operation](a, b)


print("5 * 3 =", calculator("multiply", 5, 3))

calculator_tool_spec = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform basic arithmetic operations (add, subtract, multiply, divide)",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform",
                },
                "a": {"type": "number", "description": "The first number"},
                "b": {"type": "number", "description": "The second number"},
            },
            "required": ["operation", "a", "b"],
        },
    },
}
print(json.dumps(calculator_tool_spec, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector search tool


# COMMAND ----------


def parse_vector_search_results(results):
    """Parse vector search results from array format to dict format."""
    columns = [col["name"] for col in results.get("manifest", {}).get("columns", [])]
    data_array = results.get("result", {}).get("data_array", [])
    return [dict(zip(columns, row_data, strict=False)) for row_data in data_array]


@mlflow.trace(span_type=SpanType.TOOL)
def search_papers(query: str, num_results: int = 5, year_filter: str | None = None) -> str:
    """Search for relevant papers using vector search."""
    index = vsc.get_index(
        endpoint_name=cfg.vector_search_endpoint,
        index_name=f"{cfg.catalog}.{cfg.schema}.arxiv_index",
    )
    search_params = {
        "query_text": query,
        "columns": ["text", "title", "paper_id", "authors", "year"],
        "num_results": num_results,
        "query_type": "hybrid",
    }
    if year_filter:
        search_params["filters"] = {"year": year_filter}
    results = index.similarity_search(**search_params)
    papers = [
        {
            "title": row.get("title", "N/A"),
            "paper_id": row.get("paper_id", "N/A"),
            "authors": str(row.get("authors", "N/A")),
            "year": row.get("year", "N/A"),
            "excerpt": row.get("text", "")[:200] + "...",
        }
        for row in parse_vector_search_results(results)
    ]
    return json.dumps(papers, indent=2)


print(search_papers("machine learning", num_results=2))

search_papers_tool_spec = {
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": (
            "Search for academic papers using semantic search. "
            "Returns relevant papers with titles, authors, and excerpts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing what papers to find",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5,
                },
                "year_filter": {
                    "type": "string",
                    "description": "Optional year filter to limit results (e.g., '2024')",
                },
            },
            "required": ["query"],
        },
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paper statistics tool


# COMMAND ----------


@mlflow.trace(span_type=SpanType.TOOL)
def get_paper_statistics(year: int | None = None) -> str:
    """Get statistics about papers in the database."""
    df = spark.table(f"{cfg.catalog}.{cfg.schema}.arxiv_papers")
    # published is YYYYMMDDHHMM
    df = df.withColumn("year", (F.col("published") / 100000000).cast("int"))
    if year:
        df = df.filter(F.col("year") == year)
    total_papers = df.count()
    year_range = {"min": None, "max": None}
    if total_papers > 0:
        row = df.agg(F.min("year").alias("lo"), F.max("year").alias("hi")).collect()[0]
        year_range = {"min": row["lo"], "max": row["hi"]}
    stats = {
        "total_papers": total_papers,
        "year_range": year_range,
        "filter_applied": f"year={year}" if year else "none",
    }
    return json.dumps(stats, indent=2)


get_paper_statistics_tool_spec = {
    "type": "function",
    "function": {
        "name": "get_paper_statistics",
        "description": "Get statistics (count, year range) about the papers in the database.",
        "parameters": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Optional year filter (e.g. 2026)"}
            },
        },
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool registry and execution


# COMMAND ----------

calculator_tool = ToolInfo(name="calculator", spec=calculator_tool_spec, exec_fn=calculator)
search_papers_tool = ToolInfo(
    name="search_papers", spec=search_papers_tool_spec, exec_fn=search_papers
)
stats_tool = ToolInfo(
    name="get_paper_statistics",
    spec=get_paper_statistics_tool_spec,
    exec_fn=get_paper_statistics,
)


class ToolRegistry:
    """Registry for managing agent tools."""

    def __init__(self):
        self._tools: dict[str, ToolInfo] = {}

    def register(self, tool: ToolInfo) -> None:
        self._tools[tool.name] = tool
        print(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> ToolInfo:
        if name not in self._tools:
            raise ValueError(f"Tool not found: {name}")
        return self._tools[name]

    def get_all_specs(self) -> list[dict]:
        return [tool.spec for tool in self._tools.values()]

    def execute(self, name: str, args: dict):
        return self.get_tool(name).exec_fn(**args)

    def list_tools(self) -> list[str]:
        return list(self._tools)


registry = ToolRegistry()
for t in (calculator_tool, search_papers_tool, stats_tool):
    registry.register(t)

# COMMAND ----------

print("calculator:", registry.execute("calculator", {"operation": "add", "a": 10, "b": 5}))
print(
    "search_papers:",
    registry.execute("search_papers", {"query": "neural networks", "num_results": 3}),
)
print("get_paper_statistics:", registry.execute("get_paper_statistics", {}))
