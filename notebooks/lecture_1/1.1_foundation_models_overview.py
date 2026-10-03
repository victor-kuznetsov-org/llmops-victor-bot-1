# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 1.1: Foundation Models Overview
# MAGIC Adapted: calls the course AI Gateway chat service from `project_config.yml`.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI

from arxiv_curator.config import load_config

cfg = load_config("project_config.yml", "dev")
w = WorkspaceClient()

# COMMAND ----------

# Call the model through the gateway (no personal access token needed)
client = DatabricksOpenAI(use_ai_gateway=True)
response = client.chat.completions.create(
    model=cfg.llm_endpoint,
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain LLMOps in 3 sentences."},
    ],
    max_tokens=200,
    temperature=0.7,
)
print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.total_tokens}")

# COMMAND ----------


def calculate_api_cost(
    input_tokens: int, output_tokens: int, input_dbu_per_1m: float, output_dbu_per_1m: float
) -> float:
    """Calculate DBU cost for pay-per-token API."""
    return (input_tokens / 1_000_000) * input_dbu_per_1m + (
        output_tokens / 1_000_000
    ) * output_dbu_per_1m


def calculate_provisioned_cost(hours: int, dbu_per_hour: float) -> float:
    """Calculate DBU cost for provisioned throughput."""
    return hours * dbu_per_hour


print(f"Pay-per-token: {calculate_api_cost(1_000_000, 500_000, 7.143, 21.429):.2f} DBUs")
print(f"Provisioned 24h: {calculate_provisioned_cost(24, 42.857):.2f} DBUs")
