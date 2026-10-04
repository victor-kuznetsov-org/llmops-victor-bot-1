# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.4: Embeddings & Vector Search
# MAGIC Adapted: profile-based clients; the endpoint `vs-llmops` is shared and must already exist.

# COMMAND ----------

import time

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.reranker import DatabricksReranker

from arxiv_curator.config import load_config

PROFILE = "student-bot-1"
cfg = load_config("project_config.yml", "dev")
w = WorkspaceClient(profile=PROFILE)
# Locally, VectorSearchClient() cannot find credentials by itself (mlflow tracking URI is None):
# hand it the token the profile's SDK client resolves.
vsc = VectorSearchClient(
    workspace_url=w.config.host,
    personal_access_token=w.config.authenticate()["Authorization"].split()[1],
    disable_notice=True,
)

# The endpoint is created for us; never create it.
endpoint = cfg.vector_search_endpoint
names = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
assert endpoint in names, f"endpoint {endpoint} not listed: {names}"
print("endpoint listed:", endpoint)

# COMMAND ----------

index_name = f"{cfg.catalog}.{cfg.schema}.arxiv_index"
try:
    index = vsc.get_index(endpoint_name=endpoint, index_name=index_name)
    print("index exists")
except Exception:  # noqa: BLE001
    index = vsc.create_delta_sync_index(
        endpoint_name=endpoint,
        source_table_name=f"{cfg.catalog}.{cfg.schema}.arxiv_chunks",
        index_name=index_name,
        pipeline_type="TRIGGERED",
        primary_key="id",
        embedding_source_column="text",
        embedding_model_endpoint_name=cfg.embedding_endpoint,
    )
    print("index created")

# COMMAND ----------

for _ in range(60):
    st = index.describe().get("status", {})
    print(st.get("detailed_state"), st.get("message"))
    if st.get("ready"):
        break
    time.sleep(20)

# COMMAND ----------


def rows(results):
    cols = [c["name"] for c in results["manifest"]["columns"]]
    return [dict(zip(cols, r)) for r in results["result"]["data_array"]]


def show(label, results):
    print(f"--- {label}")
    for r in rows(results):
        print(f"  {r.get('title', '')[:60]} | {r.get('text', '')[:100]}")


q = "attention mechanisms in transformers"
show(
    "semantic",
    index.similarity_search(query_text=q, columns=["text", "id", "title"], num_results=3),
)
show(
    "filter year=2026",
    index.similarity_search(
        query_text=q,
        columns=["text", "id", "title", "year"],
        filters={"year": "2026"},
        num_results=3,
    ),
)
show(
    "hybrid",
    index.similarity_search(
        query_text=q, columns=["text", "id", "title"], num_results=3, query_type="hybrid"
    ),
)

# COMMAND ----------

try:
    show(
        "hybrid + rerank",
        index.similarity_search(
            query_text=q,
            columns=["text", "id", "title", "summary"],
            num_results=3,
            query_type="hybrid",
            reranker=DatabricksReranker(columns_to_rerank=["text", "title", "summary"]),
        ),
    )
except Exception as e:  # noqa: BLE001
    print("rerank failed:", repr(e)[:500])
