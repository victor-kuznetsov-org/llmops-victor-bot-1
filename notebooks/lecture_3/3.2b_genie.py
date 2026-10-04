# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.2b: Genie Space Integration
# MAGIC Adapted: the course creates a SQL warehouse; we use our own provisioned warehouse
# MAGIC (`warehouse_id` of `project_config.yml`) and never create one.

# COMMAND ----------

import json

from databricks.sdk import WorkspaceClient

from arxiv_curator.config import load_config

cfg = load_config("project_config.yml", "dev")
w = WorkspaceClient(profile="student-bot-1")
TITLE = "genie-llmops-victor-bot-1"

# COMMAND ----------

serialized_space = {
    "version": 1,
    "data_sources": {
        "tables": [
            {
                "identifier": f"{cfg.catalog}.{cfg.schema}.arxiv_papers",
                "column_configs": [
                    {"column_name": "authors"},
                    {"column_name": "ingest_ts", "get_example_values": True},
                    {"column_name": "paper_id", "get_example_values": True},
                    {
                        "column_name": "pdf_url",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                    {"column_name": "processed", "get_example_values": True},
                    {"column_name": "published", "get_example_values": True},
                    {
                        "column_name": "summary",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                    {
                        "column_name": "title",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                    {
                        "column_name": "volume_path",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                ],
            }
        ]
    },
}

space = w.genie.create_space(
    warehouse_id=cfg.warehouse_id,
    serialized_space=json.dumps(serialized_space),
    title=TITLE,
)
space_id = space.space_id
print("Created Genie space:", TITLE, space_id)

# COMMAND ----------

space = w.genie.get_space(space_id=space_id, include_serialized_space=True)
print(json.loads(space.serialized_space)["data_sources"]["tables"][0]["identifier"])

# COMMAND ----------

conversation = w.genie.start_conversation_and_wait(
    space_id=space_id, content="Find the last 10 papers published"
)
print(json.dumps(conversation.as_dict(), indent=2, default=str)[:3000])

# COMMAND ----------

message = w.genie.create_message_and_wait(
    space_id=space_id,
    conversation_id=conversation.conversation_id,
    content="Return the list of authors of the last 10 papers published",
)
print(json.dumps(message.as_dict(), indent=2, default=str)[:3000])
