# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.3: Chunking strategies
# MAGIC Adapted: Databricks Connect session with the student profile, config from `project_config.yml`.

# COMMAND ----------

import re

import pandas as pd
from databricks.connect import DatabricksSession
from pyspark.sql import functions as F

from arxiv_curator.config import load_config

spark = DatabricksSession.builder.profile("student-bot-1").serverless(True).getOrCreate()
cfg = load_config("project_config.yml", "dev")
chunks_df = spark.table(f"{cfg.catalog}.{cfg.schema}.arxiv_chunks")
print("Total chunks:", chunks_df.count())
chunks_df.show(5, truncate=50)

# COMMAND ----------

stats = chunks_df.select(
    F.avg(F.length("text")).alias("avg_length"),
    F.min(F.length("text")).alias("min_length"),
    F.max(F.length("text")).alias("max_length"),
    F.count("*").alias("total_chunks"),
).collect()[0]
print(stats)

# COMMAND ----------


def fixed_size_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


def sentence_chunking(text: str, max_sentences: int = 5) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [" ".join(sentences[i : i + max_sentences]) for i in range(0, len(sentences), max_sentences)]


# longest chunk, so that the strategies have something to split
sample_text = chunks_df.orderBy(F.length("text").desc()).select("text").first()["text"]
fixed = fixed_size_chunking(sample_text)
sent = sentence_chunking(sample_text)
print(f"original {len(sample_text)} chars -> fixed {len(fixed)} chunks, sentence {len(sent)} chunks")
print(fixed[0][:200])
print(sent[0][:200])

# COMMAND ----------

print(
    pd.DataFrame(
        {
            "Strategy": ["AI Parse (Databricks)", "Fixed-Size", "Sentence-Based"],
            "Pros": [
                "Structure detection, preserves tables",
                "Simple, predictable size",
                "Preserves semantic units",
            ],
            "Cons": ["Requires AI Parse", "May break sentences", "Variable chunk sizes"],
        }
    ).to_string(index=False)
)
