# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 1.3: arXiv Data Ingestion with Databricks Connect
# MAGIC Adapted: values from `project_config.yml`; table `arxiv_papers` in the student's schema.

# COMMAND ----------

from datetime import datetime

import arxiv
from databricks.connect import DatabricksSession
from pyspark.sql.types import ArrayType, LongType, StringType, StructField, StructType

from arxiv_curator.config import load_config

spark = DatabricksSession.builder.profile("student-bot-1").serverless(True).getOrCreate()
cfg = load_config("project_config.yml", "dev")
TABLE = f"{cfg.catalog}.{cfg.schema}.arxiv_papers"
print(TABLE)

# COMMAND ----------


def fetch_arxiv_papers(query: str = "cat:cs.AI OR cat:cs.LG", max_results: int = 50) -> list:
    """Fetch recent arXiv paper metadata."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    return [
        {
            "arxiv_id": r.entry_id.split("/")[-1],
            "title": r.title,
            "authors": [a.name for a in r.authors],
            "summary": r.summary,
            "published": int(r.published.strftime("%Y%m%d%H%M")),
            "updated": r.updated.isoformat() if r.updated else None,
            "categories": ", ".join(r.categories),
            "pdf_url": r.pdf_url,
            "primary_category": r.primary_category,
            "ingestion_timestamp": datetime.now().isoformat(),
            "processed": None,
            "volume_path": None,
        }
        for r in arxiv.Client().results(search)
    ]


papers = fetch_arxiv_papers(max_results=50)
print(f"Fetched {len(papers)} papers")

# COMMAND ----------

schema = StructType(
    [
        StructField("arxiv_id", StringType(), False),
        StructField("title", StringType(), False),
        StructField("authors", ArrayType(StringType()), True),
        StructField("summary", StringType(), True),
        StructField("published", LongType(), True),
        StructField("updated", StringType(), True),
        StructField("categories", StringType(), True),
        StructField("pdf_url", StringType(), True),
        StructField("primary_category", StringType(), True),
        StructField("ingestion_timestamp", StringType(), True),
        StructField("processed", LongType(), True),
        StructField("volume_path", StringType(), True),
    ]
)
df = spark.createDataFrame(papers, schema=schema)
df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(TABLE)

# COMMAND ----------

papers_df = spark.table(TABLE)
print(papers_df.count())
papers_df.select("arxiv_id", "title", "primary_category", "published").show(5, truncate=50)
