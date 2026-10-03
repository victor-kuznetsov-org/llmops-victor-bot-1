# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.2: PDF parsing with ai_parse_document
# MAGIC Adapted: values from `project_config.yml`; PDFs are downloaded under the temp dir and
# MAGIC uploaded with the Files API (no `/Volumes` on a laptop); 3 papers only.

# COMMAND ----------

import json
import re
import tempfile
import time
from pathlib import Path

import arxiv
from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import col, concat_ws, explode, udf

from arxiv_curator.config import load_config

PROFILE = "student-bot-1"
spark = DatabricksSession.builder.profile(PROFILE).serverless(True).getOrCreate()
w = WorkspaceClient(profile=PROFILE)
cfg = load_config("project_config.yml", "dev")
catalog, schema, volume = cfg.catalog, cfg.schema, cfg.volume
metadata_table = f"{catalog}.{schema}.arxiv_papers"
N_PAPERS = 3

# COMMAND ----------

# Unprocessed papers from lecture 1.3
unprocessed = spark.table(metadata_table).filter(F.col("processed").isNull())
papers_to_process = unprocessed.limit(N_PAPERS).collect()
print(f"Will process {len(papers_to_process)} papers")

# COMMAND ----------

end = time.strftime("%Y%m%d%H%M", time.gmtime())
pdf_dir = f"/Volumes/{catalog}/{schema}/{volume}/{end}"
tmp_dir = Path(tempfile.mkdtemp())
client = arxiv.Client()
records = []

for row in papers_to_process:
    paper_id = row.arxiv_id
    try:
        paper = next(client.results(arxiv.Search(id_list=[paper_id])))
        local = paper.download_pdf(dirpath=str(tmp_dir), filename=f"{paper_id}.pdf")
        with open(local, "rb") as f:
            w.files.upload(f"{pdf_dir}/{paper_id}.pdf", f, overwrite=True)
        records.append(
            {
                "paper_id": row.arxiv_id,
                "processed": int(end),
                "volume_path": f"{pdf_dir}/{paper_id}.pdf",
            }
        )
        print("downloaded", paper_id)
    except Exception as e:
        logger.warning(f"Paper {paper_id} was not successfully processed: {e}")
    time.sleep(3)
print(len(records), "PDFs uploaded to", pdf_dir)

# COMMAND ----------

if records:
    upd = spark.createDataFrame(
        records,
        T.StructType(
            [
                T.StructField("paper_id", T.StringType(), False),
                T.StructField("processed", T.LongType(), True),
                T.StructField("volume_path", T.StringType(), True),
            ]
        ),
    )
    upd.createOrReplaceTempView("pdf_updates")
    spark.sql(f"""
        MERGE INTO {metadata_table} target USING pdf_updates source
        ON target.arxiv_id = source.paper_id
        WHEN MATCHED THEN UPDATE SET target.processed = source.processed,
                                     target.volume_path = source.volume_path
    """)

# COMMAND ----------

spark.sql(f"""CREATE TABLE IF NOT EXISTS {catalog}.{schema}.ai_parsed_docs (
    path STRING, parsed_content STRING, processed LONG)""")
if records:
    spark.sql(f"""
        INSERT INTO {catalog}.{schema}.ai_parsed_docs
        SELECT path, ai_parse_document(content) AS parsed_content, {end} AS processed
        FROM READ_FILES("{pdf_dir}", format => 'binaryFile')
    """)
spark.table(f"{catalog}.{schema}.ai_parsed_docs").select(
    "path", F.length("parsed_content").alias("len"), "processed"
).show(truncate=80)

# COMMAND ----------


def extract_chunks(parsed_content_json: str) -> list:
    parsed = json.loads(parsed_content_json)
    return [
        (e.get("id", ""), e.get("content", ""))
        for e in parsed.get("document", {}).get("elements", [])
        if e.get("type") == "text"
    ]


chunk_schema = T.ArrayType(
    T.StructType(
        [
            T.StructField("chunk_id", T.StringType(), True),
            T.StructField("content", T.StringType(), True),
        ]
    )
)
extract_chunks_udf = udf(extract_chunks, chunk_schema)
extract_paper_id_udf = udf(lambda p: p.replace(".pdf", "").split("/")[-1], T.StringType())


def clean_chunk(text: str) -> str:
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    t = re.sub(r"\s*\n\s*", " ", t)
    return re.sub(r"\s+", " ", t).strip()


clean_chunk_udf = udf(clean_chunk, T.StringType())

# COMMAND ----------

df = spark.table(f"{catalog}.{schema}.ai_parsed_docs").where(f"processed = {end}")
metadata_df = spark.table(metadata_table).select(
    col("arxiv_id").alias("paper_id"),
    "title",
    "summary",
    concat_ws(", ", col("authors")).alias("authors"),
    (col("published") / 100000000).cast("int").alias("year"),
    ((col("published") % 100000000) / 1000000).cast("int").alias("month"),
    ((col("published") % 1000000) / 10000).cast("int").alias("day"),
)
chunks_df = (
    df.withColumn("paper_id", extract_paper_id_udf(col("path")))
    .withColumn("chunks", extract_chunks_udf(col("parsed_content")))
    .withColumn("chunk", explode(col("chunks")))
    .select(
        "paper_id",
        col("chunk.chunk_id").alias("chunk_id"),
        clean_chunk_udf(col("chunk.content")).alias("text"),
        concat_ws("_", col("paper_id"), col("chunk.chunk_id")).alias("id"),
    )
    .join(metadata_df, "paper_id", "left")
)
chunks_table = f"{catalog}.{schema}.arxiv_chunks"
chunks_df.write.mode("append").saveAsTable(chunks_table)
spark.sql(f"ALTER TABLE {chunks_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(spark.table(chunks_table).count(), "chunks in", chunks_table)
