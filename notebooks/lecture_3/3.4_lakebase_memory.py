# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.4: Lakebase Memory for Agents
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What is agent memory?
# MAGIC - Types of memory (short-term vs long-term)
# MAGIC - Implementing memory with Delta tables
# MAGIC - Session management
# MAGIC - Memory retrieval strategies
# MAGIC - Best practices for agent memory

# Adapted: Databricks Connect session with our profile; config from project_config.yml.

# COMMAND ----------

import json
from datetime import datetime
from uuid import uuid4

from databricks.connect import DatabricksSession
from pyspark.sql import SparkSession
from pyspark.sql import types as T

from arxiv_curator.config import load_config

# COMMAND ----------

spark = DatabricksSession.builder.profile("student-bot-1").serverless(True).getOrCreate()

cfg = load_config("project_config.yml", "dev")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understanding Agent Memory
# MAGIC
# MAGIC **Memory** allows agents to remember past interactions and context.
# MAGIC
# MAGIC ### Why Memory Matters:
# MAGIC
# MAGIC - **Personalization**: Remember user preferences
# MAGIC - **Context**: Understand conversation history
# MAGIC - **Efficiency**: Avoid repeating questions
# MAGIC - **Learning**: Improve over time
# MAGIC
# MAGIC ### Types of Memory:
# MAGIC
# MAGIC 1. **Short-term Memory** (Session Memory)
# MAGIC    - Current conversation
# MAGIC    - Stored in-memory or cache
# MAGIC    - Cleared after session ends
# MAGIC
# MAGIC 2. **Long-term Memory** (Persistent Memory)
# MAGIC    - Historical interactions
# MAGIC    - User preferences
# MAGIC    - Learned patterns
# MAGIC    - Stored in database (Delta Lake)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Memory Architecture with Delta Lake
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────┐
# MAGIC │     User Interaction                     │
# MAGIC └──────────────┬──────────────────────────┘
# MAGIC                │
# MAGIC                ↓
# MAGIC ┌─────────────────────────────────────────┐
# MAGIC │     Agent                                │
# MAGIC └──────────────┬──────────────────────────┘
# MAGIC                │
# MAGIC        ┌───────┴───────┐
# MAGIC        │               │
# MAGIC        ↓               ↓
# MAGIC ┌──────────────┐  ┌──────────────┐
# MAGIC │ Short-term   │  │ Long-term    │
# MAGIC │ Memory       │  │ Memory       │
# MAGIC │ (In-memory)  │  │ (Delta Lake) │
# MAGIC └──────────────┘  └──────────────┘
# MAGIC                        │
# MAGIC                        ↓
# MAGIC              ┌──────────────────────┐
# MAGIC              │ conversation_history │
# MAGIC              │ user_preferences     │
# MAGIC              │ session_metadata     │
# MAGIC              └──────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Creating Memory Tables

# COMMAND ----------

# Create conversation history table
# Note: Delta Lake doesn't support partitioning by expressions like DATE(timestamp)
# Instead, we add a date column for partitioning
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {cfg.catalog}.{cfg.schema}.conversation_history (
    session_id STRING,
    request_id STRING,
    timestamp TIMESTAMP,
    date DATE,
    role STRING,
    content STRING,
    tool_calls STRING,
    metadata STRING
)
USING DELTA
PARTITIONED BY (date)
""")

print(f"✓ Created conversation_history table")

# COMMAND ----------

# Create user preferences table
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {cfg.catalog}.{cfg.schema}.user_preferences (
    user_id STRING,
    preference_key STRING,
    preference_value STRING,
    updated_at TIMESTAMP,
    PRIMARY KEY (user_id, preference_key)
)
USING DELTA
""")

print(f"✓ Created user_preferences table")

# COMMAND ----------

# Create session metadata table
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {cfg.catalog}.{cfg.schema}.session_metadata (
    session_id STRING PRIMARY KEY,
    user_id STRING,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    total_messages INT,
    total_tool_calls INT,
    metadata STRING
)
USING DELTA
""")

print(f"✓ Created session_metadata table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Memory Manager Class

# COMMAND ----------

class MemoryManager:
    """Manage agent memory using Delta Lake."""
    
    def __init__(self, catalog: str, schema: str):
        """Initialize memory manager.
        
        Args:
            catalog: Unity Catalog name
            schema: Schema name
        """
        self.catalog = catalog
        self.schema = schema
        self.spark = spark
        
        self.conversation_table = f"{catalog}.{schema}.conversation_history"
        self.preferences_table = f"{catalog}.{schema}.user_preferences"
        self.session_table = f"{catalog}.{schema}.session_metadata"
    
    def save_message(
        self,
        session_id: str,
        request_id: str,
        role: str,
        content: str,
        tool_calls: list = None,
        metadata: dict = None
    ):
        """Save a message to conversation history.
        
        Args:
            session_id: Session identifier
            request_id: Request identifier
            role: Message role (user, assistant, tool)
            content: Message content
            tool_calls: Optional tool calls
            metadata: Optional metadata
        """
        now = datetime.now()
        message_data = [{
            "session_id": session_id,
            "request_id": request_id,
            "timestamp": now,
            "date": now.date(),  # Add date column for partitioning
            "role": role,
            "content": content,
            "tool_calls": json.dumps(tool_calls) if tool_calls else None,
            "metadata": json.dumps(metadata) if metadata else None
        }]
        
        # Define schema explicitly to handle None values
        schema = T.StructType([
            T.StructField("session_id", T.StringType(), False),
            T.StructField("request_id", T.StringType(), False),
            T.StructField("timestamp", T.TimestampType(), False),
            T.StructField("date", T.DateType(), False),
            T.StructField("role", T.StringType(), False),
            T.StructField("content", T.StringType(), False),
            T.StructField("tool_calls", T.StringType(), True),
            T.StructField("metadata", T.StringType(), True)
        ])
        
        df = self.spark.createDataFrame(message_data, schema=schema)
        df.write.format("delta").mode("append").saveAsTable(
            self.conversation_table
        )
    
    def get_conversation_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> list[dict]:
        """Retrieve conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries
        """
        df = self.spark.sql(f"""
            SELECT role, content, timestamp
            FROM {self.conversation_table}
            WHERE session_id = '{session_id}'
            ORDER BY timestamp DESC
            LIMIT {limit}
        """)
        
        # Convert to list of dicts (reverse to get chronological order)
        messages = [row.asDict() for row in df.collect()]
        return list(reversed(messages))
    
    def save_preference(
        self,
        user_id: str,
        key: str,
        value: str
    ):
        """Save or update a user preference.
        
        Args:
            user_id: User identifier
            key: Preference key
            value: Preference value
        """
        pref_data = [{
            "user_id": user_id,
            "preference_key": key,
            "preference_value": value,
            "updated_at": datetime.now()
        }]
        
        df = self.spark.createDataFrame(pref_data)
        df.write.format("delta").mode("append").option(
            "mergeSchema", "true"
        ).saveAsTable(self.preferences_table)
    
    def get_preferences(self, user_id: str) -> dict:
        """Get all preferences for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of preferences
        """
        df = self.spark.sql(f"""
            SELECT preference_key, preference_value
            FROM {self.preferences_table}
            WHERE user_id = '{user_id}'
        """)
        
        return {
            row.preference_key: row.preference_value
            for row in df.collect()
        }
    
    def start_session(
        self,
        session_id: str,
        user_id: str = None,
        metadata: dict = None
    ):
        """Start a new session.
        
        Args:
            session_id: Session identifier
            user_id: Optional user identifier
            metadata: Optional session metadata
        """
        session_data = [{
            "session_id": session_id,
            "user_id": user_id,
            "started_at": datetime.now(),
            "ended_at": None,
            "total_messages": 0,
            "total_tool_calls": 0,
            "metadata": json.dumps(metadata) if metadata else None
        }]
        
        # Define schema explicitly to handle None values
        schema = T.StructType([
            T.StructField("session_id", T.StringType(), False),
            T.StructField("user_id", T.StringType(), True),
            T.StructField("started_at", T.TimestampType(), False),
            T.StructField("ended_at", T.TimestampType(), True),
            T.StructField("total_messages", T.IntegerType(), False),
            T.StructField("total_tool_calls", T.IntegerType(), False),
            T.StructField("metadata", T.StringType(), True)
        ])
        
        df = self.spark.createDataFrame(session_data, schema=schema)
        df.write.format("delta").mode("append").saveAsTable(
            self.session_table
        )
    
    def end_session(self, session_id: str):
        """End a session and update statistics.
        
        Args:
            session_id: Session identifier
        """
        # Count messages and tool calls
        stats = self.spark.sql(f"""
            SELECT
                COUNT(*) as total_messages,
                SUM(CASE WHEN tool_calls IS NOT NULL THEN 1 ELSE 0 END) as total_tool_calls
            FROM {self.conversation_table}
            WHERE session_id = '{session_id}'
        """).collect()[0]
        
        # Update session
        self.spark.sql(f"""
            UPDATE {self.session_table}
            SET
                ended_at = current_timestamp(),
                total_messages = {stats.total_messages},
                total_tool_calls = {stats.total_tool_calls}
            WHERE session_id = '{session_id}'
        """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Using Memory Manager

# COMMAND ----------

# Initialize memory manager
memory = MemoryManager(catalog=cfg.catalog, schema=cfg.schema)

# Create a test session
session_id = f"session_{uuid4().hex[:8]}"
user_id = "test_user_123"

print(f"Session ID: {session_id}")
print(f"User ID: {user_id}")

# COMMAND ----------

# Start session
memory.start_session(
    session_id=session_id,
    user_id=user_id,
    metadata={"source": "notebook_test", "environment": "dev"}
)

print(f"✓ Started session: {session_id}")

# COMMAND ----------

# Save some messages
memory.save_message(
    session_id=session_id,
    request_id=f"req_{uuid4().hex[:8]}",
    role="user",
    content="What papers discuss transformers?"
)

memory.save_message(
    session_id=session_id,
    request_id=f"req_{uuid4().hex[:8]}",
    role="assistant",
    content="Here are papers about transformers...",
    tool_calls=[{"name": "search_papers", "args": {"query": "transformers"}}]
)

memory.save_message(
    session_id=session_id,
    request_id=f"req_{uuid4().hex[:8]}",
    role="user",
    content="Tell me more about the first one"
)

print("✓ Saved conversation messages")

# COMMAND ----------

# Retrieve conversation history
history = memory.get_conversation_history(session_id=session_id)

print("Conversation History:")
print("=" * 80)
for msg in history:
    print(f"[{msg['timestamp']}] {msg['role']}: {msg['content'][:60]}...")

# COMMAND ----------

# Save user preferences
memory.save_preference(user_id=user_id, key="language", value="python")
memory.save_preference(user_id=user_id, key="expertise_level", value="advanced")
memory.save_preference(user_id=user_id, key="preferred_topics", value="machine learning,NLP")

print("✓ Saved user preferences")

# COMMAND ----------

# Retrieve preferences
prefs = memory.get_preferences(user_id=user_id)

print("User Preferences:")
print("=" * 80)
for key, value in prefs.items():
    print(f"{key}: {value}")

# COMMAND ----------

# End session
memory.end_session(session_id=session_id)

print(f"✓ Ended session: {session_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Memory-Aware Agent

# COMMAND ----------

class MemoryAwareAgent:
    """Agent with memory capabilities."""
    
    def __init__(
        self,
        agent,  # Base agent
        memory_manager: MemoryManager,
        user_id: str = None
    ):
        """Initialize memory-aware agent.
        
        Args:
            agent: Base agent instance
            memory_manager: Memory manager instance
            user_id: Optional user identifier
        """
        self.agent = agent
        self.memory = memory_manager
        self.user_id = user_id
        self.current_session_id = None
    
    def start_conversation(self, session_id: str = None):
        """Start a new conversation session."""
        self.current_session_id = session_id or f"session_{uuid4().hex[:8]}"
        self.memory.start_session(
            session_id=self.current_session_id,
            user_id=self.user_id
        )
        return self.current_session_id
    
    def chat(self, message: str) -> str:
        """Send a message and get response.
        
        Args:
            message: User message
            
        Returns:
            Agent response
        """
        if not self.current_session_id:
            self.start_conversation()
        
        # Save user message
        request_id = f"req_{uuid4().hex[:8]}"
        self.memory.save_message(
            session_id=self.current_session_id,
            request_id=request_id,
            role="user",
            content=message
        )
        
        # Get conversation history
        history = self.memory.get_conversation_history(
            session_id=self.current_session_id,
            limit=10
        )
        
        # Build messages with history
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in history
        ]
        
        # Call agent
        from mlflow.types.responses import ResponsesAgentRequest
        request = ResponsesAgentRequest(input=messages)
        response = self.agent.predict(request)
        
        # Save assistant response
        assistant_message = response.output[-1].content
        self.memory.save_message(
            session_id=self.current_session_id,
            request_id=request_id,
            role="assistant",
            content=assistant_message
        )
        
        return assistant_message
    
    def end_conversation(self):
        """End the current conversation."""
        if self.current_session_id:
            self.memory.end_session(self.current_session_id)
            self.current_session_id = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Memory Retrieval Strategies

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 1: Recent Messages
# MAGIC ```python
# MAGIC # Get last N messages
# MAGIC history = memory.get_conversation_history(session_id, limit=10)
# MAGIC ```
# MAGIC
# MAGIC ### Strategy 2: Semantic Search
# MAGIC ```python
# MAGIC # Search for relevant past conversations
# MAGIC relevant = vector_search(
# MAGIC     query=current_message,
# MAGIC     index="conversation_embeddings"
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ### Strategy 3: Time-based
# MAGIC ```python
# MAGIC # Get messages from specific time period
# MAGIC history = get_messages_since(session_id, hours=24)
# MAGIC ```
# MAGIC
# MAGIC ### Strategy 4: Topic-based
# MAGIC ```python
# MAGIC # Get messages about specific topics
# MAGIC history = get_messages_by_topic(session_id, topic="transformers")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Memory Analytics

# COMMAND ----------

# Analyze conversation patterns
conversation_stats = spark.sql(f"""
    SELECT
        DATE(timestamp) as date,
        COUNT(DISTINCT session_id) as total_sessions,
        COUNT(*) as total_messages,
        AVG(LENGTH(content)) as avg_message_length
    FROM {cfg.catalog}.{cfg.schema}.conversation_history
    GROUP BY DATE(timestamp)
    ORDER BY date DESC
    LIMIT 7
""")

print("Conversation Statistics (Last 7 Days):")
conversation_stats.show(truncate=False)

# COMMAND ----------

# Analyze tool usage
tool_usage = spark.sql(f"""
    SELECT
        session_id,
        COUNT(*) as tool_calls,
        COUNT(DISTINCT request_id) as unique_requests
    FROM {cfg.catalog}.{cfg.schema}.conversation_history
    WHERE tool_calls IS NOT NULL
    GROUP BY session_id
    ORDER BY tool_calls DESC
    LIMIT 10
""")

print("Top Sessions by Tool Usage:")
tool_usage.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Best Practices

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ Do:
# MAGIC 1. **Partition by date** for efficient queries
# MAGIC 2. **Set retention policies** to manage storage costs
# MAGIC 3. **Index frequently queried columns**
# MAGIC 4. **Implement privacy controls** (user data deletion)
# MAGIC 5. **Monitor memory usage** and query performance
# MAGIC 6. **Use session IDs** for conversation grouping
# MAGIC 7. **Store metadata** for analytics
# MAGIC 8. **Implement memory limits** (don't load entire history)
# MAGIC
# MAGIC ### ❌ Don't:
# MAGIC 1. Store sensitive information without encryption
# MAGIC 2. Load unlimited conversation history
# MAGIC 3. Forget to clean up old sessions
# MAGIC 4. Skip session management
# MAGIC 5. Ignore privacy regulations (GDPR, CCPA)
# MAGIC 6. Store PII without proper controls

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Data Retention and Privacy

# COMMAND ----------

# Set retention policy (example: 90 days)
spark.sql(f"""
    ALTER TABLE {cfg.catalog}.{cfg.schema}.conversation_history
    SET TBLPROPERTIES (
        'delta.deletedFileRetentionDuration' = 'interval 7 days',
        'delta.logRetentionDuration' = 'interval 30 days'
    )
""")

print("✓ Set retention policies")

# COMMAND ----------

# Delete old conversations (example: older than 90 days)
spark.sql(f"""
    DELETE FROM {cfg.catalog}.{cfg.schema}.conversation_history
    WHERE timestamp < current_timestamp() - INTERVAL 90 DAYS
""")

print("✓ Cleaned up old conversations")

# COMMAND ----------

# Delete user data (GDPR compliance example)
def delete_user_data(user_id: str):
    """Delete all data for a user (GDPR right to be forgotten).
    
    Args:
        user_id: User identifier
    """
    # Delete conversations
    spark.sql(f"""
        DELETE FROM {cfg.catalog}.{cfg.schema}.conversation_history
        WHERE session_id IN (
            SELECT session_id
            FROM {cfg.catalog}.{cfg.schema}.session_metadata
            WHERE user_id = '{user_id}'
        )
    """)
    
    # Delete preferences
    spark.sql(f"""
        DELETE FROM {cfg.catalog}.{cfg.schema}.user_preferences
        WHERE user_id = '{user_id}'
    """)
    
    # Delete sessions
    spark.sql(f"""
        DELETE FROM {cfg.catalog}.{cfg.schema}.session_metadata
        WHERE user_id = '{user_id}'
    """)
    
    print(f"✓ Deleted all data for user: {user_id}")

# Example (commented out for safety)
# delete_user_data("test_user_123")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook, we learned:
# MAGIC
# MAGIC 1. ✅ Types of agent memory (short-term vs long-term)
# MAGIC 2. ✅ Implementing memory with Delta Lake
# MAGIC 3. ✅ Creating memory tables for conversations and preferences
# MAGIC 4. ✅ Memory manager class for CRUD operations
# MAGIC 5. ✅ Building memory-aware agents
# MAGIC 6. ✅ Memory retrieval strategies
# MAGIC 7. ✅ Analytics on conversation data
# MAGIC 8. ✅ Best practices for memory management
# MAGIC 9. ✅ Data retention and privacy compliance
# MAGIC
# MAGIC **Next**: Lecture 4 - MLflow for GenAI
