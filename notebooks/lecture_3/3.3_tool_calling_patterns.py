# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.3: Tool Calling Patterns and Agent Implementation
# MAGIC Adapted: the model is the course gateway chat service through
# MAGIC `DatabricksOpenAI(use_ai_gateway=True)`; profile-based clients; the agent also records the
# MAGIC assistant `function_call` item before the tool output so the next LLM call is a valid
# MAGIC chat history; streamed text is gathered into one output item per LLM call.

# COMMAND ----------

import asyncio
import json
import warnings
from collections.abc import Generator
from typing import Any
from uuid import uuid4

import backoff
import mlflow
import nest_asyncio
import openai
from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)

from arxiv_curator.config import load_config
from arxiv_curator.mcp import ToolInfo, create_mcp_tools

nest_asyncio.apply()

cfg = load_config("project_config.yml", "dev")
w = WorkspaceClient(profile="student-bot-1")

# COMMAND ----------


class SimpleAgent(ResponsesAgent):
    """A simple agent that can call tools."""

    def __init__(self, llm_endpoint: str, system_prompt: str, tools: list | None = None):
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self.model_serving_client = DatabricksOpenAI(use_ai_gateway=True)
        self._tools_dict = {tool.name: tool for tool in (tools or [])}

    def get_tool_specs(self) -> list[dict]:
        return [tool.spec for tool in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        if tool_name not in self._tools_dict:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self._tools_dict[tool_name].exec_fn(**args)

    @mlflow.trace(span_type=SpanType.LLM)
    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=5)
    def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
            kwargs: dict[str, Any] = {}
            if self._tools_dict:  # the gateway rejects tools=None
                kwargs["tools"] = self.get_tool_specs()
            for chunk in self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=to_chat_completions_input(messages),
                stream=True,
                **kwargs,
            ):
                yield chunk.to_dict()

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        messages = [{"role": "system", "content": self.system_prompt}] + [
            i.model_dump() for i in request.input
        ]
        yield from self.call_and_run_tools(messages)

    @mlflow.trace(span_type=SpanType.CHAIN)
    def call_and_run_tools(
        self, messages: list[dict[str, Any]], max_iter: int = 10
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        for _ in range(max_iter):
            tool_calls: list[dict[str, Any]] = []
            text = ""
            for chunk in self.call_llm(messages):
                choices = chunk.get("choices") or [{}]
                delta = choices[0].get("delta", {})
                for tc_delta in delta.get("tool_calls") or []:
                    idx = tc_delta.get("index", 0)
                    while len(tool_calls) <= idx:
                        tool_calls.append({"call_id": None, "name": None, "arguments": ""})
                    current = tool_calls[idx]
                    if tc_delta.get("id"):
                        current["call_id"] = tc_delta["id"]
                    if function := tc_delta.get("function"):
                        if function.get("name"):
                            current["name"] = function["name"]
                        if function.get("arguments"):
                            current["arguments"] += function["arguments"]
                if content := delta.get("content"):
                    text += content

            if text:
                item = self.create_text_output_item(text, id=str(uuid4()))
                messages.append({"role": "assistant", "content": text})
                yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)

            calls = [tc for tc in tool_calls if tc.get("name")]
            if not calls:
                return
            for tool_call in calls:
                yield from self.handle_tool_call(tool_call, messages)

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(
                "Maximum iterations reached. Please try again.", id=str(uuid4())
            ),
        )

    def handle_tool_call(
        self, tool_call: dict[str, Any], messages: list[dict[str, Any]]
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        call_item = self.create_function_call_item(
            id=str(uuid4()),
            call_id=tool_call["call_id"],
            name=tool_call["name"],
            arguments=tool_call["arguments"],
        )
        messages.append(call_item)
        yield ResponsesAgentStreamEvent(type="response.output_item.done", item=call_item)
        try:
            args = json.loads(tool_call["arguments"] or "{}")
            result = str(self.execute_tool(tool_name=tool_call["name"], args=args))
        except Exception as e:  # noqa: BLE001  # the LLM sees the error and can try another approach
            result = f"Error executing tool {tool_call['name']}: {e}"
        tool_output = self.create_function_call_output_item(tool_call["call_id"], result)
        messages.append(tool_output)
        yield ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_output)


# COMMAND ----------


def calculator(operation: str, a: float, b: float) -> float:
    """Perform arithmetic operations."""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float("inf"),
    }
    return operations[operation](a, b)


calculator_tool = ToolInfo(
    name="calculator",
    spec={
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform arithmetic operations",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                    },
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
            },
        },
    },
    exec_fn=calculator,
)

# Vector search over arxiv_index through the managed MCP server
vs_url = f"{w.config.host}/api/2.0/mcp/vector-search/{cfg.catalog}/{cfg.schema}"
mcp_tools = asyncio.run(create_mcp_tools(w, [vs_url]))
print("MCP tools:", [t.name for t in mcp_tools])

# COMMAND ----------

# Without tools
agent = SimpleAgent(cfg.llm_endpoint, "You are a helpful AI assistant.")
resp = agent.predict(
    ResponsesAgentRequest(input=[{"role": "user", "content": "What is machine learning?"}])
)
print(resp.output[-1].content[0]["text"][:500])

# COMMAND ----------

agent_with_tools = SimpleAgent(
    cfg.llm_endpoint,
    "You are a helpful assistant for research papers. Use the calculator for arithmetic and "
    "the search tool to find papers.",
    tools=[calculator_tool, *mcp_tools],
)

# A conversation of a few turns: each turn's output is fed back as history
history: list[dict] = []
for question in [
    "What is 234 multiplied by 567?",
    "Now divide that by 3.",
    "Find a paper about attention mechanisms in transformers and name it.",
]:
    history.append({"role": "user", "content": question})
    resp = agent_with_tools.predict(ResponsesAgentRequest(input=history))
    print("USER:", question)
    for item in resp.output:
        d = item.model_dump()
        if d["type"] == "function_call":
            print("  TOOL CALL:", d["name"], d["arguments"][:150])
        elif d["type"] == "function_call_output":
            print("  TOOL OUTPUT:", d["output"][:150])
        history.append(d)
    print("ASSISTANT:", resp.output[-1].content[0]["text"][:500])

# COMMAND ----------

# Error handling: an unknown tool name and bad arguments come back to the LLM as text
bad = agent_with_tools.execute_tool
try:
    bad("no_such_tool", {})
except ValueError as e:
    print("unknown tool ->", e)

events = list(
    agent_with_tools.handle_tool_call(
        {"call_id": "c1", "name": "calculator", "arguments": '{"operation": "pow"}'}, []
    )
)
print("bad args ->", events[-1].item["output"])
