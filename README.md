# llmops-victor-bot-1

Your repository for the LLMOps course on Databricks. Everything you need in
the course workspace was provisioned for you before you got this repo, and every id is already in
`project_config.yml`: you configure nothing.

## What was provisioned for you

| What | Value |
|---|---|
| Your schema in development, which you manage | `llmops_dev.victor_bot_1` |
| Your schema in acceptance, which you read | `llmops_acc.victor_bot_1` |
| Your schema in production, which you read | `llmops_prd.victor_bot_1` |
| Volume, in each of them | `arxiv_files`, such as `llmops_dev.victor_bot_1.arxiv_files` |
| SQL warehouse id | `b6a0f7db8a1da913` |
| Usage policy id | `2b379ae0-d42c-352c-923a-b63d6ba6bf3e` |
| MLflow experiment | `/Users/victor-bot-1@cauchy.io/llmops` |
| LLM endpoint | `course_ops.gateway.chat` |
| Embedding endpoint | `databricks-gte-large-en` |
| Vector Search endpoint | `vs-llmops` |

The LLM endpoint is a model service of the course's AI Gateway, with a monthly budget per student.
Call it through the gateway, not `/serving-endpoints`, with the `databricks-openai` package:

```python
from databricks_openai import DatabricksOpenAI

client = DatabricksOpenAI(use_ai_gateway=True)
client.chat.completions.create(model="course_ops.gateway.chat", messages=[...])
```

This repo does not install `databricks-openai` yet: it needs newer `mlflow` and `pydantic` than
the course pins in `pyproject.toml`.

## What is in the repo

- `src/arxiv_curator/`: the course package; `config.py` loads `project_config.yml`,
  `vector_search.py` builds the Vector Search index of lecture 2, and lecture 3's agent uses
  `mcp.py` (tools from Databricks' MCP servers), `memory.py` (session memory in Lakebase) and
  `agent.py` (the arXiv agent); the later lectures add the rest week by week.
- `lecture_1/`, `lecture_2/` and so on, when you add them: copy a lecture's folder from the course
  code as it is, to the root of this repo, where its notebooks find `project_config.yml` one level up.
  Commit it with the hooks installed (below): the first commit stops once, after the hooks have
  formatted the notebooks and fixed what ruff can, and goes through when you add their changes and
  commit again. In these folders only, five rules that the course's notebooks break and ruff cannot
  fix are off; `pyproject.toml` names them, and also tells ruff that `display` and `spark` exist
  without an import, as they do in a Databricks notebook, everywhere in the repo.
- `databricks.yml` and `resources/`: the bundle, with three targets, `dev` (your sandbox, in
  development mode, the default), `acc` and `prd`. It builds the package as a wheel with `uv build`
  and deploys the `hello` job, named after its target (`job-hello-victor-bot-1-dev`,
  `-acc`, `-prd`), one serverless notebook task that installs the wheel and writes one row to the
  `hello` table of your schema in the target's catalog, such as
  `llmops_dev.victor_bot_1.hello`.
- `pyproject.toml` and `version.txt`: the package and its version, bumped as the lectures ask.
- `tests/`: your tests; `test_config.py` checks that `project_config.yml` loads.
- `.github/workflows/ci.yml`: on every pull request and push to `main`, installs the package with
  uv and runs the pre-commit hooks and the tests.
- `.github/workflows/deploy.yml`: deploys the bundle, one job per target, each as that target's own
  deploy principal, signed in by OpenID Connect through the GitHub environment of the same name:
  there is no secret, and the job runs as that principal with your usage policy. A push to `main`
  deploys `acc`, then `prd` once you approve it; run the workflow by hand to deploy one target. A
  target whose principal does not exist yet is skipped, and the job "no principal yet" names it.
- `.github/CODEOWNERS`: who reviews a pull request to `main`, which needs one approving review.

## Getting started

```bash
uv sync --extra dev
uvx pre-commit install
databricks bundle validate -t dev
```
