# llmops-victor-bot-1

Your repository for the LLMOps course on Databricks, victor-bot-1@cauchy.io. Everything you need in
the course workspace was provisioned for you before you got this repo, and every id is already in
`project_config.yml`: you configure nothing.

## What was provisioned for you

| What | Value |
|---|---|
| Catalog | `victor_bot_1` |
| Course schema | `victor_bot_1.arxiv` |
| Volume | `victor_bot_1.arxiv.arxiv_files` |
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

## What is in the repo

- `src/arxiv_curator/`: the course package; `config.py` loads `project_config.yml`, and the lectures
  add the rest week by week.
- `databricks.yml` and `resources/`: the bundle. It builds the package as a wheel with `uv build` and
  deploys the `hello` job, one serverless notebook task that installs the wheel and writes one row to
  `victor_bot_1.arxiv.hello`.
- `pyproject.toml` and `version.txt`: the package and its version, bumped as the lectures ask.
- `tests/`: your tests; `test_config.py` checks that `project_config.yml` loads.
- `.github/workflows/ci.yml`: on every pull request and push to `main`, installs the package with
  uv and runs the pre-commit hooks and the tests.
- `.github/workflows/deploy.yml`: on every push to `main`, deploys the bundle as your own deploy
  principal `sp-deploy-victor-bot-1`, signed in by OpenID Connect: there is no
  secret, and the job runs as that principal with your usage policy.

## Getting started

```bash
uv sync --extra dev
uvx pre-commit install
databricks bundle validate -t dev
```
