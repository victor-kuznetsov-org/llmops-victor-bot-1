"""Your first test: project_config.yml loads with the course's ProjectConfig for dev.

`ci.yml` runs every test under tests/ on each pull request and each push to main; add yours here.
"""

from pathlib import Path

from arxiv_curator.config import ProjectConfig

PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "project_config.yml"


def test_project_config_loads_for_dev() -> None:
    cfg = ProjectConfig.from_yaml(str(PROJECT_CONFIG), "dev")

    assert cfg.full_volume_path == "victor_bot_1.arxiv.arxiv_files"
    assert cfg.warehouse_id == "b6a0f7db8a1da913"
