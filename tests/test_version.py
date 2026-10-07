"""The package version is declared once in pyproject and mirrored by the package and the API."""
import tomllib
from pathlib import Path

from skynet_app import __version__
from skynet_app.tracking import USER_AGENT


def test_package_api_and_user_agent_share_the_project_version():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    assert __version__ == project["version"]
    assert USER_AGENT.endswith("/" + __version__)
    assert "0.2.0" not in Path("skynet_app/main.py").read_text()
