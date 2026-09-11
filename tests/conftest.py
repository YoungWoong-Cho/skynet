"""Keep generated execution artifacts inside each test's temporary directory."""

import pytest


@pytest.fixture(autouse=True)
def isolate_pipeline_capsules(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", tmp_path / "capsules"
    )
