"""Keep generated execution artifacts inside each test's temporary directory."""

from pathlib import Path

import pytest

pytest_plugins = ["tests.postgres_backend_plugin"]


@pytest.fixture(autouse=True)
def isolate_pipeline_capsules(tmp_path, monkeypatch):
    # Portable execution capsules import the shared recording helpers by filename.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "skynet_app"))
    monkeypatch.setattr(
        "skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", tmp_path / "capsules"
    )


@pytest.fixture
def prepared_hand_store(tmp_path, monkeypatch):
    from skynet_app.hand_bundles import definition

    def build(robot):
        spec = definition(robot)
        manifest = dict(
            digest="b" * 64,
            robot=robot,
            source_revision=spec["revision"],
            name=spec["name"],
            action_dimension=56 if spec["side"] == "both" else 28,
            retargeting_scheme="dexpilot",
            hand_asset={"schema": "skynet.hand-bundle/v1", "digest": "a" * 64},
        )
        return tmp_path / robot, manifest

    def upload(path, work_root, *args):
        return str(work_root) + "/hands/" + str(path).split("/")[-1] + "/" + "b" * 64

    monkeypatch.setattr("skynet_app.live_xr.build_hand", build)
    monkeypatch.setattr("skynet_app.live_xr.upload_hand", upload)
    monkeypatch.setattr("skynet_app.capture_processing.service.build_hand", build)
    monkeypatch.setattr("skynet_app.capture_processing.service.upload_hand", upload)
    return build
