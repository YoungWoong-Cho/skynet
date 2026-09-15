from __future__ import annotations

import json

from skynet_app.data_preview import build_data_bundle_preview


class NoRemoteCluster:
    def run_with_fallback(self, *args, **kwargs):
        raise AssertionError("provider-backed test must not read remote files")


def _bundle() -> dict:
    return {
        "id": "bundle-1",
        "name": "robot-training",
        "version": "v1",
        "manifest_sha256": "a" * 64,
        "assignments": [
            {
                "role": "training_data",
                "position": 0,
                "required": True,
                "version": {
                    "id": "version-1",
                    "revision": "b" * 40,
                    "format": "lerobot-v2.0",
                    "path": "/coc/flash7/ycho420/datasets/example",
                    "status": "READY",
                    "resource": {
                        "provider": "memory",
                        "namespace": "tests",
                        "source_key": "robot",
                        "display_name": "Robot recordings",
                        "kind": "demonstrations",
                    },
                },
            }
        ],
    }


def test_bundle_usage_does_not_claim_training_data_is_evaluation_data(monkeypatch):
    files = {
        "meta/info.json": json.dumps(
            {
                "codebase_version": "v2.0",
                "robot_type": "testbot",
                "total_episodes": 1,
                "total_frames": 3,
                "total_tasks": 1,
                "total_videos": 1,
                "chunks_size": 1000,
                "fps": 20,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [8]},
                    "action": {"dtype": "float32", "shape": [7]},
                    "observation.images.front": {"dtype": "video", "shape": [256, 256, 3]},
                },
            }
        ),
        "meta/tasks.jsonl": json.dumps({"task_index": 0, "task": "pick object"}),
        "meta/episodes.jsonl": json.dumps({"episode_index": 0, "tasks": ["pick object"], "length": 3}),
    }
    monkeypatch.setattr(
        "skynet_app.data_preview._load_metadata",
        lambda cluster, version, resource, gateway: (files, "test", []),
    )

    preview = build_data_bundle_preview(_bundle(), NoRemoteCluster())

    assert preview["schema_version"] == "skynet.data-bundle-preview/v1"
    assert preview["bundle"]["usage"]["training"]["declared"] is True
    assert preview["bundle"]["usage"]["evaluation"]["declared"] is False
    assert "no evaluation dataset" in preview["bundle"]["usage"]["relationship"]
    assignment = preview["assignments"][0]
    assert assignment["dataset"]["summary"]["total_episodes"] == 1
    assert assignment["dataset"]["features"][0]["name"] == "observation.state"
    assert assignment["sample_media"][0]["relative_path"].endswith("episode_000000.mp4")
    assert assignment["sample_media"][0]["mime_type"] == "video/mp4"
    assert assignment["sample_media"][0]["url"].startswith("/api/data/bundles/bundle-1/preview/media?")


def test_unknown_format_is_explicitly_unsupported():
    bundle = _bundle()
    bundle["manifest_sha256"] = "c" * 64
    bundle["assignments"][0]["version"]["format"] = "custom-binary-v9"

    preview = build_data_bundle_preview(bundle, NoRemoteCluster())

    inspection = preview["assignments"][0]["inspection"]
    assert inspection["status"] == "unsupported"
    assert "No static dataset preview adapter" in inspection["warnings"][0]


def test_cached_preview_uses_current_resource_display_name():
    bundle = _bundle()
    bundle["manifest_sha256"] = "d" * 64
    bundle["assignments"][0]["version"]["format"] = "custom-binary-v9"
    first = build_data_bundle_preview(bundle, NoRemoteCluster())
    bundle["assignments"][0]["version"]["resource"]["display_name"] = "Renamed recordings"
    renamed = build_data_bundle_preview(bundle, NoRemoteCluster())
    assert first["assignments"][0]["resource"]["display_name"] == "Robot recordings"
    assert renamed["assignments"][0]["resource"]["display_name"] == "Renamed recordings"
    assert renamed["assignments"][0]["resource"]["source_key"] == "robot"
    assert renamed["assignments"][0]["inspection"] == first["assignments"][0]["inspection"]
