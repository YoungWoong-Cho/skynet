from __future__ import annotations

from skynet_app.db_backend import INTEGRITY_ERRORS
import tempfile
import unittest
from pathlib import Path

from skynet_app.database import Database, content_sha256
from skynet_app.adapters import builtin_adapter_manifests
from skynet_app.pipeline_api import PipelineService


COMMIT = "e17cf98fe4bc234c564b37abc9e155f25e76d566"


class _Cluster:
    hosts = ("sky1", "sky2")


class DataRegistryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.temporary_directory.name) / "skynet.db")
        self.assets = self.database.create_data_resource(
            category="file",
            provider="huggingface",
            namespace="dexverse",
            source_key="DexVerse_release",
            kind="simulation_assets",
            metadata={"license": "upstream"},
        )
        self.assets_version = self.database.create_data_resource_version(
            self.assets["id"],
            revision="a" * 40,
            format="isaac-usd",
            path="/coc/flash7/ycho420/datasets/resources/huggingface/dexverse/DexVerse_release/revisions/aaaaaaaa",
            source_uri="hf://datasets/dexverse/DexVerse_release",
            manifest_sha256="1" * 64,
            size_bytes=5843000000,
            metadata={"file_count": 1969},
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _version(self, name: str, revision: str, digest: str) -> dict[str, object]:
        resource = self.database.create_data_resource(
            category="dataset",
            provider="derived",
            namespace="dexverse",
            source_key=name,
            kind="demonstrations",
        )
        return self.database.create_data_resource_version(
            resource["id"],
            revision=revision,
            format="lerobot-v3",
            path=f"/coc/flash7/ycho420/datasets/derivatives/dexverse/{name}/{revision}",
            manifest_sha256=digest * 64,
        )

    def test_resource_versions_are_immutable_and_resources_are_archived(self) -> None:
        loaded = self.database.get_data_resource(self.assets["id"])

        self.assertEqual(loaded["version_count"], 1)
        self.assertEqual(loaded["latest_version"]["revision"], "a" * 40)
        self.assertEqual(loaded["metadata"]["license"], "upstream")
        with self.assertRaises(INTEGRITY_ERRORS):
            with self.database.transaction() as connection:
                connection.execute(
                    "UPDATE data_resource_versions SET status = 'BROKEN' WHERE id = ?",
                    (self.assets_version["id"],),
                )

        archived = self.database.update_data_resource(self.assets["id"], archived=True)

        self.assertIsNotNone(archived["archived_at"])
        self.assertEqual(self.database.list_data_resources(), [])
        self.assertEqual(len(self.database.list_data_resources(include_archived=True)), 1)

    def test_derivation_resolves_lineage_and_rejects_cycles(self) -> None:
        converted = self._version("demos-lerobot", "converter-v1", "2")
        derivation = self.database.create_data_derivation(
            output_version_id=converted["id"],
            inputs=[{"version_id": self.assets_version["id"], "role": "assets"}],
            converter_repository="https://github.com/example/dataset-converters",
            converter_commit="b" * 40,
            converter_config={"fps": 30},
            runtime_lock_sha256="3" * 64,
        )

        self.assertEqual(derivation["inputs"][0]["version"]["resource"]["source_key"], "DexVerse_release")
        self.assertEqual(derivation["output_version"]["format"], "lerobot-v3")
        self.assertEqual(derivation["converter_config"], {"fps": 30})
        with self.assertRaisesRegex(ValueError, "lineage cycle"):
            self.database.create_data_derivation(
                output_version_id=self.assets_version["id"],
                inputs=[{"version_id": converted["id"]}],
                converter_repository="https://github.com/example/dataset-converters",
                converter_commit="c" * 40,
            )

    def test_unsafe_bundle_mounts_are_rejected_by_storage(self) -> None:
        unsafe_paths = [
            "/absolute", "../escape", "data/../escape", "./data", "data/./file",
            "data//file", "data/", "C:/data", "data\\file", "data\nfile", "data\x00file",
        ]
        for mount_path in unsafe_paths:
            with self.subTest(mount_path=mount_path):
                assignment = {"role": "simulation_assets", "version_id": self.assets_version["id"], "mount_path": mount_path}
                with self.assertRaisesRegex(ValueError, "Mount path must be a relative directory"):
                    self.database.create_data_bundle(name="unsafe", version="v1", assignments=[assignment])
        self.assertEqual(self.database.list_data_bundles(), [])

    def test_relative_bundle_mounts_are_normalized_in_persisted_manifest(self) -> None:
        for index, (mount_path, expected) in enumerate([(None, None), ("", None), ("   ", None), (" data/nested-dir_v1 ", "data/nested-dir_v1")]):
            with self.subTest(mount_path=mount_path):
                assignment = {"role": "simulation_assets", "version_id": self.assets_version["id"], "mount_path": mount_path}
                bundle = self.database.create_data_bundle(name="relative", version=str(index), assignments=[assignment])
                self.assertEqual(bundle["assignments"][0]["mount_path"], expected)
                self.assertEqual(bundle["manifest"]["assignments"][0]["mount_path"], expected)

    def test_legacy_bundle_snapshot_cannot_bypass_mount_validation(self) -> None:
        manifest = next(item for item in builtin_adapter_manifests() if item.slug == "groot")
        document = {"data": {"bundle": {"assignments": [{"role": "training_data", "position": 0, "mount_path": "../../escape", "version": {"path": "/data/training", "format": "groot-lerobot-v2.0"}}]}}}
        with self.assertRaisesRegex(ValueError, "Mount path must be a relative directory"):
            PipelineService._apply_manifest_data_bindings(document, manifest)

    def test_bundle_manifest_is_content_addressed_and_snapshotted_into_spec(self) -> None:
        demonstrations = self._version("training-demos", "hf-revision-1", "4")
        bundle = self.database.create_data_bundle(
            name="dexverse-training",
            version="2026-09-01",
            description="Pinned simulation and training inputs",
            assignments=[
                {
                    "role": "simulation_assets",
                    "version_id": self.assets_version["id"],
                    "mount_path": "assets",
                },
                {
                    "role": "training_data",
                    "version_id": demonstrations["id"],
                    "mount_path": "demonstrations",
                },
            ],
            metadata={"framework": "independent"},
        )

        self.assertEqual(bundle["manifest_sha256"], content_sha256(bundle["manifest"]))
        self.assertEqual(bundle["assignment_count"], 2)
        self.assertNotIn("id", bundle["manifest"]["assignments"][0]["version"])
        service = PipelineService(self.database, _Cluster())
        data = service._normalize_data_input({
            "apiVersion": "skynet.rl2/v1",
            "identity": {"project": "tests", "experiment": "data-snapshot"},
            "source": {
                "repository": "https://github.com/GaTech-RL2/EgoVerse",
                "revision": COMMIT,
                "adapter": "egoverse",
            },
            "runtime": {"backend": "existing", "bootstrap_uv": False},
            "data": {
                "datasets": [
                    {"name": "legacy", "uri": "file:///legacy", "revision": "v1"}
                ]
            },
            "data_bundle_id": bundle["id"],
            "resources": {
                "queue_policy": "normal",
                "account": "rl2-lab",
                "partition": "rl2-lab",
                "gpu": {"mode": "explicit", "count": 1, "type": "a40"},
                "time_limit": "00:10:00",
            },
        })

        self.assertEqual(data["bundle"]["id"], bundle["id"])
        self.assertEqual(data["bundle"]["manifest_sha256"], bundle["manifest_sha256"])
        self.assertEqual(len(data["bundle"]["assignments"]), 2)
        self.assertEqual(data["datasets"][0]["name"], "legacy")

        self.database.archive_data_bundle(bundle["id"])
        with self.assertRaisesRegex(ValueError, "archived"):
            service.normalize_spec({
                "apiVersion": "skynet.rl2/v1",
                "identity": {"project": "tests", "experiment": "archived"},
                "source": {
                    "repository": "https://github.com/GaTech-RL2/EgoVerse",
                    "revision": COMMIT,
                    "adapter": "egoverse",
                },
                "runtime": {"backend": "existing", "bootstrap_uv": False},
                "data_bundle_id": bundle["id"],
            })


if __name__ == "__main__":
    unittest.main()
