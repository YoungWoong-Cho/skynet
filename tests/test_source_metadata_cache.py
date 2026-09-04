from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from skynet_app.database import Database
from skynet_app.source_metadata_cache import SourceMetadataStore


class SourceMetadataStoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.temporary_directory.name) / "skynet.db")
        self.store = SourceMetadataStore(self.database)
        self.repository = "https://github.com/example/project"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_metadata_survives_store_recreation_and_is_replaced_explicitly(self) -> None:
        parameters = {"branch": "main", "limit": 50}
        first = self.store.put(
            "commits", self.repository, parameters, {"commits": [{"sha": "a" * 40}]}
        )

        reopened = SourceMetadataStore(self.database)
        cached = reopened.get("commits", self.repository, {"limit": 50, "branch": "main"})

        self.assertIsNotNone(cached)
        self.assertEqual(cached["cache_key"], first["cache_key"])
        self.assertEqual(cached["payload"]["commits"][0]["sha"], "a" * 40)

        reopened.put(
            "commits", self.repository, parameters, {"commits": [{"sha": "b" * 40}]}
        )
        replaced = self.store.get("commits", self.repository, parameters)
        self.assertEqual(replaced["payload"]["commits"][0]["sha"], "b" * 40)

    def test_repository_selection_persists_commits_per_branch(self) -> None:
        self.store.save_selection(self.repository, "main", "a" * 40)
        self.store.save_selection(self.repository, "release", "b" * 40)

        selection = SourceMetadataStore(self.database).get_selection(self.repository)

        self.assertEqual(selection["branch"], "release")
        self.assertEqual(selection["commits"], {"main": "a" * 40, "release": "b" * 40})

    def test_invalid_selected_commit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "full 40-character SHA"):
            self.store.save_selection(self.repository, "main", "not-a-commit")


if __name__ == "__main__":
    unittest.main()
