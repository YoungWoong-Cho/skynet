"""Recording ownership plugged into the shared history deletion engine."""

import json
import re
from pathlib import Path
from uuid import UUID

from .cluster_config import CLUSTER
from .cluster_runtime import WORK_ROOT
from .data_resource_policy import resource_recording_ids
from .live_xr_archive import archive_descriptor, TERMINAL_STATES
from .maintenance import Maintenance, rows, in_ids


def mentions(value, identifier):
    if isinstance(value, dict):
        return any(mentions(v, identifier) for v in value.values())
    if isinstance(value, list):
        return any(mentions(v, identifier) for v in value)
    return isinstance(value, str) and (
        value == identifier or identifier in value.split("/")
    )


class RecordingMaintenance(Maintenance):
    def __init__(self, database, live, reviews, videos, previews, *, conversion_root=None):
        super().__init__(database, live.archive.cluster)
        self.live, self.reviews, self.videos = live, reviews, videos
        self.previews = previews
        # Historical receipts and files remain deletable without starting a worker.
        self.conversion_root = Path(conversion_root or live.root / "data/live-conversions")

    def remote(self, operation, root, items=None, protected=None, gateway="auto"):
        # These are shared collection archives, independent of the caller's
        # training base path and currently selected submission gateway.
        return super().remote(operation, root, items, protected, "sky2")

    def _target(self, c, kind, identifier):
        if kind != "recording":
            raise ValueError("Unknown recording deletion type")
        if str(UUID(identifier)) != identifier:
            raise ValueError("Invalid recording ID")
        record = super()._target(c, kind, identifier)
        job = json.loads(record["payload_json"])
        if job.get("id") != identifier:
            raise ValueError("Recording identity does not match its stored session")
        return dict(
            record,
            name=(job.get("profile") or {}).get("task_name") or identifier,
            owner_id=self.db.workspace_id or "legacy",
        )

    def _graph(self, c, kind, identifier, *, pending_target=None):
        target = self._target(c, kind, identifier)
        job = json.loads(target["payload_json"])
        graph = {"live_xr_sessions": [target], "live_conversions": []}
        blockers = []

        def block(kind, key, label, reason):
            entry = dict(kind=kind, id=key, label=label, reason=reason)
            if entry not in blockers:
                blockers.append(entry)

        for pending in rows(c, "maintenance_operations"):
            previous = json.loads(pending["plan_json"])
            if identifier in previous["records"].get("live_xr_sessions", []) and (pending["target_kind"], pending["target_id"]) != (pending_target or (kind, identifier)):
                block(pending["target_kind"], pending["target_id"], previous["label"], "Finish this pending deletion first")

        lock = self.live.recording_lock(identifier)
        if lock.acquire(blocking=False):
            lock.release()
        else:
            block(
                "recording",
                identifier,
                target["name"],
                "Wait for recording review, transfer or conversion work to finish",
            )
        if job.get("state") not in TERMINAL_STATES or (
            job.get("job_id") and not job.get("scheduler_final")
        ):
            block(
                "recording",
                identifier,
                target["name"],
                "Stop the collection session and wait for image processing to finish",
            )
        archive = job.get("archive") or {}
        if archive.get("state") != "READY" or not archive.get("source_removed"):
            block(
                "recording",
                identifier,
                target["name"],
                "Wait for transfer to sky2 and collection workstation cleanup to finish",
            )
        if (
            identifier in self.live.archive.active
            or any(k[0] == identifier for k in self.reviews.active | self.videos.active)
            or any(
                k[0] == identifier and v.get("state") == "PREPARING"
                for k, v in self.previews.states.items()
            )
        ):
            block(
                "recording",
                identifier,
                target["name"],
                "Wait for active recording processing to finish",
            )
        for resource in rows(c, "data_resources"):
            versions = rows(
                c, "data_resource_versions", "resource_id=?", (resource["id"],)
            )
            resource["metadata"] = json.loads(resource["metadata_json"])
            for version in versions:
                version["metadata"] = json.loads(version["metadata_json"])
            if identifier in resource_recording_ids(resource, versions):
                block(
                    "dataset",
                    resource["id"],
                    resource["display_name"],
                    "Delete this dataset first",
                )
        for record in rows(c, "live_conversions"):
            conversion = json.loads(record["payload_json"])
            if conversion.get("session_id") != identifier:
                continue
            graph["live_conversions"].append(record)
            if conversion.get("state") not in {"READY", "FAILED", "CANCELLED"}:
                block(
                    "recording",
                    identifier,
                    conversion.get("name") or "Dataset conversion",
                    "Wait for this dataset conversion to finish",
                )
            resource_id = conversion.get("resource_id")
            if (
                resource_id
                and c.execute(
                    "SELECT 1 FROM data_resources WHERE id=?", (resource_id,)
                ).fetchone()
            ):
                block(
                    "dataset",
                    resource_id,
                    conversion.get("name") or resource_id,
                    "Delete the converted dataset first",
                )
        for record in rows(c, "live_xr_sessions", "id<>?", (identifier,)):
            if mentions(json.loads(record["payload_json"]), identifier):
                child = json.loads(record["payload_json"])
                block(
                    "recording",
                    record["id"],
                    child.get("profile", {}).get("task_name") or record["id"],
                    "Delete this dependent recording first",
                )
        # Legacy/manual configuration snapshots may reference a recording directly.
        for table, kind, parent in (
            ("experiment_revisions", "experiment", "experiment_id"),
            ("variants", "experiment", "experiment_id"),
            ("evaluations", "evaluation", "id"),
        ):
            for record in rows(c, table):
                if any(
                    mentions(json.loads(v), identifier)
                    for k, v in record.items()
                    if k.endswith("_json") and v
                ):
                    if table == "variants":
                        record[parent] = c.execute(
                            "SELECT experiment_id FROM experiment_revisions WHERE id=?",
                            (record["experiment_revision_id"],),
                        ).fetchone()[0]
                    visible = self.db.workspace_id is None or record.get(
                        "owner_id"
                    ) in (None, self.db.workspace_id)
                    label = record.get("suite_name") or record[parent]
                    if kind == "experiment" and visible:
                        experiment = c.execute(
                            "SELECT name FROM experiments WHERE id=?", (record[parent],)
                        ).fetchone()
                        if experiment:
                            label = experiment[0]
                    block(
                        kind,
                        record[parent] if visible else None,
                        label if visible else "Another workspace's item",
                        "Delete this dependent item first",
                    )
        return target, graph, blockers

    def _references(self, c, graph=None):
        result = super()._references(c, graph)
        excluded = {r["id"] for r in (graph or {}).get("live_conversions", [])}
        from .maintenance import paths_in

        for record in rows(c, "live_conversions"):
            if record["id"] not in excluded:
                result.extend(
                    (p, "live_conversions", record["id"])
                    for p in paths_in(json.loads(record["payload_json"]))
                )
        return result

    def _file_groups(self, c, kind, target, graph, blockers):
        job = json.loads(target["payload_json"])
        archive_descriptor(
            job, self.cluster
        )  # Validate canonical identity, never trust a saved arbitrary path.
        root = CLUSTER.paths.datasets
        groups = {
            root: [f"{root}/raw/dexverse-live/{job['id']}"],
            "local:" + str(self.reviews.root): [str(self.reviews.root / job["id"])],
        }
        for record in graph["live_conversions"]:
            item = json.loads(record["payload_json"])
            key = str(UUID(item["id"]))
            paths = [
                f"{WORK_ROOT}/jobs/runs/{key}",
                f"{WORK_ROOT}/datasets/derivatives/dexverse-live/{job['id']}/{key}",
            ]
            if (
                item.get("root") != paths[0]
                or item.get("dataset_root", paths[1]) != paths[1]
            ):
                raise ValueError("Conversion storage does not match its immutable ID")
            groups.setdefault(WORK_ROOT, []).extend(paths)
            groups.setdefault("local:" + str(self.conversion_root), []).append(
                str(self.conversion_root / key)
            )
        # Rendered video attempts have immutable generation capsules outside the
        # archive. Their receipts stay in the local review status directory.
        local = self.reviews.root / job["id"]
        if local.is_symlink():
            raise ValueError("Recording review directory must not be a symbolic link")
        for status in sorted(local.glob("*/video-*/status.json")):
            if (
                any(
                    p.is_symlink()
                    for p in (status, status.parent, status.parent.parent)
                )
                or status.stat().st_size > 1_000_000
            ):
                raise ValueError("Invalid video preparation receipt")
            value = json.loads(status.read_text())
            remote = value.get("remote_root")
            if not remote:
                continue
            if remote.startswith(job["root"] + "/output/"):
                continue  # Verified archive cleanup already removed workstation attempts.
            generation = value.get("generation", "")
            if (
                not re.fullmatch(r"[a-f0-9]{32}", generation)
                or remote != f"{WORK_ROOT}/jobs/runs/{generation}"
            ):
                raise ValueError(
                    "Video storage does not match its immutable generation"
                )
            if value.get("state") not in {"READY", "CANCELLED", "FAILED"}:
                blockers.append(
                    dict(
                        kind="recording",
                        id=job["id"],
                        label="Recording video",
                        reason="Finish or cancel video preparation before deleting this recording",
                    )
                )
            groups.setdefault(WORK_ROOT, []).append(remote)
        retained = self._references(c, graph)
        for paths in groups.values():
            for path in paths:
                if self._referenced(path, retained, graph):
                    blockers.append(
                        dict(
                            kind="storage",
                            id=None,
                            label=path,
                            reason="Another retained item references these files; delete that dependency first",
                        )
                    )
        return groups

    def delete(self, kind, identifier, token, gateway="auto"):
        lock = self.live.recording_lock(identifier)
        if not lock.acquire(blocking=False):
            raise ValueError(
                "Recording processing is active. Review deletion again after it finishes."
            )
        try:
            result = super().delete(kind, identifier, token, gateway)
            with self.previews.lock:
                for key in list(self.previews.states):
                    if key[0] == identifier:
                        self.previews.states.pop(key)
            return result
        finally:
            lock.release()

    def _delete_rows(self, c, kind, identifier, graph):
        # The parent engine removes metadata bodies, events and the durable intent.
        query, params = in_ids("id", [r["id"] for r in graph["live_conversions"]])
        c.execute("DELETE FROM live_conversions WHERE " + query, params)
        super()._delete_rows(c, kind, identifier, graph)
