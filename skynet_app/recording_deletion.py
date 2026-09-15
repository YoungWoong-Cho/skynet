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


def canonical_source_id(row):
    # Stable only inside deletion previews; the table itself uses a composite PK.
    from .database import canonical_json
    return canonical_json([row["artifact_key"], row["session_id"], row["recording_path"]])


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

    def _graph(self, c, kind, identifier, *, pending_target=None, selected_path=None):
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
                "Stop the collection session and wait for processing to finish",
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
        self._observation_graph(c, job, selected_path, graph, blockers)
        return target, graph, blockers

    def _observation_graph(self, c, job, selected_path, graph, blockers):
        """Plan observation ownership before any remote bytes can be removed."""
        sources = rows(c, "observation_sources")
        selected = [row for row in sources if row["session_id"] == job["id"]
                    and (selected_path is None or row["recording_path"] == selected_path)]
        if not selected:
            return
        affected = {row["artifact_key"] for row in selected}
        detached = {(row["artifact_key"], row["session_id"], row["recording_path"]) for row in selected}
        shared = {row["artifact_key"] for row in sources
                  if (row["artifact_key"], row["session_id"], row["recording_path"]) not in detached}
        candidates = affected - shared
        artifacts = {row["artifact_key"]: row for row in rows(c, "observation_artifacts")}
        inputs = rows(c, "observation_artifact_inputs")
        consumers = [("dataset version", row["version_id"], row["artifact_key"])
                     for row in rows(c, "observation_version_inputs")]
        consumers += [("dataset preparation", row["job_id"], row["artifact_key"])
                      for row in rows(c, "observation_job_inputs")]
        producers = {row["id"]: row for row in rows(c, "observation_producers")}
        def block(identifier, label, reason):
            entry = dict(kind="observation", id=identifier, label=label, reason=reason)
            if entry not in blockers:
                blockers.append(entry)
        for key in sorted(affected):
            producer = producers.get(artifacts[key].get("producer_id"))
            if producer and producer["state"] not in {"READY", "FAILED"}:
                block(producer["id"], "Observation preparation",
                      "Wait for the shared observation producer to finish before deleting its recording")
        for kind, identifier, key in consumers:
            if key in candidates:
                block(identifier, "Shared " + kind,
                      "Delete this observation consumer before deleting its only source recording")
        for edge in inputs:
            if edge["input_key"] in candidates and edge["artifact_key"] not in candidates:
                block(edge["artifact_key"], "Derived observations",
                      "A retained observation uses this recording; remove that dependency first")
        # Shared artifacts retain their bytes and other source aliases. Only the
        # selected source edges are detached after the recording is safely removed.
        graph["observation_sources"] = [dict(row, id=canonical_source_id(row))
                                         for row in sorted(selected, key=canonical_source_id)]
        if candidates:
            remaining, ordered = set(candidates), []
            while remaining:
                dependencies = {edge["input_key"] for edge in inputs if edge["artifact_key"] in remaining}
                leaves = sorted(remaining - dependencies)
                if not leaves:
                    raise ValueError("Observation dependency cycle prevents safe deletion")
                for key in leaves:
                    item = dict(artifacts[key], id=key)
                    item["inputs"] = sorted(edge["input_key"] for edge in inputs if edge["artifact_key"] == key)
                    ordered.append(item)
                remaining.difference_update(leaves)
            graph["observation_artifacts"] = ordered
            removable_producers = []
            for producer in producers.values():
                if producer["state"] not in {"READY", "FAILED"}:
                    continue
                owned = {key for key, artifact in artifacts.items() if artifact.get("producer_id") == producer["id"]}
                # Retried failures may no longer own the artifact FK; keep their
                # original request keys in the lifetime check too.
                payload = json.loads(producer["payload_json"])
                requested = {node["artifact_key"] for node in payload.get("request", {}).get("requests", [])}
                related = owned | (requested & artifacts.keys())
                if related & candidates and related <= candidates:
                    removable_producers.append(producer)
            if removable_producers:
                graph["observation_producers"] = sorted(removable_producers, key=lambda row: row["id"])

    @staticmethod
    def _observation_producer_path(producer):
        identifier = producer["id"]
        if str(UUID(identifier)) != identifier:
            raise ValueError("Invalid observation producer ID")
        root = f"{WORK_ROOT}/jobs/runs/{identifier}"
        payload = json.loads(producer["payload_json"])
        remote = payload.get("root")
        if remote is not None and remote != root + "/observations/" + producer["attempt_token"]:
            raise ValueError("Observation producer storage differs from its immutable attempt")
        return root

    @staticmethod
    def _observation_path(artifact):
        spec = json.loads(artifact["spec_json"])
        source = spec.get("source_sha256", "")
        key = artifact["artifact_key"]
        if not re.fullmatch(r"[a-f0-9]{64}", source) or not re.fullmatch(r"[a-f0-9]{64}", key):
            raise ValueError("Invalid observation storage identity")
        # The manifest specification, rather than a mutable stored path, owns bytes.
        import hashlib
        from .database import canonical_json
        if hashlib.sha256(canonical_json(spec).encode()).hexdigest() != key:
            raise ValueError("Observation recipe differs from its storage identity")
        expected = f"{WORK_ROOT}/datasets/observations/{source}/{key}"
        if artifact.get("path") is not None and artifact["path"] != expected:
            raise ValueError("Observation storage differs from its immutable identity")
        return expected

    @classmethod
    def _observation_lock_path(cls, artifact):
        path = cls._observation_path(artifact)
        parent, name = path.rsplit("/", 1)
        return parent + "/." + name + ".publish.lock"

    @staticmethod
    def _delete_observations(c, graph):
        for source in graph.get("observation_sources", []):
            c.execute("DELETE FROM observation_sources WHERE artifact_key=? AND session_id=? AND recording_path=?",
                      (source["artifact_key"], source["session_id"], source["recording_path"]))
        for artifact in graph.get("observation_artifacts", []):
            # The graph is ordered outputs first so input FKs keep guarding cleanup.
            c.execute("DELETE FROM observation_artifacts WHERE artifact_key=?", (artifact["artifact_key"],))
        for producer in graph.get("observation_producers", []):
            # FK ownership and the terminal state remain final database guards.
            c.execute("DELETE FROM observation_producers WHERE id=? AND state IN ('READY','FAILED') AND NOT EXISTS (SELECT 1 FROM observation_artifacts WHERE producer_id=?)",
                      (producer["id"], producer["id"]))

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
        excluded_observations = {row["artifact_key"] for row in (graph or {}).get("observation_artifacts", [])}
        for row in rows(c, "observation_artifacts"):
            if row["artifact_key"] not in excluded_observations:
                result.append((self._observation_path(row), "observation_artifacts", row["artifact_key"]))
                result.append((self._observation_lock_path(row), "observation_artifacts", row["artifact_key"]))
        excluded_producers = {row["id"] for row in (graph or {}).get("observation_producers", [])}
        for row in rows(c, "observation_producers"):
            if row["id"] not in excluded_producers:
                result.append((self._observation_producer_path(row), "observation_producers", row["id"]))
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
        for artifact in graph.get("observation_artifacts", []):
            groups.setdefault(WORK_ROOT, []).extend([
                self._observation_path(artifact), self._observation_lock_path(artifact),
            ])
        for producer in graph.get("observation_producers", []):
            groups.setdefault(WORK_ROOT, []).append(self._observation_producer_path(producer))
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
        self._delete_observations(c, graph)
        # The parent engine removes metadata bodies, events and the durable intent.
        query, params = in_ids("id", [r["id"] for r in graph["live_conversions"]])
        c.execute("DELETE FROM live_conversions WHERE " + query, params)
        super()._delete_rows(c, kind, identifier, graph)
