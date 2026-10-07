"""Recording ownership plugged into the shared history deletion engine."""

import json
from pathlib import PurePosixPath
import re
from pathlib import Path
from uuid import UUID

from .cluster_config import CLUSTER
from .cluster_runtime import DEFAULT_GATEWAY, WORK_ROOT
from .dataset_catalog import recording_ids
from .live_xr_archive import archive_descriptor, TERMINAL_STATES
from .maintenance import Maintenance, references, rows, in_ids
from .workspace_schema import LEGACY_WORKSPACE


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


def document_filter(identifier, needles=()):
    """A jsonpath accepting every string value mentions() or references() can match.

    The database selects candidate rows with it and the Python predicates still
    decide, so a rejected document could never have matched: recording_ids()
    returns values stored in the documents it reads, mentions() needs the
    identifier as a whole value or path segment, references() a needle or a
    child of an absolute needle path.
    """
    clauses = ["@ like_regex " + json.dumps("(^|/)" + re.escape(identifier) + "(/|$)")]
    variables = {}
    if needles:
        variables["needles"] = sorted(needles)
        clauses.append("@ == $needles[*]")
    for index, prefix in enumerate(sorted({path.rstrip("/") + "/" for path in needles if path.startswith("/")})):
        variables[f"prefix{index}"] = prefix
        clauses.append(f"@ starts with $prefix{index}")
    return 'lax $.** ? (@.type() == "string" && (' + " || ".join(clauses) + "))", json.dumps(variables)


# Every non-empty *_json column of a row, as the snapshot scan reads them. The
# CASE keeps the cast off non-document columns whatever the planner's order.
DOCUMENT_COLUMNS = (
    "EXISTS (SELECT 1 FROM jsonb_each_text(to_jsonb({table})) AS document WHERE jsonb_path_exists("
    "CASE WHEN right(document.key, 5) = '_json' AND document.value <> '' THEN document.value::jsonb END, "
    "?::jsonpath, ?::jsonb))"
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
        return super().remote(operation, root, items, protected, DEFAULT_GATEWAY)

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
            owner_id=self.db.workspace_id or LEGACY_WORKSPACE,
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
                "Wait for transfer to the training cluster and collection workstation cleanup to finish",
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
        source_references = self._dataset_graph(c, identifier, graph, block)
        for record in rows(c, "live_conversions", "payload_json::jsonb->>'session_id'=?", (identifier,), order_by="id"):
            conversion = json.loads(record["payload_json"])
            graph["live_conversions"].append(record)
            if conversion.get("state") not in {"READY", "FAILED", "CANCELLED"}:
                block(
                    "recording",
                    identifier,
                    conversion.get("name") or "Dataset conversion",
                    "Wait for this dataset conversion to finish",
                )
        path, variables = document_filter(identifier)
        for record in rows(c, "live_xr_sessions", "id<>? AND jsonb_path_exists(payload_json::jsonb, ?::jsonpath, ?::jsonb)",
                           (identifier, path, variables), order_by="id"):
            child = json.loads(record["payload_json"])
            if mentions(child, identifier):
                block(
                    "recording",
                    record["id"],
                    child.get("profile", {}).get("task_name") or record["id"],
                    "Delete this dependent recording first",
                )
        # Experiment and evaluation snapshots embed dataset sources that name
        # recordings. Their documents stay in SQL (payload_store.FIELDS offloads
        # none of these tables), so the database reads them and only candidates travel.
        path, variables = document_filter(identifier, source_references)
        for table, kind, parent in (
            ("experiment_revisions", "experiment", "experiment_id"),
            ("variants", "experiment", "experiment_id"),
            ("evaluations", "evaluation", "id"),
        ):
            for record in rows(c, table, DOCUMENT_COLUMNS.format(table=table), (path, variables), order_by="id"):
                if any(
                    mentions(json.loads(v), identifier) or references(json.loads(v), source_references)
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

    def _dataset_graph(self, c, identifier, graph, block):
        """Block on output versions; collect only unused internal source metadata."""
        def label(version_id, default):
            row = c.execute("SELECT display_name FROM data_dataset_presentations WHERE version_id=?",
                            (version_id,)).fetchone()
            return row[0] if row else default

        def documents(records):
            for record in records:
                record["metadata"] = json.loads(record["metadata_json"])
            return records

        # recording_ids() reads a version's metadata, its resource's and its
        # source version's; a recording named in none of them is not a member.
        member, member_variables = document_filter(identifier)
        candidates = documents([dict(row) for row in c.execute(
            "SELECT v.* FROM data_resource_versions v JOIN data_resources r ON r.id=v.resource_id "
            "LEFT JOIN data_resource_versions s ON s.id=v.metadata_json::jsonb->>'source_version_id' "
            "WHERE r.category='dataset' AND (jsonb_path_exists(v.metadata_json::jsonb, ?::jsonpath, ?::jsonb) "
            "OR jsonb_path_exists(r.metadata_json::jsonb, ?::jsonpath, ?::jsonb) "
            "OR jsonb_path_exists(s.metadata_json::jsonb, ?::jsonpath, ?::jsonb)) ORDER BY v.id",
            (member, member_variables) * 3,
        ).fetchall()])
        resources = {row["id"]: row for row in documents(rows(
            c, "data_resources", "id = ANY(?)", (sorted({v["resource_id"] for v in candidates}),)))} if candidates else {}
        source_keys = sorted({v["metadata"].get("source_version_id") for v in candidates
                              if isinstance(v["metadata"].get("source_version_id"), str)})
        sources = {row["id"]: row for row in documents(rows(
            c, "data_resource_versions", "id = ANY(?)", (source_keys,)))} if source_keys else {}
        source_versions = []
        for version in candidates:
            resource = resources[version["resource_id"]]
            source = sources.get(version["metadata"].get("source_version_id"))
            if identifier not in recording_ids(resource, version, source):
                continue
            if version["format"] != "skynet.episodes/v1":
                block("dataset", version["id"], label(version["id"], resource["display_name"]),
                      "Delete this dataset first")
            elif (resource["provider"], resource["namespace"]) == ("collection", "datasets"):
                source_versions.append(version)
            else:
                block("storage", None, resource["display_name"],
                      "A retained source registration references this recording; remove that dependency first")
        source_ids = sorted(version["id"] for version in source_versions)
        locations = rows(c, "data_locations", "version_id = ANY(?)", (source_ids,), order_by="id") if source_ids else []
        needles = {value for version in source_versions
                   for value in (version["id"], version["manifest_sha256"], version["path"])}
        needles.update(row["path"] for row in locations)
        if source_ids:
            # These rows have real foreign keys or immutable JSON receipts. A raw
            # source can be hidden from the catalog and still have retained consumers.
            for row in c.execute(
                "SELECT d.output_version_id FROM data_derivation_inputs i JOIN data_derivations d ON d.id=i.derivation_id "
                "WHERE i.input_version_id = ANY(?) ORDER BY d.output_version_id", (source_ids,),
            ).fetchall():
                block("dataset", row[0], label(row[0], "Derived dataset"), "Delete this dataset first")
            for table, columns in (
                ("data_bundle_assignments", ("version_id",)),
                ("data_derivations", ("output_version_id",)),
                ("collection_sessions", ("registered_version_id",)),
                ("data_imports", ("version_id",)),
                ("data_version_retirements", ("version_id", "replacement_version_id")),
            ):
                if c.execute(f"SELECT 1 FROM {table} WHERE " + " OR ".join(f"{column} = ANY(?)" for column in columns)
                             + " LIMIT 1", (source_ids,) * len(columns)).fetchone():
                    block("storage", None, "Recording source metadata",
                          "A retained registry item references this recording source; remove that dependency first")
        path, variables = document_filter(identifier, needles)
        exports = [(row["id"], json.loads(row["payload_json"])) for row in rows(
            c, "policy_exports", "jsonb_path_exists(payload_json::jsonb, ?::jsonpath, ?::jsonb)", (path, variables), order_by="id")]
        exports = [(key, export) for key, export in exports if mentions(export, identifier) or references(export, needles)]
        output_keys = sorted({export["version_id"] for _, export in exports if isinstance(export.get("version_id"), str)})
        outputs = {row[0] for row in c.execute(
            "SELECT id FROM data_resource_versions WHERE id = ANY(?)", (output_keys,)).fetchall()} if output_keys else set()
        for key, export in exports:
            if export.get("version_id") in outputs:
                output = export["version_id"]
                block("dataset", output, label(output, export.get("name", "Dataset")), "Delete this dataset first")
            else:
                block("prepared", key, export.get("name", "Dataset conversion"), "Delete this conversion attempt first")
        if source_versions:
            graph["data_resource_versions"] = source_versions
            graph["data_locations"] = locations
        # Keep groups used by any other versions or import/collection records.
        removable = [resource for resource in documents(rows(
            c, "data_resources",
            "provider='collection' AND namespace='datasets' AND category='dataset' "
            "AND jsonb_path_exists(metadata_json::jsonb, ?::jsonpath, ?::jsonb) "
            "AND NOT EXISTS (SELECT 1 FROM data_resource_versions v WHERE v.resource_id=data_resources.id AND NOT (v.id = ANY(?::text[]))) "
            "AND NOT EXISTS (SELECT 1 FROM data_imports i WHERE i.resource_id=data_resources.id) "
            "AND NOT EXISTS (SELECT 1 FROM collection_sessions s WHERE s.registered_resource_id=data_resources.id)",
            (member, member_variables, source_ids), order_by="id"))
            if identifier in recording_ids(resource, {})]
        if removable:
            graph["data_resources"] = removable
            needles.update(resource["id"] for resource in removable)
        return needles

    def _observation_graph(self, c, job, selected_path, graph, blockers):
        """Plan observation ownership before any remote bytes can be removed."""
        detached = "session_id=? AND (?::text IS NULL OR recording_path=?)"
        identity = (job["id"], selected_path, selected_path)
        selected = rows(c, "observation_sources", detached, identity)
        if not selected:
            return
        affected = sorted({row["artifact_key"] for row in selected})
        # An artifact another source edge still names keeps its bytes.
        shared = {row[0] for row in c.execute(
            f"SELECT DISTINCT artifact_key FROM observation_sources WHERE artifact_key = ANY(?) AND NOT ({detached})",
            (affected, *identity)).fetchall()}
        candidates = set(affected) - shared
        keys = sorted(candidates)
        artifacts = {row["artifact_key"]: row for row in rows(c, "observation_artifacts", "artifact_key = ANY(?)", (affected,))}
        inputs = rows(c, "observation_artifact_inputs", "artifact_key = ANY(?) OR input_key = ANY(?)", (keys, keys),
                      order_by="artifact_key, input_key") if keys else []
        removed_versions = {row["id"] for row in graph.get("data_resource_versions", [])}
        consumers = []
        if keys:
            consumers = [("dataset version", row["version_id"], row["artifact_key"])
                         for row in rows(c, "observation_version_inputs", "artifact_key = ANY(?)", (keys,),
                                         order_by="version_id, artifact_key")
                         if row["version_id"] not in removed_versions]
            consumers += [("dataset preparation", row["job_id"], row["artifact_key"])
                          for row in rows(c, "observation_job_inputs", "artifact_key = ANY(?)", (keys,),
                                          order_by="job_id, artifact_key")]
        producer_ids = sorted({row["producer_id"] for row in artifacts.values() if row.get("producer_id")})
        producers = {row["id"]: row for row in rows(c, "observation_producers", "id = ANY(?)", (producer_ids,))} if producer_ids else {}
        def block(identifier, label, reason):
            entry = dict(kind="observation", id=identifier, label=label, reason=reason)
            if entry not in blockers:
                blockers.append(entry)
        for key in affected:
            producer = producers.get(artifacts[key].get("producer_id"))
            if producer and producer["state"] not in {"READY", "FAILED"}:
                block(producer["id"], "Observation preparation",
                      "Wait for the shared observation producer to finish before deleting its recording")
        for kind, identifier, key in consumers:
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
            # Retried failures may no longer own the artifact FK; keep their
            # original request keys in the lifetime check too.
            terminal = rows(
                c, "observation_producers",
                "state IN ('READY','FAILED') AND (EXISTS (SELECT 1 FROM observation_artifacts a "
                "WHERE a.producer_id=observation_producers.id AND a.artifact_key = ANY(?)) "
                "OR jsonb_path_exists(payload_json::jsonb, ?::jsonpath, ?::jsonb))",
                (keys, "lax $.request.requests[*].artifact_key ? (@ == $keys[*])", json.dumps({"keys": keys})),
                order_by="id")
            owned, requested = {}, {}
            if terminal:
                for row in c.execute("SELECT producer_id, artifact_key FROM observation_artifacts WHERE producer_id = ANY(?)",
                                     ([producer["id"] for producer in terminal],)).fetchall():
                    owned.setdefault(row[0], set()).add(row[1])
            for producer in terminal:
                payload = json.loads(producer["payload_json"])
                requested[producer["id"]] = {node["artifact_key"] for node in payload.get("request", {}).get("requests", [])}
            requested_keys = sorted(set().union(*requested.values())) if requested else []
            existing = {row[0] for row in c.execute("SELECT artifact_key FROM observation_artifacts WHERE artifact_key = ANY(?)",
                                                    (requested_keys,)).fetchall()} if requested_keys else set()
            removable_producers = []
            for producer in terminal:
                related = owned.get(producer["id"], set()) | (requested[producer["id"]] & existing)
                if related & candidates and related <= candidates:
                    removable_producers.append(producer)
            if removable_producers:
                graph["observation_producers"] = removable_producers

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
        if spec.get("schema") == "skynet.recording-file/v1":
            from pathlib import PurePosixPath
            path = PurePosixPath(spec.get("path", ""))
            root = PurePosixPath(WORK_ROOT) / "datasets"
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError("Invalid shared recording path")
            if spec.get("owned") is True:
                if (path.parent.parent != root / "recordings" / source / "state"
                        or path.name != "values.hdf5" or not re.fullmatch(r"[a-f0-9]{64}", path.parent.name)):
                    raise ValueError("Shared state storage differs from its recording identity")
            elif spec.get("owned") is False:
                if not path.is_relative_to(root / "raw/dexverse-live") or path.suffix != ".hdf5":
                    raise ValueError("Capture reference is outside the raw recording archive")
            else:
                raise ValueError("Shared recording ownership is missing")
            expected = str(path)
        else:
            modality, camera = spec.get("modality", ""), spec.get("camera_id", "")
            if modality not in {"rgb", "depth", "point_cloud"} or not re.fullmatch(r"[A-Za-z0-9_.-]+", camera) or camera in {'.', '..'}:
                raise ValueError("Invalid shared observation identity")
            expected = f"{WORK_ROOT}/datasets/recordings/{source}/{modality}/{camera}/{key}"
        if artifact.get("path") is not None and artifact["path"] != expected:
            raise ValueError("Observation storage differs from its immutable identity")
        return expected

    @classmethod
    def _observation_lock_path(cls, artifact):
        path = cls._observation_path(artifact)
        spec = json.loads(artifact["spec_json"])
        if spec.get("schema") == "skynet.recording-file/v1":
            if not spec["owned"]:
                return None
            parent = str(PurePosixPath(path).parent)
            return str(PurePosixPath(parent).parent / ("." + PurePosixPath(parent).name + ".lock"))
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
                lock_path = self._observation_lock_path(row)
                if lock_path:
                    result.append((lock_path, "observation_artifacts", row["artifact_key"]))
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
            spec = json.loads(artifact["spec_json"])
            if spec.get("schema") == "skynet.recording-file/v1" and not spec["owned"]:
                continue  # The raw archive owns this file, not an individual stream.
            path = self._observation_path(artifact)
            if spec.get("schema") == "skynet.recording-file/v1":
                path = str(PurePosixPath(path).parent)
            groups.setdefault(WORK_ROOT, []).extend(p for p in [path, self._observation_lock_path(artifact)] if p)
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
        self._delete_source_provenance(c, graph)
        self._delete_observations(c, graph)
        # The parent engine removes metadata bodies, events and the durable intent.
        query, params = in_ids("id", [r["id"] for r in graph["live_conversions"]])
        c.execute("DELETE FROM live_conversions WHERE " + query, params)
        super()._delete_rows(c, kind, identifier, graph)

    @staticmethod
    def _delete_source_provenance(c, graph):
        if not graph.get("data_resource_versions") and not graph.get("data_resources"):
            return
        c.execute("SET LOCAL skynet.delete_history='on'")
        c.execute("SET LOCAL skynet.allow_dataset_delete='on'")
        for table in ("data_locations", "data_resource_versions", "data_resources"):
            query, params = in_ids("id", [row["id"] for row in graph.get(table, [])])
            c.execute(f"DELETE FROM {table} WHERE " + query, params)
