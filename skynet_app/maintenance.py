"""Workspace-scoped deletion plans and storage inspection.

Internal children are removed together. Independently usable evaluations and
restarted runs are explicit blockers. SQL deletion happens only after files are
removed; a durable intent protects a partially completed deletion until retry.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path, PurePosixPath

from . import history_journals, storage_files
from .database import canonical_json, utc_now
from .workspace_schema import visible_sql
from .workspace_storage import WorkspaceStorage
from .registry_dependencies import extend_graph, suite_removal_notices
from .registry_policy import suppress_defaults

KINDS = {"experiment": "experiments", "run": "runs", "evaluation": "evaluations",
         "adapter": "adapters", "suite": "evaluation_suites"}
TERMINAL = frozenset(
    {
        "SUCCEEDED",
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "CANCELED",
        "TIMEOUT",
        "SKIPPED",
        "DRAFT",
        "BLOCKED",
        "SUBMISSION_FAILED",
    }
)
PATH_COLUMNS = {
    "runs": ("run_directory",),
    "evaluations": ("result_path",),
    "checkpoints": ("path",),
    "artifacts": ("path",),
    "manifests": ("path",),
    "data_resource_versions": ("path",),
    "data_locations": ("path",),
    "evaluation_episodes": ("video_path", "raw_result_path"),
    "job_attempts": ("stdout_path", "stderr_path", "sbatch_path"),
}
JSON_PATH_COLUMNS = {
    "experiment_revisions": "requested_spec_json",
    "variants": "resolved_spec_json",
    "live_xr_sessions": "payload_json",
    "policy_exports": "payload_json",
    "data_bundles": "manifest_json",
}


def paths_in(value):
    if isinstance(value, str):
        if value.startswith("/"):
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from paths_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from paths_in(item)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def rows(connection, table, condition="1=1", params=()):
    return [
        dict(row)
        for row in connection.execute(
            f"SELECT * FROM {table} WHERE {condition}", params
        ).fetchall()
    ]


def in_ids(column, identifiers):
    return (
        (f"{column} IN ({','.join('?' for _ in identifiers)})", tuple(identifiers))
        if identifiers
        else ("1=0", ())
    )


class Maintenance:
    def __init__(self, database, cluster, *, local_capsules=None):
        self.db, self.cluster = database, cluster
        self.storage = WorkspaceStorage(database)
        self.local_capsules = Path(local_capsules) if local_capsules else None

    def remote(self, operation, root, items=None, protected=None, gateway="auto"):
        host = self.cluster.resolve_gateway(gateway)
        script = Path(storage_files.__file__).read_text()
        result = self.cluster.ssh(
            host,
            shlex.join(["python3", "-c", script]),
            stdin=json.dumps(
                {
                    "operation": operation,
                    "root": root,
                    "items": items or [],
                    "protected": protected or [],
                }
            ),
            timeout=120,
        )
        return json.loads(result)

    def _target(self, c, kind, identifier):
        table = KINDS.get(kind)
        if table is None:
            raise ValueError("Unknown history type")
        condition, params = self._identity(kind, identifier)
        target = c.execute(
            f"SELECT * FROM {table} WHERE {condition} AND {visible_sql(table)}"
            + (" ORDER BY version_number DESC" if kind == "adapter" else ""), params
        ).fetchone()
        if target is None:
            raise KeyError("History item not found")
        if kind == "adapter" and self.db.workspace_id is not None and target["owner_id"] != self.db.workspace_id:
            raise KeyError("Adapter is not editable in this workspace")
        if kind == "suite" and self.db.workspace_id not in (None, "legacy"):
            raise KeyError("Only the installation owner can remove default evaluation suites")
        return dict(target)

    @staticmethod
    def _identity(kind, identifier):
        return ("(id=? OR adapter_key=?)", (identifier, identifier)) if kind == "adapter" else ("id=?", (identifier,))

    def _graph(self, c, kind, identifier):
        target = self._target(c, kind, identifier)
        graph = {KINDS[kind]: [target]}
        blockers = []

        def add(table, column, values):
            query, params = in_ids(column, values)
            graph[table] = rows(c, table, query, params)
            return [row["id"] for row in graph[table]]

        def block(kind, record, reason):
            # Dependencies in other workspaces are never disclosed by identifier.
            visible = self.db.workspace_id is None or record.get("owner_id") in (
                None,
                self.db.workspace_id,
            )
            entry = {
                "kind": kind,
                "id": record["id"] if visible else None,
                "label": (
                    record.get("name") or record.get("suite_name") or record["id"]
                )
                if visible
                else "Another workspace's item",
                "reason": reason,
            }
            if not any((b["kind"], b["id"], b["label"]) == (entry["kind"], entry["id"], entry["label"]) for b in blockers):
                blockers.append(entry)

        if kind in {"adapter", "suite"}:
            extend_graph(c, kind, target, graph, block, rows)
        elif kind == "experiment":
            revisions = add("experiment_revisions", "experiment_id", [identifier])
            variants = add("variants", "experiment_revision_id", revisions)
            query, params = in_ids("variant_id", variants)
            for record in rows(c, "runs", query, params):
                block("run", record, "Delete this training run first")
        else:
            run_id = identifier if kind == "run" else target["run_id"]
            if kind == "run":
                for record in rows(c, "evaluations", "run_id=?", (run_id,)):
                    block("evaluation", record, "Delete this evaluation first")
                for record in rows(c, "runs", "restarted_from_run_id=?", (run_id,)):
                    block("run", record, "Delete this restarted run first")
                stages = add("workflow_stages", "run_id", [run_id])
                add("checkpoints", "run_id", [run_id])
                add("manifests", "run_id", [run_id])
                add("artifacts", "run_id", [run_id])
                add("training_progress_samples", "run_id", [run_id])
                add("metrics", "run_id", [run_id])
            else:
                stages = add(
                    "workflow_stages",
                    "id",
                    [target["stage_id"]] if target["stage_id"] else [],
                )
                add("evaluation_episodes", "evaluation_id", [identifier])
                add("metrics", "evaluation_id", [identifier])
                add("artifacts", "evaluation_id", [identifier])
                for record in rows(
                    c,
                    "evaluations",
                    "stage_id=? AND id<>?",
                    (target["stage_id"], identifier),
                ):
                    block(
                        "evaluation",
                        record,
                        "Evaluation shares the same execution stage",
                    )
            attempts = add("job_attempts", "stage_id", stages)
            if kind == "evaluation":
                query, params = in_ids("attempt_id", attempts)
                graph["manifests"] = rows(c, "manifests", query, params)
                known = {r["id"] for r in graph["artifacts"]}
                for record in rows(c, "artifacts", "run_id=?", (run_id,)):
                    metadata = json.loads(record["metadata_json"])
                    if record["id"] not in known and (
                        record["stage_id"] in stages
                        or metadata.get("attempt_id") in attempts
                    ):
                        graph["artifacts"].append(record)
            for record in [
                target,
                *graph.get("workflow_stages", []),
                *graph.get("job_attempts", []),
            ]:
                if record.get("status", "DRAFT") not in TERMINAL:
                    block(
                        kind,
                        target,
                        "Cancel active work and wait for its final state before deleting",
                    )
                    break
            checkpoint_ids = [r["id"] for r in graph.get("checkpoints", [])]
            if checkpoint_ids:
                query, params = in_ids("resume_checkpoint_id", checkpoint_ids)
                for record in rows(c, "job_attempts", query, params):
                    if record["id"] not in attempts:
                        stage = c.execute(
                            "SELECT run_id FROM workflow_stages WHERE id=?",
                            (record["stage_id"],),
                        ).fetchone()
                        parent = dict(
                            c.execute(
                                "SELECT * FROM runs WHERE id=?", (stage[0],)
                            ).fetchone()
                        )
                        block(
                            "run",
                            parent,
                            "This run resumes from a checkpoint being deleted",
                        )
            query, params = in_ids("depends_on_stage_id", stages)
            for dependency in rows(c, "stage_dependencies", query, params):
                if dependency["stage_id"] not in stages:
                    stage = c.execute(
                        "SELECT run_id FROM workflow_stages WHERE id=?",
                        (dependency["stage_id"],),
                    ).fetchone()
                    parent = dict(
                        c.execute(
                            "SELECT * FROM runs WHERE id=?", (stage[0],)
                        ).fetchone()
                    )
                    block(
                        "run",
                        parent,
                        "This run depends on an execution stage being deleted",
                    )
        return target, graph, blockers

    def _references(self, c, graph=None):
        graph = graph or {}
        excluded = {table: {r["id"] for r in data} for table, data in graph.items()}
        refs = []
        for table, columns in PATH_COLUMNS.items():
            for record in rows(c, table):
                if record["id"] not in excluded.get(table, set()):
                    refs.extend(
                        (str(record[key]), table, record["id"])
                        for key in columns
                        if record.get(key)
                    )
        for table, column in JSON_PATH_COLUMNS.items():
            for record in rows(c, table):
                if record["id"] not in excluded.get(table, set()):
                    refs.extend(
                        (path, table, record["id"])
                        for path in paths_in(json.loads(record[column]))
                    )
        return refs

    @staticmethod
    def _referenced(path, references, graph):
        evaluation_runs = {row["run_id"] for row in graph.get("evaluations", [])}
        for reference, table, identifier in references:
            # The retained run owns the container, while evaluation attempts own
            # their children. An exact reference or another consumer still blocks.
            if (
                table == "runs"
                and identifier in evaluation_runs
                and PurePosixPath(reference) in PurePosixPath(path).parents
            ):
                continue
            if storage_files.overlaps(path, reference):
                return True
        return False

    def _file_groups(self, c, kind, target, graph, blockers):
        groups = {}
        if kind in {"experiment", "adapter", "suite"}:
            return groups
        run = (
            target
            if kind == "run"
            else dict(
                c.execute(
                    "SELECT * FROM runs WHERE id=?", (target["run_id"],)
                ).fetchone()
            )
        )
        suffix = "/jobs/runs/" + run["id"]
        if not run["run_directory"].endswith(suffix):
            raise ValueError("Run directory does not match its immutable run ID")
        root = run["run_directory"].removesuffix(suffix)
        files = set()
        if kind == "run":
            files.add(run["run_directory"])
        elif target["stage_id"]:
            expected = f"{root}/eval/runs/{target['stage_id']}"
            if (
                target["result_path"]
                and str(PurePosixPath(target["result_path"]).parent) != expected
            ):
                raise ValueError(
                    "Evaluation result directory does not match its execution stage"
                )
            files.add(expected)
            for attempt in graph.get("job_attempts", []):
                job_id = str(attempt.get("slurm_job_id") or "")
                if job_id.isdecimal():
                    files.add(f"{run['run_directory']}/attempts/{job_id}")
        for table in (
            "checkpoints",
            "artifacts",
            "manifests",
            "evaluation_episodes",
            "job_attempts",
        ):
            for record in graph.get(table, []):
                for field in PATH_COLUMNS[table]:
                    path = record.get(field)
                    if not path or "%" in path:
                        continue
                    # The mutable transport job.sbatch belongs to the parent run;
                    # each attempt's immutable receipt is a separate artifact.
                    if (
                        kind == "evaluation"
                        and field == "sbatch_path"
                        and str(path).startswith(run["run_directory"] + "/")
                    ):
                        continue
                    files.add(str(path))
        object_root = (
            str(self.db.payload_store.objects.root) if self.db.payload_store else None
        )
        references = self._references(c, graph)
        for path in sorted(files, key=len):
            if any(
                PurePosixPath(parent) in PurePosixPath(path).parents or parent == path
                for values in groups.values()
                for parent in values
            ):
                continue
            if self._referenced(path, references, graph):
                # Reusing a file is a dependency, never permission to unlink it.
                blockers.append(
                    {
                        "kind": "storage",
                        "id": None,
                        "label": path,
                        "reason": "Referenced by another retained item; delete that dependency first",
                    }
                )
                continue
            if object_root and path.startswith(object_root + "/"):
                group = object_root
            elif path.startswith(root + "/"):
                group = root
            elif self.local_capsules and path.startswith(
                str(self.local_capsules / run["id"]) + "/"
            ):
                group = "local:" + str(self.local_capsules)
            else:
                blockers.append(
                    {
                        "kind": "storage",
                        "id": None,
                        "label": path,
                        "reason": "File is outside managed storage; relocate it before deleting",
                    }
                )
                continue
            groups.setdefault(group, []).append(path)
        if kind == "run" and self.local_capsules:
            groups["local:" + str(self.local_capsules)] = [
                str(self.local_capsules / run["id"])
            ]
        return groups

    def _exclusive_payloads(self, c, kind, identifier, graph):
        if not self.db.payload_store:
            return []
        owned = {(table, row["id"]) for table, data in graph.items() for row in data}
        if kind == "run":
            owned.add(("tracking_journals", identifier))
        by_digest = {}
        for ref in rows(c, "metadata_payload_refs"):
            by_digest.setdefault(ref["sha256"], set()).add(
                (ref["table_name"], ref["record_id"])
            )
        return [
            dict(row)
            for row in c.execute("SELECT * FROM metadata_payloads").fetchall()
            if by_digest.get(row["sha256"]) and by_digest[row["sha256"]] <= owned
        ]

    def preview(self, kind, identifier, gateway="auto"):
        with self.db.connection() as c:
            target, graph, blockers = self._graph(c, kind, identifier)
            notices = suite_removal_notices(c, target, self.db.workspace_id) if kind == "suite" else []
            groups = (
                {} if blockers else self._file_groups(c, kind, target, graph, blockers)
            )
            payloads = self._exclusive_payloads(c, kind, identifier, graph)
            needles = [item["id"] for data in graph.values() for item in data]
            needles.extend(path for paths in groups.values() for path in paths)
            journals = (
                history_journals.changes(c, self.db, target["run_id"], needles)
                if kind == "evaluation"
                else []
            )
            payloads.extend(
                item for journal in journals for item in journal.get("retired", [])
            )
            pending = c.execute(
                "SELECT plan_json FROM maintenance_operations WHERE target_kind=? AND target_id=?",
                (kind, identifier),
            ).fetchone()
            if pending:
                previous = json.loads(pending[0])
                # Previously committed journal cleanup can leave bodies awaiting
                # deletion after a network failure. Keep them in the retry plan.
                for item in previous["files"]:
                    groups.setdefault(item["root"], []).append(item["path"])
                if self.db.payload_store:
                    for digest in previous["payloads"]:
                        row = c.execute(
                            "SELECT * FROM metadata_payloads WHERE sha256=?", (digest,)
                        ).fetchone()
                        if row:
                            payloads.append(dict(row))
            if payloads:
                groups.setdefault(str(self.db.payload_store.objects.root), []).extend(
                    item["path"] for item in payloads
                )
        files = []
        if not blockers:
            for root, paths in groups.items():
                request = [{"path": path} for path in sorted(set(paths))]
                result = (
                    storage_files.execute(
                        {"operation": "inspect", "root": root[6:], "items": request}
                    )
                    if root.startswith("local:")
                    else self.remote("inspect", root, request, gateway=gateway)
                )
                files.extend({**item, "root": root} for item in result["items"])
        plan = {
            "kind": kind,
            "id": identifier,
            "owner_id": target.get("owner_id") or "legacy",
            "label": (target.get("description") if kind == "suite" else None) or target.get("name") or target.get("suite_name") or identifier,
            "counts": {table: len(data) for table, data in graph.items()},
            "records": {
                table: [item["id"] for item in data] for table, data in graph.items()
            },
            "payloads": sorted({item["sha256"] for item in payloads}),
            "journals": journals,
            "needles": needles,
            "blockers": blockers,
            "notices": notices,
            "files": files,
            "graph_hash": fingerprint(graph),
            "retry": bool(pending),
        }
        plan["token"] = fingerprint(plan)
        return plan

    def delete(self, kind, identifier, token, gateway="auto"):
        # Same lock as submission/reconciliation; SQL serialization also covers
        # concurrent API writes and other app hosts.
        with self.db.operation_lock("pipeline"):
            with self.db.connection() as c:
                try:
                    self._target(c, kind, identifier)
                except KeyError:
                    condition, params = self._identity(kind, identifier)
                    if c.execute(
                        f"SELECT 1 FROM {KINDS[kind]} WHERE {condition}", params,
                    ).fetchone():
                        raise
                    return {"deleted": True, "already_deleted": True}
            plan = self.preview(kind, identifier, gateway)
            if plan["blockers"]:
                raise ValueError("Delete the listed dependencies first")
            if plan["token"] != token:
                raise ValueError(
                    "The item or its files changed. Review the new deletion preview"
                )
            with self.db.transaction() as c:
                _, graph, blockers = self._graph(c, kind, identifier)
                if blockers or fingerprint(graph) != plan["graph_hash"]:
                    raise ValueError("History changed. Review deletion again")
                c.execute(
                    "INSERT INTO maintenance_operations(target_kind,target_id,owner_id,plan_json,created_at) VALUES (?,?,?,?,?) ON CONFLICT(target_kind,target_id) DO UPDATE SET plan_json=excluded.plan_json",
                    (
                        kind,
                        identifier,
                        plan["owner_id"],
                        canonical_json(plan),
                        utc_now(),
                    ),
                )
                history_journals.apply(c, self.db, plan["journals"], plan["needles"])
            # Intent is committed before the first irreversible filesystem change.
            with self.db.transaction() as c:
                _, graph, blockers = self._graph(c, kind, identifier)
                if blockers or fingerprint(graph) != plan["graph_hash"]:
                    raise ValueError("History changed. Review deletion again")
                retained = self._references(c, graph)
                for item in plan["files"]:
                    if self._referenced(item["path"], retained, graph):
                        raise ValueError(
                            "A retained item now references a file in this deletion. Review dependencies again"
                        )
                owned = {
                    (table, row["id"]) for table, data in graph.items() for row in data
                }
                if kind == "run":
                    owned.add(("tracking_journals", identifier))
                for digest in plan["payloads"]:
                    refs = c.execute(
                        "SELECT table_name,record_id FROM metadata_payload_refs WHERE sha256=?",
                        (digest,),
                    ).fetchall()
                    if any((row[0], row[1]) not in owned for row in refs):
                        raise ValueError(
                            "A metadata body is now shared. Review dependencies again"
                        )
                for root in sorted({item["root"] for item in plan["files"]}):
                    items = [item for item in plan["files"] if item["root"] == root]
                    if root.startswith("local:"):
                        storage_files.execute(
                            {"operation": "delete", "root": root[6:], "items": items}
                        )
                    else:
                        self.remote("delete", root, items, gateway=gateway)
                self._delete_rows(c, kind, identifier, graph)
                for digest in plan["payloads"]:
                    c.execute(
                        "DELETE FROM metadata_payloads WHERE sha256=? AND NOT EXISTS (SELECT 1 FROM metadata_payload_refs WHERE sha256=?)",
                        (digest, digest),
                    )
                c.execute(
                    "DELETE FROM maintenance_operations WHERE target_kind=? AND target_id=?",
                    (kind, identifier),
                )
            return {"deleted": True}

    def _delete_rows(self, c, kind, identifier, graph):
        ids = [identifier, *[record["id"] for data in graph.values() for record in data]]
        ids.extend(row["adapter_key"] for row in graph.get("adapters", []))
        query, params = in_ids("entity_id", ids)
        c.execute("SET LOCAL skynet.delete_history='on'")
        c.execute("DELETE FROM events WHERE " + query, params)
        query, params = in_ids("scope_id", ids)
        c.execute("DELETE FROM tracking_bindings WHERE " + query, params)
        for table, data in graph.items():
            query, params = in_ids("record_id", [r["id"] for r in data])
            c.execute(
                "DELETE FROM metadata_payload_refs WHERE table_name=? AND " + query,
                (table, *params),
            )
        if kind == "run":
            c.execute(
                "DELETE FROM metadata_payload_refs WHERE table_name='tracking_journals' AND record_id=?",
                (identifier,),
            )
        if kind in {"adapter", "suite"}:
            table = KINDS[kind]
            suppress_defaults(c, kind, graph[table])
            if kind == "adapter":
                query, params = in_ids("id", [r["id"] for r in graph["adapter_validations"]])
                c.execute("DELETE FROM adapter_validations WHERE " + query, params)
            query, params = in_ids("id", [r["id"] for r in graph[table]])
            c.execute(f"DELETE FROM {table} WHERE " + query, params)
        elif kind == "evaluation":
            for table in ("artifacts", "manifests"):
                query, params = in_ids("id", [r["id"] for r in graph.get(table, [])])
                c.execute(f"DELETE FROM {table} WHERE " + query, params)
            c.execute("DELETE FROM evaluations WHERE id=?", (identifier,))
            query, params = in_ids(
                "id", [r["id"] for r in graph.get("workflow_stages", [])]
            )
            c.execute("DELETE FROM workflow_stages WHERE " + query, params)
        else:
            c.execute(f"DELETE FROM {KINDS[kind]} WHERE id=?", (identifier,))

    def inspect_storage(self, gateway="auto"):
        root = self.storage.require_root()
        with self.db.connection() as c:
            protected = [path for path, _, _ in self._references(c)]
            pending = rows(
                c,
                "maintenance_operations",
                "owner_id=?",
                (self.db.workspace_id or "legacy",),
            )
            unreferenced_payloads = []
            if self.db.payload_store:
                protected.extend(
                    row[0]
                    for row in c.execute(
                        "SELECT path FROM metadata_payloads p WHERE EXISTS (SELECT 1 FROM metadata_payload_refs r WHERE r.sha256=p.sha256)"
                    ).fetchall()
                )
                unreferenced_payloads = [
                    row[0]
                    for row in c.execute(
                        "SELECT path FROM metadata_payloads p WHERE NOT EXISTS (SELECT 1 FROM metadata_payload_refs r WHERE r.sha256=p.sha256)"
                    ).fetchall()
                ]
        result = self.remote("scan", root, protected=protected, gateway=gateway)
        for item in result["items"]:
            item["root"] = root
        if self.db.payload_store:
            object_root = str(self.db.payload_store.objects.root)
            extra = self.remote(
                "scan_objects", object_root, protected=protected, gateway=gateway
            )
            for item in extra["items"]:
                item["root"] = object_root
            result["items"].extend(extra["items"])
            known = {item["path"] for item in result["items"]}
            # Catalog rows from an interrupted cleanup must also be removable.
            missing = [
                path
                for path in unreferenced_payloads
                if path not in known
                and not any(storage_files.overlaps(path, ref) for ref in protected)
            ]
            if missing:
                checked = self.remote(
                    "inspect",
                    object_root,
                    [{"path": path} for path in missing],
                    gateway=gateway,
                )
                result["items"].extend(
                    {
                        **item,
                        "root": object_root,
                        "reason": "Unused metadata reference",
                        "selectable": True,
                    }
                    for item in checked["items"]
                    if not item["exists"]
                )
        result["pending_deletions"] = [
            {"kind": item["target_kind"], "id": item["target_id"]} for item in pending
        ]
        result["token"] = fingerprint(result)
        return result

    def clean_storage(self, selected, token, gateway="auto"):
        with self.db.operation_lock("pipeline"), self.db.transaction() as c:
            report = self.inspect_storage(gateway)
            if token != report["token"]:
                raise ValueError("Storage changed. Scan again before cleaning")
            items = [
                item
                for item in report["items"]
                if item["path"] in selected and item["selectable"]
            ]
            if set(selected) != {item["path"] for item in items} or not items:
                raise ValueError("Select unreferenced files from the current scan")
            deleted = []
            for root in sorted({item["root"] for item in items}):
                result = self.remote(
                    "delete",
                    root,
                    [item for item in items if item["root"] == root],
                    gateway=gateway,
                )
                deleted.extend(result["deleted"])
            if self.db.payload_store:
                for path in deleted:
                    c.execute(
                        "DELETE FROM metadata_payloads WHERE path=? AND NOT EXISTS (SELECT 1 FROM metadata_payload_refs WHERE sha256=metadata_payloads.sha256)",
                        (path,),
                    )
            return {"deleted": deleted}
