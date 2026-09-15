"""Remove a selected native recording through the shared deletion transaction."""

import base64
import copy
import json
import re
from pathlib import PurePosixPath
from uuid import UUID

from .database import canonical_json, utc_now
from .maintenance import Maintenance, fingerprint, in_ids
from .recording_deletion import RecordingMaintenance
from .live_xr_archive import archive_descriptor


def recording_identity(identifier):
    session, separator, encoded = identifier.partition(":")
    try:
        if not separator or str(UUID(session)) != session or len(encoded) > 4096:
            raise ValueError()
        name = base64.b64decode(
            encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True
        ).decode("utf-8")
        if base64.urlsafe_b64encode(name.encode()).decode().rstrip("=") != encoded:
            raise ValueError()
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.is_relative_to("recordings")
            or path.suffix != ".pkl"
            or str(path) != name
            or "\x00" in name
        ):
            raise ValueError()
    except (ValueError, UnicodeError) as error:
        raise ValueError("Invalid recording file identity") from error
    return session, name


class RecordingFileMaintenance(RecordingMaintenance):
    def _identity(self, kind, identifier):
        if kind == "recording-file":
            recording_identity(identifier)
            # A missing selected path is already deleted, even if its session remains.
            return "1=0", ()
        return super()._identity(kind, identifier)

    def _target(self, c, kind, identifier):
        if kind == "recording":
            return super()._target(c, kind, identifier)
        session, name = recording_identity(identifier)
        target = super()._target(c, "recording", session)
        job = json.loads(target["payload_json"])
        if name not in job.get("recordings", []):
            raise KeyError("Recording file not found")
        index = job["recordings"].index(name)
        slot = self.reviews.slot(job, index)
        return dict(
            target,
            selected_path=name,
            selected_slot=slot,
            name=f"Recording {slot + 1} · {target['name']}",
        )

    def _graph(self, c, kind, identifier):
        target = self._target(c, kind, identifier)
        _, graph, blockers = super()._graph(
            c, "recording", target["id"], pending_target=(kind, identifier)
        )
        graph["live_xr_sessions"] = [target]
        return target, graph, blockers

    def preview(self, kind, identifier, gateway="auto"):
        plan = super().preview(kind, identifier, gateway)
        plan["notices"].append(
            "Only the selected recording and its related files will be deleted. Other recordings and the collection session will remain."
        )
        plan["counts"] = {
            "recordings": 1,
            "conversion_attempts": plan["counts"].get("live_conversions", 0),
        }
        plan.pop("token")
        plan["token"] = fingerprint(plan)
        return plan

    def _removed_paths(self, job, name, slot):
        path = PurePosixPath(name)
        siblings = {
            str(path.with_suffix(suffix))
            for suffix in (".pkl", ".hdf5", ".mp4", ".json")
        }
        image = (job.get("recording_images") or {}).get(name)
        if image and image.get("path"):
            image_path = PurePosixPath(image["path"])
            if image_path.parent != path.parent or image_path.suffix != ".hdf5":
                raise ValueError("Recording image location does not match its source")
            siblings.add(str(image_path))
        selected = {"output/" + item for item in siblings}
        prefixes = [f"output/reviews/{slot}/"]
        checksum = (job.get("recording_checksums") or {}).get(name)
        other_checksums = {
            v for k, v in (job.get("recording_checksums") or {}).items() if k != name
        }
        if checksum and checksum not in other_checksums:
            if not re.fullmatch(r"[a-f0-9]{64}", checksum):
                raise ValueError("Invalid recording checksum")
            prefixes.append(f"output/review-videos/{checksum}/")
        retained = {"output/" + item for item in job["recordings"] if item != name}
        retained.update(
            "output/" + image["path"]
            for key, image in (job.get("recording_images") or {}).items()
            if key != name and image.get("path")
        )
        if selected & retained:
            raise ValueError("Another recording shares this recording's files")
        return {
            item["path"]
            for item in job["archive"]["manifest"]["files"]
            if item["path"] in selected
            or any(item["path"].startswith(prefix) for prefix in prefixes)
        }

    def _file_groups(self, c, kind, target, graph, blockers):
        # Reuse canonical path and video-generation validation from session deletion.
        groups = super()._file_groups(c, "recording", target, graph, blockers)
        job = json.loads(target["payload_json"])
        archive = archive_descriptor(job, self.cluster)
        name, slot = target["selected_path"], target["selected_slot"]
        removed = self._removed_paths(job, name, slot)
        if "output/" + name not in removed:
            raise ValueError(
                "The selected recording is absent from its verified archive"
            )
        root = str(PurePosixPath(archive["root"]).parent)
        base = str(PurePosixPath(root).parent)
        dataset_root = next(key for key, paths in groups.items() if base in paths)
        groups[dataset_root] = [root + "/" + path for path in sorted(removed)]
        groups[dataset_root].extend(
            [
                self.live.archive.derived_root(job) + f"/reviews/{slot}",
                base + "/manifests/" + archive["manifest_sha256"] + ".json",
            ]
        )
        local = self.reviews.root / job["id"] / str(slot)
        groups["local:" + str(self.reviews.root)] = [str(local)]
        selected_videos = {
            json.loads(path.read_text()).get("remote_root")
            for path in local.glob("video-*/status.json")
        }
        for key, paths in groups.items():
            groups[key] = [
                path
                for path in paths
                if not re.search(r"/jobs/runs/[a-f0-9]{32}$", path)
                or path in selected_videos
            ]
        return groups

    def delete(self, kind, identifier, token, gateway="auto"):
        session, _ = recording_identity(identifier)
        lock = self.live.recording_lock(session)
        if not lock.acquire(blocking=False):
            raise ValueError(
                "Recording processing is active. Review deletion again after it finishes."
            )
        try:
            result = Maintenance.delete(self, kind, identifier, token, gateway)
            # These caches are keyed by the visible list index, which may have shifted.
            with self.previews.lock:
                for key in list(self.previews.states):
                    if key[0] == session:
                        self.previews.states.pop(key)
            with self.videos.lock:
                for key in list(self.videos.generations):
                    if key[0] == session:
                        self.videos.generations.pop(key)
            return result
        finally:
            lock.release()

    def _delete_rows(self, c, kind, identifier, graph):
        target = graph["live_xr_sessions"][0]
        job = json.loads(target["payload_json"])
        name, slot = target["selected_path"], target["selected_slot"]
        removed = self._removed_paths(job, name, slot)
        archive = copy.deepcopy(job["archive"])
        archive["storage_key"] = archive.get("storage_key", archive["manifest_sha256"])
        archive["manifest"]["files"] = [
            item for item in archive["manifest"]["files"] if item["path"] not in removed
        ]
        archive["manifest_sha256"] = fingerprint(archive["manifest"])
        # Verify remaining bytes and atomically publish the new inventory before SQL commits.
        from .cluster_config import CLUSTER

        result = self.live.archive._call(
            self.cluster,
            "sky2",
            "revise",
            session_id=job["id"],
            datasets_root=CLUSTER.paths.datasets,
            manifest=archive["manifest"],
            manifest_sha256=archive["manifest_sha256"],
            storage_key=archive["storage_key"],
        )
        if (
            not result.get("verified")
            or result.get("manifest_sha256") != archive["manifest_sha256"]
            or result.get("root") != archive["root"]
        ):
            raise ValueError("Remaining recording archive verification did not match")
        job["recording_slots"] = {
            path: self.reviews.slot(job, i)
            for i, path in enumerate(job["recordings"])
            if path != name
        }
        job["recordings"] = [path for path in job["recordings"] if path != name]
        for field in ("recording_checksums", "recording_images"):
            if field in job:
                job[field].pop(name, None)
        job.pop("recording_summary", None)
        job.update(archive=archive, updated_at=utc_now())
        c.execute("SET LOCAL skynet.delete_history='on'")
        children = [row["id"] for row in graph["live_conversions"]]
        for table, field in (("events", "entity_id"), ("tracking_bindings", "scope_id")):
            query, params = in_ids(field, children)
            c.execute(f"DELETE FROM {table} WHERE " + query, params)
        query, params = in_ids("id", [row["id"] for row in graph["live_conversions"]])
        c.execute("DELETE FROM live_conversions WHERE " + query, params)
        for table, records in graph.items():
            query, params = in_ids("record_id", [row["id"] for row in records])
            c.execute(
                "DELETE FROM metadata_payload_refs WHERE table_name=? AND " + query,
                (table, *params),
            )
        c.execute(
            "UPDATE live_xr_sessions SET payload_json=? WHERE id=?",
            (canonical_json(job), job["id"]),
        )
