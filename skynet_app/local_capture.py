"""Offline capture ingestion. Providers validate native files without executing them.

Adding a recorder means implementing CaptureProvider and registering it in PROVIDERS.
Storage and dataset registration do not depend on a particular headset or simulator.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from .database import Database, canonical_json, utc_now
from .local_capture_storage import CaptureStorage, MemoryCapture, MAX_CAPTURE_BYTES

MAX_LINE_BYTES = 256 * 1024


class CaptureProvider(Protocol):
    key: str
    native_format: str

    def inspect(self, path: Path) -> dict[str, Any]: ...


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


def _matrix(value: Any, label: str) -> None:
    if not isinstance(value, list) or len(value) != 16:
        raise ValueError(f"{label} must contain 16 column-major numbers")
    for number in value:
        _number(number, label)


class VisionProTracking:
    key = "visionpro-local"
    native_format = "visionpro_tracking_jsonl_v1"

    def inspect(self, path: Path) -> dict[str, Any]:
        header = None
        footer = None
        frames = 0
        first_time = last_time = None
        hands = {"left": 0, "right": 0}
        head_frames = 0
        with path.open("rb") as source:
            line_number = 0
            while line := source.readline(MAX_LINE_BYTES + 1):
                line_number += 1
                if len(line) > MAX_LINE_BYTES:
                    raise ValueError(f"Line {line_number} exceeds the 256 KB record limit")
                if not line.strip():
                    raise ValueError(f"Empty record at line {line_number}")
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError, RecursionError) as error:
                    raise ValueError(f"Invalid JSON at line {line_number}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"Line {line_number} must be a JSON object")
                if footer is not None:
                    raise ValueError("Unexpected records after the recording footer")
                kind = record.get("type")
                if header is None:
                    if kind != "header" or record.get("schema") != "skynet.visionpro-tracking/v1":
                        raise ValueError("Unsupported capture. Choose a Skynet Capture tracking JSONL file (schema v1).")
                    try:
                        UUID(record.get("session_id", ""))
                    except (ValueError, TypeError, AttributeError) as error:
                        raise ValueError("The recording has an invalid session ID") from error
                    for name, expected in {"units": "meters", "matrix_order": "column-major",
                                           "clock": "arkit-monotonic-seconds", "robot_actions": False,
                                           "images_recorded": False}.items():
                        if type(record.get(name)) is not type(expected) or record[name] != expected:
                            raise ValueError(f"Unsupported capture metadata: {name} must be {expected!r}")
                    if not isinstance(record.get("task"), str) or not record["task"].strip():
                        raise ValueError("A recording must name its task")
                    header = record
                elif kind == "frame":
                    if type(record.get("index")) is not int or record["index"] != frames:
                        raise ValueError(f"Missing or out-of-order frame at line {line_number}")
                    timestamp = _number(record.get("timestamp"), "Frame timestamp")
                    _number(record.get("source_timestamp"), "Source timestamp")
                    if "head_timestamp" in record:
                        _number(record["head_timestamp"], "Head timestamp")
                    if last_time is not None and timestamp < last_time:
                        raise ValueError(f"Receive timestamps go backwards at frame {frames}")
                    first_time = timestamp if first_time is None else first_time
                    last_time = timestamp
                    side = record.get("hand")
                    if side not in hands or type(record.get("tracked")) is not bool or type(record.get("head_tracked")) is not bool:
                        raise ValueError(f"Invalid tracking status at frame {frames}")
                    _matrix(record.get("origin_from_hand"), "Hand pose")
                    if record.get("origin_from_head") is not None:
                        _matrix(record["origin_from_head"], "Head pose")
                    elif record["head_tracked"]:
                        raise ValueError("A tracked head must include its pose")
                    joints = record.get("joints")
                    if not isinstance(joints, list) or len(joints) > 64:
                        raise ValueError("A hand record must contain a list of at most 64 joints")
                    names = set()
                    for joint in joints:
                        if not isinstance(joint, dict) or not isinstance(joint.get("name"), str) or not joint["name"] or joint["name"] in names:
                            raise ValueError("Joint names must be nonempty and unique within a hand frame")
                        names.add(joint["name"])
                        if type(joint.get("tracked")) is not bool:
                            raise ValueError("Each joint needs an explicit tracking status")
                        _matrix(joint.get("anchor_from_joint"), "Joint pose")
                    if record["tracked"] and not joints:
                        raise ValueError("A tracked hand must include joint data")
                    hands[side] += int(record["tracked"])
                    head_frames += int(record["head_tracked"])
                    frames += 1
                elif kind == "footer":
                    footer = record
                else:
                    raise ValueError(f"Unsupported record type at line {line_number}: {kind!r}")
        if header is None or footer is None:
            raise ValueError("Recording is incomplete: header or final save record is missing. Keep the original file for recovery and record again.")
        if footer.get("completed") is not True:
            raise ValueError(f"Recording was interrupted: {footer.get('stop_reason', 'unknown reason')}. It is not a completed capture; keep the original file and record again.")
        if type(footer.get("frames")) is not int or footer["frames"] != frames:
            raise ValueError("Saved frame count does not match the recording contents")
        if not frames or not sum(hands.values()):
            raise ValueError("No tracked hand frames were recorded. Keep your hands visible and try again.")
        return {"header": header, "frames": frames, "tracked_hand_frames": hands,
                "head_tracked_frames": head_frames, "duration_seconds": last_time - first_time,
                "warnings": (["No head poses were tracked."] if not head_frames else [])
                + [f"No {side} hand frames were tracked." for side, count in hands.items() if not count],
                "training_compatibility": "Raw human tracking. Requires an explicit robot retargeting/conversion pipeline; no training adapter consumes this format directly."}


PROVIDERS: dict[str, CaptureProvider] = {"visionpro-local": VisionProTracking()}


class LocalCaptureService:
    def __init__(self, database: Database, root: Path | None = None, *, storage=None):
        self.database = database
        # Only used to identify legacy files during explicit verified migration.
        self.root = root or database.data_root / "local-captures"
        self.storage = storage or CaptureStorage()
        self._lock = threading.RLock()
        with database.transaction() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS local_captures (
                sha256 TEXT PRIMARY KEY, provider TEXT NOT NULL, session_id TEXT NOT NULL,
                filename TEXT NOT NULL, size_bytes INTEGER NOT NULL, summary_json TEXT NOT NULL,
                version_id TEXT NOT NULL, created_at TEXT NOT NULL,
                UNIQUE(provider, session_id))""")

    def list(self) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute("SELECT * FROM local_captures ORDER BY created_at DESC").fetchall()
        return [self._public(row) for row in rows]

    def _public(self, row: Any) -> dict[str, Any]:
        item = dict(row)
        item["summary"] = json.loads(item.pop("summary_json"))
        item["location"] = "sky2" if self._location(item) else "Awaiting cluster migration"
        return item

    def _location(self, row):
        version = self.database.get_data_resource_version(row["version_id"])
        expected = self.storage.path(row["sha256"])
        return next((location for location in (version or {}).get("locations", [])
                     if location["kind"] == "cluster" and location["host"] == "skynet"
                     and location["status"] == "AVAILABLE" and location["path"] == expected
                     and location["manifest_sha256"] == row["sha256"]), None)

    @staticmethod
    def _record_location(connection, row, path):
        version = connection.execute("SELECT manifest_sha256,size_bytes FROM data_resource_versions WHERE id=?", (row["version_id"],)).fetchone()
        if version is None or version["manifest_sha256"] != row["sha256"] or version["size_bytes"] != row["size_bytes"]:
            raise ValueError("Recording registration no longer matches its original bytes")
        connection.execute("""INSERT INTO data_locations VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(version_id,host,path) DO UPDATE SET status=excluded.status,verified_at=excluded.verified_at""",
            (str(uuid4()), row["version_id"], "cluster", "skynet", path, row["sha256"], "AVAILABLE", utc_now()))

    def import_file(self, provider_key: str, path: Path) -> dict[str, Any]:
        if path.is_symlink() or not path.is_file():
            raise ValueError("Choose a regular recording file")
        if path.stat().st_size > MAX_CAPTURE_BYTES:
            raise ValueError("Recording exceeds the 512 MB import limit. Record shorter sessions.")
        with path.open("rb") as stream:
            data = stream.read(MAX_CAPTURE_BYTES + 1)
        return self.import_bytes(provider_key, data)

    def import_bytes(self, provider_key: str, data: bytes) -> dict[str, Any]:
        provider = PROVIDERS.get(provider_key)
        if provider is None:
            raise ValueError(f"Unsupported collection provider: {provider_key}")
        if len(data) > MAX_CAPTURE_BYTES:
            raise ValueError("Recording exceeds the 512 MB import limit. Record shorter sessions.")
        digest = hashlib.sha256(data).hexdigest()
        with self._lock:
            with self.database.connection() as connection:
                existing = connection.execute("SELECT * FROM local_captures WHERE sha256=?", (digest,)).fetchone()
            if existing is not None and existing["provider"] != provider_key:
                raise ValueError(f"This file was imported with provider {existing['provider']}; choose that collection provider")
            summary = json.loads(existing["summary_json"]) if existing else provider.inspect(MemoryCapture(data))
            session_id = summary["header"]["session_id"]
            with self.database.connection() as connection:
                conflict = connection.execute("SELECT sha256 FROM local_captures WHERE provider=? AND session_id=?",
                                              (provider_key, session_id)).fetchone()
            if conflict is not None and conflict["sha256"] != digest:
                raise ValueError("This session ID was already imported with different contents. The original capture is immutable.")
            # Transfer before the atomic local metadata publication. No database lock spans SSH.
            destination = self.storage.publish(data, digest)
            with self.database.transaction() as connection:
                existing = connection.execute("SELECT * FROM local_captures WHERE sha256=?", (digest,)).fetchone()
                conflict = connection.execute("SELECT sha256 FROM local_captures WHERE provider=? AND session_id=?",
                                              (provider_key, session_id)).fetchone()
                if conflict is not None and conflict["sha256"] != digest:
                    raise ValueError("This session ID was already imported with different contents. The original capture is immutable.")
                if existing is not None:
                    self._record_location(connection, existing, destination)
                    row, imported = existing, False
                else:
                    resource = connection.execute("SELECT id FROM data_resources WHERE provider='collection' AND namespace=? AND name=?",
                                                  (provider_key, session_id)).fetchone()
                    if resource is None:
                        resource = self.database._insert_data_resource(connection, category="dataset", provider="collection", namespace=provider_key,
                            name=session_id, kind="raw_capture", description=summary["header"]["task"], metadata={"storage_location": "cluster"})
                    version = connection.execute("SELECT id FROM data_resource_versions WHERE resource_id=? AND revision=?",
                                                 (resource["id"], digest)).fetchone()
                    if version is None:
                        version = self.database._insert_data_resource_version(connection, resource["id"], revision=digest,
                            format=provider.native_format, path=destination, manifest_sha256=digest,
                            status="READY", size_bytes=len(data), source_uri=f"collection:{provider_key}:{session_id}",
                            metadata={"native_raw_preserved": True, "storage_location": "cluster", "capture_summary": summary,
                                      "training_blocker": summary["training_compatibility"]})
                    connection.execute("INSERT INTO local_captures VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (digest, provider_key, session_id, f"{digest}.jsonl", len(data), canonical_json(summary), version["id"], utc_now()))
                    row = connection.execute("SELECT * FROM local_captures WHERE sha256=?", (digest,)).fetchone()
                    self._record_location(connection, row, destination)
                    imported = True
            return {"capture": self._public(row), "imported": imported}

    def _row(self, digest):
        with self.database.connection() as connection:
            row = connection.execute("SELECT * FROM local_captures WHERE sha256=?", (digest,)).fetchone()
        if row is None:
            raise KeyError("Capture not found")
        return row

    def file(self, digest):
        row = self._row(digest)
        if not self._location(row):
            raise ValueError("This recording is awaiting verified migration to sky2. Its saved original has not been removed.")
        return self.storage.artifact(digest, row["size_bytes"])

    def read(self, digest):
        row = self._row(digest)
        data = self.file(digest).read_bytes()
        if len(data) != row["size_bytes"] or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Original recording size or checksum changed on sky2")
        return MemoryCapture(data)

    def stage(self, digest, run_id, gateway):
        row = self._row(digest)
        if not self._location(row):
            raise ValueError("Migrate the original recording to sky2 before processing")
        return self.storage.stage(digest, row["size_bytes"], run_id, gateway)

    def migrate_capture(self, digest):
        """Verify and publish one exact legacy file; deletion remains a separate step."""
        row = self._row(digest)
        if row["filename"] != f"{digest}.jsonl":
            raise ValueError("Legacy recording filename does not match its registered identity")
        path = self.root / row["filename"]
        if path.is_symlink() or not path.is_file():
            if self._location(row):
                self.storage.verify(digest, row["size_bytes"])
                return self._public(row)
            raise ValueError("The legacy recording is missing; no local file was removed")
        if path.stat().st_size != row["size_bytes"]:
            raise ValueError("Legacy recording size does not match its registration")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Legacy recording checksum does not match its registration")
        destination = self.storage.publish(data, digest)
        with self.database.transaction() as connection:
            self._record_location(connection, row, destination)
        return self._public(row)
