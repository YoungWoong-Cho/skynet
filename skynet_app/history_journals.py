"""Remove deleted evaluation references from retained run delivery journals."""

from __future__ import annotations

import hashlib
import json

from .tracking_journal import (
    CHUNK_BYTES,
    decode_payload,
    encode_payload,
)

_DROP = object()


def scrub(value, needles):
    def matches(value):
        return isinstance(value, str) and any(needle in value for needle in needles)

    if isinstance(value, dict):
        # Objects such as {key: evaluation/id/metric, value: 1} are one record.
        if any(matches(item) for item in value.values() if isinstance(item, str)):
            return _DROP
        result = {}
        for key, item in value.items():
            if matches(key):
                continue
            cleaned = scrub(item, needles)
            if cleaned is not _DROP:
                result[key] = cleaned
        return result
    if isinstance(value, list):
        return [
            cleaned for item in value if (cleaned := scrub(item, needles)) is not _DROP
        ]
    return _DROP if matches(value) else value


def cleaned_payload(filename, body, needles):
    if not any(needle.encode() in body for needle in needles):
        return body
    if filename.endswith(".jsonl"):
        events = []
        for line in body.splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            cleaned = scrub(event, needles)
            if cleaned is not _DROP:
                # Artifact deliveries without their URI cannot be replayed.
                if cleaned.get("operation") in {
                    "artifact_link",
                    "log_artifact_link",
                } and not cleaned.get("payload", {}).get("uri"):
                    continue
                events.append(
                    json.dumps(cleaned, sort_keys=True, separators=(",", ":")).encode()
                    + b"\n"
                )
        return b"".join(events)
    cleaned = scrub(json.loads(body), needles)
    return json.dumps(
        {} if cleaned is _DROP else cleaned, sort_keys=True, separators=(",", ":")
    ).encode()


def changes(connection, database, run_id, needles):
    result = []
    for row in connection.execute(
        "SELECT owner_id,filename,payload FROM tracking_journals WHERE run_id=?",
        (run_id,),
    ).fetchall():
        original = bytes(row[2])
        body = decode_payload(database.payload_store, original)
        clean = cleaned_payload(row[1], body, needles)
        if clean == body:
            continue
        item = {
            "owner": row[0],
            "filename": row[1],
            "run_id": run_id,
            "before": hashlib.sha256(original).hexdigest(),
            "after": hashlib.sha256(clean).hexdigest(),
        }
        if database.payload_store:
            # Identify only old chunks that the replacement does not reuse.
            text = clean.decode()
            chunks = (
                [text]
                if row[1] == "tracking-artifact-links.json"
                else [
                    text[start : start + CHUNK_BYTES]
                    for start in range(0, len(text), CHUNK_BYTES)
                ]
            )
            retained = {hashlib.sha256(chunk.encode()).hexdigest() for chunk in chunks}
            old_refs = connection.execute(
                "SELECT r.sha256,p.path,p.size_bytes,r.field_name FROM metadata_payload_refs r JOIN metadata_payloads p ON p.sha256=r.sha256 WHERE r.table_name='tracking_journals' AND r.record_id=?",
                (run_id,),
            ).fetchall()
            retired = []
            for ref in old_refs:
                if ref["sha256"] in retained or not (
                    ref["field_name"] == row[1]
                    or ref["field_name"].startswith(row[1] + ":")
                ):
                    continue
                shared = connection.execute(
                    "SELECT table_name,record_id,field_name FROM metadata_payload_refs WHERE sha256=?",
                    (ref["sha256"],),
                ).fetchall()
                if all(
                    r[0] == "tracking_journals"
                    and r[1] == run_id
                    and (r[2] == row[1] or r[2].startswith(row[1] + ":"))
                    for r in shared
                ):
                    retired.append(
                        {
                            "sha256": ref["sha256"],
                            "path": ref["path"],
                            "size_bytes": ref["size_bytes"],
                        }
                    )
            item["retired"] = retired
        result.append(item)
    return result


def apply(connection, database, plans, needles):
    for plan in plans:
        key = (plan["owner"], plan["run_id"], plan["filename"])
        row = connection.execute(
            "SELECT payload FROM tracking_journals WHERE owner_id=? AND scope=? AND filename=?",
            key,
        ).fetchone()
        original = bytes(row[0])
        if hashlib.sha256(original).hexdigest() != plan["before"]:
            raise ValueError("Tracking journal changed. Review deletion again")
        content = cleaned_payload(
            plan["filename"], decode_payload(database.payload_store, original), needles
        )
        if hashlib.sha256(content).hexdigest() != plan["after"]:
            raise ValueError("Tracking cleanup changed. Review deletion again")
        if database.payload_store:
            if plan["filename"] == "tracking-artifact-links.json":
                content = database.payload_store.put(
                    connection,
                    "tracking_journals",
                    plan["run_id"],
                    plan["filename"],
                    content.decode(),
                ).encode()
            else:
                content = encode_payload(
                    connection,
                    database.payload_store,
                    plan["run_id"],
                    plan["filename"],
                    content,
                    original,
                )
        connection.execute(
            "UPDATE tracking_journals SET payload=? WHERE owner_id=? AND scope=? AND filename=?",
            (content, *key),
        )
