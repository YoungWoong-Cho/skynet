"""Installation-owned defaults and persistent opt-outs from automatic seeding."""

import json


def suite_key(environment, name):
    return json.dumps([environment, name], separators=(",", ":"))


def suppress_defaults(connection, kind, records):
    keys = {row.get("seed_key") for row in records} if kind == "adapter" else {
        suite_key(row["evaluator_adapter"], row["name"]) for row in records
    }
    for key in keys - {None, ""}:
        connection.execute(
            "INSERT INTO registry_exclusions(kind,seed_key) VALUES (?,?) ON CONFLICT DO NOTHING",
            (kind, key),
        )
