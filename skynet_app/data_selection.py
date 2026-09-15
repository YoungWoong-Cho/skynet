"""Resolve registered files directly into an experiment's immutable input snapshot.

The serialized envelope remains compatible with existing experiment revisions.
New selections do not create a separate data_bundles row.
"""

from .database import content_sha256


def snapshot(database, selections):
    if not isinstance(selections, list) or not selections:
        raise ValueError("Choose at least one registered dataset")
    assignments, seen, names = [], set(), []
    with database.connection() as connection:
        for item in selections:
            if not isinstance(item, dict) or not item.get("version_id"):
                raise ValueError("Each input must identify a registered dataset or file")
            role, position = str(item.get("role", "training_data")), item.get("position", 0)
            if type(position) is not int or position < 0 or (role, position) in seen:
                raise ValueError("Each dataset input must have a unique role and position")
            seen.add((role, position))
            version_id = str(item["version_id"])
            row = connection.execute(
                "SELECT r.* FROM data_resources r JOIN data_resource_versions v "
                "ON v.resource_id=r.id WHERE v.id=?", (version_id,),
            ).fetchone()
            if row is None or row["archived_at"]:
                raise ValueError("The selected dataset is missing or archived")
            if role == "training_data" and row["category"] != "dataset":
                raise ValueError("Choose a dataset for training data; files belong to other inputs")
            assignment = database._bundle_manifest_assignment(connection, {
                "role": role, "position": position, "version_id": version_id,
                "config": {"location_id": item.get("location_id")},
            })
            if assignment["version"]["format"] == "skynet.episodes/v1":
                raise ValueError("Choose a prepared dataset, not its original recording reference")
            assignment["version"]["metadata"] = {
                **assignment["version"].get("metadata", {}),
                "registered_version_id": version_id,
            }
            assignments.append(assignment)
            names.append(row["name"])
    return _manifest(assignments, names)


def _manifest(assignments, names):
    manifest = {
        "schema_version": "skynet.data-bundle/v1", "name": ", ".join(names),
        "version": "experiment-inputs", "metadata": {"direct_selection": True},
        "assignments": sorted(assignments, key=lambda a: (a["role"], a["position"])),
    }
    digest = content_sha256(manifest)
    return {"id": "selection:" + digest, **manifest, "manifest_sha256": digest}


def choices(database):
    """List actual prepared results, including incompatible formats for explanation."""
    result = []
    for resource in database.list_data_resources(category="dataset", include_versions=True):
        for version in resource["versions"]:
            if version["format"] == "skynet.episodes/v1":
                continue
            locations = [loc for loc in version.get("locations", [])
                         if loc["kind"] == "cluster" and loc["status"] == "AVAILABLE"
                         and loc["manifest_sha256"] == version["manifest_sha256"]]
            if not locations and version["status"] != "READY":
                continue
            # One deterministic cluster location per result; a saved experiment keeps its receipt.
            location = min(locations, key=lambda loc: str(loc["id"])) if locations else None
            selection = {"version_id": version["id"], "location_id": location["id"] if location else None}
            assignment = database.bundle_manifest_assignment({
                "role": "training_data", "position": 0, "version_id": version["id"],
                "config": {"location_id": selection["location_id"]},
            }, resource, version)
            assignment["version"]["metadata"] = {
                **assignment["version"]["metadata"], "registered_version_id": version["id"],
            }
            frozen = _manifest([assignment], [resource["name"]])
            result.append({
                **frozen, "id": version["id"], "selection": selection,
                "name": resource.get("metadata", {}).get("display_name") or resource["name"],
                "format": version["format"], "created_at": version["created_at"],
                "resource_id": resource["id"],
            })
    return result


def describe(spec):
    """Human-readable run list context without loading full manifests in the browser."""
    bundle = spec.get("data", {}).get("bundle") or {}
    result = []
    for assignment in bundle.get("assignments", []):
        if assignment.get("role") != "training_data":
            continue
        version = assignment.get("version") or {}
        metadata = version.get("metadata") or {}
        episodes = metadata.get("num_episodes", metadata.get("episodes"))
        result.append(dict(name=metadata.get("display_name") or bundle.get("name") or assignment.get("resource", {}).get("name"),
                           format=version.get("format"), episodes=len(episodes) if isinstance(episodes, list) else episodes,
                           resource_id=assignment.get("resource", {}).get("id"),
                           version_id=metadata.get("registered_version_id"),
                           manifest_sha256=version.get("manifest_sha256")))
    return result


def attach_links(database, runs):
    """Resolve legacy snapshots by content identity without changing their receipts."""
    inputs = [item for run in runs for item in run.get("training_data", [])]
    digests = list({item["manifest_sha256"] for item in inputs if item.get("manifest_sha256")})
    found = {}
    with database.connection() as connection:
        for start in range(0, len(digests), 400):
            batch = digests[start:start + 400]
            placeholders = ",".join("?" for _ in batch)
            for row in connection.execute(f"SELECT id, resource_id, manifest_sha256 FROM data_resource_versions WHERE manifest_sha256 IN ({placeholders}) ORDER BY created_at, id", batch).fetchall():
                found.setdefault(row["manifest_sha256"], row)
    for item in inputs:
        registered = found.get(item.get("manifest_sha256"))
        if registered:
            item.update(resource_id=registered["resource_id"], version_id=registered["id"])
