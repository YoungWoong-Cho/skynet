"""Resolve registered files directly into an experiment's immutable input snapshot.

The serialized envelope remains compatible with existing experiment revisions.
New selections do not create a separate data_bundles row.
"""

from .database import content_sha256


def assert_available(database, *execution_inputs):
    """Reject obsolete pinned inputs before a launch can change run state."""
    from .data_version_retirement import references

    with database.connection() as connection:
        versions = connection.execute(
            "SELECT v.id,v.manifest_sha256,v.path FROM data_resource_versions v "
            "JOIN data_version_retirements r ON r.version_id=v.id"
        ).fetchall()
        if not versions:
            return
        identifiers = [version["id"] for version in versions]
        needles = {value for version in versions for value in
                   (version["id"], version["manifest_sha256"], version["path"])}
        needles.update(row["path"] for row in connection.execute(
            "SELECT path FROM data_locations WHERE version_id=ANY(?)", (identifiers,)).fetchall())
        for row in connection.execute(
            "SELECT b.id,b.manifest_sha256 FROM data_bundles b "
            "JOIN data_bundle_assignments a ON a.bundle_id=b.id WHERE a.version_id=ANY(?)",
            (identifiers,),
        ).fetchall():
            needles.update((row["id"], row["manifest_sha256"]))
        if any(references(value, needles) for value in execution_inputs):
            raise ValueError(
                "This execution uses retired converted data. Use the experiment's new revision "
                "with its replacement dataset. Historical checkpoints remain available for inspection."
            )


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
            if connection.execute(
                "SELECT 1 FROM data_version_retirements WHERE version_id=?", (version_id,)
            ).fetchone():
                raise ValueError("This converted dataset has been retired; select its replacement")
            row = connection.execute(
                "SELECT r.*, p.archived_at AS dataset_archived_at, p.display_name AS dataset_display_name "
                "FROM data_resources r JOIN data_resource_versions v ON v.resource_id=r.id "
                "LEFT JOIN data_dataset_presentations p ON p.version_id=v.id WHERE v.id=?", (version_id,),
            ).fetchone()
            if row is None or (row["dataset_archived_at"] if row["category"] == "dataset" else row["archived_at"]):
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
                "display_name": row["dataset_display_name"] or row["display_name"],
            }
            assignments.append(assignment)
            names.append(row["dataset_display_name"] or row["display_name"])
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
    for version in database.list_datasets(include_presets=False):
        locations = [loc for loc in version.get("locations", [])
                     if loc["kind"] == "cluster" and loc["status"] == "AVAILABLE"
                     and loc["manifest_sha256"] == version["manifest_sha256"]]
        if not locations and version["status"] != "READY":
            continue
        # One deterministic cluster location per result; saved experiments retain theirs.
        location = min(locations, key=lambda loc: str(loc["id"])) if locations else None
        selection = {"version_id": version["id"], "location_id": location["id"] if location else None}
        assignment = database.bundle_manifest_assignment({
            "role": "training_data", "position": 0, "version_id": version["id"],
            "config": {"location_id": selection["location_id"]},
        }, version, version)
        assignment["version"]["metadata"] = {
            **assignment["version"]["metadata"], "registered_version_id": version["id"],
            "display_name": version["display_name"],
        }
        frozen = _manifest([assignment], [version["display_name"]])
        result.append({
            **frozen, "id": version["id"], "selection": selection,
            "name": version["display_name"],
            "format": version["format"], "created_at": version["created_at"],
            "resource_id": version["resource_id"],
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
