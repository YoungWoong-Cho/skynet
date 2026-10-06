"""The dataset catalog identifies individual published output versions.

Presentation is separate from immutable data manifests. Internal source resource
rows support imports and provenance, but never determine dataset membership.
"""

from .database import utc_now


def recording_ids(resource, version, source_version=None):
    """Use this output's sources only, falling back to the dataset's own recording."""
    parent = resource.get("metadata") or {}
    owner = parent.get("recording_session_id")
    session = parent.get("session_id")
    metadata = version.get("metadata") or {}
    sources = metadata.get("sources")
    if not isinstance(sources, list):
        sources = (source_version or {}).get("metadata", {}).get("sources")
    if not isinstance(sources, list):
        sources = [episode.get("source", {}) for episode in metadata.get("episodes", [])
                   if isinstance(episode, dict)] if isinstance(metadata.get("episodes"), list) else []
    result = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        value = source.get("recording_session_id") or source.get("session_id")
        if isinstance(value, str) and value:
            result.add(value)
    # Never add sibling versions' sessions to this result.
    if owner and not result:
        result.add(owner)
    elif not result:
        value = metadata.get("recording_session_id") or metadata.get("session_id")
        if not value and (resource.get("provider") == "collection" or parent.get("managed_dataset")):
            value = session
        if isinstance(value, str) and value:
            result.add(value)
    return sorted(result)


def list_datasets(database, *, include_archived=False, version_id=None, include_presets=True):
    clauses = ["r.category='dataset'", "v.format<>'skynet.episodes/v1'", "t.version_id IS NULL"]
    parameters = []
    if not include_archived:
        clauses.append("p.archived_at IS NULL")
    if version_id is not None:
        clauses.append("v.id=?")
        parameters.append(version_id)
    with database.read_snapshot() as connection:
        rows = connection.execute(
            "SELECT v.*, p.display_name, p.description, p.archived_at, p.updated_at "
            "FROM data_resource_versions v JOIN data_resources r ON r.id=v.resource_id "
            "JOIN data_dataset_presentations p ON p.version_id=v.id "
            "LEFT JOIN data_version_retirements t ON t.version_id=v.id "
            "WHERE " + " AND ".join(clauses) + " ORDER BY v.created_at DESC, v.id DESC", parameters,
        ).fetchall()
        versions = database._data_version_payloads(connection, rows, include_resource=True)
        source_ids = list({version["metadata"].get("source_version_id") for version in versions
                           if version["metadata"].get("source_version_id")})
        source_rows = connection.execute(
            "SELECT * FROM data_resource_versions WHERE id=ANY(?)", (source_ids,),
        ).fetchall() if source_ids else []
        sources = {row["id"]: database._decode(row) for row in source_rows}
        for source in sources.values():
            source["metadata"] = source.pop("metadata_json", {})
        for version in versions:
            resource = version.pop("resource")
            version.update({key: resource[key] for key in
                            ("provider", "namespace", "source_key", "category", "kind")})
            version["recording_ids"] = recording_ids(
                resource, version, sources.get(version["metadata"].get("source_version_id")))
    links = database.dataset_preset_links() if include_presets else []
    by_version = {}
    for link in links:
        by_version.setdefault(link["version_id"], []).append(link)
    for version in versions:
        version["experiment_presets"] = by_version.get(version["id"], [])
    return versions


def get_dataset(database, version_id):
    return next(iter(list_datasets(database, include_archived=True, version_id=version_id)), None)


def update_dataset(database, version_id, *, display_name=None, description=None, archived=None):
    fields = {"updated_at": utc_now()}
    if display_name is not None:
        if not isinstance(display_name, str) or not display_name.strip():
            raise ValueError("Display name must not be empty")
        fields["display_name"] = display_name.strip()
    if description is not None:
        fields["description"] = description
    if archived is not None:
        fields["archived_at"] = utc_now() if archived else None
    with database.transaction() as connection:
        row = connection.execute(
            "SELECT p.version_id FROM data_dataset_presentations p "
            "LEFT JOIN data_version_retirements t ON t.version_id=p.version_id "
            "WHERE p.version_id=? AND t.version_id IS NULL", (version_id,),
        ).fetchone()
        if row is None:
            raise KeyError("Dataset not found")
        connection.execute(
            "UPDATE data_dataset_presentations SET " + ", ".join(f"{key}=?" for key in fields)
            + " WHERE version_id=?", (*fields.values(), version_id),
        )
    return get_dataset(database, version_id)
