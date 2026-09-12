"""Use the shared deletion dialog for prepared data and its exact dependencies."""
from .database import content_sha256


def preview(service, workspace, kind, identifier):
    if kind == "local-copy":
        return local_copy_preview(service, workspace, identifier)
    job_id = identifier if kind == "prepared" else None
    resource_id = service.get(identifier)["resource_id"] if job_id else identifier
    resource = service.database.get_data_resource(resource_id)
    if resource is None:
        raise KeyError("Dataset not found")
    versions = [v for v in resource.get("versions", []) if not job_id or v["id"] == service.get(job_id).get("version_id")]
    blockers = []
    for version in versions:
        for use in service.database.data_version_usage(version["manifest_sha256"]):
            visible = workspace.owns("experiments", use["experiment_id"])
            blocker = dict(kind="experiment", id=use["experiment_id"] if visible else None,
                           label=use["name"] if visible else "Another workspace's experiment",
                           reason="Delete this experiment and its runs first")
            if blocker not in blockers:
                blockers.append(blocker)
    try:
        graph = service.database.delete_prepared_dataset(resource_id, None, identifier=job_id, preview=True)
    except ValueError as error:
        graph = None
        if not blockers:
            blockers.append(dict(kind="dataset", id=None, label=resource["name"], reason=str(error)))
    locations = graph["locations"] if graph else []
    managed = (resource["provider"], resource["namespace"]) == ("collection", "datasets")
    paths = {l["path"] for l in locations if managed and l["version_id"] in {v["id"] for v in versions if v["format"] != "skynet.episodes/v1"}}
    from .policy_exports import WORK_ROOT
    for job in (graph or {}).get("jobs", []):
        paths.add(str(service.root / job["id"]))
        if job.get("version_id") and job.get("target")=="cluster":
            paths.add(f"{WORK_ROOT}/jobs/runs/{job['id']}")
    plan = dict(label=resource.get("metadata", {}).get("display_name") or resource["name"],
                blockers=blockers, counts={"prepared_results": sum(v["format"] != "skynet.episodes/v1" for v in versions)},
                files=[dict(path=path, size_bytes=None, exists=None) for path in sorted(paths)],
                notices=["Original recordings are kept. Prepared files and their registration history are removed." if managed else
                         "Dataset registration and its completed import history are removed. Externally registered source files are kept."],
                retry=any(j.get("state") == "DELETE_FAILED" for j in (graph or {}).get("jobs", [])))
    plan["token"] = content_sha256(dict(kind=kind, identifier=identifier, plan=plan,
                                      versions=[(v["id"],v["manifest_sha256"]) for v in versions],
                                      jobs=[(j["id"],j.get("state"),j.get("version_id")) for j in (graph or {}).get("jobs", [])],
                                      imports=[(i["id"], i["state"]) for i in (graph or {}).get("imports", [])]))
    return plan


def delete(service, workspace, kind, identifier, token):
    def validate():
        plan = preview(service, workspace, kind, identifier)
        if plan["token"] != token or plan["blockers"]:
            raise ValueError("Dataset or dependencies changed. Review deletion again.")
    if kind == "local-copy":
        return service.remove_local_copy(identifier, validate=validate)
    with service.lock:
        validate()
        resource_id = service.get(identifier)["resource_id"] if kind == "prepared" else identifier
        # The existing deletion transaction rechecks references before any file is removed.
        return service.delete_dataset(resource_id, identifier if kind == "prepared" else None)


def local_copy_preview(service, workspace, identifier):
    """Keep the same verified-copy and dependency requirements as the deletion itself."""
    job = service.get(identifier)
    version = service.database.get_data_resource_version(job.get("version_id", ""))
    if not version:
        raise ValueError("This preparation has no registered files")
    resource = service.database.get_data_resource(job["resource_id"])
    blockers = []
    if identifier in service.active or job["state"] != "READY":
        blockers.append(dict(kind="prepared", id=identifier, label=resource["name"], reason="Wait for preparation to finish"))
    for use in service.database.data_version_usage(version["manifest_sha256"]):
        visible = workspace.owns("experiments", use["experiment_id"])
        blockers.append(dict(kind="experiment", id=use["experiment_id"] if visible else None,
                             label=use["name"] if visible else "Another workspace's experiment",
                             reason="Delete the experiment and its runs first; their saved paths must remain available"))
    cluster = [item for item in version.get("locations", []) if item["kind"] == "cluster" and item["status"] == "AVAILABLE"
               and item["manifest_sha256"] == version["manifest_sha256"]]
    if not cluster:
        blockers.append(dict(kind="prepared", id=identifier, label=resource["name"], reason="Transfer and verify a cluster copy first"))
    directory = service.root / identifier
    # Only converted local data is removed, never metadata or source recordings.
    files = [dict(path=str(path), exists=path.exists(), size_bytes=path.stat().st_size if path.is_file() else None)
             for path in (directory / "output", directory / "dataset.zip")]
    plan = dict(label=resource.get("metadata", {}).get("display_name") or resource["name"],
                counts={"local_copies": 1}, blockers=blockers, files=files,
                notices=["Only this computer's prepared files are removed. Original recordings, dataset registration, and the verified cluster copy are kept. The cluster copy is checked again before deletion."], retry=False)
    plan["token"] = content_sha256(dict(plan=plan, state=job["state"], version=version["manifest_sha256"],
                                        locations=version.get("locations"), local_removed=job.get("local_removed")))
    return plan
