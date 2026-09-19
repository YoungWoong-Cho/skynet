"""Use the shared deletion dialog for prepared data and its exact dependencies."""
from .database import content_sha256


def preview(service, workspace, kind, identifier):
    if kind not in {"dataset", "prepared"}:
        raise ValueError("Unknown prepared-data deletion kind")
    job_id = identifier if kind == "prepared" else None
    job = service.get(identifier) if job_id else None
    dataset = service.database.get_dataset(identifier) if not job_id else None
    if not job and not dataset:
        raise KeyError("Dataset not found")
    resource_id = job["resource_id"] if job else dataset["resource_id"]
    resource = service.database.get_data_resource(resource_id)
    if resource is None:
        raise KeyError("Dataset not found")
    version_id = job.get("version_id") if job else identifier
    versions = [v for v in resource.get("versions", []) if v["id"] == version_id]
    label = dataset["display_name"] if dataset else job.get("name", resource["display_name"])
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
        graph = service.database.delete_prepared_dataset(resource_id, None, identifier=job_id, version_id=None if job_id else identifier, preview=True)
    except ValueError as error:
        graph = None
        if not blockers:
            blockers.append(dict(kind="dataset", id=None, label=label, reason=str(error)))
    locations = graph["locations"] if graph else []
    managed = (resource["provider"], resource["namespace"]) == ("collection", "datasets")
    paths = {l["path"] for l in locations if managed and l["version_id"] in {v["id"] for v in versions if v["format"] != "skynet.episodes/v1"}}
    from .policy_exports import WORK_ROOT
    for job in (graph or {}).get("jobs", []):
        paths.add(str(service.root / job["id"]))
        if job.get("version_id") and job.get("target")=="cluster":
            paths.add(f"{WORK_ROOT}/jobs/runs/{job['id']}")
    plan = dict(label=label,
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
    with service.lock:
        validate()
        resource_id = (service.get(identifier)["resource_id"] if kind == "prepared"
                       else service.database.get_dataset(identifier)["resource_id"])
        # Recheck references inside the deletion transaction before removing any file.
        return service.delete_dataset(resource_id, identifier if kind == "prepared" else None,
                                      version_id=identifier if kind == "dataset" else None)
