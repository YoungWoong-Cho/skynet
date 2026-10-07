"""Record chains that many tests need as scaffolding rather than as the subject under test."""
from types import SimpleNamespace


def make_run_chain(
    db,
    *,
    project_name="project",
    project_description="",
    experiment_name="experiment",
    requested_spec=None,
    variant_name="one",
    parameters=None,
    resolved_spec=None,
    seed=1,
    adapter_name="generic",
    adapter_version="1",
    run_directory="/fixture/run",
    **run_fields,
):
    """Create project -> experiment -> variant -> run in ``db`` and return all four records.

    ``project_name=None`` files the experiment under no project (``project`` is then None).
    ``resolved_spec`` defaults to ``requested_spec``; ``run_fields`` such as ``status`` go to
    ``create_run`` unchanged, so its own defaults apply when they are omitted.
    """
    requested_spec = {} if requested_spec is None else requested_spec
    project = None if project_name is None else db.create_project(project_name, project_description)
    experiment = db.create_experiment(
        project_id=None if project is None else project["id"], name=experiment_name, requested_spec=requested_spec
    )
    variant = db.create_variant(
        experiment["latest_revision"]["id"],
        name=variant_name,
        parameters={} if parameters is None else parameters,
        resolved_spec=requested_spec if resolved_spec is None else resolved_spec,
    )
    run = db.create_run(
        variant["id"],
        seed=seed,
        adapter_name=adapter_name,
        adapter_version=adapter_version,
        run_directory=run_directory,
        **run_fields,
    )
    return SimpleNamespace(project=project, experiment=experiment, variant=variant, run=run)
