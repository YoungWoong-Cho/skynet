from skynet_app.database import Database
from skynet_app.pipeline_api import _attach_run_progress_summaries


def make_run(tmp_path):
    database = Database(tmp_path / "test.db")
    project = database.create_project("qa")
    experiment = database.create_experiment(
        project_id=project["id"], name="qa",
        requested_spec={"source": {"revision": "a" * 40}},
    )
    variant = database.create_variant(
        experiment["latest_revision"]["id"], name="one",
        parameters={}, resolved_spec={"resources": {"gpu": {"count": 1, "type": "l40s"}}},
    )
    run = database.create_run(variant["id"], seed=0, adapter_name="groot", adapter_version="1", run_directory="/tmp/qa")
    return database, run


def test_run_summary_includes_latest_training_attempt_and_resources(tmp_path):
    database, run = make_run(tmp_path)
    stage = database.create_stage(run["id"], stage_type="TRAIN", name="train")
    first = database.create_job_attempt(stage["id"], slurm_job_id="10")
    database.update_job_attempt(first["id"], status="FAILED")
    latest = database.create_job_attempt(stage["id"], slurm_job_id="11", gpu_count=1)
    # An evaluation attempt must not replace the training attempt in the ledger.
    evaluation_stage = database.create_stage(run["id"], stage_type="EVAL", name="eval")
    database.create_job_attempt(evaluation_stage["id"], slurm_job_id="12")
    rows = database.list_runs()
    _attach_run_progress_summaries(database, rows)
    assert rows[0]["attempt_count"] == 2
    assert rows[0]["latest_attempt"]["id"] == latest["id"]
    assert rows[0]["latest_attempt"]["slurm_job_id"] == "11"
    assert rows[0]["resources"]["gpu"]["count"] == 1


def test_evaluation_list_contains_result_before_opening_details(tmp_path):
    database, run = make_run(tmp_path)
    evaluation = database.create_evaluation(
        run["id"], evaluator_adapter="groot", evaluator_version="1",
        suite_name="tabletop", suite_version="1", tasks=["task"], seeds=[0], episodes_per_task=1,
    )
    aggregate = [{"metric": "success_rate", "mean": 0, "task": None}]
    database.create_artifact(run["id"], evaluation_id=evaluation["id"], artifact_type="EVALUATION_RESULT", path="/tmp/result.json", metadata={"aggregate": aggregate})
    assert database.list_evaluations()[0]["aggregate"] == aggregate


def test_experiment_list_exposes_pinned_revision(tmp_path):
    database, _ = make_run(tmp_path)
    assert database.list_experiments()[0]["git_revision"] == "a" * 40


def test_bundle_selection_is_consumed_or_rejected_explicitly():
    import pytest
    from skynet_app.adapters import builtin_adapter_manifests
    from skynet_app.pipeline_api import PipelineService
    manifests = {manifest.slug: manifest for manifest in builtin_adapter_manifests()}
    def document(format='groot-lerobot-v2', path=''):
        return {
            'data': {'bundle': {'assignments': [{'role': 'training_data', 'position': 0,
                'version': {'format': format, 'path': '/data/gr00t', 'status': 'READY'}}]}},
            'native': {'config': {'dataset_path': path}},
        }
    resolved = PipelineService._apply_manifest_data_bindings(document(), manifests['groot'])
    assert resolved['native']['config']['dataset_path'] == '/data/gr00t'
    assert PipelineService._apply_manifest_data_bindings(resolved, manifests['groot']) == resolved
    with pytest.raises(ValueError, match='conflicts'):
        PipelineService._apply_manifest_data_bindings(document(path='/different'), manifests['groot'])
    with pytest.raises(ValueError, match='incompatible'):
        PipelineService._apply_manifest_data_bindings(document(format='isaac-usd'), manifests['groot'])
    with pytest.raises(ValueError, match='incompatible'):
        PipelineService._apply_manifest_data_bindings(document(), manifests['openpi'])
    assert PipelineService._apply_manifest_data_bindings(document(format='lerobot-v2.0'), manifests['openpi'])['native']['config']['dataset_path'] == '/data/gr00t'
    unavailable = document()
    unavailable['data']['bundle']['assignments'][0]['version']['status'] = 'LOCAL'
    with pytest.raises(ValueError, match='not ready on the cluster'):
        PipelineService._apply_manifest_data_bindings(unavailable, manifests['groot'])
    workstation = document()
    workstation['data']['bundle']['assignments'][0]['version']['metadata'] = {'storage_location':'workstation'}
    with pytest.raises(ValueError, match='collection workstation'):
        PipelineService._apply_manifest_data_bindings(workstation, manifests['groot'])
    extra = document()
    import copy
    another = copy.deepcopy(extra['data']['bundle']['assignments'][0])
    another['position'] = 1
    extra['data']['bundle']['assignments'].append(another)
    with pytest.raises(ValueError, match='cannot consume'):
        PipelineService._apply_manifest_data_bindings(extra, manifests['groot'])



def test_evaluation_readiness_rejects_busy_run_before_expensive_probe(tmp_path):
    from types import SimpleNamespace
    from skynet_app.pipeline_api import PipelineService
    database, run = make_run(tmp_path)
    database.create_checkpoint(run['id'], path='/tmp/checkpoint', checkpoint_type='INFERENCE',
        is_selected_for_inference=True)
    database.create_stage(run['id'], stage_type='EVALUATE', name='existing evaluation', status='RUNNING')
    service = PipelineService(database, SimpleNamespace(hosts=('sky1',)))
    service._resolve_evaluation_suite_selection = lambda *args: (_ for _ in ()).throw(AssertionError('Busy target should stop before suite/runtime checks'))
    validation = service.validate_evaluation_target(run['id'], None, suite_id='any-suite')
    assert validation['run_valid'] and validation['checkpoint_valid']
    assert not validation['valid']
    assert not validation['plan_valid']
    assert 'already has active' in validation['plan_message']


def test_homepage_revalidates_and_versions_changed_assets(tmp_path, monkeypatch):
    from skynet_app import main
    monkeypatch.setattr(main, 'STATIC_ROOT', tmp_path)
    assets = ('app.js', 'collection-ui.js', 'live-xr.js', 'live-conversion.js', 'styles.css')
    (tmp_path / 'index.html').write_text(''.join(f'<script src="/static/{name}?v=old"></script>' for name in assets))
    for name in assets:
        (tmp_path / name).write_text('first')
    first = main.index()
    assert first.headers['cache-control'] == 'no-cache'
    assert b'?v=old' not in first.body
    for name in assets:
        (tmp_path / name).write_text('updated interface')
        second = main.index()
        assert first.body != second.body, name
        assert f'/static/{name}?v='.encode() in second.body
        first = second
