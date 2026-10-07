from factories import make_run_chain
from skynet_app.database import Database
from skynet_app.pipeline_api import _attach_run_progress_summaries


QA_RUN = dict(
    project_name="qa", experiment_name="qa", requested_spec={"source": {"revision": "a" * 40}},
    resolved_spec={"resources": {"gpu": {"count": 1, "type": "l40s"}}},
    seed=0, adapter_name="groot", run_directory="/tmp/qa",
)


def test_run_summary_includes_latest_training_attempt_and_resources(tmp_path):
    database = Database(tmp_path / "test.db")
    run = make_run_chain(database, **QA_RUN).run
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
    database = Database(tmp_path / "test.db")
    run = make_run_chain(database, **QA_RUN).run
    evaluation = database.create_evaluation(
        run["id"], evaluator_adapter="groot", evaluator_version="1",
        suite_name="tabletop", suite_version="1", tasks=["task"], seeds=[0], episodes_per_task=1,
    )
    aggregate = [{"metric": "success_rate", "mean": 0, "task": None}]
    database.create_artifact(run["id"], evaluation_id=evaluation["id"], artifact_type="EVALUATION_RESULT", path="/tmp/result.json", metadata={"aggregate": aggregate})
    assert database.list_evaluations()[0]["aggregate"] == aggregate


def test_experiment_list_exposes_pinned_revision(tmp_path):
    database = Database(tmp_path / "test.db")
    make_run_chain(database, **QA_RUN)
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
    database = Database(tmp_path / "test.db")
    run = make_run_chain(database, **QA_RUN).run
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
    assets = ('app.js', 'workspace-navigation.js', 'live-xr.js', 'live-conversion.js', 'styles.css')
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


def test_homepage_offers_the_configured_gateways_queues_gpus_and_defaults(monkeypatch):
    import re
    from skynet_app import main
    from skynet_app.cluster_config import CLUSTER
    from skynet_app import page_markup
    from skynet_app.page_markup import cluster_markup
    def options(page, select):
        body = re.search(rf'<select id="{select}"[^>]*>(.*?)</select>', page, re.S).group(1)
        return re.findall(r'<option value="([^"]*)"[^>]*>([^<]*)</option>', body)
    source = (main.STATIC_ROOT / 'index.html').read_text(encoding='utf-8')
    page = main.index().body.decode()
    for placeholder in cluster_markup():
        assert placeholder in source, placeholder
        assert placeholder not in page, placeholder
    hosts = list(CLUSTER.gateways)
    expected = [('auto', 'Auto: ' + ', then '.join(hosts)), *((host, f'Prefer {host}') for host in hosts)]
    for select in ('gateway', 'collection-gateway', 'data-import-gateway'):
        assert options(source, select) == []
        assert options(page, select) == expected
    # Queue and GPU choices are exactly the configured ones; the forms carry no copies.
    assert [value for value, _ in options(page, 'resource-policy')] == ['auto', *CLUSTER.queues]
    assert [value for value, _ in options(page, 'data-import-queue')] == list(CLUSTER.queues)
    assert [value for value, _ in options(page, 'experiment-gpu-type')] == list(CLUSTER.gpu_aliases)
    for name, queue in CLUSTER.queues.items():
        assert queue.partition in dict(options(page, 'resource-policy'))[name]
    default = CLUSTER.queue(CLUSTER.defaults.queue_policy)
    assert re.search(r'<option value="%s" selected' % CLUSTER.defaults.gpu_type, page)
    assert re.search(r'id="collection-account"\s+value="%s"' % re.escape(default.account), page)
    assert re.search(r'id="collection-partition"\s+value="%s"' % re.escape(default.partition), page)
    headers = re.findall(r'<th data-gpu-column="([^"]+)">', page)
    assert headers == list(CLUSTER.dashboard.gpu_usage_columns)
    # Placeholder rows span the header app.js reads; the page states no column counts.
    assert 'colspan' not in page
    # Request-model bounds and server registries reach the page instead of retyped copies.
    import html
    import json
    import pytest
    from pydantic import ValidationError
    from skynet_app import retargeting
    from skynet_app.experiments import DEFAULT_EVALUATION_EPISODES, MAX_EVALUATION_EPISODES, CheckpointPolicy, SweepSpec
    from skynet_app.hands_api import Pose
    from skynet_app.live_xr_archive import TERMINAL_STATES
    from skynet_app.policy_exports_api import ExportRequest
    from skynet_app.tracking import TRACKING_PROVIDERS
    from skynet_app.workspaces import EmailRequest
    def attributes(element_id):
        tag = re.search(rf'<\w+\b[^>]*\bid="{element_id}"[^>]*>', page).group(0)
        return {name: html.unescape(value) for name, value in re.findall(r'([\w-]+)="([^"]*)"', tag)}
    def export_split(**split):
        return ExportRequest(session_id='s', adapter_id='a', adapter_version_id='v', name='n', **split)
    for element_id, attribute, accepts in (
        ('checkpoint-max-attempts', 'max', lambda n: CheckpointPolicy(max_attempts=n)),
        ('hand-pose-name', 'maxlength', lambda n: Pose(name='p' * n, revision='r', joints={})),
        ('workspace-email', 'maxlength', lambda n: EmailRequest(email='e' * n)),
        ('preparation-validation', 'max', lambda n: export_split(validation_percent=n)),
        ('preparation-seed', 'max', lambda n: export_split(seed=n)),
    ):
        limit = int(attributes(element_id)[attribute])
        accepts(limit)
        with pytest.raises(ValidationError):
            accepts(limit + 1)
    split = export_split()
    assert attributes('preparation-validation')['value'] == str(split.validation_percent)
    assert attributes('preparation-seed')['value'] == str(split.seed)
    assert json.loads(attributes('collection-view-live')['data-terminal-states']) == sorted(TERMINAL_STATES)
    sweep = SweepSpec()
    assert json.loads(attributes('sweep-definition')['data-sweep-defaults']) == {
        name: getattr(sweep, name) for name in ('seeds', 'max_parallel', 'confirmation_threshold')}
    episodes = attributes('evaluation-episodes')
    assert (episodes['value'], episodes['data-max-episodes']) == (str(DEFAULT_EVALUATION_EPISODES), str(MAX_EVALUATION_EPISODES))
    assert attributes('live-xr-retargeter')['data-default-retargeter'] == retargeting.DEFAULT
    assert options(page, 'live-xr-retargeter') == [
        (retargeting.DEFAULT, next(method['name'] for method in retargeting.METHODS if method['key'] == retargeting.DEFAULT))]
    assert json.loads(attributes('settings-view-connections')['data-tracking-providers']) == TRACKING_PROVIDERS
    # The list follows the configuration, and configured names are escaped.
    def configured(**changes):
        monkeypatch.setattr(page_markup, 'CLUSTER', CLUSTER.model_copy(update=changes))
        return main.index().body.decode()
    escaped = 'login&lt;b&gt;&amp;&quot;c&quot;'
    assert options(configured(gateways=['login-a', 'login<b>&"c"']), 'gateway') == [
        ('auto', f'Auto: login-a, then {escaped}'), ('login-a', 'Prefer login-a'), (escaped, f'Prefer {escaped}')]
    assert options(configured(gateways=['login-a']), 'gateway') == [('auto', 'Auto: login-a'), ('login-a', 'Prefer login-a')]
    one_gpu = configured(gpu_aliases={'any': None, 'h100': 'h100'})
    assert options(one_gpu, 'experiment-gpu-type') == [('any', 'Any compatible'), ('h100', 'H100')]
