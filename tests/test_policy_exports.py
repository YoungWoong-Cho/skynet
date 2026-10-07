"""Recording preparation orchestration with real CPU shared-stream fixtures."""
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from skynet_app.database import Database, canonical_json
from skynet_app.live_xr_review import LiveReviewService
from skynet_app.policy_exports import PolicyExportService, digest, conversion_failure
from skynet_app.adapters import builtin_adapter_manifests
from skynet_app.adapters.act_manifest import manifest as act_manifest
from skynet_app.observation_contracts import rgb_requirements
from policy_export_offline import prepare as offline_prepare

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops/xr"))
from recording_metadata import joint_layout

def capture(root, index=0, both=False):
    wrists = [f"{axis}_translation_joint" for axis in "xyz"] + [f"{axis}_rotation_joint" for axis in "zyx"]
    names = wrists + ["finger_a", "finger_b"]
    if both:
        names = ["rh_" + n for n in wrists] + ["lh_" + n for n in wrists] + ["rh_finger_a", "rh_finger_b", "lh_finger_a", "lh_finger_b"]
    meta = dict(robot="floating_shadow_bimanual" if both else "floating_shadow_hand", task="test-task", hand="both" if both else "right", source_revision="abc", action_joint_names=names, robot_joint_names=list(reversed(names)), groups=joint_layout(names, "both" if both else "right"), step_dt=1 / 60, color_space="RGB", cameras={k: {"mount": "fixed_scene"} for k in ["scene_front", "scene_left", "scene_right"]})
    actions = np.arange(3 * len(names), dtype="f4").reshape(3, -1) / 100 + index
    states = actions + 10
    rgb = np.zeros((256, 256, 3), dtype="u1")
    rgb[:, :, 0] = np.arange(256, dtype="u1")[:, None]
    rgb[:, :, 1] = 23
    rgb[:, :, 2] = 170
    directory = root / 'recordings/live'
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / f'images-{index:06d}.hdf5'
    with h5py.File(image_path,'w') as file:
        file.attrs.update(schema='skynet.rgb-trajectory/v1', complete=True, metadata=json.dumps(meta))
        file['state'] = states
        file['action'] = actions
        file['timestamps'] = np.arange(3)/60
        file['wall_times'] = 1000+np.arange(3)/60
        for name, offset in [('scene_front',0),('scene_left',2),('scene_right',3)]:
            file['images/'+name] = np.stack([rgb+np.uint8(offset)]*3)
    images = dict(path=str(image_path.relative_to(root)),sha256=digest(image_path),size_bytes=image_path.stat().st_size,steps=3,schema='skynet.rgb-trajectory/v1')
    raw_states = [dict(articulation={"robot": {"joint_position": s[::-1][None]}}) for s in [*states, states[-1] + 1]]
    ep = dict(actions=actions, states=raw_states, num_steps=3, success=True, skynet_images=images)
    path = root / f"recordings/live/episode-{index:06d}.pkl"
    path.write_bytes(pickle.dumps(dict(format="dexverse_trajectory", schema_version=3, num_episodes=1, task=meta["task"], robot_type=meta["robot"], episodes=[ep])))
    return dict(path=str(path.relative_to(root)), sha256=digest(path), images=images), meta, actions, states


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = tmp_path / "remote/output"
    source.mkdir(parents=True)
    receipts = [capture(source, i, both=True)[0] for i in range(2)]
    session = dict(id="session-1", state="STOPPED", root=str(source.parent), gateway="test-host", created_at="2026-09-08", profile=dict(display_name="Test hand", task="test-task", robot="floating_shadow_bimanual"), recordings=[r["path"] for r in receipts], recording_checksums={r["path"]: r["sha256"] for r in receipts}, recording_images={r["path"]: r["images"] for r in receipts})
    live = SimpleNamespace(archive=SimpleNamespace(resolve=lambda job, path: (None, "test", path)), root=ROOT, database=Database(tmp_path / "db.store"), get=lambda _:session,list=lambda **_:[session])
    service = PolicyExportService(LiveReviewService(live,root=tmp_path/'reviews'),root=tmp_path/'exports')
    install_adapters(service)
    monkeypatch.setattr(service,'dispatch',lambda _:None)
    monkeypatch.setattr(service,'prepare',offline_prepare.__get__(service))
    yield service,session,source
    service.stop()


def fixture_recording_manifest():
    """Synthetic service contract: no model training is invoked by these tests.

    Reuse ACT's frozen shared-data reader files, while explicitly declaring the
    state/RGB variants exercised by orchestration tests as test-only contracts.
    """
    declared = act_manifest().model_copy(deep=True)
    declared.slug = 'test-recording-inputs'
    declared.display_name = 'Synthetic recording inputs'
    declared.description = 'Test-only recording orchestration fixture, not a model adapter.'
    conversion = declared.train.data_requirements.recording_conversion
    rgb = conversion.presets[0]
    rgb.contract = 'test.recording-rgb/v1'
    rgb.training_setup['adapter'] = declared.slug
    state = rgb.model_copy(deep=True)
    state.id = 'state'
    state.name = 'Joint states'
    state.contract = 'test.recording-state/v1'
    state.observations = ['state']
    state.observation_requirements = rgb_requirements([])
    state.loader_validation.mode = 'state'
    conversion.presets = [state, rgb]
    conversion.default_preset = 'state'
    for field in declared.train.input_fields:
        if field.data_binding:
            field.data_binding.contracts = [state.contract, rgb.contract]
    return declared


def install_adapters(service):
    service.test_adapters = {}
    for manifest in [*builtin_adapter_manifests(), fixture_recording_manifest()]:
        if manifest.train.data_requirements and manifest.train.data_requirements.recording_conversion:
            adapter = service.database.create_adapter(name=manifest.display_name,manifest=manifest.model_dump(mode='json'))
            service.test_adapters[manifest.slug] = adapter


def create(service, session_id, kind, name, **kwargs):
    # Test shorthand resolves real registry IDs; the production API accepts no recipe aliases.
    slug, preset = {
        'fixture-rgb':('test-recording-inputs','rgb'), 'fixture-state':('test-recording-inputs','state'),
        'act':('xpolicylab-act','rgb'),'act-native':('xpolicylab-act-native','rgb'),
        'egoverse':('egoverse-hpt','hpt_joints'),
    }[kind]
    adapter = service.test_adapters[slug]
    return service.create(session_id,adapter['id'],name,
        adapter_version_id=adapter['latest_version']['id'],adapter_data_preset=preset,**kwargs)


def set_requirements(service, slug, requirements):
    adapter = service.test_adapters[slug]
    manifest = adapter['latest_version']['manifest']
    declared = manifest['train']['data_requirements']['recording_conversion']
    preset = next(p for p in declared['presets'] if p['id'] == 'rgb')
    preset['observation_requirements'] = requirements
    service.test_adapters[slug] = service.database.edit_adapter(adapter['id'], manifest=manifest)


def manifest_path(service, job):
    return service.root/job['id']/'output/manifest.json'

def test_worker_failure_distinguishes_progress_from_termination():
    log = '\n'.join(['{"episodes_done": 36, "episodes_total": 51}',
                     'resource_tracker.py: UserWarning: 1 leaked semaphore', 'warnings.warn("cleanup")'])
    message = conversion_failure(-15, log)
    assert "SIGTERM" in message and "36/51 episodes" in message
    assert "without an exception report" in message
    assert "leaked semaphore" not in message
    assert "ModuleNotFoundError: missing" in conversion_failure(1, log + '\nModuleNotFoundError: missing')

def test_conversion_results_keep_independent_names_and_source_identity(setup):
    service, session, _ = setup
    first = create(service, session["id"], "fixture-rgb", "RGB demonstrations")
    before = service.database.get_data_resource(first["resource_id"])
    second = create(service, session["id"], "act", "ACT demonstrations")
    after = service.database.get_data_resource(first["resource_id"])
    assert second["resource_id"] == first["resource_id"]
    assert second["source_version_id"] == first["source_version_id"]
    assert second["name"] == "ACT demonstrations"
    assert service.get(first["id"])["name"] == "RGB demonstrations"
    assert after["display_name"] == before["display_name"]
    assert after["metadata"] == before["metadata"]
    assert after["description"] == before["description"]


def test_preparation_options_expose_recording_and_adapters_without_source_group(setup):
    service, session, _ = setup
    create(service, session["id"], "fixture-rgb", "Named conversion")
    options = service.preparation_options(session["id"])
    assert options["session"]["id"] == session["id"]
    assert options["session"]["name"] == session["profile"]["display_name"]
    assert options["session"]["eligible"]
    assert options["adapters"]
    assert "resource" not in options
    assert "resource_id" not in options["session"]

@pytest.mark.parametrize("mutation, message", [
    ("active", "End the collection"),
    ("path", "Invalid saved image path"),
    ("hash", "checksums are missing"),
])
def test_rejects_ineligible_sources_before_dispatch(setup, mutation, message):
    service, session, _ = setup
    first = session["recordings"][0]
    if mutation == "active": session["state"] = "RUNNING"
    if mutation == "path": session["recording_images"][first]["path"] = "../../outside.hdf5"
    if mutation == "hash": session["recording_checksums"][first] = ""
    with pytest.raises(ValueError, match=message):
        create(service, session["id"], "fixture-rgb", "Test")
    assert service.list() == []
    assert not service.preparation_options(session["id"])["session"]["eligible"]

def test_missing_images_queue_conversion_with_declared_observations(setup):
    service, session, _ = setup
    session["recording_images"] = {}
    job = create(service, session["id"], "fixture-rgb", "Render when converting", target="cluster")
    assert job["state"] == "QUEUED"
    assert job["observation_contract"]["streams"]
    assert all("image_path" not in source for source in job["sources"])
    assert service.preparation_options(session["id"])["session"]["eligible"]

def test_checksum_failure_does_not_register_a_version_and_can_retry(setup):
    service, session, source = setup
    job = create(service, session["id"], "fixture-rgb", "Corrupt")
    raw = source / session["recordings"][0]
    raw.write_bytes(raw.read_bytes() + b"changed")
    service.prepare(job["id"])
    assert service.get(job["id"])["state"] == "FAILED"
    assert all(v["format"] == "skynet.episodes/v1" for r in service.database.list_data_resources() for v in service.database.get_data_resource(r["id"])["versions"])
    assert service.pending() == []
    retry = create(service, session["id"], "fixture-rgb", "Retry")
    assert retry["id"] != job["id"]
    # The monitor's pending scan skips finished and legacy rows and lists newest first.
    with service.database.transaction() as c:
        for row in (dict(id="legacy", format="legacy", state="QUEUED", created_at="2999-01-01T00:00:00.000Z"),
                    dict(id="older", format=retry["format"], state="PENDING", created_at="2000-01-01T00:00:00.000Z")):
            c.execute("INSERT INTO policy_exports VALUES (?,?)", (row["id"], canonical_json(row)))
    assert [j["id"] for j in service.pending()] == [retry["id"], "older"]

def test_shared_preparation_redacts_other_workspace_experiment_details(setup, monkeypatch):
    from skynet_app.workspaces import WorkspaceDirectory
    service, session, _ = setup
    created = create(service, session['id'], 'act', 'Shared preparation')
    job = service.get(created['id'])
    job['state'] = 'READY'
    job['version_id'] = job['source_version_id']
    monkeypatch.setattr(service, 'list', lambda: [dict(job)])
    version = service.database.get_data_resource_version(job['version_id'])
    spec = {'data': {'bundle': {'assignments': [{'version': {'manifest_sha256': version['manifest_sha256']}}]}}}
    private = service.database.create_experiment(name='Private name', requested_spec=spec)
    workspace, _ = WorkspaceDirectory(service.database).open('teammate@example.com')
    scoped = Database(service.database.path, workspace_id=workspace['id'])
    result = service.job_overview(workspace_database=scoped)
    assert result['exports'][0]['usage'] == [{'other_workspace': True}]
    assert 'Private name' not in json.dumps(result)
    # Internal deletion checks retain the actual dependency information.
    assert service.database.data_version_usage(version['manifest_sha256'])[0]['experiment_id'] == private['id']

def test_duplicate_preparation_does_not_scan_other_datasets_or_upload_metadata(setup, monkeypatch):
    service, session, _ = setup
    first = create(service, session["id"], "fixture-rgb", "Saved dataset")
    def unexpected(*args, **kwargs):
        pytest.fail("Duplicate preparation must use its saved identity without registry scan or metadata upload")
    monkeypatch.setattr(service.database, "list_data_resources", unexpected)
    monkeypatch.setattr(service, "register_source", unexpected)
    duplicate = create(service, session["id"], "fixture-rgb", "Duplicate")
    assert duplicate["id"] == first["id"]
    assert duplicate["name"] == "Saved dataset"
    assert duplicate["source_version_id"] == first["source_version_id"]
