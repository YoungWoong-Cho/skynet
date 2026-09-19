"""A frozen hand identity includes manifest bytes, not only its recipe ID."""
import hashlib
import json

import pytest

from skynet_app.database import canonical_json
from skynet_app.cluster_runtime import WORK_ROOT
from test_observation_preparation import context
from test_policy_exports import create


@pytest.mark.parametrize('local', [True, False])
def test_observation_profile_pins_original_hand_manifest(context, tmp_path, monkeypatch, local):
    import skynet_app.simulation_hands as hands
    session = context.sessions['first']
    digest = 'f' * 64
    session['profile']['hand_bundle'] = {'digest': digest}
    raw = canonical_json(dict(schema='skynet.simulation-hand/v1', robot=session['profile']['robot'], digest=digest, files={}))
    expected = hashlib.sha256(raw.encode()).hexdigest()
    directory = tmp_path / 'hand'
    directory.mkdir()
    (directory / 'manifest.json').write_text(raw)
    def find(*args, **kwargs):
        if local:
            return directory
        raise ValueError('Archived hand is only on sky2')
    monkeypatch.setattr(hands, 'find_bundle', find)
    path = f"{WORK_ROOT}/hands/{session['profile']['robot']}/{digest}/manifest.json"
    context.cluster.files[path] = raw
    profile = context.service.observations.profile(session)
    assert profile['hand_bundle']['manifest_sha256'] == expected
    session['profile']['hand_bundle']['manifest_sha256'] = expected
    changed = json.loads(raw)
    changed['files']['changed.urdf'] = dict(sha256='a' * 64, size_bytes=100)
    changed = canonical_json(changed)
    (directory / 'manifest.json').write_text(changed)
    context.cluster.files[path] = changed
    with pytest.raises(ValueError, match='manifest checksum changed'):
        context.service.observations.profile(session)


@pytest.mark.parametrize('change_manifest_after_upload', [False, True])
def test_stage_uploads_shared_hand_once_and_preserves_manifest_checks(
        context, tmp_path, monkeypatch, change_manifest_after_upload):
    import skynet_app.simulation_hands as hands
    first = context.sessions['first']
    second = context.add_session('second', checksum='b' * 64)
    digest = 'f' * 64
    manifest = dict(schema='skynet.simulation-hand/v1', robot=first['profile']['robot'], digest=digest, files={})
    raw = canonical_json(manifest)
    manifest_sha = hashlib.sha256(raw.encode()).hexdigest()
    directory = tmp_path / 'shared-hand'
    directory.mkdir()
    (directory / 'manifest.json').write_text(raw)
    for session in (first, second):
        session['profile']['hand_bundle'] = dict(digest=digest, manifest_sha256=manifest_sha)
    monkeypatch.setattr(hands, 'find_bundle', lambda *args, **kwargs: directory)
    uploads = []
    def upload(local, remote_root, transport, gateway):
        assert local == directory and transport is context.cluster and gateway == 'sky2'
        uploads.append((str(local), manifest_sha))
        if change_manifest_after_upload:
            changed = dict(manifest, files={'changed.urdf': dict(sha256='c' * 64, size_bytes=1)})
            (directory / 'manifest.json').write_text(canonical_json(changed))
        return f"{remote_root}/hands/{manifest['robot']}/{digest}"
    monkeypatch.setattr(hands, 'upload', upload)
    service = context.service
    job = create(service, 'first', 'fixture-rgb', 'Two recordings with one frozen hand',
        selections=[dict(session_id='first', indices=None), dict(session_id='second', indices=None)])
    service.prepare(job['id'])
    assert len(uploads) == 1
    current = service.get(job['id'])
    if change_manifest_after_upload:
        assert current['state'] == 'FAILED'
        assert 'Frozen hand manifest changed' in current['error']
        assert context.cluster.submissions == []
    else:
        assert current['stage'] == 'OBSERVATIONS' and current['state'] == 'QUEUED'
        producers = service.observations.store.producers()
        assert len(producers) == 1 and len(producers[0]['request']['sources']) == 2
        assert len(context.cluster.submissions) == 1
