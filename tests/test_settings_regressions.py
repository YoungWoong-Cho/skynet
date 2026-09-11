from unittest.mock import patch

import pytest

from skynet_app.credential_store import StoredCredential
from skynet_app.database import Database
from skynet_app.pipeline_api import PipelineService, TrackingConnectionRequest
from skynet_app.tracking import MLflowBridge, SessionCredentialStore, WandBBridge


class MemoryCredentials:
    def __init__(self):
        self.records = {}

    def load(self, provider):
        return self.records.get(provider)

    def save(self, provider, endpoint, credentials):
        self.records[provider] = StoredCredential(provider, endpoint, dict(credentials))

    def delete(self, provider):
        self.records.pop(provider, None)


@pytest.fixture
def service(tmp_path, monkeypatch):
    for name in ('WANDB_BASE_URL', 'WANDB_API_KEY', 'WANDB_ENTITY', 'MLFLOW_TRACKING_URI',
                 'MLFLOW_TRACKING_TOKEN', 'MLFLOW_TRACKING_USERNAME', 'MLFLOW_TRACKING_PASSWORD'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr('skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT', tmp_path / 'capsules')
    result = PipelineService(Database(tmp_path / 'test.db'), object(),
                             credential_store=MemoryCredentials(), session_credentials=SessionCredentialStore())
    monkeypatch.setattr(result, '_flush_tracking_provider', lambda _: {})
    return result


@pytest.fixture
def validated(monkeypatch):
    captured = []
    def mlflow(bridge):
        captured.append(('mlflow', bridge.settings))
        return {'reachable': True}
    def wandb(bridge):
        captured.append(('wandb', bridge.settings))
        return {'viewer_id': '1', 'username': 'qa', 'entity': 'qa'}
    monkeypatch.setattr(MLflowBridge, 'validate_connection', mlflow)
    monkeypatch.setattr(WandBBridge, 'validate_connection', wandb)
    monkeypatch.setattr(WandBBridge, 'validate_entity', lambda *args: None)
    return captured


def connect(service, provider='mlflow', **kwargs):
    return service.configure_tracking_connection(provider, TrackingConnectionRequest(remember=False, **kwargs))


def test_repeated_basic_connect_reuses_secret_only_for_same_identity(service, validated):
    connect(service, tracking_uri='https://mlflow.test', username='alice', password='fixture-secret', verify_tls=False)
    response = connect(service, tracking_uri='https://mlflow.test', username='alice', verify_tls=False)
    assert validated[-1][1].password == 'fixture-secret'
    assert response['connection']['verify_tls'] is False
    assert response['connection']['username'] == 'alice'
    assert 'fixture-secret' not in str(response)
    with pytest.raises(ValueError, match='new MLflow credentials'):
        connect(service, tracking_uri='https://different.test', username='alice')
    assert service._mlflow_settings().tracking_uri == 'https://mlflow.test'


def test_replacing_basic_identity_does_not_reuse_password(service, validated):
    connect(service, tracking_uri='https://mlflow.test', username='alice', password='fixture-secret')
    connect(service, tracking_uri='https://mlflow.test', username='bob')
    assert validated[-1][1].password is None
    assert service._mlflow_settings().username == 'bob'
    assert service._mlflow_settings().password is None


@pytest.mark.parametrize('provider, endpoint_field, secret_field', [
    ('mlflow', 'tracking_uri', 'token'), ('wandb', 'base_url', 'api_key')])
def test_endpoint_change_requires_new_secret_before_any_validation(service, validated, provider, endpoint_field, secret_field):
    connect(service, provider, **{endpoint_field: 'https://trusted.test', secret_field: 'old-fixture-secret'})
    before = len(validated)
    with pytest.raises(ValueError, match='new .*credentials|new W&B API key'):
        connect(service, provider, **{endpoint_field: 'https://other.test'})
    assert len(validated) == before
    connect(service, provider, **{endpoint_field: 'https://other.test', secret_field: 'new-fixture-secret'})
    assert getattr(validated[-1][1], secret_field) == 'new-fixture-secret'
    assert service.credentials.endpoint(provider) == 'https://other.test'
    settings = service._wandb_settings() if provider == 'wandb' else service._mlflow_settings()
    assert getattr(settings, endpoint_field) == 'https://other.test'
    assert getattr(settings, secret_field) == 'new-fixture-secret'
    assert 'old-fixture-secret' not in str(service.credentials.get(provider))


def test_disconnected_credential_is_not_reused_for_anonymous_new_endpoint(service, validated):
    connect(service, tracking_uri='https://trusted.test', token='old-fixture-secret')
    service.disconnect_tracking_connection('mlflow')
    response = connect(service, tracking_uri='https://public.test')
    assert validated[-1][1].token is None
    assert service._mlflow_settings().token is None
    assert response['connection']['credential_source'] is None
    assert service.database.get_tracking_connection('mlflow')['config_json']['authentication'] == 'none'


@pytest.mark.parametrize('provider, uri_name, secret_name, field', [
    ('mlflow', 'MLFLOW_TRACKING_URI', 'MLFLOW_TRACKING_TOKEN', 'token'),
    ('wandb', 'WANDB_BASE_URL', 'WANDB_API_KEY', 'api_key')])
def test_environment_credential_never_falls_back_to_another_endpoint(service, monkeypatch, provider, uri_name, secret_name, field):
    monkeypatch.setenv(uri_name, 'https://environment.test')
    monkeypatch.setenv(secret_name, 'environment-fixture-secret')
    service.database.upsert_tracking_connection(provider, endpoint='https://saved.test', config={'authentication': 'none' if provider == 'mlflow' else 'api_key', 'credential_source': 'session'})
    settings = service._mlflow_settings() if provider == 'mlflow' else service._wandb_settings()
    assert getattr(settings, field) is None
    assert service.tracking_connections()['connections'][provider]['credential_source'] is None


def test_anonymous_connection_does_not_inherit_even_same_endpoint_environment_auth(service, validated, monkeypatch):
    connect(service, tracking_uri='https://public.test')
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'https://public.test')
    monkeypatch.setenv('MLFLOW_TRACKING_TOKEN', 'environment-fixture-secret')
    assert service._mlflow_settings().token is None
    connect(service, tracking_uri=service._mlflow_settings().tracking_uri)
    assert validated[-1][1].token is None


def test_session_token_does_not_merge_environment_basic_fields(service, validated, monkeypatch):
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'https://mlflow.test')
    monkeypatch.setenv('MLFLOW_TRACKING_USERNAME', 'environment-user')
    monkeypatch.setenv('MLFLOW_TRACKING_PASSWORD', 'environment-password')
    response = connect(service, tracking_uri='https://mlflow.test', token='session-token')
    settings = service._mlflow_settings()
    assert settings.token == 'session-token'
    assert settings.username is None and settings.password is None
    assert response['connection']['username'] is None
    connect(service, tracking_uri='https://mlflow.test')
    assert validated[-1][1].token == 'session-token'
    assert validated[-1][1].username is None


def test_same_endpoint_authentication_mode_transition_replaces_all_fields(service, validated):
    connect(service, tracking_uri='https://mlflow.test', username='alice', password='basic-secret')
    connect(service, tracking_uri='https://mlflow.test', token='token-secret')
    assert service.credentials.get('mlflow') == {'token': 'token-secret'}
    connect(service, tracking_uri='https://mlflow.test', username='bob', password='new-basic-secret')
    assert service.credentials.get('mlflow') == {'username': 'bob', 'password': 'new-basic-secret'}


def test_new_secret_cannot_pair_with_previous_host_after_failed_database_commit(service, validated, monkeypatch):
    connect(service, tracking_uri='https://trusted.test', token='old-fixture-secret')
    monkeypatch.setattr(service.database, 'upsert_tracking_connection', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('fixture write failure')))
    with pytest.raises(RuntimeError, match='fixture write failure'):
        connect(service, tracking_uri='https://other.test', token='new-fixture-secret')
    assert service._mlflow_settings().tracking_uri == 'https://trusted.test'
    assert service._mlflow_settings().token is None
    assert service.tracking_connections()['connections']['mlflow']['connected'] is False


def test_environment_secret_is_redacted_from_connect_errors(service, monkeypatch):
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'https://environment.test')
    monkeypatch.setenv('MLFLOW_TRACKING_TOKEN', 'environment-fixture-secret')
    def fail(_):
        raise RuntimeError('rejected environment-fixture-secret')
    monkeypatch.setattr(MLflowBridge, 'validate_connection', fail)
    with pytest.raises(Exception) as raised:
        connect(service, tracking_uri=service._mlflow_settings().tracking_uri)
    assert 'environment-fixture-secret' not in str(raised.value)
    assert 'environment-fixture-secret' not in str(service.tracking_connections())


def test_late_connect_response_cannot_overwrite_a_changed_connection(service, validated, monkeypatch):
    connect(service, tracking_uri='https://first.test', token='first-token')
    def validate(_):
        service._commit_tracking_connection('mlflow', 'https://second.test', {'token': 'second-token'},
                                           origin='session', remember=False, verify_tls=True,
                                           expected_settings=service._mlflow_settings(),
                                           expected_revision=service._tracking_connection_revisions['mlflow'])
        return {'reachable': True}
    monkeypatch.setattr(MLflowBridge, 'validate_connection', validate)
    with pytest.raises(ValueError, match='changed while'):
        connect(service, tracking_uri=service._mlflow_settings().tracking_uri)
    assert service._mlflow_settings().tracking_uri == 'https://second.test'
    assert service._mlflow_settings().token == 'second-token'
    assert service.tracking_connections()['connections']['mlflow']['connected'] is True


def test_late_validation_failure_does_not_mark_a_new_connection_failed(service, validated, monkeypatch):
    connect(service, tracking_uri='https://first.test', token='first-token')
    def validate(_):
        service._commit_tracking_connection('mlflow', 'https://second.test', {'token': 'second-token'},
                                           origin='session', remember=False, verify_tls=True,
                                           expected_settings=service._mlflow_settings(),
                                           expected_revision=service._tracking_connection_revisions['mlflow'])
        raise RuntimeError('late first-token validation failure')
    monkeypatch.setattr(MLflowBridge, 'validate_connection', validate)
    with pytest.raises(Exception):
        connect(service, tracking_uri=service._mlflow_settings().tracking_uri)
    assert service.tracking_connections()['connections']['mlflow']['connected'] is True
    assert service._mlflow_settings().token == 'second-token'


@pytest.mark.parametrize('provider, endpoint_field, secret_field, bridge_class', [
    ('mlflow', 'tracking_uri', 'token', MLflowBridge),
    ('wandb', 'base_url', 'api_key', WandBBridge)])
@pytest.mark.parametrize('newer_action', ['disconnect', 'connect'])
def test_late_configure_cannot_replace_a_newer_connection_action(
    service, validated, monkeypatch, provider, endpoint_field, secret_field, bridge_class, newer_action,
):
    connect(service, provider, **{endpoint_field: 'https://first.test', secret_field: 'first-token'})
    settings_for = service._wandb_settings if provider == 'wandb' else service._mlflow_settings

    def validate(_):
        if newer_action == 'disconnect':
            service.disconnect_tracking_connection(provider)
        else:
            service._commit_tracking_connection(
                provider, 'https://second.test', {secret_field: 'second-token'},
                origin='session', remember=False, verify_tls=True,
                expected_settings=settings_for(), workspace='qa' if provider == 'wandb' else None,
                expected_revision=service._tracking_connection_revisions[provider],
            )
        return {'reachable': True, 'entity': 'qa', 'username': 'qa'}

    monkeypatch.setattr(bridge_class, 'validate_connection', validate)
    with pytest.raises(ValueError, match='changed while'):
        connect(service, provider, **{endpoint_field: 'https://first.test'})
    if newer_action == 'disconnect':
        assert not service.credentials.get(provider)
        assert service.database.get_tracking_connection(provider) is None
    else:
        assert getattr(settings_for(), endpoint_field) == 'https://second.test'
        assert getattr(settings_for(), secret_field) == 'second-token'
        assert service.tracking_connections()['connections'][provider]['connected'] is True


def test_disconnect_invalidates_inflight_configure_even_when_environment_settings_are_unchanged(service, monkeypatch):
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'https://environment.test')
    monkeypatch.setenv('MLFLOW_TRACKING_TOKEN', 'environment-token')
    before = service._mlflow_settings()

    def validate(_):
        service.disconnect_tracking_connection('mlflow')
        assert service._mlflow_settings() == before
        return {'reachable': True}

    monkeypatch.setattr(MLflowBridge, 'validate_connection', validate)
    with pytest.raises(ValueError, match='changed while'):
        connect(service, tracking_uri='https://environment.test')
    assert service.database.get_tracking_connection('mlflow') is None
    assert service.tracking_connections()['connections']['mlflow']['connected'] is False


def test_stored_secret_keeps_endpoint_binding_after_restart(service, validated):
    service.configure_tracking_connection('mlflow', TrackingConnectionRequest(
        tracking_uri='https://trusted.test', token='stored-fixture-token', remember=True))
    restarted = PipelineService(service.database, object(), credential_store=service.credential_store,
                                session_credentials=SessionCredentialStore())
    assert restarted._mlflow_settings().token == 'stored-fixture-token'
    assert restarted.credentials.endpoint('mlflow') == 'https://trusted.test'
    restarted.database.upsert_tracking_connection('mlflow', endpoint='https://other.test', config={'authentication': 'token', 'credential_source': 'credential_store'})
    assert restarted._mlflow_settings().token is None
