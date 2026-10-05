"""Retired standalone routes cannot mutate state outside the browser workflow."""
import re
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from skynet_app import pipeline_api, collection_api, policy_exports_api

REMOVED = [('GET', '/api/data/exports/datasets/{version_id}/retirement'), ('POST', '/api/data/exports/datasets/{version_id}/retirement'), ('DELETE', '/api/data/exports/{identifier}'), ('GET', '/api/data/versions/{version_id}'), ('GET', '/api/data/derivations/{derivation_id}'), ('GET', '/api/data/bundles'), ('POST', '/api/data/bundles'), ('GET', '/api/data/bundles/{bundle_id}'), ('GET', '/api/data/bundles/{bundle_id}/preview'), ('GET', '/api/data/bundles/{bundle_id}/preview/media'), ('DELETE', '/api/data/bundles/{bundle_id}'), ('PATCH', '/api/adapters/{adapter_id}'), ('POST', '/api/adapters/{adapter_id}/archive'), ('GET', '/api/adapters/{adapter_id}/validations'), ('GET', '/api/capabilities'), ('GET', '/api/runtime-profiles'), ('POST', '/api/reconcile'), ('POST', '/api/tracking/connections/{provider}'), ('DELETE', '/api/tracking/connections/{provider}'), ('GET', '/api/collection/capabilities'), ('PATCH', '/api/collection/adapters/{adapter_id}'), ('GET', '/api/collection/sessions/{session_id}/manifest')]

@pytest.fixture
def client():
    app = FastAPI()
    for module in (pipeline_api, collection_api, policy_exports_api):
        app.include_router(module.router)
    return TestClient(app)

@pytest.mark.parametrize("method,path", REMOVED)
def test_removed_routes_are_not_callable(client, method, path):
    url = re.sub(r"\{[^}]+\}", "unused-id", path)
    response = client.request(method, url, json={})
    assert response.status_code in {404, 405}, response.text


def test_browser_submission_and_recovery_routes_remain():
    routes = {(method, route.path) for route in pipeline_api.router.routes for method in route.methods}
    assert {
        ("POST", "/api/experiments/{experiment_id}/submit"),
        ("POST", "/api/runs/{run_id}/recover-submission"),
        ("POST", "/api/runs/{run_id}/resume"),
        ("POST", "/api/runs/{run_id}/rerun"),
        ("POST", "/api/evaluations"),
        ("POST", "/api/tracking/connections/{provider}/connect"),
        ("POST", "/api/tracking/connections/{provider}/disconnect"),
    } <= routes
