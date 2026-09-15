-- Immutable observations shared by preparation jobs and published dataset versions.
CREATE TABLE observation_producers (
    id TEXT PRIMARY KEY,
    attempt_token TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK (state IN ('QUEUED','SUBMITTING','PENDING','RUNNING','READY','FAILED')),
    payload_json TEXT NOT NULL,
    lease_owner TEXT,
    lease_until DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX observation_producers_active ON observation_producers(state) WHERE state NOT IN ('READY','FAILED');
CREATE TABLE observation_artifacts (
    artifact_key TEXT PRIMARY KEY CHECK (artifact_key ~ '^[a-f0-9]{64}$'),
    spec_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('MISSING','QUEUED','RUNNING','READY','FAILED')),
    producer_id TEXT REFERENCES observation_producers(id),
    path TEXT,
    manifest_sha256 TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (state <> 'READY' OR (path IS NOT NULL AND manifest_sha256 ~ '^[a-f0-9]{64}$'))
);
CREATE TABLE observation_sources (
    artifact_key TEXT NOT NULL REFERENCES observation_artifacts(artifact_key) ON DELETE CASCADE,
    session_id TEXT NOT NULL REFERENCES live_xr_sessions(id),
    recording_path TEXT NOT NULL,
    PRIMARY KEY (artifact_key, session_id, recording_path)
);
CREATE INDEX observation_sources_recording ON observation_sources(session_id, recording_path);
CREATE TABLE observation_artifact_inputs (
    artifact_key TEXT NOT NULL REFERENCES observation_artifacts(artifact_key) ON DELETE CASCADE,
    input_key TEXT NOT NULL REFERENCES observation_artifacts(artifact_key),
    PRIMARY KEY (artifact_key, input_key),
    CHECK (artifact_key <> input_key)
);
CREATE TABLE observation_job_inputs (
    job_id TEXT NOT NULL REFERENCES policy_exports(id) ON DELETE CASCADE,
    artifact_key TEXT NOT NULL REFERENCES observation_artifacts(artifact_key),
    PRIMARY KEY (job_id, artifact_key)
);
CREATE TABLE observation_version_inputs (
    version_id TEXT NOT NULL REFERENCES data_resource_versions(id) ON DELETE CASCADE,
    artifact_key TEXT NOT NULL REFERENCES observation_artifacts(artifact_key),
    PRIMARY KEY (version_id, artifact_key)
);
CREATE FUNCTION guard_ready_observation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.state = 'READY' AND ROW(NEW.spec_json, NEW.state, NEW.path, NEW.manifest_sha256)
       IS DISTINCT FROM ROW(OLD.spec_json, OLD.state, OLD.path, OLD.manifest_sha256) THEN
        RAISE EXCEPTION 'Published observation artifacts are immutable';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER immutable_observation BEFORE UPDATE ON observation_artifacts
    FOR EACH ROW EXECUTE FUNCTION guard_ready_observation();
CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON observation_artifacts
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('data,exports', 'shared');
CREATE TRIGGER skynet_change AFTER INSERT OR DELETE ON observation_version_inputs
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('data,exports', 'shared');
