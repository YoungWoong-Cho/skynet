CREATE TABLE metadata_payloads (
    sha256 TEXT PRIMARY KEY CHECK(length(sha256)=64),
    path TEXT NOT NULL UNIQUE,
    size_bytes BIGINT NOT NULL CHECK(size_bytes>=0)
);
CREATE TABLE metadata_payload_refs (
    table_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    field_name TEXT NOT NULL,
    sha256 TEXT NOT NULL REFERENCES metadata_payloads(sha256) ON DELETE RESTRICT,
    PRIMARY KEY(table_name,record_id,field_name)
);
CREATE INDEX metadata_payload_refs_digest ON metadata_payload_refs(sha256);

-- The offline relocation changes only representation, never receipt content.
CREATE OR REPLACE FUNCTION adapter_versions_immutable_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF current_setting('skynet.relocate_payloads', true)='on'
 AND (to_jsonb(NEW)-'manifest_json')=(to_jsonb(OLD)-'manifest_json')
 AND NEW.manifest_json::jsonb->'$skynet_object_v1'->>'sha256'=encode(sha256(convert_to(OLD.manifest_json,'UTF8')),'hex') THEN
 RETURN NEW;
END IF;
RAISE EXCEPTION 'adapter versions are immutable' USING ERRCODE='23514';
END; $$;
CREATE OR REPLACE FUNCTION attempt_snapshot_immutable_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF current_setting('skynet.relocate_payloads', true)='on'
 AND (to_jsonb(NEW)-'execution_snapshot_json')=(to_jsonb(OLD)-'execution_snapshot_json')
 AND NEW.execution_snapshot_json::jsonb->'$skynet_object_v1'->>'sha256'=encode(sha256(convert_to(OLD.execution_snapshot_json,'UTF8')),'hex') THEN
 RETURN NEW;
END IF;
RAISE EXCEPTION 'attempt execution snapshots are immutable' USING ERRCODE='23514';
END; $$;
