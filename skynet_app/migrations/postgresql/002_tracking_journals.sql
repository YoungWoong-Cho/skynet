-- Shared delivery cursors and sanitized tracking queues; credentials stay in
-- each host's credential store and are never persisted in these journals.
CREATE UNIQUE INDEX runs_journal_owner ON runs(id, owner_id);

CREATE TABLE tracking_journals (
    owner_id TEXT NOT NULL REFERENCES workspaces(id),
    scope TEXT NOT NULL,
    filename TEXT NOT NULL,
    run_id TEXT,
    payload BYTEA NOT NULL,
    PRIMARY KEY(owner_id, scope, filename),
    FOREIGN KEY(run_id, owner_id) REFERENCES runs(id, owner_id) ON DELETE CASCADE
);
CREATE FUNCTION tracking_journals_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE owner TEXT;
BEGIN
    IF TG_OP='INSERT' THEN owner:=NEW.owner_id; ELSE owner:=OLD.owner_id; END IF;
    IF current_workspace_id() IS NOT NULL AND owner IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' AND NEW.owner_id IS DISTINCT FROM OLD.owner_id THEN
        RAISE EXCEPTION 'Journal ownership is immutable' USING ERRCODE='23514';
    END IF;
    IF TG_OP='DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
END; $$;
CREATE TRIGGER tracking_journals_guard BEFORE INSERT OR UPDATE OR DELETE ON tracking_journals
FOR EACH ROW EXECUTE FUNCTION tracking_journals_guard();
