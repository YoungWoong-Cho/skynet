CREATE TABLE maintenance_operations (
    target_kind TEXT NOT NULL,
    target_id TEXT NOT NULL,
    owner_id TEXT NOT NULL REFERENCES workspaces(id),
    plan_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(target_kind,target_id)
);
CREATE OR REPLACE FUNCTION attempt_common_hyperparameter_receipt_no_delete_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF OLD.entity_type='job_attempt' AND OLD.event_type='COMMON_HYPERPARAMETERS_ENRICHED_V1'
 AND current_setting('skynet.delete_history', true) IS DISTINCT FROM 'on' THEN
 RAISE EXCEPTION 'attempt common hyperparameter receipts are immutable' USING ERRCODE='23514';
END IF;
RETURN OLD;
END; $$;
