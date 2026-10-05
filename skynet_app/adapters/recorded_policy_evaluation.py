"""Compose a saved policy with the selected recorded-simulation suite."""

import argparse
import json
from pathlib import Path

from recording_dataset import verify_dataset
from policy_contract import recorded_contract, unidex_contract, contract_issues
from policy_loading import load_policy
from policy_simulator import run_simulator, validate_simulation
from xpolicy_runtime import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    parser.add_argument("--source-dir", required=True)
    args = parser.parse_args()
    context = json.loads(Path(args.context).read_text())
    report = Path(context["result_path"]).parent / "preflight.json"
    write_json(report, {"status": "RUNNING", "phase": "load_policy"})
    try:
        config = context["policy"]["native_config"]
        expected = context["compatibility"]["io_contract"]
        unidex = context["compatibility"]["policy_loader"] == "unidex_faas"
        hat = context["compatibility"]["policy_loader"] == "hat_cartesian"
        if unidex or hat:
            target = context.get("target_dataset") or {}
            manifest = verify_dataset(target["path"], target["manifest_sha256"])
            if hat:
                from hat_evaluation import hat_contract
                actual = hat_contract(manifest, control_hz=1 / expected["step_dt"])
            else:
                actual = unidex_contract(manifest, control_hz=1 / expected["step_dt"])
        else:
            manifest = verify_dataset(config["dataset_path"], config["dataset_manifest_sha256"])
            if hat:
                from hat_evaluation import hat_contract
                actual = hat_contract(manifest, control_hz=config.get("control_hz"))
            else:
                actual = recorded_contract(manifest, images=bool(expected["cameras"]), control_hz=config.get("control_hz"))
        if actual != expected:
            raise ValueError("The dataset I/O contract differs from the submitted compatibility report")
        issues = contract_issues(actual)
        if issues:
            raise ValueError("; ".join(i["message"] for i in issues))
        policy = load_policy(context, args.source_dir, manifest)
        expected_mode = expected.get("observation_mode", "rgb" if expected["cameras"] else "state")
        if policy.mode != expected_mode:
            raise ValueError("Checkpoint camera mode differs from the submitted I/O contract")
        validate_simulation(context, manifest)
        run_simulator(context, args.context, policy)
    except Exception as error:
        previous = json.loads(report.read_text()) if report.exists() else {}
        if previous.get("status") != "PASSED":
            write_json(report, {"status": "FAILED", "error": str(error)})
            print(json.dumps({"event": "evaluation_preflight", "status": "FAILED", "error": str(error)}), flush=True)
        raise


if __name__ == "__main__":
    main()
