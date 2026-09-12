"""Compose a saved policy with the selected recorded-simulation suite."""

import argparse
import json
from pathlib import Path

from artifacts import verify
from policy_contract import recorded_contract, contract_issues
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
        manifest = verify(config["dataset_path"], config["dataset_manifest_sha256"])
        expected = context["compatibility"]["io_contract"]
        actual = recorded_contract(manifest, images=bool(expected["cameras"]))
        if actual != expected:
            raise ValueError("The dataset I/O contract differs from the submitted compatibility report")
        issues = contract_issues(actual)
        if issues:
            raise ValueError("; ".join(i["message"] for i in issues))
        policy = load_policy(context, args.source_dir, manifest)
        if (policy.mode == "rgb") != bool(expected["cameras"]):
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
