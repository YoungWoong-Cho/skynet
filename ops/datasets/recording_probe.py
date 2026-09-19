"""CPU-only source checks before any camera work is scheduled."""
import json
import os
from pathlib import Path
import sys


def validate_source_split(split, sources):
    split = split or {}
    train, validation = split.get("train", []), split.get("validation", [])
    if not train or any(type(i) is not int for i in train + validation) or sorted(train + validation) != list(range(len(sources))):
        raise ValueError("Recording splits must be disjoint, complete and contain training episodes")
    identities = [source["sha256"] for source in sources]
    if len(set(identities)) != len(identities):
        raise ValueError("Dataset cannot contain copies of the same raw recording")


def probe(request):
    from recording_prepare import preflight_source
    results = []
    requirements = request["requirements"]
    for source in request["sources"]:
        inspected = preflight_source(source, requirements["observation_requirements"],
                                     action_representation=requirements.get("action_representation"))
        results.append(dict(source_sha256=source["sha256"],
                            shared_image_streams=inspected["streams"],
                            capture=inspected["capture"], steps=inspected["steps"]))
    validate_source_split(request.get("split"), request["sources"])
    return dict(schema="skynet.recording-preflight/v1", job_id=request["job_id"],
                attempt_id=request["attempt_id"], verified=True, sources=results)


def main(path):
    request = json.loads(Path(path).read_text())
    try:
        result, code = probe(request), 0
    except Exception as error:
        result = dict(schema="skynet.recording-preflight/v1", job_id=request["job_id"],
                      attempt_id=request["attempt_id"], verified=False, error=str(error))
        code = 1
    output = Path(request["receipt_path"])
    temporary = output.with_suffix(".part")
    temporary.write_text(json.dumps(result, allow_nan=False))
    os.replace(temporary, output)
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
