"""CPU-only source checks before any camera work is scheduled."""
import json
import math
import os
from pathlib import Path
import sys


def validate_temporal(temporal, source, inspected):
    if not temporal:
        return
    stride, horizon, execute = (temporal.get(key) for key in ("frame_stride", "action_horizon", "execution_horizon"))
    if any(type(value) is not int or value < 1 for value in (stride, horizon, execute)) or execute > horizon:
        raise ValueError("Adapter temporal requirements need positive stride/action/execution horizons")
    source_hz, control_hz = (float(temporal.get(key, 0)) for key in ("source_fps", "control_hz"))
    if (not all(math.isfinite(value) and value > 0 for value in (source_hz, control_hz))
            or not math.isclose(source_hz / stride, control_hz, rel_tol=1e-6)):
        raise ValueError("Adapter control frequency must equal recording frequency / frame stride")
    dt = float(inspected["capture"].get("step_dt", 0))
    if not math.isfinite(dt) or not math.isclose(dt * source_hz, 1.0, rel_tol=1e-6):
        raise ValueError("Adapter temporal requirements differ from the recorded control timing")
    minimum = (horizon - 1) * stride + 1
    count = int(inspected["steps"])
    if count < minimum:
        recording = source.get("recording_id") or source["sha256"][:12]
        raise ValueError(
            f"Recording {recording} has {count} frames; this adapter needs at least {minimum} "
            f"recorded frames for a complete {horizon}-step action chunk at {control_hz:g} Hz "
            f"from {source_hz:g} Hz recordings. Collect a longer episode or select a shorter-horizon "
            "adapter preset. No camera work has started."
        )


def validate_native_split(requirements, split, sources):
    if requirements.get("contract") != "skynet.unidex-pointcloud-faas/v1":
        return
    split = split or {}
    train, validation = split.get("train", []), split.get("validation", [])
    if (not train or not validation or any(type(i) is not int for i in train + validation)
            or sorted(train + validation) != list(range(len(sources)))):
        raise ValueError("UniDex requires separate nonempty training and validation recordings, split by whole episode before action chunks. Select at least two distinct complete recordings and a nonzero validation split. No camera work has started.")
    identities = [source["sha256"] for source in sources]
    if len(set(identities)) != len(identities):
        raise ValueError("UniDex cannot place copies of the same raw recording in its source-episode split")


def probe(request):
    from recording_prepare import preflight_source
    results = []
    requirements = request["requirements"]
    for source in request["sources"]:
        inspected = preflight_source(source, requirements["observation_requirements"],
                                     action_representation=requirements.get("action_representation"))
        validate_temporal(requirements.get("temporal"), source, inspected)
        results.append(dict(source_sha256=source["sha256"],
                            shared_image_streams=inspected["streams"],
                            capture=inspected["capture"], steps=inspected["steps"]))
    validate_native_split(requirements, request.get("split"), request["sources"])
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
