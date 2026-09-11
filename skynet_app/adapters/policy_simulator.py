"""Run policy inference and Isaac in their independently pinned Python runtimes."""

from multiprocessing.connection import Listener
import os
from pathlib import Path
import secrets
import subprocess
import threading
from tempfile import TemporaryDirectory

def simulator_environment(runtime):
    env = os.environ.copy()
    for key in [
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
    ]:
        env.pop(key, None)
    root = runtime["environment_path"]
    env.update(runtime.get("environment", {}))
    env.update(runtime.get("operator_environment", {}))
    for op in runtime.get("profile_snapshot", {}).get("environment_operations", []):
        name = op["name"]
        value = op.get("value", "").replace("{{runtime.environment_path}}", root)
        old = env.get(name, "")
        separator = op.get("separator", ":")
        if op["operation"] == "unset":
            env.pop(name, None)
        elif op["operation"] == "set":
            env[name] = value
        elif op["operation"] == "prepend":
            env[name] = value + (separator + old if old else "")
        elif op["operation"] == "append":
            env[name] = (old + separator if old else "") + value
        else:
            raise ValueError("Unsupported runtime environment operation")
    env.update(
        CONDA_PREFIX=root,
        PATH=root + "/bin:" + env.get("PATH", ""),
        PYTHONNOUSERSITE="1",
        PYTHONPATH=runtime["source_dir"] + "/source/dexverse",
    )
    return env


def validate_simulation(context, manifest):
    if len(context["tasks"]) != 1 or context["parallelism"] != 1:
        raise ValueError("Each evaluation worker requires one assigned task")
    if context["tasks"][0] not in context["suite"]["tasks"]:
        raise ValueError("Task is outside the pinned evaluation catalog")
    runtime = context["evaluator_runtime"]
    source = Path(runtime["source_dir"])
    revision = (source / ".skynet-source-revision").read_text().strip()
    if (
        revision != manifest["capture"]["source_revision"]
        or revision != runtime["source"]["revision"]
    ):
        raise ValueError("Evaluation source does not match collection")

def run_simulator(context, context_path, policy):
    runtime = context["evaluator_runtime"]
    source = Path(runtime["source_dir"])
    auth = secrets.token_bytes(32)
    listener = Listener(("127.0.0.1", 0), authkey=auth)

    def serve():
        try:
            with listener.accept() as conn:
                while True:
                    request = conn.recv()
                    try:
                        if request["command"] == "reset":
                            policy.reset(request["seed"])
                            result = None
                        elif request["command"] == "step":
                            result = policy.step(
                                request["observation"], request["predict"]
                            )
                        else:
                            raise ValueError("Unknown policy request")
                        conn.send({"result": result})
                    except Exception as exc:
                        conn.send({"error": str(exc)})
        except (EOFError, OSError):
            pass

    threading.Thread(target=serve, daemon=True).start()
    env = simulator_environment(runtime)
    env["SKYNET_POLICY_AUTH"] = auth.hex()
    env["SKYNET_POLICY_PORT"] = str(listener.address[1])
    env["SKYNET_POLICY_IMAGES"] = "1" if policy.mode == "rgb" else "0"
    temporary_root = Path(context["result_path"]).parent
    temporary_root.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(prefix="simulator-", dir=temporary_root) as temporary:
            env["TMPDIR"] = temporary
            subprocess.run(
                [
                    runtime["python_executable"],
                    str(Path(__file__).with_name("dexverse_evaluation.py")),
                    "--context",
                    context_path,
                ],
                env=env,
                cwd=source,
                check=True,
            )
    finally:
        listener.close()

