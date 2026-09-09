from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .capture_processing.api import router as capture_processing_router
from .cluster_config import CLUSTER
from .collection_api import router as collection_router
from .local_capture_api import router as local_capture_router
from .hands_api import router as hands_router
from .live_xr_api import router as live_xr_router
from .live_xr_api import conversions as live_conversions
from .policy_exports_api import router as policy_exports_router, service as policy_exports
from .pipeline_api import router as pipeline_router
from .pipeline_api import service as pipeline_service


APP_ROOT = Path(__file__).resolve().parent.parent
STATIC_ROOT = APP_ROOT / "static"

WORK_ROOT = CLUSTER.paths.work_root
SLURM_BIN = CLUSTER.commands.slurm_bin
GPU_USAGE = CLUSTER.commands.gpu_usage
GPU_USAGE_LONG_COMMAND = CLUSTER.commands.gpu_usage_shell_command("-l")
GPU_USAGE_USER_COMMAND = CLUSTER.commands.gpu_usage_shell_command("-u")
SSH_HOSTS = tuple(CLUSTER.gateways)
GPU_USAGE_COLUMNS = tuple(CLUSTER.dashboard.gpu_usage_columns)
OVERFLOW_PARTITIONS = frozenset(CLUSTER.dashboard.overflow_partitions)
OVERFLOW_ACCOUNT = CLUSTER.dashboard.overflow_account_label
ALLOWED_GATEWAYS = frozenset(("auto", *SSH_HOSTS))

QUERY_COMMAND = rf'''export PATH={SLURM_BIN}:$PATH
LC_ALL=C squeue -h -o '%i|%u|%T|%P|%D|%N|%b|%M|%l|%j|%R|%a'
printf '\n__SKYNET_JOBS__\n'
printf '\n__SKYNET_USAGE__\n'
LC_ALL=C {GPU_USAGE_LONG_COMMAND} 2>/dev/null || true
printf '\n__SKYNET_USER_USAGE__\n'
LC_ALL=C {GPU_USAGE_USER_COMMAND} 2>/dev/null || true
'''

INIT_COMMAND = "mkdir -p " + " ".join(
    [
        f"{WORK_ROOT}/workspace",
        f"{WORK_ROOT}/repos",
        f"{WORK_ROOT}/datasets",
        f"{WORK_ROOT}/artifacts",
        f"{WORK_ROOT}/logs",
        f"{WORK_ROOT}/jobs",
        f"{WORK_ROOT}/eval/catalogs",
        f"{WORK_ROOT}/eval/datasets",
        f"{WORK_ROOT}/eval/assets",
        f"{WORK_ROOT}/eval/runs",
        f"{WORK_ROOT}/mlflow/db",
        f"{WORK_ROOT}/mlflow/artifacts",
        f"{WORK_ROOT}/.cache/uv",
        f"{WORK_ROOT}/.cache/huggingface",
        f"{WORK_ROOT}/.cache/torch",
    ]
)

@asynccontextmanager
async def lifespan(_: FastAPI):
    pipeline_service.start()
    live_conversions.start()
    policy_exports.start()
    try:
        yield
    finally:
        policy_exports.stop()
        live_conversions.stop()
        pipeline_service.stop()


app = FastAPI(
    title="Skynet Slurm Console",
    version="0.2.0",
    docs_url="/api/docs",
    redoc_url=None,
    lifespan=lifespan,
)


class ClusterUnavailable(RuntimeError):
    pass


def _ssh(host: str, command: str, *, stdin: str | None = None, timeout: int = 20) -> str:
    process = subprocess.run(
        [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=6",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=1",
            host,
            command,
        ],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout or "SSH command failed").strip()
        raise ClusterUnavailable(f"{host}: {detail}")
    return process.stdout


def _gateway_candidates(gateway: str) -> tuple[str, ...]:
    if gateway not in ALLOWED_GATEWAYS:
        raise HTTPException(status_code=422, detail="Unknown SSH gateway")
    if gateway == "auto":
        return SSH_HOSTS
    return (gateway,)


def _run_with_fallback(command: str, gateway: str, *, timeout: int = 20) -> tuple[str, str]:
    errors: list[str] = []
    for host in _gateway_candidates(gateway):
        try:
            return host, _ssh(host, command, timeout=timeout)
        except (ClusterUnavailable, subprocess.TimeoutExpired) as error:
            errors.append(str(error))
    raise ClusterUnavailable("; ".join(errors) or "No SSH gateways are configured")


def _tres_gpu_count(tres: str) -> int:
    if not tres or tres in {"N/A", "(null)"}:
        return 0
    generic = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", tres)
    if generic:
        return int(generic.group(1))
    typed = re.findall(r"(?:^|,)gres/gpu:[^=,]+=(\d+)", tres)
    if typed:
        return sum(int(count) for count in typed)
    gres = re.findall(r"(?:^|,)(?:gres/)?gpu(?::[^:,()=]+)?[:=](\d+)", tres)
    return sum(int(count) for count in gres)


def _tres_gpu_model(tres: str) -> str | None:
    match = re.search(r"(?:^|,)(?:gres/)?gpu:([^:,()=]+)[:=]\d+", tres or "")
    return match.group(1).lower() if match else None


def _parse_job(line: str) -> dict[str, object] | None:
    fields = line.split("|", 11)
    if len(fields) != 12:
        return None
    (
        job_id,
        user,
        state,
        partition,
        node_count,
        node_list,
        tres,
        runtime,
        time_limit,
        name,
        reason,
        account,
    ) = fields
    return {
        "id": job_id,
        "user": user,
        "state": state,
        "partition": partition,
        "node_count": int(node_count) if node_count.isdigit() else 0,
        "node_list": node_list,
        "gpus": _tres_gpu_count(tres),
        "gpu_model": _tres_gpu_model(tres),
        "runtime": runtime,
        "time_limit": time_limit,
        "name": name,
        "reason": reason,
        "account": account,
    }


def _parse_usage_value(value: str) -> dict[str, int | None]:
    parts = [part.strip() for part in value.split("/", 1)]
    usage = int(parts[0]) if parts[0].isdigit() else 0
    limit = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else None
    return {"usage": usage, "limit": limit}


def _normalize_usage_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _parse_account_usage(output: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    header_indexes: dict[str, int] = {}
    for line in output.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells:
            continue

        normalized_first = _normalize_usage_column(cells[0])
        if normalized_first == "account":
            header_indexes = {}
            for index, name in enumerate(cells):
                normalized_name = _normalize_usage_column(name)
                if normalized_name in {"cpu", "cpus"}:
                    header_indexes["cpu"] = index
                elif normalized_name.startswith("total"):
                    header_indexes["total_gpus"] = index
                else:
                    for column in GPU_USAGE_COLUMNS:
                        if normalized_name == _normalize_usage_column(column):
                            header_indexes[column] = index
                            break
            for column in GPU_USAGE_COLUMNS:
                header_indexes.setdefault(column, -1)
            header_indexes.setdefault("cpu", -1)
            header_indexes.setdefault("total_gpus", -1)
            continue

        if normalized_first == "username":
            continue

        if not header_indexes and len(cells) < len(GPU_USAGE_COLUMNS) + 1:
            continue

        row: dict[str, object] = {"account": cells[0]}
        if header_indexes:
            for index, column in enumerate(GPU_USAGE_COLUMNS):
                value_index = header_indexes.get(column, -1)
                if value_index < 0:
                    value_index = index + 1
                row[column] = (
                    _parse_usage_value(cells[value_index]) if value_index < len(cells) else {"usage": 0, "limit": None}
                )

            cpu_index = header_indexes.get("cpu", -1)
            total_index = header_indexes.get("total_gpus", -1)
            if cpu_index < 0:
                cpu_index = len(cells) - 2 if len(cells) >= len(GPU_USAGE_COLUMNS) + 3 else -1
            if total_index < 0:
                total_index = len(cells) - 1 if len(cells) >= len(GPU_USAGE_COLUMNS) + 2 else -1
        else:
            for index, column in enumerate(GPU_USAGE_COLUMNS, start=1):
                row[column] = (
                    _parse_usage_value(cells[index]) if index < len(cells) else {"usage": 0, "limit": None}
                )
            cpu_index = len(cells) - 2 if len(cells) >= len(GPU_USAGE_COLUMNS) + 3 else -1
            total_index = len(cells) - 1 if len(cells) >= len(GPU_USAGE_COLUMNS) + 2 else -1

        row["cpu"] = _parse_usage_value(cells[cpu_index]) if cpu_index >= 0 and cpu_index < len(cells) else {"usage": 0, "limit": None}
        row["total_gpus"] = _parse_usage_value(cells[total_index]) if total_index >= 0 and total_index < len(cells) else {"usage": 0, "limit": None}
        rows.append(row)
    return rows


def _parse_user_usage(output: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metric_pattern = re.compile(r"(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*\((\d+)\)")
    for line in output.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < len(GPU_USAGE_COLUMNS) + 1 or cells[0] == "Username":
            continue
        metrics: dict[str, dict[str, int]] = {}
        for key, value in zip(GPU_USAGE_COLUMNS, cells[1:], strict=False):
            match = metric_pattern.fullmatch(value)
            if not match:
                break
            lab, overcap, scavenger, total = (int(part) for part in match.groups())
            metrics[key] = {
                "lab": lab,
                "overcap": overcap,
                "scavenger": scavenger,
                "total": total,
            }
        if len(metrics) == len(GPU_USAGE_COLUMNS):
            rows.append({"user": cells[0], "metrics": metrics})
    return rows


def _parse_elapsed_seconds(value: str) -> int | None:
    value = (value or "").strip()
    if not value:
        return None

    days = 0
    clock = value
    if "-" in value:
        day_part, clock = value.split("-", 1)
        if day_part.isdigit():
            days = int(day_part)
        else:
            return None

    chunks = clock.split(":")
    if len(chunks) == 1 and chunks[0].isdigit():
        seconds = int(chunks[0])
    elif len(chunks) == 2 and all(chunk.isdigit() for chunk in chunks):
        minutes = int(chunks[0])
        seconds = int(chunks[1]) + (minutes * 60)
    elif len(chunks) == 3 and all(chunk.isdigit() for chunk in chunks):
        hours = int(chunks[0])
        minutes = int(chunks[1])
        seconds = int(chunks[2]) + (hours * 3600) + (minutes * 60)
    else:
        return None

    return seconds + (days * 86400)


def _attach_account_users(
    rows: list[dict[str, object]],
    jobs: list[dict[str, object]],
    user_usage: list[dict[str, object]],
) -> None:
    allocations: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    max_elapsed: dict[str, dict[str, dict[str, dict[str, object]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    gpu_columns = set(GPU_USAGE_COLUMNS)
    for job in jobs:
        if job["state"] not in {"RUNNING", "COMPLETING"} or not job["gpus"]:
            continue
        partitions = {
            partition.strip()
            for partition in str(job["partition"]).split(",")
            if partition.strip()
        }
        account = OVERFLOW_ACCOUNT if partitions & OVERFLOW_PARTITIONS else str(job["account"])
        user = str(job["user"])
        gpu_count = int(job["gpus"]) * max(1, int(job["node_count"]))
        elapsed_seconds = _parse_elapsed_seconds(str(job.get("runtime", "")))
        runtime = str(job.get("runtime", "")).strip() or None
        allocations[account]["total_gpus"][user] += gpu_count
        gpu_model = str(job["gpu_model"] or "")
        if gpu_model in gpu_columns:
            allocations[account][gpu_model][user] += gpu_count
            if elapsed_seconds is not None and runtime:
                existing = max_elapsed[account][gpu_model].get(user)
                if existing is None or elapsed_seconds > existing.get("elapsed_seconds", -1):
                    max_elapsed[account][gpu_model][user] = {"elapsed_seconds": elapsed_seconds, "runtime": runtime}
        if elapsed_seconds is not None and runtime:
            existing_total = max_elapsed[account]["total_gpus"].get(user)
            if existing_total is None or elapsed_seconds > existing_total.get("elapsed_seconds", -1):
                max_elapsed[account]["total_gpus"][user] = {"elapsed_seconds": elapsed_seconds, "runtime": runtime}

    overcap_account = allocations[OVERFLOW_ACCOUNT]
    for column in (*sorted(gpu_columns), "total_gpus"):
        overcap_account[column].clear()
    for user_row in user_usage:
        username = str(user_row["user"])
        user_total = 0
        metrics = user_row["metrics"]
        if not isinstance(metrics, dict):
            continue
        for column in gpu_columns:
            metric = metrics.get(column)
            if not isinstance(metric, dict):
                continue
            count = int(metric["overcap"]) + int(metric["scavenger"])
            if count:
                overcap_account[column][username] = count
                user_total += count
        if user_total:
            overcap_account["total_gpus"][username] = user_total

    for row in rows:
        account = str(row["account"])
        for column in (*sorted(gpu_columns), "total_gpus"):
            metric = row.get(column)
            if not isinstance(metric, dict):
                continue
            column_elapsed = max_elapsed.get(account, {}).get(column, {})
            metric["users"] = [
                {
                    "name": user,
                    "usage": usage,
                    "elapsed": column_elapsed.get(user, {}).get("runtime"),
                }
                for user, usage in sorted(
                    allocations[account][column].items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ]


def _parse_snapshot(output: str, gateway: str) -> dict[str, object]:
    job_output, separator, remaining_output = output.partition("__SKYNET_JOBS__")
    if not separator:
        raise ClusterUnavailable("Slurm returned an unexpected response")
    usage_with_marker, _, user_usage_output = remaining_output.partition("__SKYNET_USER_USAGE__")
    usage_output = usage_with_marker.split("__SKYNET_USAGE__", 1)[-1] if "__SKYNET_USAGE__" in usage_with_marker else usage_with_marker

    jobs = [job for line in job_output.splitlines() if line.strip() if (job := _parse_job(line))]
    jobs.sort(key=lambda job: ({"RUNNING": 0, "COMPLETING": 1, "PENDING": 2}.get(str(job["state"]), 3), str(job["id"])))
    account_usage = _parse_account_usage(usage_output)
    _attach_account_users(account_usage, jobs, _parse_user_usage(user_usage_output))
    total_gpu_limit = 0
    total_gpu_allocated = 0
    for row in account_usage:
        metric = row.get("total_gpus")
        if not isinstance(metric, dict):
            continue
        total_gpu_allocated += int(metric.get("usage") or 0)
        limit = metric.get("limit")
        if isinstance(limit, int):
            total_gpu_limit += limit

    return {
        "gateway": gateway,
        "jobs": jobs,
        "account_usage": account_usage,
        "summary": {
            "gpu_total": total_gpu_limit,
            "gpu_allocated": total_gpu_allocated,
            "gpu_available": max(0, total_gpu_limit - total_gpu_allocated),
            "running_jobs": sum(job["state"] == "RUNNING" for job in jobs),
            "pending_jobs": sum(job["state"] == "PENDING" for job in jobs),
        },
    }


def _as_http_error(error: Exception) -> HTTPException:
    if isinstance(error, subprocess.TimeoutExpired):
        detail = "SSH operation timed out"
    else:
        detail = str(error)
    return HTTPException(status_code=503, detail=detail)


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"ok": True, "gateways": list(SSH_HOSTS)}


@app.get("/api/cluster")
def cluster(gateway: str = Query(default="auto")) -> dict[str, object]:
    try:
        active_gateway, output = _run_with_fallback(QUERY_COMMAND, gateway)
        return _parse_snapshot(output, active_gateway)
    except (ClusterUnavailable, subprocess.TimeoutExpired) as error:
        raise _as_http_error(error) from error


@app.post("/api/workspace/init")
def initialize_workspace(gateway: str = Query(default="auto")) -> dict[str, object]:
    try:
        active_gateway, _ = _run_with_fallback(INIT_COMMAND, gateway)
        return {"ok": True, "gateway": active_gateway, "work_root": WORK_ROOT}
    except (ClusterUnavailable, subprocess.TimeoutExpired) as error:
        raise _as_http_error(error) from error


@app.get("/", include_in_schema=False)
def index() -> HTMLResponse:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    for name in set(re.findall(r"/static/([A-Za-z0-9_.-]+\.(?:css|js))", html)):
        stat = (STATIC_ROOT / name).stat()
        version = f"{stat.st_mtime_ns:x}-{stat.st_size:x}"
        html = re.sub(rf'/static/{re.escape(name)}(?:\?[^"\s]*)?',
                      f"/static/{name}?v={version}", html)
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


app.include_router(pipeline_router)
app.include_router(collection_router)
app.include_router(local_capture_router)
app.include_router(hands_router)
app.include_router(live_xr_router)
app.include_router(policy_exports_router)
app.include_router(capture_processing_router)
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")
