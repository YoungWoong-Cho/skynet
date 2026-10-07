"""Page fragments built from the cluster configuration and the server's models, so forms offer exactly what the server accepts.

index() fills these placeholders in static/index.html; browser tests take the same
fragments from `python -m skynet_app.page_markup`.
"""
from __future__ import annotations

from html import escape
import json
from pathlib import PurePosixPath

from .cluster_config import CLUSTER, format_slurm_duration
from .collection import CollectionResources
from .data_imports import HuggingFaceImportRequest
from .experiments import DEFAULT_EVALUATION_EPISODES, MAX_EVALUATION_EPISODES, CheckpointPolicy, SweepSpec
from .hands_api import Pose
from .live_xr_archive import TERMINAL_STATES as LIVE_XR_TERMINAL_STATES
from .policy_exports import DEFAULT_SPLIT_SEED, DEFAULT_VALIDATION_PERCENT, MAX_VALIDATION_PERCENT, SPLIT_SEED_LIMIT
from . import retargeting
from .tracking import TRACKING_PROVIDERS
from .workspaces import EmailRequest


def _option(value: str, label: str, *, selected: bool = False, **data: str) -> str:
    attributes = "".join(f' data-{name.replace("_", "-")}="{escape(item)}"' for name, item in data.items())
    return f'<option value="{escape(value)}"{" selected" if selected else ""}{attributes}>{escape(label)}</option>'


def _queue_label(name: str) -> str:
    queue = CLUSTER.queues[name]
    return f"{'Preemptible' if queue.preemptible else name.capitalize()} / {queue.partition}"


def _gpu_label(name: str) -> str:
    return name.upper().replace("_", " ")


def _bound(model, name: str, key: str):
    """A field's default or an upper bound from its Field metadata."""
    field = model.model_fields[name]
    if key == "default":
        return field.get_default(call_default_factory=True)
    return next(getattr(item, key) for item in field.metadata if getattr(item, key, None) is not None)


def _queue_options(selected: str, *, auto: bool) -> str:
    options = [_option("auto", "Auto", selected=selected == "auto")] if auto else []
    options += [
        _option(name, _queue_label(name), selected=name == selected,
                max_time_seconds=str(queue.max_time_seconds), max_time=format_slurm_duration(queue.max_time_seconds))
        for name, queue in CLUSTER.queues.items()
    ]
    return "".join(options)


def cluster_markup() -> dict[str, str]:
    hosts = list(CLUSTER.gateways)
    defaults, limits = CLUSTER.defaults, CLUSTER.limits
    collection = CollectionResources()
    return {
        "<!-- default-time-limit -->": escape(defaults.time_limit),
        "<!-- default-memory-gb -->": str(defaults.memory_gb),
        "<!-- default-max-attempts -->": str(defaults.max_attempts),
        "<!-- cpus-per-gpu -->": str(defaults.cpus_per_gpu),
        "<!-- max-memory-gb -->": str(limits.max_memory_gb),
        "<!-- max-cpus-per-task -->": str(limits.max_cpus_per_task),
        "<!-- max-gpus-per-node -->": str(limits.max_gpus_per_node),
        "<!-- max-nodes -->": str(limits.max_nodes),
        "<!-- collection-time-limit -->": escape(collection.time_limit),
        "<!-- collection-memory-gb -->": str(collection.memory_gb),
        "<!-- import-time-limit -->": escape(_bound(HuggingFaceImportRequest, "time_limit", "default")),
        "<!-- import-cpus -->": str(_bound(HuggingFaceImportRequest, "cpus", "default")),
        "<!-- import-max-cpus -->": str(_bound(HuggingFaceImportRequest, "cpus", "le")),
        "<!-- import-memory-gb -->": str(_bound(HuggingFaceImportRequest, "memory_gb", "default")),
        "<!-- import-max-memory-gb -->": str(_bound(HuggingFaceImportRequest, "memory_gb", "le")),
        "<!-- checkpoint-warning-seconds -->": str(_bound(CheckpointPolicy, "save_before_timeout_seconds", "default")),
        "<!-- checkpoint-warning-min-seconds -->": str(_bound(CheckpointPolicy, "save_before_timeout_seconds", "ge")),
        "<!-- checkpoint-warning-max-seconds -->": str(_bound(CheckpointPolicy, "save_before_timeout_seconds", "le")),
        "<!-- checkpoint-max-attempts-limit -->": str(_bound(CheckpointPolicy, "max_attempts", "le")),
        # The sweep values the experiment form leaves implicit; a restored sweep with others cannot be shown.
        "<!-- sweep-defaults -->": escape(json.dumps(
            {name: _bound(SweepSpec, name, "default") for name in ("seeds", "max_parallel", "confirmation_threshold")})),
        "<!-- evaluation-episodes -->": str(DEFAULT_EVALUATION_EPISODES),
        "<!-- max-evaluation-episodes -->": str(MAX_EVALUATION_EPISODES),
        "<!-- pose-name-max-length -->": str(_bound(Pose, "name", "max_length")),
        "<!-- default-validation-percent -->": str(DEFAULT_VALIDATION_PERCENT),
        "<!-- max-validation-percent -->": str(MAX_VALIDATION_PERCENT),
        "<!-- default-split-seed -->": str(DEFAULT_SPLIT_SEED),
        "<!-- max-split-seed -->": str(SPLIT_SEED_LIMIT - 1),
        "<!-- live-xr-terminal-states -->": escape(json.dumps(sorted(LIVE_XR_TERMINAL_STATES))),
        "<!-- email-max-length -->": str(_bound(EmailRequest, "email", "max_length")),
        "<!-- default-retargeter -->": escape(retargeting.DEFAULT),
        "<!-- default-retargeter-option -->": _option(
            retargeting.DEFAULT, next(method["name"] for method in retargeting.METHODS if method["key"] == retargeting.DEFAULT)),
        "<!-- tracking-providers -->": escape(json.dumps(TRACKING_PROVIDERS)),
        "<!-- artifacts-path-example -->": escape(f"{CLUSTER.paths.artifacts}/.../last.ckpt"),
        "<!-- collection-output-example -->": escape(f"{CLUSTER.paths.datasets}/.staging/collection/..."),
        "<!-- data-version-path-example -->": escape(f"{CLUSTER.paths.datasets}/resources/..."),
        "<!-- work-root-example -->": escape(str(PurePosixPath(CLUSTER.paths.work_root).parent / "yourname")),
        "<!-- gateway-options -->": _option("auto", f"Auto: {', then '.join(hosts)}")
        + "".join(_option(host, f"Prefer {host}") for host in hosts),
        "<!-- queue-policy-options -->": _queue_options(defaults.queue_policy, auto=True),
        "<!-- import-queue-options -->": _queue_options(defaults.import_queue_policy, auto=False),
        # The alias without a target means any compatible GPU; evaluation forms leave it out.
        "<!-- gpu-type-options -->": "".join(
            _option(alias, "Any compatible" if target is None else _gpu_label(target),
                    selected=alias == CLUSTER.defaults.gpu_type, **({"any_gpu": "true"} if target is None else {}))
            for alias, target in CLUSTER.gpu_aliases.items()),
        "<!-- gpu-usage-headers -->": "".join(
            f'<th data-gpu-column="{escape(column)}">{escape(_gpu_label(column))}</th>'
            for column in CLUSTER.dashboard.gpu_usage_columns),
        "<!-- collection-account -->": escape(collection.account),
        "<!-- collection-partition -->": escape(collection.partition),
        "<!-- collection-gpu-type -->": escape(collection.gpu_type or ""),
    }


if __name__ == "__main__":
    print(json.dumps(cluster_markup()))
