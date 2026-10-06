"""Page fragments built from the cluster configuration, so forms offer exactly what the server accepts.

index() fills these placeholders in static/index.html; browser tests take the same
fragments from `python -m skynet_app.page_markup`.
"""
from __future__ import annotations

from html import escape
import json

from .cluster_config import CLUSTER
from .collection import CollectionResources


def _option(value: str, label: str, *, selected: bool = False, **data: str) -> str:
    attributes = "".join(f' data-{name.replace("_", "-")}="{escape(item)}"' for name, item in data.items())
    return f'<option value="{escape(value)}"{" selected" if selected else ""}{attributes}>{escape(label)}</option>'


def _queue_label(name: str) -> str:
    queue = CLUSTER.queues[name]
    return f"{'Preemptible' if queue.preemptible else name.capitalize()} / {queue.partition}"


def _gpu_label(name: str) -> str:
    return name.upper().replace("_", " ")


def cluster_markup() -> dict[str, str]:
    hosts = list(CLUSTER.gateways)
    queues = list(CLUSTER.queues)
    # Imports are CPU-only background work: they start on a preemptible queue when one exists.
    import_default = next((name for name in queues if CLUSTER.queues[name].preemptible), queues[0])
    collection = CollectionResources()
    return {
        "<!-- gateway-options -->": _option("auto", f"Auto: {', then '.join(hosts)}")
        + "".join(_option(host, f"Prefer {host}") for host in hosts),
        "<!-- queue-policy-options -->": _option("auto", "Auto", selected=CLUSTER.defaults.queue_policy == "auto")
        + "".join(_option(name, _queue_label(name), selected=name == CLUSTER.defaults.queue_policy) for name in queues),
        "<!-- import-queue-options -->": "".join(
            _option(name, _queue_label(name), selected=name == import_default) for name in queues),
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
