"""Cluster placement policy shared by evaluation planning and Slurm submission."""

from collections.abc import Mapping
from typing import Any

from .cluster_config import CLUSTER
from .experiments import ResourceSpec


def uses_isaac_sim(
    context: Mapping[str, Any], *, runtime_profile_id: str | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> bool:
    """Use evaluator declarations and runtime evidence, not policy/adapter names."""
    suite = context.get("suite") or {}
    config = suite.get("config") or {}
    if any(value in {"isaac_sim", "isaac_lab"} for value in (
        context.get("environment"), config.get("evaluator"),
        (context.get("evaluator") or {}).get("adapter"),
    )):
        return True
    runtimes = [runtime or {}, context.get("evaluator_runtime") or {}]
    if runtime_profile_id:
        runtimes.append(CLUSTER.runtime_profile_snapshot(runtime_profile_id))
    for candidate in runtimes:
        snapshot = candidate.get("profile_snapshot") or candidate
        versions = snapshot.get("versions") or {}
        distributions = (snapshot.get("verification") or {}).get("distributions") or {}
        if "isaac_sim" in versions or "isaac_lab" in versions or "isaacsim" in distributions:
            return True
    return False


def resolve_evaluation_resources(
    resources: ResourceSpec, context: Mapping[str, Any], *,
    runtime_profile_id: str | None = None, runtime: Mapping[str, Any] | None = None,
    gpu_count: int | None = None, gpu_type: str | None = None,
) -> ResourceSpec:
    if not uses_isaac_sim(context, runtime_profile_id=runtime_profile_id, runtime=runtime):
        return resources
    policy = CLUSTER.isaac_evaluation_placement
    if policy is None:
        raise ValueError("Isaac Sim evaluation requires configured compatible nodes; submission is blocked")
    requested_type = gpu_type or resources.gpu.gpu_type
    count = gpu_count if gpu_count is not None else resources.gpu.count
    if count is None:
        raise ValueError("Resolve the Isaac Sim evaluation GPU count before choosing a node")
    allowed = ", ".join(policy.nodes)
    if resources.node.mode == "manual":
        name = resources.node.name
        if name not in policy.nodes:
            raise ValueError(f"Isaac Sim evaluations may run only on {allowed}; node {name} is not allowed")
        if requested_type not in {"any", policy.nodes[name].gpu_type}:
            raise ValueError(f"Isaac Sim evaluation node {name} requires {policy.nodes[name].gpu_type} GPUs")
    elif requested_type == "any":
        name = policy.default_node
    else:
        name = next((name for name, node in policy.nodes.items() if node.gpu_type == requested_type), None)
        if name is None:
            raise ValueError(f"Isaac Sim evaluations have no compatible {requested_type} node; allowed nodes: {allowed}")
    node = policy.nodes[name]
    if count > node.gpu_count:
        raise ValueError(f"Isaac Sim evaluation node {name} supports at most {node.gpu_count} GPUs")
    # One concrete node is intentional: a Slurm nodelist of two nodes requests
    # both nodes, rather than allowing the scheduler to choose either one.
    document = resources.model_dump(mode="json", by_alias=True)
    document["node"] = {"mode": "manual", "name": name}
    document["gpu"] = {"mode": "explicit", "count": count, "type": node.gpu_type}
    return ResourceSpec.model_validate(document)
