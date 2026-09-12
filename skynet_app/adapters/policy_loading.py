"""Policy-side loaders; independent of suite, task selection and simulator."""


def load_policy(context, source_dir, manifest):
    loader = context["compatibility"]["policy_loader"]
    if loader == "egoverse_joints":
        from egoverse_simulation import RecordedPolicy
        return RecordedPolicy(context, source_dir)
    if loader == "xpolicy_joints":
        from xpolicy_evaluation import RecordedPolicy
        return RecordedPolicy(context, source_dir, manifest)
    raise ValueError(f"Unknown policy loader: {loader}")
