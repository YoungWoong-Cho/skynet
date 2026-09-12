# Evaluation compatibility

Training runs retain their original model configuration and checkpoint. The
evaluation picker lists every enabled suite, including incompatible combinations.
Selecting a suite does not submit a GPU job.

`evaluation_compatibility.inspect_compatibility` is the shared metadata check for
the picker, target validation and submission. It reports `compatible`,
`mapping_required`, `unknown`, or `incompatible`, with field-specific reasons.
Missing metadata is not treated as proof of compatibility. Actual checkpoint
availability is resolved through the run's registered checkpoints. Unsupported
submissions are rejected before creating an evaluation stage.

## Independent recorded-policy execution

The policy loader and suite executor are separate. EgoVerse ACT/HPT recorded-joint
policies and XPolicyLab ACT/DP policies share the recorded simulator executor.
They can use the registered `dexverse_recorded` suite or the single-episode suite
when its dataset conditions hold. Deleting a suite still removes it from the
picker; this module never registers or restores suites.

Policy inference uses the training receipt, source revision, normalization and
checkpoint. `policy_loading.py` loads the appropriate implementation, while
`dexverse_evaluation.py` owns environment reset, task scoring and rollout video.
`policy_contract.py` supplies both metadata checks and actual environment checks.
The evaluation capsule freezes the chosen loader, executor source and I/O report,
including an implementation hash. Retries retain that capsule.

Matching covers named joints, the policy-to-source permutation, raw joint-position
command semantics, action scales/offsets, control period and RGB camera dimensions.
The recorded format's translation/rotation semantics remain unchanged. Reordering
identical joint names is supported; arbitrary embodiment, unit or frequency
conversion is not invented. Such combinations require an explicit bridge.

Each worker loads the checkpoint, verifies its dataset hash and checks that the
on-disk I/O contract matches the submitted snapshot. It then resets the environment
and policy, reads a real observation, predicts an action chunk and advances one
simulator step. It resets again before any episode is scored or video recorded.
The unscored check is written to the worker's `preflight.json` and stdout; failures
stop evaluation. Normal rollout observations/actions remain checked. A passed
check establishes execution compatibility, not task success or generalization.
Isaac placement continues to use the existing grom/megazord policy.

## Other integrations

Existing native model/evaluator commands remain available through their versioned
adapter declarations and native runtime checks. They are not routed through the
recorded-joint protocol. No cross-model bridge is implied by merely having the
same tensor dimensions. Add a loader and a suite executor/observation-action
mapping before marking a new combination compatible.

Explicit custom commands remain an opt-in override, labeled unverified, and own
their compatibility checks. They retain suite, checkpoint and GPU-placement
validation. Nothing automatically executes a custom command while inspecting a
suite.
