# Skynet Training and Evaluation Console

Skynet is a web application for a trusted team for inspecting the live Slurm cluster and managing reproducible training and evaluation workflows. It provides GPU/account usage, node and queue views, a versioned Adapter Registry, exact-commit repository inspection, canonical experiment specifications, deterministic `sbatch` generation, submission and retry history, evaluation progress, logs, artifacts, and optional W&B and MLflow synchronization.

The application talks to the cluster through the `sky1` and `sky2` SSH login aliases. It does not install training repositories or simulator runtimes for you.

## Safety boundary

Run the service on a trusted workstation or server. Use a loopback binding with an SSH tunnel, or restrict direct network access to the trusted team. The application can submit and cancel Slurm jobs and execute adapter-defined workloads as your cluster user. Adapter manifests use structured argv rather than shell command strings, but a manifest can still select any executable available to that user. Registry authors are therefore trusted operators. Email workspaces separate saved experiment configurations, custom training adapters, repository selections, tracking credentials, training runs and evaluations. Emails are deliberately unverified: anyone who enters an email can open its workspace. This is workspace organization, not identity authentication or a cluster execution sandbox. All workloads still use the configured cluster SSH account.

```bash
uv run uvicorn skynet_app.main:app --host 127.0.0.1 --port 8080
```

If the browser is on another machine, use an SSH tunnel or restrict direct access to your trusted team network. See [Linux team deployment](docs/deployment.md) for the persistent service, workspace cutover and operations guide.

## Prerequisites

- Python 3.11 or newer on the machine running the web application.
- [`uv`](https://docs.astral.sh/uv/) locally. Repository runtime profiles may use `uv`, Conda, or a site-approved container.
- Non-interactive SSH authentication for at least one of `sky1` or `sky2`.
- Access to `/coc/flash7/ycho420` from the Slurm login and compute nodes.
- Slurm commands at `/opt/slurm/Ubuntu-20.04/current/bin` on the login nodes.
- The cluster `gpu_usage` utility at `/coc/testnvme/admin/tools/skynet-utilities/gpu_usage`.
- Pinned training repositories, datasets, checkpoints, simulator assets, and any gated model credentials required by the selected adapter.
- A framework-specific checkpoint/resume command for any workload that should auto-resume.
- A structured evaluator argv for evaluations that do not yet have a built-in adapter command.
- An operator-reviewed cluster profile. The included profile is `config/clusters/skynet.json`.

Confirm SSH before starting the app:

```bash
ssh -T -o BatchMode=yes sky1 'hostname; squeue -h | head'
ssh -T -o BatchMode=yes sky2 'hostname; squeue -h | head'
```

Only one alias needs to work. The second command is useful for confirming fallback behavior.

## Install, run, and test

Install the application and development dependencies:

```bash
uv sync --group dev
```

Start the local server. Restrict development reloads to source folders so saved conversion workers and run capsules do not restart active jobs:

```bash
export SKYNET_SSH_HOSTS=sky1,sky2
export SKYNET_DATABASE_PATH="$PWD/data/skynet.db"
uv run uvicorn skynet_app.main:app --reload --reload-dir skynet_app --reload-dir ops --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080` and enter your email. No password or email verification is required. See [email workspaces](docs/email-workspaces.md) for migration and shared-data behavior. The generated API documentation is available at `http://127.0.0.1:8080/api/docs`.

Run the test suite:

```bash
uv run pytest
npm ci
npm run test:live
npm run test:hands
npm run test:shared
```

The tests use temporary databases and mocked transports where appropriate. Cluster integration tests require active SSH access and submit real Slurm jobs, so review their markers and payloads before running them.

## Slack notifications

Open **Settings → Slack notifications** to connect a personal Slack webhook and receive submission, start, cancellation, failure and completion updates for your training and evaluation jobs. Settings and delivery queues are isolated by email workspace. See [Slack setup and delivery behavior](docs/slack-notifications.md).

## UI workflow

1. Open the cluster dashboard and select one or more partitions to filter the job queue.
2. Confirm the account GPU usage and limits, GPU users, pending reasons, and gateway health.
3. Open **Experiments → Adapters** to review, clone, edit, or validate the versioned repository adapter that will compile the workload.
4. Open **Experiments**, select a repository branch and exact commit, then choose automatic or explicit runtime resolution and provide resources, hyperparameters, checkpoint policy, sweep axes, and evaluation plan.
5. Preview the resolved variants and canonical `sbatch` before submission.
6. Submit the experiment. A logical run may contain multiple Slurm attempts after preemption, timeout, or node failure.
7. Open a run to inspect its stages, exact argv, generated script, checkpoint lineage, stdout, stderr, events, and artifacts.
8. Use **Resume** for an interrupted resumable run, **Cancel** for active work, or fork the experiment revision when changing scientific inputs.
9. Create evaluations from a retained inference checkpoint and monitor the episode ledger, logs, videos, and normalized result.

For multi-GPU training, choose **Experiments → Slurm resources → GPU allocation → Manual**
and set **GPUs / node**. DP, ACT and DexMimicGen support up to eight GPUs on one
node. Each GPU runs a separate DDP worker; per-device batch size is multiplied
by the GPU count before accumulation. Losses and gradients are weighted by the
actual sample count, including uneven final batches. Only the main worker
writes logs and checkpoints, which remain compatible with single-GPU evaluation.
DexMimicGen uses the native robomimic BC family and keeps its configured global
batch size. Other adapters retain their native launchers; custom commands must
use the allocation exposed through `SKYNET_ASSIGNED_GPU_COUNT`.

The Skynet cluster profile pins `NCCL_P2P_DISABLE=1` into new experiment runtimes
to avoid unreliable direct peer transfers on this cluster. NCCL uses host-memory
communication within the node. Other cluster profiles may leave this unset.
The shared trainers validate collective communication before loading the policy
and fail if loss values or sample counts are invalid. Existing experiment revisions
retain their pinned code; create a new revision to use an updated trainer.

For DP and ACT, open **Training Runs → View attempts → Start evaluation**. In
**Evaluations**, choose one or more configured DexVerse tasks, episodes, seeds,
and up to eight parallel jobs. The recorded task is the default. Choose the
queue, GPU type, CPUs and RAM per worker, and wall time; the total allocation is
shown before submission. Parallel workers occupy separate GPUs in one Slurm
allocation, so two workers request two GPUs. Completed episodes and their videos
are retained across automatic attempts.

**Results** opens immediately. Its **Rollout videos** table lists every requested
episode; **Detail** opens the shared modal with that episode's video, canonical
results, Slurm attempt, stdout and stderr. Logs load independently of the panel.
Changing evaluation tasks measures transfer to those tasks; it does not change
the trained policy. Robot joint layout, control frequency and checkpoint/data
provenance must still match.

Edits create a new immutable experiment revision. Previously submitted variants and runs continue to reference the configuration under which they were created.

## Canonical experiment model

The canonical specification separates scientific inputs from scheduler placement:

```yaml
apiVersion: skynet.rl2/v1
kind: Experiment
identity:
  project: manipulation
  experiment: encoder-comparison
source:
  repository: https://github.com/NVIDIA/Isaac-GR00T
  revision: <full-commit-sha>
  adapter: groot
runtime:
  backend: uv
  profile: groot-uv
  lock_file: uv.lock
  lock_sha256: <sha256-of-uv.lock>
train:
  learning_rate: 0.0001
  batch:
    declared_semantics: per_device
    value: 16
    gradient_accumulation_steps: 2
  num_workers_per_rank: 8
  seed: 42
  checkpoint:
    save_every_steps: 1000
    keep_last: 3
    auto_resume: true
    max_attempts: 5
resources:
  gateway: auto
  queue_policy: normal
  account: rl2-lab
  partition: rl2-lab
  nodes: 1
  node:
    mode: auto
  gpu:
    mode: explicit
    count: 4
    type: any
  cpus_per_task: 32
  memory_gb: 128
  time_limit: "04:00:00"
```

This example is illustrative. Adapter capability schemas determine which fields are accepted and what their batch/step semantics mean. Use the UI preview or OpenAPI schema for the exact installed adapter version. Unsupported canonical fields fail validation rather than being silently discarded; repository-specific settings belong in namespaced native overrides.

## Operator cluster profile

Cluster topology and policy are data, not adapter behavior. The default operator-managed profile is [`config/clusters/skynet.json`](config/clusters/skynet.json). Set `SKYNET_CLUSTER_CONFIG=/absolute/path/to/profile.json` to load another profile, then restart the server. There is deliberately no API that edits this file.

The profile defines gateways, filesystem roots, Slurm and `gpu_usage` command paths, an optional deterministic `gpu_usage_interpreter`, atomic partition/account queue pairs, time and preemption policy, GPU aliases, dashboard columns, application defaults, and hard resource limits. It is schema-validated at startup; unknown fields, duplicate gateways, invalid default references, and duplicate partition/account pairs fail closed.

The following environment variables can override deployment-specific values without changing application code:

- `SKYNET_SSH_HOSTS`
- `SKYNET_HOME_ROOT`
- `SKYNET_WORK_ROOT`
- `SKYNET_SLURM_BIN`
- `SKYNET_GPU_USAGE_COMMAND`
- `SKYNET_GPU_USAGE_INTERPRETER`
- `SKYNET_QUEUE_<POLICY>_PARTITION`
- `SKYNET_QUEUE_<POLICY>_ACCOUNT`
- `SKYNET_QUEUE_<POLICY>_MAX_TIME_SECONDS`

The active, non-secret profile is returned by `GET /api/settings` and `GET /api/capabilities`. Do not put credentials in the profile because these endpoints expose it to the browser.

## Versioned Adapter Registry

The application seeds the six experiment adapters into SQLite, then treats adapters as database records rather than UI hardcoding. An adapter manifest declares repository matching metadata, allowed and recommended runtimes, capabilities, defaults, a structured training command template, checkpoint/resume behavior, evaluation metadata, warnings, and TODOs.

The DexVerse experiment adapter is retired and existing seeded records are archived on startup. Its historical manifest handler remains available for saved experiments. DexVerse simulation and the separate `dexverse-cloudxr` collection registry are unaffected.

Registry lifecycle semantics are:

- **Create** starts an independent adapter at version 1.
- **Edit** appends a new immutable version; it never mutates an older version. `expected_latest_version` provides optimistic conflict detection.
- **Clone** copies any selected version into a new adapter at version 1 and records its origin.
- **Validate** schema-checks the manifest and may inspect a selected exact repository commit. Validation does not execute repository code or launch training.
- **Archive** hides the adapter from normal selection while retaining every version and historical reference. `DELETE` is an archive alias, not permanent deletion.
- **Restore** makes an archived adapter selectable again.

Created experiment revisions embed the selected manifest and its SHA-256. Later registry edits or archival therefore cannot change an existing experiment or submitted run.

Non-runtime defaults fill only fields omitted by the request. Precedence is explicit experiment input, adapter manifest defaults, then operator cluster defaults. A runtime recommendation is only a recommendation and follows the stricter resolution rules in the next section.

The complete JSON Schema is returned as `adapter_manifest_schema` by `GET /api/capabilities`. A minimal declarative example is:

```json
{
  "schema_version": "skynet.adapter/v1",
  "slug": "my_policy",
  "display_name": "My Policy",
  "description": "Canonical trainer for my policy repository",
  "repository_patterns": ["https://github.com/example/my-policy"],
  "default_repository": "https://github.com/example/my-policy",
  "runtime": {
    "allowed_backends": ["uv", "conda", "apptainer"],
    "recommended_backend": "uv"
  },
  "capabilities": {
    "name": "my_policy",
    "version": 1,
    "runtime_backends": ["uv", "conda", "apptainer"],
    "supports_multi_gpu_single_node": true,
    "supports_resume": true,
    "minimum_gpus": 1,
    "recommended_gpus": 1,
    "maximum_gpus": 8
  },
  "defaults": {
    "workdir": ".",
    "hyperparameters": {
      "learning_rate": 0.0001,
      "batch_size": 16,
      "gradient_accumulation_steps": 1,
      "num_workers_per_rank": 8,
      "seed": 42
    },
    "resources": {
      "gpu_mode": "auto",
      "gpu_profile": "recommended"
    },
    "checkpoint": {
      "save_every_steps": 1000,
      "save_before_timeout_seconds": 300,
      "keep_last": 3,
      "auto_resume": true,
      "max_attempts": 5,
      "final_selector": "latest",
      "remove_training_state_after_success": true
    }
  },
  "train": {
    "argv": ["python", "train.py"],
    "parameter_flags": {
      "train.learning_rate": {"flag": "--learning-rate", "style": "separate"},
      "train.batch.value": {"flag": "--batch-size", "style": "separate"}
    },
    "resume_argv": ["--resume", "{{tokens.resume_checkpoint}}"],
    "checkpoint_globs": ["checkpoints/*.pt"],
    "required_values": ["train.learning_rate", "train.batch.value"]
  },
  "evaluations": []
}
```

`runtime.allowed_backends` must equal `capabilities.runtime_backends`, and `capabilities.name` must equal `slug`. Templates support canonical value placeholders and explicit flag bindings; they are not arbitrary Python plugins. Built-in compatibility manifests may delegate to their versioned legacy handler, while new adapters can remain fully declarative.

## Repository runtime inspection

Automatic runtime selection inspects metadata at the resolved full commit and selected project subdirectory without importing or executing repository code. Branches and tags are resolved to a commit SHA before the experiment is stored.

Resolution precedence is:

1. An explicitly selected `uv`, `conda`, `apptainer`, or `existing` backend wins, must be allowed by the adapter, and records that repository inspection was skipped.
2. In `auto` mode, a valid `.skynet.json` or `.skynet.toml` `runtime` object is preferred.
3. Otherwise, exactly one strong runnable dependency-file candidate is selected.
4. Multiple strong backends, no strong backend, or missing required runtime data fails validation and requires an explicit choice.

Adapter `recommended_backend` is included in the evidence and error message but is never a silent fallback. The application also never selects an existing environment implicitly.

Strong automatic evidence currently means `uv.lock` with `pyproject.toml`, or `conda-lock.yml`/`conda-lock.yaml`. `pyproject.toml` without `uv.lock`, `environment.yml`, requirements files, and container definition files are weak evidence and require operator input. `README.md` and `README.rst` are not parsed into commands.

Only `.skynet.json` and `.skynet.toml` are machine-applied repository runtime metadata. `.skynet.yml` and `.skynet.yaml` are advisory and produce a warning. A repository runtime file may declare only runtime configuration, for example:

```toml
[runtime]
backend = "uv"
lock_file = "uv.lock"
bootstrap_uv = true
uv_version = "0.8.14"
```

For Conda, automatic resolution requires a lock file or an explicit deterministic environment path. For Apptainer, a definition file is not an image: provide a built immutable `container_image` and declare its `container_digest`. The current implementation records and syntax-validates that digest but does not remotely fetch the image and verify its bytes against the declaration.

## API workflow

The UI uses the same JSON API exposed to automation:

- `GET /api/settings` returns non-secret application and active cluster-profile settings.
- `GET /api/capabilities` returns the adapter manifest schema, experiment schema, runtime backends, cluster profile, and safety limits.
- `GET /api/adapters?include_archived=true` lists registry entries; `POST /api/adapters` creates one.
- `GET /api/adapters/{id}` returns version history; `PUT` or `PATCH` appends a version.
- `POST /api/adapters/{id}/clone`, `/archive`, and `/restore` implement lifecycle operations; `DELETE /api/adapters/{id}` also archives.
- `POST /api/adapters/validate` validates an unsaved manifest. `POST /api/adapters/{id}/validate` records validation for a stored version, and `GET /api/adapters/{id}/validations` returns its reports.
- `GET /api/source/branches`, `/api/source/commits`, and `/api/source/inspect` expose exact repository discovery and runtime evidence.
- `GET /api/evaluation-suites` lists installed evaluation catalogs.
- `POST /api/experiments/preview` resolves and validates variants without submission.
- `GET /api/experiments` and `POST /api/experiments` list and create immutable experiment revisions.
- `GET /api/experiments/{id}` returns experiment lineage and variants.
- `POST /api/experiments/{id}/submit` compiles and submits the selected revision.
- `GET /api/runs` and `GET /api/runs/{id}` expose logical runs, stages, attempts, checkpoints, and artifacts.
- `GET /api/runs/{id}/logs` exposes bounded stdout/stderr reads for failure diagnosis.
- Run cancellation and resume endpoints call Slurm and preserve attempt history.
- Evaluation list, create, and submit endpoints live under `/api/evaluations`.
- Cluster, workspace, and direct job endpoints remain available for dashboard and low-level operation.

Use `/api/docs` for request schemas and the precise cancel/resume/evaluation route suffixes.

## Real Slurm behavior

There is one Slurm cluster behind two interchangeable login gateways:

- `gateway=auto` tries `sky1`, then `sky2`.
- Selecting a gateway tries it first and then falls back to the other alias if SSH cannot be established.
- Cluster reads may be retried safely on either gateway.
- Submission resolves one healthy gateway and invokes `sbatch --parsable` once. It does not retry an ambiguous submission on a second host, which avoids accidental duplicate jobs.
- Automatic node placement omits `#SBATCH --nodelist`; manual placement validates and emits the selected node expression.
- Jobs are restricted to one node but may use multiple same-type GPUs on that node.
- Mixed allocations such as A40 GPUs plus L40S GPUs and multi-node distributed jobs are intentionally disabled.
- Slurm remains the source of truth for scheduling, placement, preemption, exit state, and pending reasons.

Account and partition must be selected as an atomic pair:

| Scheduling mode | `--partition` | `--account` | Behavior |
| --- | --- | --- | --- |
| Normal | `rl2-lab` | `rl2-lab` | Uses the lab allocation |
| Preemptible | `overcap` | `overcap` | May run above the lab cap and may be preempted |

`rl2-lab` requests are limited to `04:00:00`; `overcap` requests may run for up to `2-00:00:00` and remain preemptible. Long-running workloads should checkpoint and resume across attempts.

Generated jobs use separate output files:

```bash
#SBATCH --chdir=/coc/flash7/ycho420/workspace
#SBATCH --output=/coc/flash7/ycho420/logs/%x-%j.out
#SBATCH --error=/coc/flash7/ycho420/logs/%x-%j.err
```

The application preserves every failed attempt's script, exit status, Slurm reason, stdout, and stderr. A resumed job is a new attempt under the same logical run.

## Cluster paths

The workspace initializer creates:

```text
/coc/flash7/ycho420/
|-- workspace/
|-- repos/
|-- datasets/
|   |-- resources/
|   |-- derivatives/
|   |-- bundles/
|   `-- .staging/
|-- artifacts/
|-- logs/
|-- jobs/
|-- eval/
|   |-- catalogs/
|   `-- runs/
`-- .cache/
    |-- uv/
    |-- huggingface/
    `-- torch/
```

Jobs receive:

```bash
export HOME=/nethome/ycho420
export WORK_ROOT=/coc/flash7/ycho420
export UV_CACHE_DIR="$WORK_ROOT/.cache/uv"
export HF_HOME="$WORK_ROOT/.cache/huggingface"
export TORCH_HOME="$WORK_ROOT/.cache/torch"
```

`uv` is preferred for ordinary Python repositories. Isaac Gym, Isaac Sim, and Isaac Lab workloads may require a pinned Conda environment or immutable Apptainer/container runtime instead.

## Canonical dataset store

Dataset storage is source-centric and independent of any training framework or simulator:

```text
$WORK_ROOT/datasets/
|-- resources/<provider>/<namespace>/<name>/revisions/<exact-revision>/
|   |-- manifest.json
|   |-- inventory.sha256
|   |-- data/
|   `-- READY
|-- derivatives/<resource-id>/<format>/<derivation-id>/
|   |-- manifest.json
|   |-- inventory.sha256
|   |-- data/
|   `-- READY
|-- bundles/<bundle-id>/versions/<version>/
|   |-- manifest.json
|   |-- view/
|   `-- READY
`-- .staging/
```

`resources` contains exact upstream revisions or an explicitly declared subset of a revision. Provider, namespace, name, full source revision, format, license status, payload inventory, byte count, and provenance belong in the resource manifest. A resource's `kind`, such as `demonstrations` or `simulation_assets`, describes its contents; it does not hard-code whether an experiment uses that resource for training or evaluation.

`derivatives` contains deterministic transformations and never replaces its source. For example, raw DexVerse trajectory pickle or HDF5 data may be converted into LeRobot format. The derivative manifest must pin the source resource URI and inventory hash, converter repository and full commit, converter configuration, runtime lock or image digest, output format version, and output inventory hash. LeRobot data is therefore a derived representation with traceable lineage, not a second unversioned copy treated as the source.

`bundles` assigns immutable resources and derivatives to experiment roles such as `simulation_assets`, `training_data`, and `evaluation_data`. A bundle may provide a codebase-specific symlink view without copying payload bytes. The DexVerse asset bundle, for example, assigns both `DexVerse_release` and the selected `ManiTwin-100K` subset to `simulation_assets`, then mounts them under the paths expected by DexVerse. Another bundle can reuse either resource without duplicating it.

`.staging` is the only location for incomplete downloads and conversions. Publishers write data and inventories there, then atomically publish a new revision. A resource, derivative, or bundle is consumable only when `READY` contains a valid SHA-256 for its immutable `manifest.json`. Publishers must never overwrite a published revision, and jobs must never resolve mutable `current` or `latest` aliases.

The pinned DexVerse asset publisher is [`ops/datasets/download_dexverse_assets.sbatch`](ops/datasets/download_dexverse_assets.sbatch). It publishes independent Hugging Face resources and a reference-only bundle; it does not create the previous combined `datasets/dexverse/snapshots` layout.

An experiment revision must store exact data identities rather than only filesystem paths:

```yaml
data:
  bundle:
    uri: bundle:dexverse/simulation-assets@release-<full-release-revision>--manitwin-<full-manitwin-revision>
    manifest_sha256: <bundle-manifest-sha256>
  inputs:
    simulation_assets:
      - uri: resource:huggingface/dexverse/DexVerse_release@<full-revision>
        manifest_sha256: <resource-manifest-sha256>
        inventory_sha256: <payload-inventory-sha256>
    training_data:
      - uri: derivative:<resource-id>/lerobot-v3@<derivation-id>
        manifest_sha256: <derivative-manifest-sha256>
        inventory_sha256: <derived-payload-inventory-sha256>
```

The run capsule records these exact URIs and hashes along with the code commit, adapter version, runtime lock, generated configuration, checkpoint lineage, and evaluation suite version. A strict replay refuses a missing manifest or any manifest/inventory hash mismatch.

## Database and artifact storage

The application database defaults to `data/skynet.db` beside the source tree and can be overridden with `SKYNET_DATABASE_PATH`. Keep SQLite on the application host, not on a network filesystem. The schema is designed to migrate to PostgreSQL when concurrent users or multiple server processes are needed.

SQLite stores experiment lineage, run/stage/attempt state, evaluation progress, metadata, and file indexes. Large checkpoints, logs, videos, native configs, and result files stay below `/coc/flash7/ycho420`. Run capsules remain sufficient to audit and reconstruct application records if the local database is lost.

Adapter registry rows store immutable numbered manifest versions, manifest hashes, clone provenance, archive state, and exact-commit validation reports. Archiving never removes versions referenced by experiments.

MLflow uses its own database. The application never writes directly to MLflow's internal tables.

## Checkpoints, retention, and resume

- `save_every_steps` is interpreted by the repository adapter in optimizer-step, epoch, or iteration units as explicitly reported by that adapter.
- During training, retain the three newest validated full-resume checkpoints by default.
- A full-resume checkpoint should include model, optimizer, scheduler, scaler, RNG, and framework state where supported.
- Auto-resume creates a new Slurm attempt for `PREEMPTED`, `TIMEOUT`, `NODE_FAIL`, and declared transient failures.
- Syntax errors, invalid configs, OOM, and user cancellation are not blindly retried.
- After successful training, keep only the selected policy checkpoint. Training-state cleanup is enabled by default (`train.checkpoint.remove_training_state_after_success`). OpenPI retains `params` and `assets` and removes `train_state`. GR00T retains its checkpoint's weights, configuration, processor, normalization statistics, and embodiment map; it removes optimizer/scheduler/RNG state and the trainer's redundant export at the root of `artifacts`.
- Adapter contracts declare checkpoint-relative `checkpoint_prune_globs` and run-relative `training_output_prune_globs`. Cleanup validates inference files and every indexed model shard before deleting anything, rejects paths outside the run or overlapping the retained policy, then writes the final checkpoint path, digest, and cleanup receipt to `checkpoints/selected-for-inference.json`. The app registers this final identity for later evaluation. Unknown checkpoint formats are not unpacked or modified speculatively.
- Logs, videos, failure records, and checkpoint lineage remain retained.
- Evaluation resumes from its episode ledger. Completed `(checkpoint, suite, version, task, seed, episode_index)` entries are skipped and incomplete episodes restart.

Auto-resume is only as complete as the adapter's checkpoint and resume implementation. The application cannot infer a safe framework-specific resume flag from an arbitrary Python command.

## Evaluation commands and data

Immutable evaluation inputs and simulator assets use the canonical dataset store under `$WORK_ROOT/datasets`. Evaluation task catalogs, progress ledgers, results, logs, and videos remain below `$WORK_ROOT/eval`; bundle manifests assign selected dataset resources to the `evaluation_data` or `simulation_assets` role.

Each evaluation selects an evaluator, immutable suite version, compatible tasks, checkpoint, episode count, seeds, horizon, video policy, and resources. Until a suite adapter defines its own launch command, the request must provide `argv` as a structured JSON array in the UI, for example:

```json
["uv", "run", "python", "evaluate.py", "--suite", "libero_10"]
```

Do not provide a shell command string. Structured argv preserves exact argument boundaries, avoids implicit shell expansion, and is recorded in the run capsule. The application does not fabricate a placeholder evaluation command. An evaluation without a built-in adapter command or `argv` is saved as `BLOCKED` and is not submitted. Evaluators receive result/progress paths plus suite, task, seed, episode-count, and checkpoint metadata through `SKYNET_EVAL_*`. A successful job must write a schema-valid canonical result with the complete episode ledger; malformed or incomplete results fail visibly.

## Reproducibility capsule

Before persistence, a symbolic source revision is resolved to a full commit, the selected adapter version is embedded as a complete hashed manifest, and runtime resolution is embedded with its inspection evidence and hash. The resolved experiment specification is therefore independent of later branch movement, registry edits, or adapter archival. An explicit runtime choice records that automatic inspection was skipped rather than pretending it was inferred.

The operator cluster profile is not copied verbatim into a separate capsule file. Its concrete effects are captured in the resolved resource request and generated `sbatch`, while the attempt records its gateway, partition, account, allocation, and live cluster snapshot. Preserve the profile file itself if operator-policy history must be reconstructed independently.

Every submitted run records an immutable capsule containing, when applicable:

- Requested and fully resolved experiment specifications and their hashes.
- Adapter name/version and generated native configuration.
- Exact argv and canonical `sbatch` script.
- Repository commit, submodules, Git LFS revisions, and captured dirty patch.
- Runtime lock or container digest and package/environment manifest.
- Exact dataset resource, derivative, and bundle URIs plus manifest and inventory hashes; normalization, asset, and initial-checkpoint identifiers and hashes.
- Seeds, determinism settings, checkpoint lineage, and evaluation episode keys.
- Slurm job/account/partition/node/allocation data and every resolved `auto` choice.
- GPU, driver, CUDA, framework, simulator, OS, and CPU metadata captured on-node.
- The exact selected Python executable and installed distribution versions in `runtime-manifest.json`.
- A final `capsule-manifest.json` containing SHA-256 and size for every reproducibility metadata file.
- Logs, selected checkpoints, raw results, canonical results, videos, and checksums.

Every Slurm attempt is archived independently under `$WORK_ROOT/jobs/runs/<run-id>/attempts/<slurm-job-id>/`. Later training retries or evaluation stages therefore cannot overwrite an earlier attempt's script or provenance capsule.

Replay modes distinguish strict replay, compatible replay, and an explicit fork. Strict replay refuses missing or changed immutable inputs. This guarantees reconstruction of the submitted inputs and execution envelope; it does not promise bitwise-identical model weights across different hardware, framework releases, kernels, or nondeterministic simulators.

Secret values are never written to the capsule. Only secret references may be recorded.

## W&B and MLflow setup

Central tracking is optional. Configure either provider under **Settings**, test the connection, and enable it in an experiment. Skynet derives the remote project/experiment from the Skynet experiment and uses the Skynet run UUID as the remote run identity. Remote links and delivery status then appear on experiment and run views.

Secrets entered in Settings are saved by default in the operating-system credential manager (`macOS Keychain`, Windows Credential Locker, or a Linux Secret Service/KWallet backend). They are never stored in SQLite, plaintext files, experiment specifications, `sbatch` files, run capsules, logs, or API responses. A connection can opt out with `remember: false`, which keeps the credential only in backend process memory. Environment credentials remain supported and are never copied into the credential manager.

The tracking connection API contract is:

- `POST /api/tracking/connections/{provider}/connect` accepts `remember` (default `true`) alongside the provider fields. A remembered credential is written only after remote validation succeeds.
- `GET /api/tracking/connections` and all connection mutation responses expose only nonsecret connection metadata. `credential_source` is `credential_store`, `session`, `environment`, or `null`; `status` reports the connection state. Secret values are never returned.
- On restart, Skynet restores a remembered credential only when its credential-manager record is pinned to the same validated endpoint stored in SQLite. A session-only credential must be entered again.
- Disconnect deletes an app-managed credential-manager record and the nonsecret connection metadata. An environment-backed credential cannot be deleted by Skynet and remains externally managed.
- If no supported secure credential manager is available, remembered connect/disconnect operations fail explicitly with HTTP 503. Use `remember: false` for process-only storage or configure environment variables. Plaintext/file keyring backends are rejected.

W&B setup:

1. Open **Settings** and enter the W&B API key.
2. Optionally enter an entity/team. If omitted, Skynet uses the verified default entity for the key.
3. Use `https://api.wandb.ai` unless connecting to a self-managed HTTPS endpoint.
4. Click **Connect**, then enable W&B in the experiment form. The entity must already exist; W&B creates the project when Skynet creates its first run.

For externally managed restart-safe configuration, set `WANDB_API_KEY`; `WANDB_BASE_URL` and `WANDB_ENTITY` may also be set when needed. Do not put the key in an experiment tag, repository URL, tracking URL, adapter value, or job environment recorded by Skynet.

The central W&B bridge records canonical parameters, provenance, Slurm identity, final summaries, evaluation aggregates, terminal outcome, and artifact references. It does not upload artifact files or synthesize live per-step curves. Adapter-native tracking remains preserved for repositories that implement live W&B logging themselves.

MLflow tracking operations are sanitized and written first to an atomic JSONL spool in each run capsule. The REST bridge drains that spool idempotently when the tracking server is available. Only the web application must reach the server for central tracking. Slurm compute nodes need access only when a selected adapter uses MLflow natively.

A minimal loopback-only MLflow deployment can use:

```bash
mkdir -p /coc/flash7/ycho420/mlflow/{db,artifacts}

uvx --from mlflow mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:////coc/flash7/ycho420/mlflow/db/mlflow.db \
  --artifacts-destination file:///coc/flash7/ycho420/mlflow/artifacts \
  --serve-artifacts
```

Use one server worker with SQLite. Prefer PostgreSQL before adding workers or concurrent users. Plain HTTP is accepted only for loopback endpoints. A server on another host must be exposed through HTTPS, preferably with authentication; do not expose an unauthenticated tracking server publicly.

Configure the application and jobs with:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# For a server on another host, use an HTTPS endpoint instead:
# export MLFLOW_TRACKING_URI=https://mlflow.example.edu

# Choose bearer token or basic authentication when the server requires it.
export MLFLOW_TRACKING_TOKEN='<runtime-secret>'
# export MLFLOW_TRACKING_USERNAME='<runtime-user>'
# export MLFLOW_TRACKING_PASSWORD='<runtime-secret>'

export MLFLOW_HTTP_REQUEST_TIMEOUT=5
export SKYNET_MLFLOW_ENABLED=true
export SKYNET_MLFLOW_AUTO_FLUSH=true
```

The same values can be entered and tested under **Settings** instead of exporting them. Credentials are consumed only at runtime and are redacted from spool records, public settings, errors, and manifests. Each queued record is pinned to its validated provider endpoint; reconnecting to a different endpoint does not replay old records into the new service. When the original provider becomes available again, reconnecting triggers bounded delivery recovery and updates the stored remote IDs, links, and status.

## Selecting an OpenPI dataset

The current OpenPI adapter can use a registered `training_data` bundle or an explicit absolute dataset path. Select a LeRobot LIBERO training configuration such as `pi05_libero`, a READY bundle in `lerobot-v2.0`, `lerobot-v2.1`, or `openpi-libero-lerobot-v2` format, and a matching **Dataset normalization file** (`norm_stats.json`) on the compute node. The observation contract includes `image`, `wrist_image`, `state[8]` and `actions[7]`; matching a format label alone does not prove compatible observations or action semantics.

The versioned dataset bridge checks metadata, episode/file presence and normalization dimensions, then passes the exact selected root to LeRobot. Dataset download/fallback is disabled. The chosen normalization data is copied into the run under its content hash while preserving the checkpoint asset identity used during evaluation. Statistics must have been computed for the selected data and OpenPI configuration; statistics from a different action transform are unsuitable even when dimensions match.

Leave both dataset inputs blank to retain the pinned repository's own dataset configuration. Other embodiments or data-loader APIs require an explicit bridge; they fail with an unsupported message. Custom training commands cannot silently ignore a selected bundle. A bundle with additional unconsumed training inputs or LOCAL-only files is rejected by both the browser and backend.

For a read-only metadata and file-presence check, run `python skynet_app/adapters/openpi_dataset_bridge.py --dataset-root /absolute/path/to/dataset --norm-stats /absolute/path/to/norm_stats.json --check-only` in the dataset's environment. This does not decode all observations or run training. Runtime data loading remains the final check for corrupt Parquet/images, broken symlinks and framework compatibility.

## Known environment requirements and limits

- Exact repository commits and dataset/asset revisions must exist and be readable from compute nodes.
- The repository-specific adapters remain the authority for native hyperparameter semantics and checkpoint discovery.
- DexMimicGen delegates training to the compatible robomimic branch.
- GET-Zero requires legacy Isaac Gym Preview 4 and may rely on W&B for orchestration.
- Isaac Sim/Lab/Gym stacks are not guaranteed to work in a generic `uv` environment.
- Runtime inspection is nonexecuting and intentionally cannot prove that a selected dependency set, simulator, dataset, or container will work on a compute node.
- A declared container digest is recorded but is not remotely verified against the image bytes; pin and verify immutable images outside the application.
- Explicit runtime selection skips dependency-file inspection, so strict reproduction depends on the user supplying the corresponding lock, environment, or image metadata.
- Simulator evaluation may require headless graphics configuration, compatible NVIDIA drivers, dynamic ports, and substantial CPU/RAM in addition to GPUs.
- A generic evaluator is blocked until explicit structured argv is provided.
- Single-node, homogeneous-GPU jobs are supported; mixed GPU types and multi-node training are not.
- The console is currently single-user and trusted-network only.
- Statistical reproducibility may be the strongest available guarantee for nondeterministic frameworks and simulators.

Policy-aware dataset preparation and management are described in [Dataset preparation](docs/policy-data-exports.md). One dataset groups original revisions, prepared formats, verified local/cluster copies and experiment usage. DP and ACT support training and evaluation through the pinned XPolicyLab adapters; shared XPolicyLab HDF5 is an export format. All entry points use the same preparation workflow.
