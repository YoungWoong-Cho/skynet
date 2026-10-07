"""Lazy, durable observation preparation ahead of format conversion."""
import hashlib
import inspect
import json
from pathlib import Path
import shlex
from time import time

from .cluster_config import CLUSTER, cpus_for_gpus
from .cluster_runtime import DEFAULT_GATEWAY, WORK_ROOT, ClusterError, SubmissionOutcomeUnknown
from .database import canonical_json
from .live_xr_review import ArrayUnpickler
from .isaac_job import isaac_environment
from .observation_contracts import plan_artifacts, reuse_recorded_captures, PREPARE_SCHEMA
from .observation_store import ObservationStore
from .preparation_states import TERMINAL_FAILURE_STATES, observed_state
from .gpu_preflight import GPU_MISSING_MESSAGE, gpu_missing_exit, gpu_preflight_lines
from .sbatch import SHEBANG, sbatch_header, shell_prelude


# Leave space below the worker's 4 MB JSON limit for producer IDs, paths and
# framing added at submission. Long selections become independent bounded jobs.
MAX_BATCH_SOURCES = 32
# Sampling cost scales with every point in every frame; isolate CPU episodes.
MAX_DERIVE_BATCH_SOURCES = 1
MAX_BATCH_BYTES = 3_500_000
MAX_REQUEST_BYTES = 4_000_000
# Completed jobs may need a short interval for shared-filesystem visibility.
COMPLETED_RESULT_GRACE_SECONDS = 60


def request_batches(requests, worker_sources, *, max_sources=None):
    max_sources = MAX_BATCH_SOURCES if max_sources is None else max_sources
    sources = {source['sha256']: source for source in worker_sources}
    source_bytes = {key: len(canonical_json(source).encode()) for key, source in sources.items()}
    batch, keys, size = [], set(), 1024
    for request in requests:
        key = request['source_sha256']
        node_bytes = len(canonical_json(request).encode()) + 1
        added = node_bytes + (source_bytes[key] + 1 if key not in keys else 0)
        if batch and (size + added > MAX_BATCH_BYTES or (key not in keys and len(keys) >= max_sources)):
            yield batch, [sources[k] for k in sorted(keys)]
            batch, keys, size = [], set(), 1024
            added = node_bytes + source_bytes[key] + 1
        if size + added > MAX_BATCH_BYTES:
            raise ValueError('One recording observation request exceeds the preparation size limit')
        batch.append(request)
        keys.add(key)
        size += added
    if batch:
        yield batch, [sources[k] for k in sorted(keys)]



def prepared_source(source, contract, planned, existing):
    """Preserve stream identity when a camera has multiple shapes or cloud crops."""
    candidates, streams = {}, {}
    for node in planned:
        if node['source_sha256'] != source['sha256']:
            continue
        ref = {key: existing[node['artifact_key']][key] for key in ('artifact_key', 'path', 'manifest_sha256')}
        candidates.setdefault((node['camera_id'], node['modality']), {})[node['artifact_key']] = ref
        for stream in node.get('streams', []):
            streams[stream] = ref
    by_camera = {}
    for (camera, modality), choices in candidates.items():
        # Native single-recipe converters use this convenient view. Ambiguous
        # variants are available by declared stream name; never select the last.
        if len(choices) == 1:
            by_camera.setdefault(camera, {})[modality] = next(iter(choices.values()))
    return dict(source, observation_contract=contract,
                observation_artifacts=by_camera, observation_streams=streams)


class ObservationsPending(ValueError):
    pass


class JobStatusLookup:
    """Scheduler records for every job one monitor pass may consult.

    The monitor asks Slurm once per pass for the cluster and preflight jobs of
    every pending preparation together with every observation producer's job,
    then hands the result to each consumer. A job the read did not list is
    None, exactly as a direct read reports a missing status; a read that
    failed raises its ClusterError at every lookup, so each consumer keeps the
    recovery it applies to a failed read of its own.
    """

    def __init__(self, cluster, job_ids):
        self.statuses, self.error = {}, None
        job_ids = [job_id for job_id in dict.fromkeys(job_ids) if job_id]
        if job_ids:
            try:
                _, self.statuses = cluster.job_statuses(job_ids, DEFAULT_GATEWAY)
            except ClusterError as exc:
                self.error = exc

    def get(self, job_id):
        if self.error is not None:
            raise self.error
        return self.statuses.get(job_id)


def job_status(cluster, statuses, job_id):
    """One job's record from the pass read, or its own read outside the monitor."""
    return (JobStatusLookup(cluster, [job_id]) if statuses is None else statuses).get(job_id)


class ObservationPreparation:
    def __init__(self, service):
        self.service = service
        self.store = ObservationStore(service.database)

    @property
    def cluster(self):
        return self.service.cluster

    def frozen_files(self):
        root = self.service.live.root
        names = ['observation_prepare.py', 'observation_geometry.py', 'observation_render.py', 'observation_supervisor.py']
        files = {name: (root / 'ops/datasets' / name).read_text() for name in names}
        for name in ('wrist.py', 'scene_restore.py', 'recording_metadata.py'):
            files[name] = (root / 'ops/xr' / name).read_text()
        files['trajectory.py'] = Path(__file__).with_name('trajectory.py').read_text()
        files['observation_contracts.py'] = Path(__file__).with_name('observation_contracts.py').read_text()
        files['arrays.py'] = 'import pickle\nimport numpy as np\n' + inspect.getsource(ArrayUnpickler)
        files['scene_geometry.py'] = (root / 'skynet_app/adapters/scene_geometry.py').read_text()
        return files

    def profile(self, session):
        from .simulation_profiles import frozen_cluster_profile
        return frozen_cluster_profile(self.service.live.root, self.cluster, session)

    def ensure(self, job, sources, statuses=None):
        """Attach the immutable plan, schedule missing layers, return ready refs."""
        contract = job.get('observation_contract')
        if not contract or not contract['streams']:
            return sources
        from ops.datasets.observation_prepare import render_identity
        # Use the capsule frozen when Convert was requested, even after app upgrades.
        directory = self.service.root / job['id'] / 'observation-worker'
        files = {p.name: p.read_text() for p in directory.iterdir() if p.is_file()}
        if hashlib.sha256(canonical_json(files).encode()).hexdigest() != job['observation_worker_sha256']:
            raise ValueError('Frozen observation worker changed; request a new conversion')
        renderer_revision = hashlib.sha256(canonical_json(files).encode()).hexdigest()
        plan_path = self.service.root / job['id'] / 'observation-plan.json'
        if job.get('observation_plan_sha256'):
            raw = plan_path.read_text()
            if hashlib.sha256(raw.encode()).hexdigest() != job['observation_plan_sha256']:
                raise ValueError('Frozen observation plan changed')
            plan = json.loads(raw)
            planned, worker_sources = plan['nodes'], plan['sources']
        else:
            planned, profiles, worker_sources = [], {}, []
            captures = self.store.ready_captures(sources)
            for source in sources:
                required = dict(contract, streams=[stream for stream in contract['streams']
                    if stream['name'] not in source.get('shared_image_streams', {})])
                if not required['streams']:
                    continue
                session_id = source['session_id']
                if session_id not in profiles:
                    profiles[session_id] = self.profile(self.service.live.get(session_id))
                profile = profiles[session_id]
                identity = render_identity(profile, renderer_revision)
                nodes = plan_artifacts(source['sha256'], {'episode_index': 0, 'episode_key': source['sha256']}, required, identity)
                nodes = reuse_recorded_captures(nodes, captures.get(source['sha256'], []))
                planned.extend(nodes)
                worker_sources.append(dict(source, episode_key=source['sha256'], episode_index=0,
                                           profile=profile, render_recipe=identity))
            raw = canonical_json(dict(nodes=planned, sources=worker_sources))
            temporary = plan_path.with_suffix('.tmp')
            temporary.write_text(raw)
            temporary.replace(plan_path)
            self.service.update(job['id'], observation_plan_sha256=hashlib.sha256(raw.encode()).hexdigest())
        self.store.attach(job['id'], planned, sources)
        existing = self.store.for_job(job['id'])
        failed = [a for a in existing.values() if a['state'] == 'FAILED']
        if failed:
            raise ValueError(failed[0]['error'] or 'Observation preparation failed; retry this conversion')
        for mode, modalities in [('render', {'rgb', 'depth'}), ('derive', {'point_cloud'})]:
            requests = []
            for node in planned:
                if node['modality'] not in modalities:
                    continue
                dependencies = [existing[key] for key in node['dependencies']]
                if any(d['state'] != 'READY' for d in dependencies):
                    continue
                requests.append(dict(node,
                    output_dir=f"{WORK_ROOT}/datasets/recordings/{node['source_sha256']}/{node['modality']}/{node['camera_id']}/{node['artifact_key']}",
                    dependencies=[{k: d[k] for k in ('artifact_key','path','manifest_sha256')} for d in dependencies]))
            # claim() needs dependency keys as well as the worker's concrete refs.
            for request in requests:
                request['dependency_keys'] = [d['artifact_key'] for d in request['dependencies']]
            groups = {}
            source_by_key = {source['sha256']: source for source in worker_sources}
            for request in requests:
                source = source_by_key[request['source_sha256']]
                group = canonical_json([source['profile'], request['recipe']['width'], request['recipe']['height']]) if mode == 'render' else 'derive'
                groups.setdefault(group, []).append(request)
            for group in groups.values():
                for batch, batch_sources in request_batches(group, worker_sources,
                        max_sources=MAX_DERIVE_BATCH_SOURCES if mode == 'derive' else MAX_BATCH_SOURCES):
                    self.store.claim(batch, dict(schema=PREPARE_SCHEMA, mode=mode,
                        sources=batch_sources, worker_files=files))
        # A monitor pass shares its scheduler read; a one-off conversion lets
        # the tick read the producers' status for itself.
        if statuses is None:
            self.tick()
        else:
            self.tick(statuses)
        existing = self.store.for_job(job['id'])
        failed = next((a for a in existing.values() if a['state'] == 'FAILED'), None)
        if failed:
            raise ValueError(failed['error'] or 'Observation preparation failed')
        ready = sum(a['state'] == 'READY' for a in existing.values())
        producers = self.store.progress(job['id'])
        self.service.update(job['id'], observation_progress=dict(ready=ready, total=len(existing),
            reused=sum(len(source.get('shared_image_streams', {})) for source in sources), producers=producers))
        if ready != len(existing):
            running = [p for p in producers if p['state'] == 'RUNNING']
            pending = [p for p in producers if p['state'] in {'QUEUED','SUBMITTING','PENDING'}]
            if running:
                phase = 'Rendering required camera data' if any(p['mode'] == 'render' for p in running) else 'Building point clouds from saved depth'
            elif pending:
                phase = 'Waiting for a simulation GPU' if any(p['mode'] == 'render' for p in pending) else 'Waiting for point-cloud preparation CPUs'
            else:
                phase = 'Preparing required observations'
            raise ObservationsPending(f'{phase}: {ready}/{len(existing)} observations ready')
        return [prepared_source(source, contract, planned, existing) for source in sources]

    def _stage(self, producer):
        request = dict(producer['request'])
        files = request.pop('worker_files')
        identifier, token = producer['id'], producer['attempt_token']
        root = f'{WORK_ROOT}/jobs/runs/{identifier}/observations/{token}'
        request['result_path'] = root + '/result.json'
        request['receipt_path'] = root + '/result.json'
        request['staging_root'] = root + '/staging'
        # Dependency edges are persisted by the store; workers receive concrete refs.
        for item in request['requests']:
            item.pop('dependency_keys', None)
        from .simulation_hands import find_bundle, upload
        uploaded_hands = set()
        for source in request['sources']:
            bundle = source['profile'].get('hand_bundle')
            if not bundle:
                continue
            try:
                local = find_bundle(source['profile']['robot'], bundle['digest'], app_root=self.service.live.root)
            except ValueError:
                # Archived hosts may have the verified bundle without a local copy.
                # The worker verifies its manifest and every referenced asset before use.
                continue
            if hashlib.sha256((local / 'manifest.json').read_bytes()).hexdigest() != bundle['manifest_sha256']:
                raise ValueError('Frozen hand manifest changed after conversion was requested')
            identity = (bundle['root'], bundle['manifest_sha256'])
            if identity in uploaded_hands:
                continue
            if upload(local, WORK_ROOT, self.cluster, DEFAULT_GATEWAY) != bundle['root']:
                raise ValueError('Frozen hand upload has an unexpected path')
            uploaded_hands.add(identity)
        relative = f'observations/{token}'
        capsule = {relative + '/worker/' + name: value for name, value in files.items()}
        raw_request = canonical_json(request)
        if len(raw_request.encode()) > MAX_REQUEST_BYTES:
            raise ValueError('Observation request exceeds the worker JSON size limit')
        capsule[relative + '/request.json'] = raw_request
        self.cluster.write_capsule_files(identifier, capsule, DEFAULT_GATEWAY)
        queue = CLUSTER.queue(CLUSTER.defaults.background_queue_policy)
        rendering = request['mode'] == 'render'
        runtime = CLUSTER.runtime_profile(
            CLUSTER.defaults.rendering_runtime_profile if rendering else CLUSTER.defaults.background_runtime_profile)
        environment = str(runtime.environment_path)
        gpu = []
        if rendering:
            placement = CLUSTER.isaac_evaluation_placement
            if placement is None:
                raise ValueError('Observation rendering requires a configured Isaac-compatible cluster node')
            node = placement.default_node
            gpu = [f'#SBATCH --nodelist={node}', f'#SBATCH --gres={CLUSTER.gres(placement.nodes[node].gpu_type, 1)}']
        # Rendering holds one GPU; point-cloud derivation is a CPU-only job.
        jobs = CLUSTER.defaults.background_jobs
        shape = jobs.observation_render if rendering else jobs.observation_derive
        cpus = cpus_for_gpus(1) if rendering else shape.cpus_per_task
        script = '\n'.join([
            SHEBANG,
            *sbatch_header(job_name=f'observe-{identifier[:8]}', queue=queue, cpus=cpus, memory_gb=shape.memory_gb,
                           time_limit=shape.time_limit, output=f'{root}/observations.log', extra=gpu),
            *shell_prelude(umask='077'),
            *(gpu_preflight_lines(1) if rendering else []),
            'export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1',
            f'export OMP_NUM_THREADS={cpus} OPENBLAS_NUM_THREADS={cpus} MKL_NUM_THREADS={cpus}',
            f'export LD_LIBRARY_PATH={shlex.quote(environment + "/lib")}:"${{LD_LIBRARY_PATH:-}}"',
            *(isaac_environment(request['sources'][0]['profile'], root) if rendering else ['export CUDA_VISIBLE_DEVICES=']),
            shlex.join([environment + '/bin/python', root + '/worker/observation_prepare.py',
                        '--request', root + '/request.json', '--result', root + '/result.json']), '',
        ])
        return self.store.update(producer, state='SUBMITTING', root=root, script=script)

    def _with_failure_log(self, producer, message):
        try:
            _, output = self.cluster.read_log(producer['root'] + '/observations.log', DEFAULT_GATEWAY, lines=100, max_bytes=8000)
            if output.strip():
                message += ': ' + output.strip()[-2000:]
        except (ClusterError, ValueError):
            pass
        return message

    def cluster_job_ids(self, producers=None):
        """Slurm ids of the unfinished producers, for one shared status read."""
        if producers is None:
            producers = self.store.producers()
        return [producer['cluster_job_id'] for producer in producers if producer.get('cluster_job_id')]

    def tick(self, statuses=None):
        # Producers are recoverable even if no initiating Convert is being polled.
        snapshots = self.store.producers()
        if statuses is None:
            # Outside a monitor pass one read still covers every producer.
            statuses = JobStatusLookup(self.cluster, self.cluster_job_ids(snapshots))
        for snapshot in snapshots:
            producer = self.store.acquire(snapshot['id'])
            if not producer:
                continue
            try:
                if not producer.get('script'):
                    producer = self._stage(producer)
                if not producer.get('cluster_job_id'):
                    submitted = self.cluster.submit_script(producer['script'], producer['id'], DEFAULT_GATEWAY,
                        submission_key=f"{producer['id']}-observations-{producer['attempt_token']}")
                    producer = self.store.update(producer, state='PENDING', cluster_job_id=submitted.job_id)
                # New submissions stay pending until the next shared status read;
                # a failed read raises here and cannot mark the attempt failed.
                status = statuses.get(producer['cluster_job_id'])
                if not status:
                    continue
                state = status['State']
                if state != 'COMPLETED' and state not in TERMINAL_FAILURE_STATES:
                    self.store.update(producer, state=observed_state(state, producer['state']),
                                      reason=status.get('Reason') or f'Awaiting scheduler confirmation ({state})',
                                      result_missing_since=None)
                    continue
                if state != 'COMPLETED':
                    # Slurm has definitively stopped this attempt. A bootstrap
                    # failure may never write a result, so missing/unreadable
                    # diagnostics must not leave its artifacts queued forever.
                    message = f'Observation job ended as {state}'
                    if gpu_missing_exit(status):
                        message += f' ({GPU_MISSING_MESSAGE})'
                    try:
                        _, raw = self.cluster.read_file(producer['root'] + '/result.json', DEFAULT_GATEWAY, max_bytes=10_000_000)
                        result = json.loads(raw)
                        if (result.get('request_id') == producer['id']
                                and result.get('attempt_token') == producer['attempt_token'] and result.get('error')):
                            message += ': ' + str(result['error'])[-2000:]
                    except (ClusterError, ValueError):
                        message = self._with_failure_log(producer, message)
                    raise ValueError(message)
                _, raw = self.cluster.read_optional_file(producer['root'] + '/result.json', DEFAULT_GATEWAY, max_bytes=10_000_000)
                if raw is None:
                    now = time()
                    since = producer.get('result_missing_since')
                    if since is None:
                        producer = self.store.update(producer, result_missing_since=now,
                            reason='Job completed; waiting for its result file to become visible')
                        continue
                    if now - since < COMPLETED_RESULT_GRACE_SECONDS:
                        continue
                    raise ValueError(self._with_failure_log(producer,
                        'Observation job completed without a result file after the shared-filesystem visibility grace period'))
                receipt = json.loads(raw)
                if receipt.get('request_id') != producer['id'] or receipt.get('attempt_token') != producer['attempt_token']:
                    raise ValueError('Observation result belongs to a different producer attempt')
                if state != 'COMPLETED' or receipt.get('state') != 'READY':
                    raise ValueError(receipt.get('error') or f'Observation job ended as {state}')
                expected = {n['artifact_key']: n for n in producer['request']['requests']}
                artifacts = receipt.get('artifacts', [])
                for artifact in artifacts:
                    key = artifact['artifact_key']
                    if key not in expected or artifact.get('path') != expected[key]['output_dir']:
                        raise ValueError('Observation result has an unexpected artifact path')
                    _, content = self.cluster.read_file(artifact['path'] + '/manifest.json', DEFAULT_GATEWAY, max_bytes=4_000_000)
                    if hashlib.sha256(content.encode()).hexdigest() != artifact.get('manifest_sha256'):
                        raise ValueError('Observation manifest checksum differs from worker receipt')
                    manifest = json.loads(content)
                    if manifest.get('artifact_key') != key or manifest.get('spec') != expected[key]['spec']:
                        raise ValueError('Observation manifest identity differs from the requested specification')
                if receipt.get('cleanup_warning'):
                    producer = self.store.update(producer, cleanup_warning=str(receipt['cleanup_warning'])[:2000])
                self.store.finish(producer, artifacts=artifacts)
            except (ClusterError, SubmissionOutcomeUnknown):
                # Preserve the same attempt after network/submission uncertainty.
                continue
            except Exception as exc:
                try:
                    self.store.finish(producer, error=str(exc))
                except ValueError:
                    pass  # Another monitor recovered the expired lease.
            finally:
                self.store.release(producer)
