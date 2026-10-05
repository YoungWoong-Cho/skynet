"""Dataset input previews reuse recording scenes and immutable saved observations."""
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import threading

import numpy as np

from .adapters import recording_dataset, recording_time
from .cluster_config import CLUSTER
from .cluster_runtime import validate_remote_path
from .data_selection import verified_cluster_location
from . import dataset_preview_worker
from .dataset_preview_worker import input_streams


def preview_chunk_frames(metadata, episode):
    """Size requests from saved shapes, including worst-case PNG expansion."""
    raw_bytes, response_bytes = 0, 2048
    for item in input_streams(metadata, episode):
        reference = episode['streams'][item['name']]
        shape = reference['shape'][1:]
        elements = math.prod(shape)
        raw_bytes += elements * np.dtype(reference['dtype']).itemsize
        if item['modality'] == 'rgb':
            response_bytes += elements * 4 / 3 + 1024
        elif item['modality'] in {'point_cloud', 'depth'}:
            points = min(4096, shape[0] if item['modality'] == 'point_cloud' else math.prod(shape[:2]))
            response_bytes += points * 100
        else:
            response_bytes += elements * 24
    return max(1, min(120, int(64 * 1024 * 1024 / max(1, raw_bytes)),
                      int(20 * 1024 * 1024 / response_bytes)))


def available_dataset(database, identifier):
    dataset = database.get_dataset(identifier)
    if dataset is None:
        raise KeyError('Dataset not found')
    return dataset


def describe(database, dataset):
    if dataset['format'] != recording_dataset.FORMAT:
        return dict(state='UNAVAILABLE', detail='Input preview is available for converted recording datasets.', episodes=[], modalities=[])
    if not verified_cluster_location(dataset):
        return dict(state='UNAVAILABLE', detail='This dataset has no available verified cluster location.', episodes=[], modalities=[])
    metadata, sessions = dataset.get('metadata') or {}, {}
    episodes = metadata.get('episodes', [])
    identifiers = sorted({e.get('session_id') for e in episodes if e.get('session_id')})
    if identifiers:
        with database.connection() as connection:
            rows = connection.execute('SELECT id,payload_json FROM live_xr_sessions WHERE id=ANY(?)', (identifiers,)).fetchall()
        sessions = {row['id']: json.loads(row['payload_json']) if isinstance(row['payload_json'], str) else row['payload_json'] for row in rows}
    result, modalities = [], {}
    for episode in episodes:
        session_id = episode.get('session_id')
        session = sessions.get(session_id) or {}
        sha = episode.get('source', {}).get('sha256')
        checksums = session.get('recording_checksums') or {}
        matches = [index for index, path in enumerate(session.get('recordings', [])) if checksums.get(path) == sha]
        hz = recording_time.source_frequency(episode, metadata)
        entry = dict(index=episode['index'], steps=episode['steps'], duration=episode['steps'] / hz, hz=hz,
                     chunk_frames=preview_chunk_frames(metadata, episode),
                     source_session_id=session_id if len(matches) == 1 else None,
                     source_recording_index=matches[0] if len(matches) == 1 else None,
                     source_episode=0, source_sha256=sha)
        if len(matches) != 1:
            entry['scene_warning'] = 'The original recording scene is unavailable; saved dataset inputs remain viewable.'
        result.append(entry)
        for item in input_streams(metadata, episode):
            modalities[item['name']] = {k: v for k, v in item.items() if k != 'recipe'}
    return dict(state='READY', dataset_id=dataset['id'], manifest_sha256=dataset['manifest_sha256'],
                episodes=result, modalities=list(modalities.values()))


class DatasetPreviews:
    """Bounded, immutable chunk reuse. Permissions and availability are checked per request."""
    MAX_CACHE_BYTES = 64 * 1024 * 1024
    MAX_RESPONSE_BYTES = 24 * 1024 * 1024

    def __init__(self):
        self.cache, self.plans = OrderedDict(), OrderedDict()
        self.cache_bytes = 0
        self.lock = threading.RLock()
        self.read_locks = [threading.Lock() for _ in range(16)]
        root = Path(__file__).resolve().parents[1]
        modules = dict(recording_time=Path(recording_time.__file__).read_text(),
                       recording_dataset=Path(recording_dataset.__file__).read_text(),
                       observation_geometry=(root / 'ops/datasets/observation_geometry.py').read_text())
        # Ship the existing pure array/geometry readers, not another conversion stack.
        self.program = 'import sys, types\n'
        for name, source in modules.items():
            self.program += f'module = types.ModuleType({name!r}); sys.modules[{name!r}] = module\nexec({source!r}, module.__dict__)\n'
        self.program += Path(dataset_preview_worker.__file__).read_text()
        self.version = hashlib.sha256(self.program.encode()).hexdigest()

    def frames(self, database, cluster, dataset, episode_index, start, count, gateway='auto'):
        if dataset['format'] != recording_dataset.FORMAT:
            raise ValueError('Input preview requires a converted recording dataset')
        location = verified_cluster_location(dataset)
        if location is None:
            raise ValueError('This dataset has no available verified cluster location')
        sha = dataset.get('manifest_sha256', '')
        if not re.fullmatch('[a-f0-9]{64}', sha):
            raise ValueError('Dataset checksum is unavailable')
        episodes = dataset.get('metadata', {}).get('episodes', [])
        if type(episode_index) is not int or not 0 <= episode_index < len(episodes):
            raise ValueError('Dataset episode does not exist')
        episode = episodes[episode_index]
        if type(start) is not int or type(count) is not int or not 0 <= start < episode['steps'] or not 1 <= count <= 120:
            raise ValueError('Invalid preview frame slice')
        storage = getattr(cluster, 'storage', None)
        roots = storage.allowed_roots() if storage else {cluster.work_root}
        # New job placement does not move registered data. Reuse the transport's
        # current/historical roots, plus the independently registered shared data.
        allowed_roots = tuple(sorted({*roots, CLUSTER.paths.datasets}))
        path = validate_remote_path(location['path'], roots=allowed_roots)
        workspace = getattr(database, 'workspace_id', None)
        identity = (workspace, dataset['id'], path, sha, episode_index, self.version, allowed_roots)
        key = identity + (start, count)
        read_lock = self.read_locks[int(hashlib.sha256(repr(identity).encode()).hexdigest(), 16) % len(self.read_locks)]
        with read_lock:
            with self.lock:
                if key in self.cache:
                    self.cache.move_to_end(key)
                    return self.cache[key]
                plan = self.plans.get(identity, {})
            request = dict(path=path, sha256=sha, episode_index=episode_index, start=start, count=count,
                           allowed_roots=list(allowed_roots), **plan)
            runtime = next((p.environment_path for p in CLUSTER.runtime_profiles.values()
                            if 'isaac_lab' in p.versions and p.environment_path), None)
            if not runtime:
                raise ValueError('No configured Python environment can read saved dataset arrays')
            code = self.program + '\nprint(json.dumps(read_frames(' + repr(request) + '), separators=(",", ":"), allow_nan=False))\n'
            host, output = cluster.run_with_fallback(shlex.quote(runtime + '/bin/python') + ' -', gateway, stdin=code, timeout=120)
            if len(output.encode()) > self.MAX_RESPONSE_BYTES:
                raise ValueError('Preview slice is too large; request fewer frames')
            result = json.loads(output.strip().splitlines()[-1])
            if result['spec']['episode']['source']['sha256'] != episode['source']['sha256']:
                raise ValueError('Registered dataset episode differs from the immutable manifest')
            verified = {name: result.pop(name) for name in ('spec', 'manifest_stat', 'file_stamps')}
            payload = json.dumps(result, separators=(',', ':'), allow_nan=False).encode()
            with self.lock:
                self.plans[identity] = verified
                self.plans.move_to_end(identity)
                while len(self.plans) > 32:
                    self.plans.popitem(last=False)
                self.cache[key] = payload
                self.cache_bytes += len(payload)
                while self.cache_bytes > self.MAX_CACHE_BYTES:
                    _, old = self.cache.popitem(last=False)
                    self.cache_bytes -= len(old)
            return payload


previews = DatasetPreviews()
