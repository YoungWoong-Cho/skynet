"""CPU-only shared-manifest test harness; no application compatibility path."""
import json
from pathlib import Path

from ops.datasets.artifacts import digest
from ops.datasets.recording_prepare import prepare as build


def prepare(self, identifier):
    job = self.get(identifier)
    directory = self.root / identifier
    try:
        sources = []
        for source in job['sources']:
            session = self.live.get(source['session_id'])
            root = Path(session['root']) / 'output'
            sources.append(dict(source, recording=str(root/source['path']),
                                images=str(root/source['image_path']) if source.get('image_path') else None))
        request = dict(job['requirements'], output=str(directory/'output'), sources=sources,
            recording_root=str(self.root/'shared-recordings'), adapter=job['adapter'],
            adapter_data_preset=job['adapter_data_preset'], split=job['split'],
            source_revision=job['source_revision'], converter_sha256=job['converter_sha256'])
        manifest = build(request)
        sha = digest(directory/'output/manifest.json')
        version = self.register(job, manifest, sha, remote_path=str(directory/'output'))
        self.database.record_data_location(version['id'],kind='cluster',host='test',
            path=str(directory/'output'),manifest_sha256=sha)
        self.update(identifier,state='READY',stage='READY',version_id=version['id'],
            manifest_sha256=sha,episodes=len(manifest['episodes']),steps=manifest['steps'],
            training_ready=True,error=None)
    except Exception as error:
        self.update(identifier,state='FAILED',stage='FAILED',error=str(error))
    finally:
        self.active.discard(identifier)
