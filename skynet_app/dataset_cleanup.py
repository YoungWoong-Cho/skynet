"""Remove only generated dataset paths; shared by local and cluster cleanup."""

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from uuid import UUID


def cleanup(root, *, jobs=(), prepared=(), observations=(), cluster=False):
    root = Path(root)
    if not root.is_absolute() or root == Path('/') or '..' in root.parts:
        raise ValueError('Dataset cleanup requires an absolute owned root')
    paths = []
    for identifier in jobs:
        if str(UUID(identifier)) != identifier:
            raise ValueError("Invalid preparation identifier")
        paths.append(root / "jobs/runs" / identifier if cluster else root / identifier)
    for checksum in prepared:
        if not re.fullmatch(r"[a-f0-9]{64}", checksum):
            raise ValueError("Invalid dataset checksum")
    paths.extend(root / "datasets/prepared" / sha for sha in prepared)
    observation_manifests = {}
    for item in observations:
        source, key, camera = item['source_sha256'], item['artifact_key'], item['camera_id']
        if (not cluster or item.get('modality') != 'point_cloud'
                or not re.fullmatch(r'[a-f0-9]{64}', source) or not re.fullmatch(r'[a-f0-9]{64}', key)
                or not re.fullmatch(r'[A-Za-z0-9_.-]+', camera) or camera in {'.', '..'}
                or not re.fullmatch(r'[a-f0-9]{64}', item['manifest_sha256'])):
            raise ValueError('Invalid retired point-cloud identity')
        path = root / 'datasets/recordings' / source / 'point_cloud' / camera / key
        if item['path'] != str(path):
            raise ValueError('Point-cloud cleanup path differs from its immutable identity')
        observation_manifests[path] = item['manifest_sha256']
        paths.extend([path, path.parent / ('.' + key + '.publish.lock')])
    # Validate the entire plan before deleting anything. A linked ancestor must
    # never redirect deletion outside these generated directories.
    for path in paths:
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Dataset cleanup cannot follow symbolic links")
        if path.parent.name == "prepared" and path.exists():
            manifest = path / "manifest.json"
            if manifest.is_symlink() or not manifest.is_file():
                raise ValueError("Prepared dataset manifest is missing")
            if hashlib.sha256(manifest.read_bytes()).hexdigest() != path.name:
                raise ValueError("Prepared dataset manifest has changed")
        if path in observation_manifests and path.exists():
            manifest = path / 'manifest.json'
            if manifest.is_symlink() or not manifest.is_file():
                raise ValueError('Point-cloud manifest is missing')
            raw = manifest.read_bytes()
            if hashlib.sha256(raw).hexdigest() != observation_manifests[path]:
                raise ValueError('Point-cloud manifest has changed')
            value = json.loads(raw)
            if value.get('artifact_key') != path.name or value.get('modality') != 'point_cloud':
                raise ValueError('Retired point-cloud manifest identity differs')
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    return {"removed": True}


if __name__ == "__main__":
    print(json.dumps(cleanup(**json.loads(sys.argv[1]))))
