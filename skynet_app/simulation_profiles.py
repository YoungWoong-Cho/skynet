"""Pinned simulation profiles shared by observation preparation and evaluation."""
import hashlib
import json
from pathlib import Path
import re

from .cluster_runtime import DEFAULT_GATEWAY, WORK_ROOT
from .database import canonical_json
from .dexverse_versions import environment_profile


def frozen_cluster_profile(app_root, cluster, session, gateway=DEFAULT_GATEWAY):
    """``gateway`` routes the shared-storage manifest reads; it is not part of the profile."""
    original = session['profile']
    base = json.loads((Path(app_root) / 'config/live_video.json').read_text())
    profile = environment_profile(base, original['task'], cluster_root=WORK_ROOT)
    if profile['source_revision'] != original.get('source_revision'):
        raise ValueError('No pinned observation renderer matches this recording’s DexVerse revision')
    for key in ('robot', 'task', 'hand', 'hand_name', 'task_name', 'recording_schema_version'):
        if key in original:
            profile[key] = original[key]
    if original.get('hand_bundle'):
        bundle = original['hand_bundle']
        if not re.fullmatch('[a-f0-9]{64}', bundle.get('digest', '')):
            raise ValueError('Recording has no immutable hand identity')
        profile['hand_bundle'] = dict(bundle, root=f"{WORK_ROOT}/hands/{profile['robot']}/{bundle['digest']}")
        from .simulation_hands import find_bundle
        try:
            local = find_bundle(profile['robot'], bundle['digest'], app_root=Path(app_root))
        except ValueError:
            _, raw_manifest = cluster.read_file(profile['hand_bundle']['root'] + '/manifest.json', gateway, max_bytes=4_000_000)
            raw_manifest = raw_manifest.encode()
        else:
            manifest_file = local / 'manifest.json'
            if manifest_file.stat().st_size > 4_000_000:
                raise ValueError('Frozen hand manifest exceeds its size limit')
            raw_manifest = manifest_file.read_bytes()
        manifest = json.loads(raw_manifest)
        if (manifest.get('schema') != 'skynet.simulation-hand/v1' or manifest.get('digest') != bundle['digest']
                or manifest.get('robot') != profile['robot']):
            raise ValueError('Frozen hand manifest differs from the original recording')
        manifest_sha = hashlib.sha256(raw_manifest).hexdigest()
        if bundle.get('manifest_sha256', manifest_sha) != manifest_sha:
            raise ValueError('Frozen hand manifest checksum changed')
        profile['hand_bundle']['manifest_sha256'] = manifest_sha
    if profile.get('asset_bundle'):
        _, content = cluster.read_file(profile['asset_bundle'] + '/manifest.json', gateway, max_bytes=4_000_000)
        manifest = json.loads(content)
        inventory = (manifest.get('integrity') or {}).get('inventory_sha256', '')
        if manifest.get('schema_version') != 'dataset-bundle/v1' or not re.fullmatch('[a-f0-9]{64}', inventory):
            raise ValueError('Simulation asset bundle has no verified inventory')
        profile['asset_bundle_manifest_sha256'] = hashlib.sha256(content.encode()).hexdigest()
        profile['asset_inventory_sha256'] = inventory
    return dict(profile, device='cuda:0')


def freeze_target_simulation_profile(database, cluster, target, app_root):
    """Resolve recorded provenance without scheduling rendering or changing data."""
    metadata = target.get("metadata") or {}
    episodes = metadata.get("episodes") or []
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Evaluation target has no recorded episode provenance")
    sessions, profiles = {}, {}
    for episode in episodes:
        identifier = episode.get("session_id") or episode.get("recording_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Evaluation target episode has no source collection session")
        if identifier not in sessions:
            with database.connection() as connection:
                row = connection.execute("SELECT payload_json FROM live_xr_sessions WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise ValueError("Evaluation target source collection session is unavailable")
            payload = row["payload_json"]
            sessions[identifier] = json.loads(payload) if isinstance(payload, str) else payload
        session = sessions[identifier]
        if session.get("id") != identifier or not isinstance(session.get("profile"), dict):
            raise ValueError("Evaluation target source session has no pinned simulation profile")
        original = session["profile"]
        capture = episode.get("capture") or metadata.get("capture") or {}
        for key in ("robot", "hand", "task", "source_revision"):
            if not capture.get(key) or capture[key] != original.get(key):
                raise ValueError(f"Evaluation target {key} differs from its source collection session")
        if episode.get("hand_id", capture["robot"]) != capture["robot"]:
            raise ValueError("Evaluation target hand differs from its recorded robot")
        checksum = (episode.get("source") or {}).get("sha256")
        if not isinstance(checksum, str) or not re.fullmatch(r"[a-f0-9]{64}", checksum):
            raise ValueError("Evaluation target episode has no verified source recording checksum")
        checksums = session.get("recording_checksums") or {}
        if not any(checksums.get(path) == checksum for path in session.get("recordings", [])):
            raise ValueError("Evaluation target episode does not belong to its source collection session")
        hand_digest = capture.get("hand_adapter_digest")
        bundle = original.get("hand_bundle") or {}
        if hand_digest and hand_digest != bundle.get("digest"):
            raise ValueError("Evaluation target hand asset differs from the frozen collection hand")
        if capture["robot"].startswith("skynet_") and not bundle.get("digest"):
            raise ValueError("Custom evaluation hand requires its frozen simulation bundle")
    # Validate every episode before performing any bounded cluster manifest reads.
    for identifier, session in sessions.items():
        profiles[identifier] = frozen_cluster_profile(app_root, cluster, session)
    distinct = {canonical_json(profile) for profile in profiles.values()}
    if len(distinct) != 1:
        raise ValueError("Evaluation target episodes require one identical frozen simulation profile")
    return next(iter(profiles.values()))
