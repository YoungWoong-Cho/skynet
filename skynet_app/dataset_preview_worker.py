"""Read bounded, checksum-pinned input slices; never render or write dataset files."""
import base64
import json
from pathlib import Path
import struct
import zlib

import numpy as np


def png_data_url(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError('Saved RGB must contain uint8 HWC pixels')
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff)
    rows = b''.join(b'\0' + row.tobytes() for row in image)
    value = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', image.shape[1], image.shape[0], 8, 2, 0, 0, 0))
             + chunk(b'IDAT', zlib.compress(rows, 3)) + chunk(b'IEND', b''))
    return 'data:image/png;base64,' + base64.b64encode(value).decode()


def input_streams(manifest, episode):
    """Use the frozen adapter contract, never a union of sibling conversions."""
    streams, result = episode['streams'], []
    for requirement in manifest.get('observation_requirements', {}).get('streams', []):
        name, modality = requirement['name'], requirement['modality']
        if modality not in {'rgb', 'depth', 'point_cloud'}:
            continue
        if name not in streams:
            raise ValueError('Required saved input is missing: ' + name)
        ref = streams[name]
        cameras = requirement.get('camera_ids') or [ref.get('camera_id')]
        result.append(dict(name=name, modality=modality, camera_id=ref.get('camera_id') or cameras[0],
                           camera_ids=cameras, recipe=requirement))
    if 'state' in manifest.get('observations', []):
        name = 'faas_state_absolute' if 'faas_state_absolute' in streams else 'state'
        if name in streams:
            result.append(dict(name=name, modality='state', camera_id=None, camera_ids=[]))
    return result


def read_frames(request):
    from recording_dataset import reject_symlinks, digest, validate_manifest, verify_reference, EpisodeReader, close_handles
    from recording_time import source_frequency
    from observation_geometry import pose_from_ros, rigid_transform, intrinsic_matrix, transform_points, unproject

    def allowed(path):
        path = Path(path)
        if (not path.is_absolute() or '..' in path.parts
                or not any(path.is_relative_to(Path(root)) for root in request['allowed_roots'])):
            raise ValueError('Preview streams must stay inside the registered workspace')
        return reject_symlinks(path)

    def stamp(path):
        value = path.stat()
        return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]

    manifest_path = allowed(str(allowed(request['path']) / 'manifest.json'))
    manifest_stat = stamp(manifest_path)
    if manifest_stat != request.get('manifest_stat'):
        if manifest_stat[2] > 16_000_000 or digest(manifest_path) != request['sha256']:
            raise ValueError('Dataset manifest differs from its registered checksum')
        manifest = validate_manifest(json.loads(manifest_path.read_text()))
        index = request['episode_index']
        if type(index) is not int or not 0 <= index < len(manifest['episodes']):
            raise ValueError('Dataset episode does not exist')
        episode = manifest['episodes'][index]
        spec = dict(episode=episode, modalities=input_streams(manifest, episode), hz=source_frequency(episode, manifest))
    else:
        spec = request['spec']
        episode = spec['episode']
    start, count, steps = request['start'], request['count'], episode['steps']
    if type(start) is not int or type(count) is not int or not 0 <= start < steps or not 1 <= count <= 120:
        raise ValueError('Invalid preview frame slice')
    stop = min(steps, start + count)
    streams, names = episode['streams'], set()
    for modality in spec['modalities']:
        name = modality['name']
        names.add(name)
        names.update(k for k in (name + '_valid_mask', name + '_frame_ids', name + '_timestamps') if k in streams)
        cameras = set(modality.get('camera_ids') or [])
        recipe_frame = modality.get('recipe', {}).get('coordinate_frame', '')
        if recipe_frame.startswith('camera:'):
            cameras.add(recipe_frame[7:])
        for camera in cameras:
            names.update(k for k in (str(camera) + '_intrinsics', str(camera) + '_world_from_camera') if k in streams)
    if 'timestamps' in streams:
        names.add('timestamps')
    pins, verified = {}, set()
    for name in sorted(names):
        reference = streams[name]
        path = allowed(reference['path'])
        before = stamp(path)
        if before[2] > 4_000_000_000:
            raise ValueError('Saved input exceeds the preview file limit')
        if reference.get('source_sha256', episode['source']['sha256']) != episode['source']['sha256']:
            raise ValueError('Saved input belongs to a different recording')
        old = request.get('file_stamps', {}).get(str(path))
        identity = dict(stat=before, sha256=reference['sha256'])
        verify_reference(reference, steps=steps, verified=verified, verify_files=old != identity)
        pins[str(path)] = identity
    reader, values = EpisodeReader(episode), {}
    try:
        total_bytes = 0
        for name in names:
            reference = streams[name]
            total_bytes += int(np.prod(reference['shape'][1:])) * (stop - start) * np.dtype(reference['dtype']).itemsize
            if total_bytes > 128_000_000:
                raise ValueError('Input slice exceeds the preview memory limit')
            values[name] = reader.read(name, slice(start, stop))
        def calibration(camera, offset, reference):
            saved = reference.get('calibration') or episode.get('capture', {}).get('cameras', {}).get(camera) or {}
            intrinsics = values.get(str(camera) + '_intrinsics')
            poses = values.get(str(camera) + '_world_from_camera')
            k = intrinsic_matrix(intrinsics[offset] if intrinsics is not None else saved.get('intrinsic_matrix'))
            pose = poses[offset] if poses is not None else saved.get('world_from_camera')
            if pose is None:
                pose = pose_from_ros(saved.get('position_world'), saved.get('quaternion_world_ros'))
            return k, rigid_transform(pose)
        frames = []
        for offset, index in enumerate(range(start, stop)):
            frame = dict(index=index, time=float(values['timestamps'][offset]) if 'timestamps' in values else index / spec['hz'],
                         images=[], point_cloud=[], depth=[], state=[])
            for modality in spec['modalities']:
                name, kind, camera = modality['name'], modality['modality'], modality.get('camera_id')
                reference, value = streams[name], values[name][offset]
                label = {'scene_front': 'Front', 'scene_left': 'Left', 'scene_right': 'Right'}.get(camera, camera or name)
                if kind == 'state':
                    if not np.isfinite(value).all():
                        raise ValueError('Saved state contains nonfinite values')
                    labels = episode.get('capture', {}).get('action_joint_names', []) if name == 'state' else []
                    frame['state'].append(dict(name=name, values=np.asarray(value).reshape(-1).tolist(), labels=labels))
                    continue
                ids, times = values.get(name + '_frame_ids'), values.get(name + '_timestamps')
                if (ids is not None and int(ids[offset]) != index) or (times is not None and not np.isclose(times[offset], frame['time'], atol=1e-6)):
                    raise ValueError('Saved camera input is not aligned with the recording timeline')
                if kind == 'rgb':
                    k, pose = calibration(camera, offset, reference)
                    frame['images'].append(dict(id=name, label=label, data_url=png_data_url(value),
                        width=value.shape[1], height=value.shape[0], intrinsics=k.tolist(), world_from_camera=pose.tolist()))
                    continue
                mask = values.get(name + '_valid_mask')
                mask = mask[offset] if mask is not None else None
                if kind == 'depth':
                    k, pose = calibration(camera, offset, reference)
                    depth = value[..., 0] if value.ndim == 3 and value.shape[-1] == 1 else value
                    xyz, colors = unproject(depth, k, valid_mask=mask)
                    xyz = transform_points(xyz, pose)
                else:
                    if value.ndim != 2 or value.shape[1] not in (3, 6):
                        raise ValueError('Saved point cloud has an unsupported shape')
                    points = value[mask] if mask is not None else value
                    if not np.isfinite(points).all():
                        raise ValueError('Saved point cloud contains nonfinite values')
                    xyz, colors = points[:, :3], points[:, 3:] if points.shape[1] == 6 else None
                    recipe = modality['recipe']
                    coordinate_frame = recipe.get('coordinate_frame')
                    if coordinate_frame != 'world':
                        target_camera = coordinate_frame[7:] if coordinate_frame and coordinate_frame.startswith('camera:') else camera
                        _, pose = calibration(target_camera, offset, reference)
                        if recipe.get('camera_convention', 'ros_optical') != 'ros_optical':
                            raise ValueError('Saved point cloud camera convention is not supported')
                        xyz = transform_points(xyz, pose)
                    if colors is not None and recipe.get('color_range') == '0_255':
                        colors = colors / 255
                # Display decimation only; stored/training observations stay unchanged.
                stride = max(1, int(np.ceil(len(xyz) / 4096)))
                frame[kind].append(dict(id=name, label=label, positions=np.round(xyz[::stride], 6).tolist(),
                    colors=np.clip(colors[::stride].astype(np.float64), 0, 1).round(4).tolist() if colors is not None else None))
            frames.append(frame)
        for path, identity in pins.items():
            if stamp(Path(path)) != identity['stat']:
                raise ValueError('Saved input changed while reading preview frames')
        if stamp(manifest_path) != manifest_stat:
            raise ValueError('Dataset manifest changed while reading preview frames')
        return dict(start=start, total=steps, hz=spec['hz'], frames=frames, spec=spec,
                    manifest_stat=manifest_stat, file_stamps=pins)
    finally:
        close_handles()
