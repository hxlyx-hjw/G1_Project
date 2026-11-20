"""Download and adapt the G1-retargeted AMASS mirror for this project."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha1
from pathlib import Path
from urllib.parse import quote

import numpy as np
import mujoco
import requests

from data_paths import AMASS, AMASS_MOTIONS
from .model import build_model, joint_info
from .reference import Reference


DEFAULT_REPO = 'ember-lab-berkeley/AMASS_Retargeted_for_G1'
DEFAULT_MIRROR = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
CHUNK_SIZE = 1024 * 1024
CATALOG_NAME = 'catalog.json'


def _session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Codex AMASS downloader',
    })
    return session


def remote_files(repo=DEFAULT_REPO, mirror=DEFAULT_MIRROR, timeout=60):
    """Return the repository file list from the Hugging Face mirror API."""
    url = f'{mirror.rstrip("/")}/api/datasets/{repo}/tree/main?recursive=true'
    response = _session().get(url, timeout=timeout)
    response.raise_for_status()
    items = response.json()
    files = []
    for item in items:
        if item.get('type') != 'file':
            continue
        path = str(item.get('path', ''))
        if path == 'README.md' or path.lower().endswith('.npz'):
            files.append({'path': path, 'size': int(item.get('size') or 0)})
    return sorted(files, key=lambda value: value['path'])


def _download_one(session, mirror, repo, item, root, retries=4):
    path = item['path']
    destination = root / Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = int(item.get('size') or 0)
    if destination.is_file() and (expected <= 0 or destination.stat().st_size == expected):
        return 'skipped', destination, 0

    part = destination.with_suffix(destination.suffix + '.part')
    url = f'{mirror.rstrip("/")}/datasets/{repo}/resolve/main/{quote(path, safe="/")}'
    last_error = None
    for attempt in range(retries):
        offset = part.stat().st_size if part.is_file() else 0
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                if offset and response.status_code == 200:
                    offset = 0
                    part.unlink(missing_ok=True)
                response.raise_for_status()
                if 'text/html' in response.headers.get('Content-Type', '').lower():
                    raise RuntimeError('mirror returned HTML instead of a data file')
                mode = 'ab' if offset and response.status_code == 206 else 'wb'
                written = 0
                with part.open(mode) as handle:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:
                            handle.write(chunk)
                            written += len(chunk)
                if expected > 0 and part.stat().st_size != expected:
                    raise IOError(
                        f'incomplete file: {part.stat().st_size}/{expected} bytes'
                    )
                part.replace(destination)
                return 'downloaded', destination, written
        except (requests.RequestException, OSError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f'failed to download {path}: {last_error}')


def download(repo=DEFAULT_REPO, root=AMASS, mirror=DEFAULT_MIRROR, workers=8):
    """Download all G1 NPZ files from the mirror with resume support."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    files = remote_files(repo, mirror)
    if not files:
        raise RuntimeError('mirror returned no NPZ files')

    total_bytes = sum(item['size'] for item in files)
    print(json.dumps({
        'event': 'download_started',
        'repo': repo,
        'mirror': mirror,
        'root': str(root),
        'files': len(files),
        'bytes': total_bytes,
    }, ensure_ascii=False), flush=True)

    counts = {'downloaded': 0, 'skipped': 0, 'failed': 0}
    downloaded_bytes = 0
    lock = threading.Lock()

    def task(item):
        nonlocal downloaded_bytes
        status, path, written = _download_one(
            _session(), mirror, repo, item, root
        )
        with lock:
            counts[status] += 1
            downloaded_bytes += written
            current = counts['downloaded'] + counts['skipped'] + counts['failed']
            if current % 20 == 0 or current == len(files):
                print(json.dumps({
                    'event': 'progress',
                    'done': current,
                    'total': len(files),
                    'downloaded': counts['downloaded'],
                    'skipped': counts['skipped'],
                    'failed': counts['failed'],
                    'bytes': downloaded_bytes,
                    'latest': str(path),
                }, ensure_ascii=False), flush=True)
        return status, path

    errors = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(task, item) for item in files]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                with lock:
                    counts['failed'] += 1
                errors.append(str(exc))

    report = {
        'event': 'download_finished',
        'root': str(root),
        'files': len(files),
        'downloaded': counts['downloaded'],
        'skipped': counts['skipped'],
        'failed': counts['failed'],
        'bytes': total_bytes,
        'errors': errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise RuntimeError(f'{len(errors)} files failed to download')
    return report


def inventory(root=AMASS):
    """Summarize downloaded AMASS files."""
    root = Path(root)
    files = sorted(path for path in root.rglob('*.npz') if '.cache' not in path.parts)
    total = sum(path.stat().st_size for path in files)
    groups = {}
    for path in files:
        relative = path.relative_to(root)
        key = relative.parts[1] if relative.parts[0] == 'g1' and len(relative.parts) > 1 else relative.parts[0]
        entry = groups.setdefault(key, {'files': 0, 'bytes': 0})
        entry['files'] += 1
        entry['bytes'] += path.stat().st_size
    report = {
        'root': str(root),
        'files': len(files),
        'bytes': total,
        'groups': groups,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _motion_name(path):
    name = Path(path).stem
    for suffix in ('_poses_120_jpos', '_poses_jpos', '_poses'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def _catalog_record(path, root):
    path = Path(path)
    relative = path.relative_to(root)
    parts = relative.parts
    dataset = parts[0] if len(parts) > 1 else 'unknown'
    subject = parts[1] if len(parts) > 2 else ''
    name = _motion_name(path)
    motion_id = sha1(relative.as_posix().encode('utf-8')).hexdigest()[:12]
    frames = 0
    duration = 0.0
    fps = 0.0
    try:
        with np.load(path, allow_pickle=False) as z:
            times = np.asarray(z['times']).reshape(-1)
            frames = len(times)
            if frames:
                duration = float(times[-1] - times[0])
            if 'source_fps' in z.files:
                fps = float(np.asarray(z['source_fps']).reshape(-1)[0])
    except (OSError, ValueError, KeyError):
        pass
    return {
        'motion_id': motion_id,
        'dataset': dataset,
        'subject': subject,
        'name': name,
        'frames': frames,
        'duration_seconds': round(duration, 4),
        'fps': fps,
        'relative_path': relative.as_posix(),
        'path': str(path.resolve()),
    }


def build_catalog(root=AMASS_MOTIONS, force=True):
    """Build the stable motion index used by DeepSeek tools."""
    root = Path(root)
    files = sorted(path for path in root.rglob('*.npz')
                   if '.cache' not in path.parts and path.name != CATALOG_NAME)
    records = [_catalog_record(path, root) for path in files]
    ids = [record['motion_id'] for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError('AMASS motion_id collision')
    groups = Counter(record['dataset'] for record in records)
    durations = {}
    for record in records:
        durations[record['dataset']] = durations.get(record['dataset'], 0.0) + record['duration_seconds']
    payload = {
        'version': 1,
        'root': str(root.resolve()),
        'count': len(records),
        'datasets': {
            name: {
                'motions': count,
                'duration_seconds': round(durations.get(name, 0.0), 4),
            }
            for name, count in sorted(groups.items())
        },
        'motions': records,
    }
    root.mkdir(parents=True, exist_ok=True)
    destination = root / CATALOG_NAME
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding='utf-8')
    os.replace(temporary, destination)
    return payload


def load_catalog(root=AMASS_MOTIONS, refresh=False):
    root = Path(root)
    destination = root / CATALOG_NAME
    if refresh or not destination.is_file():
        return build_catalog(root, force=True)
    try:
        payload = json.loads(destination.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return build_catalog(root, force=True)
    if not isinstance(payload, dict) or not isinstance(payload.get('motions'), list):
        return build_catalog(root, force=True)
    return payload


def dataset_inventory(root=AMASS_MOTIONS, refresh=False):
    payload = load_catalog(root, refresh=refresh)
    return {
        'root': payload['root'],
        'count': payload['count'],
        'datasets': payload['datasets'],
    }


def list_motions(dataset=None, query=None, limit=50, offset=0,
                 root=AMASS_MOTIONS, refresh=False):
    payload = load_catalog(root, refresh=refresh)
    motions = payload['motions']
    if dataset:
        wanted = str(dataset).strip().lower()
        motions = [item for item in motions
                   if item['dataset'].lower() == wanted]
    if query:
        needle = str(query).strip().lower()
        motions = [item for item in motions
                   if needle in ' '.join((
                       item['name'], item['subject'], item['dataset'],
                       item['relative_path'])).lower()]
    total = len(motions)
    offset = max(0, int(offset))
    limit = min(200, max(1, int(limit)))
    page = motions[offset:offset + limit]
    return {
        'root': payload['root'],
        'total': total,
        'offset': offset,
        'limit': limit,
        'datasets': payload['datasets'],
        'motions': page,
    }


def resolve_motion(motion_id=None, dataset=None, name=None, path=None,
                   query=None, root=AMASS_MOTIONS):
    payload = load_catalog(root)
    motions = payload['motions']
    candidates = motions
    if motion_id:
        candidates = [item for item in candidates
                      if item['motion_id'] == str(motion_id)]
    if dataset:
        wanted = str(dataset).strip().lower()
        candidates = [item for item in candidates
                      if item['dataset'].lower() == wanted]
    if name:
        wanted = _motion_name(str(name)).lower()
        candidates = [item for item in candidates
                      if item['name'].lower() == wanted]
    if query:
        needle = str(query).strip().lower()
        candidates = [item for item in candidates
                      if needle in ' '.join((
                          item['name'], item['subject'], item['dataset'],
                          item['relative_path'])).lower()]
    if path:
        requested = Path(path).expanduser()
        if not requested.is_absolute():
            requested = Path(root) / requested
        requested = requested.resolve()
        try:
            requested.relative_to(Path(root).resolve())
        except ValueError:
            raise ValueError('AMASS path must be inside %s' % root)
        if not requested.is_file():
            raise FileNotFoundError(requested)
        candidates = [item for item in candidates
                      if Path(item['path']).resolve() == requested]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError('没有匹配的 AMASS 动作，请先调用 robot_amass_list 检索')
    sample = [{'motion_id': item['motion_id'],
               'dataset': item['dataset'],
               'name': item['name'],
               'subject': item['subject']}
              for item in candidates[:12]]
    raise ValueError('匹配到多个 AMASS 动作，请使用 motion_id 精确选择：%s'
                     % json.dumps(sample, ensure_ascii=False))


def _normalize_quaternion(values):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError('zero-length root quaternion')
    values = values / norms
    for index in range(1, len(values)):
        if float(values[index] @ values[index - 1]) < 0.0:
            values[index] *= -1.0
    return values


def _contacts_for_qpos(model, data, qpos, threshold=0.025):
    feet = {
        model.body('left_ankle_roll_link').id,
        model.body('right_ankle_roll_link').id,
    }
    spheres = [
        geom for geom in range(model.ngeom)
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE
        and int(model.geom_bodyid[geom]) in feet
    ]
    contacts = np.zeros((len(qpos), 2), bool)
    if not spheres:
        return contacts
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for side, body in enumerate((
            model.body('left_ankle_roll_link').id,
            model.body('right_ankle_roll_link').id,
        )):
            values = [
                data.geom_xpos[geom, 2] - model.geom_size[geom, 0]
                for geom in spheres
                if int(model.geom_bodyid[geom]) == body
            ]
            contacts[frame, side] = bool(values and min(values) < threshold)
    return contacts


def _foot_bottom_series(model, data, qpos):
    feet = {
        model.body('left_ankle_roll_link').id,
        model.body('right_ankle_roll_link').id,
    }
    spheres = [
        geom for geom in range(model.ngeom)
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE
        and int(model.geom_bodyid[geom]) in feet
    ]
    if not spheres:
        return np.zeros(len(qpos), dtype=np.float64)
    values = np.empty(len(qpos), dtype=np.float64)
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        values[frame] = min(
            data.geom_xpos[geom, 2] - model.geom_size[geom, 0]
            for geom in spheres
        )
    return values


def convert(src, dst, model=None, names=None, want_contacts=False):
    """Convert one mirror NPZ to the project's fullbody reference format."""
    src = Path(src)
    dst = Path(dst)
    with np.load(src, allow_pickle=False) as z:
        required = ('fps', 'dof_names', 'dof_positions', 'body_names',
                    'body_positions', 'body_rotations')
        missing = [key for key in required if key not in z.files]
        if missing:
            raise ValueError(f'{src}: missing keys {missing}')
        fps = float(np.asarray(z['fps']).reshape(-1)[0])
        dof_names = [str(value) for value in z['dof_names']]
        body_names = [str(value) for value in z['body_names']]
        dof_positions = np.asarray(z['dof_positions'], dtype=np.float64)
        body_positions = np.asarray(z['body_positions'], dtype=np.float64)
        body_rotations = np.asarray(z['body_rotations'], dtype=np.float64)

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f'{src}: invalid fps')
    if body_names[0] != 'pelvis':
        raise ValueError(f'{src}: first body must be pelvis, got {body_names[0]!r}')
    if len(dof_positions) < 2:
        raise ValueError(f'{src}: motion needs at least 2 frames')

    if model is None or names is None:
        with tempfile.TemporaryDirectory(prefix='g1_amass_model_') as folder:
            model = build_model(Path(folder))
            names = joint_info(model)[0]
            return convert(src, dst, model=model, names=names,
                           want_contacts=want_contacts)

    expected_names = joint_info(model)[0]
    if dof_names != expected_names:
        raise ValueError(f'{src}: joint order does not match the G1 model')

    frames = len(dof_positions)
    # The mirror files are produced by IsaacLab and store root rotations as
    # wxyz, despite the older README describing the fleaven xyzw layout.
    quat_wxyz = _normalize_quaternion(body_rotations[:, 0, :])
    root_position = body_positions[:, 0, :]

    ranges = model.jnt_range[model.actuator_trnid[:, 0].astype(int)]
    clipped = np.clip(dof_positions, ranges[:, 0] + 1e-5, ranges[:, 1] - 1e-5)
    clipped_count = int(np.count_nonzero(np.abs(clipped - dof_positions) > 1e-7))

    qpos = np.concatenate((root_position, quat_wxyz, clipped), axis=1)
    data = mujoco.MjData(model)
    foot_bottom = _foot_bottom_series(model, data, qpos)
    ground_offset = -float(np.median(foot_bottom))
    qpos[:, 2] += ground_offset

    qvel = np.zeros((frames, model.nv), dtype=np.float64)
    dt = 1.0 / fps
    for index in range(frames - 1):
        mujoco.mj_differentiatePos(model, qvel[index], dt, qpos[index], qpos[index + 1])
    qvel[-1] = qvel[-2]
    times = np.arange(frames, dtype=np.float64) / fps

    contacts = None
    if want_contacts:
        contacts = _contacts_for_qpos(model, data, qpos)

    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'times': times,
        'qpos': qpos,
        'qvel': qvel,
        'joint_names': np.asarray(expected_names, dtype='U64'),
        'ik_error': np.zeros(frames, dtype=np.float64),
        'rejected': np.zeros(frames, dtype=bool),
        'version': 2,
        'root_mode': 'contact_grounded',
        'root_scale': 1.0,
        'root_track_method': 'amass_retarget',
        'root_track_fallback': '',
        'source': str(src.resolve()),
        'source_fps': fps,
        'clipped_joint_values': clipped_count,
        'ground_offset_m': ground_offset,
    }
    if contacts is not None:
        payload['contacts'] = contacts
    with dst.open('wb') as handle:
        np.savez_compressed(handle, **payload)

    reference = Reference(dst, model)
    report = {
        'source': str(src),
        'output': str(dst),
        'frames': frames,
        'fps': fps,
        'duration': reference.duration,
        'clipped_joint_values': clipped_count,
        'ground_offset_m': ground_offset,
        'contacts': bool(contacts is not None),
    }
    return report


def export(root=AMASS, output=AMASS_MOTIONS, limit=None, overwrite=False,
           want_contacts=False):
    """Convert every downloaded AMASS motion into project references."""
    root = Path(root)
    output = Path(output)
    sources = sorted(path for path in root.rglob('*.npz') if '.cache' not in path.parts)
    if limit is not None:
        sources = sources[:max(0, int(limit))]
    if not sources:
        raise RuntimeError(f'no AMASS NPZ files under {root}')

    with tempfile.TemporaryDirectory(prefix='g1_amass_model_') as folder:
        model = build_model(Path(folder))
        names = joint_info(model)[0]
        records = []
        started = time.monotonic()
        for index, source in enumerate(sources, start=1):
            relative = source.relative_to(root)
            if relative.parts and relative.parts[0] == 'g1':
                relative = Path(*relative.parts[1:])
            destination = output / relative
            if destination.is_file() and not overwrite:
                records.append({'source': str(source), 'output': str(destination),
                                'status': 'skipped'})
                continue
            try:
                record = convert(source, destination, model=model, names=names,
                                 want_contacts=want_contacts)
                record['status'] = 'ok'
            except Exception as exc:
                record = {'source': str(source), 'output': str(destination),
                          'status': 'error', 'error': repr(exc)}
            records.append(record)
            if index % 20 == 0 or index == len(sources):
                print(json.dumps({
                    'event': 'export_progress',
                    'done': index,
                    'total': len(sources),
                    'seconds': round(time.monotonic() - started, 2),
                    'latest': record,
                }, ensure_ascii=False), flush=True)

    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'manifest.json'
    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2),
                        encoding='utf-8')
    catalog = None
    if limit is None:
        catalog = build_catalog(output, force=True)
    errors = [record for record in records if record['status'] == 'error']
    report = {
        'root': str(root),
        'output': str(output),
        'total': len(records),
        'ok': sum(record['status'] == 'ok' for record in records),
        'skipped': sum(record['status'] == 'skipped' for record in records),
        'errors': len(errors),
        'manifest': str(manifest),
        'catalog_motions': (catalog['count'] if catalog else None),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise RuntimeError(f'{len(errors)} AMASS files failed to convert')
    return report
