"""Synchronized two-view MediaPipe + DLT + optional SMPL-X fitting.

The two cameras may be calibrated with a JSON file, or approximated as two
orthogonal cameras with a user-supplied baseline. The output deliberately uses
the same ``times/points/visibility`` contract as the single-view extractor, so
the existing G1 retargeting and training path needs no format changes.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from .pose import detect_2d, stabilize


VIDEO_SUFFIXES = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
LIST_CANDIDATES = ('list.txt', 'list', 'pairs.txt', 'manifest.txt', 'names.txt')


def _looks_b(path):
    text = Path(path).as_posix().lower()
    return bool(re.search(r'(^|[/_\-])b(?:[_.\-/]|$)', text)
                or 'side' in text or 'lateral' in text)


def _looks_a(path):
    text = Path(path).as_posix().lower()
    return bool(re.search(r'(^|[/_\-])a(?:[_.\-/]|$)', text)
                or 'front' in text or 'forward' in text)


def _resolve_video(folder, value):
    value = str(value).strip().strip('"\'')
    if not value:
        raise ValueError('list 中存在空文件名')
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = folder / candidate
    candidate = candidate.resolve()
    if candidate.is_file():
        return candidate
    matches = [path.resolve() for path in folder.rglob(Path(value).name)
               if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError('文件中找不到视频: %s' % value)


def _sibling_variants(name):
    path = Path(name)
    stem = path.stem
    suffix = path.suffix
    variants = []
    def add(value):
        candidate = Path(value)
        if candidate not in variants:
            variants.append(candidate)
    lower = stem.lower()
    if lower.startswith('a'):
        add(path.with_name('b' + stem[1:] + suffix))
    if lower.startswith('b'):
        add(path.with_name('a' + stem[1:] + suffix))
    for token, replacement in (('front', 'side'), ('frontal', 'side'),
                               ('forward', 'side')):
        if token in lower:
            add(Path(str(path).replace(token, replacement)))
            add(Path(str(path).replace(token.upper(), replacement.upper())))
    # Common a/ and b/ sibling directories.
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if part.lower() == 'a':
            alt = parts.copy(); alt[index] = 'b'; add(Path(*alt))
        if part.lower() == 'b':
            alt = parts.copy(); alt[index] = 'a'; add(Path(*alt))
        if part.lower() in ('front', 'frontal', 'forward'):
            alt = parts.copy(); alt[index] = 'side'; add(Path(*alt))
    return variants


def discover_pairs(folder, list_file=None):
    """Read a list file and return synchronized a/b video pairs.

    Supported list lines:
    - ``a/001.mp4,b/001.mp4``
    - ``a/001.mp4 b/001.mp4``
    - ``a_001.mp4`` (the sibling ``b_001.mp4`` is inferred)
    - a and b names listed separately, grouped by a normalized key
    """
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError('folder 必须是存在的目录: %s' % folder)
    if list_file:
        manifest = Path(list_file)
        if not manifest.is_absolute():
            manifest = folder / manifest
    else:
        manifest = next((folder / name for name in LIST_CANDIDATES
                         if (folder / name).is_file()), None)
        if manifest is None:
            manifests = [p for p in folder.iterdir()
                         if p.is_file() and p.suffix.lower() == '.txt']
            if len(manifests) == 1:
                manifest = manifests[0]
    if manifest is None or not manifest.is_file():
        raise FileNotFoundError('文件夹中未找到 list/list.txt/pairs.txt')
    rows = []
    for raw in manifest.read_text(encoding='utf-8-sig', errors='replace').splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        tokens = [token for token in re.split(r'[,\t ]+', line) if token]
        rows.append(tokens)
    if not rows:
        raise ValueError('list 文件为空')
    pairs = []
    explicit = [row for row in rows if len(row) >= 2]
    singles = [row[0] for row in rows if len(row) == 1]
    for row in explicit:
        first, second = _resolve_video(folder, row[0]), _resolve_video(folder, row[1])
        if _looks_b(first) and _looks_a(second):
            first, second = second, first
        pairs.append({'a': str(first), 'b': str(second),
                      'key': Path(first).stem})
    if singles:
        for value in singles:
            source = _resolve_video(folder, value)
            if _looks_b(source):
                source = _resolve_video(folder, _sibling_variants(source)[0]
                                        if _sibling_variants(source) else value)
            partner = None
            for variant in _sibling_variants(source):
                try:
                    partner = _resolve_video(folder, variant)
                    break
                except FileNotFoundError:
                    continue
            if partner is not None:
                a, b = (source, partner)
                if _looks_b(a) and _looks_a(b):
                    a, b = b, a
                pairs.append({'a': str(a), 'b': str(b), 'key': Path(a).stem})
            else:
                # Keep single-only entries for a later grouped pairing pass.
                pairs.append({'a': str(source), 'b': None, 'key': Path(source).stem})
    if any(pair['b'] is None for pair in pairs):
        # Pair separately listed a/b names by a normalized name key.
        groups = {}
        for pair in pairs:
            if pair['b'] is not None:
                continue
            path = Path(pair['a'])
            key = re.sub(r'(?i)(^|[_\-\/])(a|b|front|frontal|forward|side|lateral)(?=([_\-\/.]|$))',
                         r'\1', path.stem)
            key = key.strip('._-') or path.stem
            groups.setdefault(key, []).append(path)
        paired = [pair for pair in pairs if pair['b'] is not None]
        for key, paths in groups.items():
            a = next((p for p in paths if _looks_a(p)), None)
            b = next((p for p in paths if _looks_b(p)), None)
            if a is None or b is None or len(paths) != 2:
                raise ValueError('无法配对 %s；请在 list 中写成 a,b 两列' % key)
            paired.append({'a': str(a), 'b': str(b), 'key': key})
        pairs = paired
    unique = {}
    for pair in pairs:
        if pair['b'] is None:
            raise ValueError('list 中存在没有侧视角的视频: %s' % pair['a'])
        unique[pair['key']] = pair
    if not unique:
        raise ValueError('list 中没有可用的 a/b 视频对')
    return {
        'folder': str(folder), 'list_file': str(manifest),
        'pairs': sorted(unique.values(), key=lambda pair: pair['key']),
    }


def video_info(path):
    import cv2
    path = Path(path).resolve()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError('无法打开视频: %s' % path)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if not np.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError('视频缺少有效帧率或尺寸: %s' % path)
    return {
        'path': str(path), 'fps': fps, 'width': width, 'height': height,
        'frames': frames, 'duration': frames / fps if frames > 0 else None,
    }


def _matrix(value, shape, name):
    array = np.asarray(value, float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError('%s 必须是有限数值的 %s 矩阵' % (name, shape))
    return array


def _intrinsics(info, fov_degrees):
    width, height = float(info['width']), float(info['height'])
    fov = math.radians(float(fov_degrees))
    if not 5.0 < float(fov_degrees) < 170.0:
        raise ValueError('fov 应在 5~170 度')
    fx = 0.5 * width / math.tan(fov / 2.0)
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fx, height / 2.0],
                     [0.0, 0.0, 1.0]], float)


def _rotation_y(degrees):
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler('y', float(degrees), degrees=True).as_matrix()


def _camera_from_entry(entry, fallback_k, fallback_r, fallback_t, name):
    if not isinstance(entry, dict):
        raise ValueError('%s 必须是相机 JSON 对象' % name)
    if 'P' in entry:
        P = _matrix(entry['P'], (3, 4), name + '.P')
        K = _matrix(entry['K'], (3, 3), name + '.K') if 'K' in entry else None
        R = None
        t = None
    else:
        K = (_matrix(entry['K'], (3, 3), name + '.K')
             if 'K' in entry else fallback_k)
        R = (_matrix(entry['R'], (3, 3), name + '.R')
             if 'R' in entry else fallback_r)
        t = (_matrix(entry['t'], (3,), name + '.t')
             if 't' in entry else fallback_t)
        P = K @ np.column_stack((R, t))
    dist = np.asarray(entry.get('dist', entry.get('D', np.zeros(5))), float).reshape(-1)
    if not np.isfinite(dist).all():
        raise ValueError('%s.dist 含非有限数值' % name)
    return {'K': K, 'R': R, 't': t, 'dist': dist, 'P': P}


def camera_pair(info_a, info_b, calibration=None, fov=60.0, baseline=2.0,
                azimuth_b=90.0):
    """Return two OpenCV projection matrices and calibration metadata."""
    K_a = _intrinsics(info_a, fov)
    K_b = _intrinsics(info_b, fov)
    if calibration:
        data = json.loads(Path(calibration).read_text(encoding='utf-8'))
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict) and isinstance(data.get('cameras'), list):
            entries = data['cameras']
        elif isinstance(data, dict) and 'camera_a' in data and 'camera_b' in data:
            entries = [data['camera_a'], data['camera_b']]
        else:
            raise ValueError('标定 JSON 需要 cameras 数组或 camera_a/camera_b')
        if len(entries) != 2:
            raise ValueError('标定 JSON 必须包含两个相机')
        camera_a = _camera_from_entry(entries[0], K_a, np.eye(3), np.zeros(3), 'camera_a')
        camera_b = _camera_from_entry(entries[1], K_b, None, None, 'camera_b')
        if camera_b['R'] is None or camera_b['t'] is None:
            if 'P' not in entries[1]:
                raise ValueError('camera_b 需要 R/t 或 P')
        mode = 'calibrated'
    else:
        R_b = _rotation_y(azimuth_b)
        center_b = np.array([float(baseline), 0.0, 0.0])
        t_b = -R_b @ center_b
        camera_a = {'K': K_a, 'R': np.eye(3), 't': np.zeros(3),
                    'dist': np.zeros(5), 'P': K_a @ np.column_stack((np.eye(3), np.zeros(3)))}
        camera_b = {'K': K_b, 'R': R_b, 't': t_b,
                    'dist': np.zeros(5), 'P': K_b @ np.column_stack((R_b, t_b))}
        mode = 'assumed_orthogonal'
    if not 0.0 < float(baseline) <= 20.0:
        raise ValueError('baseline 应在 (0,20] 米')
    return camera_a, camera_b, {
        'mode': mode,
        'fov_degrees': float(fov),
        'baseline_m': float(baseline),
        'azimuth_b_degrees': float(azimuth_b),
    }


def _nearest_indices(times, targets):
    indices = np.searchsorted(times, targets, side='left')
    indices = np.clip(indices, 0, len(times) - 1)
    previous = np.clip(indices - 1, 0, len(times) - 1)
    choose_previous = np.abs(targets - times[previous]) < np.abs(times[indices] - targets)
    return np.where(choose_previous, previous, indices)


def common_timeline(times_a, times_b, offset_a=0.0, offset_b=0.0):
    """Return the overlap window, capped by the shorter effective video."""
    times_a = np.asarray(times_a, float) + float(offset_a)
    times_b = np.asarray(times_b, float) + float(offset_b)
    if len(times_a) < 2 or len(times_b) < 2:
        raise ValueError('双视角时间轴至少各需要 2 帧')
    start = max(float(times_a[0]), float(times_b[0]))
    effective_a = float(times_a[-1]) - start
    effective_b = float(times_b[-1]) - start
    if effective_a <= 0 or effective_b <= 0:
        raise ValueError('两个视频没有足够的重叠时间段，请检查同步偏移')
    duration = min(effective_a, effective_b)
    source = 'a' if effective_a <= effective_b else 'b'
    return {
        'policy': 'shortest_common_overlap',
        'start_seconds': start,
        'end_seconds': start + duration,
        'duration_a_seconds': effective_a,
        'duration_b_seconds': effective_b,
        'common_duration_seconds': duration,
        'shorter_source': source,
        'offset_a': float(offset_a),
        'offset_b': float(offset_b),
    }


def synchronize(times_a, points_a, visibility_a, times_b, points_b, visibility_b,
                fps=30.0, offset_a=0.0, offset_b=0.0, return_info=False):
    """Resample two independently extracted sequences onto one common clock."""
    raw_a = np.asarray(times_a, float)
    raw_b = np.asarray(times_b, float)
    info = common_timeline(raw_a, raw_b, offset_a=offset_a, offset_b=offset_b)
    times_a = raw_a + float(offset_a)
    times_b = raw_b + float(offset_b)
    start = info['start_seconds']
    end = info['end_seconds']
    if end - start < 1.0 / float(fps):
        raise ValueError('两个视频没有足够的重叠时间段，请检查同步偏移')
    targets = np.arange(start, end + 1e-9, 1.0 / float(fps))
    index_a = _nearest_indices(times_a, targets)
    index_b = _nearest_indices(times_b, targets)
    result = (targets - start, np.asarray(points_a)[index_a],
              np.asarray(visibility_a)[index_a],
              np.asarray(points_b)[index_b],
              np.asarray(visibility_b)[index_b])
    if return_info:
        return result, info
    return result


def _pixel_points(points, info):
    values = np.asarray(points, float)
    return values[..., :2] * np.array([float(info['width']), float(info['height'])])


def _undistort(points, camera):
    dist = np.asarray(camera.get('dist', []), float).reshape(-1)
    if not len(dist) or not np.any(np.abs(dist) > 1e-12):
        return points
    import cv2
    values = np.asarray(points, float).reshape(-1, 1, 2)
    result = cv2.undistortPoints(values, camera['K'], dist, P=camera['K'])
    return result.reshape(np.asarray(points).shape)


def _project(P, X):
    value = P @ np.r_[X, 1.0]
    if abs(value[2]) < 1e-12:
        return None
    return value[:2] / value[2]


def dlt_point(P_a, P_b, x_a, x_b):
    """Triangulate one corresponding pixel pair; return X, reprojection error."""
    rows = np.vstack((
        x_a[0] * P_a[2] - P_a[0],
        x_a[1] * P_a[2] - P_a[1],
        x_b[0] * P_b[2] - P_b[0],
        x_b[1] * P_b[2] - P_b[1],
    ))
    _, _, vh = np.linalg.svd(rows)
    hom = vh[-1]
    if abs(hom[3]) < 1e-10:
        raise ValueError('DLT 退化')
    X = hom[:3] / hom[3]
    depth_a = float(P_a[2] @ np.r_[X, 1.0])
    depth_b = float(P_b[2] @ np.r_[X, 1.0])
    if depth_a <= 0 or depth_b <= 0:
        raise ValueError('DLT 点在相机后方')
    projected_a = _project(P_a, X)
    projected_b = _project(P_b, X)
    if projected_a is None or projected_b is None:
        raise ValueError('DLT 投影退化')
    error = float(np.linalg.norm(projected_a - x_a)
                  + np.linalg.norm(projected_b - x_b))
    return X, error


def triangulate(points_a, points_b, visibility_a, visibility_b, camera_a, camera_b,
                info_a, info_b, min_visibility=0.4, max_reprojection=40.0):
    """Triangulate all MediaPipe joints and return points/visibility/error."""
    pixel_a = _undistort(_pixel_points(points_a, info_a), camera_a)
    pixel_b = _undistort(_pixel_points(points_b, info_b), camera_b)
    count = len(pixel_a)
    points = np.zeros((count, 33, 3), np.float64)
    visibility = np.zeros((count, 33), np.float64)
    errors = np.full((count, 33), np.nan, np.float64)
    for frame in range(count):
        for joint in range(33):
            confidence = min(float(visibility_a[frame, joint]),
                             float(visibility_b[frame, joint]))
            if confidence < float(min_visibility):
                continue
            try:
                X, error = dlt_point(camera_a['P'], camera_b['P'],
                                     pixel_a[frame, joint], pixel_b[frame, joint])
            except (ValueError, np.linalg.LinAlgError):
                continue
            if error > float(max_reprojection) or not np.isfinite(X).all():
                continue
            points[frame, joint] = X
            visibility[frame, joint] = confidence
            errors[frame, joint] = error
    return points, visibility, errors


def estimate_sync_offset(times_a, points_a, visibility_a, times_b, points_b,
                         visibility_b, camera_a, camera_b, info_a, info_b,
                         offset_a=0.0, offset_b=0.0,
                         max_offset_seconds=12.0, step_seconds=0.5,
                         sample_hz=3.0, min_visibility=0.4,
                         max_reprojection=40.0):
    """Search a temporal offset using two-view triangulation consistency.

    The score rewards joints that triangulate with a valid reprojection error
    and penalizes candidates that discard too much of the shorter clip.
    """
    raw_a = np.asarray(times_a, float)
    raw_b = np.asarray(times_b, float)
    max_offset_seconds = max(0.0, float(max_offset_seconds))
    step_seconds = max(1.0 / 120.0, float(step_seconds))
    sample_hz = max(0.5, float(sample_hz))
    short_raw = min(float(raw_a[-1] - raw_a[0]),
                    float(raw_b[-1] - raw_b[0]))
    base_duration = min(
        short_raw, float(max(raw_a[-1], raw_b[-1]) - min(raw_a[0], raw_b[0])))
    min_duration = max(1.0, 0.2 * short_raw)
    best = None
    pose_pairs = (
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
        (11, 23), (12, 24), (11, 24), (12, 23))

    def pose_signature(points, visibility):
        values = np.asarray(points, float)
        vis = np.asarray(visibility, float)
        count = len(values)
        shoulders = (values[:, 11] + values[:, 12]) / 2.0
        hips = (values[:, 23] + values[:, 24]) / 2.0
        torso = np.linalg.norm(shoulders - hips, axis=1)
        shoulder_width = np.linalg.norm(
            values[:, 11] - values[:, 12], axis=1)
        hip_width = np.linalg.norm(values[:, 23] - values[:, 24], axis=1)
        widths = np.r_[shoulder_width, hip_width]
        widths = widths[np.isfinite(widths) & (widths > 1e-6)]
        fallback = float(np.median(widths)) if len(widths) else 1.0
        torso = np.where(
            np.isfinite(torso) & (torso > 1e-6), torso, fallback)
        features = []
        for first, second in pose_pairs:
            distance = np.linalg.norm(
                values[:, first] - values[:, second], axis=1)
            distance = distance / torso
            usable = ((vis[:, first] >= min_visibility)
                      & (vis[:, second] >= min_visibility)
                      & np.isfinite(distance))
            distance[~usable] = np.nan
            valid = np.flatnonzero(np.isfinite(distance))
            if len(valid) >= 2:
                distance = np.interp(
                    np.arange(count), valid, distance[valid])
            else:
                distance = np.zeros(count, float)
            distance = np.nan_to_num(
                distance, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(distance)
        feature = np.stack(features, axis=1)
        velocity = np.diff(feature, axis=0, prepend=feature[:1])
        def standardize(values):
            center = np.nanmedian(values, axis=0)
            scale = np.nanmedian(np.abs(values - center), axis=0)
            scale = np.where(scale > 1e-6, scale, 1.0)
            return (values - center) / scale
        signature = np.concatenate(
            (standardize(feature), standardize(velocity)), axis=1)
        return np.nan_to_num(signature, nan=0.0, posinf=0.0, neginf=0.0)

    signature_a = pose_signature(points_a, visibility_a)
    signature_b = pose_signature(points_b, visibility_b)
    def motion_weights(points, visibility):
        values = np.asarray(points, float)
        vis = np.asarray(visibility, float)
        weights = np.zeros((len(values), 33), float)
        if len(values) > 1:
            weights[1:] = np.linalg.norm(
                values[1:] - values[:-1], axis=2)
            weights[0] = weights[1]
        weights[vis < min_visibility] = 0.0
        return weights
    motion_a = motion_weights(points_a, visibility_a)
    motion_b = motion_weights(points_b, visibility_b)

    def evaluate(candidate_b):
        try:
            timeline = common_timeline(
                raw_a, raw_b, offset_a=offset_a, offset_b=candidate_b)
        except ValueError:
            return None
        start = timeline['start_seconds']
        end = timeline['end_seconds']
        duration = end - start
        if duration < min_duration:
            return None
        step = 1.0 / sample_hz
        targets = np.arange(start, end + 1e-9, step)
        if len(targets) > 180:
            targets = targets[np.linspace(
                0, len(targets) - 1, 180).astype(int)]
        if len(targets) < 3:
            return None
        shifted_a = raw_a + float(offset_a)
        shifted_b = raw_b + float(candidate_b)
        index_a = _nearest_indices(shifted_a, targets)
        index_b = _nearest_indices(shifted_b, targets)
        _, sampled_vis, _ = triangulate(
            points_a[index_a], points_b[index_b],
            visibility_a[index_a], visibility_b[index_b],
            camera_a, camera_b, info_a, info_b,
            min_visibility=min_visibility,
            max_reprojection=max_reprojection)
        valid_fraction = float(np.mean(sampled_vis >= min_visibility))
        visible_pair = (
            (visibility_a[index_a] >= min_visibility)
            & (visibility_b[index_b] >= min_visibility))
        weights = np.maximum(
            motion_a[index_a], motion_b[index_b]) + 0.02
        weights[~visible_pair] = 0.0
        weight_sum = float(weights.sum())
        triangulated = sampled_vis >= min_visibility
        motion_valid = (
            float((weights * triangulated).sum()) / weight_sum
            if weight_sum > 1e-9 else 0.0)
        pose_difference = np.abs(
            signature_a[index_a] - signature_b[index_b])
        pose_error = float(np.mean(pose_difference))
        pose_score = float(np.exp(-pose_error / 0.8))
        coverage = duration / max(1e-6, base_duration)
        score = (0.82 * pose_score + 0.18 * valid_fraction) * (
            0.55 + 0.45 * min(1.0, coverage))
        return {
            'offset_b': float(candidate_b),
            'valid_fraction': valid_fraction,
            'motion_valid_fraction': motion_valid,
            'pose_error': pose_error,
            'pose_score': pose_score,
            'coverage': float(coverage),
            'duration_seconds': float(duration),
            'score': float(score),
        }

    baseline = evaluate(float(offset_b))
    count = int(round(max_offset_seconds / step_seconds))
    candidates = [float(offset_b)]
    for index in range(1, count + 1):
        delta = index * step_seconds
        candidates.extend([float(offset_b) - delta,
                           float(offset_b) + delta])
    for candidate in candidates:
        result = evaluate(candidate)
        if result is None:
            continue
        if best is None or result['score'] > best['score']:
            best = result
    if best is not None and abs(best['offset_b'] - float(offset_b)) > 1e-9:
        center = best['offset_b']
        fine_step = max(1.0 / 120.0, step_seconds / 5.0)
        for candidate in np.arange(
                center - step_seconds, center + step_seconds + 1e-9,
                fine_step):
            result = evaluate(float(candidate))
            if result is not None and result['score'] > best['score']:
                best = result
    if best is None:
        return {
            'applied': False, 'offset_a': float(offset_a),
            'offset_b': float(offset_b), 'reason': 'no_valid_candidate',
        }
    if baseline is not None and baseline['valid_fraction'] >= 0.55:
        return {
            'applied': False,
            'reason': 'baseline_already_good',
            'offset_a': float(offset_a),
            'offset_b': float(offset_b),
            'baseline_valid_fraction': baseline['valid_fraction'],
            'baseline_pose_error': baseline['pose_error'],
            'coverage': baseline['coverage'],
            'max_offset_seconds': max_offset_seconds,
            'step_seconds': step_seconds,
            'sample_hz': sample_hz,
        }
    applied = False
    if baseline is not None:
        improved_pose = (
            best['pose_error'] <= baseline['pose_error'] * 0.50)
        improved_valid = (
            best['valid_fraction'] >= baseline['valid_fraction'] + 0.05)
        applied = ((improved_pose or improved_valid)
                   and best['coverage'] >= 0.60)
    else:
        applied = best['pose_score'] >= 0.25
    chosen = best if applied else baseline
    if chosen is None:
        chosen = {'offset_b': float(offset_b), 'valid_fraction': 0.0,
                  'coverage': 0.0, 'score': 0.0}
    return {
        'applied': bool(applied),
        'offset_a': float(offset_a),
        'offset_b': float(chosen['offset_b']),
        'searched_offset_b': float(best['offset_b']),
        'baseline_valid_fraction': (
            None if baseline is None else baseline['valid_fraction']),
        'baseline_pose_error': (
            None if baseline is None else baseline['pose_error']),
        'best_valid_fraction': best['valid_fraction'],
        'best_pose_error': best['pose_error'],
        'best_pose_score': best['pose_score'],
        'chosen_valid_fraction': chosen['valid_fraction'],
        'coverage': chosen.get('coverage', 0.0),
        'max_offset_seconds': max_offset_seconds,
        'step_seconds': step_seconds,
        'sample_hz': sample_hz,
    }


def repair_tracks(points, visibility, errors, min_visibility=0.4,
                  max_gap_frames=8, corrected_visibility=0.55):
    """Repair short joint gaps with bounded temporal interpolation.

    Only gaps bracketed by reliable triangulated observations (or short
    leading/trailing gaps) are filled. Longer occlusions stay missing, so the
    downstream retargeter can still carry the previous pose instead of
    inventing a large motion.
    """
    values = np.asarray(points, float).copy()
    vis = np.asarray(visibility, float).copy()
    err = np.asarray(errors, float)
    count = len(values)
    corrected = np.zeros((count, 33), bool)
    max_gap_frames = max(0, int(max_gap_frames))
    if count == 0 or max_gap_frames == 0:
        return values, vis, corrected, {
            'enabled': False, 'max_gap_frames': 0,
            'corrected_joint_cells': 0, 'corrected_frames': 0,
            'corrected_fraction': 0.0}
    for joint in range(33):
        valid = ((vis[:, joint] >= float(min_visibility))
                 & np.isfinite(values[:, joint]).all(axis=1)
                 & np.isfinite(err[:, joint]))
        indices = np.flatnonzero(valid)
        if not len(indices):
            continue
        first, last = int(indices[0]), int(indices[-1])
        for frame in range(max(0, first - max_gap_frames), first):
            values[frame, joint] = values[first, joint]
            corrected[frame, joint] = True
        for left, right in zip(indices[:-1], indices[1:]):
            left, right = int(left), int(right)
            gap = right - left - 1
            if gap <= 0 or gap > max_gap_frames:
                continue
            ratio = (np.arange(left + 1, right) - left) / float(right - left)
            values[left + 1:right, joint] = (
                values[left, joint][None, :] * (1.0 - ratio[:, None])
                + values[right, joint][None, :] * ratio[:, None])
            corrected[left + 1:right, joint] = True
        for frame in range(last + 1, min(count, last + 1 + max_gap_frames)):
            values[frame, joint] = values[last, joint]
            corrected[frame, joint] = True
    vis[corrected] = np.maximum(
        vis[corrected], float(corrected_visibility))
    frame_count = int(np.any(corrected, axis=1).sum()) if count else 0
    cells = int(corrected.sum())
    return values, vis, corrected, {
        'enabled': True,
        'max_gap_frames': max_gap_frames,
        'corrected_joint_cells': cells,
        'corrected_frames': frame_count,
        'corrected_fraction': float(cells / max(1, corrected.size)),
    }


def normalize_scale(points, visibility, target_leg_length=0.85):
    """Scale triangulated points to a plausible human leg length.

    If full legs are unavailable, use a complete thigh+shin chain, then a
    doubled thigh/shin, then an anthropometric torso ratio. A completely
    unscalable sequence is returned unscaled with a warning instead of
    aborting the whole dual-view extraction.
    """
    values = np.asarray(points, float)
    vis = np.asarray(visibility, float)
    lengths = []
    thighs = []
    shins = []
    torsos = []
    def usable(frame, indices):
        return min(float(vis[frame, index]) for index in indices) >= 0.4
    for frame in range(len(values)):
        if usable(frame, (23, 25, 27)):
            lengths.append(
                np.linalg.norm(values[frame, 25] - values[frame, 23])
                + np.linalg.norm(values[frame, 27] - values[frame, 25]))
        if usable(frame, (24, 26, 28)):
            lengths.append(
                np.linalg.norm(values[frame, 26] - values[frame, 24])
                + np.linalg.norm(values[frame, 28] - values[frame, 26]))
        if usable(frame, (23, 25)):
            thighs.append(np.linalg.norm(
                values[frame, 25] - values[frame, 23]))
        if usable(frame, (24, 26)):
            thighs.append(np.linalg.norm(
                values[frame, 26] - values[frame, 24]))
        if usable(frame, (25, 27)):
            shins.append(np.linalg.norm(
                values[frame, 27] - values[frame, 25]))
        if usable(frame, (26, 28)):
            shins.append(np.linalg.norm(
                values[frame, 28] - values[frame, 26]))
        if usable(frame, (11, 12, 23, 24)):
            shoulders = (values[frame, 11] + values[frame, 12]) / 2.0
            hips = (values[frame, 23] + values[frame, 24]) / 2.0
            torsos.append(np.linalg.norm(shoulders - hips))
    def clean(items):
        return np.asarray([
            value for value in items
            if np.isfinite(value) and value > 1e-6])
    lengths = clean(lengths)
    thighs = clean(thighs)
    shins = clean(shins)
    torsos = clean(torsos)
    warning = None
    if len(lengths):
        measured = float(np.median(lengths))
        method = 'side_leg_chain'
    elif len(thighs) or len(shins):
        measured = float(np.median(
            2.0 * np.r_[thighs, shins]))
        method = 'two_times_thigh_or_shin'
    elif len(torsos):
        measured = float(np.median(torsos) * 1.55)
        method = 'torso_ratio_1.55'
    else:
        measured = None
        method = 'unscaled'
        warning = '没有足够的腿部或躯干关节估计尺度，保留三角化原始尺度'
    factor = (float(target_leg_length) / measured
              if measured is not None else 1.0)
    valid = vis >= 0.4
    centers = values[valid]
    center = np.median(centers, axis=0) if len(centers) else np.zeros(3)
    scaled = values.copy()
    scaled[valid] = (values[valid] - center) * factor + center
    info = {
        'median_leg_length': measured,
        'scale_factor': factor,
        'target_leg_length': float(target_leg_length),
        'method': method,
    }
    if warning is not None:
        info['warning'] = warning
    return scaled, info


def _fit_smplx(points, visibility, times, model_dir, gender, device, iterations,
               progress=None):
    from .smplx_fit import available, describe, fit_sequence, map_joints_to_mediapipe
    if not available(model_dir):
        raise RuntimeError('SMPL-X 不可用：%s' % describe(model_dir))
    callback = None
    if progress is not None:
        callback = lambda frame, total, error: progress('smplx_frame', frame, total)
    fit = fit_sequence(points, visibility, times, model_dir_path=model_dir,
                       gender=gender, device=device, iterations=iterations,
                       progress=callback)
    mapped, mapped_visibility = map_joints_to_mediapipe(fit, visibility)
    return fit, mapped, mapped_visibility


def extract(video_a, video_b, output, fps=30.0, max_seconds=60.0, quality=None,
            calibration=None, fov=60.0, baseline=2.0, azimuth_b=90.0,
            offset_a=0.0, offset_b=0.0, min_visibility=0.4,
            max_reprojection=40.0, target_leg_length=0.85,
            self_correct=True, repair_gap_seconds=0.25,
            auto_sync=True, max_sync_offset_seconds=12.0,
            sync_step_seconds=0.5, min_valid_joint_fraction=0.05,
            backend='auto', smplx_model_dir=None, smplx_gender='neutral',
            smplx_device='auto', smplx_iterations=40,
            detect_progress_a=None, detect_progress_b=None, progress=None):
    """Extract, synchronize, triangulate and optionally fit a two-view action."""
    if backend not in ('auto', 'dlt', 'smplx'):
        raise ValueError('backend 必须为 auto/dlt/smplx')
    if not 0.2 <= float(target_leg_length) <= 2.0:
        raise ValueError('target_leg_length 应在 0.2~2.0 米')
    if not 0.0 <= float(min_visibility) <= 1.0:
        raise ValueError('min_visibility 应在 0~1')
    if float(max_reprojection) <= 0:
        raise ValueError('max_reprojection 必须为正数')
    if not 0 < float(max_seconds) <= 180:
        raise ValueError('max_seconds 应在 (0,180]')
    if not 0 <= float(repair_gap_seconds) <= 5:
        raise ValueError('repair_gap_seconds 应在 0~5 秒')
    if not 0 <= float(max_sync_offset_seconds) <= 60:
        raise ValueError('max_sync_offset_seconds 应在 0~60 秒')
    if not 1.0/120.0 <= float(sync_step_seconds) <= 5:
        raise ValueError('sync_step_seconds 应在 1/120~5 秒')
    if not 0.0 <= float(min_valid_joint_fraction) <= 1.0:
        raise ValueError('min_valid_joint_fraction 应在 0~1')
    info_a = video_info(video_a)
    info_b = video_info(video_b)
    durations = [info.get('duration') for info in (info_a, info_b)]
    scan_limit = float(max_seconds)
    if all(isinstance(value, (int, float)) and value > 0
           for value in durations):
        scan_limit = min(
            scan_limit,
            min(float(value) for value in durations)
            + abs(float(offset_a)) + abs(float(offset_b)))
    times_a, points_a, visibility_a, _ = detect_2d(
        video_a, fps=fps, max_seconds=scan_limit, quality=quality,
        progress=detect_progress_a)
    times_b, points_b, visibility_b, _ = detect_2d(
        video_b, fps=fps, max_seconds=scan_limit, quality=quality,
        progress=detect_progress_b)
    camera_a, camera_b, camera_info = camera_pair(
        info_a, info_b, calibration=calibration, fov=fov,
        baseline=baseline, azimuth_b=azimuth_b)
    sync_search = None
    if auto_sync:
        sync_search = estimate_sync_offset(
            times_a, points_a, visibility_a, times_b, points_b, visibility_b,
            camera_a, camera_b, info_a, info_b,
            offset_a=offset_a, offset_b=offset_b,
            max_offset_seconds=max_sync_offset_seconds,
            step_seconds=sync_step_seconds,
            min_visibility=min_visibility,
            max_reprojection=max_reprojection)
        if sync_search.get('applied'):
            offset_b = float(sync_search['offset_b'])
    synchronized, timeline = synchronize(
        times_a, points_a, visibility_a, times_b, points_b, visibility_b,
        fps=fps, offset_a=offset_a, offset_b=offset_b, return_info=True)
    times, first, first_vis, second, second_vis = synchronized
    points, visibility, reprojection = triangulate(
        first, second, first_vis, second_vis, camera_a, camera_b,
        info_a, info_b, min_visibility=min_visibility,
        max_reprojection=max_reprojection)
    max_gap_frames = (max(1, int(round(float(repair_gap_seconds) * float(fps))))
                      if self_correct else 0)
    points, visibility, corrected, correction = repair_tracks(
        points, visibility, reprojection, min_visibility=min_visibility,
        max_gap_frames=max_gap_frames)
    points, scale_info = normalize_scale(points, visibility, target_leg_length)
    points, visibility = stabilize(times, points, visibility, window=5)
    valid_joint_fraction = float(np.mean(visibility >= .4))
    if valid_joint_fraction < float(min_valid_joint_fraction):
        raise ValueError(
            '双视角有效关节比例过低（%.4f < %.4f）；请检查时间偏移、'
            '相机标定或视角重叠。已拒绝将低质量三角化用于训练。'
            % (valid_joint_fraction, float(min_valid_joint_fraction)))
    backend_used = 'multiview_dlt'
    smplx_error = ''
    if backend != 'dlt':
        try:
            if progress:
                progress('smplx', 0, 1)
            fit, points, visibility = _fit_smplx(
                points, visibility, times, smplx_model_dir, smplx_gender,
                smplx_device, smplx_iterations, progress=progress)
            backend_used = 'multiview_smplx'
            if progress:
                progress('smplx', 1, 1)
        except Exception as exc:
            if backend == 'smplx':
                raise
            smplx_error = str(exc)
            fit = None
    else:
        fit = None
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'times': times,
        'points': points,
        'visibility': visibility,
        'source_a': str(Path(video_a).resolve()),
        'source_b': str(Path(video_b).resolve()),
        'source_fps': float(max(info_a['fps'], info_b['fps'])),
        'sample_fps': float(fps),
        'backend': backend_used,
        'stabilized': True,
        'multiview': json.dumps({
            'camera': camera_info,
            'info_a': info_a,
            'info_b': info_b,
            'scale': scale_info,
            'sync': {'offset_a': float(offset_a), 'offset_b': float(offset_b)},
            'auto_sync': sync_search,
            'timeline': timeline,
            'self_correction': correction,
            'min_visibility': float(min_visibility),
            'max_reprojection': float(max_reprojection),
        }, ensure_ascii=False),
        'reprojection_error_px': reprojection,
        'corrected_joints': corrected,
    }
    if smplx_error:
        payload['smplx_error_message'] = smplx_error
    if fit is not None:
        payload.update(
            smplx_joints=np.asarray(fit['joints'], np.float32),
            smplx_global_orient=np.asarray(fit['global_orient'], np.float32),
            smplx_body_pose=np.asarray(fit['body_pose'], np.float32),
            smplx_transl=np.asarray(fit['transl'], np.float32),
            smplx_betas=np.asarray(fit['betas'], np.float32),
            smplx_scale=np.float32(fit['scale']),
            smplx_valid=np.asarray(fit['valid'], bool),
            smplx_model=str(fit['model_dir']),
            smplx_gender=str(fit['gender']),
            smplx_device=str(fit['device']),
            smplx_iterations=int(fit['iterations']))
    with output.open('wb') as handle:
        np.savez_compressed(handle, **payload)
    return {
        'output': str(output),
        'frames': len(times),
        'duration_seconds': float(times[-1] - times[0]),
        'scan_seconds': float(scan_limit),
        'source_duration_a_seconds': durations[0],
        'source_duration_b_seconds': durations[1],
        'timeline': timeline,
        'sync_search': sync_search,
        'self_correction': correction,
        'backend': backend_used,
        'camera_mode': camera_info['mode'],
        'median_reprojection_error_px': float(np.nanmedian(reprojection)),
        'valid_joint_fraction': float(np.mean(visibility >= .4)),
        'min_valid_joint_fraction': float(min_valid_joint_fraction),
        'scale': scale_info,
        'smplx_error': smplx_error,
    }
