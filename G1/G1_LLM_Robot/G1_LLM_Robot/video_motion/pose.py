"""MediaPipe Tasks 视频模式 + SMPL-X 运动学拟合。

默认使用精度更高的 PoseLandmarker Full 模型，对逐帧世界关键点做置信度
感知的插值与 Savitzky-Golay 时序平滑，再以 ``D:\\SMPL`` 的 SMPL-X v1.1
权重做逐帧运动学拟合（``backend=auto`` 优先 SMPL-X，失败回退 MediaPipe）。
接口仍输出 ``times/points/visibility``，旧的重定向、训练和技能代码无需改格式。
"""
from pathlib import Path
import os
import urllib.request
import numpy as np

MODEL_URLS = {
    'lite': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
    'full': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
    'heavy': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',
}
_QUALITY_ALIASES = {'lite': 'lite', 'full': 'full', 'heavy': 'heavy',
                    'medium': 'full', 'balanced': 'full'}


def _quality(value=None):
    value = (value or os.environ.get('G1_POSE_QUALITY', 'full')).strip().lower()
    return _QUALITY_ALIASES.get(value, 'full')


def _download(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.download')
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'G1VideoLearning/1.0'})
        with urllib.request.urlopen(request, timeout=45) as response, temporary.open('wb') as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
        if temporary.stat().st_size < 100000:
            raise RuntimeError('姿态模型下载不完整')
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def ensure_model(path=None, quality=None):
    """返回可用的姿态模型路径。

    ``path`` 显式指定时优先使用；否则按 ``quality`` 选择 full/lite/heavy。
    默认 full 比原 lite 更准；若下载失败会自动回退到已有的 lite 模型。
    """
    if path:
        path = Path(path)
        if path.is_file():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        selected = _quality(quality)
        try:
            _download(MODEL_URLS[selected], path)
            return path
        except Exception as exc:
            raise RuntimeError('无法下载姿态模型到 %s: %s' % (path, exc)) from exc

    selected = _quality(quality)
    order = [selected] + [name for name in ('full', 'lite') if name != selected]
    assets = Path(__file__).parent / 'assets'
    for name in order:
        candidate = assets / ('pose_landmarker_%s.task' % name)
        if candidate.is_file() and candidate.stat().st_size >= 100000:
            return candidate
    errors = []
    for name in order:
        candidate = assets / ('pose_landmarker_%s.task' % name)
        try:
            _download(MODEL_URLS[name], candidate)
            return candidate
        except Exception as exc:
            errors.append('%s: %s' % (name, exc))
    raise RuntimeError('姿态模型不可用；请在联网环境运行 setup 或设置 G1_POSE_MODEL。\n' + '\n'.join(errors))


def stabilize(times, points, visibility, window=5):
    """置信度感知的插值和时序平滑，减少单目深度抖动与遮挡跳变。

    只在有效点段内插值，不把长时完全缺失动作“猜”成有效数据；可见度数组
    原样保留，后续 trim/convert 仍会按可见度筛选可靠帧。
    """
    from scipy.signal import savgol_filter

    times = np.asarray(times, float)
    points = np.asarray(points, float).copy()
    visibility = np.asarray(visibility, float).copy()
    count = len(times)
    if count < 3 or points.shape != (count, 33, 3):
        return points, visibility
    window = int(window)
    if window % 2 == 0:
        window += 1
    window = min(window, count if count % 2 else count - 1)
    if window < 3:
        return points, visibility
    for joint in range(points.shape[1]):
        valid = np.isfinite(points[:, joint]).all(axis=1) & (visibility[:, joint] >= 0.35)
        indices = np.flatnonzero(valid)
        if len(indices) < 3:
            continue
        first, last = int(indices[0]), int(indices[-1])
        segment_length = last - first + 1
        local_window = min(window, segment_length if segment_length % 2 else segment_length - 1)
        if local_window < 3:
            continue
        for axis in range(3):
            values = points[indices, joint, axis]
            interpolated = np.interp(np.arange(count), indices, values)
            points[first:last + 1, joint, axis] = savgol_filter(
                interpolated[first:last + 1], local_window, 2, mode='interp')
    return points, visibility


def _detect_subprocess(video, fps, max_seconds, model_path, progress=None,
                       space='world'):
    """在独立进程中运行 MediaPipe，规避 close() 的长时间阻塞。

    子进程通过 ``G1_PROGRESS current total`` 行报告骨架检测进度；父进程
    实时解析并转发给 ``progress`` 回调。
    """
    import subprocess
    import sys
    import tempfile
    import threading
    import time
    project_root = Path(__file__).resolve().parents[1]
    descriptor, name = tempfile.mkstemp(suffix='.npz', prefix='g1_pose_')
    os.close(descriptor)
    temporary = Path(name)
    try:
        command = [
            sys.executable, '-m', 'video_motion.mediapipe_worker',
            '--video', str(video), '--output', str(temporary),
            '--fps', str(float(fps)), '--max-seconds', str(float(max_seconds)),
            '--model-path', str(model_path), '--space', str(space)]
        environment = os.environ.copy()
        previous = environment.get('PYTHONPATH', '')
        environment['PYTHONPATH'] = str(project_root) + (os.pathsep + previous if previous else '')
        process = subprocess.Popen(
            command, cwd=str(project_root), env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        output = []

        def consume():
            if process.stdout is None:
                return
            for raw in process.stdout:
                line = raw.rstrip('\r\n')
                if line.startswith('G1_PROGRESS '):
                    parts = line.split()
                    if progress is not None and len(parts) >= 3:
                        try:
                            progress(int(parts[1]), int(parts[2]))
                        except Exception:
                            pass
                elif line:
                    output.append(line)
                    if len(output) > 200:
                        del output[:-200]

        reader = threading.Thread(target=consume, name='g1-mediapipe-reader', daemon=True)
        reader.start()
        timeout = max(120., float(max_seconds) * 2. + 60.)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            raise RuntimeError('MediaPipe 检测超时（%.0f 秒）' % timeout)
        finally:
            if process.poll() is None:
                process.kill()
            reader.join(timeout=2.0)
        if code != 0:
            detail = '\n'.join(output[-40:]).strip()
            raise RuntimeError('MediaPipe 检测子进程失败（退出码 %d）：%s'
                               % (code, detail[-2000:]))
        with np.load(temporary, allow_pickle=False) as data:
            times = np.asarray(data['times'], float)
            points = np.asarray(data['points'], float)
            visibility = np.asarray(data['visibility'], float)
            source_fps = float(data['source_fps'])
        return times, points, visibility, source_fps
    finally:
        temporary.unlink(missing_ok=True)


def extract(video, output, fps=15., max_seconds=180., model=None, quality=None,
            backend='auto', smplx_model_dir=None, smplx_gender='neutral',
            smplx_device='auto', smplx_iterations=60, progress=None,
            mediapipe_subprocess=True, detect_progress=None):
    if not np.isfinite([fps, max_seconds]).all() or not 1 <= fps <= 60 or not 0 < max_seconds <= 180:
        raise ValueError('fps 应在 1~60，max_seconds 应在 (0,180]')
    if backend not in ('auto', 'mediapipe', 'smplx'):
        raise ValueError('backend 必须为 auto/mediapipe/smplx')
    video = Path(video).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    model_path = ensure_model(model, quality)
    inprocess = os.environ.get('G1_MEDIAPIPE_INPROCESS', '').strip().lower() in ('1', 'true', 'yes')
    if mediapipe_subprocess and not inprocess:
        times, points, visibility, source_fps = _detect_subprocess(
            video, fps, max_seconds, model_path, progress=detect_progress)
    else:
        from .mediapipe_backend import detect_frames
        times, points, visibility, source_fps = detect_frames(
            video, fps, max_seconds, model_path, close_detector=True,
            progress=detect_progress)
    if not np.any(np.min(visibility[:, [11,12,13,14,15,16,23,24]], axis=1) >= .5):
        raise ValueError('未检测到清晰的肩、肘、腕及髋部，请使用单人无遮挡视频')
    points, visibility = stabilize(
        times, points, visibility, window=5 if fps <= 20 else 7)
    times = np.asarray(times, float)
    points = np.asarray(points, float)
    visibility = np.asarray(visibility, float)
    backend_used = 'mediapipe'
    smplx_data = None
    smplx_error = ''
    if backend != 'mediapipe':
        from .smplx_fit import available, describe, fit_sequence, map_joints_to_mediapipe
        if available(smplx_model_dir):
            try:
                if progress:
                    progress(0, len(times), None)
                smplx_data = fit_sequence(
                    points, visibility, times, model_dir_path=smplx_model_dir,
                    gender=smplx_gender, device=smplx_device,
                    iterations=smplx_iterations, progress=progress)
                fitted, fitted_visibility = map_joints_to_mediapipe(smplx_data, visibility)
                usable = smplx_data['valid']
                points = points.copy()
                visibility = visibility.copy()
                points[usable] = fitted[usable]
                visibility[usable] = fitted_visibility[usable]
                backend_used = 'smplx'
            except Exception as exc:
                if backend == 'smplx':
                    raise
                backend_used = 'mediapipe_fallback'
                smplx_error = str(exc)
        elif backend == 'smplx':
            raise RuntimeError('SMPL-X 不可用，请检查 D:\\SMPL 或 G1_SMPLX_MODEL_DIR：%s'
                               % describe(smplx_model_dir))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        times=times, points=points, visibility=visibility,
        source=str(video), source_fps=float(source_fps),
        sample_fps=float(fps), pose_model=model_path.stem,
        stabilized=True, backend=backend_used)
    if smplx_data is not None:
        payload.update(
            smplx_joints=np.asarray(smplx_data['joints'], np.float32),
            smplx_global_orient=np.asarray(smplx_data['global_orient'], np.float32),
            smplx_body_pose=np.asarray(smplx_data['body_pose'], np.float32),
            smplx_transl=np.asarray(smplx_data['transl'], np.float32),
            smplx_betas=np.asarray(smplx_data['betas'], np.float32),
            smplx_scale=np.float32(smplx_data['scale']),
            smplx_basis=np.asarray(smplx_data['basis'], np.float32),
            smplx_pelvis_ref=np.asarray(smplx_data['pelvis_ref'], np.float32),
            smplx_fit_error=np.asarray(smplx_data['errors'], np.float32),
            smplx_valid=np.asarray(smplx_data['valid'], bool),
            smplx_model=str(smplx_data['model_dir']),
            smplx_gender=str(smplx_data['gender']),
            smplx_device=str(smplx_data['device']),
            smplx_iterations=int(smplx_data['iterations']))
    if smplx_error:
        payload['smplx_error_message'] = smplx_error
    with output.open('wb') as f:
        np.savez_compressed(f, **payload)
    return output


def detect_2d(video, fps=30., max_seconds=180., model=None, quality=None,
              progress=None, mediapipe_subprocess=True):
    """Return normalized image-space MediaPipe landmarks for multi-view DLT.

    The returned points are ``(T, 33, 3)`` with ``x/y`` in normalized image
    coordinates and ``z=0``. They are intentionally not world landmarks:
    triangulation needs the original per-camera image observations.
    """
    if not np.isfinite([fps, max_seconds]).all() or not 1 <= fps <= 60 or not 0 < max_seconds <= 180:
        raise ValueError('fps 应在 1~60，max_seconds 应在 (0,180]')
    video = Path(video).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    model_path = ensure_model(model, quality)
    inprocess = os.environ.get('G1_MEDIAPIPE_INPROCESS', '').strip().lower() in ('1', 'true', 'yes')
    if mediapipe_subprocess and not inprocess:
        return _detect_subprocess(video, fps, max_seconds, model_path,
                                  progress=progress, space='image')
    from .mediapipe_backend import detect_frames
    return detect_frames(video, fps, max_seconds, model_path,
                         close_detector=True, progress=progress,
                         landmark_space='image')


def load(path):
    with np.load(path, allow_pickle=False) as data:
        times, points, vis = (np.array(data[k], dtype=float) for k in ('times', 'points', 'visibility'))
    if (times.ndim != 1 or len(times) < 2 or points.shape != (len(times), 33, 3)
            or vis.shape != (len(times), 33) or not all(np.isfinite(a).all() for a in (times, points, vis))
            or np.any(np.diff(times) <= 0) or times[-1] - times[0] > 180.01):
        raise ValueError('动作文件格式错误：需为有限数值、递增时间戳且不超过180秒')
    times -= times[0]
    return times, points, vis
