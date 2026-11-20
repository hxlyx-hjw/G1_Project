"""MediaPipe PoseLandmarker 逐帧检测。"""
import os
from pathlib import Path

import numpy as np


def _write_payload(output, times, points, visibility, source_fps):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp.npz')
    try:
        np.savez_compressed(
            temporary, times=times, points=points,
            visibility=visibility, source_fps=source_fps)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def detect_frames(video, fps, max_seconds, model_path, close_detector=True,
                  result_handler=None, exit_after_handler=False, progress=None,
                  landmark_space='world'):
    """返回 times/points/visibility/source_fps。

    ``close_detector=False`` 供子进程使用：当前 MediaPipe/Windows 组合下
    ``PoseLandmarker.close()`` 可能阻塞数十秒，子进程写盘后直接 os._exit
    可绕过该问题。

    ``result_handler(times, points, visibility, source_fps)`` 在检测循环内、
    MediaPipe 对象仍然存活时调用。子进程通过该回调先落盘再 ``os._exit``，
    避免函数返回时局部 ``PoseLandmarker`` 析构触发数十秒阻塞。

    ``progress(current, total)`` 在每个采样帧后调用，用于 CLI 或训练状态
    展示骨架提取进度。
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    video = Path(video).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if landmark_space not in ('world', 'image'):
        raise ValueError('landmark_space 必须为 world 或 image')
    cap = cv2.VideoCapture(str(video))
    detector = None
    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not np.isfinite(source_fps) or source_fps <= 0:
            raise ValueError('视频无法解码或没有有效帧率')
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = (frame_count / source_fps
                    if np.isfinite(frame_count) and frame_count > 0 else max_seconds)
        total_samples = max(1, int(min(float(max_seconds), float(duration)) * fps) + 1)
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=.5, min_pose_presence_confidence=.5,
            min_tracking_confidence=.5)
        times, points, visibility = [], [], []
        sampled = 0
        index, next_sample, last_ms = 0, 0., -1
        detector = vision.PoseLandmarker.create_from_options(options)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = index / source_fps
            index += 1
            if t > max_seconds:
                break
            if t + 1e-9 < next_sample:
                continue
            next_sample = t + 1 / fps
            timestamp = max(last_ms + 1, round(t * 1000))
            last_ms = timestamp
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector.detect_for_video(image, timestamp)
            if landmark_space == 'image':
                lm = result.pose_landmarks[0] if result.pose_landmarks else None
                if lm is not None:
                    points.append([[p.x, p.y, 0.] for p in lm])
                    visibility.append([p.visibility for p in lm])
                else:
                    points.append(np.zeros((33, 3)))
                    visibility.append(np.zeros(33))
            else:
                if result.pose_world_landmarks:
                    lm = result.pose_world_landmarks[0]
                    points.append([[p.x, p.y, p.z] for p in lm])
                    visibility.append([p.visibility for p in lm])
                else:
                    points.append(np.zeros((33, 3)))
                    visibility.append(np.zeros(33))
            times.append(t)
            sampled += 1
            if progress is not None:
                progress(sampled, total_samples)
        if len(times) < 2:
            raise ValueError('视频至少需要两帧可解码采样')
        if progress is not None and total_samples != len(times):
            progress(len(times), len(times))
        payload = (np.asarray(times, float), np.asarray(points, float),
                   np.asarray(visibility, float), float(source_fps))
        if result_handler is not None:
            result_handler(*payload)
            if exit_after_handler:
                os._exit(0)
        return payload
    finally:
        cap.release()
        if close_detector and detector is not None:
            detector.close()
