"""g1_exports 渲染脚本的公共辅助函数。

只依赖标准库；OpenCV 在函数内部导入，便于 ``--help`` 等纯参数操作在
没有安装 OpenCV 的环境中也能正常执行。
"""
from __future__ import annotations

import os
from pathlib import Path


DEFAULT_CODEC = "avc1"


def open_video_writer(path, fps, size, preferred=None):
    """创建视频写入器，返回 ``(writer, codec)``。

    优先使用 H.264（``avc1``），失败时回退到广泛可用的 ``mp4v``。
    可通过环境变量 ``G1_EXPORT_CODEC`` 指定首选编码器。
    """
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = preferred or os.environ.get("G1_EXPORT_CODEC") or DEFAULT_CODEC
    candidates = []
    for codec in (preferred, "mp4v"):
        if codec and codec not in candidates:
            candidates.append(codec)

    for codec in candidates:
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), float(fps), tuple(size)
        )
        if writer.isOpened():
            return writer, codec
        writer.release()

    raise RuntimeError(
        "无法创建视频文件：%s（已尝试编码器：%s）。"
        "请检查 OpenCV 的视频后端或安装 OpenH264。" % (path, ", ".join(candidates))
    )
