"""把 retarget 后的参考动作渲染成 MP4。

默认沿用此前的导出位置，也可以全部通过命令行覆盖::

    python render_reference_video.py --help
    python render_reference_video.py --run <导出目录>
    python render_reference_video.py --run <目录> --max-seconds 5 \
        --output preview.mp4

依赖 ``G1-CPU-Training-Package``（含 fullbody 代码和 runtime）。默认路径只
在本机有效，迁移后请用 ``--package`` 指定新位置。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_PACKAGE = Path(
    r"E:\TEST_RECORD\Server_Deploy\CPU_Server\G1-CPU-Training-Package"
)
DEFAULT_RUN = Path(
    r"E:\TEST_RECORD\G1_Project\g1_exports\g1_test7_20261026_212752"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="渲染 G1 retarget 参考动作视频（MP4）"
    )
    parser.add_argument(
        "--package",
        default=os.environ.get("G1_CPU_PACKAGE", str(DEFAULT_PACKAGE)),
        help="G1-CPU-Training-Package 目录，默认 %(default)s",
    )
    parser.add_argument(
        "--run",
        default=os.environ.get("G1_EXPORT_RUN", str(DEFAULT_RUN)),
        help="包含 reference.npz 的导出目录，默认 %(default)s",
    )
    parser.add_argument("--output", help="输出 MP4；默认 <run>/test7_reference_720p.mp4")
    parser.add_argument("--poster", help="封面 PNG；默认与输出同名的 _poster.png")
    parser.add_argument("--report", help="报告 JSON；默认与输出同名的 .report.json")
    parser.add_argument("--max-seconds", type=float, default=None, help="只渲染前 N 秒")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--distance", type=float, default=2.9, help="等距相机距离")
    parser.add_argument("--title", default="G1 REFERENCE MOTION")
    parser.add_argument("--label", default="test7.mp4", help="右上角来源标签")
    return parser


def prepare_runtime(parser: argparse.ArgumentParser, package: Path) -> Path:
    """校验训练包并把其代码加入 import 路径。返回 code 目录。"""
    code_dir = package / "code"
    if not (code_dir / "fullbody").is_dir() or not (
        code_dir / "viewer_camera.py"
    ).is_file():
        parser.error(
            "训练包代码不完整：%s（需要 code/fullbody 和 code/viewer_camera.py）"
            % package
        )

    os.environ.setdefault("G1_DATA_ROOT", str(package / "runtime"))
    os.environ.setdefault(
        "MUJOCO_MENAGERIE", str(package / "assets" / "mujoco_menagerie")
    )
    entry = str(code_dir)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return code_dir


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    package = Path(args.package).expanduser()
    run = Path(args.run).expanduser()
    reference_path = run / "reference.npz"
    if not reference_path.is_file():
        parser.error("找不到参考动作：%s" % reference_path)
    prepare_runtime(parser, package)

    import cv2
    import mujoco
    import numpy as np
    from fullbody.model import build_model
    from fullbody.reference import Reference
    from viewer_camera import isometric_camera

    from render_common import open_video_writer

    output = Path(args.output) if args.output else run / "test7_reference_720p.mp4"
    poster = (
        Path(args.poster)
        if args.poster
        else output.with_name(output.stem + "_poster.png")
    )
    report_path = (
        Path(args.report) if args.report else output.with_suffix(".report.json")
    )

    model = build_model(run / "render_reference_model")
    data = mujoco.MjData(model)
    reference = Reference(reference_path, model)
    camera = isometric_camera(model, distance=args.distance, lookat=(0.0, 0.0, 0.62))
    duration = reference.duration
    if args.max_seconds is not None:
        duration = min(duration, max(0.0, float(args.max_seconds)))
    fps = float(args.fps)
    width, height = int(args.width), int(args.height)
    frames = 0
    next_frame = 0.0
    last_frame = None
    poster_written = False

    writer, codec = open_video_writer(output, fps, (width, height))
    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for step in range(int(np.ceil(duration / 0.02)) + 1):
                t = min(duration, step * 0.02)
                q, v = reference.sample(t)
                data.qpos[:] = q
                data.qvel[:] = v
                mujoco.mj_forward(model, data)
                if t + 1e-8 >= next_frame or step == 0:
                    camera.lookat[:] = data.qpos[:3]
                    camera.lookat[2] = 0.62
                    renderer.update_scene(data, camera=camera)
                    frame = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)
                    panel = frame.copy()
                    cv2.rectangle(panel, (0, 0), (width, 62), (20, 25, 32), -1)
                    cv2.rectangle(
                        panel, (0, height - 70), (width, height), (20, 25, 32), -1
                    )
                    frame[:] = cv2.addWeighted(panel, 0.76, frame, 0.24, 0)
                    cv2.putText(
                        frame,
                        args.title,
                        (22, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (235, 242, 250),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        args.label,
                        (max(22, width - 170), 39),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (140, 210, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        "t=%05.2fs   retargeted reference   root mode: %s"
                        % (t, reference.root_mode),
                        (22, height - 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (220, 228, 236),
                        1,
                        cv2.LINE_AA,
                    )
                    progress = min(1.0, t / max(0.01, reference.duration))
                    cv2.rectangle(
                        frame,
                        (0, height - 4),
                        (int(width * progress), height),
                        (90, 180, 255),
                        -1,
                    )
                    writer.write(frame)
                    last_frame = frame
                    if not poster_written and frames == int(fps * 5):
                        cv2.imwrite(str(poster), frame)
                        poster_written = True
                    frames += 1
                    next_frame += 1.0 / fps
    finally:
        writer.release()

    if not poster_written and last_frame is not None:
        cv2.imwrite(str(poster), last_frame)
        poster_written = True

    report = {
        "output": str(output),
        "poster": str(poster) if poster_written else None,
        "reference": str(reference_path),
        "package": str(package),
        "codec": codec,
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "duration_seconds": duration,
        "reference_duration_seconds": reference.duration,
        "root_mode": reference.root_mode,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
