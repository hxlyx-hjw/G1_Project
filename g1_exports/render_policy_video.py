"""把训练好的 G1 策略回放渲染成 MP4。

默认沿用此前的导出位置，也可以全部通过命令行覆盖::

    python render_policy_video.py --help
    python render_policy_video.py --run <导出目录> --checkpoint latest.pt
    python render_policy_video.py --run <目录> --max-seconds 5 --output preview.mp4

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


def overlay(cv2, frame, iteration, elapsed, info, title):
    height, width = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (width, 62), (20, 25, 32), -1)
    cv2.rectangle(panel, (0, height - 96), (width, height), (20, 25, 32), -1)
    frame[:] = cv2.addWeighted(panel, 0.76, frame, 0.24, 0)
    cv2.putText(
        frame,
        title,
        (22, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (235, 242, 250),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "iteration %d" % iteration,
        (max(22, width - 250), 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (140, 210, 255),
        1,
        cv2.LINE_AA,
    )
    motion = "AIRBORNE" if info.get("airborne") else "CONTACT"
    status = "FALLEN" if info.get("fallen") else motion
    color = (90, 110, 255) if info.get("fallen") else (120, 235, 140)
    cv2.putText(
        frame,
        "t=%05.2fs  %s" % (elapsed, status),
        (22, height - 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "joint MSE %.4f   root error %.3fm   balance %.3f"
        % (
            info.get("joint_mse", 0.0),
            info.get("root_xy_error_m", 0.0),
            info.get("balance_margin", 0.0),
        ),
        (22, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 228, 236),
        1,
        cv2.LINE_AA,
    )
    progress = min(
        1.0,
        elapsed / max(0.01, info.get("max_episode_seconds", elapsed + 0.01)),
    )
    cv2.rectangle(
        frame,
        (0, height - 4),
        (int(width * progress), height),
        (90, 180, 255),
        -1,
    )
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="回放 G1 训练策略并渲染 MP4")
    parser.add_argument(
        "--package",
        default=os.environ.get("G1_CPU_PACKAGE", str(DEFAULT_PACKAGE)),
        help="G1-CPU-Training-Package 目录，默认 %(default)s",
    )
    parser.add_argument(
        "--run",
        default=os.environ.get("G1_EXPORT_RUN", str(DEFAULT_RUN)),
        help="包含 reference.npz 和 checkpoint 的导出目录，默认 %(default)s",
    )
    parser.add_argument("--checkpoint", help="策略权重；默认 <run>/latest.pt")
    parser.add_argument("--output", help="输出 MP4；默认 <run>/test7_latest_720p.mp4")
    parser.add_argument("--poster", help="封面 PNG；默认与输出同名的 _poster.png")
    parser.add_argument("--report", help="报告 JSON；默认与输出同名的 .report.json")
    parser.add_argument("--max-seconds", type=float, default=62.0, help="最长回放秒数")
    parser.add_argument("--distance", type=float, default=2.9, help="等距相机距离")
    parser.add_argument("--title", default="G1 POLICY REPLAY")
    return parser


def prepare_runtime(parser: argparse.ArgumentParser, package: Path) -> None:
    code_dir = package / "code"
    required = (
        code_dir / "fullbody",
        code_dir / "coordination" / "video.py",
        code_dir / "runtime_io.py",
        code_dir / "viewer_camera.py",
    )
    if not all(path.exists() for path in required):
        parser.error(
            "训练包代码不完整：%s（需要 code/fullbody、code/coordination、"
            "runtime_io.py 和 viewer_camera.py）" % package
        )

    os.environ.setdefault("G1_DATA_ROOT", str(package / "runtime"))
    os.environ.setdefault(
        "MUJOCO_MENAGERIE", str(package / "assets" / "mujoco_menagerie")
    )
    os.environ.setdefault("G1_POLICY_PATH", str(package / "assets" / "motion.pt"))
    entry = str(code_dir)
    if entry not in sys.path:
        sys.path.insert(0, entry)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    package = Path(args.package).expanduser()
    run = Path(args.run).expanduser()
    if not (run / "reference.npz").is_file():
        parser.error("找不到参考动作：%s" % (run / "reference.npz"))
    prepare_runtime(parser, package)

    checkpoint = Path(args.checkpoint) if args.checkpoint else run / "latest.pt"
    if not checkpoint.is_file():
        parser.error("找不到策略权重：%s" % checkpoint)

    import cv2
    import mujoco
    import numpy as np
    import torch
    from fullbody.balanced import BalancedEnv
    from fullbody.ppo import ActorCritic
    from coordination.video import restore_controller
    from runtime_io import load_checkpoint
    from viewer_camera import isometric_camera

    from render_common import open_video_writer

    output = Path(args.output) if args.output else run / "test7_latest_720p.mp4"
    poster = (
        Path(args.poster)
        if args.poster
        else output.with_name(output.stem + "_poster.png")
    )
    report_path = (
        Path(args.report) if args.report else output.with_suffix(".report.json")
    )

    payload = load_checkpoint(checkpoint, map_location="cpu", weights_only=True)
    iteration = int(payload.get("iteration", 0))
    env = BalancedEnv(
        run / "reference.npz",
        run / ("render_model_" + output.stem),
        max_episode_seconds=payload.get("max_episode_seconds"),
    )
    writer = None
    try:
        env.video_leg_weight = payload.get("leg_weight", 0.15)
        env.video_arm_weight = payload.get("arm_weight", 1.0)
        env.airborne_reward = bool(payload.get("airborne_reward", True))
        restore_controller(env, payload)
        policy = ActorCritic.from_checkpoint(payload, eval_mode=True)
        torch.set_num_threads(1)
        observation, _ = env.reset(options={"phase": 0})
        camera = isometric_camera(
            env.model, distance=args.distance, lookat=(0.0, 0.0, 0.62)
        )
        fps = 15.0
        width, height = 960, 720
        duration = min(env.reference.duration, max(0.0, float(args.max_seconds)))
        next_frame = 0.0
        frames = 0
        max_drift = 0.0
        fallen = False
        last_frame = None
        poster_written = False

        writer, codec = open_video_writer(output, fps, (width, height))
        with mujoco.Renderer(env.model, height=height, width=width) as renderer:
            for _ in range(int(np.ceil(duration / 0.02))):
                with torch.no_grad():
                    action = policy.mean(
                        torch.as_tensor(observation).unsqueeze(0)
                    )[0].numpy()
                observation, _, fallen, ended, info = env.step(action)
                max_drift = max(
                    max_drift, float(info.get("root_xy_error_m", 0.0))
                )
                if env.elapsed + 1e-8 >= next_frame or fallen or ended:
                    camera.lookat[:] = env.data.qpos[:3]
                    camera.lookat[2] = 0.62
                    renderer.update_scene(env.data, camera=camera)
                    frame = cv2.cvtColor(
                        renderer.render().copy(), cv2.COLOR_RGB2BGR
                    )
                    frame = overlay(
                        cv2,
                        frame,
                        iteration,
                        env.elapsed,
                        {**info, "fallen": bool(fallen)},
                        args.title,
                    )
                    writer.write(frame)
                    last_frame = frame
                    if not poster_written and frames == int(fps * 5):
                        cv2.imwrite(str(poster), frame)
                        poster_written = True
                    frames += 1
                    next_frame += 1.0 / fps
                if fallen or ended:
                    break

        simulation_seconds = float(env.elapsed)
        reference_completed = bool(
            not fallen and env.phase >= env.reference.duration
        )
    finally:
        if writer is not None:
            writer.release()
        env.close()

    if not poster_written and last_frame is not None:
        cv2.imwrite(str(poster), last_frame)
        poster_written = True

    report = {
        "output": str(output),
        "poster": str(poster) if poster_written else None,
        "checkpoint": str(checkpoint),
        "package": str(package),
        "iteration": iteration,
        "codec": codec,
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "simulation_seconds": simulation_seconds,
        "fallen": bool(fallen),
        "max_root_xy_error_m": max_drift,
        "reference_completed": reference_completed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
