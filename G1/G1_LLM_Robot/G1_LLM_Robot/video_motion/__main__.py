"""python -m video_motion {setup,extract,play,learn}"""
import argparse
import importlib
import json
from pathlib import Path


_detect_percent = [-1]


def _detect_progress(frame, total):
    percent = int(round(100.0 * frame / max(1, total)))
    if frame == 1 or frame == total or percent >= _detect_percent[0] + 10:
        _detect_percent[0] = percent
        print('[骨架检测] %d/%d（%d%%）' % (frame, total, percent), flush=True)


def _progress(frame, total, error):
    if frame == 0:
        print('SMPL-X 拟合中...', flush=True)
    elif frame == 1 or frame % 10 == 0 or frame == total:
        percent = int(round(100.0 * frame / max(1, total)))
        detail = '' if error is None else '，关节误差 %.3f m' % error
        print('  SMPL-X %d/%d（%d%%）%s' % (frame, total, percent, detail), flush=True)


def _multiview_progress(stage, current, total):
    percent = int(round(100.0 * current / max(1, total)))
    print('[多视角] %s %d/%d（%d%%）' % (stage, current, total, percent), flush=True)


def main():
    parser = argparse.ArgumentParser(description='从单人视频提取动作并由 G1 模仿上半身姿态')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('setup', help='检查依赖、机器人资源并下载姿态模型')
    for name in ('extract','learn'):
        p = sub.add_parser(name)
        p.add_argument('video')
        p.add_argument('--output', default='motions/video_pose.npz')
        p.add_argument('--fps', type=float, default=15)
        p.add_argument('--quality', choices=('lite','full','heavy'), default=None)
        p.add_argument('--backend', choices=('auto','mediapipe','smplx'), default='auto')
        p.add_argument('--smplx-model-dir', default=None)
        p.add_argument('--smplx-gender', choices=('neutral','male','female'), default='neutral')
        p.add_argument('--smplx-device', choices=('auto','cpu','cuda'), default='auto')
        p.add_argument('--smplx-iterations', type=int, default=60)
        p.add_argument('--max-seconds',type=float,default=30)
        if name == 'learn':
            p.add_argument('--viewer',action='store_true')
            p.add_argument('--gif')
            p.add_argument('--speed',type=float,default=1)
    p = sub.add_parser('multiview', help='同步双视角 MediaPipe 2D + DLT/SMPL-X')
    p.add_argument('video_a')
    p.add_argument('video_b')
    p.add_argument('--output', default='motions/multiview/pose.npz')
    p.add_argument('--fps', type=float, default=30)
    p.add_argument('--max-seconds', type=float, default=60)
    p.add_argument('--quality', choices=('lite','full','heavy'), default=None)
    p.add_argument('--backend', choices=('auto','dlt','smplx'), default='auto')
    p.add_argument('--calibration', default=None)
    p.add_argument('--fov', type=float, default=60)
    p.add_argument('--baseline', type=float, default=2.0)
    p.add_argument('--azimuth-b', type=float, default=90.0)
    p.add_argument('--offset-a', type=float, default=0.0)
    p.add_argument('--offset-b', type=float, default=0.0)
    p.add_argument('--auto-sync', action=argparse.BooleanOptionalAction,
                   default=True, help='自动搜索双视角时间偏移')
    p.add_argument('--max-sync-offset-seconds', type=float, default=12.0)
    p.add_argument('--sync-step-seconds', type=float, default=0.5)
    p.add_argument('--min-visibility', type=float, default=0.4)
    p.add_argument('--max-reprojection', type=float, default=40.0)
    p.add_argument('--min-valid-joint-fraction', type=float, default=0.05)
    p.add_argument('--self-correct', action=argparse.BooleanOptionalAction,
                   default=True, help='对短时缺失关节做有上限的时序修复')
    p.add_argument('--repair-gap-seconds', type=float, default=0.25)
    p.add_argument('--target-leg-length', type=float, default=0.85)
    p.add_argument('--smplx-model-dir', default=None)
    p.add_argument('--smplx-gender', choices=('neutral','male','female'), default='neutral')
    p.add_argument('--smplx-device', choices=('auto','cpu','cuda'), default='auto')
    p.add_argument('--smplx-iterations', type=int, default=40)
    p = sub.add_parser('play')
    p.add_argument('motion')
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--gif')
    p.add_argument('--speed',type=float,default=1)
    args = parser.parse_args()
    if args.command == 'setup':
        failed = []
        for name in ('numpy','scipy','cv2','mediapipe','mujoco','torch','smplx'):
            try:
                module = importlib.import_module(name)
                print(f'OK {name}: {getattr(module,"__version__","available")}')
            except Exception as exc:
                failed.append(name)
                print(f'FAIL {name}: {exc}')
        if failed:
            raise SystemExit('请先安装 requirements-video.txt: '+', '.join(failed))
        from .pose import ensure_model
        from .smplx_fit import describe
        from g1_walkfree.resources import find_menagerie, find_policy
        print('姿态模型:', ensure_model())
        print('SMPL-X:', describe())
        print('机器人模型:', find_menagerie())
        print('行走策略:', find_policy())
        return
    if args.command == 'multiview':
        from .multiview import extract as extract_multiview
        report = extract_multiview(
            args.video_a, args.video_b, args.output, fps=args.fps,
            max_seconds=args.max_seconds, quality=args.quality,
            calibration=args.calibration, fov=args.fov,
            baseline=args.baseline, azimuth_b=args.azimuth_b,
            offset_a=args.offset_a, offset_b=args.offset_b,
            auto_sync=args.auto_sync,
            max_sync_offset_seconds=args.max_sync_offset_seconds,
            sync_step_seconds=args.sync_step_seconds,
            min_visibility=args.min_visibility,
            max_reprojection=args.max_reprojection,
            min_valid_joint_fraction=args.min_valid_joint_fraction,
            self_correct=args.self_correct,
            repair_gap_seconds=args.repair_gap_seconds,
            target_leg_length=args.target_leg_length,
            backend=args.backend, smplx_model_dir=args.smplx_model_dir,
            smplx_gender=args.smplx_gender, smplx_device=args.smplx_device,
            smplx_iterations=args.smplx_iterations,
            detect_progress_a=lambda i,n: _multiview_progress('view_a',i,n),
            detect_progress_b=lambda i,n: _multiview_progress('view_b',i,n),
            progress=_multiview_progress)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    from .pose import extract
    if args.command in ('extract','learn'):
        _detect_percent[0] = -1
        motion = extract(args.video,args.output,args.fps,args.max_seconds,quality=args.quality,
            backend=args.backend,smplx_model_dir=args.smplx_model_dir,
            smplx_gender=args.smplx_gender,smplx_device=args.smplx_device,
            smplx_iterations=args.smplx_iterations,progress=_progress,
            detect_progress=_detect_progress)
        print('动作文件:',motion)
        if args.command == 'extract':
            return
    else:
        motion = Path(args.motion)
    from robot_bridge import RobotSession
    from .retarget import play
    with RobotSession(show_viewer=args.viewer,record_frames=bool(args.gif)) as bot:
        report = play(bot,motion,args.speed)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        report_path = motion.with_suffix('.report.json')
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        if args.gif:
            bot.save_gif(args.gif)
        if args.viewer:
            bot.wait_until_viewer_closed()
        if not report['completed']:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
    from viewer_camera import finish_process
    finish_process(0)
