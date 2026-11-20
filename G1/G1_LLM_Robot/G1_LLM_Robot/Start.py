"""Visual Studio 启动文件：设为启动文件后按 Ctrl+F5/F5。

默认显示菜单；也支持 python Start.py --mode video --headless --seconds 8。
API Key 使用环境变量 DEEPSEEK_API_KEY 或本目录 api_key.txt。
"""
from pathlib import Path
import argparse
import os
import sys

PROJECT_DIR = Path(__file__).resolve().parent
# 可直接修改这几项作为 IDE 默认配置。
DEFAULT_MODE = 'menu'  # menu / video / chat
DEFAULT_VIDEO = PROJECT_DIR / 'vedio' / 'example.mp4'
DEFAULT_SECONDS = 8.0


def _pose_progress(frame, total, error):
    if frame == 0:
        print('SMPL-X 拟合中...', flush=True)
    elif frame == 1 or frame % 10 == 0 or frame == total:
        percent = int(round(100.0 * frame / max(1, total)))
        detail = '' if error is None else '，关节误差 %.3f m' % error
        print('  SMPL-X %d/%d（%d%%）%s' % (frame, total, percent, detail), flush=True)


def _detect_progress(frame, total):
    percent = int(round(100.0 * frame / max(1, total)))
    bucket = percent // 10
    last = getattr(_detect_progress, 'last_bucket', -1)
    if frame == 1 or frame == total or bucket != last:
        _detect_progress.last_bucket = bucket
        print('[骨架检测] %d/%d（%d%%）' % (frame, total, percent), flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description='G1 视频学习 / DeepSeek 聊天启动器')
    parser.add_argument('--mode', choices=('menu','video','chat','ollama','fullbody-retarget','fullbody-train','fullbody-preview','fullbody-eval'), default=DEFAULT_MODE)
    parser.add_argument('--video', type=Path, default=DEFAULT_VIDEO)
    parser.add_argument('--seconds', type=float, default=DEFAULT_SECONDS)
    parser.add_argument('--fps', type=float, default=10, help='姿态提取采样率')
    parser.add_argument('--pose-quality', choices=('lite','full','heavy'), default='full', help='MediaPipe 姿态模型精度')
    parser.add_argument('--pose-backend', choices=('auto','mediapipe','smplx'), default='auto', help='姿态后端；auto 优先 SMPL-X')
    parser.add_argument('--smplx-model-dir', type=Path, default=None, help='SMPL-X 模型 smplx 目录')
    parser.add_argument('--smplx-gender', choices=('neutral','male','female'), default='neutral')
    parser.add_argument('--smplx-device', choices=('auto','cpu','cuda'), default='auto')
    parser.add_argument('--smplx-iterations', type=int, default=60)
    parser.add_argument('--speed', type=float, default=1., help='回放倍率，0.25~2')
    parser.add_argument('--headless', action='store_true', help='不显示3D窗口，适合验证')
    parser.add_argument('--reference', type=Path, default=PROJECT_DIR/'motions/fullbody/reference.npz')
    parser.add_argument('--root-mode', choices=('in_place','contact_grounded','auto'), default='contact_grounded', help='全身参考根轨迹模式')
    parser.add_argument('--refine', action=argparse.BooleanOptionalAction, default=True, help='足底支撑与接触修正，默认开启')
    parser.add_argument('--output', type=Path, default=PROJECT_DIR/'runs/fullbody')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--coordinator',type=Path,help='全身训练启用指定质心协调网络')
    parser.add_argument('--jump-times',type=float,nargs='*',default=[],help='相对参考起点的起跳秒数，需要coordinator')
    parser.add_argument('--leg-weight',type=float,default=.15,help='腿部参考幅度，0~2')
    parser.add_argument('--arm-weight',type=float,default=1.,help='手臂参考幅度，.25~1.5')
    parser.add_argument('--motion-gain',type=float,default=1.35,help='平衡感知动作放大系数，1~2')
    parser.add_argument('--max-episode-seconds',type=float,default=None,help='单回合最长仿真秒数')
    parser.add_argument('--joint-assist',action=argparse.BooleanOptionalAction,default=True,help='每关节神经网络辅助，默认开启')
    parser.add_argument('--joint-assist-gain',type=float,default=.18,help='每关节辅助残差增益，0~2')
    parser.add_argument('--airborne-reward',action=argparse.BooleanOptionalAction,default=True,help='腾空支撑凸包与放宽惩罚，默认开启')
    parser.add_argument('--proactive-interval-minutes',type=float,default=None,help='聊天训练主动报告间隔，默认10分钟')
    parser.add_argument('--no-proactive',action='store_true',help='关闭聊天训练主动报告')
    parser.add_argument('--gif', help='保存仿真GIF路径')
    return parser


def run_video(args):
    from video_motion.pose import extract
    from video_motion.retarget import play
    from robot_bridge import RobotSession
    import json
    video = args.video.resolve()
    motion = PROJECT_DIR / 'vedio' / (video.stem + '.npz')
    print('正在从视频提取上半身动作：', video)
    extract(video, motion, fps=args.fps, max_seconds=args.seconds, quality=args.pose_quality,
            backend=args.pose_backend, smplx_model_dir=args.smplx_model_dir,
            smplx_gender=args.smplx_gender, smplx_device=args.smplx_device,
            smplx_iterations=args.smplx_iterations, progress=_pose_progress,
            detect_progress=_detect_progress)
    with RobotSession(show_viewer=not args.headless, record_frames=bool(args.gif)) as bot:
        report = play(bot, motion, speed=args.speed)
        text = json.dumps(report, ensure_ascii=False, indent=2)
        print(text)
        motion.with_suffix('.report.json').write_text(text, encoding='utf-8')
        if args.gif:
            bot.save_gif(args.gif)
        if not args.headless:
            if bot.viewer_running:
                print('回放结束，关闭3D窗口退出。')
                bot.wait_until_viewer_closed()
            else:
                print('未检测到可用图形显示，已按无界面模式完成回放。')
        return 0 if report['completed'] else 2


def run_chat(args):
    from deepseek_robot import RobotChatBot
    bot = RobotChatBot(
        robot_viewer=not args.headless,
        enable_proactive=not args.no_proactive,
        proactive_interval_seconds=(args.proactive_interval_minutes * 60.0
                                    if args.proactive_interval_minutes is not None else None),
    )
    try:
        print('示例视频：', DEFAULT_VIDEO)
        print('可以输入：学习 vedio/example.mp4 前8秒的上半身动作并回放。')
        bot.run_cli()
    finally:
        bot.close()
    return 0


def run_ollama(args):
    from deepseek_robot import RobotChatBot
    base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
    model = os.environ.get('OLLAMA_MODEL', 'llama:8b')
    bot = RobotChatBot(
        deepseek_api_key='ollama',      # Ollama 无需 key，占位即可
        model=model,
        base_url=base_url,
        thinking=False,                 # Ollama 不支持 DeepSeek 思考模式
        enable_robot=False,             # 本地 8b 模型工具调用不可靠，先纯对话
        enable_local_files=False,
        system_prompt='你是一个运行在本地 Ollama 的 AI 助手，简洁、直接地回答用户问题。',
    )
    try:
        print('Ollama 本地模型:', model)
        print('接口地址:', base_url)
        print('提示：若模型名报错，请运行 ollama list 查看已装模型，并设置 OLLAMA_MODEL。')
        bot.run_cli()
    finally:
        bot.close()
    return 0


def run_fullbody(args):
    from fullbody.__main__ import main as fullbody_main
    if args.mode == 'fullbody-retarget':
        from video_motion.pose import extract
        pose = PROJECT_DIR/'vedio'/ (args.video.stem+'.npz')
        extract(args.video,pose,fps=args.fps,max_seconds=args.seconds,quality=args.pose_quality,
            backend=args.pose_backend,smplx_model_dir=args.smplx_model_dir,
            smplx_gender=args.smplx_gender,smplx_device=args.smplx_device,
            smplx_iterations=args.smplx_iterations,progress=_pose_progress,
            detect_progress=_detect_progress)
        command=['retarget',str(pose),'--output',str(args.reference),'--root-mode',args.root_mode]
        command += ['--refine' if args.refine else '--no-refine']
        return fullbody_main(command)
    if args.mode == 'fullbody-train':
        if args.coordinator:
            from fullbody.train_balanced import train
            config={'version':1,'coordinator_checkpoint':str(args.coordinator.resolve()),'jump_times':args.jump_times,'jump_mode':'explicit' if args.jump_times else 'off'}
            if any(t<.7 for t in args.jump_times): raise ValueError('起跳时刻至少0.7秒')
            train(args.reference,args.output,iterations=args.iterations,num_envs=1,horizon=args.steps,device=args.device,
                  resume=args.resume,video_controller=config,leg_weight=args.leg_weight,arm_weight=args.arm_weight,
                  motion_gain=args.motion_gain,max_episode_seconds=args.max_episode_seconds,
                  joint_assist=args.joint_assist,joint_assist_gain=args.joint_assist_gain,airborne_reward=args.airborne_reward)
            return 0
        if args.jump_times: raise ValueError('跳跃视频训练需要--coordinator模型')
        command=['train',str(args.reference),'--output',str(args.output),'--iterations',str(args.iterations),
                 '--steps',str(args.steps),'--device',args.device]
        command += ['--joint-assist' if args.joint_assist else '--no-joint-assist',
                    '--joint-assist-gain',str(args.joint_assist_gain),
                    '--airborne-reward' if args.airborne_reward else '--no-airborne-reward']
        if args.resume: command += ['--resume',str(args.resume)]
        return fullbody_main(command)
    command=['preview' if args.mode=='fullbody-preview' else 'evaluate',str(args.reference),
             '--output',str(args.output/('preview' if args.mode=='fullbody-preview' else 'evaluation'))]
    if not args.headless: command += ['--viewer']
    if args.mode=='fullbody-eval':
        checkpoint=args.checkpoint or args.output/'policy.pt'
        if checkpoint.is_file(): command += ['--checkpoint',str(checkpoint)]
        elif args.checkpoint: raise FileNotFoundError(checkpoint)
        else: print('未找到训练权重，评估纯PD基线。')
    return fullbody_main(command)


def main(argv=None):
    # IDE 可从任意工作目录启动，模型、密钥及视频统一按项目目录定位。
    os.chdir(PROJECT_DIR)
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    args = build_parser().parse_args(argv)
    try:
        if args.mode == 'menu':
            print('G1 启动菜单\n  1. 上半身视频示例（默认）\n  2. AI聊天控制机器人\n  3. 视频全身重定向\n  4. 全身参考预览（运动学）\n  5. PPO全身模仿训练\n  6. 全身策略物理评估（未指定权重时为PD基线）\n  7. Ollama 本地对话 (llama:8b)\n  0. 退出')
            choice = input('请选择 [1]：').strip()
            if choice == '0':
                return 0
            modes={'':'video','1':'video','2':'chat','3':'fullbody-retarget','4':'fullbody-preview','5':'fullbody-train','6':'fullbody-eval','7':'ollama'}
            if choice not in modes:
                raise ValueError('请输入 0~7')
            args.mode=modes[choice]
        if args.mode.startswith('fullbody-'):
            return run_fullbody(args)
        if args.mode == 'ollama':
            return run_ollama(args)
        return run_video(args) if args.mode == 'video' else run_chat(args)
    except (KeyboardInterrupt, EOFError):
        print('\n程序已退出。')
        return 0
    except Exception as exc:
        print(f'启动或执行失败：{exc}')
        print('环境检查：python environment_check.py；详细说明见 README.md')
        return 1


if __name__ == '__main__':
    from viewer_camera import finish_process
    finish_process(main())
