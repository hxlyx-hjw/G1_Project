"""Run an autonomous web-video training job and wait for completion.

Designed for ``docker run -d``: the launcher starts TrainingJobs.learn_action,
prints JSON status lines to stdout, and exits when the worker reaches a
terminal state.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from fullbody.jobs import TrainingJobs


TERMINAL = {'completed', 'failed', 'cancelled'}


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description='Autonomous video action training')
    parser.add_argument('action', help='动作描述，例如：挥手、原地小跳、向前走')
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='cuda')
    parser.add_argument('--max-seconds', type=float, default=60)
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--max-candidates', type=int, default=3)
    parser.add_argument('--render', action='store_true', help='打开训练渲染（后台运行不建议）')
    parser.add_argument('--leg-weight', type=float, default=.15)
    parser.add_argument('--arm-weight', type=float, default=1.)
    parser.add_argument('--motion-gain', type=float, default=1.35)
    parser.add_argument('--joint-assist', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--airborne-reward', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--source-url', action='append', default=None, help='可重复；跳过自动搜索')
    parser.add_argument('--skill-name', default=None)
    parser.add_argument('--poll-seconds', type=float, default=15.)
    args = parser.parse_args(argv)

    jobs = TrainingJobs()
    jobs.tavily_key = os.environ.get('TAVILY_API_KEY')
    jobs.model_settings = {
        key: os.environ.get(key, '')
        for key in ('DEEPSEEK_API_KEY', 'G1_MODEL_BASE_URL', 'G1_MODEL_NAME')
    }
    state = jobs.learn_action(
        args.action,
        source_urls=args.source_url,
        skill_name=args.skill_name or args.action,
        iterations=args.iterations,
        device=args.device,
        render=args.render,
        leg_weight=args.leg_weight,
        arm_weight=args.arm_weight,
        motion_gain=args.motion_gain,
        max_seconds=args.max_seconds,
        fps=args.fps,
        num_envs=args.num_envs,
        max_candidates=args.max_candidates,
        joint_assist=args.joint_assist,
        airborne_reward=args.airborne_reward,
    )
    job_id = state['job_id']
    print(json.dumps({'event': 'started', **state}, ensure_ascii=False), flush=True)

    while True:
        status = jobs.status(job_id)
        printable = {
            key: status.get(key)
            for key in ('job_id', 'state', 'message', 'device', 'gpu', 'iteration',
                        'total_iterations', 'elapsed_text', 'training_text',
                        'eta_text', 'extraction', 'metrics', 'evaluation',
                        'checkpoint', 'output_dir', 'error')
            if status.get(key) is not None
        }
        print(json.dumps(printable, ensure_ascii=False), flush=True)
        if status.get('state') in TERMINAL:
            return 0 if status.get('state') == 'completed' else 2
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == '__main__':
    raise SystemExit(main())
