"""Docker-backed training jobs for DeepSeek tools.

The chat process stays on the host. Training runs in the existing
``g1-llm-robot:cu128`` image with GPU and model mounts, while JSON status is
written to a host directory under ``runs/docker_jobs``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path

from proactive import format_duration
from data_paths import (RUNS, OUTPUTS, MOTIONS, SKILLS, LOGS, VEDIO,
                        MODELS, ASSETS, ensure_data_dirs)
from .jobs import atomic_json, read_json
from .jobs import rl_pd_options


IMAGE = os.environ.get('G1_DOCKER_IMAGE', 'g1-llm-robot:cu128')
DOCKER = os.environ.get('G1_DOCKER', 'docker')
SMPL_DIR = Path(os.environ.get(
    'G1_SMPLX_MODEL_DIR',
    str(MODELS / 'SMPL' / 'models_smplx_v1_1' / 'models' / 'smplx')))
WHAM_DIR = Path(os.environ.get('G1_WHAM_DIR', str(MODELS / 'WHAM')))
VIDEO_SUFFIXES = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
TERMINAL = {'completed', 'failed', 'cancelled'}


def _under(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


class DockerJobs:
    def __init__(self):
        self.jobs = {}
        self.latest = None

    # ------------------------------------------------------------------
    # process helpers
    # ------------------------------------------------------------------
    def _docker(self, args, timeout=120):
        return subprocess.run(
            [DOCKER, *args], capture_output=True, text=True, timeout=timeout)

    def _require_docker(self):
        if shutil.which(DOCKER) is None:
            raise RuntimeError(
                '当前聊天进程无法调用 Docker CLI。请在 Windows 宿主机运行 '
                'Start.py --mode chat；若聊天本身在容器内，需要额外挂载 Docker socket。')

    def _inspect(self, name):
        try:
            result = self._docker(
                ['inspect', '-f', '{{.State.Status}} {{.State.ExitCode}}', name],
                timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split()
                return parts[0], (int(parts[1]) if len(parts) > 1 else None)
        except Exception:
            pass
        return 'missing', None

    def _task(self, job_id=None):
        if job_id is None and self.latest is None:
            self.discover()
        job_id = job_id or self.latest
        if job_id and job_id not in self.jobs:
            self._load_job(job_id)
        if job_id not in self.jobs:
            raise ValueError('没有此 Docker 训练任务，请使用 robot_docker_train 返回的 job_id')
        return job_id, self.jobs[job_id]

    def _load_job(self, job_id):
        folder = RUNS / 'docker_jobs' / str(job_id)
        config = read_json(folder / 'config.json', {}) or {}
        if not config.get('job_id'):
            return False
        self.jobs[job_id] = {
            'folder': folder,
            'container': config.get('container') or ('g1-docker-' + job_id),
            'container_id': config.get('container_id', ''),
            'started_at': config.get('started_at', ''),
            'started_monotonic': 0.0,
        }
        self.latest = job_id
        return True

    def discover(self):
        """Adopt the newest host job after a chat process restart."""
        base = RUNS / 'docker_jobs'
        if not base.is_dir():
            return None
        folders = sorted((p for p in base.iterdir() if p.is_dir()),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        for folder in folders:
            config = read_json(folder / 'config.json', {}) or {}
            if not config.get('job_id'):
                continue
            job_id = folder.name
            if self._load_job(job_id):
                return job_id
        return None

    # ------------------------------------------------------------------
    # start training
    # ------------------------------------------------------------------
    def start(self, directory, reference=None, video=None, iterations=500,
              device='cuda', num_envs=4, leg_weight=.15, arm_weight=1.,
              motion_gain=1.35, max_seconds=60, fps=15, skill_name=None,
              coordination=True, joint_assist=True, joint_assist_gain=.18,
              airborne_reward=True, root_mode='contact_grounded',
              curriculum_target=None, max_episode_seconds=None,
              rl_pd_influence=1.0, hand_rl_pd_range=None,
              foot_rl_pd_range=None):
        self._require_docker()
        ensure_data_dirs()
        directory = Path(directory).expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError('directory 必须是存在的目录：%s' % directory)
        if reference and video:
            raise ValueError('reference 与 video 只能指定一个')
        if device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('device 必须为 auto/cpu/cuda')
        if not 1 <= iterations <= 100000 or not 1 <= num_envs <= 16:
            raise ValueError('iterations/num_envs 超出范围')
        if not 0 <= leg_weight <= 2 or not .25 <= arm_weight <= 1.5:
            raise ValueError('手腿权重超出范围')
        if not 1 <= motion_gain <= 2 or not 0 <= joint_assist_gain <= 2:
            raise ValueError('motion_gain 或 joint_assist_gain 超出范围')
        if not 0 < max_seconds <= 180 or not 1 <= fps <= 60:
            raise ValueError('max_seconds/fps 超出范围')
        if max_episode_seconds is not None and not 1 <= max_episode_seconds <= 600:
            raise ValueError('max_episode_seconds 应在 1~600')
        if curriculum_target is not None and not leg_weight <= curriculum_target <= 2:
            raise ValueError('curriculum_target 应在 leg_weight~2')
        rl_pd_influence, hand_rl_pd_range, foot_rl_pd_range = rl_pd_options(
            rl_pd_influence, hand_rl_pd_range, foot_rl_pd_range)

        def child(value):
            path = Path(value)
            path = (path if path.is_absolute() else directory / path).resolve()
            if path != directory and not _under(path, directory):
                raise ValueError('指定文件必须位于 directory 内：%s' % value)
            return path

        reference_path = child(reference) if reference else None
        video_path = child(video) if video else None
        if reference_path is None and video_path is None:
            candidate = directory / 'reference.npz'
            if candidate.is_file():
                reference_path = candidate
            else:
                videos = sorted(p for p in directory.iterdir()
                                if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
                if videos:
                    video_path = videos[0]
        if reference_path is not None and not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        if video_path is not None and not video_path.is_file():
            raise FileNotFoundError(video_path)
        if reference_path is None and video_path is None:
            raise ValueError('目录中未找到 reference.npz 或可解码视频')

        for pending_id, task in self.jobs.items():
            state, _ = self._inspect(task['container'])
            if state in ('running', 'created', 'restarting'):
                raise RuntimeError('已有 Docker 训练正在运行：%s' % pending_id)

        job_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
        folder = RUNS / 'docker_jobs' / job_id
        folder.mkdir(parents=True, exist_ok=False)
        mode = 'reference' if reference_path is not None else 'local_video'
        if reference_path is not None:
            shutil.copy2(reference_path, folder / 'reference.npz')

        config = {
            'job_id': job_id, 'folder': str(folder), 'mode': mode,
            'skill_name': skill_name or directory.name,
            'reference': str(folder / 'reference.npz') if reference_path is not None else None,
            'video': '/data/' + video_path.name if video_path is not None else None,
            'iterations': int(iterations), 'device': device,
            'render': False, 'num_envs': int(num_envs),
            'leg_weight': float(leg_weight), 'arm_weight': float(arm_weight),
            'motion_gain': float(motion_gain), 'max_seconds': float(max_seconds),
            'fps': float(fps), 'coordination': bool(coordination),
            'jump_mode': 'off', 'jump_times': None,
            'auto_adjust': True, 'model_review': False,
            'joint_assist': bool(joint_assist),
            'joint_assist_gain': float(joint_assist_gain),
            'airborne_reward': bool(airborne_reward),
            'root_mode': root_mode, 'curriculum_target': curriculum_target,
            'max_episode_seconds': max_episode_seconds,
            'rl_pd_influence': rl_pd_influence,
            'hand_rl_pd_range': hand_rl_pd_range,
            'foot_rl_pd_range': foot_rl_pd_range,
            'input_directory': str(directory),
            'container': 'g1-docker-' + job_id,
        }
        started_at = datetime.now().astimezone().isoformat(timespec='seconds')
        config['started_at'] = started_at
        atomic_json(folder / 'config.json', config)
        atomic_json(folder / 'status.json', {
            'job_id': job_id, 'state': 'queued', 'message': 'Docker 训练容器启动中',
            'started_at': started_at, 'elapsed_seconds': 0, 'elapsed_text': '0秒',
        })

        container = 'g1-docker-' + job_id
        command = [
            DOCKER, 'run', '-d', '--name', container,
            '--gpus', 'all', '--shm-size=4g',
            '-v', '%s:/workspace/G1_LLM_Robot/runs' % RUNS,
            '-v', '%s:/data:ro' % directory,
            '-e', 'G1_SMPLX_MODEL_DIR=/models/SMPL',
        ]
        for host_dir, container_dir in (
                (OUTPUTS, '/workspace/G1_LLM_Robot/outputs'),
                (MOTIONS, '/workspace/G1_LLM_Robot/motions'),
                (SKILLS, '/workspace/G1_LLM_Robot/skills'),
                (LOGS, '/workspace/G1_LLM_Robot/logs'),
                (VEDIO, '/workspace/G1_LLM_Robot/vedio')):
            host_dir.mkdir(parents=True, exist_ok=True)
            command += ['-v', '%s:%s' % (host_dir, container_dir)]
        if SMPL_DIR.is_dir():
            command += ['-v', '%s:/models/SMPL:ro' % SMPL_DIR]
        if WHAM_DIR.is_dir():
            command += ['-v', '%s:/models/WHAM:ro' % WHAM_DIR]
        for key in ('DEEPSEEK_API_KEY', 'TAVILY_API_KEY',
                    'G1_MODEL_BASE_URL', 'G1_MODEL_NAME'):
            if os.environ.get(key):
                command += ['-e', key]
        command += [IMAGE, 'python', '-u', '-m', 'fullbody.job_worker',
                    '/workspace/G1_LLM_Robot/runs/docker_jobs/%s' % job_id]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        except Exception as exc:
            atomic_json(folder / 'status.json', {
                'job_id': job_id, 'state': 'failed', 'message': 'Docker 启动失败：%s' % exc,
                'started_at': started_at,
            })
            raise
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or '').strip()
            atomic_json(folder / 'status.json', {
                'job_id': job_id, 'state': 'failed',
                'message': 'Docker 启动失败：%s' % detail[-1500:],
                'started_at': started_at,
            })
            raise RuntimeError('Docker 启动失败：%s' % detail[-1500:])
        container_id = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ''
        self.jobs[job_id] = {
            'folder': folder, 'container': container,
            'container_id': container_id, 'started_at': started_at,
            'started_monotonic': time.monotonic(),
        }
        self.latest = job_id
        return self.status(job_id)

    # ------------------------------------------------------------------
    # status / summary
    # ------------------------------------------------------------------
    def status(self, job_id=None):
        job_id, task = self._task(job_id)
        status = read_json(task['folder'] / 'status.json', {})
        docker_state, exit_code = self._inspect(task['container'])
        if docker_state == 'exited' and status.get('state') not in TERMINAL:
            status.update(state='failed', message='训练容器意外退出',
                          docker_exit_code=exit_code)
        elapsed = status.get('duration_seconds')
        if not isinstance(elapsed, (int, float)):
            elapsed = status.get('elapsed_seconds')
        if not isinstance(elapsed, (int, float)):
            elapsed = time.monotonic() - task.get('started_monotonic', time.monotonic())
        status.update(
            job_id=job_id, output_dir=str(task['folder']),
            docker_image=IMAGE, container=task['container'],
            container_id=task.get('container_id'), docker_state=docker_state,
            worker_running=docker_state in ('running', 'created', 'restarting'),
            elapsed_seconds=round(float(elapsed), 3),
            elapsed_text=status.get('elapsed_text') or format_duration(elapsed),
        )
        return status

    def summary(self, job_id=None, write_report=True):
        status = self.status(job_id)
        job_id, task = self._task(job_id)
        folder = task['folder']
        best = read_json(folder / 'best.json', {}) or {}
        evaluation = status.get('evaluation') or read_json(
            folder / 'evaluation' / 'evaluation.json', {}) or best
        metrics = status.get('metrics') or {}
        lines = [
            '# Docker 训练总结',
            '',
            '- 任务：`%s`' % job_id,
            '- 状态：**%s**' % status.get('state', 'unknown'),
            '- 容器：`%s`（%s）' % (task['container'], status.get('docker_state')),
            '- 输入目录：`%s`' % read_json(folder / 'config.json', {}).get('input_directory', '—'),
            '- 总耗时：%s' % status.get('duration_text', status.get('elapsed_text', '—')),
        ]
        if status.get('training_text'):
            lines.append('- 纯训练用时：%s' % status['training_text'])
        if status.get('iteration') is not None:
            lines.append('- 轮次：%s/%s' % (status.get('iteration'), status.get('total_iterations', '?')))
        if isinstance(metrics, dict):
            for key, label in (('mean_reward', '平均奖励'), ('loss', '损失'),
                               ('steps', '累计采样步数')):
                if key in metrics:
                    lines.append('- %s：%s' % (label, metrics[key]))
        history = []
        try:
            history_path = folder / 'training' / 'metrics.jsonl'
            if history_path.is_file():
                for raw in history_path.read_text(encoding='utf-8').splitlines()[-300:]:
                    try:
                        record = json.loads(raw)
                    except ValueError:
                        continue
                    if isinstance(record.get('mean_reward'), (int, float)):
                        history.append(record)
        except OSError:
            history = []
        if history:
            best = max(history, key=lambda row: row.get('mean_reward', -1e9))
            lines.append('- 奖励趋势：首轮 %.4f，最高 %.4f（轮次 %s），最近 %.4f' % (
                history[0].get('mean_reward', 0), best.get('mean_reward', 0),
                best.get('iteration', '?'), history[-1].get('mean_reward', 0)))
        if isinstance(evaluation, dict) and evaluation:
            lines.extend([
                '- 物理评估：摔倒=%s，参考完成=%s' % (
                    evaluation.get('fallen', '未报告'),
                    evaluation.get('reference_completed', evaluation.get('completed', '未报告'))),
            ])
            for key, label in (('simulation_seconds', '仿真秒数'),
                               ('joint_mse', '关节MSE'),
                               ('max_root_xy_error_m', '最大平面误差(m)'),
                               ('max_root_speed_mps', '最大根部速度(m/s)'),
                               ('termination_reason', '终止原因')):
                if key in evaluation:
                    lines.append('- %s：%s' % (label, evaluation[key]))
        checkpoint = folder / 'training' / 'policy.pt'
        if checkpoint.is_file():
            lines.append('- 最佳权重：`%s`' % checkpoint)
        if status.get('state') == 'failed':
            try:
                tail = (folder / 'worker.log').read_text(
                    encoding='utf-8', errors='replace').splitlines()[-20:]
                if tail:
                    lines.extend(['', '## 失败日志', '```text', *tail, '```'])
            except OSError:
                pass
        lines.extend(['', '结论：训练流程完成不等于动作语义已学会；请检查摔倒、漂移、'
                          '关节误差和技能登记结果。'])
        text = '\n'.join(lines)
        path = None
        if write_report:
            path = folder / 'summary.md'
            path.write_text(text, encoding='utf-8')
        return {
            'job_id': job_id, 'state': status.get('state'),
            'summary': text, 'summary_path': str(path) if path else None,
            'output_dir': str(folder), 'evaluation': evaluation,
            'checkpoint': str(checkpoint) if checkpoint.is_file() else None,
        }

    # ------------------------------------------------------------------
    # video export
    # ------------------------------------------------------------------
    def export_video(self, job_id=None, format='mp4', fps=15, max_seconds=20,
                     output=None):
        self._require_docker()
        job_id, task = self._task(job_id)
        folder = task['folder']
        checkpoint = folder / 'training' / 'policy.pt'
        if not checkpoint.is_file():
            checkpoint = folder / 'training' / 'latest.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError('任务尚无可导出的 policy.pt/latest.pt')
        reference = folder / 'reference.npz'
        if not reference.is_file():
            raise FileNotFoundError(reference)
        if format not in ('gif', 'mp4', 'video'):
            raise ValueError('format 必须为 gif 或 mp4')
        format = 'mp4' if format == 'video' else format
        if not 1 <= fps <= 30 or not 0 < max_seconds <= 60:
            raise ValueError('fps:1~30，max_seconds:(0,60]')
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if output:
            output = Path(output)
            output = (output if output.is_absolute() else folder / output).resolve()
        else:
            output = folder / ('video_%s.%s' % (stamp, format))
        if output.suffix.lower() != '.' + format:
            output = output.with_suffix('.' + format)
        if not _under(output, RUNS):
            raise ValueError('导出文件必须位于项目 runs 目录内：%s' % output)
        if output.exists():
            raise FileExistsError(output)
        gif_path = output if format == 'gif' else folder / ('video_%s.gif' % stamp)
        if gif_path.exists():
            raise FileExistsError(gif_path)

        def container_path(path):
            relative = Path(path).resolve().relative_to(RUNS.resolve())
            return '/workspace/G1_LLM_Robot/runs/' + relative.as_posix()

        status_path = folder / ('export_%s.json' % stamp)
        cmd = [
            DOCKER, 'run', '--rm', '--gpus', 'all',
            '-v', '%s:/workspace/G1_LLM_Robot/runs' % RUNS,
            '-e', 'G1_SMPLX_MODEL_DIR=/models/SMPL',
        ]
        if SMPL_DIR.is_dir():
            cmd += ['-v', '%s:/models/SMPL:ro' % SMPL_DIR]
        cmd += [IMAGE, 'python', '-m', 'fullbody.gif_export',
                container_path(checkpoint), container_path(gif_path),
                '--reference', container_path(reference),
                '--fps', str(int(fps)), '--max-seconds', str(float(max_seconds)),
                '--status', container_path(status_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if result.returncode != 0 or not gif_path.is_file():
            detail = (result.stderr or result.stdout or '').strip()
            raise RuntimeError('视频导出失败：%s' % detail[-1500:])
        report = read_json(status_path, {}) or {}
        final = gif_path
        if format == 'mp4':
            ffmpeg = [
                DOCKER, 'run', '--rm',
                '-v', '%s:/workspace/G1_LLM_Robot/runs' % RUNS,
                IMAGE, 'ffmpeg', '-y', '-loglevel', 'error',
                '-i', container_path(gif_path),
                '-movflags', '+faststart', '-pix_fmt', 'yuv420p',
                container_path(output),
            ]
            converted = subprocess.run(
                ffmpeg, capture_output=True, text=True, timeout=600)
            if converted.returncode != 0 or not output.is_file():
                detail = (converted.stderr or converted.stdout or '').strip()
                raise RuntimeError('GIF 转 MP4 失败：%s' % detail[-1500:])
            final = output
        return {
            'job_id': job_id, 'format': format,
            'output': str(final), 'gif': str(gif_path),
            'report': report,
            'message': 'Docker 物理回放视频已导出',
        }

    def stop(self, job_id=None):
        self._require_docker()
        job_id, task = self._task(job_id)
        (task['folder'] / 'STOP').touch()
        try:
            self._docker(['stop', '--time', '20', task['container']], timeout=40)
        except Exception:
            pass
        return {'job_id': job_id, 'stop_requested': True,
                'message': '已请求停止 Docker 训练并保留已保存权重'}

    def close(self):
        for task in self.jobs.values():
            state, _ = self._inspect(task['container'])
            if state in ('running', 'created', 'restarting'):
                (task['folder'] / 'STOP').touch()
