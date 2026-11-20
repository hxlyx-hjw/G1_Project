"""Read-only training status reader for G1 local chat and Docker jobs."""
from __future__ import annotations

import json
import time
from pathlib import Path

from config import RUNS, SKILLS


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def _tail_metric(path, max_bytes=262144):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with path.open('rb') as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            lines = handle.read().decode('utf-8', errors='replace').splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _metric_series(path, max_bytes=4_000_000):
    path = Path(path)
    if not path.is_file():
        return []
    try:
        with path.open('rb') as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            lines = handle.read().decode('utf-8', errors='replace').splitlines()
    except OSError:
        return []
    records = []
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


class JobStatusReader:
    def __init__(self, runs=None):
        self.runs = Path(runs or RUNS)

    def _folders(self):
        result = []
        for backend, parent in (('chat', self.runs / 'chat_training'),
                                ('docker', self.runs / 'docker_jobs')):
            if not parent.is_dir():
                continue
            for folder in parent.iterdir():
                if folder.is_dir() and (folder / 'status.json').is_file():
                    result.append((backend, folder))
        return result

    def list_jobs(self, limit=30, backend=None):
        jobs = []
        for kind, folder in self._folders():
            if backend and backend not in ('auto', kind):
                continue
            try:
                record = self._read_job(kind, folder)
            except Exception:
                continue
            jobs.append(record)
        jobs.sort(key=lambda item: item.get('mtime', 0), reverse=True)
        return jobs[:max(1, int(limit))]

    def get(self, job_id=None):
        jobs = self.list_jobs(limit=1000)
        if job_id:
            exact = [item for item in jobs if item['job_id'] == str(job_id)]
            if exact:
                return exact[0]
            prefix = [item for item in jobs if item['job_id'].startswith(str(job_id))]
            if len(prefix) == 1:
                return prefix[0]
            if len(prefix) > 1:
                raise ValueError('job_id 前缀不唯一，请使用完整 ID')
            raise ValueError('没有找到训练任务: %s' % job_id)
        if not jobs:
            raise ValueError('没有找到训练任务；请先启动训练')
        return jobs[0]

    def _read_job(self, kind, folder):
        status = _read_json(folder / 'status.json', {}) or {}
        config = _read_json(folder / 'config.json', {}) or {}
        metrics_path = folder / 'training' / 'metrics.jsonl'
        last_metric = _tail_metric(metrics_path)
        series = _metric_series(metrics_path)
        evaluation = status.get('evaluation') or _read_json(
            folder / 'evaluation' / 'evaluation.json', {}) or {}
        if not isinstance(evaluation, dict):
            evaluation = {}
        last = last_metric or {}
        rewards = [float(item['mean_reward']) for item in series
                   if isinstance(item.get('mean_reward'), (int, float))]
        record = {
            'job_id': str(status.get('job_id') or config.get('job_id') or folder.name),
            'backend': kind,
            'folder': str(folder),
            'mtime': folder.stat().st_mtime,
            'state': status.get('state', 'unknown'),
            'message': status.get('message', ''),
            'started_at': status.get('started_at') or config.get('started_at'),
            'finished_at': status.get('finished_at'),
            'elapsed_seconds': status.get('elapsed_seconds'),
            'elapsed_text': status.get('elapsed_text'),
            'training_seconds': status.get('training_seconds'),
            'training_text': status.get('training_text'),
            'iteration': status.get('iteration', last.get('iteration')),
            'total_iterations': status.get('total_iterations', config.get('iterations')),
            'device': status.get('device', config.get('device')),
            'gpu': status.get('gpu'),
            'mode': config.get('mode'),
            'reference': status.get('reference') or config.get('reference') or config.get('reference_source'),
            'source_video': config.get('video') or config.get('source_folder'),
            'checkpoint': status.get('checkpoint'),
            'last_metric': last,
            'reward_first': rewards[0] if rewards else None,
            'reward_last': rewards[-1] if rewards else None,
            'reward_best': max(rewards) if rewards else None,
            'evaluation': self._evaluation_subset(evaluation),
            'config': self._config_subset(config),
            'skills': self._skills_for_job(str(folder)),
            'age_seconds': max(0.0, time.time() - folder.stat().st_mtime),
        }
        record['outcome'], record['failure_reason'] = self._outcome(record)
        return record

    @staticmethod
    def _evaluation_subset(evaluation):
        keys = (
            'policy_used', 'fallen', 'reference_completed', 'completion_fraction',
            'simulation_seconds', 'reference_duration_seconds', 'termination_reason',
            'joint_mse', 'max_root_xy_error_m', 'max_root_speed_mps',
            'mean_reward', 'airborne_steps', 'min_support_margin_m',
            'airborne_support_compliance', 'motion_gain', 'motion_config',
            'final_state',
        )
        return {key: evaluation.get(key) for key in keys if key in evaluation}

    @staticmethod
    def _config_subset(config):
        keys = (
            'mode', 'iterations', 'num_envs', 'device', 'leg_weight', 'arm_weight',
            'motion_gain', 'max_episode_seconds', 'rl_pd_influence',
            'hand_rl_pd_range', 'foot_rl_pd_range', 'joint_assist',
            'joint_assist_gain', 'airborne_reward', 'coordination', 'reference',
            'video', 'source_folder', 'pair', 'backend', 'pair_count',
        )
        return {key: config.get(key) for key in keys if key in config}

    @staticmethod
    def _skills_for_job(folder):
        path = Path(SKILLS) / 'registry.json'
        data = _read_json(path, {}) or {}
        return [
            {'skill_id': key, 'name': item.get('name'),
             'safe_to_run': item.get('safe_to_run')}
            for key, item in (data.get('skills') or {}).items()
            if str(item.get('job_dir', '')).lower() == str(folder).lower()
        ]

    @staticmethod
    def _outcome(record):
        state = str(record.get('state', '')).lower()
        evaluation = record.get('evaluation') or {}
        if state in ('failed', 'cancelled'):
            return state, record.get('message') or state
        if state != 'completed':
            return 'running', ''
        if evaluation.get('fallen'):
            return 'failed', evaluation.get('termination_reason') or 'fallen'
        if evaluation and not evaluation.get('reference_completed', True):
            return 'partial', 'reference_not_completed'
        if not evaluation:
            return 'partial', 'evaluation_missing'
        return 'success', ''

    @staticmethod
    def summarize(record):
        eval_ = record.get('evaluation') or {}
        return (
            '%s [%s] 状态=%s，轮次=%s/%s，用时=%s；'
            '评估: fallen=%s, reference_completed=%s, completion=%s, '
            'joint_mse=%s, drift=%s, termination=%s'
            % (
                record.get('job_id'), record.get('backend'), record.get('state'),
                record.get('iteration'), record.get('total_iterations'),
                record.get('elapsed_text') or record.get('duration_text'),
                eval_.get('fallen'), eval_.get('reference_completed'),
                eval_.get('completion_fraction'), eval_.get('joint_mse'),
                eval_.get('max_root_xy_error_m'), eval_.get('termination_reason'),
            )
        )
