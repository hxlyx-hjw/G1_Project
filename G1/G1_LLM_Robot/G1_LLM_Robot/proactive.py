"""训练计时、主动报告与报告文件落盘。

DeepSeek 的对话接口本身是请求-响应模式。本模块在聊天进程内维护一个
轻量后台线程：训练任务启动后按固定间隔读取真实训练状态，生成一份报告，
通过聊天回调主动发送，并保存为 Markdown 文件。
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path


TERMINAL_STATES = frozenset(('completed', 'failed', 'cancelled'))


def format_duration(seconds):
    """把秒数格式化为中文时长。"""
    try:
        seconds = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return '未记录'
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return '%d小时%02d分%02d秒' % (hours, minutes, secs)
    if minutes:
        return '%d分%02d秒' % (minutes, secs)
    return '%d秒' % secs


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class ProactiveReporter:
    """按固定间隔发送训练报告。

    ``status_provider(job_id)`` 返回训练状态字典；``output(text)`` 负责
    发送消息（CLI 中打印，其他界面可换成消息气泡回调）。
    """

    def __init__(self, status_provider, output, interval_seconds=600,
                 model_summary=None, report_root=None, enabled=True):
        self.status_provider = status_provider
        self.output = output
        self.model_summary = model_summary
        self.report_root = Path(report_root) if report_root else None
        self._interval = max(1.0, float(interval_seconds))
        self._enabled = bool(enabled)
        self._job_id = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._next_due = None
        self._last_report_at = None

    @property
    def interval_seconds(self):
        with self._lock:
            return self._interval

    @property
    def job_id(self):
        with self._lock:
            return self._job_id

    def configure(self, enabled=None, interval_seconds=None, job_id=None):
        """更新开关、间隔或当前监听的 job。"""
        with self._lock:
            if enabled is not None:
                self._enabled = bool(enabled)
            if interval_seconds is not None:
                self._interval = max(1.0, float(interval_seconds))
                if self._job_id is not None and self._enabled:
                    self._next_due = time.monotonic() + self._interval
            if job_id:
                self._job_id = str(job_id)
                self._next_due = time.monotonic() + self._interval
                self._enabled = True
            active_job = self._job_id
        if active_job and self._enabled:
            self._ensure_thread()
        return self.summary()

    def summary(self):
        with self._lock:
            return {
                'enabled': self._enabled,
                'job_id': self._job_id,
                'interval_seconds': self._interval,
                'interval_minutes': round(self._interval / 60.0, 2),
                'last_report_seconds_ago': (
                    round(time.monotonic() - self._last_report_at, 1)
                    if self._last_report_at is not None else None),
            }

    def watch(self, job_id):
        """开始监听一个训练任务；已有监听会被替换。"""
        with self._lock:
            if not self._enabled:
                return self.summary()
            self._job_id = str(job_id)
            self._next_due = time.monotonic() + self._interval
        self._ensure_thread()
        return self.summary()

    def send_now(self, job_id=None):
        """立即生成并发送一份报告。"""
        with self._lock:
            job = str(job_id or self._job_id or '')
        if not job:
            raise ValueError('没有正在监听的训练任务；请先启动训练或传入 job_id')
        status = self.status_provider(job)
        text = self.build_report(status)
        text = self._with_model_summary(status, text)
        self.output(text)
        path = self._save_report(status, text)
        with self._lock:
            self._last_report_at = time.monotonic()
        return {'job_id': job, 'sent': True, 'path': path, 'report': text,
                'state': status.get('state')}

    def build_report(self, status):
        """把训练状态整理成可读报告。"""
        status = status or {}
        state = str(status.get('state') or 'unknown')
        job = status.get('job_id') or '—'
        lines = [
            '# G1 训练主动报告',
            '',
            '- 生成时间：%s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '- 训练任务：`%s`' % job,
            '- 当前阶段：**%s**' % state,
        ]
        message = status.get('message')
        if message:
            lines.append('- 阶段说明：%s' % message)
        started = status.get('started_at')
        if started:
            lines.append('- 开始时间：%s' % started)
        duration = status.get('duration_seconds')
        if duration is None:
            duration = status.get('elapsed_seconds')
        if _number(duration):
            label = '总耗时' if state in TERMINAL_STATES else '已用时间'
            lines.append('- %s：%s（%.1f 秒）' % (
                label, status.get('duration_text') or status.get('elapsed_text')
                or format_duration(duration), float(duration)))
        training_seconds = status.get('training_seconds')
        if _number(training_seconds):
            lines.append('- 纯训练用时：%s' % (
                status.get('training_text') or format_duration(training_seconds)))
        iteration = status.get('iteration')
        total = status.get('total_iterations')
        if _number(iteration):
            if _number(total) and total > 0:
                percent = min(100.0, 100.0 * float(iteration) / float(total))
                lines.append('- 训练轮次：%d/%d（%.1f%%）' % (
                    int(iteration), int(total), percent))
            else:
                lines.append('- 训练轮次：%d' % int(iteration))
        eta = status.get('eta_seconds')
        if _number(eta) and eta > 0 and state not in TERMINAL_STATES:
            lines.append('- 预计剩余：约 %s' % format_duration(eta))
        metrics = status.get('metrics') or {}
        if isinstance(metrics, dict):
            for key, label in (('mean_reward', '平均奖励'), ('loss', '损失'),
                               ('steps', '累计采样步数')):
                if key in metrics:
                    lines.append('- %s：%s' % (label, metrics[key]))
        extraction = status.get('extraction')
        if isinstance(extraction, dict):
            stage = 'SMPL-X 拟合' if extraction.get('stage') == 'smplx' else '骨架提取'
            lines.append('- %s：%s/%s（%s%%）' % (stage,
                extraction.get('current', '?'), extraction.get('total', '?'),
                extraction.get('percent', '?')))
        viewer = status.get('viewer') or {}
        if (isinstance(viewer, dict) and viewer
                and viewer.get('state') not in (None, 'not_requested')):
            lines.append('- 渲染窗口：%s；权重轮次 %s；渲染耗时 %sms' % (
                viewer.get('state', '—'), viewer.get('checkpoint_iteration', '—'),
                viewer.get('render_ms', '—')))
        evaluation = status.get('evaluation')
        if isinstance(evaluation, dict):
            lines.append('- 物理评估：摔倒=%s，参考完成=%s' % (
                evaluation.get('fallen', '未报告'),
                evaluation.get('reference_completed', evaluation.get('completed', '未报告'))))
            for key, label in (('simulation_seconds', '仿真秒数'),
                               ('max_root_xy_error_m', '最大平面误差(m)'),
                               ('joint_mse', '关节MSE'),
                               ('termination_reason', '终止原因')):
                if key in evaluation:
                    lines.append('  - %s：%s' % (label, evaluation[key]))
        checkpoint = status.get('checkpoint')
        if checkpoint:
            lines.append('- 策略权重：`%s`' % checkpoint)
        output_dir = status.get('output_dir')
        if output_dir:
            lines.append('- 任务目录：`%s`' % output_dir)
        if state == 'completed':
            lines.extend(['', '训练流程已结束。请同时检查物理评估与技能登记结果，'
                              '任务结束不等于动作语义已学会。'])
        elif state == 'failed':
            lines.extend(['', '训练失败：%s' % (status.get('error') or message or '未知错误')])
        elif state == 'cancelled':
            lines.extend(['', '训练已停止，已保存的权重和日志仍然保留。'])
        return '\n'.join(lines)

    def _with_model_summary(self, status, text):
        if self.model_summary is None:
            return text
        try:
            summary = self.model_summary(status, text)
        except Exception:
            summary = ''
        if summary:
            return text + '\n\n**DeepSeek 摘要**：%s' % str(summary).strip()
        return text

    def _save_report(self, status, text):
        root = self.report_root
        if root is None:
            output_dir = status.get('output_dir')
            root = Path(output_dir) if output_dir else None
        if root is None:
            return None
        try:
            folder = Path(root) / 'proactive_reports'
            folder.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = folder / ('report_%s.md' % stamp)
            path.write_text(text, encoding='utf-8')
            (folder / 'latest.md').write_text(text, encoding='utf-8')
            return str(path)
        except OSError:
            return None

    def _ensure_thread(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._loop, name='g1-proactive-reporter', daemon=True)
            self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                enabled = self._enabled
                job = self._job_id
                due = self._next_due
                interval = self._interval
            if not enabled or not job or due is None:
                self._stop.wait(0.2)
                continue
            wait = due - time.monotonic()
            if wait > 0:
                self._stop.wait(min(wait, 1.0))
                continue
            try:
                status = self.status_provider(job)
            except Exception as exc:  # 状态读取失败也要告知，而不是静默消失
                status = {'job_id': job, 'state': 'failed',
                          'message': '读取训练状态失败：%s' % exc}
            text = self._with_model_summary(status, self.build_report(status))
            try:
                self.output(text)
                self._save_report(status, text)
            except Exception:
                pass
            terminal = str(status.get('state')) in TERMINAL_STATES
            with self._lock:
                self._last_report_at = time.monotonic()
                if terminal:
                    self._job_id = None
                    self._next_due = None
                else:
                    self._next_due = time.monotonic() + interval
            if terminal:
                break
        with self._lock:
            if self._thread is threading.current_thread():
                self._thread = None

    def stop(self):
        """停止后台线程（关闭聊天时调用）。"""
        with self._lock:
            self._enabled = False
            self._job_id = None
            self._next_due = None
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        with self._lock:
            self._thread = None
