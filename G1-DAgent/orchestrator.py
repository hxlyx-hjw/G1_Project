"""Orchestrate the status and parameter agents into an auditable report."""
from __future__ import annotations

import json
import time
from pathlib import Path

from agents import build_agents
from config import REPORTS, ensure_reports
from status_reader import JobStatusReader


class Orchestrator:
    def __init__(self, runs=None, use_llm=True):
        self.reader = JobStatusReader(runs)
        self.status_agent, self.debug_agent = build_agents(use_llm)

    def run(self, job_id=None, write=True):
        record = self.reader.get(job_id)
        status = self.status_agent.analyze(record)
        debug = self.debug_agent.debug(record)
        directions = list(status.get('directions') or [])
        directions.extend(debug.get('proposal', {}).get('directions') or [])
        report = {
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'job_id': record.get('job_id'),
            'status': status,
            'debug': debug,
            'directions': directions,
            'record': record,
        }
        paths = {}
        if write:
            ensure_reports()
            stamp = time.strftime('%Y%m%d_%H%M%S')
            json_path = Path(REPORTS) / ('%s_%s.json' % (record['job_id'], stamp))
            md_path = Path(REPORTS) / ('%s_%s.md' % (record['job_id'], stamp))
            json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                                 encoding='utf-8')
            md_path.write_text(self.render_markdown(report), encoding='utf-8')
            paths = {'json': str(json_path), 'markdown': str(md_path)}
        report['report_paths'] = paths
        return report

    @staticmethod
    def render_markdown(report):
        record = report.get('record') or {}
        status = report.get('status') or {}
        proposal = (report.get('debug') or {}).get('proposal') or {}
        lines = [
            '# G1 双 Agent 训练报告',
            '',
            '- 任务：`%s`' % report.get('job_id'),
            '- 生成时间：%s' % report.get('generated_at'),
            '- 状态：`%s`，结果：`%s`' % (record.get('state'), record.get('outcome')),
            '- 轮次：%s/%s' % (record.get('iteration'), record.get('total_iterations')),
            '',
            '## 状态 Agent：事实',
        ]
        lines.extend('- ' + str(item) for item in status.get('facts', []))
        lines.extend(['', '## 状态 Agent：风险'])
        lines.extend('- ' + str(item) for item in status.get('risks', []))
        lines.extend(['', '## 参数调试 Agent'])
        lines.append('- 置信度：`%s`' % proposal.get('confidence'))
        changes = proposal.get('changes') or []
        if changes:
            for item in changes:
                lines.append('- `%s`: `%s` → `%s`（%s）'
                             % (item['parameter'], item['old'], item['new'], item['reason']))
        else:
            lines.append('- 没有满足规则的参数修改；保持当前配置。')
        if proposal.get('launch_command'):
            lines.extend(['', '```powershell', proposal['launch_command'], '```'])
        lines.extend(['', '## 下一步方向'])
        lines.extend('- ' + str(item) for item in report.get('directions', []))
        if status.get('llm_analysis'):
            lines.extend(['', '## 状态 Agent LLM 分析', status['llm_analysis']])
        review = (report.get('debug') or {}).get('llm_review')
        if review:
            lines.extend(['', '## 参数 Agent LLM 审查', review])
        if status.get('llm_error') or (report.get('debug') or {}).get('llm_error'):
            lines.extend(['', '## LLM 错误'])
            if status.get('llm_error'):
                lines.append('- 状态 Agent: %s' % status['llm_error'])
            if (report.get('debug') or {}).get('llm_error'):
                lines.append('- 参数 Agent: %s' % report['debug']['llm_error'])
        return '\n'.join(lines) + '\n'
