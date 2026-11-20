"""G1-DAgent: read-only dual-agent training status and tuning advisor."""
from __future__ import annotations

import argparse
import json
import sys

from config import DATA_ROOT, REPORTS, llm_available
from orchestrator import Orchestrator
from status_reader import JobStatusReader


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')


def _json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='G1 双 Agent：学习状态读取、参数调试、总结与方向')
    parser.add_argument('--runs', default=None, help='训练任务目录，默认 G1_DATA_ROOT/runs')
    parser.add_argument('--no-llm', action='store_true', help='只使用本地确定性分析')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('list', help='列出最近训练任务')
    p.add_argument('--limit', type=int, default=20)
    p.add_argument('--backend', choices=('auto', 'chat', 'docker'), default='auto')
    p.add_argument('--json', action='store_true')

    p = sub.add_parser('status', help='读取并总结一个训练任务')
    p.add_argument('--job', default=None)
    p.add_argument('--json', action='store_true')

    p = sub.add_parser('debug', help='只生成参数调试建议')
    p.add_argument('--job', default=None)
    p.add_argument('--json', action='store_true')
    p.add_argument('--no-llm', action='store_true', dest='no_llm_sub')

    p = sub.add_parser('report', help='运行两个 Agent 并写入报告')
    p.add_argument('--job', default=None)
    p.add_argument('--json', action='store_true')
    p.add_argument('--no-llm', action='store_true', dest='no_llm_sub')

    p = sub.add_parser('watch', help='定时生成最新任务报告')
    p.add_argument('--job', default=None)
    p.add_argument('--interval', type=float, default=60.0)
    p.add_argument('--once', action='store_true')
    p.add_argument('--no-llm', action='store_true', dest='no_llm_sub')

    args = parser.parse_args(argv)
    reader = JobStatusReader(args.runs)
    if args.command == 'list':
        rows = reader.list_jobs(limit=args.limit, backend=args.backend)
        if args.json:
            _json(rows)
        else:
            for row in rows:
                print('%s  %-6s %-10s iter=%s/%s  %s' % (
                    row['job_id'], row['backend'], row['state'],
                    row['iteration'], row['total_iterations'],
                    row.get('message', '')))
        return 0
    if args.command == 'status':
        record = reader.get(args.job)
        if args.json:
            _json(record)
        else:
            print(JobStatusReader.summarize(record))
        return 0
    no_llm = bool(args.no_llm or getattr(args, 'no_llm_sub', False))
    orchestrator = Orchestrator(args.runs, use_llm=not no_llm)
    if args.command == 'debug':
        record = reader.get(args.job)
        result = orchestrator.debug_agent.debug(record)
        _json(result)
        return 0
    if args.command == 'report':
        report = orchestrator.run(args.job)
        if args.json:
            _json(report)
        else:
            print(orchestrator.render_markdown(report))
        return 0
    if args.command == 'watch':
        import time
        while True:
            try:
                report = orchestrator.run(args.job)
                print(orchestrator.render_markdown(report))
            except Exception as exc:
                print('报告失败：%s' % exc)
            if args.once:
                break
            time.sleep(max(2.0, args.interval))
        return 0
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
