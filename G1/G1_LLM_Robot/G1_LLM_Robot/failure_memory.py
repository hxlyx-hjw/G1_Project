"""Structured action-failure memory for closed-loop robot tool calls.

The memory is intentionally lightweight: JSONL on disk, no model update.
It stores the failed tool, arguments, physical evidence, diagnosis and a
reusable correction. Retrieval is deterministic and local.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

from data_paths import LOGS


MEMORY_PATH = Path(os.environ.get(
    'G1_FAILURE_MEMORY',
    str(Path(LOGS) / 'failure_memory.jsonl'),
))
_LOCK = threading.Lock()


def _safe(value, depth=0):
    if depth > 4:
        return '<truncated>'
    if isinstance(value, dict):
        return {str(key): _safe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item, depth + 1) for item in value[:40]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > 500:
            return value[:500] + '...'
        return value
    return str(value)[:200]


def _read():
    if not MEMORY_PATH.is_file():
        return []
    records = []
    try:
        for line in MEMORY_PATH.read_text(encoding='utf-8', errors='replace').splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict):
                records.append(item)
    except OSError:
        return []
    return records


def record(tool, args, outcome, reason, evidence, correction,
           before=None, after=None, raw_result=None):
    item = {
        'id': uuid.uuid4().hex[:12],
        'timestamp': time.time(),
        'tool': str(tool),
        'args': _safe(args),
        'outcome': str(outcome),
        'reason': str(reason),
        'evidence': _safe(evidence),
        'correction': str(correction),
        'before': _safe(before),
        'after': _safe(after),
        'raw_result': _safe(raw_result),
    }
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with MEMORY_PATH.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    return item


def _arg_score(left, right):
    if not isinstance(left, dict) or not isinstance(right, dict):
        return 0
    score = 0
    for key in set(left) & set(right):
        a, b = left[key], right[key]
        if a == b:
            score += 2
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            score += 1 if abs(float(a) - float(b)) <= max(1e-6, 0.15 * max(1.0, abs(float(a)))) else 0
    return score


def retrieve(tool=None, args=None, limit=5):
    records = _read()
    if tool:
        records = [item for item in records if item.get('tool') == str(tool)]
    records.sort(key=lambda item: (-_arg_score(item.get('args', {}), args or {}),
                                  -float(item.get('timestamp', 0))))
    return records[:max(1, int(limit))]


def summary(tool=None, limit=3):
    records = retrieve(tool=tool, limit=limit)
    if not records:
        return []
    return [{
        'tool': item.get('tool'),
        'reason': item.get('reason'),
        'correction': item.get('correction'),
    } for item in records]


def clear():
    with _LOCK:
        if MEMORY_PATH.exists():
            MEMORY_PATH.unlink()


def count():
    return len(_read())
