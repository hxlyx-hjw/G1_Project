"""Central data-root paths.

On Windows the training data lives on E: so C: only keeps code and links.
In Docker/Linux the same relative layout under the project root is used.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _default_root():
    configured = os.environ.get('G1_DATA_ROOT')
    if configured:
        return Path(configured).expanduser()
    if sys.platform.startswith('win') and Path(r'E:\TEST_RECORD\G1_Project\G1').exists():
        return Path(r'E:\TEST_RECORD\G1_Project\G1')
    return PROJECT_ROOT


DATA_ROOT = _default_root()
RUNS = DATA_ROOT / 'runs'
OUTPUTS = DATA_ROOT / 'outputs'
MOTIONS = DATA_ROOT / 'motions'
SKILLS = DATA_ROOT / 'skills'
LOGS = DATA_ROOT / 'logs'
VEDIO = DATA_ROOT / 'vedio'
VEDIO_SELF = DATA_ROOT / 'vedio_self'
MODELS = DATA_ROOT / 'models'
ASSETS = DATA_ROOT / 'assets'
TOOLS = DATA_ROOT / 'tools'
DATASETS = Path(os.environ.get(
    'G1_DATASET_ROOT',
    r'D:\G1_Datasets' if sys.platform.startswith('win')
    else str(DATA_ROOT / 'datasets')))
AMASS = Path(os.environ.get(
    'G1_AMASS_ROOT',
    r'D:\AMASS\datasets\AMASS_Retargeted_for_G1' if sys.platform.startswith('win')
    else str(DATA_ROOT / 'datasets' / 'AMASS_Retargeted_for_G1'),
))
AMASS_MOTIONS = Path(os.environ.get('G1_AMASS_MOTIONS', str(MOTIONS / 'amass')))

LEGACY_DATA_ROOT = Path(r'E:\TEST_RECORD\G1')


def rebase_paths(value):
    """Rebase paths stored before the G1_Project move."""
    if isinstance(value, dict):
        return {key: rebase_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rebase_paths(item) for item in value]
    if isinstance(value, str):
        legacy = str(LEGACY_DATA_ROOT)
        if value.lower().startswith(legacy.lower()):
            return str(DATA_ROOT) + value[len(legacy):]
    return value


def ensure_data_dirs():
    for path in (RUNS, OUTPUTS, MOTIONS, SKILLS, LOGS, VEDIO, VEDIO_SELF,
                 DATASETS, AMASS_MOTIONS):
        path.mkdir(parents=True, exist_ok=True)
    return DATA_ROOT


def describe():
    return {
        'data_root': str(DATA_ROOT),
        'runs': str(RUNS),
        'outputs': str(OUTPUTS),
        'motions': str(MOTIONS),
        'skills': str(SKILLS),
        'logs': str(LOGS),
        'vedio': str(VEDIO),
        'vedio_self': str(VEDIO_SELF),
        'models': str(MODELS),
        'assets': str(ASSETS),
        'tools': str(TOOLS),
        'datasets': str(DATASETS),
        'amass': str(AMASS),
        'amass_motions': str(AMASS_MOTIONS),
    }
