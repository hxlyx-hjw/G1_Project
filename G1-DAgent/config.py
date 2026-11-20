"""Runtime paths and API configuration for G1-DAgent."""
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
except Exception:
    pass


PROJECT_ROOT = Path(os.environ.get(
    'G1_PROJECT_ROOT',
    r'E:\TEST_RECORD\G1_Project\G1\G1_LLM_Robot\G1_LLM_Robot',
))
DATA_ROOT = Path(os.environ.get('G1_DATA_ROOT', r'E:\TEST_RECORD\G1_Project\G1'))
RUNS = DATA_ROOT / 'runs'
OUTPUTS = DATA_ROOT / 'outputs'
SKILLS = DATA_ROOT / 'skills'
LOGS = DATA_ROOT / 'logs'
REPORTS = Path(os.environ.get('G1_DAGENT_REPORTS', str(ROOT / 'reports')))

DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '').strip()
DEEPSEEK_BASE_URL = os.environ.get(
    'G1_MODEL_BASE_URL', 'https://api.deepseek.com').rstrip('/')
DEEPSEEK_MODEL = os.environ.get('G1_MODEL_NAME', 'deepseek-flash')
AGENT_TEMPERATURE = float(os.environ.get('G1_DAGENT_TEMPERATURE', '0.2'))


def llm_available():
    return bool(DEEPSEEK_API_KEY)


def ensure_reports():
    REPORTS.mkdir(parents=True, exist_ok=True)
    return REPORTS
