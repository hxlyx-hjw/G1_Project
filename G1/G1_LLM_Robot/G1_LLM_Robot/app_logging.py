"""主进程及工具错误日志，日志目录不可写时回退LocalAppData。"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os

try:
    from data_paths import LOGS
except Exception:
    LOGS = Path(__file__).resolve().parent / 'logs'


def configure_logging():
    logger=logging.getLogger('g1robot')
    if logger.handlers: return logger
    locations=[LOGS,Path(os.environ.get('LOCALAPPDATA',Path.home()))/'G1Robot/logs']
    for folder in locations:
        try:
            folder.mkdir(parents=True,exist_ok=True)
            handler=RotatingFileHandler(folder/'app.log',maxBytes=5_000_000,backupCount=3,encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
            logger.addHandler(handler); logger.setLevel(logging.INFO); logger.propagate=False
            return logger
        except OSError: continue
    logger.addHandler(logging.StreamHandler()); logger.setLevel(logging.INFO)
    return logger
