"""Windows友好持久化：唯一临时文件、短重试，不破坏原文件。"""
from pathlib import Path
import os
import tempfile
import time


def retry(operation,attempts=8):
    for index in range(attempts):
        try: return operation()
        except PermissionError:
            if index==attempts-1: raise
            time.sleep(min(.05*2**index,.5))


def atomic_write(path,writer):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=path.stem+'_',suffix='.tmp',dir=path.parent)
    os.close(fd); temporary=Path(temp)
    try:
        writer(temporary)
        retry(lambda: os.replace(temporary,path))
    except PermissionError as exc:
        raise PermissionError(f'无法写入 {path}：可能被查看器/杀毒程序占用或目录不可写。已重试，原文件保留；请检查logs/app.log。') from exc
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(payload,path):
    import torch
    atomic_write(path,lambda temporary: torch.save(payload,temporary))


def load_checkpoint(path,**kwargs):
    import torch
    return retry(lambda: torch.load(path,**kwargs))
