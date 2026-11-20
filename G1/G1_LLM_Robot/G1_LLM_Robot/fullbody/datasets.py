"""Registry, manifest builder and training entry point for local datasets.

Downloaded datasets live outside the project on D: by default.  This module
keeps the raw data untouched and exposes a uniform inventory/clip/train API
for AIST++, Motion-X, AMASS, LAFAN1 and 100STYLE.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from data_paths import AMASS_MOTIONS, DATASETS


DATASET_ROOT = DATASETS

SPECS = {
    'AMASS': {
        'kind': 'mocap',
        'trainable': True,
        'license': 'AMASS research license',
        'notes': 'Existing D:\\AMASS plus converted G1 references.',
    },
    'AISTPP': {
        'kind': 'video_dual',
        'trainable': True,
        'license': 'AIST++ / AIST Dance DB research terms',
        'notes': '10 dance genres, c01/c03 calibrated pairs.',
    },
    'MotionX': {
        'kind': 'video',
        'trainable': True,
        'license': 'Motion-X dataset license',
        'notes': 'Parquet video shards and extracted MP4 clips.',
    },
    'LAFAN1': {
        'kind': 'bvh',
        'trainable': False,
        'license': 'CC BY-NC-ND 4.0',
        'notes': 'BVH; requires BVH-to-G1 conversion before PPO training.',
    },
    '100STYLE': {
        'kind': 'bvh',
        'trainable': False,
        'license': 'CC BY 4.0',
        'notes': 'BVH; requires BVH-to-G1 conversion before PPO training.',
    },
    'BEDLAM': {
        'kind': 'video',
        'trainable': False,
        'license': 'BEDLAM research license, registration required',
        'notes': 'Directory reserved; download after accepting the license.',
    },
    'Human3.6M': {
        'kind': 'video_dual',
        'trainable': False,
        'license': 'Human3.6M research license, registration required',
        'notes': 'Directory reserved; download after registration.',
    },
    'MPI_INF_3DHP': {
        'kind': 'video_dual',
        'trainable': False,
        'license': 'MPI-INF-3DHP research terms',
        'notes': 'Directory reserved; download after registration.',
    },
    'TotalCapture': {
        'kind': 'video_dual',
        'trainable': False,
        'license': 'TotalCapture research license',
        'notes': 'Directory reserved; request access from the authors.',
    },
    'CMUPanoptic': {
        'kind': 'video_dual',
        'trainable': False,
        'license': 'CMU Panoptic research-only license',
        'notes': 'Directory reserved; multi-view download is large.',
    },
    'MoVi': {
        'kind': 'video_dual',
        'trainable': False,
        'license': 'MoVi non-commercial research license',
        'notes': 'Directory reserved; request access.',
    },
    'CMU_MoCap': {
        'kind': 'mocap',
        'trainable': False,
        'license': 'CMU mocap free-use terms',
        'notes': 'Directory reserved; ASF/AMC or BVH conversion required.',
    },
}

ALIASES = {
    'AIST++': 'AISTPP',
    'AIST_PP': 'AISTPP',
    'MOTION-X': 'MotionX',
    'MOTION_X': 'MotionX',
    '100STYLE': '100STYLE',
}


def _canonical(name):
    value = str(name).strip()
    for key in SPECS:
        if key.lower() == value.lower():
            return key
    return ALIASES.get(value.upper(), value)


def dataset_dir(name):
    return DATASET_ROOT / _canonical(name)


def _files(path, pattern='*'):
    path = Path(path)
    if not path.exists():
        return []
    return sorted(item for item in path.rglob(pattern) if item.is_file())


def _bytes(path):
    return int(sum(item.stat().st_size for item in _files(path)))


def _json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def _aistpp_pairs():
    root = dataset_dir('AISTPP')
    records = _json(root / 'manifests' / 'pairs.json', [])
    result = []
    for index, item in enumerate(records):
        a = (root / item['a']).resolve()
        b = (root / item['b']).resolve()
        calibration = (root / item['calibration']).resolve()
        if not a.is_file() or not b.is_file():
            continue
        result.append({
            'dataset': 'AISTPP',
            'kind': 'dual_view',
            'index': index,
            'key': item.get('key', a.stem),
            'pair_key': item.get('pair_key', a.stem),
            'a': str(a),
            'b': str(b),
            'calibration': str(calibration),
        })
    return result


def _motionx_clips():
    root = dataset_dir('MotionX')
    return [{
        'dataset': 'MotionX',
        'kind': 'video',
        'index': index,
        'key': path.stem,
        'path': str(path),
    } for index, path in enumerate(_files(root / 'videos', '*.mp4'))]


def _amass_clips():
    return [{
        'dataset': 'AMASS',
        'kind': 'reference',
        'index': index,
        'key': path.stem,
        'path': str(path),
    } for index, path in enumerate(_files(AMASS_MOTIONS, '*.npz'))]


def _bvh_clips(name):
    root = dataset_dir(name)
    return [{
        'dataset': name,
        'kind': 'bvh',
        'index': index,
        'key': path.stem,
        'path': str(path),
    } for index, path in enumerate(_files(root, '*.bvh'))]


def clips(dataset, query=None, limit=50, offset=0):
    name = _canonical(dataset)
    if name == 'AISTPP':
        values = _aistpp_pairs()
    elif name == 'MotionX':
        values = _motionx_clips()
    elif name == 'AMASS':
        values = _amass_clips()
    elif name in ('LAFAN1', '100STYLE', 'CMU_MoCap'):
        values = _bvh_clips(name)
    else:
        values = []
    if query:
        needle = str(query).lower()
        values = [item for item in values
                  if needle in str(item.get('key', '')).lower()
                  or needle in str(item.get('path', '')).lower()]
    offset = max(0, int(offset))
    limit = max(0, int(limit))
    return values[offset:offset + limit] if limit else values[offset:]


def inventory():
    report = []
    for name, spec in SPECS.items():
        root = dataset_dir(name)
        item = {
            'name': name, 'root': str(root), 'kind': spec['kind'],
            'trainable': bool(spec['trainable']),
            'license': spec['license'], 'notes': spec['notes'],
            'exists': root.is_dir(), 'bytes': _bytes(root) if root.is_dir() else 0,
        }
        if name == 'AISTPP':
            item.update({
                'dual_view_pairs': len(_aistpp_pairs()),
                'videos': len(_files(root / 'videos', '*.mp4')),
                'calibrations': len(_files(root / 'calibrations', '*.json')),
                'annotations': len(_files(root / 'annotations', '*')),
            })
        elif name == 'MotionX':
            item.update({
                'videos': len(_files(root / 'videos', '*.mp4')),
                'parquet_shards': len(_files(root / 'parquet', '*.parquet')),
            })
        elif name == 'AMASS':
            item.update({
                'source_root': r'D:\AMASS',
                'references': len(_amass_clips()),
            })
        elif name in ('LAFAN1', '100STYLE', 'CMU_MoCap'):
            item['bvh_files'] = len(_bvh_clips(name))
        if not item['exists']:
            item['status'] = 'missing_directory'
        elif name in ('BEDLAM', 'Human3.6M', 'MPI_INF_3DHP',
                      'TotalCapture', 'CMUPanoptic', 'MoVi'):
            item['status'] = 'requires_license'
        elif name in ('LAFAN1', '100STYLE', 'CMU_MoCap'):
            if item.get('bvh_files', 0):
                item['status'] = 'downloaded_needs_conversion'
            elif name == '100STYLE' and (root / '100STYLE.zip').is_file():
                item['status'] = 'downloading_or_needs_unzip'
            else:
                item['status'] = 'empty'
        elif name == 'MotionX' and not item.get('videos') \
                and item.get('parquet_shards'):
            item['status'] = 'parquet_needs_extraction'
        else:
            item['status'] = 'ready' if item.get('videos', item.get(
                'dual_view_pairs', item.get('references', 0))) else 'empty'
        report.append(item)
    return {'root': str(DATASET_ROOT), 'datasets': report}


def build_manifests():
    root = DATASET_ROOT
    index = root / '00_INDEX'
    index.mkdir(parents=True, exist_ok=True)
    for name in ('MotionX', 'LAFAN1', '100STYLE', 'AMASS'):
        folder = dataset_dir(name) / 'manifests'
        folder.mkdir(parents=True, exist_ok=True)
    motionx = _motionx_clips()
    (dataset_dir('MotionX') / 'manifests' / 'video_list.txt').write_text(
        ''.join(item['path'] + '\n' for item in motionx), encoding='utf-8')
    for name in ('LAFAN1', '100STYLE'):
        values = _bvh_clips(name)
        (dataset_dir(name) / 'manifests' / 'bvh_list.txt').write_text(
            ''.join(item['path'] + '\n' for item in values), encoding='utf-8')
    amass = _amass_clips()
    (dataset_dir('AMASS') / 'manifests' / 'reference_list.txt').write_text(
        ''.join(item['path'] + '\n' for item in amass), encoding='utf-8')
    catalog = inventory()
    (index / 'catalog.json').write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2), encoding='utf-8')
    return catalog


def extract_motionx(parquet_path, limit=None):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError('读取 Motion-X parquet 需要 pyarrow: '
                           'python -m pip install pyarrow') from exc
    parquet_path = Path(parquet_path).resolve()
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    output = dataset_dir('MotionX') / 'videos'
    output.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(parquet_path)
    if 'video' not in table.column_names:
        raise ValueError('parquet 中没有 video 列: %s' %
                         ','.join(table.column_names))
    rows = table.to_pylist()
    if limit:
        rows = rows[:int(limit)]
    written = []
    for index, row in enumerate(rows):
        value = row.get('video')
        data = None
        name = None
        if isinstance(value, (bytes, bytearray)):
            data = bytes(value)
        elif isinstance(value, dict):
            data = value.get('bytes')
            name = value.get('path')
        if data is None:
            continue
        if name:
            stem = Path(str(name)).stem
        else:
            stem = '%s_%06d' % (parquet_path.stem.replace(
                '-of-', '_'), index)
        target = output / (stem + '.mp4')
        target.write_bytes(data)
        if target.stat().st_size > 1000:
            written.append(str(target))
        else:
            target.unlink(missing_ok=True)
    manifest = dataset_dir('MotionX') / 'manifests' / 'video_list.txt'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        ''.join(path + '\n' for path in written), encoding='utf-8')
    report = {
        'parquet': str(parquet_path),
        'rows': len(rows),
        'videos': len(written),
        'output': str(output),
        'manifest': str(manifest),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _pick_clip(name, clip):
    values = clips(name, limit=0)
    if not values:
        raise RuntimeError('%s 没有可用片段' % name)
    if clip is None:
        return values[0]
    text = str(clip)
    if text.isdigit():
        index = int(text)
        if 0 <= index < len(values):
            return values[index]
    for item in values:
        if text in (item.get('key'), item.get('pair_key'), item.get('path')):
            return item
    raise KeyError('找不到片段: %s' % clip)


def start_training(dataset, clip=None, iterations=500, envs=4, device='auto',
                   render=False, backend='dlt', fps=30., max_seconds=60.,
                   rl_pd_influence=.7, hand_range=(.2, .6),
                   foot_range=(.5, .9), skill_name=None,
                   repair_gap_seconds=.7, dry_run=False):
    name = _canonical(dataset)
    selected = _pick_clip(name, clip)
    config = {
        'dataset': name, 'clip': selected, 'iterations': int(iterations),
        'num_envs': int(envs), 'device': device, 'render': bool(render),
        'backend': backend, 'fps': float(fps), 'max_seconds': float(max_seconds),
        'rl_pd_influence': float(rl_pd_influence),
        'hand_rl_pd_range': list(hand_range),
        'foot_rl_pd_range': list(foot_range),
        'repair_gap_seconds': float(repair_gap_seconds),
    }
    if dry_run:
        return {'dry_run': True, **config}
    from .jobs import TrainingJobs
    jobs = TrainingJobs()
    common = dict(
        iterations=iterations, device=device, render=render,
        num_envs=envs, rl_pd_influence=rl_pd_influence,
        hand_rl_pd_range=list(hand_range),
        foot_rl_pd_range=list(foot_range),
        skill_name=skill_name)
    if selected['kind'] == 'dual_view':
        return jobs.start_multiview(
            str(dataset_dir('AISTPP')),
            list_file='manifests/dual_view.txt',
            pair=selected['pair_key'],
            fallback=True,
            backend=backend,
            calibration=selected['calibration'],
            fps=fps,
            max_seconds=max_seconds,
            repair_gap_seconds=repair_gap_seconds,
            **common)
    if selected['kind'] == 'reference':
        return jobs.start_reference(selected['path'], **common)
    if selected['kind'] == 'video':
        return jobs.start(
            selected['path'], max_seconds=max_seconds, fps=fps, **common)
    raise RuntimeError(
        '%s 是 %s 数据，当前训练器需要先转换为 reference.npz 或视频'
        % (name, selected['kind']))


def job_status(job_id):
    from data_paths import RUNS
    folder = Path(RUNS) / 'chat_training' / str(job_id)
    status = _json(folder / 'status.json', {})
    status['job_id'] = str(job_id)
    status['output_dir'] = str(folder)
    status['worker_log'] = str(folder / 'worker.log')
    return status


def stop_job(job_id):
    from data_paths import RUNS
    folder = Path(RUNS) / 'chat_training' / str(job_id)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    (folder / 'STOP').touch()
    return {'job_id': str(job_id), 'stop_requested': True,
            'message': '将在安全检查点停止并保留最新权重'}


def resume_job(job_id, iterations=30000, device=None, render=None):
    from data_paths import RUNS
    from .jobs import ROOT, atomic_json
    folder = Path(RUNS) / 'chat_training' / str(job_id)
    config = _json(folder / 'config.json')
    if not config:
        raise FileNotFoundError('任务配置不存在: %s' % folder)
    status = _json(folder / 'status.json', {})
    if status.get('state') == 'training':
        raise RuntimeError('任务仍在训练，请先执行 stop 并等待 cancelled')
    latest = folder / 'training' / 'latest.pt'
    if not latest.is_file():
        raise FileNotFoundError('续训checkpoint不存在: %s' % latest)
    config['mode'] = 'reference'
    config['iterations'] = int(iterations)
    config['resume'] = str(latest.resolve())
    config['reference_source'] = str(folder / 'reference.npz')
    if device:
        config['device'] = str(device)
    if render is not None:
        config['render'] = bool(render)
    atomic_json(folder / 'config.json', config)
    (folder / 'STOP').unlink(missing_ok=True)
    with (folder / 'worker.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen(
            [sys.executable, '-u', '-m', 'fullbody.job_worker', str(folder)],
            cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return {
        'job_id': str(job_id),
        'target_iterations': int(iterations),
        'resume': str(latest),
        'worker_pid': process.pid,
        'output_dir': str(folder),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='G1 本地数据集注册与训练入口')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('inventory')
    p = sub.add_parser('list')
    p.add_argument('--dataset', required=True)
    p.add_argument('--query')
    p.add_argument('--limit', type=int, default=50)
    p.add_argument('--offset', type=int, default=0)
    sub.add_parser('build')
    p = sub.add_parser('extract-motionx')
    p.add_argument('--parquet', required=True)
    p.add_argument('--limit', type=int)
    p = sub.add_parser('train')
    p.add_argument('--dataset', required=True)
    p.add_argument('--clip')
    p.add_argument('--iterations', type=int, default=500)
    p.add_argument('--envs', type=int, default=4)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--backend', choices=('auto', 'dlt', 'smplx'), default='dlt')
    p.add_argument('--fps', type=float, default=30.)
    p.add_argument('--max-seconds', type=float, default=60.)
    p.add_argument('--render', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--rl-pd-influence', type=float, default=.7)
    p.add_argument('--hand-rl-pd-range', type=float, nargs=2, default=(.2, .6))
    p.add_argument('--foot-rl-pd-range', type=float, nargs=2, default=(.5, .9))
    p.add_argument('--repair-gap-seconds', type=float, default=.7)
    p.add_argument('--skill-name')
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('status')
    p.add_argument('job_id')
    p = sub.add_parser('stop')
    p.add_argument('job_id')
    p = sub.add_parser('resume')
    p.add_argument('job_id')
    p.add_argument('--iterations', type=int, default=30000)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    p.add_argument('--render', action='store_true', default=None)
    args = parser.parse_args(argv)
    if args.command == 'inventory':
        print(json.dumps(inventory(), ensure_ascii=False, indent=2))
    elif args.command == 'list':
        values = clips(args.dataset, args.query, args.limit, args.offset)
        print(json.dumps(values, ensure_ascii=False, indent=2))
    elif args.command == 'build':
        print(json.dumps(build_manifests(), ensure_ascii=False, indent=2))
    elif args.command == 'extract-motionx':
        extract_motionx(args.parquet, args.limit)
    elif args.command == 'status':
        print(json.dumps(job_status(args.job_id),
                         ensure_ascii=False, indent=2))
    elif args.command == 'stop':
        print(json.dumps(stop_job(args.job_id),
                         ensure_ascii=False, indent=2))
    elif args.command == 'resume':
        print(json.dumps(
            resume_job(args.job_id, args.iterations, args.device, args.render),
            ensure_ascii=False, indent=2))
    else:
        report = start_training(
            dataset=args.dataset, clip=args.clip,
            iterations=args.iterations, envs=args.envs,
            device=args.device, render=args.render, backend=args.backend,
            fps=args.fps, max_seconds=args.max_seconds,
            rl_pd_influence=args.rl_pd_influence,
            hand_range=args.hand_rl_pd_range,
            foot_range=args.foot_rl_pd_range,
            skill_name=args.skill_name,
            repair_gap_seconds=args.repair_gap_seconds,
            dry_run=args.dry_run)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
