"""独立训练进程，聊天工具无需等待姿态提取或训练完成。"""
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime
from .jobs import read_json, atomic_json, ROOT
from proactive import format_duration
from data_paths import OUTPUTS as DATA_OUTPUTS, RUNS as DATA_RUNS


def run(folder):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    folder=Path(folder); config=read_json(folder/'config.json')
    job_started=time.monotonic()
    training_started=None
    training_total=None
    def status(state,**extra):
        nonlocal training_total
        previous=read_json(folder/'status.json',{})
        now=time.monotonic()
        payload={**previous,'state':state,
            'started_at':config.get('started_at') or previous.get('started_at'),
            'elapsed_seconds':round(now-job_started,3),
            'elapsed_text':format_duration(now-job_started)}
        if training_started is not None:
            training_total=now-training_started
        if training_total is not None:
            payload['training_seconds']=round(training_total,3)
            payload['training_text']=format_duration(training_total)
        payload.update(extra)
        if 'duration_seconds' in extra:
            payload['duration_text']=format_duration(extra['duration_seconds'])
        last_error=None
        for attempt in range(8):
            try:
                atomic_json(folder/'status.json',payload)
                return
            except (PermissionError,OSError) as exc:
                last_error=exc
                time.sleep(min(.05*(2**attempt),.8))
        print('[status] 状态写入暂时失败，训练继续: %s' % last_error,
              file=sys.stderr,flush=True)
    def finish(state,**extra):
        nonlocal training_total
        now=time.monotonic()
        extra.setdefault('duration_seconds',round(now-job_started,3))
        extra.setdefault('finished_at',datetime.now().astimezone().isoformat(timespec='seconds'))
        if training_started is not None:
            training_total=now-training_started
        if training_total is not None:
            extra.setdefault('training_seconds',round(training_total,3))
        status(state,**extra)
    def stopped(): return (folder/'STOP').exists()
    progress_bucket=[-1]
    fit_bucket=[-1]
    def detect_progress(current,total):
        percent=int(round(100.0*current/max(1,total)))
        bucket=percent//5
        if current==total or bucket!=progress_bucket[0]:
            progress_bucket[0]=bucket
            status('extracting',message='提取视频骨架 %d/%d（%d%%）'%(current,total,percent),
                extraction={'current':int(current),'total':int(total),'percent':percent})
    def fit_progress(frame,total,error):
        if total<=0: return
        percent=int(round(100.0*frame/max(1,total)))
        bucket=percent//5
        if frame==0 or frame==total or bucket!=fit_bucket[0]:
            fit_bucket[0]=bucket
            detail='' if error is None else '，关节误差 %.3f m'%error
            status('extracting',message='SMPL-X 拟合 %d/%d（%d%%）%s'%(frame,total,percent,detail),
                extraction={'stage':'smplx','current':int(frame),'total':int(total),'percent':percent})
    def multiview_progress(stage,current,total):
        percent=int(round(100.0*current/max(1,total)))
        status('extracting',message='双视角 %s %d/%d（%d%%）'%(stage,current,total,percent),
            extraction={'stage':stage,'current':int(current),'total':int(total),'percent':percent})
    try:
        import torch
        device=config['device']
        if device=='auto': device='cuda' if torch.cuda.is_available() else 'cpu'
        if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA不可用，请选择cpu或安装适配PyTorch')
        status('preparing',device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,message='准备动作学习数据')
        mode=config.get('mode','video')
        if mode=='web_action':
            from .web_pipeline import prepare
            prepare(config,folder,status,stopped,detect_progress,fit_progress)
        elif mode=='multiview':
            status('extracting',message='双视角同步、2D关键点与DLT三角化')
            from video_motion.multiview import extract as extract_multiview
            try:
                report=extract_multiview(
                    config['video_a'],config['video_b'],folder/'pose_full.npz',
                    fps=config.get('fps',30.),max_seconds=config.get('max_seconds',60.),
                    quality=config.get('quality'),calibration=config.get('calibration'),
                    fov=config.get('fov',60.),baseline=config.get('baseline',2.),
                    azimuth_b=config.get('azimuth_b',90.),
                    offset_a=config.get('offset_a',0.),offset_b=config.get('offset_b',0.),
                    min_visibility=config.get('min_visibility',.4),
                    max_reprojection=config.get('max_reprojection',40.),
                    self_correct=config.get('self_correct',True),
                    repair_gap_seconds=config.get('repair_gap_seconds',.25),
                    auto_sync=config.get('auto_sync',True),
                    max_sync_offset_seconds=config.get('max_sync_offset_seconds',12.),
                    sync_step_seconds=config.get('sync_step_seconds',.5),
                    min_valid_joint_fraction=config.get('min_valid_joint_fraction',.05),
                    target_leg_length=config.get('target_leg_length',.85),
                    backend=config.get('backend','auto'),
                    smplx_model_dir=config.get('smplx_model_dir'),
                    smplx_gender=config.get('smplx_gender','neutral'),
                    smplx_device=config.get('smplx_device','auto'),
                    smplx_iterations=config.get('smplx_iterations',40),
                    detect_progress_a=detect_progress,detect_progress_b=detect_progress,
                    progress=multiview_progress)
                atomic_json(folder/'multiview_report.json',report)
            except Exception as exc:
                if not config.get('fallback',True):
                    raise
                status('extracting',message='双视角失败，降级为a视角单目提取',error=str(exc))
                from video_motion.pose import extract
                extract(config['video_a'],folder/'pose_full.npz',fps=config.get('fps',30.),
                    max_seconds=config.get('max_seconds',60.),quality=config.get('quality'),
                    detect_progress=detect_progress,progress=fit_progress)
                atomic_json(folder/'multiview_report.json',{'fallback':True,'error':str(exc),'source':'single_view_a'})
            from .prepare_self import trim
            trim(folder/'pose_full.npz',folder/'pose.npz')
        elif mode=='reference':
            if not (folder/'reference.npz').is_file():
                raise FileNotFoundError('指定目录训练缺少 reference.npz')
            status('retargeting',message='使用指定目录中的参考动作',
                source_interval=config.get('source_interval'))
        else:
            status('extracting',message='提取视频骨架')
            from video_motion.pose import extract
            extract(config['video'],folder/'pose_full.npz',fps=config['fps'],max_seconds=config['max_seconds'],
                detect_progress=detect_progress,progress=fit_progress)
            if stopped(): finish('cancelled'); return
            from .prepare_self import trim
            trim(folder/'pose_full.npz',folder/'pose.npz')
        if mode!='reference':
            status('retargeting',message='全身重定向与足部支撑修正',source_interval=read_json(folder/'pose.trim.json'))
            from .reference import convert
            convert(folder/'pose.npz',folder/'reference_raw.npz',
                root_mode=config.get('root_mode','contact_grounded'),
                min_visibility=.4 if mode=='multiview' else .5)
            if stopped(): finish('cancelled'); return
            from .refine import refine
            refine(folder/'pose.npz',folder/'reference_raw.npz',folder/'reference.npz')
            if stopped(): finish('cancelled'); return
        if mode=='reference':
            events={'jump_times':config.get('jump_times') or [],'mode':config.get('jump_mode','off')}
        else:
            from coordination.video import annotate
            events=annotate(folder/'pose.npz',folder/'jump_events.json',config.get('jump_mode','off'),config.get('jump_times'))
        video_controller={'version':1,'jump_times':events['jump_times'],'jump_mode':events['mode']}
        if config.get('coordination',True):
            status('coordinator_preparing',message='准备关键节点-质心协调网络')
            import shutil
            network=folder/'com_graph.pt'
            existing=DATA_RUNS/'coordination/com_graph.pt'
            if existing.is_file(): shutil.copy2(existing,network)
            else:
                from coordination.network import train as train_coordinator
                from .model import build_model
                train_coordinator(build_model(folder/'coordinator_model'),network,device=device)
            video_controller['coordinator_checkpoint']=str(network.resolve())
        atomic_json(folder/'video_controller.json',video_controller)
        training_started=time.monotonic()
        resume=config.get('resume')
        resume_iteration=0
        train_iterations=config['iterations']
        if resume:
            resume_path=Path(resume)
            if not resume_path.is_file():
                raise FileNotFoundError('续训checkpoint不存在: '+str(resume_path))
            from runtime_io import load_checkpoint
            previous=load_checkpoint(resume_path,map_location='cpu',weights_only=True)
            resume_iteration=int(previous.get('iteration',0))
            if resume_iteration>=config['iterations']:
                finish('completed',message='已经达到目标轮次',iteration=resume_iteration,
                    total_iterations=config['iterations'],checkpoint=str(resume_path))
                return 0
            train_iterations=config['iterations']-resume_iteration
        status('training',message='PPO更新；渲染显示最近保存策略的物理回放',iteration=resume_iteration,total_iterations=config['iterations'],
            reference=str(folder/'reference.npz'),leg_weight=config['leg_weight'],arm_weight=config.get('arm_weight',1.),motion_gain=config.get('motion_gain',1.35),
            joint_assist=config.get('joint_assist',True),airborne_reward=config.get('airborne_reward',True))
        from .train_balanced import train
        def progress(record): status('training',iteration=record['iteration'],metrics=record)
        train(folder/'reference.npz',folder/'training',iterations=train_iterations,num_envs=config['num_envs'],device=device,
            leg_weight=config['leg_weight'],curriculum_target=config.get('curriculum_target'),callback=progress,should_stop=stopped,
            auto_adjust=config.get('auto_adjust',True),model_review=config.get('model_review',False),arm_weight=config.get('arm_weight',1.),
            motion_gain=config.get('motion_gain',1.35),max_episode_seconds=config.get('max_episode_seconds'),video_controller=video_controller,
            joint_assist=config.get('joint_assist',True),joint_assist_gain=config.get('joint_assist_gain',.18),airborne_reward=config.get('airborne_reward',True),
            rl_pd_influence=config.get('rl_pd_influence',1.0),hand_rl_pd_range=config.get('hand_rl_pd_range'),foot_rl_pd_range=config.get('foot_rl_pd_range'),
            resume=resume)
        if training_started is not None:
            training_total=time.monotonic()-training_started
            training_started=None
        if stopped(): finish('cancelled',message='已停止，保留已完成轮次权重'); return
        status('evaluating',message='独立物理评估最佳权重')
        from .ppo import evaluate
        report=evaluate(folder/'reference.npz',folder/'training/policy.pt',folder/'evaluation')
        skill=None
        spatial_skill=None
        registration_error=None
        if report.get('reference_completed') and not report.get('fallen'):
            try:
                from .skills import register
                skill=register(config.get('skill_name') or ('video_'+folder.name),folder,report)
                from .spatial_skills import register as register_spatial
                spatial_skill=register_spatial(config.get('skill_name') or ('video_'+folder.name),folder,report,node_skill=skill['skill_id'])
            except ValueError as exc:
                registration_error=str(exc)
        log_dir = None
        try:
            import shutil
            dest = DATA_OUTPUTS/config['job_id']
            dest.mkdir(parents=True, exist_ok=True)
            for rel in ('config.json', 'status.json', 'worker.log', 'training/metrics.jsonl', 'evaluation/evaluation.json'):
                src = folder/rel
                if src.is_file():
                    shutil.copy2(src, dest/(rel.replace('/', '_')))
            log_dir = str(dest)
        except Exception:
            log_dir = None
        finish('completed',message='训练与评估结束；入库需独立通过硬指标与跳跃事件验收',evaluation=report,
            checkpoint=str(folder/'training/policy.pt'),skill=skill,spatial_skill=spatial_skill,registration_error=registration_error,log_dir=log_dir)
    except InterruptedError:
        finish('cancelled',message='任务已停止，保留已生成数据')
        return 0
    except Exception as exc:
        finish('failed',message=str(exc),traceback=traceback.format_exc())
        traceback.print_exc()
        return 1
    return 0


if __name__=='__main__': raise SystemExit(run(sys.argv[1]))
