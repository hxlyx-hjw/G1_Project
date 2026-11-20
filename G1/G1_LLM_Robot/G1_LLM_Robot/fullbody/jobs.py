"""聊天训练任务：参数验证、后台子进程、状态与取消。"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime

from proactive import format_duration
from data_paths import RUNS as DATA_RUNS, OUTPUTS as DATA_OUTPUTS

ROOT=Path(__file__).resolve().parents[1]


def rl_pd_options(influence=1.0,hand_range=None,foot_range=None):
    """Normalize the global RL-PD influence and per-limb convergence bounds."""
    try: influence=float(influence)
    except (TypeError,ValueError) as exc: raise ValueError('rl_pd_influence必须为0~1数值') from exc
    if not 0<=influence<=1: raise ValueError('rl_pd_influence应在0~1之间')
    def bounds(value,name):
        if value is None: return None
        if not isinstance(value,(list,tuple)) or len(value)!=2:
            raise ValueError('%s必须是[min,max]两元素数组'%name)
        try: low,high=float(value[0]),float(value[1])
        except (TypeError,ValueError) as exc: raise ValueError('%s必须为数值'%name) from exc
        if not 0<=low<=high<=1: raise ValueError('%s须满足0<=min<=max<=1'%name)
        return [low,high]
    return influence,bounds(hand_range,'hand_rl_pd_range'),bounds(foot_range,'foot_rl_pd_range')


def atomic_json(path,value):
    """Atomically publish JSON with a unique temp file and Windows retries."""
    from runtime_io import atomic_write
    text=json.dumps(value,ensure_ascii=False,indent=2)
    atomic_write(Path(path),lambda temporary: temporary.write_text(text,encoding='utf-8'))


def read_json(path,default=None):
    try: return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError): return default


class TrainingJobs:
    def __init__(self):
        self.jobs={}
        self.latest=None
        # 由 RobotChatBot 注入：训练任务启动后自动开始主动报告。
        self.on_job_started=None

    def start(self,video,skill_name=None,iterations=100,device='auto',render=True,leg_weight=.15,max_seconds=60,fps=15,num_envs=4,curriculum_target=None,auto_adjust=True,model_review=False,arm_weight=1.,coordination=True,jump_mode='off',jump_times=None,motion_gain=1.35,max_episode_seconds=None,joint_assist=True,joint_assist_gain=.18,airborne_reward=True,root_mode='contact_grounded',rl_pd_influence=1.0,hand_rl_pd_range=None,foot_rl_pd_range=None):
        video=Path(video).expanduser()
        if not video.is_absolute(): video=ROOT/video
        video=video.resolve()
        if not video.is_file(): raise FileNotFoundError(video)
        if device not in ('auto','cpu','cuda'): raise ValueError('device: auto/cpu/cuda')
        if not 1<=iterations<=100000 or not 1<=num_envs<=16: raise ValueError('iterations:1~100000，num_envs:1~16')
        if not 0<=leg_weight<=2 or not 0<max_seconds<=180 or not 1<=fps<=60: raise ValueError('参数超出范围')
        if jump_mode not in ('off','auto','explicit'): raise ValueError('jump_mode:off/auto/explicit')
        if root_mode not in ('in_place','contact_grounded','auto'): raise ValueError('root_mode:in_place/contact_grounded/auto')
        if jump_mode=='explicit' and (not isinstance(jump_times,list) or not jump_times): raise ValueError('explicit需要jump_times数组')
        if not .25<=arm_weight<=1.5: raise ValueError('arm_weight:.25~1.5')
        if not 1<=motion_gain<=2: raise ValueError('motion_gain:1~2')
        if not 0<=joint_assist_gain<=2: raise ValueError('joint_assist_gain:0~2')
        if max_episode_seconds is not None and not 1<=max_episode_seconds<=600: raise ValueError('max_episode_seconds:1~600')
        if curriculum_target is not None and not leg_weight<=curriculum_target<=2: raise ValueError('curriculum_target应在leg_weight和2之间')
        rl_pd_influence,hand_rl_pd_range,foot_rl_pd_range=rl_pd_options(rl_pd_influence,hand_rl_pd_range,foot_rl_pd_range)
        for task in self.jobs.values():
            if task['worker'].poll() is None: raise RuntimeError('已有训练正在运行，请先查询或停止')
        job_id=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
        folder=DATA_RUNS/'chat_training'/job_id
        folder.mkdir(parents=True)
        config={'job_id':job_id,'skill_name':skill_name,'video':str(video),'iterations':iterations,'device':device,'render':bool(render),
            'coordination':bool(coordination),'jump_mode':jump_mode,'jump_times':jump_times,'auto_adjust':bool(auto_adjust),'model_review':bool(model_review),'arm_weight':arm_weight,'curriculum_target':curriculum_target,'leg_weight':leg_weight,'motion_gain':motion_gain,'max_episode_seconds':max_episode_seconds,'joint_assist':bool(joint_assist),'joint_assist_gain':joint_assist_gain,'airborne_reward':bool(airborne_reward),'max_seconds':max_seconds,'fps':fps,'num_envs':num_envs,'rl_pd_influence':rl_pd_influence,'hand_rl_pd_range':hand_rl_pd_range,'foot_rl_pd_range':foot_rl_pd_range,'folder':str(folder)}
        config['root_mode']=root_mode
        return self._launch(config,folder,render)

    def start_reference(self,reference,skill_name=None,dataset=None,source_motion=None,iterations=100,device='auto',render=True,leg_weight=.15,num_envs=4,curriculum_target=None,auto_adjust=True,model_review=False,arm_weight=1.,coordination=True,jump_mode='off',jump_times=None,motion_gain=1.35,max_episode_seconds=None,joint_assist=True,joint_assist_gain=.18,airborne_reward=True,root_mode='contact_grounded',rl_pd_influence=1.0,hand_rl_pd_range=None,foot_rl_pd_range=None):
        """Start a background job from an existing fullbody reference.npz."""
        reference=Path(reference).expanduser()
        if not reference.is_absolute(): reference=ROOT/reference
        reference=reference.resolve()
        if not reference.is_file(): raise FileNotFoundError(reference)
        if device not in ('auto','cpu','cuda'): raise ValueError('device: auto/cpu/cuda')
        if not 1<=iterations<=100000 or not 1<=num_envs<=16: raise ValueError('iterations:1~100000，num_envs:1~16')
        if not 0<=leg_weight<=2: raise ValueError('leg_weight超出范围')
        if jump_mode not in ('off','auto','explicit'): raise ValueError('jump_mode:off/auto/explicit')
        if root_mode not in ('in_place','contact_grounded','auto'): raise ValueError('root_mode:in_place/contact_grounded/auto')
        if jump_mode=='explicit' and (not isinstance(jump_times,list) or not jump_times): raise ValueError('explicit需要jump_times数组')
        if not .25<=arm_weight<=1.5: raise ValueError('arm_weight:.25~1.5')
        if not 1<=motion_gain<=2: raise ValueError('motion_gain:1~2')
        if not 0<=joint_assist_gain<=2: raise ValueError('joint_assist_gain:0~2')
        if max_episode_seconds is not None and not 1<=max_episode_seconds<=600: raise ValueError('max_episode_seconds:1~600')
        if curriculum_target is not None and not leg_weight<=curriculum_target<=2: raise ValueError('curriculum_target应在leg_weight和2之间')
        rl_pd_influence,hand_rl_pd_range,foot_rl_pd_range=rl_pd_options(rl_pd_influence,hand_rl_pd_range,foot_rl_pd_range)
        for task in self.jobs.values():
            if task['worker'].poll() is None: raise RuntimeError('已有训练正在运行，请先查询或停止')
        job_id=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
        folder=DATA_RUNS/'chat_training'/job_id
        folder.mkdir(parents=True)
        shutil.copy2(reference,folder/'reference.npz')
        config={'job_id':job_id,'mode':'reference','skill_name':skill_name or reference.stem,
            'dataset':dataset,'source_motion':source_motion,
            'reference_source':str(reference),'iterations':iterations,'device':device,
            'render':bool(render),'coordination':bool(coordination),'jump_mode':jump_mode,
            'jump_times':jump_times,'auto_adjust':bool(auto_adjust),'model_review':bool(model_review),
            'arm_weight':arm_weight,'curriculum_target':curriculum_target,'leg_weight':leg_weight,
            'motion_gain':motion_gain,'max_episode_seconds':max_episode_seconds,
            'joint_assist':bool(joint_assist),'joint_assist_gain':joint_assist_gain,
            'airborne_reward':bool(airborne_reward),'root_mode':root_mode,
            'rl_pd_influence':rl_pd_influence,'hand_rl_pd_range':hand_rl_pd_range,'foot_rl_pd_range':foot_rl_pd_range,
            'num_envs':num_envs,'folder':str(folder)}
        return self._launch(config,folder,render)

    def start_amass(self,motion_id=None,dataset=None,name=None,path=None,query=None,
                    skill_name=None,**options):
        """Resolve an indexed AMASS motion and train it as a local reference."""
        from .amass import resolve_motion
        motion=resolve_motion(motion_id=motion_id,dataset=dataset,name=name,
                              path=path,query=query)
        return self.start_reference(
            motion['path'],skill_name=skill_name or motion['name'],
            dataset=motion['dataset'],source_motion=motion,**options)

    def multiview_pairs(self,folder,list_file=None):
        from video_motion.multiview import discover_pairs
        return discover_pairs(folder,list_file)

    def start_multiview(self,folder,list_file=None,pair=None,fallback=True,
                        backend='auto',calibration=None,fov=60.,baseline=2.,
                        azimuth_b=90.,offset_a=0.,offset_b=0.,min_visibility=.4,
                        max_reprojection=40.,self_correct=True,
                        repair_gap_seconds=.25,auto_sync=True,
                        max_sync_offset_seconds=12.,sync_step_seconds=.5,
                        min_valid_joint_fraction=.05,target_leg_length=.85,quality=None,
                        smplx_model_dir=None,smplx_gender='neutral',
                        smplx_device='auto',smplx_iterations=40,
                        skill_name=None,iterations=500,device='auto',render=True,
                        fps=30.,max_seconds=60.,leg_weight=.15,num_envs=4,
                        curriculum_target=None,auto_adjust=True,model_review=False,
                        arm_weight=1.,coordination=True,jump_mode='off',jump_times=None,
                        motion_gain=1.35,max_episode_seconds=None,joint_assist=True,
                        joint_assist_gain=.18,airborne_reward=True,
                        root_mode='contact_grounded',rl_pd_influence=1.0,
                        hand_rl_pd_range=None,foot_rl_pd_range=None):
        if backend not in ('auto','dlt','smplx'): raise ValueError('backend:auto/dlt/smplx')
        if device not in ('auto','cpu','cuda'): raise ValueError('device:auto/cpu/cuda')
        if not 1<=iterations<=100000 or not 1<=num_envs<=16: raise ValueError('iterations/num_envs超出范围')
        if not 0<max_seconds<=180 or not 1<=fps<=60: raise ValueError('max_seconds/fps超出范围')
        if not 0<=repair_gap_seconds<=5: raise ValueError('repair_gap_seconds:0~5')
        if not 0<=max_sync_offset_seconds<=60: raise ValueError('max_sync_offset_seconds:0~60')
        if not 1/120<=sync_step_seconds<=5: raise ValueError('sync_step_seconds超出范围')
        if not 0<=min_valid_joint_fraction<=1: raise ValueError('min_valid_joint_fraction:0~1')
        if not 0<=leg_weight<=2 or not .25<=arm_weight<=1.5: raise ValueError('手腿权重超出范围')
        if not 1<=motion_gain<=2 or not 0<=joint_assist_gain<=2: raise ValueError('motion_gain/joint_assist_gain超出范围')
        if jump_mode not in ('off','auto','explicit'): raise ValueError('jump_mode:off/auto/explicit')
        if root_mode not in ('in_place','contact_grounded','auto'): raise ValueError('root_mode非法')
        if jump_mode=='explicit' and (not isinstance(jump_times,list) or not jump_times): raise ValueError('explicit需要jump_times数组')
        if max_episode_seconds is not None and not 1<=max_episode_seconds<=600: raise ValueError('max_episode_seconds:1~600')
        if curriculum_target is not None and not leg_weight<=curriculum_target<=2: raise ValueError('curriculum_target超出范围')
        rl_pd_influence,hand_rl_pd_range,foot_rl_pd_range=rl_pd_options(rl_pd_influence,hand_rl_pd_range,foot_rl_pd_range)
        from video_motion.multiview import discover_pairs
        manifest=discover_pairs(folder,list_file)
        pairs=manifest['pairs']
        if pair is None:
            selected=pairs[0]
        elif isinstance(pair,int) or (isinstance(pair,str) and pair.isdigit()):
            index=int(pair)
            if not 0<=index<len(pairs): raise ValueError('pair索引超出范围')
            selected=pairs[index]
        else:
            matches=[item for item in pairs if item['key']==str(pair) or Path(item['a']).stem==str(pair)]
            if len(matches)!=1: raise ValueError('无法唯一确定pair；请先调用pairs列表')
            selected=matches[0]
        for task in self.jobs.values():
            if task['worker'].poll() is None: raise RuntimeError('已有训练正在运行，请先查询或停止')
        job_id=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
        job_folder=DATA_RUNS/'chat_training'/job_id; job_folder.mkdir(parents=True)
        config={'job_id':job_id,'mode':'multiview','folder':str(job_folder),
            'source_folder':manifest['folder'],'list_file':manifest['list_file'],
            'pair':selected['key'],'video_a':selected['a'],'video_b':selected['b'],
            'pairs_available':len(pairs),'fallback':bool(fallback),'backend':backend,
            'calibration':calibration,'fov':float(fov),'baseline':float(baseline),
            'azimuth_b':float(azimuth_b),'offset_a':float(offset_a),'offset_b':float(offset_b),
            'min_visibility':float(min_visibility),'max_reprojection':float(max_reprojection),
            'self_correct':bool(self_correct),'repair_gap_seconds':float(repair_gap_seconds),
            'auto_sync':bool(auto_sync),'max_sync_offset_seconds':float(max_sync_offset_seconds),
            'sync_step_seconds':float(sync_step_seconds),
            'min_valid_joint_fraction':float(min_valid_joint_fraction),
            'target_leg_length':float(target_leg_length),'quality':quality,
            'smplx_model_dir':smplx_model_dir,'smplx_gender':smplx_gender,
            'smplx_device':smplx_device,'smplx_iterations':int(smplx_iterations),
            'skill_name':skill_name or selected['key'],
            'iterations':iterations,'device':device,'render':bool(render),'fps':float(fps),
            'max_seconds':float(max_seconds),'leg_weight':leg_weight,'num_envs':num_envs,
            'curriculum_target':curriculum_target,'auto_adjust':bool(auto_adjust),
            'model_review':bool(model_review),'arm_weight':arm_weight,'coordination':bool(coordination),
            'jump_mode':jump_mode,'jump_times':jump_times,'motion_gain':motion_gain,
            'max_episode_seconds':max_episode_seconds,'joint_assist':bool(joint_assist),
            'joint_assist_gain':joint_assist_gain,'airborne_reward':bool(airborne_reward),
            'root_mode':root_mode,'rl_pd_influence':rl_pd_influence,
            'hand_rl_pd_range':hand_rl_pd_range,'foot_rl_pd_range':foot_rl_pd_range}
        return self._launch(config,job_folder,render)

    def amass_datasets(self,refresh=False):
        from .amass import dataset_inventory
        return dataset_inventory(refresh=refresh)

    def amass_list(self,dataset=None,query=None,limit=50,offset=0,refresh=False):
        from .amass import list_motions
        return list_motions(dataset=dataset,query=query,limit=limit,
                            offset=offset,refresh=refresh)

    def _worker_env(self):
        env=os.environ.copy()
        if getattr(self,'tavily_key',None): env['TAVILY_API_KEY']=self.tavily_key
        for key,value in getattr(self,'model_settings',{}).items():
            if value: env[key]=value
        return env

    def _launch(self,config,folder,render):
        job_id=config['job_id']
        started_at=datetime.now().astimezone().isoformat(timespec='seconds')
        config['started_at']=started_at
        atomic_json(folder/'config.json',config)
        atomic_json(folder/'status.json',{'job_id':job_id,'state':'queued','message':'后台任务已创建','started_at':started_at,'elapsed_seconds':0.0,'elapsed_text':'0秒'})
        with (folder/'worker.log').open('w',encoding='utf-8') as log:
            worker=subprocess.Popen([sys.executable,'-u','-m','fullbody.job_worker',str(folder)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=self._worker_env())
        self.jobs[job_id]={'folder':folder,'worker':worker,'viewer':None,'started_at':started_at,'started_monotonic':time.monotonic()}
        self.latest=job_id
        if render:
            try: self.open_viewer(job_id)
            except Exception as exc: atomic_json(folder/'viewer_status.json',{'state':'failed','error':str(exc)})
        result=self.status(job_id)
        if self.on_job_started is not None:
            try: self.on_job_started(job_id)
            except Exception: pass
        return result

    def search_videos(self,action,limit=5):
        from .web_video import search
        return search(action,limit,getattr(self,'tavily_key',None))

    def learn_action(self,action,source_urls=None,iterations=100,device='auto',render=True,leg_weight=.15,
                     max_seconds=30,fps=15,num_envs=4,max_candidates=3,curriculum_target=None,auto_adjust=True,model_review=False,arm_weight=1.,coordination=True,jump_mode='off',jump_times=None,motion_gain=1.35,max_episode_seconds=None,joint_assist=True,joint_assist_gain=.18,airborne_reward=True,root_mode='contact_grounded',rl_pd_influence=1.0,hand_rl_pd_range=None,foot_rl_pd_range=None):
        from .web_video import valid_url
        if not isinstance(action,str) or not action.strip() or len(action)>300: raise ValueError('动作描述须为1~300字符')
        if not 1<=iterations<=100000 or not 1<=num_envs<=16 or not 1<=max_candidates<=5: raise ValueError('训练或候选预算超出范围')
        if device not in ('auto','cpu','cuda') or not 0<=leg_weight<=2 or not 0<max_seconds<=180 or not 1<=fps<=60: raise ValueError('参数超出范围')
        if jump_mode not in ('off','auto','explicit'): raise ValueError('jump_mode:off/auto/explicit')
        if root_mode not in ('in_place','contact_grounded','auto'): raise ValueError('root_mode:in_place/contact_grounded/auto')
        if jump_mode=='explicit' and (not isinstance(jump_times,list) or not jump_times): raise ValueError('explicit需要jump_times数组')
        if not .25<=arm_weight<=1.5: raise ValueError('arm_weight:.25~1.5')
        if not 1<=motion_gain<=2: raise ValueError('motion_gain:1~2')
        if not 0<=joint_assist_gain<=2: raise ValueError('joint_assist_gain:0~2')
        if max_episode_seconds is not None and not 1<=max_episode_seconds<=600: raise ValueError('max_episode_seconds:1~600')
        if curriculum_target is not None and not leg_weight<=curriculum_target<=2: raise ValueError('课程目标超出范围')
        rl_pd_influence,hand_rl_pd_range,foot_rl_pd_range=rl_pd_options(rl_pd_influence,hand_rl_pd_range,foot_rl_pd_range)
        if source_urls is not None:
            if not isinstance(source_urls,list) or len(source_urls)>10: raise ValueError('source_urls须为最多10个URL的数组')
            for url in source_urls: valid_url(url)
        for task in self.jobs.values():
            if task['worker'].poll() is None: raise RuntimeError('已有训练或自主学习任务正在运行')
        job_id=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
        folder=DATA_RUNS/'chat_training'/job_id; folder.mkdir(parents=True)
        config={'job_id':job_id,'folder':str(folder),'mode':'web_action','action':action,'source_urls':source_urls,
            'skill_name':action,'iterations':iterations,'device':device,'render':bool(render),'leg_weight':leg_weight,
            'max_seconds':max_seconds,'fps':fps,'num_envs':num_envs,'max_candidates':max_candidates,'curriculum_target':curriculum_target,
            'coordination':bool(coordination),'jump_mode':jump_mode,'jump_times':jump_times,'auto_adjust':bool(auto_adjust),'model_review':bool(model_review),'arm_weight':arm_weight,
            'motion_gain':motion_gain,'max_episode_seconds':max_episode_seconds,'joint_assist':bool(joint_assist),'joint_assist_gain':joint_assist_gain,'airborne_reward':bool(airborne_reward),
            'rl_pd_influence':rl_pd_influence,'hand_rl_pd_range':hand_rl_pd_range,'foot_rl_pd_range':foot_rl_pd_range}
        config['root_mode']=root_mode
        return self._launch(config,folder,render)

    def _task(self,job_id=None):
        job_id=job_id or self.latest
        if job_id not in self.jobs: raise ValueError('没有此训练任务，请使用start返回的job_id')
        return job_id,self.jobs[job_id]

    def status(self,job_id=None):
        job_id,task=self._task(job_id)
        status=read_json(task['folder']/'status.json',{})
        exitcode=task['worker'].poll()
        if exitcode is not None and status.get('state') not in ('completed','failed','cancelled'):
            status.update(state='failed',message='训练进程意外退出',exit_code=exitcode)
        started=status.get('started_at') or task.get('started_at')
        if started: status['started_at']=started
        duration=status.get('duration_seconds')
        if isinstance(duration,(int,float)):
            elapsed=float(duration)
            status['duration_text']=status.get('duration_text') or format_duration(elapsed)
        else:
            started_monotonic=task.get('started_monotonic')
            elapsed=(time.monotonic()-started_monotonic) if isinstance(started_monotonic,(int,float)) else status.get('elapsed_seconds',0.)
        if isinstance(elapsed,(int,float)):
            status['elapsed_seconds']=round(float(elapsed),3)
            status['elapsed_text']=(status.get('elapsed_text') or format_duration(elapsed)
                                    if status.get('duration_seconds') is not None
                                    else format_duration(elapsed))
        training_seconds=status.get('training_seconds')
        if isinstance(training_seconds,(int,float)):
            status['training_text']=status.get('training_text') or format_duration(training_seconds)
        iteration=status.get('iteration'); total=status.get('total_iterations')
        if isinstance(iteration,(int,float)) and isinstance(total,(int,float)) and total>0 and iteration>0 and status.get('state')=='training':
            basis=status.get('training_seconds')
            if not isinstance(basis,(int,float)) and isinstance(task.get('started_monotonic'),(int,float)):
                basis=time.monotonic()-task['started_monotonic']
            rate=(float(basis)/max(float(iteration),1e-9)) if isinstance(basis,(int,float)) else None
            if rate:
                status['eta_seconds']=round(rate*max(0.,float(total)-float(iteration)),1)
                status['eta_text']=format_duration(status['eta_seconds'])
        status.update(job_id=job_id,output_dir=str(task['folder']),worker_running=exitcode is None,
            viewer=read_json(task['folder']/'viewer_status.json',{'state':'not_requested'}),
            viewer_process_running=bool(task['viewer'] and task['viewer'].poll() is None))
        return status

    def stop(self,job_id=None):
        job_id,task=self._task(job_id)
        (task['folder']/'STOP').touch()
        return {'job_id':job_id,'stop_requested':True,'message':'请求在安全检查点停止并保留权重；渲染窗口继续保留'}

    def open_viewer(self,job_id=None):
        job_id,task=self._task(job_id)
        if task['viewer'] and task['viewer'].poll() is None: return {'job_id':job_id,'viewer':'already_open'}
        with (task['folder']/'viewer.log').open('a',encoding='utf-8') as log:
            task['viewer']=subprocess.Popen([sys.executable,'-u','-m','fullbody.training_viewer',str(task['folder'])],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        return {'job_id':job_id,'viewer':'opening'}

    def export_gif(self,job_id=None,checkpoint=None,output=None,fps=15,max_seconds=30):
        if not 1<=fps<=30 or not 0<max_seconds<=60: raise ValueError('fps:1~30，max_seconds:(0,60]')
        if checkpoint is None:
            _,task=self._task(job_id)
            checkpoint=task['folder']/'training/policy.pt'
        checkpoint=Path(checkpoint)
        if not checkpoint.is_absolute(): checkpoint=ROOT/checkpoint
        if not checkpoint.is_file(): raise FileNotFoundError('尚无可导出权重：'+str(checkpoint))
        export_id='gif_'+uuid.uuid4().hex[:10]
        folder=DATA_RUNS/'gif_exports'/export_id; folder.mkdir(parents=True)
        output=Path(output) if output else DATA_OUTPUTS/'gifs'/(export_id+'.gif')
        if not output.is_absolute(): output=ROOT/output
        if output.suffix.lower()!='.gif': raise ValueError('输出必须为.gif')
        if output.exists(): raise ValueError('输出已存在，请换一个文件名')
        status_path=folder/'status.json'
        atomic_json(status_path,{'state':'queued','output':str(output)})
        with (folder/'render.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,'-u','-m','fullbody.gif_export',str(checkpoint),str(output),
                '--fps',str(fps),'--max-seconds',str(max_seconds),'--status',str(status_path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if not hasattr(self,'exports'): self.exports={}
        self.exports[export_id]={'process':proc,'status':status_path}
        return {'export_id':export_id,'state':'queued','output':str(output),'message':'后台渲染GIF，使用robot_gif_status查询结果'}

    def gif_status(self,export_id):
        task=getattr(self,'exports',{}).get(export_id)
        if not task: raise ValueError('未知export_id')
        state=read_json(task['status'],{})
        code=task['process'].poll()
        if code is not None and state.get('state') not in ('completed','failed'): state.update(state='failed',exit_code=code)
        return {'export_id':export_id,**state}

    def skills_list(self):
        from .skills import list_skills
        return list_skills()

    def skill_get(self,skill_id):
        from .skills import get
        return get(skill_id)

    def run_skill(self,skill_id,viewer=True):
        item=self.skill_get(skill_id)
        if not item.get('safe_to_run'): raise ValueError('技能未标记为可运行')
        from .ppo import evaluate
        out=DATA_RUNS/'skill_runs'/(skill_id+'_'+uuid.uuid4().hex[:6])
        report=evaluate(item['reference'],item['checkpoint'],out,viewer=bool(viewer))
        return {'skill_id':skill_id,'name':item['name'],'run_dir':str(out),'evaluation':report}

    def spatial_skills_list(self):
        from .spatial_skills import list_skills
        return list_skills()

    def spatial_skill_get(self,skill_id):
        from .spatial_skills import get
        return get(skill_id)

    def run_spatial_skill(self,skill_id,mode='spatial',viewer=True):
        from .spatial_run import run
        return {'skill_id':skill_id,'mode':mode,'viewer':bool(viewer),'result':run(skill_id,mode,viewer)}

    def close(self):
        # 聊天结束请求停止训练；窗口由用户自行关闭，仍可查看最后权重。
        for task in self.jobs.values():
            if task['worker'].poll() is None: (task['folder']/'STOP').touch()
