"""搜索→候选下载→解码与骨架筛选；通过后交回现有训练流程。"""
from pathlib import Path
import shutil
import numpy as np
from .jobs import atomic_json
from .web_video import search,download,inspect_video


def prepare(config,folder,status,stopped,detect_progress=None,fit_progress=None):
    folder=Path(folder)
    urls=config.get('source_urls')
    if urls:
        result={'action':config['action'],'provider':'harness_or_user_urls','candidates':[{'url':url,'title':'','semantic_match_verified':False} for url in urls]}
    else:
        status('searching',message='搜索动作演示视频候选')
        result=search(config['action'],limit=config['max_candidates'])
    atomic_json(folder/'search.json',result)
    candidates=result['candidates'][:config['max_candidates']]
    if not candidates: raise ValueError('没有可用视频搜索结果；可传入Harness搜索得到的视频URL或本地视频')
    attempts=[]
    for index,candidate in enumerate(candidates):
        if stopped(): raise InterruptedError('任务已取消')
        directory=folder/'candidates'/str(index+1)
        try:
            status('downloading',message='下载并验证候选视频',candidate_index=index+1,candidate_count=len(candidates),source_url=candidate['url'])
            video,source=download(candidate['url'],directory,stopped)
            source.update(action=config['action'],search_candidate=candidate,video_info=inspect_video(video))
            if stopped(): raise InterruptedError('任务已取消')
            status('screening',message='解码成功，检测连续全身动作',candidate_index=index+1)
            from video_motion.pose import extract
            from .prepare_self import trim
            extract(video,directory/'pose_full.npz',fps=config['fps'],max_seconds=config['max_seconds'],
                detect_progress=detect_progress,progress=fit_progress)
            trim(directory/'pose_full.npz',directory/'pose.npz')
            with np.load(directory/'pose.npz') as z: duration=float(z['times'][-1]-z['times'][0])
            if duration<.5: raise ValueError('连续全身动作不足0.5秒')
            if stopped(): raise InterruptedError('任务已取消')
            for name in ('pose_full.npz','pose.npz','pose.trim.json'): shutil.copy2(directory/name,folder/name)
            source.update(selected_candidate=index+1,screening={'continuous_fullbody_seconds':duration,'passed':True},semantic_match_verified=False)
            atomic_json(folder/'source.json',source)
            attempts.append({'url':candidate['url'],'state':'selected','duration':duration})
            atomic_json(folder/'candidate_attempts.json',attempts)
            config['video']=str(video.resolve())
            atomic_json(folder/'config.json',config)
            return
        except InterruptedError: raise
        except Exception as exc:
            attempts.append({'url':candidate['url'],'state':'rejected','error':str(exc)})
            atomic_json(folder/'candidate_attempts.json',attempts)
    raise ValueError('所有候选视频均未通过下载或全身动作筛选，详见candidate_attempts.json')
