"""Harness搜索适配与可下载视频筛选；搜索结果是候选，不是动作识别。"""
from pathlib import Path
from urllib.parse import urlparse
import json
import os
import time
import requests

MAX_BYTES=200*1024*1024


def valid_url(url):
    parsed=urlparse(url)
    if parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('视频地址必须为http/https且不包含账号密码')
    return url


def video_candidate(url):
    parsed=urlparse(url); host=(parsed.hostname or '').lower(); path=parsed.path.lower()
    return (Path(path).suffix in ('.mp4','.webm','.mov','.avi','.mkv')
        or (host in ('youtube.com','www.youtube.com','m.youtube.com') and (path=='/watch' or path.startswith('/shorts/')))
        or host=='youtu.be' or (host.endswith('bilibili.com') and path.startswith('/video/'))
        or host.endswith('vimeo.com') or host.endswith('pexels.com') and '/video' in path
        or host.endswith('pixabay.com') and '/videos/' in path)


def search(action,limit=5,tavily_key=None):
    if not isinstance(action,str) or not action.strip() or len(action)>300: raise ValueError('动作描述须为1~300字符')
    if not 1<=limit<=10: raise ValueError('limit:1~10')
    query=action.strip()+' single person full body movement demonstration video'
    from harness_search import search_web
    data=search_web(query,max_results=limit,api_key=tavily_key)
    rows=[{'title':x.get('title',''),'url':x.get('url',''),'description':x.get('content','')[:500]} for x in data.get('results',[])]
    errors=[]; provider='harness_search_web'
    unique=[]; seen=set()
    for row in rows:
        try: valid_url(row['url'])
        except ValueError: continue
        if not video_candidate(row['url']): continue
        if row['url'] not in seen:
            seen.add(row['url']); unique.append({**row,'provider':provider,'semantic_match_verified':False})
    return {'action':action,'query':query,'provider':provider,'candidates':unique[:limit],'search_errors':errors,
            'message':'标题相关性不代表动作已核实；需下载解码及全身骨架验证'}


def download(url,directory,stopped=lambda:False):
    url=valid_url(url); directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    def check():
        if stopped(): raise InterruptedError('任务已取消')
        if time.monotonic()-start>180: raise TimeoutError('单候选下载超过180秒')
    # 直链流式下载，设置体积与时间预算。
    suffix=Path(urlparse(url).path).suffix.lower()
    if suffix in ('.mp4','.webm','.mov','.avi','.mkv'):
        path=directory/('video'+suffix)
        with requests.get(url,stream=True,timeout=(15,20),headers={'User-Agent':'G1VideoLearning/1.0'}) as response:
            response.raise_for_status()
            if int(response.headers.get('Content-Length','0') or 0)>MAX_BYTES: raise ValueError('视频超过200MB下载预算')
            total=0
            with path.open('wb') as output:
                for block in response.iter_content(512*1024):
                    check(); total+=len(block)
                    if total>MAX_BYTES: raise ValueError('视频超过200MB下载预算')
                    output.write(block)
        info={'source_url':url,'download_method':'direct','download_bytes':total}
    else:
        import yt_dlp
        def progress(data):
            check()
            if data.get('downloaded_bytes',0)>MAX_BYTES: raise ValueError('视频超过200MB下载预算')
        options={'outtmpl':str(directory/'video.%(ext)s'),'format':'best[height<=720][ext=mp4]/best[height<=720]',
            'noplaylist':True,'max_filesize':MAX_BYTES,'socket_timeout':20,'retries':1,'fragment_retries':1,
            'quiet':True,'no_warnings':True,'progress_hooks':[progress],'restrictfilenames':True,
            'match_filter':lambda meta,**kw: '视频超过10分钟，跳过长视频' if (meta.get('duration') or 0)>600 else None}
        with yt_dlp.YoutubeDL(options) as downloader:
            meta=downloader.extract_info(url,download=True)
            if not meta: raise ValueError('视频不可下载或被筛选')
            path=Path(downloader.prepare_filename(meta))
        info={'source_url':url,'download_method':'yt_dlp','title':meta.get('title'),'license':meta.get('license'),
              'uploader':meta.get('uploader'),'webpage_url':meta.get('webpage_url'),'duration':meta.get('duration')}
    check()
    if not path.is_file() or path.stat().st_size==0: raise ValueError('视频下载为空')
    info.update(local_video=str(path.resolve()),bytes=path.stat().st_size,semantic_match_verified=False)
    (directory/'source.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    return path,info


def inspect_video(path):
    import cv2
    cap=cv2.VideoCapture(str(path))
    try:
        fps=cap.get(cv2.CAP_PROP_FPS); count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ok,frame=cap.read()
        if not cap.isOpened() or not ok or not 0<fps<=240 or count<2: raise ValueError('下载内容无法解码为有效视频')
        return {'fps':fps,'duration_seconds':count/fps,'width':frame.shape[1],'height':frame.shape[0]}
    finally: cap.release()
