"""只读环境诊断：不输出密钥，不修改训练和技能数据。"""
from pathlib import Path
import importlib.util
import os
import sys

ROOT=Path(__file__).resolve().parent


def check():
    results=[]
    def add(name,ok,message): results.append({'name':name,'ok':bool(ok),'message':str(message)})
    add('python',sys.version.split()[0],sys.executable)
    for module in ('numpy','scipy','cv2','mediapipe','mujoco','torch','smplx','gymnasium','openai','tavily','yt_dlp'):
        spec=importlib.util.find_spec(module)
        add('module:'+module,spec is not None,'installed' if spec else 'missing')
    try:
        import torch
        add('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')
    except Exception as exc: add('cuda',False,type(exc).__name__+': '+str(exc))
    files={
        'Start.py':ROOT/'Start.py','G1Robot.pyproj':ROOT/'G1Robot.pyproj',
        'pose_model':ROOT/'video_motion/assets/pose_landmarker_full.task',
        'smplx_model':Path(r'E:\TEST_RECORD\G1_Project\G1\models\SMPL\models_smplx_v1_1\models\smplx\SMPLX_NEUTRAL.npz'),
        'menagerie':Path(r'E:\TEST_RECORD\G1_Project\G1\assets\mujoco_menagerie\unitree_g1\g1.xml'),
        'policy':Path(r'E:\TEST_RECORD\G1_Project\G1\assets\motion.pt'),
        'deepseek_key':ROOT/'api_key.txt','tavily_key':ROOT/'tavily_api_key.txt'}
    for name,path in files.items():
        add('file:'+name,path.is_file(),str(path))
    for path in (ROOT,ROOT/'runs',ROOT/'skills',ROOT/'outputs'):
        try:
            path.mkdir(parents=True,exist_ok=True)
            probe=path/'.g1_write_probe.tmp'; probe.write_text('ok',encoding='utf-8'); probe.unlink()
            add('write:'+str(path.relative_to(ROOT) or '.'),True,'writable')
        except Exception as exc: add('write:'+str(path),False,type(exc).__name__+': '+str(exc))
    for name,path in (('deepseek_key',files['deepseek_key']),('tavily_key',files['tavily_key'])):
        try: chars=len(path.read_text(encoding='utf-8-sig').strip()) if path.is_file() else 0
        except Exception: chars=0
        add('key:'+name,chars>0,f'{chars} characters; value hidden')
    return results


def render():
    rows=check(); print('G1 环境诊断')
    failed=0
    for row in rows:
        print(('[OK ]' if row['ok'] else '[BAD]')+' '+row['name']+': '+row['message'])
        failed+=not row['ok']
    print('结果：'+('通过' if not failed else f'{failed}项需要处理'))
    return 0 if not failed else 1


if __name__=='__main__':
    for _stream in (sys.stdout,sys.stderr):
        if hasattr(_stream,'reconfigure'):
            _stream.reconfigure(encoding='utf-8',errors='replace')
    raise SystemExit(render())
