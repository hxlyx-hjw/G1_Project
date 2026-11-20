"""独立协调/跳跃进程接口，不改变聊天机器人MjData。"""
from pathlib import Path
import subprocess,sys,json,uuid

ROOT=Path(__file__).resolve().parents[1]
try:
    from data_paths import RUNS as DATA_RUNS
except Exception:
    DATA_RUNS = ROOT / 'runs'


class CoordinationJobs:
    def __init__(self,session=None): self.tasks={}; self.session=session; self.pending=None

    def start(self,operation='jump',viewer=True,record=True,device='auto'):
        if operation not in ('train','jump') or device not in ('auto','cpu','cuda'): raise ValueError('操作或设备非法')
        if any(x['process'].poll() is None and not (x['operation']=='jump' and x['result'].is_file()) for x in self.tasks.values()): raise ValueError('已有协调/跳跃任务运行中')
        if self.pending: self.handoff(self.pending)
        task_id='coord_'+uuid.uuid4().hex[:10]; out=DATA_RUNS/'coordination_tasks'/task_id; out.mkdir(parents=True)
        command=[sys.executable,'-u','-m','coordination',operation]
        if operation=='train':
            target=DATA_RUNS/'coordination/com_graph.pt'; command+=['--output',str(target),'--device',device]
        else:
            target=out/'report.json'; checkpoint=DATA_RUNS/'coordination/com_graph.pt'
            if not checkpoint.is_file(): raise FileNotFoundError('请先训练协调网络：python -m coordination train')
            command+=['--output',str(out),'--checkpoint',str(checkpoint)]
            if self.session is not None:
                import numpy as np
                with self.session._lock:
                    if self.session._active: raise RuntimeError('请等待当前动作完成')
                    if self.session.ctrl.status()['fallen']: raise ValueError('行走机器人已摔倒，请先复位')
                    q=self.session.ctrl.data.qpos
                    yaw=float(np.arctan2(2*(q[3]*q[6]+q[4]*q[5]),1-2*(q[5]**2+q[6]**2)))
                    command+=['--start-pose',str(q[0]),str(q[1]),str(q[2]),str(yaw)]
            if viewer: command+=['--viewer']
            if record: command+=['--gif']
        with (out/'worker.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        self.tasks[task_id]={'process':proc,'folder':out,'result':target,'operation':operation}
        if operation=='jump': self.pending=task_id
        return {'task_id':task_id,'state':'running','operation':operation,'output_dir':str(out),'isolated_simulation':True}

    def status(self,task_id):
        task=self.tasks.get(task_id)
        if not task: raise ValueError('未知协调任务')
        code=task['process'].poll()
        report=None
        if task['operation']=='jump' and task['result'].is_file(): report=json.loads(task['result'].read_text(encoding='utf-8'))
        if report and report.get('handoff_ready') and self.session is not None and not task.get('handed_off'):
            self.handoff(task_id)
        return {'task_id':task_id,'state':('handed_off' if task.get('handed_off') else 'ready_for_handoff') if report and report.get('success') else 'failed' if report else ('running' if code is None else 'completed' if code==0 else 'failed'),
            'window_process_running':code is None,'handed_off':task.get('handed_off',False),
            'exit_code':code,'report':report,'output_dir':str(task['folder']),'checkpoint':str(task['result']) if task['operation']=='train' else None}


    def handoff(self,task_id):
        task=self.tasks.get(task_id)
        if not task or task['operation']!='jump': raise ValueError('未知跳跃任务')
        if self.session is None: raise RuntimeError('没有行走会话')
        if task.get('handed_off'): return {'imported':True,'already_imported':True,'task_id':task_id}
        if not task['result'].is_file(): raise RuntimeError('跳跃尚未稳定，请先查询状态；后续动作暂不执行')
        report=json.loads(task['result'].read_text(encoding='utf-8'))
        if not report.get('success') or not report.get('handoff_ready'): raise ValueError('跳跃失稳，不能导入行走仿真；请关闭窗口并复位')
        from .handoff import import_state
        result=import_state(self.session,task['folder']/'handoff.npz')
        task['handed_off']=True
        if self.pending==task_id: self.pending=None
        return {'task_id':task_id,**result,'jump_window_preserved':True}

    def before_action(self):
        if self.pending: return self.handoff(self.pending)

    def close_window(self,task_id):
        task=self.tasks.get(task_id)
        if not task: raise ValueError('未知协调任务')
        (task['folder']/'CLOSE_VIEWER').touch()
        return {'task_id':task_id,'close_requested':True}


def schemas():
    def tool(name,description,properties,required=()):
        return {'type':'function','function':{'name':name,'description':description,'parameters':{'type':'object','properties':properties,'required':list(required)}}}
    return [tool('robot_jump_handoff','将已稳定跳跃的完整状态导入当前行走仿真，跳跃窗口保留；后续动作会自动执行此交接。',{'task_id':{'type':'string'}},['task_id']),
        tool('robot_jump_close_window','显式关闭跳跃快照窗口，不重置已交接的行走机器人。',{'task_id':{'type':'string'}},['task_id']),
        tool('robot_coordination_start','独立进程训练关键节点-质心图网络或执行保守小跳。从当前行走位置和朝向开始，稳定落地后后续动作自动交接完整状态，窗口保持快照。跳跃成功要求连续腾空、落地后稳定保持；不可用于实机。',{
        'operation':{'type':'string','enum':['train','jump']},'viewer':{'type':'boolean'},'record':{'type':'boolean'},'device':{'type':'string','enum':['auto','cpu','cuda']}}),
        tool('robot_coordination_status','查询协调训练或跳跃结果，必须读取success/fallen，不得将进程结束当作跳跃成功。',{'task_id':{'type':'string'}},['task_id'])]
