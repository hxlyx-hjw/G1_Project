"""真实工具链短任务验证，不调用付费大模型。"""
import json
import time
from pathlib import Path
from robot_tools import RobotToolkit


def main():
    toolkit=RobotToolkit(None)
    result=json.loads(toolkit.dispatch('robot_train_video',{'video':'vedio/example.mp4','iterations':1,'num_envs':1,'device':'cuda','render':True,'max_seconds':2,'fps':10,'rl_pd_influence':.7,'hand_rl_pd_range':[.2,.6],'foot_rl_pd_range':[.5,.9]}))
    print(json.dumps(result,ensure_ascii=False),flush=True)
    job=result['job_id']
    Path('outputs/training_tool_smoke_job.txt').write_text(job)
    deadline=time.monotonic()+100
    while time.monotonic()<deadline:
        status=json.loads(toolkit.dispatch('robot_training_status',{'job_id':job}))
        if status.get('state') in ('completed','failed','cancelled'):
            print(json.dumps(status,ensure_ascii=False),flush=True)
            # 等待渲染进程观察到最终权重。
            for _ in range(20):
                time.sleep(.25)
                status=toolkit.training.status(job)
                if status.get('viewer',{}).get('state')=='rendering': break
            print('VIEWER',status.get('viewer'),flush=True)
            return 0 if status['state']=='completed' and status.get('viewer',{}).get('state')=='rendering' else 1
        time.sleep(.3)
    print('Timeout',status,flush=True)
    toolkit.close()
    return 2


if __name__=='__main__': raise SystemExit(main())
