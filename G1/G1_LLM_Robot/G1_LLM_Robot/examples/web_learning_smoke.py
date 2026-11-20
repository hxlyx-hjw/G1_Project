"""自主学习后台链路实测；用本地HTTP提供官方示例验证下载和训练。"""
from functools import partial
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
import threading,time,json
from robot_tools import RobotToolkit


def main():
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(SimpleHTTPRequestHandler,directory=str(Path('vedio').resolve())))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    toolkit=RobotToolkit(None)
    reply=toolkit.dispatch('robot_learn_action',{'action':'官方姿态示例链路验证（未核实动作语义）',
        'rl_pd_influence':.7,'hand_rl_pd_range':[.2,.6],'foot_rl_pd_range':[.5,.9],
        'source_urls':[f'http://127.0.0.1:{server.server_port}/example.mp4'],'iterations':1,'num_envs':1,
        'max_seconds':2,'fps':10,'device':'cuda','render':False})
    result=json.loads(reply); print(json.dumps(result),flush=True)
    Path('outputs/web_learning_smoke_job.txt').write_text(result['job_id'])
    try:
        for _ in range(110):
            state=toolkit.training.status(result['job_id'])
            if state['state'] in ('completed','cancelled','failed'):
                print(json.dumps(state,ensure_ascii=True),flush=True)
                return 0 if state['state']=='completed' and state.get('skill') else 1
            time.sleep(1)
        toolkit.close(); return 2
    finally: server.shutdown()


if __name__=='__main__': raise SystemExit(main())
