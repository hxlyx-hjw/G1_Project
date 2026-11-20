import json,time
from robot_tools import RobotToolkit

t=RobotToolkit(None)
r=json.loads(t.dispatch('robot_export_gif',{'checkpoint':'runs/self_test1/training/policy.pt','max_seconds':1,'fps':10}))
print(r,flush=True)
for _ in range(100):
    s=json.loads(t.dispatch('robot_gif_status',{'export_id':r['export_id']}))
    if s['state'] in ('completed','failed'):
        print(s,flush=True)
        raise SystemExit(0 if s['state']=='completed' else 1)
    time.sleep(.2)
raise SystemExit('Export timeout')
