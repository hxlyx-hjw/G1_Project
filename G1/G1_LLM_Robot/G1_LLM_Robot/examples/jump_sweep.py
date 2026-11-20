from dataclasses import replace
import coordination.jump as j
from coordination.config import JumpConfig

base=JumpConfig()
for knee in [.5,.7,.85]:
    for duration in [.08,.12,.20]:
        cfg=replace(base,crouch_hip=-knee/2,crouch_knee=knee,crouch_ankle=-knee/2,thrust_seconds=duration)
        j.JumpConfig=lambda:cfg
        r=j.run(f'runs/jump/sweep_{knee}_{duration}',coordinator='runs/coordination/com_graph.pt')
        print(knee,duration,r['success'],r['fallen'],r['airborne_seconds'],r['peak_com_rise_m'],flush=True)
