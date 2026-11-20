from fullbody.balanced import BalancedEnv
import numpy as np
import torch
from fullbody.ppo import ActorCritic

e=BalancedEnv('motions/self_test1/reference.npz','runs/self_test1/sweep_model')
p=ActorCritic.from_checkpoint(torch.load('runs/self_test1/training/policy.pt',weights_only=True),eval_mode=True)
for weight in [.3,.5,.7,1.]:
    e.video_leg_weight=weight
    o,_=e.reset(options={'phase':0})
    errors=[]
    for i in range(334):
        with torch.no_grad(): a=p.mean(torch.as_tensor(o).unsqueeze(0))[0].numpy()
        o,r,f,t,info=e.step(a); errors.append(info['joint_mse'])
        if f or t: break
    print(weight,i+1,f,np.mean(errors),flush=True)
