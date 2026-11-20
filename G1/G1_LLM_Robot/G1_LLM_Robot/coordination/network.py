"""以全身质心为中心的图网络，输出受限29关节修正；教师为质心Jacobian控制。"""
import numpy as np
import mujoco
import torch
from torch import nn
from .config import KEY_BODIES
from fullbody.model import joint_info


class CoordinatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.node=nn.Sequential(nn.Linear(6,48),nn.Tanh(),nn.Linear(48,48),nn.Tanh())
        self.com_message=nn.Sequential(nn.Linear(48+6,64),nn.Tanh(),nn.Linear(64,29))

    def forward(self,nodes,com):
        # 各节点相对COM的位置/速度经共享编码，聚合连接到COM中心节点。
        encoded=self.node(nodes).mean(dim=-2)
        return .08*torch.tanh(self.com_message(torch.cat([encoded,com],dim=-1)))


class Features:
    def __init__(self,model):
        self.model=model
        self.ids=[model.body(n).id for n in KEY_BODIES]
        self.names,self.qa,self.va=joint_info(model)
        self.pelvis=model.body('pelvis').id
        self.feet=[model.body(s+'_ankle_roll_link').id for s in ('left','right')]
        self.last_com=None

    def read(self,data):
        mujoco.mj_forward(self.model,data)
        com=data.subtree_com[self.pelvis].copy()
        jac=np.zeros((3,self.model.nv)); mujoco.mj_jacSubtreeCom(self.model,data,jac,self.pelvis)
        velocity=jac@data.qvel
        node_features=[]
        for body in self.ids:
            jp=np.zeros((3,self.model.nv)); jr=np.zeros_like(jp)
            mujoco.mj_jacBody(self.model,data,jp,jr,body)
            node_features.append(np.r_[data.xpos[body]-com,jp@data.qvel-velocity])
        support=np.mean(data.xpos[self.feet,:2],axis=0)
        central=np.r_[com[:2]-support,velocity[:2],data.qvel[3:5]]
        return np.array(node_features,np.float32),central.astype(np.float32),jac,com,support

    def teacher(self,data):
        nodes,central,jac,com,support=self.read(data)
        desired=np.clip(-.6*(com[:2]-support)-.10*central[2:4],-.04,.04)
        j=jac[:2,self.va]
        correction=j.T@np.linalg.solve(j@j.T+.015*np.eye(2),desired)
        return nodes,central,np.clip(correction,-.08,.08).astype(np.float32)


def train(model,output,samples=2000,epochs=100,device='cpu',seed=7):
    from fullbody.model import nominal
    from runtime_io import save_checkpoint
    torch.set_num_threads(2); torch.manual_seed(seed)
    rng=np.random.default_rng(seed); data=mujoco.MjData(model); feat=Features(model)
    default=nominal(model); ids=model.actuator_trnid[:,0].astype(int); bounds=model.jnt_range[ids]
    xs=[]; cs=[]; ys=[]
    for _ in range(samples):
        data.qpos[:]=default
        data.qpos[feat.qa]=np.clip(default[feat.qa]+rng.normal(0,.13,29),*bounds.T)
        data.qvel[:]=rng.normal(0,.15,model.nv)
        nodes,central,target=feat.teacher(data); xs.append(nodes); cs.append(central); ys.append(target)
    x=torch.tensor(np.array(xs),device=device); c=torch.tensor(np.array(cs),device=device); y=torch.tensor(np.array(ys),device=device)
    split=int(samples*.8); net=CoordinatorNet().to(device); opt=torch.optim.Adam(net.parameters(),lr=.002)
    for _ in range(epochs):
        for batch in torch.randperm(split,device=device).split(256):
            loss=(net(x[batch],c[batch])-y[batch]).square().mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): mse=float((net(x[split:],c[split:])-y[split:]).square().mean().cpu())
    save_checkpoint({'model':net.cpu().state_dict(),'key_bodies':list(KEY_BODIES),'joint_names':feat.names,
        'type':'com_star_graph_v1','heldout_mse':mse,'seed':seed,'teacher':'subtree_com_jacobian'},output)
    return {'samples':samples,'epochs':epochs,'heldout_mse':mse,'device':device,'output':str(output)}


class Coordinator:
    def __init__(self,model,path):
        from runtime_io import load_checkpoint
        state=load_checkpoint(path,map_location='cpu',weights_only=True)
        self.features=Features(model)
        if state['joint_names']!=self.features.names or state['key_bodies']!=list(KEY_BODIES): raise ValueError('协调网络模型/关节不匹配')
        self.net=CoordinatorNet(); self.net.load_state_dict(state['model']); self.net.eval()

    def correction(self,data):
        nodes,com,*_=self.features.read(data)
        with torch.inference_mode(): return self.net(torch.from_numpy(nodes)[None],torch.from_numpy(com)[None])[0].numpy()
