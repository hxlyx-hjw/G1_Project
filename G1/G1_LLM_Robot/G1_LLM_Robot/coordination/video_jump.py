"""供视频训练调用的接触驱动保守跳跃模板。"""
import numpy as np
from .config import JumpConfig


class VideoJumpController:
    def __init__(self,env,times):
        self.env=env; self.times=times; self.cfg=JumpConfig(); self.reset()

    def reset(self):
        self.event=None; self.liftoff=None; self.landing=None; self.airborne=0.; self.max_airborne=0.; self.phase='tracking'
        self.events=[]; self.last_time=None

    def target(self,base):
        e=self.env; now=e.phase
        events=[i for i,t in enumerate(self.times) if t-.65<=now<t+1.8]
        if not events:
            self.phase='tracking'; return base
        index=events[0]; scheduled=self.times[index]
        if self.event!=index:
            self.event=index; self.liftoff=None; self.landing=None; self.airborne=0.; self.max_airborne=0.
        foot=set(e.key_bodies[2:]); contacts=any(int(e.model.geom_bodyid[c.geom1]) in foot or int(e.model.geom_bodyid[c.geom2]) in foot for c in e.data.contact)
        delta=.02 if self.last_time is None else max(0,min(.02,now-self.last_time)); self.last_time=now
        if now>=scheduled and not contacts:
            if self.liftoff is None: self.liftoff=now
            self.airborne+=delta; self.max_airborne=max(self.max_airborne,self.airborne)
        elif contacts:
            if self.max_airborne>=.04 and self.landing is None and self.liftoff is not None:
                self.landing=now; self.events.append({'event':index,'scheduled_time':scheduled,'liftoff':self.liftoff,'landing':now,'airborne_seconds':self.max_airborne})
            self.airborne=0.
        if now<scheduled:
            self.phase='crouch'; alpha=np.clip((now-(scheduled-.65))/.65,0,1); alpha=alpha*alpha*(3-2*alpha)
            knee=.3+alpha*(self.cfg.crouch_knee-.3)
        elif self.landing is not None:
            self.phase='landing'; return base
        elif self.liftoff is not None:
            self.phase='flight'; knee=.54
        elif now<scheduled+self.cfg.thrust_seconds:
            self.phase='thrust'; knee=self.cfg.thrust_knee
        else:
            self.phase='recover'; return base
        target=base.copy()
        for side in ('left','right'):
            for joint,value in [('hip_pitch',-knee/2),('knee',knee),('ankle_pitch',-knee/2)]:
                target[e.names.index(side+'_'+joint+'_joint')]=value
        return target

    def report(self):
        return {'phase':self.phase,'events':self.events.copy(),'jump_verified':bool(self.events),
                'all_scheduled_jumps_landed':len(self.events)==len(self.times) and bool(self.times)}
