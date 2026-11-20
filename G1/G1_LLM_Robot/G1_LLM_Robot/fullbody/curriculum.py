"""仅在完整片段物理评估达标时增加腿部参考权重。"""
from dataclasses import dataclass


@dataclass
class LegCurriculum:
    weight: float=.15
    target: float=1.
    increment: float=.05
    required_passes: int=3
    passes: int=0
    max_root_xy_error: float=.15
    max_root_speed: float=1.0
    max_joint_mse: float=.05

    def update(self,report):
        ok=(report['completed'] and not report['fallen'] and report['max_root_xy_error_m']<=self.max_root_xy_error
            and report['max_root_speed_mps']<=self.max_root_speed and report['joint_mse']<=self.max_joint_mse)
        self.passes=self.passes+1 if ok else 0
        promoted=False
        if self.passes>=self.required_passes and self.weight<self.target:
            self.weight=min(self.target,self.weight+self.increment)
            self.passes=0; promoted=True
        return {'passed':ok,'promoted':promoted,'weight':self.weight,'target':self.target,'consecutive_passes':self.passes}
