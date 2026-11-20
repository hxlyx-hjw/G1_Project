"""为用户录屏提取最长连续全身段，保留来源时间。"""
from pathlib import Path
import json
import numpy as np


def trim(source,output):
    with np.load(source) as z:
        times,points,visibility=[z[k].copy() for k in ('times','points','visibility')]
    # Multiview DLT can legitimately retain a limb at 0.4-0.5 confidence.
    # The following 3D repair and retarget refinement handle these frames.
    valid=np.min(visibility[:,[11,12,23,24,25,26,27,28]],axis=1)>=.4
    indices=np.flatnonzero(valid)
    if not len(indices): raise ValueError('视频中没有肩髋膝踝同时可见的全身动作')
    groups=np.split(indices,np.where(np.diff(indices)>1)[0]+1)
    group=max(groups,key=len)
    if len(group)<10: raise ValueError('没有足够连续的全身动作')
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,times=times[group]-times[group[0]],points=points[group],visibility=visibility[group],source_times=times[group])
    report={'source':str(source),'start_seconds':float(times[group[0]]),'end_seconds':float(times[group[-1]]),'frames':len(group),'excluded':'近景讲解、镜头切换和腿部不可见区间'}
    output.with_suffix('.trim.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(report)


if __name__=='__main__':
    trim('motions/self_test1/pose_full.npz','motions/self_test1/pose.npz')
