"""足部合理支撑范围凸包与质心包含裕度；用于腾空阶段的平衡约束。"""
import numpy as np
import mujoco


def convex_hull(points):
    """二维凸包（Andrew monotone chain），返回逆时针顶点。"""
    values=np.asarray(points,float)
    if values.size==0: return np.zeros((0,2),float)
    values=np.unique(values.reshape(-1,2),axis=0)
    if len(values)<=1: return values
    values=sorted(values.tolist())

    def cross(o,a,b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])

    lower=[]
    for point in values:
        while len(lower)>=2 and cross(lower[-2],lower[-1],point)<=0: lower.pop()
        lower.append(point)
    upper=[]
    for point in reversed(values):
        while len(upper)>=2 and cross(upper[-2],upper[-1],point)<=0: upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1]+upper[:-1],float)


def _segment_distance(point,a,b):
    edge=b-a; length=float(edge@edge)
    if length<=1e-16: return float(np.linalg.norm(point-a))
    t=float(np.clip((point-a)@edge/length,0.,1.))
    return float(np.linalg.norm(point-(a+t*edge)))


def signed_distance(point,polygon):
    """凸多边形内为正（到最近边的距离），外为负（到边界的最短距离）。"""
    polygon=np.asarray(polygon,float)
    if len(polygon)<3: return 0.
    point=np.asarray(point,float).reshape(2)
    inside=True; nearest=float('inf')
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        edge=b-a; rel=point-a
        distance=float((edge[0]*rel[1]-edge[1]*rel[0])/max(np.linalg.norm(edge),1e-12))
        nearest=min(nearest,distance)
        if distance< -1e-9: inside=False
    if inside: return max(0.,nearest)
    return -min(_segment_distance(point,a,b) for a,b in zip(polygon,np.roll(polygon,-1,axis=0)))


def foot_support_points(model,data,body_ids,margin=0.,velocity_xy=None,horizon=0.):
    """把足部球体投影到水平面，并按支撑半径/落地预测扩张成点集。"""
    points=[]
    bodies=set(int(body) for body in body_ids)
    velocity=np.zeros(2) if velocity_xy is None else np.asarray(velocity_xy,float)[:2]
    shift=velocity*max(0.,float(horizon))
    directions=np.array([[np.cos(a),np.sin(a)] for a in np.linspace(0,2*np.pi,8,endpoint=False)])
    for body in bodies:
        spheres=[g for g in range(model.ngeom) if int(model.geom_bodyid[g])==body
                 and model.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE]
        if spheres:
            for geom in spheres:
                center=data.geom_xpos[geom,:2]+shift
                radius=float(model.geom_size[geom,0])+max(0.,float(margin))
                points.extend(center+radius*directions)
        else:
            center=data.xpos[body,:2]+shift
            radius=max(.06,float(margin))
            points.extend(center+radius*directions)
    return np.asarray(points,float)


def support_margin(model,data,body_ids,com_xy,margin=0.,velocity_xy=None,horizon=0.):
    """返回质心相对足部支撑凸包的有符号裕度及凸包顶点。"""
    points=foot_support_points(model,data,body_ids,margin,velocity_xy,horizon)
    hull=convex_hull(points)
    if len(hull)<3: return 0.,hull
    return signed_distance(com_xy,hull),hull
