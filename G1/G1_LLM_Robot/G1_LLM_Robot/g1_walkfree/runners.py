"""两种非图形界面的运行方式。

* :func:`run_headless`  纯计算，打印状态曲线（做回归测试、批量扫参用）
* :func:`run_view`      打开 MuJoCo 3D 窗口，一直显示到你关窗口

两者都只依赖 :class:`~g1_walkfree.controller.WalkController`，
不涉及 tkinter，方便在无显示环境（服务器/CI）里跑。
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from viewer_camera import ISOMETRIC_DISTANCE, apply_isometric_camera
from . import config as C
from .controller import WalkController


# ======================================================================
def run_headless(ctrl: WalkController,
                 secs: float = 10.0,
                 log_every: float = 1.0) -> dict:
    """不开窗口跑 ``secs`` 秒，按 ``log_every`` 秒打印一行状态。

    返回最后一次 :meth:`WalkController.status` 的结果。
    """
    n_steps = int(secs / ctrl.model.opt.timestep)
    next_log = 0.0

    for k in range(n_steps):
        ctrl.step()
        t = k * ctrl.model.opt.timestep

        if t >= next_log:
            st = ctrl.status()
            print("t=%5.2f  位置=(%+.3f,%+.3f)  z=%.3f  速度=(%+.3f,%+.3f)  "
                  "朝向=%+6.1f°  %s"
                  % (t, st["x"], st["y"], st["z"], st["vx"], st["vy"], st["yaw"],
                     "⚠ 已摔倒" if st["fallen"] else ""))
            next_log += log_every

        if st_fallen(ctrl):
            print("*** 摔倒了 t=%.2f" % t)
            break

    return ctrl.status()


def st_fallen(ctrl: WalkController) -> bool:
    """轻量摔倒判定（比 status() 便宜，避免每步都算一遍三角函数）。"""
    d = ctrl.data
    if d.qpos[2] < C.FALLEN_HEIGHT:
        return True
    return float(d.xmat[1].reshape(3, 3)[2, 2]) < C.FALLEN_UPRIGHT


# ======================================================================
def run_view(ctrl: WalkController,
             secs: float = 0.0,
             follow: bool = True,
             cam_distance: float = ISOMETRIC_DISTANCE) -> None:
    """打开 3D 查看器并按真实时间推进仿真，直到用户关闭窗口。

    参数
        secs           >0 时跑够这么多仿真秒后停止推进（窗口仍保持打开）
        follow         True = 视角跟随骨盆；False = 自由视角
        cam_distance   相机距离
    """
    from viewer_camera import passive_viewer

    model, data = ctrl.model, ctrl.data

    handle = passive_viewer(model, data)
    if handle is None:
        return
    with handle as viewer:
        # ---- 初始相机 ----
        cam = viewer.cam
        apply_isometric_camera(
            cam, distance=cam_distance, lookat=(0.0, 0.0, 0.6),
            tracking_body=ctrl.pelvis_id if follow else None)
        viewer.sync()

        print("3D 窗口已打开（鼠标可旋转/平移/缩放），关闭窗口退出。", flush=True)

        finished = False
        step = 0
        while viewer.is_running():
            t0 = time.time()

            if not finished:
                ctrl.step()
                if secs > 0 and data.time >= secs:
                    print("已达到设定时长 %.1fs；窗口保持打开，关掉即可退出。" % secs)
                    finished = True

            step += 1
            if step % 5 == 0:           # 不必每步都渲染，省 CPU
                viewer.sync()

            if finished:
                time.sleep(0.01)
                continue

            # ---- 真实时间节拍 ----
            leftover = model.opt.timestep - (time.time() - t0)
            if leftover > 0:
                time.sleep(leftover)

    print("3D 窗口已关闭。")
