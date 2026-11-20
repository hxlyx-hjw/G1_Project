"""G1 机器人会话桥接层：把"跑的仿真"封装成一个可以被外部随时调用的机器人。

为什么要单独一层
----------------
大模型给出的指令是**异步**到达的（一问一答），而 MuJoCo 仿真需要**连续**推进一步。
所以这里用一条后台线程持续跑仿真，对外只暴露线程安全的"动作 API"：

    大模型 --Function Calling--> robot_tools --普通方法--> RobotSession
                                                            |
                                                     后台线程 mj_step
                                                            |
                                                        MuJoCo 仿真

两种节奏
--------
* ``realtime=False``（默认）：仿真尽可能快跑。大模型说"走 3 秒"，
  实际可能只花 0.5 秒墙上时间，交互体验好。**时长一律指仿真秒数。**
* ``realtime=True``：按真实时间跑，方便开 3D 窗口肉眼看。

安全措施
--------
* 速度指令统一限幅（``SAFE_MAX_*``），不信任模型给的任何数值
* 每次动作后检查是否摔倒，摔倒立即停并如实上报
* ``goto`` / ``walk_for`` 都有**仿真时间超时**和**墙上时间兜底超时**，不会卡死
* 全程用一把可重入锁保护 ``MjData``，避免边步进边读状态导致数据撕裂
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
import threading
import time
from typing import Optional


# ======================================================================
# 控制台编码兜底
# ======================================================================
def ensure_console_safe() -> None:
    """让 print 不会因为特殊字符（⚠ 等）在 GBK 控制台上直接崩掉。

    Windows 控制台默认是 GBK/CP936，打印 emoji 或生僻符号会抛
    ``UnicodeEncodeError``。这里只做"遇到编不出来的字符就替换"，
    不改编码本身，避免中文变成乱码。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


ensure_console_safe()          # 导入本模块时立即生效，任何入口都能受益


# ======================================================================
# 安全限幅（不信任大模型给的数值，一律夹到安全区间）
# ======================================================================
SAFE_MAX_VX = 0.80      # 前进/后退 (m/s)
SAFE_MAX_VY = 0.45      # 左右平移 (m/s)
SAFE_MAX_WZ = 1.00      # 转向 (rad/s)
MAX_ACTION_SECONDS = 60.0        # 单条指令最长仿真秒数，防止"走 1 小时"
GOTO_TOL = 0.12                  # goto 位置容差 (m)
YAW_TOL = math.radians(12.0)     # goto 朝向容差 (rad)


class RobotSessionError(RuntimeError):
    """机器人层出错（没找到工程、依赖缺失、动作参数非法等）。"""


# ======================================================================
# 定位 g1_walkfree 包（上一个交付里的控制器）
# ======================================================================
def _ensure_walker_importable(walker_dir: Optional[str] = None) -> str:
    """确保 ``g1_walkfree`` 能被 import，返回实际使用的目录。

    查找顺序：显式参数 → 已可导入 → 脚本目录及其父目录 → 用户主目录。
    """
    if walker_dir:
        p = os.path.abspath(walker_dir)
        if not os.path.isdir(os.path.join(p, "g1_walkfree")):
            raise RobotSessionError(
                "在 %s 里找不到 g1_walkfree 包（应包含 g1_walkfree/__init__.py）" % p)
        sys.path.insert(0, p)
        return p

    try:
        import g1_walkfree  # noqa: F401
        return os.path.dirname(os.path.abspath(g1_walkfree.__file__))
    except ImportError:
        pass

    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.getcwd(), here, os.path.dirname(here),
                 os.path.join(here, ".."), os.path.expanduser("~")):
        cand = os.path.abspath(cand)
        if os.path.isdir(os.path.join(cand, "g1_walkfree")):
            sys.path.insert(0, cand)
            return cand

    raise RobotSessionError(
        "找不到 g1_walkfree 包。请把上一个交付里的 g1_walkfree/ 目录\n"
        "和本文件放在同一层（或父层），或者用 walker_dir 参数指定其所在目录。\n"
        "已尝试：当前目录、本文件目录、父目录、用户主目录")


# ======================================================================
# 装饰器：标记"这是一个动作"，执行期间才让后台线程推进仿真
# ======================================================================
def _needs_sim(method):
    """被装饰的方法在执行期间会激活仿真；结束后自动恢复空闲（暂停）状态。

    用装饰器而不是在每个方法里手写 ``with``，是为了保证**一个都不会漏**：
    漏掉任何一个动作方法，那个动作就会"下发了但仿真不动"，非常难查。
    """
    import functools

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._active_action():
            return method(self, *args, **kwargs)

    return wrapper


# ======================================================================
class RobotSession:
    """一个"活的"G1：后台跑仿真，对外提供同步的动作方法。

    典型用法::

        with RobotSession() as bot:
            print(bot.walk_for(0.5, 0, 0, seconds=3))   # 前进 3 仿真秒
            print(bot.goto(0.0, 0.0, yaw=0.0))          # 走回原点
            print(bot.set_arm_preset("双手上举"))
            print(bot.describe())                        # 给模型看的摘要
    """

    def __init__(self,
                 walker_dir: Optional[str] = None,
                 show_viewer: bool = False,
                 realtime: Optional[bool] = None,
                 verbose: bool = True,
                 max_speed_scale: float = 1.0,
                 record_frames: bool = False,
                 record_fps: int = 20,
                 record_size: tuple = (384, 288),
                 motion_log_path: Optional[str] = None,
                 motion_log_interval: int = 1):
        """启动机器人。

        参数
            walker_dir       g1_walkfree 包所在目录（None = 自动查找）
            show_viewer      是否打开 MuJoCo 3D 窗口（打开时默认按真实时间跑）
            realtime         None = 跟随 show_viewer；True/False 强制指定
            verbose          是否打印内部日志（资源路径、动作记录）
            max_speed_scale  安全限幅的整体缩放（0.5 = 更保守）
            record_frames    True = 边跑边离屏录制，结束时可用 save_gif() 导出
            record_fps       录制帧率
            record_size      录制分辨率 (高, 宽)
            motion_log_path  非空时每帧记录骨盆 x,y,z 到该文件（csv/json，关闭时保存）
            motion_log_interval 每隔多少仿真步记录一次（1=每帧，500Hz）
        """
        self.verbose = verbose
        self.show_viewer = show_viewer
        self.realtime = show_viewer if realtime is None else realtime
        self._speed_scale = float(max(0.1, min(2.0, max_speed_scale)))

        # 录制相关（离屏渲染，和 3D 窗口互不影响）
        # 注意：名字不要叫 _record —— 那会盖掉下面的 _record() 写日志方法
        self._recording = bool(record_frames)
        self._record_fps = int(record_fps)
        self._record_size = record_size
        self._renderer = None
        self._record_cam = None
        self._steps_per_frame = 1
        self._record_render_failed = False
        self.frames: list = []

        # 位置日志（默认关闭；motion_log_path 非空才启用，不破坏原有行为）
        self._motion_logger = None
        if motion_log_path:
            from motion_logger import MotionLogger
            self._motion_logger = MotionLogger(motion_log_path, step_interval=motion_log_interval)
            self._log("已启用位置日志: %s" % motion_log_path)
        self._motion_log_result = None

        # ---- 1) 定位并导入控制器 ----
        self.walker_dir = _ensure_walker_importable(walker_dir)
        from g1_walkfree.controller import WalkController
        self._log("g1_walkfree 目录: %s" % self.walker_dir)

        self.ctrl = WalkController(verbose=verbose)

        # ---- 2) 线程间共享状态 ----
        self._lock = threading.RLock()       # 保护 MjData 的读写
        self._viewer_lock = threading.Lock() # 串行化窗口同步与关闭，避免跨线程竞态
        self._cmd = [0.0, 0.0, 0.0]          # 当前速度指令（由动作方法设置）
        self._stop_evt = threading.Event()   # 通知后台线程退出
        self._thread: Optional[threading.Thread] = None
        self._viewer = None
        self._last_fall = False              # 上一轮是否摔倒（用于"新摔倒"上报）
        self._action_log: list[str] = []     # 动作历史（给模型/调试看）

        # 「有动作在执行」的计数器（可重入）。
        # 关键：**空闲时后台线程不推进仿真**。
        # 否则大模型两次工具调用之间（网络往返好几秒）仿真会自己跑掉几十秒，
        # 机器人会在这个空档里漂到别处，模型看到的坐标全是乱的。
        self._active = 0

        # ---- 3) 启动 ----
        if self._recording:
            self._setup_recorder()
        if show_viewer:
            self._open_viewer()
        self._thread = threading.Thread(target=self._loop, name="g1-sim", daemon=True)
        self._thread.start()
        self._warm_up()
        self._log("仿真线程已启动（realtime=%s，空闲时暂停）" % self.realtime)

    # ------------------------------------------------------------------
    # 便捷上下文管理
    # ------------------------------------------------------------------
    def __enter__(self) -> "RobotSession":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        """停掉后台线程并关闭 3D 窗口。"""
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        viewer = None
        if self._viewer_lock.acquire(timeout=2.0):
            try:
                viewer = self._viewer
                self._viewer = None
            finally:
                self._viewer_lock.release()
        if viewer is not None:
            try:
                if viewer.is_running():
                    viewer.close()
            except Exception:
                pass
        if self._motion_logger is not None:
            try:
                path = self._motion_logger.save()
                self._log("位置日志已保存: %s（%d 帧）" % (path, len(self._motion_logger)))
            except Exception as exc:
                self._log("位置日志保存失败: %s" % exc)
        self._log("机器人已关闭")

    # ==================================================================
    # 后台线程
    # ==================================================================
    def _loop(self) -> None:
        """后台循环：**只在有动作执行时**推进仿真，空闲就真睡觉。

        为什么不一直跑？
            大模型是"一问一答"的异步节奏，两次工具调用之间隔着网络往返（好几秒）。
            如果仿真一直全速跑，这几秒里仿真会推进几十秒，机器人会自己漂走，
            模型读到的坐标前后矛盾（实测过：两次读数相差好几米）。
            所以把仿真和"动作"绑定：有动作才推进，空闲就冻结在当前姿态。

        快跑模式一次推进 ``batch`` 步再让出 GIL；实时模式一次一批并 sleep 到点上。
        """
        batch = 25                       # 0.05 仿真秒（25 × 0.002）
        dt = self.ctrl.model.opt.timestep

        while not self._stop_evt.is_set():
            t0 = time.perf_counter()
            with self._lock:
                active = self._active > 0
                if active:
                    self.ctrl.set_command(*self._cmd)
                    for _ in range(batch):
                        self.ctrl.step()
                        self._maybe_capture()      # 录制（必须在锁内，避免数据撕裂）
                        self._maybe_log_position() # 位置日志（默认关闭）

                # 摔倒保护：立刻清空速度指令（避免躺着还在"走"）
                fallen_now = self.ctrl.status()["fallen"]
                if fallen_now and not self._last_fall:
                    self._cmd = [0.0, 0.0, 0.0]
                    self._log("⚠ 检测到摔倒，已自动清零速度指令")
                self._last_fall = fallen_now
                if self._viewer is not None:
                    with self._viewer_lock:
                        viewer = self._viewer
                        if viewer is not None:
                            try:
                                if viewer.is_running():
                                    viewer.sync()
                            except Exception as exc:
                                self._log("3D 窗口同步失败，已停止刷新（%s）" % exc)
                                self._viewer = None

            if not active:
                # 空闲：冻结仿真，睡一小会儿别空转 CPU
                time.sleep(0.005)
                continue

            if self.realtime:
                want = batch * dt
                left = want - (time.perf_counter() - t0)
                if left > 0:
                    time.sleep(left)
            else:
                time.sleep(0)            # 让出 GIL，保证工具调用能及时插进来

        # 渲染器创建于此线程，也必须在此线程销毁，避免退出时 GLFW/WGL
        # 在错误线程清理上下文导致进程返回非零退出码或打印驱动错误。
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    # ------------------------------------------------------------------
    # 离屏录制
    # ------------------------------------------------------------------
    def _maybe_log_position(self) -> None:
        """记录一帧骨盆世界坐标（默认关闭，必须在锁内调用）。"""
        if self._motion_logger is None:
            return
        d = self.ctrl.data
        self._motion_logger.record(d.time, d.qpos[0], d.qpos[1], d.qpos[2], d.qvel[0], d.qvel[1], d.qvel[2], d.qacc[0], d.qacc[1], d.qacc[2])

    def _finish_motion_log(self) -> None:
        """动作结束时保存日志并绘制图谱（默认关闭）。"""
        try:
            if self._motion_logger is None or len(self._motion_logger) == 0:
                return
            data_path = self._motion_logger.save()
            plot_path = self._motion_logger.plot()
            self._motion_log_result = {'data_path': data_path, 'plot_path': plot_path, 'frames': len(self._motion_logger)}
            self._log("动作日志: %s | 图谱: %s" % (data_path, plot_path))
        except Exception as exc:
            self._log("动作日志处理失败: %s" % exc)

    def set_motion_log(self, path: Optional[str] = None) -> str:
        """动态开启/关闭位置日志。path 非空开启，None 关闭。"""
        with self._lock:
            if path:
                from motion_logger import MotionLogger
                self._motion_logger = MotionLogger(path)
                self._motion_log_result = None
                self._log("已开启位置日志: %s" % path)
                return "已开启位置日志: %s" % path
            self._motion_logger = None
            return "已关闭位置日志"

    def motion_log_summary(self):
        """返回最近一次动作日志的数据/图谱路径（无则 None）。"""
        return self._motion_log_result

    def _setup_recorder(self) -> None:
        """准备离屏录制参数；渲染器在仿真线程内首次采样时创建。

        OpenGL 上下文是线程局部的。若在构造函数所在线程创建
        ``mujoco.Renderer`` 却在 ``g1-sim`` 线程调用，WGL 会报
        “Failed to make context current”，渲染结果全黑。因此这里只记录
        参数，把渲染器创建推迟到真正执行 ``render()`` 的线程。
        """
        try:
            dt = self.ctrl.model.opt.timestep
            self._steps_per_frame = max(1, int(round(1.0 / (self._record_fps * dt))))
            self._record_counter = 0
            self._log("已准备离屏录制（%dx%d @ %d fps，仿真线程内初始化）"
                      % (self._record_size[1], self._record_size[0], self._record_fps))
        except Exception as exc:
            self._recording = False
            self._log("录制初始化失败（%s），已关闭录制" % exc)

    def _ensure_renderer(self) -> bool:
        """在当前线程创建离屏渲染器；必须与 ``render()`` 同线程调用。"""
        if self._renderer is not None:
            return True
        if self._record_render_failed:
            return False
        try:
            import mujoco
            from viewer_camera import isometric_camera

            h, w = self._record_size
            self._renderer = mujoco.Renderer(self.ctrl.model, height=h, width=w)
            self._record_cam = isometric_camera(
                self.ctrl.model, lookat=(0.0, 0.0, 0.6))
            return True
        except Exception as exc:
            self._record_render_failed = True
            self._recording = False
            self._log("离屏渲染器初始化失败（%s），已关闭录制" % exc)
            return False

    def _maybe_capture(self) -> None:
        """在仿真线程里按帧率抓一帧（必须在锁内调用）。"""
        if not self._recording:
            return
        if not self._ensure_renderer():
            return
        self._record_counter += 1
        if self._record_counter % self._steps_per_frame:
            return
        try:
            self._renderer.update_scene(self.ctrl.data, camera=self._record_cam)
            # MuJoCo.Renderer.render() 返回内部复用缓冲；必须拷贝后保存，
            # 否则所有 GIF 帧会指向同一块内存，出现重复帧或花屏。
            self.frames.append(self._renderer.render().copy())
        except Exception:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None        # 渲染出错就放弃录制，别拖垮仿真
            self._recording = False

    def save_gif(self, path: str = "g1_llm_demo.gif") -> str:
        """把录到的帧存成 GIF，返回文件路径。"""
        frames = list(self.frames)
        if not frames:
            return "没有录到任何帧（构造时请设 record_frames=True）"
        import imageio
        imageio.mimsave(path, frames, fps=self._record_fps, loop=0)
        return "已保存 %s（%d 帧）" % (path, len(frames))

    def _warm_up(self, seconds: float = 0.15) -> None:
        """启动后先让仿真跑一小段，确保线程正常、机器人站稳。"""
        with self._active_action():
            self._wait_sim(seconds, wall_cap=15.0)

    @contextlib.contextmanager
    def _active_action(self):
        """标记"有动作在执行"，期间后台线程推进仿真。

        用计数器而不是布尔，因为动作会嵌套（``sequence`` 里调 ``goto``）。
        ``finally`` 保证异常时也一定会退出激活状态，不会把仿真卡在一直跑。
        """
        with self._lock:
            self._active += 1
            if self._active == 1 and self._motion_logger is not None:
                self._motion_logger.clear()
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
            if self._active == 0 and self._motion_logger is not None:
                self._finish_motion_log()

    def _open_viewer(self) -> None:
        """打开 MuJoCo 被动查看器（内部用；失败时降级为无画面）。"""
        try:
            from viewer_camera import apply_isometric_camera, passive_viewer

            v = passive_viewer(self.ctrl.model, self.ctrl.data)
            if v is None:
                raise RuntimeError('当前环境没有可用的图形显示')
            apply_isometric_camera(
                v.cam, lookat=(0.0, 0.0, 0.6),
                tracking_body=self.ctrl.pelvis_id)
            v.sync()
            with self._viewer_lock:
                self._viewer = v
            self._log("3D 窗口已打开")
        except Exception as exc:            # 无显示环境下优雅降级
            with self._viewer_lock:
                self._viewer = None
            self._log("3D 窗口打不开（%s），改为无界面运行" % exc)

    # ==================================================================
    # 3D 窗口的公开接口（模块调用时用）
    # ==================================================================
    def open_viewer(self, realtime: Optional[bool] = True) -> str:
        """随时打开 3D 窗口。

        构造时也可以直接 ``RobotSession(show_viewer=True)``；
        这个方法用于"先无界面跑，需要时再开窗口看"的场景。

        参数
            realtime  **默认 True**：打开窗口就切到按真实时间跑。
                      否则仿真全速跑，一个 2 秒的动作零点几秒就过去了，
                      窗口里几乎什么都看不清。
                      传 False 可以保留原来的快跑节奏。

        注意：窗口里的仿真**只在有动作执行时才推进**（空闲冻结），
        所以打开后如果没动作，画面会停在当前姿态——这是正常现象，
        不是卡死（鼠标依然能转动视角）。
        """
        if realtime is not None:
            self.realtime = bool(realtime)

        if self._viewer is not None and self._viewer.is_running():
            return "3D 窗口已经打开了。"

        self._open_viewer()
        if self._viewer is None:
            return ("3D 窗口打不开（可能没有图形界面 / 缺少 OpenGL 环境），"
                    "仿真仍在无界面模式下正常运行。")
        return ("3D 窗口已打开（realtime=%s）。窗口里可以用鼠标左键旋转、"
                "右键平移、滚轮缩放；关掉窗口不会终止仿真，"
                "需要停止请调用 close()。" % self.realtime)

    def close_viewer(self) -> str:
        """关掉 3D 窗口（仿真继续跑，只是不显示）。"""
        if not self._viewer_lock.acquire(timeout=2.0):
            return "3D 窗口正在刷新，请稍后再关闭。"
        try:
            viewer = self._viewer
            self._viewer = None
        finally:
            self._viewer_lock.release()
        if viewer is None:
            return "当前没有打开的 3D 窗口。"
        try:
            if viewer.is_running():
                viewer.close()
        except Exception:
            pass
        return "3D 窗口已关闭（仿真继续，只是不显示画面）。"

    @property
    def viewer_running(self) -> bool:
        """3D 窗口当前是否开着（且用户没关掉）。"""
        if not self._viewer_lock.acquire(timeout=1.0):
            return True
        try:
            viewer = self._viewer
        finally:
            self._viewer_lock.release()
        if viewer is None:
            return False
        try:
            return bool(viewer.is_running())
        except Exception:
            return False

    def wait_until_viewer_closed(self, timeout: Optional[float] = None,
                                 poll: float = 0.2) -> str:
        """阻塞等用户关掉 3D 窗口 —— 脚本末尾用它"看完再退出"。

        参数
            timeout  最长等待秒数（墙上时间）。None = 一直等，直到窗口被关。
            poll     轮询间隔

        为什么需要它：MuJoCo 的被动查看器**不阻塞**，
        脚本执行完就会退出、窗口跟着消失。想看完整段演示就得在这里挂住。
        """
        if self._viewer is None:
            return "当前没有 3D 窗口，无需等待。"
        t0 = time.perf_counter()
        while self.viewer_running:
            if timeout is not None and (time.perf_counter() - t0) > timeout:
                return "已等待 %.1f 秒，窗口仍开着。" % (time.perf_counter() - t0)
            time.sleep(poll)
        return "3D 窗口已被关闭。"

    # ==================================================================
    # 基础查询（线程安全）
    # ==================================================================
    def sim_time(self) -> float:
        """当前仿真时间（秒）。这是所有"时长"的计时基准。"""
        return float(self.ctrl.data.time)

    def status(self) -> dict:
        """读取一帧机器人状态（含位置、速度、朝向、是否摔倒、手臂姿态）。"""
        with self._lock:
            st = self.ctrl.status()
            pose = {name: float(v) for name, v in zip(
                self._pose_names(), self.ctrl.pose)}
        st["pose"] = pose
        st["cmd"] = tuple(self._cmd)
        from g1_walkfree import config as C
        upright = float(st.get("upright", 0.0))
        height = float(st.get("z", 0.0))
        stability = max(0.0, min(1.0, (upright - C.FALLEN_UPRIGHT)
                                / max(1e-6, 1.0 - C.FALLEN_UPRIGHT)))
        height_quality = max(0.0, min(1.0, (height - C.FALLEN_HEIGHT)
                                     / max(1e-6, 0.8 - C.FALLEN_HEIGHT)))
        quality = 0.75 * stability + 0.25 * height_quality
        st["reliability"] = {
            "stability": round(stability, 4),
            "height_quality": round(height_quality, 4),
            "quality": round(quality, 4),
            "fallen": bool(st.get("fallen")),
            "source": "simulated_ground_truth",
        }
        return st

    def describe(self) -> str:
        """把状态翻译成一段中文，方便直接回传给大模型。"""
        st = self.status()
        state = "已摔倒（需要复位）" if st["fallen"] else "站立/行走中"
        speed = st["speed"]
        moving = "移动中" if speed > 0.05 else "基本静止"
        reliability = st.get("reliability", {})
        return (
            "机器人状态：%s，%s。\n"
            "  位置 x=%+.3f m, y=%+.3f m, 高度 z=%.3f m；朝向 %+.1f°\n"
            "  线速度 %.3f m/s（vx=%+.3f, vy=%+.3f）；相对出发点 Δ=(%+.3f, %+.3f)\n"
            "  当前速度指令 vx=%+.2f vy=%+.2f ωz=%+.2f\n"
            "  仿真时间 %.2f s\n"
            "  可靠性：稳定性 %.2f，高度质量 %.2f，综合 %.2f（%s）"
            % (state, moving, st["x"], st["y"], st["z"], st["yaw"], speed,
               st["vx"], st["vy"], st["dx"], st["dy"],
               st["cmd"][0], st["cmd"][1], st["cmd"][2], st["time"],
               reliability.get("stability", 0.0),
               reliability.get("height_quality", 0.0),
               reliability.get("quality", 0.0),
               reliability.get("source", "unknown")))

    @staticmethod
    def _pose_names() -> list:
        from g1_walkfree import config as C
        return list(C.POSE_NAMES)

    # ==================================================================
    # 内部工具
    # ==================================================================
    def _log(self, msg: str) -> None:
        if self.verbose:
            print("[机器人] %s" % msg, flush=True)

    def _record(self, msg: str) -> None:
        stamp = "t=%.2fs" % self.sim_time()
        self._action_log.append("%s %s" % (stamp, msg))

    def _clamp(self, v: float, limit: float) -> float:
        lim = limit * self._speed_scale
        return float(max(-lim, min(lim, v)))

    def _wait_sim(self, seconds: float, wall_cap: float = 60.0) -> bool:
        """阻塞直到**仿真时间**推进 ``seconds`` 秒。

        返回 True 表示正常推进；False 表示墙上时间兜底超时（仿真卡住）。

        两个要点：
        * 用**仿真时间**而不是墙上时间计时 —— 快跑和实时两种模式都能对上。
        * 如果当前没有动作在执行，会**自动临时激活仿真**。
          否则"等仿真时间"和"空闲时暂停仿真"会互相死锁：没人激活 -> 仿真不走
          -> 一直等 -> 撞到墙上超时才返回（这个 bug 真踩过）。
        """
        if self._active > 0:
            return self._wait_sim_inner(seconds, wall_cap)
        with self._active_action():                 # 临时激活，保证仿真会动
            return self._wait_sim_inner(seconds, wall_cap)

    def _wait_sim_inner(self, seconds: float, wall_cap: float) -> bool:
        """真正干活的等待循环（调用前需确保仿真已被激活）。"""
        target = self.sim_time() + seconds
        deadline = time.perf_counter() + wall_cap
        while self.sim_time() < target:
            if self._stop_evt.is_set():
                return False
            if time.perf_counter() > deadline:
                return False
            time.sleep(0.002)
        return True

    def _set_cmd(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> None:
        with self._lock:
            self._cmd = [vx, vy, wz]

    def _stop_and_settle(self, settle: float = 0.35) -> None:
        """刹车：先清指令让策略收回动作，再等一小会儿站稳。"""
        self._set_cmd(0.0, 0.0, 0.0)
        self._wait_sim(settle, wall_cap=10.0)

    # ==================================================================
    # 动作 API（全部线程安全、全部返回中文描述）
    # ==================================================================
    @_needs_sim
    def walk_for(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0,
                 seconds: float = 1.0) -> str:
        """按速度指令走指定**仿真秒数**，然后自动刹停。

        参数
            vx       前进速度（正=身体前方），限幅 ±0.8 m/s
            vy       左移速度（正=身体左侧），限幅 ±0.45 m/s
            vyaw     转向角速度（正=左转），限幅 ±1.0 rad/s
            seconds  时长（仿真秒），限幅 0.1~60
        """
        seconds = max(0.1, min(MAX_ACTION_SECONDS, float(seconds)))
        vx, vy, vyaw = self._clamp(vx, SAFE_MAX_VX), self._clamp(vy, SAFE_MAX_VY), \
            self._clamp(vyaw, SAFE_MAX_WZ)

        if self.status()["fallen"]:
            return "机器人已经摔倒了，无法行走。请先调用 robot_reset 复位。"

        before = self.status()
        self._set_cmd(vx, vy, vyaw)
        ok = self._wait_sim(seconds)
        self._stop_and_settle()
        after = self.status()

        if not ok:
            return "行走被中断（仿真未按时推进），请检查仿真线程是否正常。"
        if after["fallen"]:
            self._record("walk_for 摔倒")
            return ("⚠ 执行中摔倒了！指令 vx=%+.2f vy=%+.2f ωz=%+.2f 共 %.1fs。"
                    "\n位移 Δ=(%+.3f, %+.3f)，请调用 robot_reset 复位后再试更保守的参数。"
                    % (vx, vy, vyaw, seconds, after["x"] - before["x"],
                       after["y"] - before["y"]))

        self._record("walk_for vx=%+.2f vy=%+.2f ωz=%+.2f %.1fs" % (vx, vy, vyaw, seconds))
        return ("已行走 %.1f 仿真秒（vx=%+.2f vy=%+.2f ωz=%+.2f）。"
                "\n  位移 Δ=(%+.3f, %+.3f) m，终点=(%+.3f, %+.3f)，朝向 %+.1f°"
                % (seconds, vx, vy, vyaw,
                   after["x"] - before["x"], after["y"] - before["y"],
                   after["x"], after["y"], after["yaw"]))

    @_needs_sim
    def goto(self, x: float, y: float, yaw: Optional[float] = None,
             timeout: float = 20.0) -> str:
        """闭环走到指定坐标（世界系）。位置控制由内部比例控制器完成。

        参数
            x, y    目标坐标 (m)，相对出发时机器人所在位置
            yaw     目标朝向（度，None = 不管朝向）
            timeout 最长仿真秒数，超时返回当前位置

        说明：这是**闭环**的 —— 每 0.1 仿真秒读一次实际位姿，
        把世界系误差旋转到机体系再下发速度，所以策略的漂移会被自动纠正。
        """
        if self.status()["fallen"]:
            return "机器人已经摔倒了，请先调用 robot_reset 复位。"

        timeout = max(1.0, min(MAX_ACTION_SECONDS, float(timeout)))
        t_end = self.sim_time() + timeout
        start = self.status()
        target_yaw = None if yaw is None else math.radians(float(yaw))

        k_pos = 1.2         # 位置误差 -> 速度的比例增益
        k_yaw = 1.5         # 朝向误差 -> 角速度的比例增益
        tick = 0.1          # 控制周期（仿真秒）

        while self.sim_time() < t_end:
            st = self.status()
            if st["fallen"]:
                self._stop_and_settle()
                return "⚠ 走到一半摔倒了，请调用 robot_reset 复位。"

            dx, dy = x - st["x"], y - st["y"]
            dist = math.hypot(dx, dy)
            yaw_now = math.radians(st["yaw"])

            # 朝向误差
            yaw_err = 0.0
            if target_yaw is not None:
                yaw_err = _wrap_angle(target_yaw - yaw_now)

            if dist < GOTO_TOL and abs(yaw_err) < YAW_TOL:
                break

            # 世界系误差 -> 机体系（机体系 x 朝前、y 朝左，与策略的 vx/vy 一致）
            c, s = math.cos(yaw_now), math.sin(yaw_now)
            ex_b = dx * c + dy * s
            ey_b = -dx * s + dy * c

            # 位置误差（米）乘增益得到速度；距离远了最多到限幅
            cmd_vx = self._clamp(k_pos * ex_b, SAFE_MAX_VX)
            cmd_vy = self._clamp(k_pos * ey_b, SAFE_MAX_VY)
            cmd_wz = self._clamp(k_yaw * yaw_err, SAFE_MAX_WZ)

            # 距离很近时只做朝向微调，避免在目标点附近来回蹭
            if dist < 0.05 and target_yaw is None:
                cmd_vx = cmd_vy = 0.0

            self._set_cmd(cmd_vx, cmd_vy, cmd_wz)
            self._wait_sim(tick, wall_cap=10.0)

        self._stop_and_settle()
        st = self.status()
        dist = math.hypot(x - st["x"], y - st["y"])
        yaw_txt = ""
        if target_yaw is not None:
            yaw_txt = "，朝向误差 %.1f°" % math.degrees(
                _wrap_angle(target_yaw - math.radians(st["yaw"])))

        self._record("goto (%+.2f,%+.2f)" % (x, y))
        if dist < GOTO_TOL:
            return ("已到达目标点 (%.2f, %.2f)。"
                    "\n  实际位置=(%+.3f, %+.3f)，距目标 %.3f m%s"
                    % (x, y, st["x"], st["y"], dist, yaw_txt))
        return ("未能精确到达：目标 (%.2f, %.2f)，实际 (%+.3f, %+.3f)，"
                "还剩 %.3f m%s（可能超时或受限幅影响）。"
                "\n  本轮从 (%+.3f, %+.3f) 出发，位移 (%+.3f, %+.3f)。"
                % (x, y, st["x"], st["y"], dist, yaw_txt,
                   start["x"], start["y"], st["x"] - start["x"], st["y"] - start["y"]))

    def turn(self, degrees: float, timeout: float = 10.0) -> str:
        """原地转向指定角度（度，正=左转/逆时针）。"""
        st = self.status()
        target_yaw = math.radians(st["yaw"] + float(degrees))
        # 原地转：把目标点设成"当前所在位置"，只让朝向来闭环
        return self.goto(st["x"], st["y"], yaw=math.degrees(target_yaw), timeout=timeout)

    @_needs_sim
    def stop(self) -> str:
        """立即刹停（清空速度指令并等一小会儿站稳）。"""
        self._stop_and_settle()
        st = self.status()
        self._record("stop")
        return ("已停止。当前位置 (%+.3f, %+.3f)，朝向 %+.1f°，%s。"
                % (st["x"], st["y"], st["yaw"],
                   "已摔倒" if st["fallen"] else "站立正常"))

    @_needs_sim
    def reset(self) -> str:
        """复位到站立姿态（摔倒后用它把机器人扶起来）。"""
        with self._lock:
            self.ctrl.reset()
            self._cmd = [0.0, 0.0, 0.0]
        self._last_fall = False
        self._wait_sim(0.3, wall_cap=10.0)
        st = self.status()
        self._record("reset")
        return ("已复位到站立姿态。位置归零，朝向 %+.1f°，高度 %.3f m。"
                % (st["yaw"], st["z"]))

    @_needs_sim
    def set_arm_preset(self, preset: str) -> str:
        """套用手臂预设姿态。可用预设见 ``available_arm_presets()``。"""
        from g1_walkfree import config as C
        if preset not in C.ARM_PRESETS:
            return ("没有名为「%s」的手臂预设。可用：%s"
                    % (preset, "、".join(C.ARM_PRESETS)))
        with self._lock:
            self.ctrl.apply_arm_preset(preset)
        self._wait_sim(0.4, wall_cap=10.0)      # 让手臂摆到位
        self._record("arm preset=%s" % preset)
        return ("已切换手臂姿态为「%s」。\n  当前手臂角度：%s"
                % (preset, _fmt_pose(self.ctrl.pose)))

    @_needs_sim
    def set_arm_joints(self, side: str = "both", **joints) -> str:
        """直接设置手臂关节角（弧度）。侧别可选 left / right / both。

        支持的关节名：shoulder_pitch, shoulder_roll, shoulder_yaw,
        elbow, wrist_roll, wrist_pitch, wrist_yaw。
        注意 shoulder_pitch **负值才是手臂向前**。
        """
        from g1_walkfree import config as C
        side = (side or "both").lower()
        if side not in ("left", "right", "both"):
            return "side 只能是 left / right / both，收到：%s" % side

        unknown = [k for k in joints if k not in C.ARM_JOINTS]
        if unknown:
            return ("不认识的关节：%s。可用：%s"
                    % ("、".join(unknown), "、".join(C.ARM_JOINTS)))

        sides = ("left", "right") if side == "both" else (side,)
        with self._lock:
            for sd in sides:
                for name, val in joints.items():
                    # 肩外张左右手符号相反，自动取反
                    v = float(val) * (-1.0 if (sd == "right" and name == "shoulder_roll")
                                      else 1.0)
                    self.ctrl.set_joint("%s_%s" % (sd, name), v)
        self._wait_sim(0.4, wall_cap=10.0)
        self._record("arm joints %s %s" % (side, joints))
        return ("已设置%s手臂关节 %s。\n  当前手臂角度：%s"
                % ("双侧" if side == "both" else ("左" if side == "left" else "右"),
                   ", ".join("%s=%+.2f" % (k, v) for k, v in joints.items()),
                   _fmt_pose(self.ctrl.pose)))

    @_needs_sim
    def set_waist(self, yaw: Optional[float] = None, roll: Optional[float] = None,
                  pitch: Optional[float] = None) -> str:
        """设置腰部三个关节角（弧度）。None 表示保持不变。"""
        with self._lock:
            if yaw is not None:
                self.ctrl.set_joint("waist_yaw", float(yaw))
            if roll is not None:
                self.ctrl.set_joint("waist_roll", float(roll))
            if pitch is not None:
                self.ctrl.set_joint("waist_pitch", float(pitch))
        self._wait_sim(0.3, wall_cap=10.0)
        self._record("waist yaw=%s roll=%s pitch=%s" % (yaw, roll, pitch))
        return ("已设置腰部：yaw=%s roll=%s pitch=%s（弧度）。\n  当前腰部：%s"
                % ("-" if yaw is None else "%+.2f" % yaw,
                   "-" if roll is None else "%+.2f" % roll,
                   "-" if pitch is None else "%+.2f" % pitch,
                   _fmt_pose(self.ctrl.pose, only_waist=True)))

    @_needs_sim
    def sequence(self, steps: list) -> str:
        """按顺序执行一串动作。``steps`` 是动作字典列表。

        每个动作形如::

            {"type": "walk",  "vx": 0.5, "vy": 0, "vyaw": 0, "seconds": 2}
            {"type": "goto",  "x": 1.0, "y": 0.0, "yaw": 0}
            {"type": "turn",  "degrees": 90}
            {"type": "arm",   "preset": "双手上举"}
            {"type": "arm",   "side": "left", "joints": {"elbow": 1.8}}
            {"type": "waist", "yaw": 0.3}
            {"type": "wait",  "seconds": 1.0}
            {"type": "stop"} / {"type": "reset"}
        """
        if not isinstance(steps, list) or not steps:
            return "steps 必须是非空的动作列表。"

        lines = []
        for i, step in enumerate(steps, 1):
            if not isinstance(step, dict):
                lines.append("[%d] 跳过：不是字典" % i)
                continue
            kind = str(step.get("type", "")).lower()
            try:
                if kind == "walk":
                    msg = self.walk_for(step.get("vx", 0.0), step.get("vy", 0.0),
                                        step.get("vyaw", 0.0),
                                        step.get("seconds", 1.0))
                elif kind == "goto":
                    msg = self.goto(step.get("x", 0.0), step.get("y", 0.0),
                                    step.get("yaw"), step.get("timeout", 20.0))
                elif kind == "turn":
                    msg = self.turn(step.get("degrees", 90.0))
                elif kind == "arm":
                    if "preset" in step:
                        msg = self.set_arm_preset(step["preset"])
                    else:
                        msg = self.set_arm_joints(step.get("side", "both"),
                                                  **step.get("joints", {}))
                elif kind == "waist":
                    msg = self.set_waist(step.get("yaw"), step.get("roll"),
                                         step.get("pitch"))
                elif kind == "wait":
                    self._wait_sim(min(30.0, max(0.0, step.get("seconds", 1.0))))
                    msg = "等待 %.1f 秒" % step.get("seconds", 1.0)
                elif kind == "stop":
                    msg = self.stop()
                elif kind == "reset":
                    msg = self.reset()
                else:
                    msg = "未知动作类型：%s" % kind
            except Exception as exc:                     # 单步失败不影响后续
                msg = "执行出错：%s" % exc
            lines.append("[%d] %s" % (i, msg.replace("\n", " ")))

        st = self.status()
        return ("动作序列执行完毕（共 %d 步）：\n%s\n最终位置 (%+.3f, %+.3f)，朝向 %+.1f°%s"
                % (len(steps), "\n".join(lines), st["x"], st["y"], st["yaw"],
                   "，⚠ 已摔倒" if st["fallen"] else ""))


# ======================================================================
# 小工具
# ======================================================================
def _wrap_angle(a: float) -> float:
    """把角度归一到 (-π, π]，避免"绕远路"转弯。"""
    while a > math.pi:
        a -= 2 * math.pi
    while a <= -math.pi:
        a += 2 * math.pi
    return a


def _fmt_pose(pose, only_waist: bool = False) -> str:
    """把姿态向量格式化成一行中文，方便直接喂给模型。"""
    from g1_walkfree import config as C
    items = []
    for name, v in zip(C.POSE_NAMES, pose):
        if only_waist and not name.startswith("waist"):
            continue
        items.append("%s=%+.2f" % (name, v))
    return "  ".join(items)


def available_arm_presets() -> list:
    """列出所有可用的手臂预设名。"""
    from g1_walkfree import config as C
    return list(C.ARM_PRESETS)
