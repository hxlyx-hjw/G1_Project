"""统一的 MuJoCo 轴测观察相机。

G1 模型朝向 ``+x``。``azimuth=135`` 时相机位于机器人前方偏右，形成
正面 3/4 轴测视角；``elevation=-24`` 略微俯视，既能看清全身，也能
看到足够的训练场地。所有交互式 viewer 和离屏渲染都从这里取初始视角，
避免不同入口出现“斜后方”或角度不一致的问题。
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Sequence

ISOMETRIC_AZIMUTH = 135.0
ISOMETRIC_ELEVATION = -24.0
ISOMETRIC_DISTANCE = 3.6

#: 训练监视窗口需要展示更大的活动范围，因此比普通查看器稍远。
TRAINING_DISTANCE = 4.2

DEFAULT_LOOKAT = (0.0, 0.0, 0.62)

_viewer_warning_shown = False
_x_error_handler = None
_viewer_used = False


def _install_x_error_tolerance():
    """VcXsrv/GLX 在窗口销毁时可能报告 GLXBadDrawable 并终止进程。

    安装一个仅打印一次的 Xlib 错误处理器，让查看器关闭阶段不再触发
    进程级退出。设置 ``G1_X11_STRICT=1`` 可恢复 Xlib 默认行为。
    """
    global _x_error_handler
    if _x_error_handler is not None or os.environ.get('G1_X11_STRICT') == '1':
        return
    if not sys.platform.startswith('linux'):
        return
    try:
        import ctypes

        class XErrorEvent(ctypes.Structure):
            _fields_ = [
                ('type', ctypes.c_int), ('display', ctypes.c_void_p),
                ('resourceid', ctypes.c_ulong), ('serial', ctypes.c_ulong),
                ('error_code', ctypes.c_ubyte),
                ('request_code', ctypes.c_ubyte),
                ('minor_code', ctypes.c_ubyte),
            ]

        callback = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(XErrorEvent))
        reported = [False]

        def handler(_display, event):
            if not reported[0]:
                code = event.contents.error_code if event else -1
                print('[viewer] 已忽略 X11/GLX 窗口关闭错误（error_code=%s），'
                      '进程继续运行；严格模式可设 G1_X11_STRICT=1。' % code,
                      file=sys.stderr, flush=True)
                reported[0] = True
            return 0

        _x_error_handler = callback(handler)
        lib = ctypes.CDLL('libX11.so.6')
        lib.XSetErrorHandler.argtypes = [callback]
        lib.XSetErrorHandler.restype = ctypes.c_void_p
        lib.XSetErrorHandler(_x_error_handler)
    except Exception:
        _x_error_handler = None


def display_available() -> bool:
    """判断当前进程是否有可用的图形显示。"""
    if sys.platform.startswith('linux'):
        return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
    return True


def passive_viewer(model, data):
    """安全创建 MuJoCo 被动查看器；无显示环境返回 None。

    Docker/SSH 无 X11/Wayland 时，GLFW 会打印
    ``ERROR: could not initialize GLFW`` 并让调用方崩溃。这里先检查显示变量，
    再捕获驱动异常，保证训练/评估可以继续以无界面模式运行。
    """
    global _viewer_warning_shown, _viewer_used
    if not display_available():
        if not _viewer_warning_shown:
            print('[viewer] 未检测到 DISPLAY/WAYLAND_DISPLAY，已切换无界面模式；'
                  '如需窗口请配置 X11/VcXsrv/WSLg 或使用 --headless。',
                  file=sys.stderr, flush=True)
            _viewer_warning_shown = True
        return None
    try:
        import mujoco.viewer
        handle = mujoco.viewer.launch_passive(model, data)
        _viewer_used = True
        _install_x_error_tolerance()
        return handle
    except Exception as exc:
        if not _viewer_warning_shown:
            print('[viewer] 图形窗口初始化失败（%s），已切换无界面模式。' % exc,
                  file=sys.stderr, flush=True)
            _viewer_warning_shown = True
        return None


def finish_process(code=0):
    """结束进程；使用过 VcXsrv/GLX 查看器时跳过不安全的解释器析构。

    VcXsrv 在 Xlib/GLX 退出清理时可能触发 GLXBadDrawable 并导致
    SIGSEGV（退出码 139）。业务层已经完成 close()/保存后，用
    ``os._exit`` 保留原退出码并跳过该析构路径。
    """
    if _viewer_used and sys.platform.startswith('linux') \
            and os.environ.get('G1_X11_HARD_EXIT', '1') != '0':
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(int(code or 0))
    raise SystemExit(code)


def apply_isometric_camera(cam, *, distance: float = ISOMETRIC_DISTANCE,
                           lookat: Sequence[float] = DEFAULT_LOOKAT,
                           tracking_body: Optional[int] = None):
    """把已有 ``MjvCamera`` 设置为统一的轴测视角。

    参数
        cam             MuJoCo ``MjvCamera``（viewer.cam 或离屏相机）
        distance        相机到观察点的距离
        lookat          自由视角下的观察点
        tracking_body   需要跟随的 body id；``None``/负数表示自由视角
    """
    import mujoco

    cam.azimuth = ISOMETRIC_AZIMUTH
    cam.elevation = ISOMETRIC_ELEVATION
    cam.distance = float(distance)
    cam.lookat[:] = list(lookat)
    if tracking_body is None or int(tracking_body) < 0:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.trackbodyid = -1
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = int(tracking_body)
    return cam


def isometric_camera(model=None, *, distance: float = ISOMETRIC_DISTANCE,
                     lookat: Sequence[float] = DEFAULT_LOOKAT,
                     tracking_body: Optional[int] = None):
    """创建一个已设置好轴测视角的新 ``MjvCamera``。"""
    import mujoco

    cam = mujoco.MjvCamera()
    if model is not None:
        mujoco.mjv_defaultFreeCamera(model, cam)
    return apply_isometric_camera(
        cam, distance=distance, lookat=lookat, tracking_body=tracking_body)


def training_window_size(screen_width: int, screen_height: int) -> tuple[int, int]:
    """按屏幕尺寸计算训练监视窗口大小（宽, 高）。

    优先占据屏幕约 94% 宽度，给图像区域留出尽可能大的空间；
    同时保留任务栏和窗口边框所需的余量。
    """
    screen_width = max(1, int(screen_width))
    screen_height = max(1, int(screen_height))
    width = min(1920, max(1280, int(screen_width * 0.94)),
                max(1024, screen_width - 60))
    height = min(1080, max(800, int(screen_height * 0.90)),
                 max(600, screen_height - 90))
    width = min(width, max(800, screen_width - 20))
    height = min(height, max(600, screen_height - 40))
    return width, height

