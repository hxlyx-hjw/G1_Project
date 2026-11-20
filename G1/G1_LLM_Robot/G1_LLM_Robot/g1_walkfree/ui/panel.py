"""主控制面板：把布局拼起来，并用定时器驱动仿真。

界面结构（放大版，1280×900 起）
--------------------------------------------------------------------------
┌ 标题 ─────────────────────────────────────────────────────────────────┐
├ 运行控制： ▶/⏸  ⛔ 紧急停  ⟲ 复位  ✕ 退出    [✓] 视角跟随             │
├──────────── 左栏 (行走 + 状态) ──────────┬──── 右栏 (手臂) ──────────┤
│ 行走指令                                  │ 手臂控制                    │
│   前进 vx   [=========|=======] +0.50     │   (•)左臂 ( )右臂           │
│   横移 vy   [====|============] +0.00     │   肩 pitch [======|====]    │
│   转向 ωz   [======|==========] +0.00     │   ... 共 7 个               │
│   [前进][后退][左移][右移][左转][右转][停] │   预设: [...] [...]        │
│                                           │                             │
│ 状态                                      │                             │
│   时间 / 位置 / 速度 / 朝向 ...            │                             │
├───────────────────────────────────────────┴─────────────────────────────┤
│ 键盘说明                                                                 │
└──────────────────────────────────────────────────────────────────────────┘

仿真时钟：用 ``root.after`` 每 8ms 醒一次，按墙上时间累积步数，
所以**仿真速度与真实时间一致**，不会因为刷新率不同而快慢不一。
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import messagebox, ttk

from .. import config as C
from ..controller import WalkController
from . import theme as T
from .keymap import bind_keys
from .widgets import LabeledSlider, QuickButtonRow, ScrollableFrame, StatusCard


class WalkFreePanel:
    """自由可控行走控制台。"""

    #: 主循环唤醒间隔（毫秒）。8ms ≈ 125Hz，足够跟手也不吃 CPU。
    TICK_MS = 8

    def __init__(self, root: tk.Tk, ctrl: WalkController, show_viewer: bool = True):
        self.root = root
        self.ctrl = ctrl
        self.dt = ctrl.model.opt.timestep

        # ---- 运行状态 ----
        self.running = True          # 是否推进仿真
        self.closed = False
        self._last_wall = None       # 上一帧的墙上时间
        self._acc = 0.0              # 累积的"待推进仿真时间"
        self.side = "left"           # 当前在调哪只手臂
        self._last_arm_joint = C.ARM_JOINTS[0]   # 方向键微调的对象

        # ---- 窗口外观（尺寸按屏幕自适应，避免控件被裁掉）----
        self.fonts = T.install(root)
        (win_w, win_h), slider_len = T.compute_layout(root)
        T.Sizes.SLIDER_LENGTH = slider_len          # 供 make_slider 读取
        self.col_min_w = slider_len + T.SLIDER_CHROME_PX

        root.title("宇树 G1 · 自由可控行走 + 手臂控制台")
        root.geometry("%dx%d" % (win_w, win_h))
        root.minsize(T.WINDOW_MIN_SIZE[0], T.WINDOW_MIN_SIZE[1])
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        # ---- 拼装界面 ----
        self._build_header()
        self._build_body()
        self._build_footer()

        # ---- 键盘 ----
        bind_keys(root, self)

        # ---- 3D 查看器 ----
        self.viewer = None
        if show_viewer:
            self._start_viewer()

        # ---- 启动定时器 ----
        self.root.after(self.TICK_MS, self._tick)

    # ==================================================================
    # 一、界面拼装
    # ==================================================================
    def _build_header(self) -> None:
        """顶部：标题 + 运行控制按钮。"""
        head = tk.Frame(self.root, bg=T.Colors.BG)
        head.pack(fill="x", padx=T.PAD_OUTER, pady=(T.PAD_OUTER, 0))

        ttk.Label(head, text="宇树 G1 · 自由可控行走 + 手臂控制",
                  style="Title.TLabel").pack(side="left")

        bar = ttk.LabelFrame(self.root, text="运行控制")
        bar.pack(fill="x", padx=T.PAD_OUTER, pady=(T.PAD_INNER, 0))

        self.btn_run = ttk.Button(bar, text="⏸ 暂停", width=T.Sizes.BUTTON_W_MED,
                                  command=self.toggle_run)
        self.btn_run.grid(row=0, column=0, padx=6, pady=T.PAD_INNER)
        ttk.Button(bar, text="⛔ 紧急停", width=T.Sizes.BUTTON_W_MED,
                   command=self.emergency_stop).grid(row=0, column=1, padx=6)
        ttk.Button(bar, text="⟲ 复位", width=T.Sizes.BUTTON_W_MED,
                   command=self.reset).grid(row=0, column=2, padx=6)
        ttk.Button(bar, text="✕ 退出", width=T.Sizes.BUTTON_W_MED,
                   command=self.on_close).grid(row=0, column=3, padx=6)

        self.var_follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="视角跟随骨盆", variable=self.var_follow,
                        command=self.apply_camera).grid(row=0, column=4, padx=(24, 6))

        self.btn_view = ttk.Button(bar, text="打开 3D 窗口",
                                   command=self._start_viewer)
        self.btn_view.grid(row=0, column=5, padx=6)

    def _build_body(self) -> None:
        """中部：左右两栏。"""
        body = tk.Frame(self.root, bg=T.Colors.BG)
        body.pack(fill="both", expand=True, padx=T.PAD_OUTER, pady=T.PAD_INNER)
        # 两栏等宽；minsize 由实际滑杆长度算出来，保证不会被裁
        body.grid_columnconfigure(0, weight=1, minsize=self.col_min_w)
        body.grid_columnconfigure(1, weight=1, minsize=self.col_min_w)
        body.grid_rowconfigure(0, weight=1)

        self._build_drive_column(body)
        self._build_arm_column(body)

    # ---------------- 左栏：行走指令 + 状态 ----------------
    def _build_drive_column(self, parent) -> None:
        col = tk.Frame(parent, bg=T.Colors.BG)
        col.grid(row=0, column=0, sticky="nsew", padx=(0, T.PAD_GUTTER))

        # ---- 行走指令 ----
        drive = ttk.LabelFrame(col, text="行走指令（也可以用键盘 W/S A/D Q/E）")
        drive.pack(fill="x")

        self.sl_vx = LabeledSlider(drive, self.fonts, "前进 vx", -C.MAX_VX, C.MAX_VX,
                                   0.0, fmt="%+.2f m/s", on_change=self.on_cmd_changed)
        self.sl_vx.grid(row=0, column=0, sticky="w", pady=6, padx=T.PAD_INNER)

        self.sl_vy = LabeledSlider(drive, self.fonts, "横移 vy", -C.MAX_VY, C.MAX_VY,
                                   0.0, fmt="%+.2f m/s", on_change=self.on_cmd_changed)
        self.sl_vy.grid(row=1, column=0, sticky="w", pady=6, padx=T.PAD_INNER)

        self.sl_wz = LabeledSlider(drive, self.fonts, "转向 ωz", -C.MAX_WZ, C.MAX_WZ,
                                   0.0, fmt="%+.2f rad/s", on_change=self.on_cmd_changed)
        self.sl_wz.grid(row=2, column=0, sticky="w", pady=6, padx=T.PAD_INNER)

        quick = QuickButtonRow(drive, self.fonts, [
            ("前进 0.5", lambda: self.set_cmd(0.5, 0.0, 0.0)),
            ("慢走 0.2", lambda: self.set_cmd(0.2, 0.0, 0.0)),
            ("后退", lambda: self.set_cmd(-0.4, 0.0, 0.0)),
            ("左移", lambda: self.set_cmd(0.0, 0.3, 0.0)),
            ("右移", lambda: self.set_cmd(0.0, -0.3, 0.0)),
            ("左转", lambda: self.set_cmd(0.0, 0.0, 0.6)),
            ("右转", lambda: self.set_cmd(0.0, 0.0, -0.6)),
            ("原地站住", lambda: self.set_cmd(0.0, 0.0, 0.0)),
        ], per_row=4)
        quick.grid(row=3, column=0, sticky="ew", padx=T.PAD_INNER,
                   pady=(T.PAD_INNER, T.PAD_INNER))

        # ---- 状态 ----
        st = ttk.LabelFrame(col, text="状态")
        st.pack(fill="x", pady=(T.PAD_GUTTER, 0))
        self.status = StatusCard(st, self.fonts, [
            "仿真时间", "身体位置", "相对起点", "线速度", "朝向",
            "速度指令", "行走状态",
        ])
        self.status.pack(fill="x", padx=T.PAD_INNER, pady=T.PAD_INNER)

    # ---------------- 右栏：手臂 ----------------
    def _build_arm_column(self, parent) -> None:
        col = tk.Frame(parent, bg=T.Colors.BG)
        col.grid(row=0, column=1, sticky="nsew")

        outer = ttk.LabelFrame(col, text="手臂 / 腰部（自由位置控制）")
        outer.pack(fill="both", expand=True)

        # 内容较高（10 个滑杆 + 预设 + 提示），放进可滚动容器：
        # 大屏上一屏放得下不出现滚动条，小屏上自动出现滚动条而不是被裁掉。
        scroll = ScrollableFrame(outer)
        scroll.pack(fill="both", expand=True)
        arm = scroll.body

        # 左右臂切换
        side_row = tk.Frame(arm, bg=T.Colors.BG)
        side_row.grid(row=0, column=0, sticky="w", padx=T.PAD_INNER,
                      pady=(T.PAD_INNER, 0))
        ttk.Label(side_row, text="正在调节：", style="Section.TLabel").pack(side="left")
        self.var_side = tk.StringVar(value="left")
        for value, text in (("left", "左臂"), ("right", "右臂")):
            ttk.Radiobutton(side_row, text=text, value=value,
                            variable=self.var_side,
                            command=self.reload_arm_sliders).pack(side="left", padx=8)

        # 7 个手臂关节滑杆（量程取自模型关节限位）
        limits = self.ctrl.pose_limits()
        self.arm_sliders: dict[str, LabeledSlider] = {}
        for i, joint in enumerate(C.ARM_JOINTS):
            key = "left_%s" % joint
            lo, hi = limits.get(key, (-2.6, 2.6))
            sl = LabeledSlider(arm, self.fonts, C.ARM_JOINT_LABELS.get(joint, joint),
                               lo, hi,
                               self.ctrl.joint(key), fmt="%+.2f rad",
                               on_change=lambda v, j=joint: self.on_arm_changed(j, v))
            sl.grid(row=1 + i, column=0, sticky="w", padx=T.PAD_INNER, pady=4)
            self.arm_sliders[joint] = sl

        # 腰部 3 个关节
        row = 1 + len(C.ARM_JOINTS)
        ttk.Separator(arm, orient="horizontal").grid(
            row=row, column=0, sticky="ew", padx=T.PAD_INNER, pady=T.PAD_INNER)
        self.waist_sliders: dict[str, LabeledSlider] = {}
        for i, joint in enumerate(C.WAIST_JOINTS):
            lo, hi = limits.get(joint, (-1.0, 1.0))
            sl = LabeledSlider(arm, self.fonts, C.WAIST_JOINT_LABELS.get(joint, joint),
                               lo, hi, self.ctrl.joint(joint), fmt="%+.2f rad",
                               on_change=lambda v, j=joint: self.on_waist_changed(j, v))
            sl.grid(row=row + 1 + i, column=0, sticky="w", padx=T.PAD_INNER, pady=4)
            self.waist_sliders[joint] = sl

        # 预设按钮
        preset_row = tk.Frame(arm, bg=T.Colors.BG)
        preset_row.grid(row=row + 1 + 3, column=0, sticky="ew",
                        padx=T.PAD_INNER, pady=(T.PAD_INNER, 4))
        ttk.Label(preset_row, text="一键预设：", style="Body.TLabel").pack(side="left")
        for name in C.ARM_PRESETS:
            ttk.Button(preset_row, text=name, width=T.Sizes.BUTTON_W_SMALL,
                       command=lambda n=name: self.apply_preset(n)
                       ).pack(side="left", padx=4)

        ttk.Label(arm, style="Hint.TLabel", justify="left",
                  text=("⚠ 肩 pitch 的轴向下：负值 = 手臂向前，正值 = 手臂向后"
                        "（与直觉相反）。\n"
                        "   预设已按正确方向设好，直接点预设最省事。\n"
                        "提示：手臂摆得越开，整机质心越偏，行走速度会被带偏\n"
                        "（策略是按固定手臂姿态训练的），但不会摔倒。")
                  ).grid(row=row + 1 + 4, column=0, sticky="w",
                         padx=T.PAD_INNER, pady=(0, T.PAD_INNER))

        self._reload_waist_sliders()

    def _build_footer(self) -> None:
        """底部：键盘说明。"""
        foot = tk.Frame(self.root, bg=T.Colors.BG)
        foot.pack(fill="x", padx=T.PAD_OUTER, pady=(0, T.PAD_OUTER))
        ttk.Label(foot, style="Hint.TLabel", justify="left",
                  text=("键盘（先点一下本窗口）："
                        "W/S 前进后退 · A/D 左右平移 · Q/E 左右转 · 空格急停 · "
                        "R 复位 · X 退出 · 1~5 手臂预设 · ←/→ 切换左右臂 · ↑/↓ 微调当前关节    "
                        "｜  3D 窗口：鼠标左键旋转 / 右键平移 / 滚轮缩放")
                  ).pack(anchor="w")

    # ==================================================================
    # 二、回调
    # ==================================================================
    def on_cmd_changed(self, _value: float = 0.0) -> None:
        """任一速度滑杆变化 -> 写进控制器。"""
        self.ctrl.set_command(self.sl_vx.get(), self.sl_vy.get(), self.sl_wz.get())

    def set_cmd(self, vx: float, vy: float, vyaw: float) -> None:
        """快捷按钮用：先同步滑杆显示，再写控制器。"""
        self.sl_vx.set(vx)
        self.sl_vy.set(vy)
        self.sl_wz.set(vyaw)
        self.on_cmd_changed()

    def on_arm_changed(self, joint: str, value: float) -> None:
        """手臂滑杆变化 -> 写到 "<side>_<joint>"。"""
        self._last_arm_joint = joint          # 记住它，方向键微调用
        self.ctrl.set_joint("%s_%s" % (self.side, joint), value)

    def on_waist_changed(self, joint: str, value: float) -> None:
        self.ctrl.set_joint(joint, value)

    def reload_arm_sliders(self) -> None:
        """切左右臂时，把滑杆显示同步成该臂当前的目标角。"""
        for joint, sl in self.arm_sliders.items():
            sl.set(self.ctrl.joint("%s_%s" % (self.side, joint)))

    # ---- 供键盘映射调用 ----
    def switch_side(self, side: str) -> None:
        """切换"正在调节"的手臂（← / →）。"""
        if side not in ("left", "right"):
            return
        self.var_side.set(side)
        self.reload_arm_sliders()

    def nudge_current_arm(self, delta: float) -> None:
        """微调当前手臂的"最后碰过的那个关节"（↑ / ↓）。"""
        sl = self.arm_sliders.get(self._last_arm_joint)
        if sl is not None:
            sl.nudge(delta)

    def _reload_waist_sliders(self) -> None:
        for joint, sl in self.waist_sliders.items():
            sl.set(self.ctrl.joint(joint))

    def apply_preset(self, name: str) -> None:
        """套用预设到双臂，并刷新界面。"""
        self.ctrl.apply_arm_preset(name)
        self.reload_arm_sliders()

    # ---- 运行控制 ----
    def toggle_run(self) -> None:
        self.running = not self.running
        self.btn_run.config(text="⏸ 暂停" if self.running else "▶ 继续")
        self._last_wall = None          # 避免暂停期间累积时间
        self._acc = 0.0

    def emergency_stop(self) -> None:
        """急停：速度指令归零（机器人原地站住），但仿真继续跑。"""
        self.set_cmd(0.0, 0.0, 0.0)

    def reset(self) -> None:
        """复位到站立姿态，并把速度指令和滑杆都清零。"""
        self.ctrl.reset()
        self.set_cmd(0.0, 0.0, 0.0)
        self.reload_arm_sliders()
        self._reload_waist_sliders()
        self._acc = 0.0
        self._last_wall = None

    # ---- 3D 窗口 ----
    def _start_viewer(self) -> None:
        """打开（或重新打开）MuJoCo 被动查看器。"""
        if self.viewer is not None:
            try:
                if self.viewer.is_running():
                    return
            except Exception:
                pass
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
        try:
            import mujoco
            from viewer_camera import apply_isometric_camera, passive_viewer

            v = passive_viewer(self.ctrl.model, self.ctrl.data)
            if v is None:
                self.viewer = None
                messagebox.showwarning(
                    "无法打开 3D 窗口",
                    "当前环境没有可用的图形显示，控制台仍可正常使用。")
                return
            apply_isometric_camera(
                v.cam, lookat=(0.0, 0.0, 0.6),
                tracking_body=self.ctrl.pelvis_id if self.var_follow.get() else None)
            self._mujoco = mujoco
            self.viewer = v
            self.apply_camera()
            v.sync()
            self.btn_view.config(text="3D 窗口已打开")
        except Exception as exc:                      # 无显示环境时优雅降级
            self.viewer = None
            messagebox.showwarning("无法打开 3D 窗口",
                                   "原因：%s\n\n控制台仍可用（只是没有画面）。" % exc)

    def apply_camera(self) -> None:
        """切换"跟随骨盆 / 自由视角"。"""
        if self.viewer is None:
            return
        cam = self.viewer.cam
        if self.var_follow.get():
            cam.type = self._mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = self.ctrl.pelvis_id
        else:
            cam.type = self._mujoco.mjtCamera.mjCAMERA_FREE

    def on_close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.viewer is not None:
            try:
                if self.viewer.is_running():
                    self.viewer.close()
            except Exception:
                pass
            self.viewer = None
        try:
            self.root.destroy()
        except Exception:
            pass

    # ==================================================================
    # 三、主循环
    # ==================================================================
    def _tick(self) -> None:
        """定时器回调：推进仿真 + 刷新画面。

        用**墙上时间累积器**而不是"每次固定步数"，
        这样无论界面刷新快慢，仿真速度都严格等于真实时间。
        一次最多补 800 步（≈1.6s）防止卡顿后一次性追太多。
        """
        if self.closed:
            return

        # 用户直接关掉了 3D 窗口 -> 整个程序退出
        if self.viewer is not None:
            try:
                alive = self.viewer.is_running()
            except Exception:
                alive = False
            if not alive:
                self.viewer = None
                self.on_close()
                return

        now = time.perf_counter()
        if self._last_wall is None:
            self._last_wall = now
        wall = min(now - self._last_wall, 0.2)   # 卡顿超过 0.2s 就丢掉，不追
        self._last_wall = now

        if self.running:
            self._acc += wall
            steps = 0
            while self._acc >= self.dt and steps < 800:
                self.ctrl.step()
                self._acc -= self.dt
                steps += 1
        else:
            self._acc = 0.0

        if self.viewer is not None:
            try:
                if self.viewer.is_running():
                    self.viewer.sync()
            except Exception:
                self.viewer = None

        self._refresh_status()
        self.root.after(self.TICK_MS, self._tick)

    def _refresh_status(self) -> None:
        """把 controller.status() 映射成状态卡的字符串。"""
        st = self.ctrl.status()
        if st["fallen"]:
            walk_state = "⚠ 已摔倒（按 R 复位）"
        elif not self.running:
            walk_state = "已暂停"
        elif abs(st["cmd"][0]) + abs(st["cmd"][1]) + abs(st["cmd"][2]) < 1e-3:
            walk_state = "原地站立"
        else:
            walk_state = "行走中"

        self.status.update_values({
            "仿真时间": "%7.2f s" % st["time"],
            "身体位置": "x=%+.3f  y=%+.3f  z=%.3f" % (st["x"], st["y"], st["z"]),
            "相对起点": "Δ=(%+.3f, %+.3f)" % (st["dx"], st["dy"]),
            "线速度": "%.3f m/s  (vx=%+.3f vy=%+.3f)" % (st["speed"], st["vx"], st["vy"]),
            "朝向": "%+7.1f°" % st["yaw"],
            "速度指令": "vx=%+.2f  vy=%+.2f  ωz=%+.2f"
                        % (st["cmd"][0], st["cmd"][1], st["cmd"][2]),
            "行走状态": walk_state,
        }, warn=st["fallen"])


# ======================================================================
def launch(ctrl: WalkController, show_viewer: bool = True) -> None:
    """创建窗口并进入事件循环（阻塞直到窗口关闭）。"""
    root = tk.Tk()
    WalkFreePanel(root, ctrl, show_viewer=show_viewer)
    root.mainloop()
