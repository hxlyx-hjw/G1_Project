"""GUI 与 G1 后端之间的桥接层。

设计原则
--------
1. **界面线程永不阻塞**：机器人初始化、模型对话、工具执行、环境检查全部在
   单独的后台线程里按队列顺序执行。
2. **不改现有源码**：通过运行时子类化 ``ExtendedRobotChatBot`` 接管事件，
   ``deepseek_robot.py`` / ``robot_tools.py`` / 扩展包保持原样。
3. **事件驱动**：后台只发 Qt 信号，界面只订阅信号，不直接操作后端对象。
"""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
from typing import Any, Callable

from PyQt5.QtCore import QObject, pyqtSignal


class BackendSignals(QObject):
    """后台线程 -> 界面线程的全部事件。"""

    ready = pyqtSignal(dict)
    failed = pyqtSignal(str)
    log = pyqtSignal(str)
    notice = pyqtSignal(str, str)              # level, text
    chat_started = pyqtSignal(str)             # user text
    chat_finished = pyqtSignal(str, str)       # answer, reasoning
    chat_error = pyqtSignal(str)
    tool_started = pyqtSignal(str, str)        # name, args json
    tool_finished = pyqtSignal(str, str, float)  # name, result, elapsed
    proactive = pyqtSignal(str)
    env_result = pyqtSignal(str, str)          # target, text
    training_changed = pyqtSignal()


def _make_bot_class():
    """在 G1 源码已加入 sys.path 后，动态创建事件化子类。"""

    from g1_deepseek_ext.chat import ExtendedRobotChatBot

    class GuiRobotChatBot(ExtendedRobotChatBot):
        def __init__(self, *args, event_sink: Callable[..., None] | None = None, **kwargs):
            self._events = event_sink or (lambda *a, **k: None)
            super().__init__(*args, **kwargs)

        # 后台日志进入状态栏，而不是打印到控制台。
        def _log(self, msg: str) -> None:
            self._events("log", msg=msg)

        # 主动报告要显示成聊天气泡，同时仍写入对话历史。
        def _emit_proactive(self, text: str) -> None:
            message = "[DeepSeek 主动消息]\n" + str(text)
            with self._chat_lock:
                self.history.append({"role": "assistant", "content": message})
            self._events("proactive", text=message)

        def _dispatch_tool(self, tool_call) -> None:
            name = tool_call.function.name
            args = tool_call.function.arguments or "{}"
            started = time.perf_counter()
            self._events("tool_start", name=name, args=args)
            try:
                super()._dispatch_tool(tool_call)
            except Exception as exc:
                # 工具异常不能把对话历史留在“有 tool_calls、无 tool 结果”的
                # 非法状态，否则下一轮请求会被 API 拒绝。补一条 tool 消息。
                result = json.dumps(
                    {"success": False, "error": "%s: %s" % (type(exc).__name__, exc)},
                    ensure_ascii=False,
                )
                self.history.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": result}
                )
                self._events(
                    "tool_end",
                    name=name,
                    result=result,
                    elapsed=time.perf_counter() - started,
                )
                return
            result = ""
            if self.history and self.history[-1].get("role") == "tool":
                result = self.history[-1].get("content", "")
            self._events(
                "tool_end",
                name=name,
                result=result,
                elapsed=time.perf_counter() - started,
            )

    return GuiRobotChatBot


class BackendService:
    """后台线程 + 命令队列。所有公开方法都可以从界面线程直接调用。"""

    def __init__(self, options: dict[str, Any]):
        self.options = dict(options)
        self.signals = BackendSignals()
        self._queue: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="g1-backend", daemon=True)
        self._bot = None
        self._session = None
        self._integration = None
        self._ready = threading.Event()
        self._closing = threading.Event()
        self._started = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()
        self._submit(self._initialize)

    def retry(self) -> None:
        self._ready.clear()
        self._submit(self._initialize)

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    def _loop(self) -> None:
        while not self._closing.is_set():
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                break
            try:
                job()
            except Exception:
                self.signals.log.emit("后台任务异常：\n" + traceback.format_exc())

    def _submit(self, job: Callable[[], None]) -> None:
        if self._closing.is_set():
            return
        self._queue.put(job)

    def shutdown(self) -> None:
        if self._closing.is_set():
            return
        self._closing.set()
        if not self._started:
            return
        self._queue.put(self._do_close)
        self._thread.join(timeout=8.0)
        if self._thread.is_alive():
            # 模型请求可能还在网络上；直接停掉仿真并让守护线程随进程退出。
            session = self._session
            if session is not None:
                try:
                    session._stop_evt.set()
                    session._set_cmd(0.0, 0.0, 0.0)
                except Exception:
                    pass

    def _do_close(self) -> None:
        bot = self._bot
        if bot is not None:
            try:
                bot.close()
            except Exception:
                self.signals.log.emit("关闭后端失败：\n" + traceback.format_exc())

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _initialize(self) -> None:
        if self._closing.is_set():
            return
        if self._bot is not None:
            try:
                self._bot.close()
            except Exception:
                pass
            self._bot = None
            self._session = None

        self.signals.notice.emit("info", "正在初始化 G1 仿真与 DeepSeek 客户端…")
        try:
            from g1_deepseek_ext.adapters import DeepSeekIntegration, IntegrationConfig

            integration = DeepSeekIntegration(IntegrationConfig.from_env())
            self._integration = integration
            bot_cls = _make_bot_class()
            opts = self.options
            bot = bot_cls(
                deepseek_api_key=opts.get("key"),
                model=opts.get("model", "deepseek-flash"),
                base_url=opts.get("base_url", "https://api.deepseek.com"),
                temperature=float(opts.get("temperature", 0.3)),
                reasoning_effort=opts.get("reasoning_effort", "medium"),
                thinking=True,
                enable_robot=bool(opts.get("enable_robot", True)),
                robot_viewer=bool(opts.get("viewer", False)),
                robot_realtime=True if opts.get("viewer") else None,
                walker_dir=opts.get("walker_dir"),
                robot_speed_scale=float(opts.get("speed_scale", 1.0)),
                enable_search=bool(opts.get("enable_search", False)),
                enable_local_files=True,
                verbose=False,
                enable_proactive=bool(opts.get("proactive", True)),
                proactive_interval_seconds=float(opts.get("proactive_interval", 600.0)),
                integration=integration,
                event_sink=self._emit,
            )
        except Exception as exc:
            detail = traceback.format_exc()
            self.signals.log.emit(detail)
            self.signals.failed.emit("%s: %s" % (type(exc).__name__, exc))
            return

        self._bot = bot
        self._session = bot.session
        self._ready.set()
        self.signals.ready.emit(self._describe_ready(bot, integration))

    def _describe_ready(self, bot, integration) -> dict:
        tools = []
        for schema in bot.tools:
            fn = schema.get("function", {})
            name = fn.get("name", "")
            if name.startswith("robot_supermimic"):
                source = "SuperMimic"
            elif name.startswith("robot_rax"):
                source = "RAX"
            elif name.startswith("robot_"):
                source = "机器人"
            else:
                source = "基础"
            tools.append(
                {
                    "name": name,
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                    "source": source,
                }
            )

        presets: list[str] = []
        if bot.session is not None:
            try:
                from robot_bridge import available_arm_presets

                presets = available_arm_presets()
            except Exception:
                presets = []

        cfg = integration.config
        proactive = bot.proactive.summary() if getattr(bot, "proactive", None) else None
        return {
            "model": bot.model,
            "base_url": str(getattr(bot.client, "base_url", "")),
            "temperature": bot.temperature,
            "reasoning_effort": bot.reasoning_effort,
            "robot": bot.session is not None,
            "viewer": bool(bot.viewer_running),
            "arm_presets": presets,
            "tools": tools,
            "proactive": proactive,
            "paths": {
                "g1_root": str(cfg.g1_root),
                "supermimic": str(cfg.supermimic_root),
                "rax": str(cfg.rax_root),
                "wsl": cfg.wsl_distro,
                "venv": cfg.supermimic_venv,
                "jobs": str(cfg.jobs_root),
            },
            "initial_status": bot.session.describe() if bot.session is not None else "",
        }

    # ------------------------------------------------------------------
    # 事件出口
    # ------------------------------------------------------------------
    def _emit(self, kind: str, **payload) -> None:
        if kind == "log":
            self.signals.log.emit(str(payload.get("msg", "")))
        elif kind == "proactive":
            self.signals.proactive.emit(str(payload.get("text", "")))
        elif kind == "tool_start":
            self.signals.tool_started.emit(
                str(payload.get("name", "")), str(payload.get("args", "{}"))
            )
        elif kind == "tool_end":
            self.signals.tool_finished.emit(
                str(payload.get("name", "")),
                str(payload.get("result", "")),
                float(payload.get("elapsed", 0.0)),
            )

    # ------------------------------------------------------------------
    # 对话
    # ------------------------------------------------------------------
    def ask(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._submit(lambda: self._chat(text))

    def _chat(self, text: str) -> None:
        bot = self._bot
        if bot is None:
            self.signals.chat_error.emit("后端尚未就绪。")
            return
        self.signals.chat_started.emit(text)
        try:
            answer = bot.chat(text)
        except Exception as exc:
            self.signals.log.emit(traceback.format_exc())
            self.signals.chat_error.emit("%s: %s" % (type(exc).__name__, exc))
            return
        reasoning = self._last_reasoning(bot)
        self.signals.chat_finished.emit(answer or "（模型没有返回文本）", reasoning)

    @staticmethod
    def _last_reasoning(bot) -> str:
        for message in reversed(bot.history):
            if message.get("role") != "assistant":
                continue
            reasoning = message.get("reasoning_content")
            if reasoning:
                return str(reasoning)
        return ""

    def reset_chat(self) -> None:
        self._submit(lambda: self._bot.reset() if self._bot else None)

    # ------------------------------------------------------------------
    # 机器人
    # ------------------------------------------------------------------
    def robot_status(self) -> dict | None:
        session = self._session
        if session is None:
            return None
        try:
            return session.status()
        except Exception:
            return None

    @property
    def viewer_running(self) -> bool:
        bot = self._bot
        if bot is None:
            return False
        try:
            return bool(bot.viewer_running)
        except Exception:
            return False

    def run_robot_tool(self, name: str, args: dict | None = None) -> None:
        args = args or {}
        self._submit(lambda: self._run_manual_tool(name, args))

    def _run_manual_tool(self, name: str, args: dict) -> None:
        bot = self._bot
        if bot is None or bot.robot is None:
            self.signals.notice.emit("warn", "机器人未启用，无法执行该操作。")
            return
        args_text = json.dumps(args, ensure_ascii=False)
        started = time.perf_counter()
        self.signals.tool_started.emit(name, args_text)
        try:
            result = bot.robot.dispatch(name, args)
        except Exception as exc:
            result = json.dumps(
                {"success": False, "error": "%s: %s" % (type(exc).__name__, exc)},
                ensure_ascii=False,
            )
        self.signals.tool_finished.emit(name, str(result), time.perf_counter() - started)

    def emergency_stop(self) -> None:
        """界面线程直接清零速度指令，不排队，保证急停即时生效。"""
        session = self._session
        if session is None:
            return
        try:
            session._set_cmd(0.0, 0.0, 0.0)
            self.signals.notice.emit("warn", "已发送急停：速度指令清零。")
        except Exception as exc:
            self.signals.notice.emit("danger", "急停失败：%s" % exc)

    def open_viewer(self) -> None:
        self._submit(self._open_viewer)

    def _open_viewer(self) -> None:
        bot = self._bot
        if bot is None:
            return
        try:
            message = bot.open_viewer()
        except Exception as exc:
            message = "打开 3D 窗口失败：%s" % exc
        self.signals.notice.emit("info", str(message))

    def close_viewer(self) -> None:
        self._submit(self._close_viewer)

    def _close_viewer(self) -> None:
        bot = self._bot
        if bot is None:
            return
        try:
            message = bot.close_viewer()
        except Exception as exc:
            message = "关闭 3D 窗口失败：%s" % exc
        self.signals.notice.emit("info", str(message))

    # ------------------------------------------------------------------
    # 训练任务
    # ------------------------------------------------------------------
    def training_jobs(self) -> list[dict]:
        bot = self._bot
        if bot is None or bot.robot is None:
            return []
        jobs = getattr(bot.robot.training, "jobs", {}) or {}
        result: list[dict] = []
        for job_id in reversed(list(jobs.keys())[-12:]):
            try:
                result.append(bot.robot.training.status(job_id))
            except Exception:
                continue
        return result

    def job_log_tail(self, job_id: str, limit: int = 8000) -> str:
        bot = self._bot
        if bot is None or bot.robot is None:
            return ""
        task = getattr(bot.robot.training, "jobs", {}).get(job_id)
        if not task:
            return ""
        path = task["folder"] / "worker.log"
        try:
            size = path.stat().st_size
            with path.open("rb") as handle:
                if size > limit:
                    handle.seek(size - limit)
                data = handle.read()
            return data.decode("utf-8", errors="replace")
        except OSError:
            return ""

    def training_action(self, action: str, job_id: str) -> None:
        self._submit(lambda: self._training_action(action, job_id))

    def _training_action(self, action: str, job_id: str) -> None:
        bot = self._bot
        if bot is None or bot.robot is None:
            return
        try:
            if action == "stop":
                message = json.dumps(
                    bot.robot.training.stop(job_id), ensure_ascii=False
                )
            elif action == "viewer":
                message = json.dumps(
                    bot.robot.training.open_viewer(job_id), ensure_ascii=False
                )
            elif action == "report":
                report = bot.proactive.send_now(job_id) if bot.proactive else {}
                message = json.dumps(report, ensure_ascii=False, default=str)
            elif action == "gif":
                message = json.dumps(
                    bot.robot.training.export_gif(job_id=job_id), ensure_ascii=False
                )
            else:
                message = "未知操作：%s" % action
        except Exception as exc:
            message = "操作失败：%s: %s" % (type(exc).__name__, exc)
        self.signals.notice.emit("info", message)
        self.signals.training_changed.emit()

    def proactive_state(self) -> dict | None:
        bot = self._bot
        if bot is None or getattr(bot, "proactive", None) is None:
            return None
        try:
            return bot.proactive.summary()
        except Exception:
            return None

    def configure_proactive(self, enabled: bool, interval_minutes: float) -> None:
        self._submit(lambda: self._configure_proactive(enabled, interval_minutes))

    def _configure_proactive(self, enabled: bool, interval_minutes: float) -> None:
        bot = self._bot
        if bot is None or getattr(bot, "proactive", None) is None:
            self.signals.notice.emit("warn", "主动报告未启用。")
            return
        try:
            info = bot.proactive.configure(
                enabled=bool(enabled),
                interval_seconds=max(6.0, float(interval_minutes) * 60.0),
            )
            self.signals.notice.emit(
                "info",
                "主动报告：%s，间隔 %.1f 分钟"
                % ("开启" if info["enabled"] else "关闭", info["interval_minutes"]),
            )
        except Exception as exc:
            self.signals.notice.emit("danger", "设置主动报告失败：%s" % exc)

    # ------------------------------------------------------------------
    # 环境检查
    # ------------------------------------------------------------------
    def check_environment(self) -> None:
        self._submit(self._check_environment)

    def _check_environment(self) -> None:
        integration = self._integration
        if integration is None:
            self.signals.failed.emit("集成适配器尚未初始化。")
            return
        for target, tool in (
            ("supermimic", "robot_supermimic_status"),
            ("rax", "robot_rax_status"),
        ):
            try:
                text = integration.dispatch(tool, {})
            except Exception as exc:
                text = json.dumps(
                    {"success": False, "error": "%s: %s" % (type(exc).__name__, exc)},
                    ensure_ascii=False,
                )
            self.signals.env_result.emit(target, str(text))
