"""
================================================================
DeepSeek 对话 + 宇树 G1 机器人控制（Function Calling 模块版）
================================================================

在 ``deepseek_loop.py`` 那套 Harness 的基础上，把「操作 G1 人形机器人」
做成一组工具，于是**运行命令可以由大模型现场生成**：

    用户："往前走一米，然后举起双手，最后转个身"

    模型（自动决策）：
        robot_status()                      先看一眼现在在哪
        robot_goto(x=1.0, y=0.0)            闭环走过去（不是简单定时走）
        robot_set_arm(preset="双手上举")     举手
        robot_turn(degrees=180)             转身
        robot_status()                      回报结果
    用户："好了，现在在 (0.98, 0.03)，双手上举，朝向 179°"

关键点：模型不只能"发指令"，还能通过 ``robot_status`` **看到执行结果**，
所以它能做闭环——走不够就再走一点，摔倒了就 reset 重来。

依赖安装：
    pip install openai          # 必需
    pip install tavily-python   # 可选，联网搜索
    pip install pypdf python-docx  # 可选，本地文件解析

【直接运行】（命令行反复对话）
    python deepseek_robot.py
    python deepseek_robot.py --viewer          # 同时开 3D 窗口看机器人
    python deepseek_robot.py --task "走到(1,0)然后举手"   # 单条指令
    python deepseek_robot.py --selftest        # 不连大模型，脚本化验证机器人链路
    python deepseek_robot.py --list-tools      # 打印所有工具声明（调试用）

【作为模块调用】
    from deepseek_robot import RobotChatBot

    bot = RobotChatBot(deepseek_api_key="sk-xxx")
    print(bot.chat("往前走一米"))
    print(bot.chat("再举一下手"))     # 自动记住上下文
    bot.close()

API Key 可以：传参 / 环境变量 DEEPSEEK_API_KEY / 当前目录下的 api_key.txt
退出：输入 exit / quit / 退出 / q
"""

from __future__ import annotations

import json                                  # 解析模型返回的工具调用参数（JSON 字符串）
import os                                    # 读取环境变量中的 API Key
import threading                             # 主动报告线程与对话历史锁
from pathlib import Path                     # 处理文件路径
from typing import Iterator, List, Optional  # 类型注解

from openai import OpenAI                    # DeepSeek 官方推荐用 OpenAI SDK（兼容协议）

from robot_bridge import RobotSession        # 机器人会话（后台仿真 + 线程安全动作）
from robot_tools import RobotToolkit         # 机器人工具分发器

__all__ = ["RobotChatBot", "DEFAULT_SYSTEM_PROMPT"]


# ======================================================================
# 系统提示词：告诉模型"你是谁、机器人能干什么、应该怎么用工具"
# ======================================================================
DEFAULT_SYSTEM_PROMPT = """你是一台宇树 G1 人形机器人的操作员。
你可以通过工具直接驱动仿真里的 G1 走路、转身、摆手臂。

【坐标约定】以机器人**出发位置**为原点：
    +x = 出发时的前方     +y = 出发时的左侧     朝向 0° = 出发时的朝向
所有速度、角度都用这个坐标系，机器人会自动处理世界系/机体系的换算。

【工作方式】
1. 拿到指令后先想清楚要分几步。要走到具体地点就用 robot_goto（闭环、精确）；
   只是"走一会儿"才用 robot_walk。
2. 动作前后都可以调用 robot_status 查看实际结果。**建议做完了复查一次**，
   这样你能如实告诉用户"现在在哪、姿态如何"，而不是只说"我发过指令了"。
3. 多个动作连续做时，优先用 robot_sequence 一次下发，省往返、时序更准。
4. 如果状态显示"已摔倒"，必须先 robot_reset 才能继续。
5. 只要用户表示"停""别动""等一下"，立刻 robot_stop。

【手臂方向的坑】
shoulder_pitch 的轴让 **负值 = 手臂向前**，正值 = 手臂向后（与直觉相反）。
所以想举手/挥手/前伸，优先用 robot_set_arm 的 preset（预设已按正确方向调好），
不要自己推算关节角。

【安全】
速度指令超过上限会被自动夹住；任何异常都要如实告诉用户，不要假装成功。
"""


# ======================================================================
# 辅助：找 API Key
# ======================================================================
def _resolve_key(explicit: Optional[str], env_name: str,
                 file_name: str = "api_key.txt") -> Optional[str]:
    """按 显式传参 -> 环境变量 -> 当前目录 api_key.txt 的顺序找 Key。"""
    if explicit:
        return explicit.strip()
    env = os.getenv(env_name)
    if env:
        return env.strip()
    p = Path(file_name)
    if p.is_file():
        text = p.read_text(encoding="utf-8").strip()
        if text:
            return text
    return None


# ======================================================================
class RobotChatBot:
    """DeepSeek 对话 + G1 机器人控制。

    工具家族（可按开关裁剪）：
        robot_*      机器人动作（核心，默认开启）
        search_web   联网搜索（需要 tavily_api_key）
        read_file / write_file / search_files  本地文件（默认开启）
        consult_expert  专家 AI 咨询（需要另一个 key）
    """

    def __init__(
        self,
        deepseek_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        model: str = "deepseek-flash",
        base_url: str = "https://api.deepseek.com",
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,          # 操作机器人要稳，温度调低
        reasoning_effort: str = "medium",  # 思考强度：low / medium / high / max（medium 实际映射到 high）
        thinking: bool = True,             # 思考模式开关；本地模型(Ollama)设为 False
        max_tool_rounds: Optional[int] = 100,  # 普通工具限制；训练调用自动切换为不限轮数
        # ---- 机器人相关 ----
        enable_robot: bool = True,
        robot_viewer: bool = False,        # 开 3D 窗口
        robot_realtime: Optional[bool] = None,  # None=跟随 viewer
        walker_dir: Optional[str] = None,  # g1_walkfree 包所在目录
        robot_speed_scale: float = 1.0,    # 安全限幅缩放，0.5 = 更保守
        # ---- 其他可选能力 ----
        enable_search: bool = False,
        enable_local_files: bool = True,
        verbose: bool = True,
        enable_proactive: bool = True,
        proactive_interval_seconds: Optional[float] = None,
        proactive_use_model: bool = True,
    ):
        # ---- 1) DeepSeek 客户端 ----
        key = _resolve_key(deepseek_api_key, "DEEPSEEK_API_KEY")
        if not key:
            raise ValueError(
                "缺少 DeepSeek API Key：请传 deepseek_api_key、"
                "设置环境变量 DEEPSEEK_API_KEY，或在当前目录放 api_key.txt")
        if not isinstance(base_url,str) or not base_url.startswith(('https://','http://')):
            raise ValueError('base_url 必须使用 HTTP/HTTPS，例如 https://api.deepseek.com 或 http://localhost:11434/v1')
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.max_tool_rounds = max_tool_rounds
        self.verbose = verbose
        self._chat_lock = threading.RLock()
        self.proactive = None
        env_proactive = os.environ.get('G1_PROACTIVE_REPORT', '1').strip().lower()
        self.enable_proactive = bool(enable_proactive) and env_proactive not in ('0', 'false', 'no', 'off')
        if proactive_interval_seconds is None:
            try:
                proactive_interval_seconds = float(os.environ.get('G1_PROACTIVE_REPORT_SECONDS', '600'))
            except ValueError:
                proactive_interval_seconds = 600.
        self.proactive_interval_seconds = max(1.0, float(proactive_interval_seconds))
        self.proactive_use_model = bool(proactive_use_model)

        # ---- 2) 工具集（列表 + 名字集合，两者必须一一对应）----
        self.tools: List[dict] = []
        self._tool_names: set = set()
        self.robot: Optional[RobotToolkit] = None
        self.session: Optional[RobotSession] = None

        # ---- 3) 机器人（核心能力）----
        if enable_robot:
            self.session = RobotSession(
                walker_dir=walker_dir,
                show_viewer=robot_viewer,
                realtime=robot_realtime,
                verbose=verbose,
                max_speed_scale=robot_speed_scale,
            )
            self.robot = RobotToolkit(self.session, log=self._log)
            self._register(self.robot.schemas(), self.robot.names())
            self._log("机器人工具已注册：%s" % "、".join(sorted(self.robot.names())))
            from proactive import ProactiveReporter
            self.proactive = ProactiveReporter(
                status_provider=self._training_status,
                output=self._emit_proactive,
                interval_seconds=self.proactive_interval_seconds,
                model_summary=(self._proactive_summary if self.proactive_use_model else None),
                enabled=self.enable_proactive)
            self.robot.proactive_reporter = self.proactive
            self.robot.training.on_job_started = self.proactive.watch
            self._log('主动训练报告：%s（每 %.1f 分钟）' % (
                '开启' if self.enable_proactive else '关闭',
                self.proactive_interval_seconds / 60.0))

        # ---- 4) 联网搜索（可选）----
        self.tavily = None
        from harness_search import resolve_tavily_key
        tkey = resolve_tavily_key(tavily_api_key)
        if enable_search or tkey:
            if not tkey:
                self._log("未提供 Tavily Key，跳过联网搜索工具")
            else:
                from tavily import TavilyClient
                self.tavily = TavilyClient(api_key=tkey)
                self._register([{
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "description": "需要最新信息、实时数据或答案不确定时，联网搜索。",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string",
                                                     "description": "搜索关键词"}},
                            "required": ["query"],
                        },
                    },
                }], {"search_web"})

        if self.robot is not None:
            # 同一个Harness的搜索凭证仅通过进程环境传递，不写入训练配置或技能库。
            self.robot.training.tavily_key=tkey
            self.robot.training.model_settings={'DEEPSEEK_API_KEY':key,'G1_MODEL_BASE_URL':base_url,'G1_MODEL_NAME':model}

        # ---- 5) 本地文件（可选，复用 file_parser）----
        self.enable_local_files = enable_local_files
        if enable_local_files:
            self._register([
                {
                    "type": "function",
                    "function": {
                        "name": "read_local_file",
                        "description": "读取并解析本地文件内容（txt/md/pdf/docx）。",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string",
                                                    "description": "文件完整路径"}},
                            "required": ["path"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "write_text_file",
                        "description": "把文本写入本地文件（自动建目录）。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "文件完整路径"},
                                "content": {"type": "string", "description": "要写入的文本"},
                                "mode": {"type": "string", "enum": ["overwrite", "append"]},
                            },
                            "required": ["path", "content"],
                        },
                    },
                },
            ], {"read_local_file", "write_text_file"})

        # ---- 6) 系统提示词 + 对话历史 ----
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.system_prompt += '\n当用户要求学习某个动作时，可先调用robot_search_action_videos或search_web，再将候选URL传给robot_learn_action；也可直接调用robot_learn_action自动搜索。使用返回job_id查询进度。不得把任务完成、未摔倒或视频标题相关当成动作已学会；必须说明来源、语义未核实、腿部缩幅及评估结果。训练启动后系统默认每10分钟主动发送报告；用户可要求调整间隔、关闭或立即发送，使用robot_proactive_report工具。需要隔离环境、指定宿主机目录训练、训练总结或导出训练回放视频时，使用robot_docker_train、robot_docker_status、robot_docker_summary、robot_docker_export_video、robot_docker_stop工具；启动前确认目录存在，导出后返回宿主机绝对路径。\n本机已接入AMASS G1动作集。涉及AMASS、公开动作数据集或按数据集自主训练时，先调用robot_amass_datasets查看数据集，再调用robot_amass_list按dataset/query分页检索，最后把返回的motion_id传给robot_amass_train启动。不要猜测AMASS文件路径，也不要为每个动作重复扫描磁盘。除用户明确要求Docker且g1-llm-robot:cu128已验证可用外，AMASS训练一律使用backend=local，避免误选不存在的镜像。训练后仍用robot_training_status查询，技能登记和物理评估规则与视频训练相同。\n视频读取有两类，第一类单视频工具robot_learn_video/robot_train_video必须保留，不能因为第二类功能删除或改写。第二类是一个文件夹内含list/list.txt，正向视频命名为a、侧向视频命名为b：先调用robot_multiview_pairs检查配对，再调用robot_multiview_train；默认fallback=true，双视角提取失败时自动降级为a视角单目MediaPipe。优先显式给出a,b两列；单列a时自动寻找同名b。双视角默认以较短有效视频确定共同时间轴，并开启短时缺失关节自修正（self_correct=true，最大修复间隙 repair_gap_seconds=0.25 秒）；超过该间隙的遮挡保持缺失，不得宣称已恢复。\n每个物理动作工具返回的末尾会有[VERIFY]字段：success/outcome、物理证据、失败原因和correction。执行完一个动作必须检查该字段，不能把“工具已返回”当成“目标已达成”。失败后先调用robot_recall_failures检索同类历史修正，再决定重试、缩小动作幅度或重规划；失败记忆只作为建议，当前robot_status才是事实来源。\n每次启动视频、AMASS或Docker训练前，必须先询问用户三项控制参数：全局RL-PD影响因子rl_pd_influence（0~1）、手部收敛因子范围hand_rl_pd_range=[min,max]、脚部收敛因子范围foot_rl_pd_range=[min,max]。不得自行假定、复用上次值或跳过询问；未得到用户明确回答时不得调用训练工具。RL-PD影响因子的含义是：数值越高越偏向预训练RL-PD平衡/默认关节姿态，越低越偏向神经网络与参考动作控制；手部和脚部会分别按各自范围裁剪该因子。'
        self.history: List[dict] = [{"role": "system", "content": self.system_prompt}]

    # ==================================================================
    # 内部：注册工具
    # ==================================================================
    def _register(self, schemas: list, names: set) -> None:
        """把一批工具加进工具集，并记录它们的名字。"""
        self.tools.extend(schemas)
        self._tool_names |= set(names)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print("  · %s" % msg, flush=True)

    # ==================================================================
    # 对外方法
    # ==================================================================
    def chat(self, user_input: str) -> str:
        """发一条消息，返回助手回复（自动维护上下文、自动调用工具）。"""
        if not user_input.strip():
            return ""
        with self._chat_lock:
            self.history.append({"role": "user", "content": user_input})
            return self._run_until_final_text()

    def chat_stream(self, user_input: str) -> Iterator[str]:
        """逐字返回回复（先算完整回复再逐字吐出，做打字机效果）。"""
        for ch in self.chat(user_input):
            yield ch

    def reset(self) -> None:
        """清空对话上下文（机器人状态不动）。"""
        with self._chat_lock:
            self.history = [{"role": "system", "content": self.system_prompt}]

    def close(self) -> None:
        """关闭机器人和连接。"""
        if getattr(self, 'proactive', None) is not None:
            self.proactive.stop()
        if self.robot is not None:
            self.robot.close()
        if self.session is not None:
            self.session.close()

    # ==================================================================
    # 训练计时与主动报告
    # ==================================================================
    def _training_status(self, job_id=None):
        if self.robot is None:
            raise RuntimeError('未启用机器人训练工具')
        return self.robot.training.status(job_id)

    def _emit_proactive(self, text: str) -> None:
        """把主动报告发送到聊天控制台，并写入对话历史。"""
        message = '[DeepSeek 主动消息]\n' + str(text)
        with self._chat_lock:
            self.history.append({'role': 'assistant', 'content': message})
        print('\n\n' + message + '\n', flush=True)

    def _proactive_summary(self, status, report: str) -> str:
        """可选的 DeepSeek 摘要；失败时保留确定性报告。"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': (
                        '你是G1机器人训练监护助手。根据给出的训练报告，用不超过120字中文'
                        '总结进度、耗时、风险和下一步建议。只使用报告中的数据，不要编造。')},
                    {'role': 'user', 'content': report},
                ],
                temperature=0.2,
                timeout=30,
            )
            return (response.choices[0].message.content or '').strip()
        except Exception as exc:
            self._log('主动报告摘要生成失败：%s' % exc)
            return ''

    def timer_text(self) -> str:
        """返回当前训练任务的计时摘要。"""
        if self.robot is None:
            return '未启用机器人，没有训练计时。'
        try:
            status = self.robot.training.status()
        except Exception:
            return '当前没有训练任务。'
        parts = ['任务 %s' % status.get('job_id', '—'),
                 '阶段 %s' % status.get('state', '—'),
                 '已用 %s' % status.get('elapsed_text', '—')]
        if status.get('training_text'):
            parts.append('训练 %s' % status['training_text'])
        if status.get('duration_text'):
            parts.append('总耗时 %s' % status['duration_text'])
        if status.get('eta_text'):
            parts.append('预计剩余 %s' % status['eta_text'])
        if status.get('iteration') is not None:
            parts.append('轮次 %s/%s' % (status.get('iteration'), status.get('total_iterations', '?')))
        return '；'.join(parts)

    def proactive_command(self, argument: str) -> str:
        """处理 /proactive 控制台命令。"""
        if getattr(self, 'proactive', None) is None:
            return '当前会话未启用主动报告。'
        value = (argument or '').strip().lower()
        if value in ('', 'status'):
            info = self.proactive.summary()
            return ('主动报告：%s；间隔 %.1f 分钟；当前任务 %s' % (
                '开启' if info['enabled'] else '关闭', info['interval_minutes'],
                info['job_id'] or '无'))
        if value in ('off', '0', '关闭'):
            self.proactive.configure(enabled=False)
            return '已关闭主动训练报告。'
        if value in ('on', '1', '开启'):
            self.proactive.configure(enabled=True)
            if self.proactive.job_id is None and self.robot is not None:
                try:
                    status = self.robot.training.status()
                    if status.get('job_id') and status.get('state') not in ('completed', 'failed', 'cancelled'):
                        self.proactive.watch(status['job_id'])
                except Exception:
                    pass
            return '已开启主动训练报告。'
        if value in ('now', 'report', '立即'):
            try:
                result = self.proactive.send_now()
                return '已发送主动报告：%s' % (result.get('path') or '仅控制台输出')
            except Exception as exc:
                return '发送主动报告失败：%s' % exc
        try:
            minutes = float(value)
        except ValueError:
            return '用法：/proactive [status|on|off|now|分钟数]'
        if not 0.1 <= minutes <= 1440:
            return '报告间隔应在 0.1~1440 分钟。'
        self.proactive.configure(interval_seconds=minutes * 60.0)
        return '主动报告间隔已设为 %.1f 分钟。' % minutes

    # ==================================================================
    # 3D 窗口的转发接口（模块调用时用）
    # ==================================================================
    #: 说明：构造时用 ``RobotChatBot(robot_viewer=True)`` 就够了；
    #: 下面几个方法用于"运行时再开/关窗口""看完再退出"。

    def open_viewer(self, realtime: Optional[bool] = True) -> str:
        """打开 G1 的 3D 窗口（默认同时切到按真实时间跑，否则看不清）。"""
        if self.session is None:
            return "没有启用机器人，无法显示仿真。"
        return self.session.open_viewer(realtime=realtime)

    def close_viewer(self) -> str:
        """关闭 3D 窗口（仿真继续运行，只是不显示）。"""
        if self.session is None:
            return "没有启用机器人。"
        return self.session.close_viewer()

    @property
    def viewer_running(self) -> bool:
        """3D 窗口是否正开着。"""
        return bool(self.session and self.session.viewer_running)

    def wait_until_viewer_closed(self, timeout: Optional[float] = None) -> str:
        """阻塞到用户关掉 3D 窗口为止（脚本末尾"看完再退出"用）。"""
        if self.session is None:
            return "没有启用机器人。"
        return self.session.wait_until_viewer_closed(timeout=timeout)

    # ==================================================================
    # 核心：调用模型 -> 执行工具 -> 回传结果 -> 直到给出最终答复
    # ==================================================================
    def _ask(self):
        """调用一次模型，把返回的消息对象存进历史并返回。"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            tools=self.tools or None,
            reasoning_effort=self.reasoning_effort if self.thinking else None,
            extra_body={"thinking": {"type": "enabled"}} if self.thinking else None,
        )
        msg = resp.choices[0].message
        self.history.append(self._to_dict(msg))     # 存历史，保持上下文
        return msg

    def _run_until_final_text(self) -> str:
        """执行至模型最终答复；训练/GIF工具出现后本次对话不限工具轮数。"""
        from tool_output import TRAINING_TOOLS
        rounds = 0
        unlimited = self.max_tool_rounds is None
        while unlimited or rounds < self.max_tool_rounds:
            msg = self._ask()
            if not msg.tool_calls:
                return msg.content or ""
            if any(call.function.name in TRAINING_TOOLS for call in msg.tool_calls):
                if not unlimited:
                    self._log('训练/导出工具已启用：本次工具调用循环不设轮数上限。')
                unlimited = True
            for tool_call in msg.tool_calls:
                self._dispatch_tool(tool_call)
            rounds += 1
        return ("（已达到普通工具调用轮次 %d 轮。训练工具调用不受此限制。）" % self.max_tool_rounds)

    def _dispatch_tool(self, tool_call) -> None:
        """根据模型要求的工具名，执行对应函数并把结果以 role=tool 回传。"""
        name = tool_call.function.name

        # 解析参数（模型给的是 JSON 字符串）
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}

        # ---- 机器人工具走 RobotToolkit ----
        if name in self._tool_names and name.startswith("robot_") and self.robot:
            result = self.robot.dispatch(name, args)
        elif name == "search_web":
            result = self._search(args.get("query", ""))
        elif name == "read_local_file":
            result = self._read_file(args.get("path", ""))
        elif name == "write_text_file":
            result = self._write_file(args.get("path", ""), args.get("content", ""),
                                      args.get("mode", "overwrite"))
        else:
            result = "未知工具：%s" % name

        from tool_output import format_result
        if self.verbose:
            print(format_result(name, result), flush=True)

        # 关键：tool_call_id 必须和模型要求调用的 id 对应，模型才能对上号
        self.history.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result),
        })

    @staticmethod
    def _to_dict(msg) -> dict:
        """把 SDK 返回的消息对象转成普通字典，方便再次传给 API。"""
        if hasattr(msg, "model_dump"):                 # pydantic v2
            d = msg.model_dump(exclude_none=True)
        else:
            d = msg.dict(exclude_none=True)
        # 思考模式下带 tools 的请求必须回传 reasoning_content，否则 API 返回 400。
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning and "reasoning_content" not in d:
            d["reasoning_content"] = reasoning
        return d

    # ==================================================================
    # 可选工具的实现
    # ==================================================================
    def _search(self, query: str) -> str:
        """联网搜索（Tavily）。失败时降级为提示，不让流程崩。"""
        try:
            from harness_search import search_web,format_search
            return format_search(search_web(query,client=self.tavily))
        except Exception as exc:
            return '联网搜索失败：%s' % exc

    def _read_file(self, path: str) -> str:
        """读本地文件（复用 file_parser，支持 pdf/docx）。"""
        try:
            from file_parser import parse_file
        except ImportError:
            return "缺少 file_parser 模块，无法解析文件。"
        text = parse_file(path)
        if len(text) > 8000:
            text = text[:8000] + "\n...(内容过长，已截断)"
        return text

    def _write_file(self, path: str, content: str, mode: str = "overwrite") -> str:
        """写本地文本文件。"""
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            if mode == "append":
                with p.open("a", encoding="utf-8") as f:
                    f.write(content)
            else:
                p.write_text(content, encoding="utf-8")
            return "已写入 %s（%d 字符）" % (p, len(content))
        except Exception as exc:
            return "写入失败：%s" % exc

    # ==================================================================
    # 命令行交互
    # ==================================================================
    def run_cli(self) -> None:
        """反复提问，直到输入退出关键字。"""
        print("=" * 62)
        print("DeepSeek + 宇树 G1 机器人 控制台")
        print("试着说：往前走一米 / 走到(1,0)再转身 / 举手挥一下 / 停下")
        print("输入 exit / quit / 退出 / q 结束")
        print("特殊命令：/status 机器人状态   /timer 训练计时   /proactive [on|off|now|分钟] 主动报告")
        print("          /report 动作历史   /new 清空对话   /tools 列工具")
        print("=" * 62 + "\n")

        while True:
            try:
                question = input("你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n对话结束。")
                break

            if question.lower() in ("exit", "quit", "退出", "q"):
                print("对话结束，再见！")
                break
            if not question:
                continue

            # ---- 特殊命令 ----
            if question.lower() == "/status":
                print(self.session.describe() + "\n" if self.session else "未启用机器人。\n")
                continue
            if question.lower() == "/report":
                if self.session:
                    for line in (self.session._action_log or ["（还没有动作）"]):
                        print("   ", line)
                print()
                continue
            if question.lower() == "/timer":
                print(self.timer_text() + "\n")
                continue
            if question.lower().startswith("/proactive"):
                print(self.proactive_command(question[len("/proactive"):]) + "\n")
                continue
            if question.lower() == "/new":
                self.reset()
                print("已清空对话上下文。\n")
                continue
            if question.lower() == "/tools":
                for t in self.tools:
                    print("  - %s" % t["function"]["name"])
                print()
                continue

            # ---- 正常对话 ----
            print("助手：", end="", flush=True)
            for ch in self.chat_stream(question):
                print(ch, end="", flush=True)
            print("\n")


# ======================================================================
# 自测：不连大模型，直接按脚本调用工具，验证"机器人链路"通不通
# ======================================================================
def selftest(args) -> int:
    """把模型**本该生成**的那串工具调用，硬编码跑一遍。

    这能验证：工具声明 -> 参数解析 -> RobotSession -> 仿真执行 -> 中文回报
    这整条链路是否正常，从而把"大模型不听话"和"机器人不听话"两类问题分开。
    """
    print("=" * 62)
    print("自测：用脚本模拟大模型的工具调用（不连 DeepSeek）")
    print("=" * 62)

    session = RobotSession(walker_dir=args.walker_dir,
                           show_viewer=getattr(args, "viewer", False),
                           verbose=True)
    tk = RobotToolkit(session, log=lambda m: print("  ->", m))

    # (工具名, 参数) —— 等价于模型返回的一串 tool_calls
    script = [
        ("robot_status", {}),
        ("robot_goto", {"x": 1.0, "y": 0.0, "yaw": 0}),
        ("robot_set_arm", {"preset": "双手上举"}),
        ("robot_turn", {"degrees": 90}),
        ("robot_sequence", {"steps": [
            {"type": "arm", "preset": "自然下垂（推荐）"},
            {"type": "goto", "x": 0.0, "y": 0.0, "yaw": 90},
            {"type": "arm", "side": "left", "joints": {"shoulder_pitch": -1.6}},
            {"type": "stop"},
        ]}),
        ("robot_status", {}),
    ]

    for name, params in script:
        print("\n>>> 模型调用：%s(%s)" % (name, json.dumps(params, ensure_ascii=False)))
        print(tk.dispatch(name, params))

    print("\n" + "=" * 62)
    print("自测完成。动作历史：")
    for line in session._action_log:
        print("   ", line)
    session.close()
    return 0


# ======================================================================
# 程序入口
# ======================================================================
def build_parser() -> "argparse.ArgumentParser":
    import argparse

    ap = argparse.ArgumentParser(
        prog="deepseek_robot",
        description="DeepSeek 对话 + 宇树 G1 机器人控制（Function Calling）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ---- 模型侧 ----
    ap.add_argument("--key", default=None, help="DeepSeek API Key（默认读环境变量/文件）")
    ap.add_argument("--tavily-key", default=None, help="Tavily Key，给了才开联网搜索")
    ap.add_argument("--model", default="deepseek-flash", help="对话模型")
    ap.add_argument("--base-url", default="https://api.deepseek.com", help="DeepSeek HTTPS API地址")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--search", action="store_true", help="启用联网搜索工具")
    ap.add_argument("--no-files", action="store_true", help="关闭本地文件工具")
    ap.add_argument("--quiet", action="store_true", help="不打印工具调用日志")

    # ---- 机器人侧 ----
    ap.add_argument("--viewer", action="store_true", help="打开 G1 的 3D 窗口")
    ap.add_argument("--realtime", action="store_true",
                    help="按真实时间跑（默认尽量快跑，仿真秒 ≠ 墙上秒）")
    ap.add_argument("--walker-dir", default=None,
                    help="g1_walkfree 包所在目录（默认自动查找）")
    ap.add_argument("--speed-scale", type=float, default=1.0,
                    help="安全限幅缩放，0.5 = 更保守")
    ap.add_argument("--no-robot", action="store_true", help="不启用机器人（纯聊天）")
    ap.add_argument("--proactive-interval-minutes", type=float, default=None,
                    help="训练主动报告间隔（分钟），默认10")
    ap.add_argument("--no-proactive", action="store_true", help="关闭训练主动报告")
    ap.add_argument("--proactive-no-model", action="store_true",
                    help="主动报告只发送结构化数据，不额外调用DeepSeek摘要")

    # ---- 运行方式 ----
    ap.add_argument("--task", default=None, help="单条指令，执行完就退出")
    ap.add_argument("--selftest", action="store_true",
                    help="不连大模型，脚本化跑一遍机器人工具")
    ap.add_argument("--list-tools", action="store_true", help="打印所有工具声明后退出")
    return ap


def main(argv: Optional[list] = None) -> int:
    import sys
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    args = build_parser().parse_args(argv)

    # ---- 只打印工具声明 ----
    if args.list_tools:
        from robot_tools import robot_tool_schemas
        from video_motion.tools import schemas as video_schemas
        from fullbody.tools import schemas as training_schemas
        print(json.dumps(RobotToolkit(None).schemas(), ensure_ascii=False, indent=2))
        return 0

    # ---- 自测（不连模型）----
    if args.selftest:
        return selftest(args)

    # ---- 正常模式：连 DeepSeek ----
    try:
        bot = RobotChatBot(
            deepseek_api_key=args.key,
            tavily_api_key=args.tavily_key,
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            enable_search=args.search,
            enable_local_files=not args.no_files,
            enable_robot=not args.no_robot,
            robot_viewer=args.viewer,
            robot_realtime=True if args.realtime else (True if args.viewer else None),
            walker_dir=args.walker_dir,
            robot_speed_scale=args.speed_scale,
            verbose=not args.quiet,
            enable_proactive=not args.no_proactive,
            proactive_interval_seconds=(args.proactive_interval_minutes * 60.0
                                        if args.proactive_interval_minutes is not None else None),
            proactive_use_model=not args.proactive_no_model,
        )
    except Exception as exc:
        print("[启动失败] %s" % exc)
        return 2

    try:
        if args.task:
            print("任务：%s\n" % args.task)
            print(bot.chat(args.task))
        else:
            bot.run_cli()
    finally:
        bot.close()
    return 0


if __name__ == "__main__":
    from viewer_camera import finish_process
    finish_process(main())
