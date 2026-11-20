"""把 G1 的能力声明成 Function Calling 工具，供大模型调用。

这一层很薄，只做三件事：
    1. 用 JSON Schema 描述"机器人会做什么、参数是什么"（``ROBOT_TOOL_SCHEMAS``）
    2. 把模型给的参数原样转交给 :class:`~robot_bridge.RobotSession`
    3. 把返回值整理成字符串回传（tool 消息的 content 必须是字符串）

设计要点
--------
* **描述写得像给新同事的交接文档**：模型完全靠 description 决定用哪个工具、怎么传参。
  所以每个工具都写清楚"什么时候用"、"参数单位"、"正负方向"。
* **返回也写得像人话**：模型拿到的是"已行走 2.0 秒…位移 Δ=(+0.78, -0.11) m"，
  它才能据此做闭环决策（"还差 0.2 米，再走一次"）。
* **危险动作单列**：``robot_stop`` 描述里明确写了"任何时候都可以调用"，
  保证模型在异常时有个急停抓手。
"""

from __future__ import annotations

import traceback
import math
from app_logging import configure_logging
logger=configure_logging()
import failure_memory


from robot_bridge import RobotSession, available_arm_presets


# ======================================================================
# 一、工具声明（JSON Schema）
# ======================================================================
#: 手臂预设的名字是动态的（来自 config.ARM_PRESETS），所以枚举也动态生成
_PRESET_NAMES = None


def _preset_names() -> list:
    global _PRESET_NAMES
    if _PRESET_NAMES is None:
        try:
            _PRESET_NAMES = available_arm_presets()
        except Exception:
            _PRESET_NAMES = []
    return _PRESET_NAMES


def robot_tool_schemas() -> list:
    """返回机器人工具集（OpenAI Function Calling 格式）。

    每次调用都重新构造，保证手臂预设枚举与当前工程一致。
    """
    presets = _preset_names()

    return [
        # ---------- 1. 查询状态 ----------
        {
            "type": "function",
            "function": {
                "name": "robot_status",
                "description": (
                    "查询 G1 人形机器人的当前状态：位置坐标、速度、朝向、是否摔倒、"
                    "仿真时间。**在动手之前或做完动作之后都应该先看一眼状态**，"
                    "这样才能知道现在在哪、上一步动作有没有生效。"
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },

        # ---------- 2. 按速度行走 ----------
        {
            "type": "function",
            "function": {
                "name": "robot_walk",
                "description": (
                    "让机器人按指定速度走一段**固定时长**，到时自动刹停。"
                    "适合'往前走两秒''边转边走一会儿'这类指令。"
                    "如果要走到某个**具体地点**，请改用 robot_goto。\n"
                    "速度上限：vx±0.8 m/s、vy±0.45 m/s、vyaw±1.0 rad/s，超了会被自动夹住。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "vx": {"type": "number",
                               "description": "前进速度 m/s，正数=身体前方，负数=后退"},
                        "vy": {"type": "number",
                               "description": "左右平移 m/s，正数=身体左侧（横移）"},
                        "vyaw": {"type": "number",
                                 "description": "转向角速度 rad/s，正数=左转（逆时针）"},
                        "seconds": {"type": "number",
                                    "description": "行走时长（仿真秒），建议 0.5~10"},
                    },
                    "required": ["seconds"],
                },
            },
        },

        # ---------- 3. 闭环走到坐标 ----------
        {
            "type": "function",
            "function": {
                "name": "robot_goto",
                "description": (
                    "让机器人**闭环走到指定的平面坐标**（世界系，单位 m，以机器人出发位置为原点）。"
                    "内部每 0.1 秒读一次实际位姿、把误差转成速度指令，所以能自动纠正漂移，"
                    "比 robot_walk 精确得多。适合'走到(1,0)''回到原点''走到斜前方'。\n"
                    "可以同时指定目标朝向。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "目标 x 坐标 m，正数=出发时的前方"},
                        "y": {"type": "number", "description": "目标 y 坐标 m，正数=出发时的左侧"},
                        "yaw": {"type": "number",
                                "description": "目标朝向（度），0=出发时的朝向，正数=左转。不填则不管朝向"},
                        "timeout": {"type": "number",
                                    "description": "最长允许的仿真秒数，默认 20"},
                    },
                    "required": ["x", "y"],
                },
            },
        },

        # ---------- 4. 原地转向 ----------
        {
            "type": "function",
            "function": {
                "name": "robot_turn",
                "description": (
                    "原地转身指定角度，不移动位置（会有一点位置漂移，属正常）。"
                    "正数=左转（逆时针），负数=右转。例如'向右转90度'传 -90。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "degrees": {"type": "number", "description": "转向角度（度），正=左转"},
                        "timeout": {"type": "number", "description": "最长仿真秒数，默认 10"},
                    },
                    "required": ["degrees"],
                },
            },
        },

        # ---------- 5. 手臂 ----------
        {
            "type": "function",
            "function": {
                "name": "robot_set_arm",
                "description": (
                    "设置手臂姿态。两种用法二选一：\n"
                    "  (a) 用 preset 套用预设姿态，例如 '双手上举'、'张开双臂'、'双手前平举'；\n"
                    "  (b) 用 joints 直接给关节角（弧度），例如 {\"shoulder_pitch\": -0.95, \"elbow\": 0.35}。\n"
                    "⚠ 关键：shoulder_pitch 的轴向决定了 **负值才是手臂向前**，正值是向后（与直觉相反）！\n"
                    "想'举手/挥手'优先用 preset，比手算关节角可靠。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "preset": {
                            "type": "string",
                            "enum": presets or ["自然下垂（推荐）"],
                            "description": "预设姿态名（推荐优先使用）",
                        },
                        "side": {
                            "type": "string",
                            "enum": ["left", "right", "both"],
                            "description": "改哪只手臂，默认 both（双臂）",
                        },
                        "joints": {
                            "type": "object",
                            "description": (
                                "关节角（弧度），键为关节名。可用：shoulder_pitch（负=向前）、"
                                "shoulder_roll（外张，正=张开）、shoulder_yaw、elbow（正=弯曲）、"
                                "wrist_roll、wrist_pitch、wrist_yaw"
                            ),
                            "additionalProperties": {"type": "number"},
                        },
                    },
                    "required": [],
                },
            },
        },

        # ---------- 6. 腰部 ----------
        {
            "type": "function",
            "function": {
                "name": "robot_set_waist",
                "description": (
                    "设置腰部三个关节（弧度）。yaw=上半身左右转，roll=侧倾，pitch=前后俯仰"
                    "（正=前倾）。不填的关节保持原值。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "yaw": {"type": "number", "description": "腰部左右转（弧度）"},
                        "roll": {"type": "number", "description": "腰部侧倾（弧度）"},
                        "pitch": {"type": "number", "description": "腰部前后俯仰（弧度），正=前倾"},
                    },
                    "required": [],
                },
            },
        },

        # ---------- 7. 组合动作序列 ----------
        {
            "type": "function",
            "function": {
                "name": "robot_sequence",
                "description": (
                    "一次下发**多个动作按顺序执行**（比如'先走到A点，再举手，再转个圈'）。"
                    "比连续调用多个工具更省往返、时序也更准。\n"
                    "每步是一个对象，type 可选："
                    "walk{vx,vy,vyaw,seconds}、goto{x,y,yaw}、turn{degrees}、"
                    "arm{preset} 或 arm{side,joints}、waist{yaw,roll,pitch}、"
                    "wait{seconds}、stop、reset。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "description": "动作列表，按顺序执行",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["walk", "goto", "turn", "arm",
                                                 "waist", "wait", "stop", "reset"],
                                    },
                                },
                                "required": ["type"],
                            },
                        },
                    },
                    "required": ["steps"],
                },
            },
        },

        # ---------- 8. 停止 ----------
        {
            "type": "function",
            "function": {
                "name": "robot_stop",
                "description": (
                    "立即让机器人刹停（清空速度指令，机器人原地站住）。"
                    "**任何时候都可以调用**，包括察觉不对、想打断动作、"
                    "或者用户说'停''别动'的时候。"
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },

        # ---------- 9. 复位 ----------
        {
            "type": "function",
            "function": {
                "name": "robot_reset",
                "description": (
                    "把机器人复位到站立姿态、位置归零。**摔倒了必须调用它才能继续**，"
                    "状态里显示'已摔倒'时就用这个。"
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        # ---------- 10. 记录动作参数并绘图 ----------
        {
            "type": "function",
            "function": {
                "name": "robot_log_motion",
                "description": (
                    "让机器人执行一个动作（walk 行走 / goto 走到坐标 / turn 转向），"
                    "同时记录骨盆的位置、速度、加速度，执行完自动保存 CSV 数据并绘制图谱 PNG。"
                    "返回数据文件路径和图谱路径。适合用户想查看某次动作的运动曲线时使用。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["walk", "goto", "turn"],
                                   "description": "要记录的动作类型"},
                        "vx": {"type": "number", "description": "前进速度 m/s（walk）"},
                        "vy": {"type": "number", "description": "左右平移 m/s（walk）"},
                        "vyaw": {"type": "number", "description": "转向角速度 rad/s（walk）"},
                        "seconds": {"type": "number", "description": "行走时长秒（walk）"},
                        "x": {"type": "number", "description": "目标 x 坐标（goto）"},
                        "y": {"type": "number", "description": "目标 y 坐标（goto）"},
                        "degrees": {"type": "number", "description": "转向角度（turn）"},
                        "output": {"type": "string", "description": "日志文件路径，默认 motions/log_motion.csv"},
                    },
                    "required": ["action"],
                },
            },
        },
        # ---------- 11. 创建文件夹 ----------
        {
            "type": "function",
            "function": {
                "name": "robot_create_folder",
                "description": (
                    "在 outputs 目录下创建一个新文件夹，用于存放训练日志等输出。"
                    "默认按当前时间戳命名（YYYYMMDD_HHMMSS），也可用 name 指定名称。"
                    "返回文件夹路径。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "文件夹名，省略则用当前时间戳"},
                        "parent": {"type": "string", "description": "父目录，默认 outputs"},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "robot_recall_failures",
                "description": (
                    "检索历史动作失败及其可复用修正。每次重试 goto/walk/turn 前，"
                    "尤其刚发生过摔倒或未到达时，先调用此工具；返回失败原因、"
                    "物理证据和建议修正，不替代当前 robot_status。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string", "description": "可选工具名，如 robot_goto"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "required": [],
                },
            },
        },
    ]


# ======================================================================
# 二、工具分发器
# ======================================================================
class RobotToolkit:
    """把 :class:`RobotSession` 包装成"名字 -> 调用"的分发表。

    用法::

        with RobotSession() as bot:
            tk = RobotToolkit(bot)
            tk.schemas()                  # 交给 OpenAI 的 tools 参数
            tk.names()                    # {'robot_walk', ...} 供主循环判断
            tk.dispatch("robot_walk", {"vx": 0.5, "seconds": 2})
    """

    def __init__(self, session: RobotSession, log: Optional[Callable[[str], None]] = None):
        self.session = session
        self._log = log or (lambda _m: None)
        self.failure_memory = failure_memory
        self._action_tools = {
            'robot_walk', 'robot_goto', 'robot_turn', 'robot_sequence',
            'robot_set_arm', 'robot_set_waist', 'robot_stop', 'robot_reset',
            'robot_play_motion', 'robot_log_motion',
        }

        # 名字 -> 处理函数（参数是已解析好的 dict，返回值是字符串）
        from fullbody.jobs import TrainingJobs
        self.training = TrainingJobs()
        from fullbody.docker_jobs import DockerJobs
        self.docker = DockerJobs()
        from coordination.tools import CoordinationJobs
        self.coordination=CoordinationJobs(session)
        self.proactive_reporter=None
        self._handlers: dict[str, Callable[[dict], str]] = {
            "robot_search_action_videos": lambda args: self._training_result(self.training.search_videos(**args)),
            "robot_learn_action": lambda args: self._training_result(self.training.learn_action(**self._require_rl_pd(args))),
            "robot_amass_datasets": lambda args: self._training_result(self.training.amass_datasets(args.get('refresh',False))),
            "robot_amass_list": lambda args: self._training_result(self.training.amass_list(**args)),
            "robot_amass_train": lambda args: self._training_result(self._amass_train(args)),
            "robot_multiview_pairs": lambda args: self._training_result(self.training.multiview_pairs(**args)),
            "robot_multiview_train": lambda args: self._training_result(self.training.start_multiview(**self._require_rl_pd(args))),
            "robot_jump_handoff": lambda args: self._training_result(self.coordination.handoff(args['task_id'])),
            "robot_jump_close_window": lambda args: self._training_result(self.coordination.close_window(args['task_id'])),
            "robot_coordination_start": lambda args: self._training_result(self.coordination.start(**args)),
            "robot_coordination_status": lambda args: self._training_result(self.coordination.status(args['task_id'])),
            "robot_spatial_skill_list": lambda args: self._training_result(self.training.spatial_skills_list()),
            "robot_spatial_skill_get": lambda args: self._training_result(self.training.spatial_skill_get(args['skill_id'])),
            "robot_spatial_skill_run": lambda args: self._training_result(self.training.run_spatial_skill(args['skill_id'],args.get('mode','spatial'),args.get('viewer',True))),
            "robot_skill_list": lambda args: self._training_result(self.training.skills_list()),
            "robot_skill_get": lambda args: self._training_result(self.training.skill_get(args['skill_id'])),
            "robot_skill_run": lambda args: self._training_result(self.training.run_skill(args['skill_id'],args.get('viewer',True))),
            "robot_export_gif": lambda args: self._training_result(self.training.export_gif(**args)),
            "robot_gif_status": lambda args: self._training_result(self.training.gif_status(args['export_id'])),
            "robot_train_video": lambda args: self._training_result(self.training.start(**self._require_rl_pd(args))),
            "robot_training_status": lambda args: self._training_result(self.training.status(args.get('job_id'))),
            "robot_proactive_report": self._proactive_report,
            "robot_training_stop": lambda args: self._training_result(self.training.stop(args.get('job_id'))),
            "robot_training_viewer": lambda args: self._training_result(self.training.open_viewer(args.get('job_id'))),
            "robot_docker_train": lambda args: self._training_result(self.docker.start(**self._require_rl_pd(args))),
            "robot_docker_status": lambda args: self._training_result(self.docker.status(args.get('job_id'))),
            "robot_docker_summary": lambda args: self._training_result(self.docker.summary(**args)),
            "robot_docker_export_video": lambda args: self._training_result(self.docker.export_video(**args)),
            "robot_docker_stop": lambda args: self._training_result(self.docker.stop(args.get('job_id'))),
            "robot_status": self._status,
            "robot_walk": self._walk,
            "robot_goto": self._goto,
            "robot_turn": self._turn,
            "robot_set_arm": self._set_arm,
            "robot_set_waist": self._set_waist,
            "robot_sequence": self._sequence,
            "robot_stop": self._stop,
            "robot_learn_video": self._learn_video,
            "robot_play_motion": self._play_motion,
            "robot_reset": self._reset,
            "robot_log_motion": self._log_motion,
            "robot_create_folder": self._create_folder,
            "robot_recall_failures": self._recall_failures,
        }

    # ---- 给主循环用的元信息 ----
    def schemas(self) -> list:
        from video_motion.tools import schemas
        from fullbody.tools import schemas as training_schemas
        from fullbody.tools_spatial import schemas as spatial_schemas
        from coordination.tools import schemas as coordination_schemas
        return robot_tool_schemas() + schemas() + training_schemas() + spatial_schemas() + coordination_schemas()

    def names(self) -> set:
        return set(self._handlers)

    # ---- 统一入口 ----
    def dispatch(self, name: str, args: Any) -> str:
        """执行一个工具调用，返回要回传给模型的字符串。

        这里对**所有异常兜底**：机器人崩了也不能让整个对话流程挂掉，
        而是把错误当结果回传，让模型知道发生了什么、可以换个做法再试。
        """
        handler = self._handlers.get(name)
        if handler is None:
            return "未知的机器人工具：%s。可用工具：%s" % (
                name, "、".join(sorted(self._handlers)))

        if not isinstance(args, dict):
            args = {}

        self._log("调用 %s(%s)" % (name, _fmt_args(args)))
        before = None
        if name in self._action_tools and self.session is not None:
            try:
                value = self.session.status()
                before = value if isinstance(value, dict) else None
            except Exception:
                before = None
        try:
            if name in {'robot_walk','robot_goto','robot_turn','robot_sequence','robot_set_arm','robot_set_waist','robot_stop'}:
                self.coordination.before_action()
            result = handler(args)
            if name in self._action_tools:
                after = None
                if self.session is not None:
                    try:
                        value = self.session.status()
                        after = value if isinstance(value, dict) else None
                    except Exception:
                        after = None
                verification = self._verify_action(name, args, before, after, result)
                if verification is not None:
                    if verification.get('outcome') != 'success':
                        self.failure_memory.record(
                            name, args, verification.get('outcome', 'failed'),
                            verification.get('reason', 'postcondition_failed'),
                            verification.get('evidence', {}),
                            verification.get('correction', ''),
                            before, after, result)
                    result = result + "\n[VERIFY] " + __import__('json').dumps(
                        verification, ensure_ascii=False)
        except Exception as exc:                       # noqa: BLE001
            logger.exception('机器人工具 %s 执行失败，参数=%r',name,args)
            result = "机器人工具 %s 执行出错：%s（详细堆栈已写入 logs/app.log）" % (name, exc)
            if name in self._action_tools and isinstance(before, dict):
                evidence = {'error': str(exc), 'exception': type(exc).__name__}
                correction = self._correction_for('execution_error')
                self.failure_memory.record(
                    name, args, 'failed', 'execution_error', evidence,
                    correction, before, None, result)
                result += "\n[VERIFY] " + __import__('json').dumps({
                    'tool': name, 'success': False, 'outcome': 'failed',
                    'reason': 'execution_error', 'evidence': evidence,
                    'correction': correction}, ensure_ascii=False)
        self._log("返回：%s" % str(result).replace("\n", " ")[:160])
        return str(result)

    def _recall_failures(self, args):
        records = self.failure_memory.retrieve(
            tool=args.get('tool'), limit=int(args.get('limit') or 5))
        return __import__('json').dumps({
            'count': len(records),
            'failures': records,
        }, ensure_ascii=False)

    @staticmethod
    def _correction_for(reason):
        return {
            'fallen': '先复位；缩短动作时长、降低速度/步幅，再分段重试并检查质心',
            'goal_not_reached': '先 robot_status 确认位置；增大 timeout 或拆成两段 goto，避免持续饱和',
            'turn_not_reached': '降低转动角速度，分段转向并重新测量朝向',
            'motion_underachieved': '目标位移不足；检查限幅、地面阻力和时长，先小步验证',
            'execution_error': '检查参数和前置状态；复位后以更保守参数重试',
            'postcondition_failed': '执行结果不等于目标达成；重新测量状态后再决定重试或重规划',
        }.get(reason, '重新读取状态，缩小动作幅度后重试')

    def _verify_action(self, name, args, before, after, result):
        if not isinstance(before, dict) or not isinstance(after, dict):
            return None
        evidence = {
            'before': {key: before.get(key) for key in
                       ('x', 'y', 'yaw', 'z', 'speed', 'fallen')},
            'after': {key: after.get(key) for key in
                      ('x', 'y', 'yaw', 'z', 'speed', 'fallen')},
            'reliability': after.get('reliability'),
        }
        outcome, reason = 'success', ''
        if after.get('fallen'):
            outcome, reason = 'failed', 'fallen'
        elif name == 'robot_goto':
            target_x = float(args.get('x', before.get('x', 0.0)))
            target_y = float(args.get('y', before.get('y', 0.0)))
            distance = math.hypot(target_x - after.get('x', 0.0),
                                  target_y - after.get('y', 0.0))
            yaw_arg = args.get('yaw')
            yaw_error = None
            if yaw_arg is not None:
                yaw_error = abs((float(yaw_arg) - after.get('yaw', 0.0) + 180.0) % 360.0 - 180.0)
            evidence.update(target=[target_x, target_y], distance_m=distance,
                            yaw_error_deg=yaw_error)
            if distance > 0.10 or (yaw_error is not None and yaw_error > 8.0):
                outcome, reason = 'partial', 'goal_not_reached'
        elif name == 'robot_turn':
            target = before.get('yaw', 0.0) + float(args.get('degrees', 0.0))
            error = abs((target - after.get('yaw', 0.0) + 180.0) % 360.0 - 180.0)
            evidence['yaw_error_deg'] = error
            if error > 6.0:
                outcome, reason = 'partial', 'turn_not_reached'
        elif name == 'robot_walk':
            seconds = float(args.get('seconds', 1.0))
            vx, vy = float(args.get('vx', 0.0)), float(args.get('vy', 0.0))
            yaw = math.radians(float(before.get('yaw', 0.0)))
            ex = math.cos(yaw) * vx * seconds - math.sin(yaw) * vy * seconds
            ey = math.sin(yaw) * vx * seconds + math.cos(yaw) * vy * seconds
            expected = math.hypot(ex, ey)
            actual_x = after.get('x', 0.0) - before.get('x', 0.0)
            actual_y = after.get('y', 0.0) - before.get('y', 0.0)
            actual = math.hypot(actual_x, actual_y)
            evidence.update(expected_displacement_m=expected,
                            actual_displacement_m=actual)
            if expected > 0.05 and actual < 0.35 * expected:
                outcome, reason = 'partial', 'motion_underachieved'
        elif '摔倒' in str(result) or '执行出错' in str(result):
            outcome, reason = 'failed', 'execution_error'
        hints = self.failure_memory.summary(tool=name, limit=2)
        correction = hints[0].get('correction') if hints else ''
        if outcome == 'success':
            correction = '动作后条件满足；可以推进下一步'
        else:
            correction = correction or self._correction_for(reason or 'postcondition_failed')
        return {
            'tool': name, 'success': outcome == 'success', 'outcome': outcome,
            'reason': reason or 'postcondition_satisfied',
            'evidence': evidence, 'correction': correction,
            'prior_corrections': hints,
        }

    # ==================================================================
    # 各工具的具体实现（都很薄：转调 RobotSession）
    # ==================================================================
    @staticmethod
    def _training_result(result):
        import json
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _require_rl_pd(args):
        values=dict(args)
        if (values.get('rl_pd_influence') is None or values.get('hand_rl_pd_range') is None
                or values.get('foot_rl_pd_range') is None):
            raise ValueError('启动训练前必须向用户询问RL-PD影响因子、手部和脚部收敛因子范围[min,max]')
        return values

    def _amass_train(self,args):
        from pathlib import Path
        from fullbody.amass import resolve_motion
        values=dict(args)
        backend=values.pop('backend','local')
        if (values.get('rl_pd_influence') is None or values.get('hand_rl_pd_range') is None
                or values.get('foot_rl_pd_range') is None):
            raise ValueError('启动AMASS训练前必须向用户询问RL-PD影响因子、手部和脚部收敛因子范围[min,max]')
        motion=resolve_motion(
            motion_id=values.pop('motion_id',None),
            dataset=values.pop('dataset',None),
            name=values.pop('name',None),
            path=values.pop('path',None),
            query=values.pop('query',None))
        skill_name=values.pop('skill_name',None) or motion['name']
        if backend=='docker':
            allowed={'iterations','device','num_envs','leg_weight','arm_weight',
                     'motion_gain','skill_name','coordination','joint_assist',
                     'joint_assist_gain','airborne_reward','root_mode',
                     'curriculum_target','max_episode_seconds','rl_pd_influence',
                     'hand_rl_pd_range','foot_rl_pd_range'}
            kwargs={key:value for key,value in values.items() if key in allowed}
            return self.docker.start(
                directory=str(Path(motion['path']).parent),
                reference=Path(motion['path']).name,
                skill_name=skill_name,**kwargs)
        allowed={'iterations','device','render','num_envs','leg_weight',
                 'arm_weight','motion_gain','coordination','jump_mode',
                 'jump_times','auto_adjust','model_review','curriculum_target',
                 'max_episode_seconds','joint_assist','joint_assist_gain',
                 'airborne_reward','root_mode','rl_pd_influence',
                 'hand_rl_pd_range','foot_rl_pd_range'}
        kwargs={key:value for key,value in values.items() if key in allowed}
        return self.training.start_reference(
            motion['path'],skill_name=skill_name,dataset=motion['dataset'],
            source_motion=motion,**kwargs)

    def close(self):
        self.training.close()
        self.docker.close()

    def _learn_video(self, args: dict) -> str:
        import json
        from video_motion.pose import extract
        from video_motion.retarget import play
        last_percent=[-1]
        def detect_progress(frame,total):
            percent=int(round(100.0*frame/max(1,total)))
            if frame==total or percent>=last_percent[0]+10:
                last_percent[0]=percent
                self._log('骨架检测 %d/%d（%d%%）'%(frame,total,percent))
        def smplx_progress(frame,total,error):
            if frame==0:
                self._log('SMPL-X 拟合中…')
            elif frame==1 or frame==total or frame%20==0:
                self._log('SMPL-X %d/%d'%(frame,total))
        path = extract(args['video'], args.get('output', 'motions/video_pose.npz'),
                       max_seconds=float(args.get('max_seconds', 30)),
                       detect_progress=detect_progress, progress=smplx_progress)
        result = {'motion_file': str(path), 'extracted': True}
        if args.get('play', True):
            result['playback'] = play(self.session, path)
        return json.dumps(result, ensure_ascii=False)

    def _play_motion(self, args: dict) -> str:
        import json
        from video_motion.retarget import play
        return json.dumps(play(self.session, args['motion'], float(args.get('speed', 1))), ensure_ascii=False)

    def _status(self, _args: dict) -> str:
        text = self.session.describe()
        hints = self.failure_memory.summary(limit=3)
        if hints:
            text += "\n  历史失败修正：" + "；".join(
                '%s→%s' % (item.get('tool'), item.get('correction'))
                for item in hints)
        return text

    def _proactive_report(self, args: dict) -> str:
        import json
        reporter = getattr(self, 'proactive_reporter', None)
        if reporter is None:
            return json.dumps({'enabled': False,
                               'message': '当前会话未启用主动训练报告'}, ensure_ascii=False)
        job_id = args.get('job_id')
        if not job_id and (args.get('enabled') is True or args.get('send_now')):
            try: job_id = self.training.status().get('job_id')
            except Exception: job_id = None
        interval = args.get('interval_minutes')
        result = reporter.configure(
            enabled=args.get('enabled'),
            interval_seconds=(float(interval) * 60.0 if interval is not None else None),
            job_id=job_id)
        if args.get('send_now'):
            try:
                result['sent_report'] = reporter.send_now(job_id)
            except Exception as exc:
                result['send_error'] = str(exc)
        return json.dumps(result, ensure_ascii=False)

    def _walk(self, args: dict) -> str:
        return self.session.walk_for(
            vx=float(args.get("vx", 0.0)),
            vy=float(args.get("vy", 0.0)),
            vyaw=float(args.get("vyaw", 0.0)),
            seconds=float(args.get("seconds", 1.0)),
        )

    def _goto(self, args: dict) -> str:
        yaw = args.get("yaw")
        return self.session.goto(
            x=float(args.get("x", 0.0)),
            y=float(args.get("y", 0.0)),
            yaw=None if yaw is None else float(yaw),
            timeout=float(args.get("timeout", 20.0)),
        )

    def _turn(self, args: dict) -> str:
        return self.session.turn(float(args.get("degrees", 90.0)),
                                 timeout=float(args.get("timeout", 10.0)))

    def _set_arm(self, args: dict) -> str:
        preset = args.get("preset")
        if preset:
            return self.session.set_arm_preset(str(preset))
        joints = args.get("joints") or {}
        if not joints:
            return ("robot_set_arm 需要给出 preset（预设名）或 joints（关节角）之一。"
                    "可用预设：%s" % "、".join(self._preset_names_safe()))
        return self.session.set_arm_joints(side=str(args.get("side", "both")), **joints)

    def _set_waist(self, args: dict) -> str:
        def opt(key):
            v = args.get(key)
            return None if v is None else float(v)
        return self.session.set_waist(yaw=opt("yaw"), roll=opt("roll"), pitch=opt("pitch"))

    def _sequence(self, args: dict) -> str:
        return self.session.sequence(args.get("steps") or [])

    def _stop(self, _args: dict) -> str:
        return self.session.stop()

    def _reset(self, _args: dict) -> str:
        return self.session.reset()

    def _log_motion(self, args: dict) -> str:
        import json
        output = args.get('output', 'motions/log_motion.csv')
        action = str(args.get('action', 'walk')).lower()
        self.session.set_motion_log(output)
        result = ''
        try:
            if action == 'walk':
                result = self.session.walk_for(
                    vx=float(args.get('vx', 0.5)), vy=float(args.get('vy', 0.0)),
                    vyaw=float(args.get('vyaw', 0.0)), seconds=float(args.get('seconds', 1.0)))
            elif action == 'goto':
                result = self.session.goto(
                    x=float(args.get('x', 0.0)), y=float(args.get('y', 0.0)))
            elif action == 'turn':
                result = self.session.turn(float(args.get('degrees', 90.0)))
            else:
                result = '未知动作类型：%s' % action
        finally:
            summary = self.session.motion_log_summary()
            self.session.set_motion_log(None)
        return json.dumps({'action_result': result, 'log': summary}, ensure_ascii=False)

    def _create_folder(self, args: dict) -> str:
        import json
        import time
        from pathlib import Path
        name = args.get('name') or time.strftime('%Y%m%d_%H%M%S')
        parent = Path(args.get('parent', 'outputs'))
        folder = parent / name
        folder.mkdir(parents=True, exist_ok=True)
        return json.dumps({'folder': str(folder), 'absolute': str(folder.resolve()), 'created': True}, ensure_ascii=False)

    @staticmethod
    def _preset_names_safe() -> list:
        try:
            return available_arm_presets()
        except Exception:
            return []


# ======================================================================
def _fmt_args(args: dict) -> str:
    """把参数压缩成一行，方便打日志。"""
    parts = []
    for k, v in args.items():
        s = str(v)
        parts.append("%s=%s" % (k, s if len(s) <= 40 else s[:37] + "..."))
    return ", ".join(parts)
