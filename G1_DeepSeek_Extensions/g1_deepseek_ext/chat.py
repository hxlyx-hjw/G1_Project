"""G1 DeepSeek bot subclass with external research tools."""

from __future__ import annotations

import json
from typing import Optional

from deepseek_robot import RobotChatBot

from .adapters import DeepSeekIntegration


class ExtendedRobotChatBot(RobotChatBot):
    """Add tools at runtime without changing any existing G1 source file."""

    def __init__(self, *args, integration: Optional[DeepSeekIntegration] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.integration = integration or DeepSeekIntegration()
        collisions = self.integration.names() & set(self._tool_names)
        if collisions:
            raise RuntimeError(
                "扩展工具名与 G1 现有工具冲突：%s" % sorted(collisions)
            )
        self._register(self.integration.schemas(), self.integration.names())
        self.system_prompt += self.integration.PROMPT
        self.history[0]["content"] = self.system_prompt
        if self.verbose:
            print(
                "  · 研究工具已注册：%s"
                % "、".join(sorted(self.integration.names())),
                flush=True,
            )

    def _dispatch_tool(self, tool_call) -> None:
        name = tool_call.function.name
        if name not in self.integration.names():
            super()._dispatch_tool(tool_call)
            return
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            result = json.dumps(
                {"success": False, "error": "工具参数不是有效 JSON：%s" % exc},
                ensure_ascii=False,
            )
        else:
            result = self.integration.dispatch(name, args)
        from tool_output import format_result

        if self.verbose:
            print(format_result(name, result), flush=True)
        self.history.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            }
        )
