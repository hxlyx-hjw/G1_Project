"""
================================================================
多智能体协作：让两个 AI 交流，共同实现一个功能（模块调用版）
================================================================

核心思想：
    "另一个 AI"本质上就是一次 API 调用，和联网搜索、文件解析没有区别。

依赖：pip install openai

【作为模块调用】
    from multi_agent import AIClient, Dialogue

    coder = AIClient("程序员", "sk-xxx", "https://api.deepseek.com", "deepseek-flash")
    reviewer = AIClient("审查员", "sk-xxx", "https://api.deepseek.com", "deepseek-flash")

    team = Dialogue(coder, reviewer,
                    system_a="你是资深程序员，写高质量代码",
                    system_b="你是严格审查员，挑毛病并给改进建议")
    transcript = team.run("写一个判断质数的函数", rounds=3)
    print(team.final())          # 最终结论
    print(team.transcript)       # 完整对话记录

【直接运行】
    python multi_agent.py
"""

from typing import Callable, List, Optional

from openai import OpenAI

__all__ = ["AIClient", "Dialogue", "make_expert_tool", "two_ai_dialogue"]


class AIClient:
    """通用 AI 客户端，可接任意 OpenAI 兼容的 API 服务。

    通过改 base_url / model，可以接 DeepSeek、OpenAI、通义千问、
    Kimi、智谱、本地 Ollama 等几乎所有大模型。
    """

    def __init__(self, name: str, api_key: str, base_url: str, model: str,
                 temperature: float = 0.7):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def ask(self, question: str, system: Optional[str] = None,
            history: Optional[List[str]] = None) -> str:
        """问这个 AI 一个问题，返回它的回答。

        question: 要问的话
        system:   系统提示词（设定它的角色，如"你是代码审查专家"）
        history:  之前的历史消息（字符串列表），让它看到上下文
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            for h in history:
                messages.append({"role": "user", "content": h})
        messages.append({"role": "user", "content": question})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content


class Dialogue:
    """两个 AI 的多轮交流。

    用法：
        team = Dialogue(ai_a, ai_b, system_a=..., system_b=...)
        team.run(task, rounds=3)   # 开始交流
        team.final()               # 最终结论
        team.transcript            # 完整对话记录
    """

    def __init__(self, ai_a: AIClient, ai_b: AIClient,
                 system_a: Optional[str] = None,
                 system_b: Optional[str] = None):
        self.ai_a = ai_a
        self.ai_b = ai_b
        self.system_a = system_a
        self.system_b = system_b
        self.transcript: List[str] = []

    def run(self, task: str, rounds: int = 3) -> List[str]:
        """开始交流：两个 AI 轮流发言，各自看到对方之前的所有内容。

        返回完整对话记录（字符串列表），最后一条通常是最终结论。
        """
        self.transcript = [f"【任务】{task}"]

        for _ in range(rounds):
            # A 发言：能看到任务 + 之前所有对话
            a_reply = self.ai_a.ask("\n".join(self.transcript), system=self.system_a)
            self.transcript.append(f"【{self.ai_a.name}】{a_reply}")

            # B 回应：能看到 A 刚说的话
            b_reply = self.ai_b.ask("\n".join(self.transcript), system=self.system_b)
            self.transcript.append(f"【{self.ai_b.name}】{b_reply}")

        return self.transcript

    def final(self) -> str:
        """返回最后一条发言（通常是最终结论）。"""
        return self.transcript[-1] if self.transcript else ""


def make_expert_tool(expert: AIClient, role_prompt: str) -> Callable[[str], str]:
    """把另一个 AI 封装成一个工具函数，供主模型 Function Calling 调用。

    返回的函数签名是 (question: str) -> str，可以直接当作"工具实现"
    接入主模型，主模型在需要时自动调用它来咨询这个专家。
    """
    def consult(question: str) -> str:
        return expert.ask(question, system=role_prompt)
    return consult


def two_ai_dialogue(ai_a: AIClient, ai_b: AIClient, task: str, rounds: int = 3,
                    system_a: Optional[str] = None,
                    system_b: Optional[str] = None) -> List[str]:
    """便捷函数：内部用 Dialogue 实现，直接返回对话记录。"""
    return Dialogue(ai_a, ai_b, system_a, system_b).run(task, rounds)


# ============================================================
# 演示：写代码 + 审查代码 的双 AI 协作
# ============================================================
if __name__ == "__main__":
    import os

    KEY = os.getenv("DEEPSEEK_API_KEY")
    BASE = "https://api.deepseek.com"

    coder = AIClient("程序员", KEY, BASE, "deepseek-flash")
    reviewer = AIClient("审查员", KEY, BASE, "deepseek-flash")

    team = Dialogue(
        coder, reviewer,
        system_a="你是一名资深程序员，负责写出高质量代码。",
        system_b="你是一名严格的代码审查员，指出代码的问题并给出修改建议。",
    )
    result = team.run(
        "写一个 Python 函数，判断一个整数是否为质数，要求高效。",
        rounds=2,
    )

    for line in result:
        print(line)
        print("-" * 60)
