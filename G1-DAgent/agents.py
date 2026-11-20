"""Two-agent layer: status analyst and parameter debugger."""
from __future__ import annotations

import json

from config import (AGENT_TEMPERATURE, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
                    DEEPSEEK_MODEL, llm_available)
from param_debugger import ParameterDebugger
from status_reader import JobStatusReader


class AgentClient:
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = (api_key or DEEPSEEK_API_KEY).strip()
        self.base_url = (base_url or DEEPSEEK_BASE_URL).rstrip('/')
        self.model = model or DEEPSEEK_MODEL
        self.client = None
        if self.api_key:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def ask(self, system, user):
        if self.client is None:
            raise RuntimeError('未配置 DEEPSEEK_API_KEY')
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
            temperature=AGENT_TEMPERATURE,
        )
        return response.choices[0].message.content


class StatusAgent:
    """Reads a job record, states evidence, and gives bounded directions."""

    SYSTEM = (
        '你是 G1 人形机器人训练状态分析 Agent。只依据输入的本地训练事实回答，'
        '不得虚构训练成功、动作学会或未给出的指标。输出中文，分为'
        '“事实”“风险”“下一步方向”三部分。方向必须是可在本机执行的短步骤。'
        '不要把“训练进程完成”等同于“动作成功”，必须同时检查 fallen、'
        'reference_completed、完成度、漂移和关节误差。'
    )

    def __init__(self, client=None):
        self.client = client

    def analyze(self, record):
        evaluation = record.get('evaluation') or {}
        facts = [
            JobStatusReader.summarize(record),
            '训练来源: %s' % (record.get('reference') or record.get('source_video') or '未记录'),
            '最近指标: %s' % json.dumps(record.get('last_metric') or {}, ensure_ascii=False),
        ]
        risks = []
        directions = []
        if record.get('outcome') in ('running', 'queued'):
            risks.append('训练尚未结束或仍排队，当前指标不是最终结果。')
            directions.extend([
                '先等待并查看 status/evaluation，不提前启动重复训练。',
                '如果长时间无进度，再检查 worker/container 日志和 GPU 状态。',
            ])
        elif record.get('outcome') == 'failed':
            risks.append('训练或评估失败：%s' % (record.get('failure_reason') or '未给出原因'))
            directions.extend([
                '先确认失败是进程错误、CUDA/资源问题还是物理评估摔倒。',
                '若是摔倒/失稳，先降低动作幅度并用短轮次烟测；不要直接扩大训练。',
            ])
        elif record.get('outcome') == 'partial':
            risks.append('没有证据证明参考动作完成；当前只能视为部分完成。')
            directions.extend([
                '检查 reference_completed、完成度和终止原因。',
                '优先修复参考质量、脚部可见性和接触约束，再增加训练轮数。',
            ])
        else:
            risks.append('评估显示已通过当前门槛，但这不等于动作语义已核实。')
            directions.extend([
                '保留当前 checkpoint，做多随机种子/多次评估确认不是偶然通过。',
                '再逐步提高动作幅度或迁移到新参考，不要一次改变多个变量。',
            ])
        if evaluation.get('fallen'):
            risks.append('物理评估摔倒，当前策略不能视为稳定。')
        if evaluation.get('max_root_xy_error_m') is not None:
            drift = float(evaluation.get('max_root_xy_error_m'))
            if drift > .25:
                risks.append('平面漂移 %.3f m 偏大。' % drift)
        if evaluation.get('joint_mse') is not None:
            mse = float(evaluation.get('joint_mse'))
            if mse > .08:
                risks.append('关节 MSE %.4f 偏大。' % mse)

        result = {
            'job_id': record.get('job_id'),
            'facts': facts,
            'risks': risks,
            'directions': directions,
            'llm_analysis': None,
            'llm_error': None,
        }
        if self.client is not None:
            try:
                result['llm_analysis'] = self.client.ask(
                    self.SYSTEM,
                    json.dumps(record, ensure_ascii=False, default=str),
                )
            except Exception as exc:
                result['llm_error'] = '%s: %s' % (type(exc).__name__, exc)
        return result


class ParameterDebugAgent:
    """Uses deterministic bounded rules and optionally asks an LLM to review."""

    SYSTEM = (
        '你是 G1 训练参数调试 Agent。只审查给定的参数建议和训练事实，'
        '不得改变参数边界，不得声称已修改文件或已启动训练。'
        '重点判断建议是否会造成反复摔倒、漂移或无效重试。'
        '输出中文短评，最后给出“接受/缩小/拒绝”之一。'
    )

    def __init__(self, client=None):
        self.client = client
        self.debugger = ParameterDebugger()

    def debug(self, record):
        proposal = self.debugger.propose(record)
        result = {
            'proposal': proposal,
            'llm_review': None,
            'llm_error': None,
        }
        if self.client is not None:
            try:
                result['llm_review'] = self.client.ask(
                    self.SYSTEM,
                    json.dumps({
                        'job': record,
                        'proposal': proposal,
                    }, ensure_ascii=False, default=str),
                )
            except Exception as exc:
                result['llm_error'] = '%s: %s' % (type(exc).__name__, exc)
        return result


def build_agents(use_llm=True):
    client = AgentClient() if use_llm and llm_available() else None
    return StatusAgent(client), ParameterDebugAgent(client)
