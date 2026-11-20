import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock
from deepseek_robot import RobotChatBot
from tool_output import format_result


def message(name=None):
    return SimpleNamespace(tool_calls=[SimpleNamespace(function=SimpleNamespace(name=name))] if name else [],content='完成')


class ToolLoopTest(unittest.TestCase):
    def bot(self):
        bot=RobotChatBot.__new__(RobotChatBot)
        bot.max_tool_rounds=12; bot.verbose=True; bot.history=[]
        bot._dispatch_tool=Mock()
        return bot

    def test_training_exceeds_twelve(self):
        bot=self.bot(); bot._ask=Mock(side_effect=[message('robot_training_status') for _ in range(25)]+[message()])
        with redirect_stdout(io.StringIO()): result=bot._run_until_final_text()
        self.assertEqual(result,'完成'); self.assertEqual(bot._dispatch_tool.call_count,25)

    def test_normal_limit_retained(self):
        bot=self.bot(); bot._ask=Mock(return_value=message('robot_status'))
        self.assertIn('普通工具',bot._run_until_final_text())
        self.assertEqual(bot._dispatch_tool.call_count,12)

    def test_result_interpretation_and_raw_payload(self):
        text=format_result('robot_training_status',json.dumps({'state':'completed','evaluation':{'fallen':True,'reference_completed':False},'leg_weight':.15}))
        self.assertIn('失稳',text); self.assertIn('缩幅',text); self.assertIn('原始结构化输出',text)

    def test_dispatch_preserves_result(self):
        bot=RobotChatBot.__new__(RobotChatBot); bot.verbose=True; bot.history=[]; bot.robot=Mock(); bot._tool_names={'robot_training_status'}
        payload='{"state":"training","iteration":5}'
        bot.robot.dispatch.return_value=payload
        call=SimpleNamespace(id='call_1',function=SimpleNamespace(name='robot_training_status',arguments='{}'))
        with redirect_stdout(io.StringIO()) as console: bot._dispatch_tool(call)
        self.assertIn('训练轮次',console.getvalue())
        self.assertEqual(bot.history[-1]['content'],payload)


if __name__=='__main__': unittest.main()
