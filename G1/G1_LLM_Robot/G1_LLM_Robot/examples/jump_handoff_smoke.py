"""真实窗口保持→自动交接→后续行走验证；结束时关闭本测试窗口。"""
import json,time
from pathlib import Path
from robot_bridge import RobotSession
from robot_tools import RobotToolkit


def main():
    with RobotSession(verbose=False) as bot:
        tools=RobotToolkit(bot)
        started=json.loads(tools.dispatch('robot_coordination_start',{'operation':'jump','viewer':True,'record':False}))
        print(started,flush=True); task_id=started['task_id']
        try:
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                result=json.loads(tools.dispatch('robot_coordination_status',{'task_id':task_id}))
                if result['state']!='running': break
                time.sleep(.2)
            print(result,flush=True)
            assert result['handed_off'] and result['window_process_running'],result
            initial=bot.status()
            print(tools.dispatch('robot_walk',{'vx':.2,'seconds':.5}),flush=True)
            final=bot.status()
            assert not final['fallen']
            assert tools.coordination.tasks[task_id]['process'].poll() is None
            print('window retained; handoff and walking OK',initial['x'],final['x'],flush=True)
        finally:
            tools.coordination.close_window(task_id)
            tools.coordination.tasks[task_id]['process'].wait(timeout=5)


if __name__=='__main__': main()
