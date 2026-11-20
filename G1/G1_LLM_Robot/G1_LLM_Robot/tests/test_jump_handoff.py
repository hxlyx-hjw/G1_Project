import unittest,tempfile
from pathlib import Path
import numpy as np
from coordination.jump import run
from coordination.handoff import import_state
from robot_bridge import RobotSession
from robot_tools import RobotToolkit


class HandoffTest(unittest.TestCase):
    def test_jump_to_walk(self):
        checkpoint=Path('runs/coordination/com_graph.pt').resolve()
        if not checkpoint.is_file(): self.skipTest('network missing')
        with tempfile.TemporaryDirectory() as folder,RobotSession(verbose=False) as bot:
            result=run(folder,coordinator=checkpoint,start_pose=[.5,.3,.79,0])
            self.assertTrue(result['handoff_ready'])
            with np.load(Path(folder)/'handoff.npz') as z: expected=z['root_qpos'][:3]
            received=import_state(bot,Path(folder)/'handoff.npz')
            np.testing.assert_allclose(received['position'],expected,atol=1e-10)
            bot.walk_for(vx=.2,seconds=.5)
            self.assertFalse(bot.status()['fallen'])
            self.assertGreater(bot.status()['x'],expected[0])

    def test_pending_prevents_old_position_action(self):
        from unittest.mock import Mock
        tools=RobotToolkit(None)
        tools.coordination.before_action=Mock(side_effect=RuntimeError('跳跃尚未稳定'))
        session=Mock(); tools.session=session
        result=tools.dispatch('robot_walk',{'vx':.2})
        self.assertIn('尚未稳定',result); session.walk_for.assert_not_called()


if __name__=='__main__': unittest.main()
