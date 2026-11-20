import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from harness_search import resolve_tavily_key
from fullbody.skills import list_skills


class ConfigReviewTest(unittest.TestCase):
    def test_search_key_is_separate_from_deepseek(self):
        with tempfile.TemporaryDirectory() as directory,patch('harness_search.PROJECT_ROOT',Path(directory)),patch.dict(os.environ,{},clear=True):
            root=Path(directory)
            (root/'api_key.txt').write_text('deepseek-test-key')
            self.assertIsNone(resolve_tavily_key())
            (root/'tavily_api_key.txt').write_text('  tvly-test-key  ')
            self.assertEqual(resolve_tavily_key(),'tvly-test-key')
            os.environ['TAVILY_API_KEY']='env-test-key'
            self.assertEqual(resolve_tavily_key(),'env-test-key')
            self.assertEqual(resolve_tavily_key('explicit-test-key'),'explicit-test-key')

    def test_corrupt_registry_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as directory,patch('fullbody.skills.DB',Path(directory)/'registry.json'):
            path=Path(directory)/'registry.json'; path.write_text('{bad json')
            with self.assertRaises(ValueError): list_skills()
            self.assertEqual(path.read_text(),'{bad json')

    def test_bot_enables_search_from_project_key(self):
        from deepseek_robot import RobotChatBot
        with patch('harness_search.resolve_tavily_key',return_value='tvly-test'),patch('tavily.TavilyClient') as client:
            bot=RobotChatBot(deepseek_api_key='test',enable_robot=False,enable_local_files=False,verbose=False)
            self.assertIn('search_web',bot._tool_names)
            client.assert_called_once_with(api_key='tvly-test')
            bot.client.close()


if __name__=='__main__': unittest.main()
