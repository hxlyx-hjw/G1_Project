import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch,Mock
from fullbody.web_video import valid_url,video_candidate,search
from fullbody.jobs import TrainingJobs


class WebActionTest(unittest.TestCase):
    def test_filters(self):
        self.assertTrue(video_candidate('https://www.youtube.com/watch?v=x'))
        self.assertTrue(video_candidate('https://example.org/demo.mp4'))
        self.assertFalse(video_candidate('https://dictionary.cambridge.org/standing'))
        with self.assertRaises(ValueError): valid_url('file:///C:/a.mp4')

    def test_action_background_job(self):
        with tempfile.TemporaryDirectory() as tmp,patch('fullbody.jobs.ROOT',Path(tmp)),patch('fullbody.jobs.subprocess.Popen') as spawn:
            proc=Mock(); proc.poll.return_value=None; spawn.return_value=proc
            jobs=TrainingJobs(); state=jobs.learn_action('挥手',source_urls=['https://example.org/demo.mp4'],render=False)
            self.assertTrue(state['worker_running'])
            self.assertTrue((Path(state['output_dir'])/'config.json').exists())
            jobs.stop(); self.assertTrue((Path(state['output_dir'])/'STOP').exists())


if __name__=='__main__': unittest.main()
