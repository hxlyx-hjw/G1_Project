import unittest
from fullbody.curriculum import LegCurriculum


class CurriculumTest(unittest.TestCase):
    def test_three_clean_evaluations_promote(self):
        c=LegCurriculum(target=.20)
        good={'completed':True,'fallen':False,'max_root_xy_error_m':.05,'max_root_speed_mps':.4,'joint_mse':.02}
        self.assertFalse(c.update(good)['promoted'])
        self.assertFalse(c.update(good)['promoted'])
        self.assertTrue(c.update(good)['promoted'])
        self.assertAlmostEqual(c.weight,.20)
        for _ in range(6): c.update(good)
        self.assertAlmostEqual(c.weight,.20)

    def test_drift_prevents_promotion(self):
        c=LegCurriculum()
        bad={'completed':True,'fallen':False,'max_root_xy_error_m':3.0,'max_root_speed_mps':.4,'joint_mse':.02}
        for _ in range(10): self.assertFalse(c.update(bad)['passed'])
        self.assertEqual(c.weight,.15)


if __name__=='__main__': unittest.main()
