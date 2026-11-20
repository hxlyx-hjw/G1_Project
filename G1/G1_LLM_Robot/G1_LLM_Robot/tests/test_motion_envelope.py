import unittest

from fullbody.motion import MotionConfig, amplitude, balance_margin


class MotionEnvelopeTest(unittest.TestCase):
    def test_legacy_matches_previous_scales(self):
        config = MotionConfig.legacy()
        self.assertEqual(config.max_gain, 1.0)
        self.assertEqual(config.leg_action_scale, 0.12)
        self.assertEqual(config.arm_action_scale, 0.20)
        self.assertFalse(config.guard)
        self.assertEqual(amplitude(0.0, config), 1.0)

    def test_margin_scales_amplitude_and_guards_balance(self):
        config = MotionConfig()
        stable = balance_margin(1.0, 0.0, 0.0, config)
        unstable = balance_margin(0.40, 2.0, 0.50, config)
        self.assertGreater(stable, unstable)
        self.assertAlmostEqual(amplitude(stable, config), config.max_gain)
        self.assertGreaterEqual(amplitude(unstable, config), config.min_gain)
        self.assertLess(amplitude(unstable, config), config.max_gain)

    def test_rejects_invalid_gain(self):
        with self.assertRaises(ValueError):
            MotionConfig(max_gain=0.5, min_gain=0.8).validate()

    def test_rl_pd_influence_ranges(self):
        config = MotionConfig(
            rl_pd_influence=0.7,
            hand_rl_pd_range=(0.2, 0.6),
            foot_rl_pd_range=(0.5, 0.9),
        ).validate()
        self.assertAlmostEqual(config.effective_hand_influence(), 0.6)
        self.assertAlmostEqual(config.effective_foot_influence(), 0.7)
        default = MotionConfig().validate()
        self.assertAlmostEqual(default.effective_hand_influence(), 0.0)
        self.assertAlmostEqual(default.effective_foot_influence(), 1.0)
        with self.assertRaises(ValueError):
            MotionConfig(hand_rl_pd_range=(0.6, 0.2)).validate()
        with self.assertRaises(ValueError):
            MotionConfig(rl_pd_influence=1.2).validate()


if __name__ == '__main__':
    unittest.main()
