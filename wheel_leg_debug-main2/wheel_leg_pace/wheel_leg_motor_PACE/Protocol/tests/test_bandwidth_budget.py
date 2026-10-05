from __future__ import annotations

import unittest


class BandwidthBudgetTest(unittest.TestCase):
    def test_uart_worst_case_stays_within_seventy_percent(self) -> None:
        wire_capacity = 921600 / 10
        worst_case = 126 * 500 + 128 * 10
        self.assertEqual(64280, worst_case)
        self.assertLessEqual(worst_case, wire_capacity * 0.70)
        self.assertAlmostEqual(69.7482638889, worst_case / wire_capacity * 100, places=6)

    def test_float_layout_would_not_fit(self) -> None:
        legacy_estimate = 250 * 500
        self.assertGreater(legacy_estimate, 921600 / 10)

    def test_stage_can_load_estimates(self) -> None:
        bits_per_command_feedback_pair = 2 * 130
        static_load = 6 * 100 * bits_per_command_feedback_pair
        dm_load = (4 * 500 + 2 * 100) * bits_per_command_feedback_pair
        lk_load = (2 * 500 + 4 * 100) * bits_per_command_feedback_pair
        all_six_load = 6 * 500 * bits_per_command_feedback_pair
        self.assertEqual((156000, 572000, 364000, 780000),
                         (static_load, dm_load, lk_load, all_six_load))
        self.assertLessEqual(dm_load, 650000)
        self.assertGreater(all_six_load, 700000)


if __name__ == "__main__":
    unittest.main()
