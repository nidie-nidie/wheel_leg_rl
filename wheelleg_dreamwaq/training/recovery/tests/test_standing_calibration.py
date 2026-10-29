import json
from pathlib import Path
import unittest

from recovery.standing_calibration import file_sha256, load_standing_calibration


REPORT = Path(__file__).resolve().parents[3] / "recovery/artifacts/independent-stand-implementation-20261029/standing_calibration.json"


class StandingCalibrationTests(unittest.TestCase):
    def test_real_positive_and_negative_calibration_has_required_evidence(self):
        report = load_standing_calibration(REPORT, file_sha256(REPORT), require_formal=True)
        self.assertEqual(len(report["positive_401"]), 20)
        self.assertEqual(len(report["negative_6200"]), 12)
        self.assertTrue(all(item["post_2s_qualified_fraction"] >= .99
                            and item["longest_post_2s_qualified_steps"] >= 500
                            and item["full_eval_success"] for item in report["positive_401"]))
        self.assertTrue(all(item["steps"] == item["kinematic_rejected_steps"] for item in report["negative_6200"]))

    def test_five_actual_physical_anomalies_have_no_continuous_qualification(self):
        report = load_standing_calibration(REPORT, file_sha256(REPORT), require_formal=True)
        negative = report["physical_negative"]
        self.assertTrue(negative["passed"])
        self.assertEqual({item["name"] for item in negative["episodes"]},
                         {"single_wheel", "fast_sliding", "body_supported", "inverted", "low_body"})
        for item in negative["episodes"]:
            self.assertGreaterEqual(item["samples"], 50)
            self.assertTrue(item["required_condition_observed"])
            self.assertLess(item["longest_qualified_steps"], 50)
            self.assertFalse(item["full_success"])

    def test_actual_hold_returns_favor_timely_completion(self):
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(len(report["hold_return_comparison"]), 20)
        for item in report["hold_return_comparison"]:
            self.assertGreater(item["timely"], item["one_step_delayed"])

    def test_wrong_report_hash_is_rejected(self):
        with self.assertRaises(ValueError):
            load_standing_calibration(REPORT, "0" * 64)
