import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG = (ROOT / "Config" / "pace_experiment_config.h").read_text(encoding="utf-8")
EXPERIMENT = (ROOT / "PACE" / "Src" / "pace_experiment.c").read_text(encoding="utf-8")
APP = (ROOT / "PACE" / "Src" / "pace_app.c").read_text(encoding="utf-8")


def define_int(name: str) -> int:
    match = re.search(rf"#define\s+{name}\s+(\d+)U", CONFIG)
    if not match:
        raise AssertionError(f"missing integer define {name}")
    return int(match.group(1))


def define_float(name: str) -> float:
    match = re.search(rf"#define\s+{name}\s+([0-9.]+)f", CONFIG)
    if not match:
        raise AssertionError(f"missing float define {name}")
    return float(match.group(1))


def bus_load_percent(rates_hz):
    return sum(rates_hz) * 260.0 / define_int("PACE_CAN_NOMINAL_BITRATE") * 100.0


class CanScheduleTest(unittest.TestCase):
    def test_documented_stage_loads_remain_below_target(self):
        self.assertAlmostEqual(bus_load_percent([100] * 6), 15.6, places=3)
        self.assertAlmostEqual(bus_load_percent([500] * 4 + [100] * 2), 57.2, places=3)
        self.assertAlmostEqual(bus_load_percent([100] * 4 + [500] * 2), 36.4, places=3)
        self.assertLess(bus_load_percent([500] * 4 + [100] * 2), 65.0)
        self.assertLess(bus_load_percent([100] * 4 + [500] * 2), 65.0)

    def test_all_six_at_500_hz_is_rejected_by_budget(self):
        overloaded = bus_load_percent([500] * 6)
        self.assertAlmostEqual(overloaded, 78.0, places=3)
        self.assertGreater(overloaded, 70.0)
        self.assertIn("pace_experiment_stage_bus_load_percent(stage) > 70.0f", EXPERIMENT)

    def test_baseline_dm_contract_is_frozen(self):
        self.assertEqual(define_float("PACE_DM_BASELINE_KP"), 20.0)
        self.assertEqual(define_float("PACE_DM_BASELINE_KD"), 0.6)
        self.assertIn("output->dm_velocity_rad_s[index] = 0.0f", EXPERIMENT)
        self.assertIn("output->dm_torque_nm[index] = 0.0f", EXPERIMENT)

    def test_state_and_stage_guards_exist(self):
        expected = [
            "PACE_STAGE_STATIC",
            "PACE_STAGE_DM_FIT",
            "PACE_STAGE_DM_VALIDATION",
            "PACE_STAGE_LK_TORQUE",
            "PACE_STAGE_LK_VELOCITY",
            "PACE_STAGE_STOP",
        ]
        start = EXPERIMENT.index("expected_stage_ids")
        block = EXPERIMENT[start : start + 500]
        positions = [block.index(stage) for stage in expected]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("experiment->state != PACE_STATE_CONFIGURED", EXPERIMENT)
        self.assertIn("experiment->state != PACE_STATE_RUNNING", EXPERIMENT)

    def test_final_session_remains_locked_until_contract_is_frozen(self):
        self.assertEqual(define_int("PACE_FINAL_CONTRACT_READY"), 0)
        self.assertIn("(PACE_FINAL_CONTRACT_READY == 1U)", APP)
        self.assertRegex(
            APP,
            r"PACE_SESSION_FINAL_IDENTIFICATION\)\s*&&\s*"
            r"pace_app_final_contract_is_ready\(\)",
        )

    def test_status_encoder_uses_the_full_128_byte_buffer(self):
        self.assertIn("uint8_t frame[PACE_STATUS_MAX_FRAME_SIZE];", APP)

    def test_arm_failures_have_a_header_and_terminal_record(self):
        start = APP.index("case PACE_HOST_START:")
        stop = APP.index("case PACE_HOST_STOP:")
        armed_handler = APP.index("static void pace_app_handle_armed_tick")
        finish_handler = APP.index("static void pace_app_finish_session")
        self.assertIn("pace_app_emit_session_header(now_us)", APP[start:stop])
        self.assertNotIn(
            "pace_app_emit_session_header(now_us)",
            APP[armed_handler:finish_handler],
        )
        self.assertIn("pace_app_schedule_terminal", APP[stop:APP.index("case PACE_HOST_STATUS:")])


if __name__ == "__main__":
    unittest.main()
