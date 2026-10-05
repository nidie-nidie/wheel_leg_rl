import pathlib
import unittest
import xml.etree.ElementTree as ET

import generate_keil_project


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT = ROOT / "MDK-ARM" / "wheel_leg_motor_PACE.uvprojx"
MAIN = (ROOT / "Core" / "Src" / "main.c").read_text(encoding="utf-8")
SYSTEM = (ROOT / "Core" / "Src" / "system_stm32h7xx.c").read_text(encoding="utf-8")
IOC = (ROOT / "wheel_leg_motor_PACE.ioc").read_text(encoding="utf-8")


class KeilProjectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generate_keil_project.build_project()
        generate_keil_project.build_options()
        cls.root = ET.parse(PROJECT).getroot()

    def test_target_and_output_are_renamed(self):
        self.assertEqual(self.root.findtext("./Targets/Target/TargetName"), "wheel_leg_motor_PACE")
        self.assertEqual(self.root.findtext(".//TargetCommonOption/OutputName"), "wheel_leg_motor_PACE")

    def test_every_project_file_exists(self):
        for file_path in self.root.findall(".//Groups/Group/Files/File/FilePath"):
            relative = pathlib.PureWindowsPath(file_path.text)
            local = ROOT / pathlib.Path(*relative.parts[1:])
            self.assertTrue(local.is_file(), local)

    def test_control_stack_is_not_imported(self):
        xml = PROJECT.read_text(encoding="utf-8")
        for forbidden in ("VMC", "LQR", "Chassis", "INS_Task", "Music_Task", "USB_DEVICE"):
            self.assertNotIn(forbidden, xml)

    def test_expected_groups_are_exact(self):
        groups = [node.text for node in self.root.findall(".//Groups/Group/GroupName")]
        self.assertEqual(groups, list(generate_keil_project.GROUPS))

    def test_vos1_clock_is_400_mhz_and_hse_is_consistent(self):
        self.assertIn("PWR_REGULATOR_VOLTAGE_SCALE1", MAIN)
        self.assertIn("oscillator.PLL.PLLM = 6U;", MAIN)
        self.assertIn("oscillator.PLL.PLLN = 100U;", MAIN)
        self.assertIn("oscillator.PLL.PLLP = 1U;", MAIN)
        self.assertIn("#define HSE_VALUE    ((uint32_t)24000000)", SYSTEM)
        self.assertIn("RCC.HSE_VALUE=24000000", IOC)
        self.assertIn("RCC.SYSCLKFreq_VALUE=400000000", IOC)
        self.assertIn("RCC.HCLKFreq_Value=100000000", IOC)
        self.assertIn("RCC.PLL2_VCO_SEL-AdvancedSettings=RCC_PLL2VCOMEDIUM", IOC)


if __name__ == "__main__":
    unittest.main()
