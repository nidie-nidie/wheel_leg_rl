import pathlib
import tempfile
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
        cls.output = tempfile.TemporaryDirectory()
        cls.previous_output = generate_keil_project.OUTPUT_DIR
        generate_keil_project.OUTPUT_DIR = pathlib.Path(cls.output.name)
        generate_keil_project.build_project()
        generate_keil_project.build_options()
        cls.root = ET.parse(generate_keil_project.OUTPUT_DIR / PROJECT.name).getroot()
        cls.addClassCleanup(cls.output.cleanup)
        cls.addClassCleanup(setattr, generate_keil_project, "OUTPUT_DIR", cls.previous_output)

    def test_target_and_output_are_renamed(self):
        self.assertEqual(self.root.findtext("./Targets/Target/TargetName"), "wheel_leg_motor_PACE")
        self.assertEqual(self.root.findtext(".//TargetCommonOption/OutputName"), "wheel_leg_motor_PACE")

    def test_linker_keeps_dma_outside_dtcm(self):
        self.assertEqual(self.root.findtext(".//TargetArmAds/LDads/umfTarg"), "0")
        self.assertEqual(self.root.findtext(".//TargetArmAds/LDads/useFile"), "0")
        self.assertEqual(self.root.findtext(".//TargetArmAds/LDads/ScatterFile"), "../Config/pace_memory.sct")
        scatter = (ROOT / "Config/pace_memory.sct").read_text(encoding="utf-8")
        self.assertIn("RW_DMA 0x24000000 0x00001000", scatter)
        self.assertIn("* (.pace_dma)", scatter)

    def test_target_ram_regions_are_valid(self):
        ram2 = ".//TargetArmAds/ArmAdsMisc/OnChipMemories/OCR_RVCT10"
        self.assertEqual(self.root.findtext(f"{ram2}/StartAddress"), "0x24000000")
        self.assertEqual(self.root.findtext(f"{ram2}/Size"), "0x20000")

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
