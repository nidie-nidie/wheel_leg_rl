from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
BASE_DIR = WORKSPACE / "wheel_leg_debug-main" / "rm_test-dev" / "MDK-ARM"
OUTPUT_DIR = ROOT / "MDK-ARM"

GROUPS = {
    "Startup": ["MDK-ARM/startup_stm32h723xx.s"],
    "Core": [
        "Core/Src/main.c", "Core/Src/gpio.c", "Core/Src/dma.c",
        "Core/Src/fdcan.c", "Core/Src/tim.c", "Core/Src/usart.c",
        "Core/Src/freertos.c", "Core/Src/stm32h7xx_it.c",
        "Core/Src/stm32h7xx_hal_msp.c",
        "Core/Src/stm32h7xx_hal_timebase_tim.c",
    ],
    "Board": [
        "Board/Src/pace_can_port.c", "Board/Src/pace_time.c",
        "Board/Src/pace_uart_port.c",
    ],
    "Motor": [
        "Motor/Src/dm_motor.c", "Motor/Src/lk_motor.c",
        "Motor/Src/pace_motor_registry.c",
    ],
    "PACE": [
        "PACE/Src/pace_app.c", "PACE/Src/pace_capture.c",
        "PACE/Src/pace_excitation.c", "PACE/Src/pace_experiment.c",
        "PACE/Src/pace_safety.c", "PACE/Src/pace_transport.c",
    ],
    "Protocol": [
        "Protocol/Src/pace_command.c", "Protocol/Src/pace_crc.c",
        "Protocol/Src/pace_frame.c", "Protocol/Src/pace_motor_order.c",
    ],
    "HAL": [
        "stm32h7xx_hal.c", "stm32h7xx_hal_cortex.c", "stm32h7xx_hal_dma.c",
        "stm32h7xx_hal_dma_ex.c", "stm32h7xx_hal_exti.c",
        "stm32h7xx_hal_fdcan.c", "stm32h7xx_hal_flash.c",
        "stm32h7xx_hal_flash_ex.c", "stm32h7xx_hal_gpio.c",
        "stm32h7xx_hal_pwr.c", "stm32h7xx_hal_pwr_ex.c",
        "stm32h7xx_hal_rcc.c", "stm32h7xx_hal_rcc_ex.c",
        "stm32h7xx_hal_tim.c", "stm32h7xx_hal_tim_ex.c",
        "stm32h7xx_hal_uart.c", "stm32h7xx_hal_uart_ex.c",
    ],
    "CMSIS": ["Core/Src/system_stm32h7xx.c"],
    "FreeRTOS": [
        "croutine.c", "event_groups.c", "list.c", "queue.c",
        "stream_buffer.c", "tasks.c", "timers.c",
        "CMSIS_RTOS/cmsis_os.c", "portable/MemMang/heap_4.c",
        "portable/RVDS/ARM_CM4F/port.c",
    ],
}

GROUPS["HAL"] = [
    f"Drivers/STM32H7xx_HAL_Driver/Src/{name}" for name in GROUPS["HAL"]
]
GROUPS["FreeRTOS"] = [
    f"Middlewares/Third_Party/FreeRTOS/Source/{name}" for name in GROUPS["FreeRTOS"]
]

INCLUDE_PATHS = [
    "../Core/Inc", "../Board/Inc", "../Motor/Inc", "../PACE/Inc",
    "../Protocol/Inc", "../Config", "../Drivers/STM32H7xx_HAL_Driver/Inc",
    "../Drivers/STM32H7xx_HAL_Driver/Inc/Legacy",
    "../Drivers/CMSIS/Device/ST/STM32H7xx/Include", "../Drivers/CMSIS/Include",
    "../Middlewares/Third_Party/FreeRTOS/Source/include",
    "../Middlewares/Third_Party/FreeRTOS/Source/CMSIS_RTOS",
    "../Middlewares/Third_Party/FreeRTOS/Source/portable/RVDS/ARM_CM4F",
]


def set_text(root: ET.Element, path: str, value: str) -> None:
    node = root.find(path)
    if node is None:
        raise RuntimeError(f"missing XML element {path}")
    node.text = value


def keil_path(path: str) -> str:
    return str(pathlib.PureWindowsPath("..") / pathlib.PureWindowsPath(path))


def file_type(path: str) -> str:
    return "2" if pathlib.PurePosixPath(path).suffix.lower() in {".s", ".asm"} else "1"


def build_project() -> None:
    tree = ET.parse(BASE_DIR / "rm_test.uvprojx")
    root = tree.getroot()
    target = root.find("./Targets/Target")
    if target is None:
        raise RuntimeError("base project has no target")

    replacements = {
        "./Targets/Target/TargetName": "wheel_leg_motor_PACE",
        ".//TargetCommonOption/OutputDirectory": ".\\wheel_leg_motor_PACE",
        ".//TargetCommonOption/OutputName": "wheel_leg_motor_PACE",
        ".//TargetCommonOption/SFDFile": "",
        ".//TargetArmAds/Cads/VariousControls/Define": "USE_HAL_DRIVER,STM32H723xx",
        ".//TargetArmAds/Cads/VariousControls/IncludePath": ";".join(INCLUDE_PATHS),
        ".//TargetArmAds/Aads/VariousControls/IncludePath": "../Core/Inc",
        ".//TargetArmAds/LDads/umfTarg": "1",
        ".//TargetArmAds/ArmAdsMisc/OnChipMemories/OCR_RVCT10/StartAddress": "0x24000000",
        ".//TargetArmAds/ArmAdsMisc/OnChipMemories/OCR_RVCT10/Size": "0x20000",
        # Let uVision generate the linker scatter description from the target's
        # IROM/IRAM settings.  The former path pointed at a file that does not
        # exist and made every otherwise successful build fail at link time.
        ".//TargetArmAds/LDads/ScatterFile": "",
    }
    for path, value in replacements.items():
        set_text(root, path, value)

    groups = target.find("Groups")
    if groups is None:
        raise RuntimeError("base project has no groups")
    groups.clear()
    for group_name, paths in GROUPS.items():
        group = ET.SubElement(groups, "Group")
        ET.SubElement(group, "GroupName").text = group_name
        files = ET.SubElement(group, "Files")
        for path in paths:
            if not (ROOT / path).is_file():
                raise FileNotFoundError(ROOT / path)
            file_node = ET.SubElement(files, "File")
            ET.SubElement(file_node, "FileName").text = pathlib.PurePosixPath(path).name
            ET.SubElement(file_node, "FileType").text = file_type(path)
            ET.SubElement(file_node, "FilePath").text = keil_path(path)

    for node in root.findall(".//RTE//targetInfo"):
        node.set("name", "wheel_leg_motor_PACE")
    for path in (".//LayerInfo//LayName", ".//LayerInfo//LayTitle"):
        for node in root.findall(path):
            node.text = "wheel_leg_motor_PACE"
    tree.write(OUTPUT_DIR / "wheel_leg_motor_PACE.uvprojx", encoding="utf-8", xml_declaration=True)


def add_option_file(group: ET.Element, group_number: int, number: int, path: str) -> None:
    node = ET.SubElement(group, "File")
    values = {
        "GroupNumber": str(group_number), "FileNumber": str(number),
        "FileType": file_type(path), "tvExp": "0", "tvExpOptDlg": "0",
        "bDave2": "0", "PathWithFileName": keil_path(path),
        "FilenameWithoutPath": pathlib.PurePosixPath(path).name,
        "RteFlg": "0", "bShared": "0",
    }
    for key, value in values.items():
        ET.SubElement(node, key).text = value


def build_options() -> None:
    tree = ET.parse(BASE_DIR / "rm_test.uvoptx")
    root = tree.getroot()
    set_text(root, "./Target/TargetName", "wheel_leg_motor_PACE")
    for old_group in list(root.findall("Group")):
        root.remove(old_group)
    insert_at = len(root)
    for group_number, (group_name, paths) in enumerate(GROUPS.items(), start=1):
        group = ET.Element("Group")
        for key, value in (("GroupName", group_name), ("tvExp", "1"),
                           ("tvExpOptDlg", "0"), ("cbSel", "0"), ("RteFlg", "0")):
            ET.SubElement(group, key).text = value
        for number, path in enumerate(paths, start=1):
            add_option_file(group, group_number, number, path)
        root.insert(insert_at, group)
        insert_at += 1
    tree.write(OUTPUT_DIR / "wheel_leg_motor_PACE.uvoptx", encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    build_project()
    build_options()
    print(OUTPUT_DIR / "wheel_leg_motor_PACE.uvprojx")
    print(OUTPUT_DIR / "wheel_leg_motor_PACE.uvoptx")
