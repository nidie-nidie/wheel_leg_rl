"""Build current production PACE sources for STM32H723 with Arm GNU.

No HAL fake, Keil objects or stale HEX is used. The source list is shared with
firmware/tools/generate_keil_project.py; only startup and RTOS port differ.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PACE = HERE.parent
FIRMWARE = PACE / "wheel_leg_motor_PACE"
VENDOR = HERE / "vendor"
CPU_FLAGS = ["-mcpu=cortex-m7", "-mthumb", "-mfpu=fpv5-d16", "-mfloat-abi=hard"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source_contract():
    path = FIRMWARE / "tools/generate_keil_project.py"
    spec = importlib.util.spec_from_file_location("pace_keil_source_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_compiler(requested: str | None) -> Path:
    candidates = []
    if requested:
        path = Path(requested)
        candidates.append(path if path.is_file() else path / "bin/arm-none-eabi-gcc.exe")
    if os.environ.get("PACE_ARM_GCC"):
        candidates.append(Path(os.environ["PACE_ARM_GCC"]))
    executable = shutil.which("arm-none-eabi-gcc")
    if executable:
        candidates.append(Path(executable))
    toolchain = PACE / "build/toolchain"
    if toolchain.exists():
        candidates.extend(toolchain.rglob("arm-none-eabi-gcc.exe"))
        candidates.extend(toolchain.rglob("arm-none-eabi-gcc"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError("Arm GNU compiler missing; pass --toolchain or set PACE_ARM_GCC.")


def verify_vendor() -> dict:
    payload = json.loads((VENDOR / "sources.json").read_text(encoding="utf-8"))
    for entry in payload["files"]:
        path = VENDOR / entry["path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Pinned vendor hash differs: {path}")
    return payload


def build(toolchain: str | None = None, jobs: int = 4) -> Path:
    vendor_sources = verify_vendor()
    source_contract = load_source_contract()
    compiler = find_compiler(toolchain)
    tools = {name: compiler.with_name("arm-none-eabi-" + name + compiler.suffix)
             for name in ("objcopy", "size", "nm", "objdump")}
    for tool in tools.values():
        if not tool.is_file():
            raise RuntimeError(f"Missing GNU companion tool: {tool}")
    out = PACE / "build/firmware_gcc/commissioning"
    out.mkdir(parents=True, exist_ok=True)
    (out / "obj").mkdir(exist_ok=True)
    # A failed rebuild must not leave an old success manifest discoverable.
    previous_manifest = out / "build_manifest.json"
    if previous_manifest.exists():
        previous_manifest.unlink()
    sources = []
    for group, paths in source_contract.GROUPS.items():
        for relative in paths:
            if group == "Startup" or "/portable/RVDS/" in ("/" + relative):
                continue
            sources.append(FIRMWARE / relative)
    sources.extend([VENDOR / "st/startup_stm32h723xx.s", VENDOR / "freertos/ARM_CM7/r0p1/port.c", HERE / "runtime.c"])
    includes = []
    for relative in source_contract.INCLUDE_PATHS:
        if "/portable/RVDS/" not in relative:
            includes.append((FIRMWARE / "MDK-ARM" / relative).resolve())
    includes.append(VENDOR / "freertos/ARM_CM7/r0p1")
    defines = ["USE_HAL_DRIVER", "STM32H723xx", "PACE_ONE_WAY_AUTOSTART=1", "PACE_FINAL_CONTRACT_READY=0"]
    flags = [*CPU_FLAGS, "-std=gnu11", "-O2", "-g3", "-ffunction-sections", "-fdata-sections",
             "-fstack-usage", "-Wall", "-Wextra", "-Wno-unused-parameter", "-fno-common",
             *["-D" + item for item in defines], *["-I" + str(path) for path in includes]]
    linker = HERE / "stm32h723_pace.ld"
    inputs = sorted({*sources, *[path for folder in includes if folder.is_dir() for path in folder.rglob("*.h")],
                     linker, Path(__file__).resolve(), FIRMWARE / "tools/generate_keil_project.py", VENDOR / "sources.json"})
    input_hashes = {path: sha256(path) for path in inputs}
    commands = []
    objects = []
    for index, source in enumerate(sources):
        if not source.is_file():
            raise FileNotFoundError(source)
        obj = out / "obj" / f"{index:02d}_{source.stem}.o"
        objects.append(obj)
        command = [str(compiler), *flags]
        if source.suffix.lower() == ".s":
            command.extend(["-x", "assembler-with-cpp"])
        command.extend(["-MMD", "-MP", "-MF", str(obj.with_suffix(".d")), "-c", str(source), "-o", str(obj)])
        commands.append(command)
    logs = []
    def compile_one(command):
        result = subprocess.run(command, cwd=out, capture_output=True, text=True, errors="replace")
        return result.returncode, result.stdout + result.stderr
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for result, output in pool.map(compile_one, commands):
            logs.append(output)
            if result:
                (out / "build.log").write_text("".join(logs), encoding="utf-8")
                raise RuntimeError(output or f"Compiler failed with {result}")
    elf = out / "wheel_leg_motor_PACE.elf"
    mapfile = out / "wheel_leg_motor_PACE.map"
    linker = HERE / "stm32h723_pace.ld"
    link = [str(compiler), *CPU_FLAGS, "-nostartfiles", "--specs=nano.specs", "--specs=nosys.specs",
            "-T", str(linker), "-Wl,--gc-sections", "-Wl,--print-memory-usage", "-Wl,-Map=" + str(mapfile),
            *[str(obj) for obj in objects], "-Wl,--start-group", "-lc", "-lm", "-lnosys", "-Wl,--end-group", "-o", str(elf)]
    result = subprocess.run(link, cwd=out, capture_output=True, text=True, errors="replace")
    logs.append(result.stdout + result.stderr)
    (out / "build.log").write_text("".join(logs), encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    hexfile = elf.with_suffix(".hex")
    binary = elf.with_suffix(".bin")
    subprocess.run([str(tools["objcopy"]), "-O", "ihex", str(elf), str(hexfile)], check=True)
    subprocess.run([str(tools["objcopy"]), "-O", "binary", str(elf), str(binary)], check=True)
    size = subprocess.check_output([str(tools["size"]), str(elf)], text=True)
    nm = subprocess.check_output([str(tools["nm"]), "-S", "-n", str(elf)], text=True)
    symbols = {}
    all_symbols = {}
    symbol_types = {}
    for line in nm.splitlines():
        fields = line.split()
        if len(fields) >= 3:
            all_symbols[fields[-1]] = int(fields[0], 16)
            symbol_types[fields[-1]] = fields[-2]
        if fields and fields[-1] in {"__pace_dma_start__", "__pace_dma_end__", "pace_uart_rx_dma", "_estack", "_sidata", "_sdata", "_edata", "_sbss", "_ebss"}:
            symbols[fields[-1]] = int(fields[0], 16)
    if symbols.get("__pace_dma_start__") != 0x24000000 or symbols.get("__pace_dma_end__", 0) <= 0x24000000:
        raise RuntimeError("ELF lacks the non-empty AXI DMA section.")
    vectors = struct.unpack_from("<16I", binary.read_bytes())
    if vectors[0] != 0x20020000:
        raise RuntimeError("Reset vector initial MSP differs from DTCM top.")
    vector_evidence = {"initial_msp": f"0x{vectors[0]:08X}"}
    for slot, handler in ((1, "Reset_Handler"), (11, "SVC_Handler"), (14, "PendSV_Handler"), (15, "SysTick_Handler")):
        if vectors[slot] != (all_symbols[handler] | 1):
            raise RuntimeError(f"Vector differs from Thumb handler: {handler}")
        if handler != "Reset_Handler" and symbol_types[handler].upper() != "T":
            raise RuntimeError(f"RTOS handler is not a strong code symbol: {handler}")
        vector_evidence[handler] = f"0x{vectors[slot]:08X}"
    if not (0x08000000 <= symbols["_sidata"] < 0x08100000 and
            0x20000000 == symbols["_sdata"] <= symbols["_edata"] <= symbols["_sbss"] <= symbols["_ebss"] < 0x20020000):
        raise RuntimeError("Startup data/bss bounds do not match the linker memory contract.")
    for prefix in ("__preinit_array", "__init_array"):
        if all_symbols[prefix + "_start"] > all_symbols[prefix + "_end"]:
            raise RuntimeError("C runtime array bounds invalid.")
    changed_inputs = [str(path) for path in inputs if sha256(path) != input_hashes[path]]
    if changed_inputs:
        raise RuntimeError("Sources changed during build; rerun with stable inputs: " + ", ".join(changed_inputs))
    version = subprocess.check_output([str(compiler), "--version"], text=True).splitlines()[0]
    manifest = {
        "schema_version": "PaceFirmwareGccBuildV1", "build_kind": "commissioning_passive_autostart",
        "compiler": str(compiler), "compiler_version": version, "cpu_flags": CPU_FLAGS,
        "toolchain_download": json.loads((PACE / "build/toolchain/download.json").read_text(encoding="utf-8"))
            if (PACE / "build/toolchain/download.json").is_file() else None,
        "compile_flags": flags, "compile_commands": commands, "link_command": link,
        "source_groups": source_contract.GROUPS, "vendor_sources": vendor_sources,
        "inputs": [{"path": str(path.relative_to(PACE)),
                    "sha256": input_hashes[path]} for path in inputs],
        "outputs": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                    for path in (elf, hexfile, binary, mapfile)},
        "elf_symbols": {name: f"0x{address:08X}" for name, address in symbols.items()},
        "startup_vectors": vector_evidence,
        "startup_initialization": "pinned ST startup copies .data, zeroes .bss, calls __libc_init_array then main; linker data/bss and init-array bounds checked",
        "source_consistency": "all input hashes unchanged across compile/link",
        "size": size, "memory_layout": {"flash": "0x08000000/1MB", "dtcm": "0x20000000/128KB",
                                          "axi": "0x24000000/128KB", "dma_section": ".pace_dma/AXI"},
        "board_validation": "not performed; compile/link and ELF placement only",
    }
    stack_usage = []
    for report in (out / "obj").glob("*.su"):
        for line in report.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split("\t")
            if len(fields) >= 3:
                try:
                    stack_usage.append({"function": fields[0], "bytes": int(fields[1]), "kind": fields[2]})
                except ValueError:
                    pass
    manifest["largest_individual_stack_frames"] = sorted(stack_usage, key=lambda row: row["bytes"], reverse=True)[:20]
    manifest["stack_evidence_scope"] = "compiler individual frame estimates; not ISR/task worst-case nesting or measured high-water marks"
    (out / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(version)
    print(result.stdout, end="")
    print(size, end="")
    print("DMA: " + json.dumps(manifest["elf_symbols"]))
    print(hexfile)
    return hexfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolchain")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    try:
        build(args.toolchain, args.jobs)
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print("Firmware build failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
