"""Compile the production PACE C modules against deterministic host HAL boundaries."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
FIRMWARE = HERE.parent / "wheel_leg_motor_PACE"
SOURCES = [
    HERE / "fakes/host_hal.c",
    HERE / "tests/test_native.c",
    *[FIRMWARE / name for name in (
        "Board/Src/pace_time.c", "Board/Src/pace_can_port.c", "Board/Src/pace_uart_port.c",
        "Motor/Src/dm_motor.c", "Motor/Src/lk_motor.c", "Motor/Src/pace_motor_registry.c",
        "PACE/Src/pace_app.c", "PACE/Src/pace_capture.c",
        "PACE/Src/pace_transport.c", "PACE/Src/pace_experiment.c", "PACE/Src/pace_excitation.c",
        "PACE/Src/pace_safety.c", "Protocol/Src/pace_frame.c", "Protocol/Src/pace_command.c",
        "Protocol/Src/pace_crc.c", "Protocol/Src/pace_motor_order.c",
    )],
]
INCLUDES = [HERE / "fakes", *[FIRMWARE / name for name in ("Board/Inc", "Motor/Inc", "Config", "PACE/Inc", "Protocol/Inc")]]


def _msvc_environment() -> tuple[Path, dict[str, str]]:
    roots = []
    # Discover VS without depending on the calling shell's PATH or developer prompt.
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if vswhere.is_file():
        result = subprocess.run([str(vswhere), "-latest", "-products", "*", "-requires",
                                 "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                                 "-property", "installationPath"], capture_output=True, text=True)
        roots.extend(Path(line.strip()) for line in result.stdout.splitlines() if line.strip())
    if os.environ.get("PACE_VS_ROOT"):
        roots.insert(0, Path(os.environ["PACE_VS_ROOT"]))
    roots.append(Path("E:/pow-mining-lab/.toolchain/vs2022"))
    for root in roots:
        vcvars = root / "VC/Auxiliary/Build/vcvarsall.bat"
        if not vcvars.is_file():
            continue
        command = f'cmd.exe /d /s /c ""{vcvars}" x64 >nul && set"'
        result = subprocess.run(command,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode:
            continue
        env = os.environ.copy()
        for line in result.stdout.splitlines():
            if "=" in line and not line.startswith("="):
                key, value = line.split("=", 1)
                env[key] = value
        compiler = shutil.which("cl.exe", path=env.get("Path", env.get("PATH", "")))
        if compiler:
            return Path(compiler), env
    raise RuntimeError("MSVC C tools unavailable. Set PACE_VS_ROOT to a VS installation with vcvarsall.bat.")


def input_hashes(sources: list[Path] | None = None) -> dict[str, str]:
    if sources is None:
        sources = [*SOURCES, FIRMWARE / "PACE/tests/test_experiment_contract.c"]
    inputs = set(sources)
    inputs.update((HERE / "build.py", HERE / "run_checks.py"))
    for include in INCLUDES:
        inputs.update(include.rglob("*.h"))
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(inputs)}


def build(mode: int = 1, compiler: str = "auto", sanitize: bool = False) -> Path:
    build_dir = HERE / "build" / (("passive" if mode else "active") + ("-asan" if sanitize else ""))
    build_dir.mkdir(parents=True, exist_ok=True)
    output = build_dir / ("pace_native.exe" if os.name == "nt" else "pace_native")
    sources = list(SOURCES)
    contract_test = FIRMWARE / "PACE/tests/test_experiment_contract.c"
    if not contract_test.is_file():
        raise RuntimeError(f"Required production experiment contract test missing: {contract_test}")
    sources.append(contract_test)
    use_msvc = compiler == "msvc" or (compiler == "auto" and os.name == "nt")
    if use_msvc:
        cc, env = _msvc_environment()
        command = [str(cc), "/nologo", "/TC", "/std:c11", "/W4", "/Od", "/Zi",
                   "/D_CRT_SECURE_NO_WARNINGS", f"/DPACE_ONE_WAY_AUTOSTART={mode}",
                   "/DPACE_HAS_CONTRACT_TEST=1",
                   *[f"/I{path}" for path in INCLUDES],
                   *[str(path) for path in sources], f"/Fe:{output}", f"/Fd:{build_dir / 'pace.pdb'}",
                   *(["/fsanitize=address", "/link", "/INCREMENTAL:NO"] if sanitize else [])]
    else:
        cc = Path(shutil.which("gcc") or shutil.which("cc") or "")
        if not str(cc) or str(cc) == ".":
            raise RuntimeError("GCC unavailable; install gcc or select MSVC on Windows.")
        env = os.environ.copy()
        command = [str(cc), "-std=c11", "-Wall", "-Wextra", "-Werror", "-O0", "-g",
                   f"-DPACE_ONE_WAY_AUTOSTART={mode}",
                   "-DPACE_HAS_CONTRACT_TEST=1",
                   *[f"-I{path}" for path in INCLUDES],
                   *[str(path) for path in sources], "-lm", "-o", str(output),
                   *(["-fsanitize=address", "-fno-omit-frame-pointer"] if sanitize else [])]
    before_hashes = input_hashes(sources)
    print(f"Compiler: {cc}", flush=True)
    print(f"Mode: {'passive autostart' if mode else 'host controlled'}", flush=True)
    subprocess.run(command, cwd=build_dir, env=env, check=True)
    if sanitize and use_msvc:
        runtime_dlls = list(cc.parent.glob("clang_rt.asan*dynamic-x86_64.dll"))
        if not runtime_dlls:
            raise RuntimeError("MSVC AddressSanitizer runtime DLL is missing.")
        for runtime in runtime_dlls:
            shutil.copy2(runtime, build_dir / runtime.name)
    if input_hashes(sources) != before_hashes:
        raise RuntimeError("C build inputs changed during compilation; rebuild against stable sources.")
    (build_dir / "build_info.json").write_text(json.dumps({
        "compiler": str(cc), "mode": mode, "address_sanitizer": sanitize, "sources": [str(path) for path in sources],
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs_sha256": before_hashes,
        "executable_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize", action="store_true", help="Enable AddressSanitizer and copy MSVC runtime into the isolated build folder.")
    parser.add_argument("--mode", type=int, choices=(0, 1), default=1)
    parser.add_argument("--compiler", choices=("auto", "msvc", "gcc"), default="auto")
    args = parser.parse_args()
    try:
        print(build(args.mode, args.compiler, args.sanitize))
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
