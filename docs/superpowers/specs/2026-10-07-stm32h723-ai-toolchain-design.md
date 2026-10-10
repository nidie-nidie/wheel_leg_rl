# STM32H723 AI Deployment Toolchain Design

## Objective

Prepare this Windows 10 workstation for a reproducible deployment path from a future DreamWaQ checkpoint to an STM32H723VGT6 target. The setup must be usable before the final checkpoint and final motor-control firmware are selected.

## Scope

The toolchain covers:

- checkpoint-compatible PyTorch loading and fixed-shape ONNX export;
- ONNX Runtime numerical comparison;
- ST Edge AI analysis, validation, and C-code generation;
- Keil MDK/Arm Compiler 6 command-line builds for STM32H723VGT6;
- STM32CubeMX project inspection and STM32CubeH7 firmware packages;
- STM32CubeProgrammer/ST-Link discovery and flashing;
- an isolated STM32H723 AI smoke-test project using a representative `[1,125] -> [1,6]` network.

The setup does not select a production checkpoint, freeze the final PACE/control firmware, energize motors, or claim the 50 Hz real-time target has passed. Those are later deployment acceptance stages.

## Installation Layout

- Vendor applications use their standard Windows installation directories on `C:`.
- Keil packs and compilers remain under the Keil-managed pack/compiler directories.
- Python deployment files and generated reports remain under `E:\wheel_leg_rl-main\wheelleg_dreamwaq\deployment_toolchain`.
- The active DreamWaQ training environment at `E:\wheel_leg_rl-main\wheelleg_dreamwaq\.venv` is not modified while formal training is running.

## Selected Approach

Use a command-line-first workflow with GUI tools available for inspection:

1. Install Keil MDK with Arm Compiler 6 support.
2. Install the STM32H7 device pack required by the existing project, initially `Keil.STM32H7xx_DFP.4.1.3`.
3. Install STM32CubeMX and STM32CubeH7 firmware package `1.11.2` without regenerating the existing PACE IOC file.
4. Install STM32CubeProgrammer and the ST-Link USB driver.
5. Install STM32Cube AI Studio/ST Edge AI Core and expose its CLI through a checked configuration file rather than relying only on global `PATH` state.
6. Create an isolated Python 3.11 deployment environment with pinned CPU inference/export dependencies. It may read checkpoint files but does not share mutable packages with the training environment.
7. Generate and export a representative DreamWaQ-shaped FP32 network with fixed input `[1,125]` and output `[1,6]`.
8. Validate PyTorch against ONNX Runtime, then validate ONNX through ST Edge AI.
9. Generate C sources and build a separate STM32H723 smoke-test firmware from the Keil command line.
10. When an H723 board is connected, enumerate ST-Link, flash the smoke firmware, and compare board output with stored golden vectors.

FP32 is the baseline. INT8 quantization is considered only if generated memory or measured timing does not leave adequate margin.

## Version Policy

- Record every installed application, compiler, device pack, firmware package, Python package, and executable path in a machine-readable manifest.
- Prefer the current official vendor release unless the existing STM32 project requires an exact older component.
- Preserve `STM32H7xx_DFP 4.1.3` and STM32CubeH7 `1.11.2` alongside newer versions when supported.
- Never silently migrate or regenerate the existing `wheel_leg_motor_PACE.ioc` or Keil project.

## Automation Artifacts

The configuration phase will produce:

- an environment manifest containing versions and executable paths;
- a PowerShell environment audit that reports PASS/FAIL per dependency;
- a repeatable ONNX export and parity command;
- a representative ONNX model and deterministic golden vectors;
- repeatable ST Edge AI analyze/validate/generate commands;
- a command-line Keil build entry point;
- a command-line STM32CubeProgrammer probe/flash entry point;
- setup notes for the few unavoidable GUI, license, login, and UAC steps.

## Acceptance Criteria

The workstation setup passes when:

1. The audit finds Python 3.11, PyTorch, ONNX, ONNX Runtime, Keil, Arm Compiler 6, STM32H7 DFP, CubeMX, STM32CubeH7, ST Edge AI Core, CubeProgrammer, and ST-Link support.
2. The representative model exports to ONNX and passes the defined PC numerical tolerance.
3. ST Edge AI analyzes, validates, and generates C code for STM32H723VGT6 without unsupported operators.
4. The isolated Keil smoke project builds from the command line and produces AXF and HEX files.
5. The generated reports capture Flash, RAM, operation count, and estimated or measured inference timing.
6. With hardware connected, CubeProgrammer detects the ST-Link and can program and verify the smoke firmware.

## User Interaction Boundaries

The user may need to approve Windows UAC prompts, accept vendor license terms, sign into vendor download services, choose the appropriate Keil license, connect SWD/USB, power the board, and control motor-side safety. All other repeatable software actions are handled through scripts or documented commands.

## Later Checkpoint Replacement

When the production DreamWaQ checkpoint is selected, the smoke model is replaced by a fixed-batch production ONNX export. The same validation, ST Edge AI generation, Keil build, flashing, golden-vector comparison, and timing pipeline is reused without reinstalling the toolchain.
