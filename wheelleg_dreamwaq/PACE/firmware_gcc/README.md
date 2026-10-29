# PACE STM32H723 Arm GNU build

Run from the workspace root:

```powershell
python -B wheelleg_dreamwaq/PACE/firmware_gcc/build.py
```

The builder discovers `arm-none-eabi-gcc` on PATH or under `PACE/build/toolchain`; `--toolchain <root>` or `PACE_ARM_GCC=<gcc executable>` can select another installation. It compiles current production files from the same `GROUPS` and `INCLUDE_PATHS` as `wheel_leg_motor_PACE/tools/generate_keil_project.py`, substitutes the pinned GNU startup and RTOS port, and does not reuse Keil objects or old HEX files.

Outputs are in `PACE/build/firmware_gcc/commissioning/`: fresh ELF, Intel HEX, binary, map, build.log and build_manifest.json. The manifest records compiler/flags, input and output SHA256, upstream sources/licenses, linker layout and ELF DMA symbols. It also records compiler individual stack-frame estimates, which do not replace measured task stack high-water marks or worst-case interrupt nesting.

The only supported build here is low-amplitude passive commissioning: 10-second autostart delay, 26 segments, gains20/.6, final-ready forced0. Compilation never contacts the serial port or flashes hardware.

The linker retains the original Keil declared capacities: Flash0x08000000/1MB, DTCM0x20000000/128KB and AXI0x24000000/128KB. The `.pace_dma` section starts at AXI0x24000000 because DMA1/2 cannot access DTCM. The builder checks ELF symbols, not only linker text. Newlib heap allowance is1KB and MSP allowance8KB; FreeRTOS dynamic allocation uses its existing32KB static heap and task-stack configuration.

`vendor/sources.json` pins the official ST CMSIS-device-h7 v1.10.4 startup at commit `faccfec37f82f7a1319c21638111b0f7335de7fe` (matching the local CMSIS device version), and FreeRTOS-Kernel V10.3.1-kernel-only GCC ARM_CM7/r0p1 at commit `88e32327e975ddde97c390bc5b6c1f8e7d9d239e` (matching the existing kernel). Source hashes are checked before every build, and upstream Apache-2.0/MIT license files are included. Original HAL, CMSIS, kernel and RVDS vendor files are preserved.

The build verifies compile/link, memory bounds and artifact identity. Real startup, clocks, CAN pins, IRQ/RTOS timing, UART DMA/cache behavior, motor motion and mechanical stop still require board validation using the software release procedure.
