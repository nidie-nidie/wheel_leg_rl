# Native verification of production PACE C

Run from the PACE directory:

```powershell
python host_debug/run_checks.py
```

The scripts need only Python's standard library and a C compiler. They do not install packages or use the reinforcement-learning environment. On Windows, the build discovers MSVC through vswhere or PACE_VS_ROOT, with the installed E:/pow-mining-lab/.toolchain/vs2022 toolchain as a fallback, and imports the x64 vcvarsall environment. GCC/cc is supported on Linux (`--compiler gcc`); the verified platform is Windows MSVC.

For bounds checking, run `python host_debug/run_checks.py --sanitize`. This compiles and runs both modes under AddressSanitizer (including actual HAL-length 64-byte CAN FD payload copies before rejection), writes separate `*-asan` binaries/fixtures and `native_report_asan.json`, and copies only the compiler's ASan runtime DLLs to the isolated build folders. Use `--sanitize --verify-report` to verify that report.

The two binaries compile the real Board CAN/time/UART sources, motor codecs/registry, experiment/excitation/safety, app/capture/transport, and protocol encoders/parser/CRC. Only STM32 HAL and CMSIS boundaries are replaced. Production task wrappers and tests share `pace_app_poll()` and `pace_app_transport_poll()`; the harness does not recreate the experiment or app state machine.

Tests exercise:

- 26-stage commissioning completion, recipe hash, roles/masks, independent validation, 50 Hz target holds, 500/100 Hz command scheduling, V1/V2 frame validation and actual 128-byte StageConfig encoding.
- Enqueue versus later CAN TX-event confirmation; TIM5 service timestamp with an independent FDCAN epoch, 16-bit and 32-bit wraps, future-age defense, short/64-byte FD/remote/extended/unknown/opcode-invalid receive rejection and RX loss statistics.
- Ten-second passive delay and the real paired motor-enable order at 1 ms intervals.
- Real capture queue FIFO/full behavior; UART retry/pop ownership; app scheduler, feedback/CAN loss, queue overflow and UART failure paths.
- Safe-shutdown confirmation before terminal footer, ordinary TX events that cannot substitute for safe commands, and the explicit two-second failure footer.
- Active-mode fragmented host commands, final-contract lock, UART DMA recovery in task context, STOP/ABORT and another session after a drained footer.

`build/<mode>/build_info.json` records every C/header input (including fakes and Config), build scripts, compiler and executable SHA256. Inputs must remain identical before and after compilation and across both completed test runs. `build/native_report.json` embeds these records and fixture sizes/hashes; an interrupted rerun invalidates the previous success. Run `python host_debug/run_checks.py --verify-report` to compare the saved success with the current files. `build/fixture_passive.raw` and `build/fixture_active.raw` contain full UART streams emitted by the production C app and encoders. The PC collector can use them through FakeSerial. A single fixture can be regenerated with:

```powershell
host_debug/build/passive/pace_native.exe --fixture host_debug/build/fixture_passive.raw
```

Fixtures intentionally use steady, synthetic feedback at near-zero motor coordinates. They verify the acquisition/protocol path, not motor physics or parameter identification. They are development artifacts and must not be mixed with physical acquisition sessions.

The fake advances TIM5 explicitly, calls the real HAL callbacks, and queues CAN until a separate confirmation operation. UART output is an in-memory byte sink with injected HAL failures. It does not model electrical CAN arbitration, the wireless bridge, UART wire occupancy, preemption, DMA/cache coherency or the mounted robot's closed-loop dynamics. Native success therefore does not replace the documented physical acceptance checks at 921600 baud.
