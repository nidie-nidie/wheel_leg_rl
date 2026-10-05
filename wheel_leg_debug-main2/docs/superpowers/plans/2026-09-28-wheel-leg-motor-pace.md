# wheel_leg_motor_PACE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task with review checkpoints. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an isolated STM32H723 Keil firmware project that first supports a low-risk commissioning session and later produces a final six-channel identification dataset after the RL low-level actuator contract is frozen.

**Architecture:** The firmware is split into platform, motor protocol, PACE experiment, capture, and transport modules. It supports `COMMISSIONING_SESSION` for short, low-amplitude engineering validation and `FINAL_IDENTIFICATION_SESSION` for the later production dataset. All fitting and Isaac/MuJoCo replay remain on the PC; the four DM leg motors and two LK 9025 wheel motors share one canonical order but use separate actuator-model families.

**Tech Stack:** STM32H723VGTx, STM32 HAL, FDCAN3 Classic CAN at 1 Mbit/s with Tx Event FIFO, UART7 at 921600 baud, minimal FreeRTOS, Keil MDK-ARM, a 126-byte fixed sample frame with CRC-32, Python PC decoder and PACE/Isaac/MuJoCo adapters.

---

## Task 1: Create the isolated project skeleton

**Files:**
- Create: `wheel_leg_motor_PACE/MDK-ARM/wheel_leg_motor_PACE.uvprojx`
- Create: `wheel_leg_motor_PACE/MDK-ARM/wheel_leg_motor_PACE.uvoptx`
- Create: `wheel_leg_motor_PACE/MDK-ARM/startup_stm32h723xx.s`
- Create: `wheel_leg_motor_PACE/Core/Inc/*`
- Create: `wheel_leg_motor_PACE/Core/Src/*`
- Create: `wheel_leg_motor_PACE/Board/Inc/*`
- Create: `wheel_leg_motor_PACE/Board/Src/*`
- Create: `wheel_leg_motor_PACE/Motor/Inc/*`
- Create: `wheel_leg_motor_PACE/Motor/Src/*`
- Create: `wheel_leg_motor_PACE/PACE/Inc/*`
- Create: `wheel_leg_motor_PACE/PACE/Src/*`
- Create: `wheel_leg_motor_PACE/Protocol/Inc/*`
- Create: `wheel_leg_motor_PACE/Protocol/Src/*`
- Create: `wheel_leg_motor_PACE/Config/*`
- Copy: required STM32 HAL, CMSIS, FreeRTOS, linker script, and board-generated files from `wheel_leg_debug-main/rm_test-dev`

- [x] Copy only startup, HAL, CMSIS, clock, GPIO, FDCAN3, UART7, DMA, timer, interrupt, and minimal FreeRTOS files from `rm_test-dev`.
- [x] Add only the new `Board`, `Motor`, `PACE`, and `Protocol` groups to the Keil project.
- [x] Do not add chassis, VMC, INS, remote, referee, music, image, or jump-control groups.
- [x] Set the target name and output artifact to `wheel_leg_motor_PACE`.

**Verification:**

Run:

```powershell
Test-Path wheel_leg_motor_PACE/MDK-ARM/wheel_leg_motor_PACE.uvprojx
Select-String -Path wheel_leg_motor_PACE/MDK-ARM/wheel_leg_motor_PACE.uvprojx -Pattern '<TargetName>wheel_leg_motor_PACE</TargetName>'
```

Expected: both commands find the project and target name, and the project XML contains no `ChassisL_Task.c`, `ChassisR_Task.c`, or `VMC_Calc.c` entries.

## Task 2: Freeze the canonical motor manifest

**Files:**
- Create: `wheel_leg_motor_PACE/Config/pace_motor_manifest.h`
- Create: `wheel_leg_motor_PACE/docs/motor_manifest_v1.md`
- Create: `wheel_leg_motor_PACE/Protocol/Inc/pace_motor_order.h`
- Test: `wheel_leg_motor_PACE/Protocol/Src/pace_motor_order_test.c`

- [x] Define indices `L_FRONT=0`, `L_REAR=1`, `R_REAR=2`, `R_FRONT=3`, `L_WHEEL=4`, `R_WHEEL=5`.
- [x] Define the current CAN IDs and hardware array slots in one table.
- [x] Define motor type, actuator family, sign, zero, and joint-name fields.
- [x] Add compile-time constants for `PACE_MOTOR_COUNT == 6`.
- [x] Add a unit-testable lookup function that maps `(can_id, motor_type)` to canonical index.

Required logical mapping:

```text
0 L_front  jIJ          DM[0]
1 L_rear   jIO          DM[1]
2 R_rear   jAG          DM[2]
3 R_front  jAB          DM[3]
4 L_wheel  jwheel_left  LK[0]
5 R_wheel  jwheel_right LK[1]
```

**Verification:**

Run a host-side parser or compile-time test that verifies all six indices are unique, all CAN IDs are unique, and the right-leg order remains rear then front.

Expected: one and only one manifest entry exists for every canonical index and CAN ID.

## Task 3: Extract the DM/LK motor protocol layer

**Files:**
- Create: `wheel_leg_motor_PACE/Motor/Inc/pace_motor_types.h`
- Create: `wheel_leg_motor_PACE/Motor/Inc/dm_motor.h`
- Create: `wheel_leg_motor_PACE/Motor/Inc/lk_motor.h`
- Create: `wheel_leg_motor_PACE/Motor/Src/dm_motor.c`
- Create: `wheel_leg_motor_PACE/Motor/Src/lk_motor.c`
- Create: `wheel_leg_motor_PACE/Motor/Src/pace_motor_registry.c`
- Create: `wheel_leg_motor_PACE/Motor/Inc/pace_motor_registry.h`
- Reference: `wheel_leg_debug-main/rm_test-dev/Components/Device/Src/Motor.c`

- [x] Move only DM command packing, DM feedback decoding, LK command packing, and LK feedback decoding.
- [x] Keep protocol-native raw feedback integers and decoded physical values in the motor state.
- [x] Make each command packer return a `pace_encoded_command_t` containing CAN ID, DLC, the final 8-byte payload, protocol-native command integers, and a saturation flag.
- [x] Keep requested floating-point commands separate from encoded command integers; fitting data must never substitute the requested value for the encoded value.
- [x] Add `online`, `last_rx_timestamp_us`, `rx_age_us`, `last_tx_timestamp_us`, and `tx_age_us` to every motor state.
- [x] Route every received CAN ID through `pace_motor_manifest`.
- [x] Keep command packers side-effect free; only the CAN port may update pending or Tx-confirmed command state.
- [x] Do not import controller gains, VMC, or chassis task dependencies.

**Verification:**

Build a host-side protocol test with known CAN payloads from the existing decoder. Expected: decoded position, velocity, effort/current, temperature, and online state match the reference values within the existing quantization limits.

## Task 4: Add board ports and deterministic time

**Files:**
- Create: `wheel_leg_motor_PACE/Board/Inc/pace_can_port.h`
- Create: `wheel_leg_motor_PACE/Board/Src/pace_can_port.c`
- Create: `wheel_leg_motor_PACE/Board/Inc/pace_uart_port.h`
- Create: `wheel_leg_motor_PACE/Board/Src/pace_uart_port.c`
- Create: `wheel_leg_motor_PACE/Board/Inc/pace_time.h`
- Create: `wheel_leg_motor_PACE/Board/Src/pace_time.c`
- Modify: `wheel_leg_motor_PACE/Core/Src/fdcan.c`
- Modify: `wheel_leg_motor_PACE/Core/Src/usart.c`
- Modify: `wheel_leg_motor_PACE/Core/Src/tim.c`

- [x] Expose non-blocking FDCAN transmit and receive callbacks.
- [x] Configure FDCAN3 as Classic CAN at 1 Mbit/s and preserve `AutoRetransmission = ENABLE`; do not copy FDCAN1's disabled setting.
- [x] Allocate at least eight Tx Event FIFO elements, set transmitted headers to `FDCAN_STORE_TX_EVENTS`, and assign a `MessageMarker` that is unique among outstanding transmissions.
- [x] Make the transmit API return the result of `HAL_FDCAN_AddMessageToTxFifoQ` and increment explicit queued, queue-full, and HAL-error counters.
- [x] Maintain a bounded pending-command table keyed by CAN identifier and `MessageMarker`; move an encoded command into the last-confirmed state only after its Tx Event is drained.
- [x] Enable Tx Event new-data/full/lost notifications, drain events with `HAL_FDCAN_GetTxEvent`, and retain the event timestamp and type.
- [x] Preserve `FDCAN_RxHeaderTypeDef.RxTimestamp` when routing feedback instead of passing only identifier and payload.
- [x] Expose UART7 DMA transmit completion and error status.
- [x] Expose a monotonic microsecond timestamp.
- [x] Map the 16-bit FDCAN Tx/Rx timestamp counter into the MCU monotonic time domain and handle counter wrap explicitly.
- [x] Configure a 500 Hz experiment tick without using blocking delays in the capture path.
- [x] Expose per-channel Tx/Rx counts, Tx Event FIFO loss, Tx FIFO high-water mark, CAN protocol/error state, enqueue-to-Tx latency, command period, and feedback age.

**Verification:**

Run the 500 Hz tick and UART DMA test with motors disconnected. Then use FDCAN internal loopback or a second CAN node to send a known sequence of markers. Expected: every successful enqueue has exactly one matching Tx Event, Tx/Rx timestamps remain monotonic after wrap expansion, forced queue saturation increments the failure counter, and no unconfirmed command is exposed as the last-sent command.

## Task 5: Implement the binary frame protocol

**Files:**
- Create: `wheel_leg_motor_PACE/Protocol/Inc/pace_frame.h`
- Create: `wheel_leg_motor_PACE/Protocol/Src/pace_frame.c`
- Create: `wheel_leg_motor_PACE/Protocol/Inc/pace_crc.h`
- Create: `wheel_leg_motor_PACE/Protocol/Src/pace_crc.c`
- Create: `wheel_leg_motor_PACE/Protocol/Inc/pace_command.h`
- Create: `wheel_leg_motor_PACE/Protocol/Src/pace_command.c`
- Test: `wheel_leg_motor_PACE/Protocol/tests/test_pace_frame.py`
- Test: `wheel_leg_motor_PACE/Protocol/tests/test_bandwidth_budget.py`

- [x] Define the common 10-byte little-endian header with magic bytes `0x50 0x41`, protocol version, frame type, total frame length, and one stream-wide `uint32` sequence.
- [x] Define frame types `0x01` session header, `0x02` sample, `0x03` stage config, `0x04` status, `0x05` event, and `0x06` footer.
- [x] Implement the exact 126-byte sample layout from `PROJECT_IMPLEMENTATION.md`: 26-byte header, four 16-byte DM blocks, two 16-byte LK blocks, and a 4-byte CRC-32/ISO-HDLC.
- [x] Serialize every field by explicit offset; add a compile-time size assertion and do not transmit a packed or unpacked C structure directly.
- [x] Store DM protocol-native command/feedback integers and LK command, encoder/turn, speed, and current integers without another lossy conversion.
- [x] Store `tx_age_us` and `rx_age_us` for every motor in every sample; saturate at 65535 us and clear the corresponding valid bit when no timestamp exists.
- [x] Move `command_mode`, per-motor command rate, DM `Kp/Kd` raw codes, excitation parameters, and limits into stage config frames referenced by `config_seq`.
- [x] Implement the 64-byte base status layout from `PROJECT_IMPLEMENTATION.md`, using packed 4-bit motor states and `uint16` per-channel Tx/Rx deltas since the previous status frame.
- [x] Support a bounded TLV extension region of 0..64 bytes, place CRC-32 at the actual frame end, and reject any status frame outside the 64..128-byte range.
- [x] Emit status at 10 Hz; detect counter-delta saturation, emit an event, and mark the session statistics incomplete.
- [x] Add start, stop, configure, status, and abort host commands.
- [x] Reject malformed length, version, CRC, and sequence values.
- [x] Emit a session footer with total frames, dropped frames, and overflow status.

**Verification:**

Run:

```powershell
python wheel_leg_motor_PACE/Protocol/tests/test_pace_frame.py
python wheel_leg_motor_PACE/Protocol/tests/test_bandwidth_budget.py
```

Expected: every sample encodes to exactly 126 bytes, base and maximally extended status frames encode to 64 and 128 bytes, offset fixtures decode byte-for-byte, corrupted CRC and malformed lengths are rejected, a missing sequence is reported, and the worst case `126 * 500 + 128 * 10 = 64280 byte/s` remains below the 70% UART7 limit of `64512 byte/s`.

## Task 6: Implement the capture buffer and transport task

**Files:**
- Create: `wheel_leg_motor_PACE/PACE/Inc/pace_capture.h`
- Create: `wheel_leg_motor_PACE/PACE/Src/pace_capture.c`
- Create: `wheel_leg_motor_PACE/PACE/Inc/pace_transport.h`
- Create: `wheel_leg_motor_PACE/PACE/Src/pace_transport.c`
- Modify: `wheel_leg_motor_PACE/Core/Src/freertos.c`

- [x] Use a single-producer/single-consumer ring buffer between the 500 Hz sampler and UART DMA task.
- [x] Never block the sampler while UART DMA is busy.
- [x] Set a persistent overflow flag and stop safely when the buffer cannot accept a frame.
- [x] Snapshot all six channels at 500 Hz even when inactive motors use a lower command rate.
- [x] Put only the latest Tx-confirmed encoded command and its age into each motor block; a successfully generated or enqueued command is not sufficient.
- [x] Emit the 10 Hz status frame without delaying or replacing a sample frame.
- [x] Preserve `sequence`, `config_seq`, valid masks, and event ordering across the ring buffer and UART DMA boundary.
- [x] Keep raw binary capture independent from PC-side CSV conversion.

**Verification:**

Run a synthetic six-channel producer with 126-byte samples at 500 Hz, maximally extended 128-byte status frames at 10 Hz, and a UART consumer limited to 92160 byte/s. Expected: the nominal stream runs without overflow or sequence loss; reducing the consumer below 64280 byte/s deterministically sets overflow, emits an event/footer when possible, and stops without silently dropping a frame.

## Task 7: Implement excitation, experiment states, and safety

**Files:**
- Create: `wheel_leg_motor_PACE/PACE/Inc/pace_experiment.h`
- Create: `wheel_leg_motor_PACE/PACE/Src/pace_experiment.c`
- Create: `wheel_leg_motor_PACE/PACE/Inc/pace_excitation.h`
- Create: `wheel_leg_motor_PACE/PACE/Src/pace_excitation.c`
- Create: `wheel_leg_motor_PACE/PACE/Inc/pace_safety.h`
- Create: `wheel_leg_motor_PACE/PACE/Src/pace_safety.c`
- Create: `wheel_leg_motor_PACE/PACE/Inc/pace_app.h`
- Create: `wheel_leg_motor_PACE/PACE/Src/pace_app.c`
- Test: `wheel_leg_motor_PACE/PACE/tests/test_can_schedule.py`

- [x] Implement `SAFE_IDLE`, `CONFIGURED`, `ARMED`, `RUNNING`, `STOPPING`, `COMPLETE`, and `FAULT`.
- [x] Encode `COMMISSIONING_SESSION` and `FINAL_IDENTIFICATION_SESSION` in the session header and state machine.
- [x] Implement static, DM-fit, DM-validation, LK-torque, LK-velocity, and stop stages.
- [x] Keep the snapshot rate fixed at 500 Hz while scheduling command rates independently for each motor.
- [x] Use at most 100 Hz per motor in the static stage, four DM channels at 500 Hz plus each LK at no more than 100 Hz in DM stages, and two LK channels at 500 Hz plus each DM at no more than 100 Hz in LK stages.
- [x] Compute a conservative stage bus load using actual DLC and one feedback frame per command; target at most 65% and reject configurations above 70% before entering `ARMED`.
- [x] Emit the stage config before the first sample that references its new `config_seq`.
- [x] Generate the primary four-DM excitation as a simultaneous mechanically symmetric position chirp with `dq_des=0`, `Kp=20.0`, `Kd=0.6`, and `tau_ff=0`.
- [x] Generate DM validation trajectories with different frequency, amplitude, or phase while keeping `Kp=20.0`, `Kd=0.6`, and `tau_ff=0`.
- [x] Generate separate LK 9025 torque-mode and velocity-mode excitation stages; never merge their samples without preserving `command_mode`.
- [x] Apply independent emergency position, velocity, effort, temperature, feedback-age, and duration abort thresholds.
- [x] Treat an abort-threshold hit as an invalid session instead of silently clamping the command and retaining the affected samples for fitting.
- [x] Send safe zero/off commands on abort, timeout, CAN failure, UART overflow, or watchdog warning.
- [x] Log stage transitions and safety events in the raw stream.

**Verification:**

Run:

```powershell
python wheel_leg_motor_PACE/PACE/tests/test_can_schedule.py
```

Expected: the static, DM, and LK schedules remain below the 65% target under the documented 130-bit estimate; an all-six-motors-at-500-Hz configuration computes about 78% and is rejected; illegal state transitions are rejected; abort always reaches `FAULT` or `SAFE_IDLE`; and every terminal path emits a session footer.

## Task 8: Build the PC decoder and normalized dataset

**Files:**
- Create: `wheel_leg_motor_PACE_PC/README.md`
- Create: `wheel_leg_motor_PACE_PC/pace_raw/decoder.py`
- Create: `wheel_leg_motor_PACE_PC/pace_raw/schema.py`
- Create: `wheel_leg_motor_PACE_PC/pace_raw/manifest.py`
- Create: `wheel_leg_motor_PACE_PC/pace_raw/normalize.py`
- Create: `wheel_leg_motor_PACE_PC/tests/test_decoder.py`
- Reference: `a1_motor_id_flow_20260726_clean/docs/FLOW.md`

- [x] Decode all six frame types, enforce the 126-byte sample length, verify CRC, and report sequence, `config_seq`, and timestamp discontinuities.
- [x] Preserve every protocol-native command and feedback integer before converting it to physical units.
- [x] Reconstruct command mode, DM `Kp/Kd`, command rate, and excitation parameters by joining each sample to its referenced stage config.
- [x] Reconstruct per-motor Tx/Rx timestamps from `sample_time_us`, `tx_age_us`, and `rx_age_us`; keep invalid and saturated ages explicit.
- [x] Validate the six-channel manifest before converting data.
- [x] Produce one normalized dataset with raw motor coordinates and explicit signs/zeros.
- [x] Split leg-DM and wheel-LK views without deleting the original raw data.
- [x] Export CSV for inspection and NPZ/PT for fitting.
- [x] Preserve session type, stage boundaries, and `DIAGNOSTIC`, `PROVISIONAL_FIT`, `FIT`, and `VALIDATION` roles.

**Verification:**

Run the decoder against fixtures containing exact offset sentinel values, a 32-bit sample-time wrap, one missing stage config, one malformed 125-byte sample, and one corrupted CRC. Expected: valid data converts without changing raw integers, time unwrap is monotonic, and every malformed or context-free sample is reported and excluded from fitting.

## Task 9: Add provisional/final actuator fitting and simulator adapters

**Files:**
- Create: `wheel_leg_motor_PACE_PC/fit/fit_leg_dm.py`
- Create: `wheel_leg_motor_PACE_PC/fit/fit_wheel_lk.py`
- Create: `wheel_leg_motor_PACE_PC/models/motor_model_schema.json`
- Create: `wheel_leg_motor_PACE_PC/isaac/leg_dm_adapter.py`
- Create: `wheel_leg_motor_PACE_PC/isaac/wheel_lk_adapter.py`
- Create: `wheel_leg_motor_PACE_PC/mujoco/actuator_replay.py`
- Create: `wheel_leg_motor_PACE_PC/validation/crossval.py`
- Reference: `a1_motor_id_flow_20260726_clean/pace-sim2real/source/pace_sim2real/pace_sim2real/optim/cma_es.py`
- Reference: `mujoco_control_extract/sim/main_mujoco.c`

- [x] Reuse the generic PACE optimizer for the leg-DM parameter family.
- [x] Keep encoder bias, delay, armature, viscous friction, and static/dynamic friction explicit.
- [x] Define a separate LK wheel model for velocity/effort behavior.
- [x] Reject DM fitting input whose command metadata does not report `Kp=20.0`, `Kd=0.6`, `dq_des=0`, and `tau_ff=0` for the baseline model.
- [x] Tag every fitted manifest with `model_maturity = provisional` or `model_maturity = final` based on the source session type.
- [x] Write one versioned fitted-model manifest consumed by both simulators.
- [x] Make MuJoCo replay use the fitted actuator model rather than direct controller torque only.
- [x] Make Isaac replay use the same canonical order, signs, zeros, and delays.
- [x] Report per-channel RMSE, P95 error, phase/delay error, saturation rate, and validation-stage error.

**Verification:**

Run fitting on a synthetic six-channel fixture with known parameters. Expected: the fitted model recovers the fixture parameters within the declared tolerance and both simulator adapters consume the same manifest without reordering channels.

## Task 10: Commissioning hardware acceptance

**Files:**
- Create: `wheel_leg_motor_PACE/docs/commissioning_checklist.md`
- Create: `wheel_leg_motor_PACE/docs/session_format_v1.md`
- Create: `wheel_leg_motor_PACE/README.md`

- [ ] Verify all six CAN IDs and directions on the physical robot.
- [ ] Verify zero command and emergency stop with motors unloaded.
- [ ] Verify FDCAN3 reports auto retransmission enabled, Tx Event FIFO active, no event loss, and no ignored transmit return code.
- [ ] Verify the UART stream uses 126-byte samples at 500 Hz and 64..128-byte status frames at 10 Hz; sustained encoded throughput must remain at or below 64512 byte/s even with 128-byte status frames.
- [ ] Verify the selected stage schedule has a conservative CAN load at or below 65%; configurations above 70% must fail to arm.
- [ ] Record per-channel enqueue-to-Tx latency, command period, feedback age, Tx/Rx frame count, queue failures, CAN errors, and UART overflow counters.
- [ ] Require no active 500 Hz command interval greater than or equal to 4 ms and no enqueue-to-Tx Event latency greater than or equal to 2 ms.
- [ ] Run a short, low-amplitude `COMMISSIONING_SESSION` with DM baseline gains `20.0/0.6` and zero torque feed-forward.
- [ ] Verify the commissioning raw file contains static, DM-fit, DM-validation, every executed LK diagnostic stage, and stop stages.
- [ ] Verify no overflow, missing sequence, Tx Event loss, unreported enqueue failure, or unmarked invalid sample.
- [ ] Verify raw file, firmware build ID, manifest version, and experiment configuration are linked.
- [ ] Store the raw binary file as the source of truth and generate derived files separately.
- [ ] Generate a provisional fitting report without publishing the result as the final sim2real actuator model.

**Verification:**

Run the commissioning checklist against one hardware session. Milestone A is acceptable only when all safety checks pass, UART and CAN budgets pass, every fitted command is Tx-confirmed, the decoder reports no unaccounted frame loss, timing statistics are attached to the session, and the PC path completes one provisional fit.

## Task 11: Freeze the final contract and run final identification

**Files:**
- Create: `wheel_leg_motor_PACE/docs/final_identification_contract.md`
- Create: `wheel_leg_motor_PACE/docs/release_checklist.md`
- Modify: `wheel_leg_motor_PACE/Config/pace_experiment_config.h`
- Modify: `wheel_leg_motor_PACE_PC/models/motor_model_schema.json`

- [ ] Record the approved action update rate, command hold method, DM nominal pose, expected DM position amplitude, required excitation bandwidth, DM torque-feedforward policy, final LK command mode, and emergency abort envelope.
- [x] Refuse to arm `FINAL_IDENTIFICATION_SESSION` unless every final-contract field is present and its configuration hash is stored in the session header.
- [ ] Run one final session containing independent DM/LK `FIT` and `VALIDATION` stages covering the approved low-level action envelope.
- [ ] Fit the final DM and LK models from `FIT` stages only and evaluate them on held-out `VALIDATION` stages.
- [ ] Publish one `model_maturity = final` manifest consumed by both Isaac and MuJoCo adapters.
- [ ] Record per-channel RMSE, P95, delay/phase error, saturation rate, direction errors, and all invalid/aborted stage counts.

**Verification:**

Run the final release checklist against the hardware session, fitted manifest, Isaac replay, and MuJoCo replay. Expected: the configuration hash matches across all artifacts, validation metrics are reported separately from fitting metrics, and neither simulator reorders the six canonical channels.

## Execution order

Implement tasks in order. Tasks 1-10 form Milestone A and may proceed before the complete RL project exists. Task 11 is gated only by the finalized low-level actuator contract, not by a trained checkpoint, network architecture, observation design, or reward design. Do not delete or overwrite any original raw binary session when generating derived datasets.

## Definition of done

Milestone A, commissioning-ready, is complete when:

1. The isolated Keil project builds and runs on STM32H723VGTx.
2. The board safely executes a low-amplitude commissioning experiment and captures all six channels.
3. The 126-byte sample protocol stays within the UART budget, and the stage-specific CAN schedule stays within the accepted load limit without unreported Tx/Rx failures.
4. The PC decoder detects corruption, loss, timing-context loss, and order mismatches.
5. The commissioning dataset produces explicitly provisional DM and LK results from Tx-confirmed encoded commands.
6. No chassis-control or simulation implementation is required in the firmware project.

Milestone B, final actuator model, is complete when:

1. The RL low-level actuator contract is versioned and bound to the final session configuration hash.
2. The final raw session contains separate `FIT` and `VALIDATION` stages without unaccounted frame loss.
3. The same canonical dataset produces separate final DM and LK fitted models.
4. Isaac and MuJoCo consume the same final manifest and report validation metrics.
5. Hardware safety regression and the final release checklist pass.
