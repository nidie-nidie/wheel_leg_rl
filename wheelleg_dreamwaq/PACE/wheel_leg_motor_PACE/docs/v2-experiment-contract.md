# PACE V2 experiment and content identity

This document describes the production C implementation. It does not certify motor operation, mechanical limits or zero calibration. Hardware axis signs and CAN IDs preserve the existing code confirmed by the user. Excitation multipliers remain separate from hardware axis signs.

## Wire contract

Common-header version is 2. Samples stay 126 bytes and capture slots stay 128 bytes. V2 StageConfig is exactly 128 bytes; historical V1 StageConfig is 76 bytes. Readers must distinguish the explicit version.

V2 retains offsets 10..71 of the original stage payload. All numbers use explicit little-endian encoding:

|Offset|Field|Type / meaning|
|---|---|---|
|72|role|u8: DIAGNOSTIC=0, FIT=1, VALIDATION=2, STATIC=3, STOP=4|
|73|group_mask|u8: motors receiving excitation; holding joints still have commands and feedback|
|74|segment_id|u16: recipe position 1..26, distinct from the category stage_id|
|76|action_update_rate_hz|u16: commissioning uses 50 Hz|
|78|command_hold_method|u8: ZERO_ORDER=1|
|79|center_source|u8: INITIAL_FEEDBACK=1, VERIFIED_CONTRACT=2|
|80|phase_rad|float32|
|84|dm_center_rad[4]|4 float32: the actual centers used in this session|
|100|excitation_multiplier[6]|6 float32|
|124|CRC32|u32 covering the preceding 124 bytes|

The 58-byte session header declares timestamp_timebase=1 (TIM5_ISR_SERVICE) at offset13, and control_rate_hz=500 at offset32. These are MCU event-service timestamps, not motor receipt or execution times.

PACE order is `jIJ,jIO,jAG,jAB,jwheel_left,jwheel_right`. DreamWaQ order is `jIJ,jIO,jAB,jAG,jwheel_left,jwheel_right`. Map by joint name.

## Low-amplitude 26-segment commissioning

All stages use a 50 Hz target grid with zero-order hold, active-motor CAN commands at 500 Hz, holding-motor commands at 100 Hz, and 250 Hz snapshots. Leg gains remain Kp20/Kd0.6, dq_des=0 and tau_ff=0. The recipe lasts 211 seconds, plus the 10-second autostart delay and the enable sequence.

|Segment|Category / role|Group and amplitude|
|---|---|---|
|1|STATIC / STATIC|Legs hold initial-feedback centers; zero wheel torque; 2s|
|2,3|DM_FIT / FIT|Left legs [+1,-1,0,0], .075/.15 rad; 12s each|
|4,5|DM_FIT / FIT|Right legs [0,0,-1,+1], .075/.15 rad; 12s each|
|6,7|DM_FIT / FIT|Original bilateral [+1,-1,-1,+1], .075/.15 rad; 12s each|
|8,9|DM_FIT / FIT|Second explicit bilateral recipe [+1,-1,+1,-1], .075/.15 rad; 12s each; no axis sign change|
|10..13|DM_VALIDATION / VALIDATION|The four groups above, .12 rad and 6s each; phase=.47+group*.23|
|14,15;16,17|LK_TORQUE / FIT|Left/right wheel, .05/.10 Nm; 8s each|
|18,19|LK_TORQUE / VALIDATION|Left/right wheel, .075 Nm; 6s each|
|20,21;22,23|LK_VELOCITY / FIT|Left/right wheel, 2/4 rad/s; 8s each|
|24,25|LK_VELOCITY / VALIDATION|Left/right wheel, 3 rad/s; 6s each|
|26|STOP / STOP|Leg hold and zero wheel torque for 1s; the app then performs confirmed shutdown|

Right-wheel excitation multiplier -1 preserves the original recipe. Validation combines a reverse chirp with an independent 1.37 Hz tone, normalized by 1.35 so amplitude remains an upper bound. FIT and VALIDATION are independent complete segments. Wheel-validation phase is .71+wheel*.31, with start frequency FIT*1.5 and end frequency FIT*1.1.

Targets use `floor(elapsed_us/20000)*20000`. All CAN resends within that interval have the same target. After a skipped scheduling cycle only the current target is sent; historical commands are not replayed. Internal target_update_time_us is the recipe grid, never a measured motor_execution_time.

Begin requires valid online feedback from every motor within 30ms and rejects future timestamps. Commissioning centers come from initial feedback and remain fixed during the session. These centers are not zero calibration. The wire position envelope is checked at begin; real mechanism limits still require staged hardware validation.

## Recipe hash

`pace_experiment_config_hash` uses CRC32/ISO-HDLC (init FFFFFFFF, reflected polynomial EDB88320, xorout FFFFFFFF). It never hashes C struct padding. Scalar values below are serialized as little-endian u32:

1. Serialization version2, session_type, contract_version, stage_count, control_rate500, sample_rate250, status_rate10, CAN bitrate, UART baud and declared motor_manifest_hash.
2. For each manifest entry in PACE order: canonical_index, motor_type, hardware_slot, device_id, command_can_id, feedback_can_id, position_sign, velocity_sign, torque_sign, position_zero_rad, gear_ratio, actuator_family and calibration_verified. Signed signs are extended to 32 bits; floats use their IEEE754 raw bits. Then mujoco_joint UTF8 byte length (u32) and bytes, and isaac_joint length and bytes.
3. Config dm_kp and dm_kd IEEE754 bits, and safety_limit_set_id (u32).
4. For each configured stage, the production V2 encoder produces offsets10..123 (114 bytes), using config_seq=stage_index+1, experiment_config_hash=0, and declared configuration centers. INITIAL_FEEDBACK centers are declared zero; VERIFIED_CONTRACT centers are verified references. Feed these 114 bytes into the CRC. Exclude common frame headers, frame CRCs and the hash itself.

Runtime feedback centers cannot change the recipe hash declared before begin. Actual centers are carried by recorded stage frames and bound by the immutable raw SHA256. Sidecars must retain expected and actual fields separately and cannot override online data.

Validation requires a matching content hash, finite values, unique sequential segment IDs, role/mode/group agreement, bounded commissioning amplitudes, valid command rates and CAN budget. Total recipe duration must be below half the uint32 time range, so wraparound forward time remains distinguishable from old ticks.

## Final contract boundary

`pace_experiment_default_config` constructs commissioning only. `pace_experiment_final_config` is a separate bool entry. The default ready=0 rejects it. An enabled contract must provide verified zero, version/content hash, hold method/action rate, four nominal positions, actual Kp/Kd, leg/wheel amplitudes and bandwidth, LK velocity as the main interface, and the implemented default safety-limit set ID1. These values are actually consumed. Relabeling commissioning by changing its hash cannot produce a final session.

Changes to an accepted final contract require a new content hash and replay/hardware validation. This implementation never promotes the simulation values 120/4, .35 rad or 25 rad/s to automatic commissioning settings.

## Native regression

`PACE/tests/test_experiment_contract.c` compiles with production experiment/excitation/DM/frame/CRC/motor_order and is called by host_debug. It verifies 26 stages and independent holdouts, content-sensitive identity, invalid configurations, 50 Hz holds and CAN resends, real V1/V2 CRC/length checks, uint32 wraparound, old ticks after STOP, fresh initial feedback and bounded validation excitation. It does not prove real motors are mechanically stopped.
