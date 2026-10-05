# Motor Manifest v1

This manifest fixes the actuator order shared by firmware, raw data, MuJoCo,
Isaac, and later RL code. Array position is always the canonical index.

| Index | Name | Type | Hardware slot | Device ID | Command CAN ID | Feedback CAN ID | Joint |
|---:|---|---|---:|---:|---:|---:|---|
| 0 | `L_front` | DM8009 | `DM[0]` | 1 | `0x001` | `0x011` | `jIJ` |
| 1 | `L_rear` | DM8009 | `DM[1]` | 2 | `0x002` | `0x012` | `jIO` |
| 2 | `R_rear` | DM8009 | `DM[2]` | 3 | `0x003` | `0x013` | `jAG` |
| 3 | `R_front` | DM8009 | `DM[3]` | 6 | `0x006` | `0x016` | `jAB` |
| 4 | `L_wheel` | LK9025 | `LK[0]` | 4 | `0x144` | `0x144` | `jwheel_left` |
| 5 | `R_wheel` | LK9025 | `LK[1]` | 5 | `0x145` | `0x145` | `jwheel_right` |

`Config/pace_motor_manifest.h` is the firmware source of truth. The initial
signs are `+1`, zeros are `0 rad`, and `calibration_verified` is false. Those
values must be verified with the unloaded robot before commissioning data can
be treated as usable identification data.

LK device IDs are 4 and 5. LK command and feedback frames use the protocol
offset `0x140`, so the actual bus identifiers are `0x144` and `0x145`.
