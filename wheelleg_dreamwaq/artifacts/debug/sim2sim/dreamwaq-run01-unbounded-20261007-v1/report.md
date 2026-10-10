# DreamWaQ Run-01 Unbounded MuJoCo Diagnosis

> Debug-only, non-physical evidence. Not eligible for formal evaluation or ranking.

## Configuration

- Policy: frozen DreamWaQ run-01 actor.
- Timing: MuJoCo `0.001 s x 20`, 50 Hz policy control.
- Runtime action clipping: disabled.
- Leg target clipping: disabled.
- Active effort clipping: disabled.
- Active velocity guard: disabled.
- MJCF joint, actuator control, and actuator force limits: disabled.
- Existing diagnostic termination conditions: retained.

## Result

| Profile | Completed | Mean survival | Maximum raw action | Maximum applied torque |
| --- | ---: | ---: | ---: | ---: |
| Formal limits | 0/8 | 0.095250 | 4.595667 | 18.000000 Nm |
| Unbounded | 0/8 | 0.029250 | 7.920795 | 30.813679 Nm |

The unbounded policy survived only 12-16 valid ticks per scenario. Six
scenarios terminated on base height and two on tilt. Removing the limits made
the failure substantially faster; it did not reveal a hidden recovery action.

The result is consistent with a saturated policy demanding increasingly large
commands after the physical state starts diverging. The formal limits were
containing that feedback rather than causing the original divergence.

## Viewer

The interactive nominal-stand viewer was launched with automatic reset after
each retained terminal pose. Metadata and events are stored in `viewer/`.

Evaluation report hash: `EEF28D45F39AEA7E634B03821CAEE35904D0973556AF009CE54EB2FE25659100`
