# DreamWaQ Run-01 Broad Sim2Sim Diagnosis

> Debug-only evidence. Not eligible for formal evaluation or ranking.

## Timing Gate

| Policy / timing | Completed | Mean survival | Action saturation |
| --- | ---: | ---: | ---: |
| dreamwaq_formal_1ms | 0/8 | 0.095250 | 0.701504 |
| dreamwaq_hold_5ms | 0/8 | 0.096000 | 0.701751 |
| dreamwaq_sync_5ms | 0/8 | 0.103500 | 0.702887 |
| phase1r_run03_formal_1ms | 5/8 | 0.655000 | 0.233150 |
| phase1r_run03_sync_5ms | 0/8 | 0.326750 | 0.273517 |

## Earliest Boundary

- Reset current/history max error: 4.59999995e-07.
- Initial identical-action target max error: 2.98023224e-07.
- Initial PD reference max error: 5.52596046e-05 Nm.
- First material single-action divergence: {'elapsed_time_ms': 5, 'signal': 'base_angular_velocity_control_post_step'}.

## Isolation Results

- Zero action first material divergence: {'control_tick': 1, 'signal': 'actor_obs_current_pre_step'}.
- Open-loop replay action max error: 0.
- Open-loop first material divergence: {'control_tick': 0, 'signal': 'base_angular_velocity_control_post_step_pre_reset'}.
- Closed-loop first material divergence: {'control_tick': 0, 'signal': 'base_angular_velocity_control_post_step_pre_reset'}.
- Closed-loop action max difference: 0.349195421.
- Trace sidecar hashes, dimensions, row counts, and substep continuity: verified.
- Timing/baseline report hashes and all scenario CSV hashes: verified.

## Conclusion

The reset, history, first actor output, action target, and initial PD reference align. A material state difference appears inside the first physical interval, before the second policy inference. Zero-action and identical-action replay retain the divergence, so CENet is responding to a plant mismatch rather than creating it. Closed-loop policy feedback then amplifies that mismatch. Matching the MuJoCo timestep to Isaac does not solve the failure and substantially degrades Phase 1R run-03, so it must not replace the formal timing contract.

Report hash: `FB8BF81A889B166AEFD0CE3F9D676A766FE9A3421EEE5931975232FF24EE8655`
