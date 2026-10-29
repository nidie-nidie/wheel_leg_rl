from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose, isfinite, sqrt

from .metrics import (
    AttemptContext,
    AttemptStatistics,
    EventStatistics,
    ObservationEndKind,
    RecoveryMetrics,
)


def _duration_steps(duration_s: float, dt: float) -> int:
    ratio = duration_s / dt
    steps = round(ratio)
    if steps < 1 or not isclose(ratio, steps, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Supervisor durations must be whole positive control steps")
    return steps


class ControlMode(str, Enum):
    LOCOMOTION = "LOCOMOTION"
    RECOVERY = "RECOVERY"


@dataclass(frozen=True)
class SupervisorConfig:
    dt: float = 0.02
    fall_tilt_rad: float = 0.9
    fall_hold_s: float = 0.2
    ready_hold_s: float = 1.0
    zero_stand_hold_s: float = 2.0
    recovery_budget_s: float = 8.0
    event_budget_s: float = 10.0
    stable_tilt_rad: float = 0.25
    angular_speed_limit: float = 0.5
    leg_error_limit: float = 0.2
    leg_speed_limit: float = 1.0
    wheel_speed_limit: float = 2.0
    height_range: tuple[float, float] = (0.17, 0.25)

    def __post_init__(self) -> None:
        positive = (
            self.dt, self.fall_tilt_rad, self.fall_hold_s, self.ready_hold_s,
            self.zero_stand_hold_s, self.recovery_budget_s, self.event_budget_s,
            self.stable_tilt_rad, self.angular_speed_limit, self.leg_error_limit,
            self.leg_speed_limit, self.wheel_speed_limit,
        )
        if any(not isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Supervisor scalar parameters must be finite and positive")
        if len(self.height_range) != 2 or not all(isfinite(value) for value in self.height_range):
            raise ValueError("Height range must contain two finite values")
        if self.height_range[0] >= self.height_range[1]:
            raise ValueError("Height range must be increasing")
        if self.stable_tilt_rad >= self.fall_tilt_rad:
            raise ValueError("Stable tilt must be below fall tilt")
        for value in (
            self.fall_hold_s, self.ready_hold_s, self.zero_stand_hold_s,
            self.recovery_budget_s, self.event_budget_s,
        ):
            _duration_steps(value, self.dt)
        if self.recovery_budget_s > self.event_budget_s:
            raise ValueError("Recovery budget cannot exceed event budget")


@dataclass(frozen=True)
class SupervisorMeasurements:
    frame_id: int
    tilt_rad: float
    angular_velocity: tuple[float, float, float]
    leg_position_error: tuple[float, float, float, float]
    leg_velocity: tuple[float, float, float, float]
    wheel_velocity: tuple[float, float]
    com_height_m: float
    left_wheel_support: bool
    right_wheel_support: bool
    body_or_leg_support: bool
    action_compatible: bool
    compatibility_blockers: tuple[str, ...] = ()
    physical_failure_reason: str | None = None
    finite: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int) or self.frame_id < 0:
            raise ValueError("frame_id must be a nonnegative integer")
        for name, size in (
            ("angular_velocity", 3), ("leg_position_error", 4),
            ("leg_velocity", 4), ("wheel_velocity", 2),
        ):
            if len(getattr(self, name)) != size:
                raise ValueError(f"{name} must have {size} entries")
        labels = self.compatibility_blockers
        if any(not isinstance(label, str) or not label for label in labels):
            raise ValueError("Compatibility blockers must be nonempty strings")
        if self.action_compatible and labels:
            raise ValueError("Compatible actions cannot have blocking differences")
        if not self.action_compatible and not labels:
            raise ValueError("Incompatible actions require explicit blocker labels")


@dataclass(frozen=True)
class SupervisorUpdate:
    frame_id: int | None
    mode: ControlMode
    transition: str | None
    fault_reason: str | None
    halt_requested: bool
    capture_heading: bool
    zero_command: bool
    attempt_ended: AttemptStatistics | None
    event_finished: bool
    event_id: int | None
    attempt_id: int | None
    event_steps: int
    recovery_steps: int
    attempt_steps: int
    physical_hold_steps: int
    joint_confirm_steps: int
    zero_stand_hold_steps: int


def physical_conditions(
    measurement: SupervisorMeasurements,
    config: SupervisorConfig = SupervisorConfig(),
) -> dict[str, bool]:
    return {
        "tilt": 0 <= measurement.tilt_rad < config.stable_tilt_rad,
        "angular_velocity": sqrt(sum(value * value for value in measurement.angular_velocity))
        < config.angular_speed_limit,
        **{
            f"leg_position/{index}": abs(value) < config.leg_error_limit
            for index, value in enumerate(measurement.leg_position_error)
        },
        **{
            f"leg_velocity/{index}": abs(value) < config.leg_speed_limit
            for index, value in enumerate(measurement.leg_velocity)
        },
        **{
            f"wheel_velocity/{index}": abs(value) < config.wheel_speed_limit
            for index, value in enumerate(measurement.wheel_velocity)
        },
        "height": config.height_range[0] <= measurement.com_height_m <= config.height_range[1],
        "left_wheel_support": measurement.left_wheel_support,
        "right_wheel_support": measurement.right_wheel_support,
        "body_or_leg_support": not measurement.body_or_leg_support,
    }


class RecoverySupervisor:
    def __init__(
        self,
        config: SupervisorConfig = SupervisorConfig(),
        *,
        standalone: bool = False,
        initial_recovery: bool = False,
        context: AttemptContext | None = None,
        metrics: RecoveryMetrics | None = None,
    ) -> None:
        self.config = config
        self.standalone = standalone
        self.context = context or AttemptContext(stage="independent" if standalone else "combined")
        self.metrics = metrics or RecoveryMetrics(
            dt=config.dt, ready_steps=_duration_steps(config.ready_hold_s, config.dt)
        )
        if self.metrics.dt != config.dt or self.metrics.ready_steps != _duration_steps(
            config.ready_hold_s, config.dt
        ):
            raise ValueError("Metrics timing must match supervisor timing")
        self.mode = ControlMode.LOCOMOTION
        self.fault_reason: str | None = None
        self.completed = False
        self.event: EventStatistics | None = None
        self.attempt: AttemptStatistics | None = None
        self.fall_hold_steps = 0
        self.zero_stand_hold_steps = 0
        self._last_frame_id: int | None = None
        if standalone or initial_recovery:
            self._begin_recovery(self.context)

    def snapshot(self) -> SupervisorUpdate:
        return self._update()

    def checkpoint_metrics_state(self, *, reason: str = "checkpoint_resume", seal_archive: bool = True) -> dict:
        return self.metrics.checkpoint_state(attempt=self.attempt, event=self.event,
                                             reason=reason, seal_archive=seal_archive)

    def post_step(
        self,
        measurement: SupervisorMeasurements,
        *,
        manual_recovery: bool = False,
        context: AttemptContext | None = None,
    ) -> SupervisorUpdate:
        if self.fault_reason is not None:
            raise RuntimeError("Fault is latched; explicit reenable is required")
        if self.completed:
            raise RuntimeError("Standalone recovery is complete; start a new episode explicitly")
        if self._last_frame_id is not None and measurement.frame_id != self._last_frame_id + 1:
            raise ValueError("Each consecutive action post-step must be committed exactly once")
        self._last_frame_id = measurement.frame_id
        previous_mode = self.mode
        if self.event is not None:
            self.event.event_steps += 1
            if previous_mode is ControlMode.RECOVERY:
                self.event.recovery_steps += 1
        failure = self._measurement_failure(measurement)
        if failure is not None:
            # 失败状态不参加成功确认，实际发生的动作保留在逐步日志中。
            if previous_mode is ControlMode.RECOVERY:
                self._observe_attempt(measurement, failure=failure)
            return self._fault(failure)

        conditions = physical_conditions(measurement, self.config)
        physical = all(conditions.values())
        self.fall_hold_steps = (
            self.fall_hold_steps + 1 if measurement.tilt_rad > self.config.fall_tilt_rad else 0
        )
        fallen = self.fall_hold_steps >= _duration_steps(self.config.fall_hold_s, self.config.dt)
        if previous_mode is ControlMode.RECOVERY:
            self._observe_attempt(measurement)
            ready = self.attempt.joint_confirm_steps >= self.metrics.ready_steps
            if ready and self.standalone:
                ended = self._finish_attempt("standalone_ready", ObservationEndKind.COMPLETED, True)
                self._finish_event("standalone_ready", ObservationEndKind.COMPLETED)
                self.completed = True
                return self._update(attempt_ended=ended, event_finished=True)
            event_deadline = self.event.event_steps >= _duration_steps(
                self.config.event_budget_s, self.config.dt
            )
            recovery_deadline = self.event.recovery_steps >= _duration_steps(
                self.config.recovery_budget_s, self.config.dt
            )
            if event_deadline:
                return self._fault("event_deadline")
            if ready:
                ended = self._finish_attempt("handoff", ObservationEndKind.COMPLETED, True)
                self.mode = ControlMode.LOCOMOTION
                self.zero_stand_hold_steps = 0
                self.fall_hold_steps = 0
                return self._update(
                    transition="RECOVERY->LOCOMOTION", capture_heading=True, attempt_ended=ended
                )
            if recovery_deadline:
                return self._fault("recovery_deadline")
            return self._update()

        if self.event is not None:
            previous_hold = self.zero_stand_hold_steps
            self.zero_stand_hold_steps = previous_hold + 1 if physical and not fallen else 0
            if previous_hold > 0 and self.zero_stand_hold_steps == 0:
                self.event.zero_stand_break_count += 1
            zero_complete = self.zero_stand_hold_steps >= _duration_steps(
                self.config.zero_stand_hold_s, self.config.dt
            )
            if zero_complete:
                self._finish_event("zero_stand_complete", ObservationEndKind.COMPLETED)
                self.zero_stand_hold_steps = 0
                return self._update(event_finished=True)
            if self.event.event_steps >= _duration_steps(self.config.event_budget_s, self.config.dt):
                return self._fault("event_deadline")
        if fallen or manual_recovery:
            if self.event is not None and self.event.recovery_steps >= _duration_steps(
                self.config.recovery_budget_s, self.config.dt
            ):
                return self._fault("recovery_budget_exhausted_on_fall")
            if self.event is not None:
                self.event.recurrent_fall_count += 1
            self._begin_recovery(context or self.context)
            return self._update(transition="LOCOMOTION->RECOVERY")
        return self._update()

    def interrupt(self, reason: str = "manual_stop") -> SupervisorUpdate:
        if self.fault_reason is not None or self.completed:
            raise RuntimeError("Cannot interrupt an already finished run")
        ended = None
        if self.attempt is not None:
            ended = self._finish_attempt(reason, ObservationEndKind.CENSORED)
        had_event = self.event is not None
        if had_event:
            self._finish_event(reason, ObservationEndKind.CENSORED)
        self.fault_reason = reason
        return self._update(attempt_ended=ended, event_finished=had_event)

    def reenable(self, *, context: AttemptContext | None = None) -> SupervisorUpdate:
        if self.fault_reason is None:
            raise RuntimeError("Explicit reenable requires a latched fault or interruption")
        self.fault_reason = None
        self.completed = False
        self.fall_hold_steps = 0
        self.zero_stand_hold_steps = 0
        if context is not None:
            self.context = context
        self.mode = ControlMode.LOCOMOTION
        if self.standalone:
            self._begin_recovery(self.context)
        return self._update()

    def new_episode(self, *, context: AttemptContext | None = None) -> SupervisorUpdate:
        if not self.standalone or not self.completed:
            raise RuntimeError("new_episode requires a completed standalone episode")
        self.completed = False
        if context is not None:
            self.context = context
        self._begin_recovery(self.context)
        return self._update()

    def _begin_recovery(self, context: AttemptContext) -> None:
        if self.mode is ControlMode.RECOVERY and self.attempt is not None:
            raise RuntimeError("Cannot restart an active recovery attempt")
        expected_stage = "independent" if self.standalone else "combined"
        if context.stage != expected_stage:
            raise ValueError(f"Attempt stage must be {expected_stage}")
        if self.event is None:
            self.event = self.metrics.begin_event(expected_stage)
        self.attempt = self.metrics.begin_attempt(self.event, context)
        self.mode = ControlMode.RECOVERY
        self.fall_hold_steps = 0
        self.zero_stand_hold_steps = 0

    def _observe_attempt(self, measurement: SupervisorMeasurements, *, failure: str | None = None) -> None:
        conditions = physical_conditions(measurement, self.config)
        physical = all(conditions.values()) and failure is None
        blockers = [label for label, passed in conditions.items() if not passed]
        blockers.extend(measurement.compatibility_blockers)
        if failure is not None:
            blockers.append(f"failure/{failure}")
        self.attempt.observe(
            frame_id=measurement.frame_id,
            physical=physical,
            action_compatible=measurement.action_compatible,
            blockers=blockers,
        )

    def _measurement_failure(self, measurement: SupervisorMeasurements) -> str | None:
        values = (
            measurement.tilt_rad, measurement.com_height_m,
            *measurement.angular_velocity, *measurement.leg_position_error,
            *measurement.leg_velocity, *measurement.wheel_velocity,
        )
        if not measurement.finite or not all(isfinite(value) for value in values):
            return "nonfinite_state"
        return measurement.physical_failure_reason

    def _finish_attempt(
        self,
        reason: str,
        kind: ObservationEndKind,
        accepted_handoff: bool = False,
    ) -> AttemptStatistics:
        attempt = self.attempt
        attempt.finish(reason, kind, accepted_handoff=accepted_handoff)
        self.attempt = None
        self.metrics.archive_attempt(attempt)
        return attempt

    def _finish_event(self, reason: str, kind: ObservationEndKind) -> None:
        self.event.end_reason = reason
        self.event.observation_end_kind = kind
        self.metrics.archive_event(self.event)
        self.event = None

    def _fault(self, reason: str) -> SupervisorUpdate:
        ended = None
        if self.attempt is not None:
            ended = self._finish_attempt(reason, ObservationEndKind.TASK_FAILURE)
        had_event = self.event is not None
        if had_event:
            self._finish_event(reason, ObservationEndKind.TASK_FAILURE)
        self.fault_reason = reason
        return self._update(attempt_ended=ended, event_finished=had_event)

    def _update(
        self,
        *,
        transition: str | None = None,
        capture_heading: bool = False,
        attempt_ended: AttemptStatistics | None = None,
        event_finished: bool = False,
    ) -> SupervisorUpdate:
        event = self.event or self.metrics.last_event
        attempt = self.attempt or attempt_ended
        return SupervisorUpdate(
            self._last_frame_id,
            self.mode,
            transition,
            self.fault_reason,
            self.fault_reason is not None,
            capture_heading,
            self.mode is ControlMode.RECOVERY or self.event is not None,
            attempt_ended,
            event_finished,
            event.event_id if event is not None else None,
            attempt.attempt_id if attempt is not None else None,
            event.event_steps if event is not None else 0,
            event.recovery_steps if event is not None else 0,
            len(attempt.steps) if attempt is not None else 0,
            attempt.physical_hold_steps if attempt is not None else 0,
            attempt.joint_confirm_steps if attempt is not None else 0,
            self.zero_stand_hold_steps,
        )
