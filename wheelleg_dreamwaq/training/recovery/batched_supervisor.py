from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from enum import IntEnum

import torch

from .supervisor import SupervisorConfig, _duration_steps


class ModeCode(IntEnum):
    LOCOMOTION = 0
    RECOVERY = 1


class TransitionCode(IntEnum):
    NONE = 0
    LOCO_TO_RECOVERY = 1
    RECOVERY_TO_LOCO = 2


class EndKind(IntEnum):
    NONE = 0
    COMPLETED = 1
    TASK_FAILURE = 2
    CENSORED = 3


REASON_LABELS = (
    "none", "nonfinite_state", "loop_closure", "joint_position_limit",
    "nonallowed_body_clearance", "excessive_ground_penetration", "standalone_ready",
    "handoff", "event_deadline", "recovery_deadline", "zero_stand_complete",
    "recovery_budget_exhausted_on_fall", "manual_stop", "checkpoint_resume",
    "training_run_end", "closure_error", "manual_reset",
)
REASON_CODES = {label: index for index, label in enumerate(REASON_LABELS)}
BLOCKER_LABELS = (
    "tilt", "angular_velocity", *(f"leg_position/{i}" for i in range(4)),
    *(f"leg_velocity/{i}" for i in range(4)), *(f"wheel_velocity/{i}" for i in range(2)),
    "height", "left_wheel_support", "right_wheel_support", "body_or_leg_support",
    *(f"target_{i}" for i in range(6)), *(f"torque_{i}" for i in range(6)),
    "calibration_unavailable", "failure/nonfinite_state", "failure/loop_closure",
    "failure/joint_position_limit", "failure/nonallowed_body_clearance",
    "failure/excessive_ground_penetration", "failure/closure_error",
)
BLOCKER_BITS = {label: 1 << index for index, label in enumerate(BLOCKER_LABELS)}
_COMPATIBILITY_BITS = sum(1 << index for index in range(16, 29))
_FAILURE_BITS = {code: BLOCKER_BITS[f"failure/{REASON_LABELS[code]}"]
                 for code in (1, 2, 3, 4, 5, 15)}


@dataclass(frozen=True)
class BatchedSupervisorUpdate:
    # 每列形状均为 [num_envs]，缺失的 ID/帧用 -1 表示。
    frame_id: torch.Tensor
    event_id: torch.Tensor
    attempt_id: torch.Tensor
    context_id: torch.Tensor
    generation: torch.Tensor
    mode: torch.Tensor
    transition: torch.Tensor
    fault_reason: torch.Tensor
    halt_requested: torch.Tensor
    capture_heading: torch.Tensor
    zero_command: torch.Tensor
    completed: torch.Tensor
    success: torch.Tensor
    failure: torch.Tensor
    event_active: torch.Tensor
    attempt_active: torch.Tensor
    event_steps: torch.Tensor
    recovery_steps: torch.Tensor
    attempt_steps: torch.Tensor
    physical_hold_steps: torch.Tensor
    joint_confirm_steps: torch.Tensor
    zero_stand_hold_steps: torch.Tensor
    fall_hold_steps: torch.Tensor
    zero_stand_break_count: torch.Tensor
    recurrent_fall_count: torch.Tensor
    attempt_recorded: torch.Tensor
    physical: torch.Tensor
    action_compatible: torch.Tensor
    blocker_mask: torch.Tensor
    attempt_started: torch.Tensor
    event_started: torch.Tensor
    attempt_ended: torch.Tensor
    event_finished: torch.Tensor
    attempt_end_reason: torch.Tensor
    attempt_end_kind: torch.Tensor
    event_end_reason: torch.Tensor
    event_end_kind: torch.Tensor

    def columns(self) -> dict[str, torch.Tensor]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


class BatchedRecoverySupervisor:
    """在设备上批量推进状态机，稀疏处理显式管理操作。

    观测列为倾角、角速度3、腿位差4、腿速度4、轮速度2、高度。
    FP32 样本先提升至 FP64，保持原 CPU Python 浮点比较语义。
    initial_update 含构造时的开始记录；物理重置前先单独提交中断包，
    使旧尝试的结束记录与新尝试的开始记录使用各自明确的 ID。
    """

    _STATE_FIELDS = (
        "standalone", "initial_recovery", "context_id", "active_context_id", "generation",
        "mode", "fault_reason", "completed", "event_active", "attempt_active", "frame_id",
        "event_id", "attempt_id", "next_event_id", "next_attempt_id", "event_steps",
        "recovery_steps", "attempt_steps", "physical_hold_steps", "joint_confirm_steps",
        "zero_stand_hold_steps", "fall_hold_steps", "zero_stand_break_count", "recurrent_fall_count",
    )

    def __init__(self, num_envs: int, device: torch.device | str = "cpu",
                 config: SupervisorConfig = SupervisorConfig(), *,
                 standalone: bool | torch.Tensor = False,
                 initial_recovery: bool | torch.Tensor = False,
                 context_ids: torch.Tensor | None = None,
                 next_event_ids: torch.Tensor | None = None,
                 next_attempt_ids: torch.Tensor | None = None,
                 defer_initial_begin: bool = False) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs, self.config = num_envs, config
        self.device = torch.empty(0, device=device).device
        self.ready_steps = _duration_steps(config.ready_hold_s, config.dt)
        self.fall_steps = _duration_steps(config.fall_hold_s, config.dt)
        self.zero_steps = _duration_steps(config.zero_stand_hold_s, config.dt)
        self.recovery_budget = _duration_steps(config.recovery_budget_s, config.dt)
        self.event_budget = _duration_steps(config.event_budget_s, config.dt)
        self.standalone = self._bool_option(standalone, "standalone")
        self.initial_recovery = self._bool_option(initial_recovery, "initial_recovery")
        self.context_id = self._id_option(context_ids, "context_ids", 0)
        self.active_context_id = self._int(-1)
        for name in self._STATE_FIELDS:
            if not hasattr(self, name):
                setattr(self, name, self._bool() if name in (
                    "completed", "event_active", "attempt_active") else self._int())
        self.frame_id.fill_(-1)
        self.event_id.fill_(-1)
        self.attempt_id.fill_(-1)
        self.next_event_id = self._id_option(next_event_ids, "next_event_ids", 1, minimum=1)
        self.next_attempt_id = self._id_option(next_attempt_ids, "next_attempt_ids", 1, minimum=1)
        packet = self._blank()
        if not defer_initial_begin:
            self._begin(self.standalone | self.initial_recovery, self.context_id, packet)
        self.initial_update = self._packet(packet)

    def _int(self, value: int = 0) -> torch.Tensor:
        return torch.full((self.num_envs,), value, dtype=torch.int64, device=self.device)

    def _bool(self, value: bool = False) -> torch.Tensor:
        return torch.full((self.num_envs,), value, dtype=torch.bool, device=self.device)

    def _column(self, value: torch.Tensor, name: str, dtype: torch.dtype,
                width: int | None = None) -> torch.Tensor:
        shape = (self.num_envs,) if width is None else (self.num_envs, width)
        if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype or value.device != self.device:
            raise ValueError(f"{name} must be {shape} {dtype} on {self.device}")
        return value

    def _bool_option(self, value: bool | torch.Tensor, name: str) -> torch.Tensor:
        if isinstance(value, bool):
            return self._bool(value)
        return self._column(value, name, torch.bool).clone()

    def _id_option(self, value: torch.Tensor | None, name: str, default: int,
                   minimum: int = 0) -> torch.Tensor:
        result = self._int(default) if value is None else self._column(value, name, torch.int64).clone()
        if bool((result < minimum).any()):
            raise ValueError(f"{name} must be at least {minimum}")
        return result

    def _selected(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return self._bool(True)
        if not isinstance(env_ids, torch.Tensor) or env_ids.ndim != 1 or env_ids.dtype != torch.int64:
            raise ValueError("env_ids must be a one-dimensional int64 tensor")
        ids = env_ids.to(self.device)
        if bool(((ids < 0) | (ids >= self.num_envs)).any()) or ids.unique().numel() != ids.numel():
            raise ValueError("env_ids must be unique and in range")
        mask = self._bool()
        mask[ids] = True
        return mask

    def _blank(self) -> dict[str, torch.Tensor]:
        return {name: self._bool() for name in (
            "capture_heading", "success", "failure", "attempt_recorded", "physical",
            "action_compatible", "attempt_started", "event_started", "attempt_ended", "event_finished",
        )} | {name: self._int() for name in (
            "transition", "blocker_mask", "attempt_end_reason", "attempt_end_kind", "event_end_reason", "event_end_kind",
        )}

    def _packet(self, packet: dict[str, torch.Tensor]) -> BatchedSupervisorUpdate:
        has_attempt = self.attempt_active | packet["attempt_ended"]
        columns = {name: getattr(self, name).clone() for name in (
            "frame_id", "event_id", "generation", "mode", "fault_reason", "completed",
            "event_active", "attempt_active", "event_steps", "recovery_steps", "zero_stand_hold_steps",
            "fall_hold_steps", "zero_stand_break_count", "recurrent_fall_count",
        )}
        columns.update({name: torch.where(has_attempt, getattr(self, name), 0) for name in (
            "attempt_steps", "physical_hold_steps", "joint_confirm_steps",
        )})
        columns.update(attempt_id=torch.where(has_attempt, self.attempt_id, -1),
                       context_id=torch.where(has_attempt, self.active_context_id, -1),
                       halt_requested=self.fault_reason != 0,
                       zero_command=(self.mode == ModeCode.RECOVERY) | self.event_active)
        return BatchedSupervisorUpdate(**columns, **packet)

    def snapshot(self) -> BatchedSupervisorUpdate:
        return self._packet(self._blank())

    def _begin(self, mask: torch.Tensor, contexts: torch.Tensor,
               packet: dict[str, torch.Tensor]) -> None:
        fresh = mask & ~self.event_active
        self.event_id = torch.where(fresh, self.next_event_id, self.event_id)
        self.next_event_id += fresh
        for name in ("event_steps", "recovery_steps", "zero_stand_break_count", "recurrent_fall_count"):
            setattr(self, name, torch.where(fresh, 0, getattr(self, name)))
        self.attempt_id = torch.where(mask, self.next_attempt_id, self.attempt_id)
        self.next_attempt_id += mask
        self.active_context_id = torch.where(mask, contexts, self.active_context_id)
        for name in ("attempt_steps", "physical_hold_steps", "joint_confirm_steps", "fall_hold_steps", "zero_stand_hold_steps"):
            setattr(self, name, torch.where(mask, 0, getattr(self, name)))
        self.event_active |= mask
        self.attempt_active |= mask
        self.mode = torch.where(mask, int(ModeCode.RECOVERY), self.mode)
        packet["event_started"] |= fresh
        packet["attempt_started"] |= mask

    def _finish(self, mask: torch.Tensor, reasons: torch.Tensor, kind: EndKind,
                packet: dict[str, torch.Tensor], *, finish_event: bool = True) -> None:
        attempts = mask & self.attempt_active
        packet["attempt_ended"] |= attempts
        packet["attempt_end_reason"] = torch.where(attempts, reasons, packet["attempt_end_reason"])
        packet["attempt_end_kind"] = torch.where(attempts, int(kind), packet["attempt_end_kind"])
        self.attempt_active &= ~mask
        if finish_event:
            events = mask & self.event_active
            packet["event_finished"] |= events
            packet["event_end_reason"] = torch.where(events, reasons, packet["event_end_reason"])
            packet["event_end_kind"] = torch.where(events, int(kind), packet["event_end_kind"])
            self.event_active &= ~mask

    @torch.no_grad()
    def post_step(self, values: torch.Tensor, frame_ids: torch.Tensor, *,
                  wheel_support: torch.Tensor, body_support: torch.Tensor,
                  action_compatible: torch.Tensor, compatibility_mask: torch.Tensor,
                  physical_failure: torch.Tensor, finite: torch.Tensor | None = None,
                  manual_recovery: torch.Tensor | None = None,
                  context_ids: torch.Tensor | None = None) -> BatchedSupervisorUpdate:
        if not isinstance(values, torch.Tensor) or values.shape != (self.num_envs, 15) or values.dtype not in (torch.float32, torch.float64) or values.device != self.device:
            raise ValueError("values must be [num_envs,15] float32/float64 on supervisor device")
        self._column(frame_ids, "frame_ids", torch.int64)
        self._column(wheel_support, "wheel_support", torch.bool, 2)
        for name, value in (("body_support", body_support), ("action_compatible", action_compatible)):
            self._column(value, name, torch.bool)
        for name, value in (("compatibility_mask", compatibility_mask), ("physical_failure", physical_failure)):
            self._column(value, name, torch.int64)
        finite = self._bool(True) if finite is None else self._column(finite, "finite", torch.bool)
        manual = self._bool() if manual_recovery is None else self._column(manual_recovery, "manual_recovery", torch.bool)
        contexts = self.context_id if context_ids is None else self._column(context_ids, "context_ids", torch.int64)
        bad_frame = (frame_ids < 0) | ((self.frame_id >= 0) & (frame_ids != self.frame_id + 1))
        bad_compat = ((compatibility_mask & ~_COMPATIBILITY_BITS) != 0) | (action_compatible != (compatibility_mask == 0))
        valid_failure = physical_failure == 0
        for code in _FAILURE_BITS:
            valid_failure |= physical_failure == code
        invalid = bad_frame | bad_compat | ~valid_failure | (contexts < 0)
        # 正常路径只同步一次校验整批提交，错误路径再定位具体原因。
        if bool((invalid | (self.fault_reason != 0) | self.completed).any()):
            if bool((self.fault_reason != 0).any()):
                raise RuntimeError("Fault is latched; explicit reenable is required")
            if bool(self.completed.any()):
                raise RuntimeError("Standalone recovery is complete; start a new episode explicitly")
            raise ValueError("Invalid frame, compatibility labels, context, or physical failure code")
        data = values.to(torch.float64)
        cfg = self.config
        gyro_square = data[:, 1] * data[:, 1]
        gyro_square = gyro_square + data[:, 2] * data[:, 2]
        gyro_square = gyro_square + data[:, 3] * data[:, 3]
        conditions = [(data[:, 0] >= 0) & (data[:, 0] < cfg.stable_tilt_rad),
                      torch.sqrt(gyro_square) < cfg.angular_speed_limit]
        conditions.extend(data[:, i].abs() < cfg.leg_error_limit for i in range(4, 8))
        conditions.extend(data[:, i].abs() < cfg.leg_speed_limit for i in range(8, 12))
        conditions.extend(data[:, i].abs() < cfg.wheel_speed_limit for i in range(12, 14))
        conditions.extend(((data[:, 14] >= cfg.height_range[0]) & (data[:, 14] <= cfg.height_range[1]),
                           wheel_support[:, 0], wheel_support[:, 1], ~body_support))
        blockers = compatibility_mask.clone()
        physical = self._bool(True)
        for index, passed in enumerate(conditions):
            physical &= passed
            blockers |= (~passed).to(torch.int64) << index
        failure_reason = torch.where(finite & torch.isfinite(data).all(dim=1), physical_failure, 1)
        failed = failure_reason != 0
        physical &= ~failed
        for code, bit in _FAILURE_BITS.items():
            blockers |= (failure_reason == code).to(torch.int64) * bit
        packet = self._blank()
        previous_recovery = self.mode == ModeCode.RECOVERY
        self.frame_id = frame_ids.clone()
        self.event_steps += self.event_active
        self.recovery_steps += self.event_active & previous_recovery
        self.attempt_steps += previous_recovery
        self.physical_hold_steps = torch.where(previous_recovery, torch.where(physical, self.physical_hold_steps + 1, 0), self.physical_hold_steps)
        self.joint_confirm_steps = torch.where(previous_recovery, torch.where(physical & action_compatible, self.joint_confirm_steps + 1, 0), self.joint_confirm_steps)
        packet.update(attempt_recorded=previous_recovery.clone(), physical=physical,
                      action_compatible=action_compatible.clone(), blocker_mask=blockers)
        self.fall_hold_steps = torch.where(~failed, torch.where(data[:, 0] > cfg.fall_tilt_rad, self.fall_hold_steps + 1, 0), self.fall_hold_steps)
        fallen = self.fall_hold_steps >= self.fall_steps
        recovering = previous_recovery & ~failed
        ready = self.joint_confirm_steps >= self.ready_steps
        standalone_done = recovering & ready & self.standalone
        event_deadline = recovering & ~standalone_done & (self.event_steps >= self.event_budget)
        handoff = recovering & ~standalone_done & ~event_deadline & ready
        recovery_deadline = recovering & ~standalone_done & ~event_deadline & ~ready & (self.recovery_steps >= self.recovery_budget)
        failure_reason = torch.where(event_deadline, 8, failure_reason)
        failure_reason = torch.where(recovery_deadline, 9, failure_reason)
        self._finish(standalone_done, self._int(6), EndKind.COMPLETED, packet)
        self.completed |= standalone_done
        self._finish(handoff, self._int(7), EndKind.COMPLETED, packet, finish_event=False)
        self.mode = torch.where(handoff, int(ModeCode.LOCOMOTION), self.mode)
        self.zero_stand_hold_steps = torch.where(handoff, 0, self.zero_stand_hold_steps)
        self.fall_hold_steps = torch.where(handoff, 0, self.fall_hold_steps)
        packet["capture_heading"] = handoff
        packet["success"] = standalone_done | handoff
        packet["transition"] = torch.where(handoff, int(TransitionCode.RECOVERY_TO_LOCO), packet["transition"])
        locomotion = ~previous_recovery & ~failed
        observing_zero = locomotion & self.event_active
        old_zero = self.zero_stand_hold_steps
        new_zero = torch.where(physical & ~fallen, old_zero + 1, 0)
        self.zero_stand_break_count += observing_zero & (old_zero > 0) & (new_zero == 0)
        self.zero_stand_hold_steps = torch.where(observing_zero, new_zero, old_zero)
        zero_done = observing_zero & (self.zero_stand_hold_steps >= self.zero_steps)
        self._finish(zero_done, self._int(10), EndKind.COMPLETED, packet)
        self.zero_stand_hold_steps = torch.where(zero_done, 0, self.zero_stand_hold_steps)
        loco_deadline = observing_zero & ~zero_done & (self.event_steps >= self.event_budget)
        failure_reason = torch.where(loco_deadline, 8, failure_reason)
        enter = locomotion & ~zero_done & ~loco_deadline & (fallen | manual)
        exhausted = enter & self.event_active & (self.recovery_steps >= self.recovery_budget)
        failure_reason = torch.where(exhausted, 11, failure_reason)
        enter &= ~exhausted
        self.recurrent_fall_count += enter & self.event_active
        self._begin(enter, contexts, packet)
        packet["transition"] = torch.where(enter, int(TransitionCode.LOCO_TO_RECOVERY), packet["transition"])
        faults = failure_reason != 0
        self._finish(faults, failure_reason, EndKind.TASK_FAILURE, packet)
        self.fault_reason = torch.where(faults, failure_reason, self.fault_reason)
        packet["failure"] = faults
        return self._packet(packet)

    def interrupt(self, reason: str | int = "manual_stop", *, env_ids: torch.Tensor | None = None) -> BatchedSupervisorUpdate:
        mask = self._selected(env_ids)
        code = REASON_CODES[reason] if isinstance(reason, str) else int(reason)
        if not 0 < code < len(REASON_LABELS):
            raise ValueError("Interrupt reason must be a registered nonzero code")
        if bool((mask & ((self.fault_reason != 0) | self.completed)).any()):
            raise RuntimeError("Cannot interrupt an already finished run")
        packet = self._blank()
        self._finish(mask, self._int(code), EndKind.CENSORED, packet)
        self.fault_reason = torch.where(mask, code, self.fault_reason)
        return self._packet(packet)

    def reenable(self, *, env_ids: torch.Tensor | None = None, context_ids: torch.Tensor | None = None) -> BatchedSupervisorUpdate:
        mask = self._selected(env_ids)
        if bool((mask & (self.fault_reason == 0)).any()):
            raise RuntimeError("Explicit reenable requires a latched fault or interruption")
        return self._restart(mask, context_ids, reset_frame=False, initial=False)

    def new_episode(self, *, env_ids: torch.Tensor | None = None, context_ids: torch.Tensor | None = None) -> BatchedSupervisorUpdate:
        mask = self._selected(env_ids)
        if bool((mask & (~self.standalone | ~self.completed)).any()):
            raise RuntimeError("new_episode requires a completed standalone episode")
        return self._restart(mask, context_ids, reset_frame=False, initial=False)

    def reset(self, env_ids: torch.Tensor | None = None, *, context_ids: torch.Tensor | None = None,
              next_event_ids: torch.Tensor | None = None, next_attempt_ids: torch.Tensor | None = None,
              frame_ids: torch.Tensor | None = None) -> BatchedSupervisorUpdate:
        """提交旧中断包后重置物理回合；ID 参数均为完整长度的列。"""
        mask = self._selected(env_ids)
        if bool((mask & (self.event_active | self.attempt_active)).any()):
            raise RuntimeError("Interrupt an active event before physical reset")
        if frame_ids is not None:
            self._column(frame_ids, "frame_ids", torch.int64)
            if bool((mask & ((frame_ids < -1) | (frame_ids < self.frame_id))).any()):
                raise ValueError("Reset frame must be -1 or the last committed global frame")
        checked_contexts = None if context_ids is None else self._id_option(context_ids, "context_ids", 0)
        checked_ids = {}
        for name, value in (("next_event_id", next_event_ids), ("next_attempt_id", next_attempt_ids)):
            if value is not None:
                checked_ids[name] = self._id_option(value, name, 1, minimum=1)
                if bool((mask & (checked_ids[name] < getattr(self, name))).any()):
                    raise ValueError("Reset cannot reuse an event or attempt ID")
        for name, checked in checked_ids.items():
            setattr(self, name, torch.where(mask, checked, getattr(self, name)))
        update = self._restart(mask, checked_contexts, reset_frame=True, initial=True)
        if frame_ids is not None:
            self.frame_id = torch.where(mask, frame_ids, self.frame_id)
            update = replace(update, frame_id=self.frame_id.clone())
        return update

    def _restart(self, mask: torch.Tensor, contexts: torch.Tensor | None, *, reset_frame: bool, initial: bool) -> BatchedSupervisorUpdate:
        if contexts is not None:
            checked = self._id_option(contexts, "context_ids", 0)
            self.context_id = torch.where(mask, checked, self.context_id)
        packet = self._blank()
        for name in ("fault_reason", "fall_hold_steps", "zero_stand_hold_steps", "mode"):
            setattr(self, name, torch.where(mask, 0, getattr(self, name)))
        self.completed &= ~mask
        if reset_frame:
            self.frame_id = torch.where(mask, -1, self.frame_id)
            self.generation += mask
        begin = mask & (self.standalone | (self.initial_recovery if initial else self._bool()))
        self._begin(begin, self.context_id, packet)
        return self._packet(packet)

    def state_dict(self) -> dict:
        return {"schema": "BatchedRecoverySupervisorV1", "num_envs": self.num_envs,
                "config": asdict(self.config), "reason_labels": REASON_LABELS,
                "blocker_labels": BLOCKER_LABELS,
                "state": {name: getattr(self, name).detach().cpu().clone() for name in self._STATE_FIELDS}}

    def load_state_dict(self, payload: dict) -> None:
        if payload.get("schema") != "BatchedRecoverySupervisorV1" or payload.get("num_envs") != self.num_envs or payload.get("config") != asdict(self.config) or tuple(payload.get("reason_labels", ())) != REASON_LABELS or tuple(payload.get("blocker_labels", ())) != BLOCKER_LABELS:
            raise ValueError("Batched supervisor checkpoint contract mismatch")
        state = payload.get("state", {})
        if set(state) != set(self._STATE_FIELDS):
            raise ValueError("Batched supervisor checkpoint fields mismatch")
        loaded = {}
        for name in self._STATE_FIELDS:
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.shape != (self.num_envs,) or value.dtype != getattr(self, name).dtype:
                raise ValueError(f"Invalid supervisor checkpoint column: {name}")
            loaded[name] = value.to(self.device).clone()
        invalid = self._bool()
        for name in ("context_id", "generation", "event_steps", "recovery_steps", "attempt_steps",
                     "physical_hold_steps", "joint_confirm_steps", "zero_stand_hold_steps",
                     "fall_hold_steps", "zero_stand_break_count", "recurrent_fall_count"):
            invalid |= loaded[name] < 0
        for name in ("frame_id", "event_id", "attempt_id", "active_context_id"):
            invalid |= loaded[name] < -1
        invalid |= (loaded["mode"] < 0) | (loaded["mode"] > 1)
        invalid |= (loaded["fault_reason"] < 0) | (loaded["fault_reason"] >= len(REASON_LABELS))
        invalid |= (loaded["next_event_id"] < 1) | (loaded["next_event_id"] <= loaded["event_id"])
        invalid |= (loaded["next_attempt_id"] < 1) | (loaded["next_attempt_id"] <= loaded["attempt_id"])
        invalid |= loaded["recovery_steps"] > loaded["event_steps"]
        invalid |= loaded["attempt_steps"] > loaded["recovery_steps"]
        invalid |= loaded["physical_hold_steps"] > loaded["attempt_steps"]
        invalid |= loaded["joint_confirm_steps"] > loaded["physical_hold_steps"]
        invalid |= loaded["event_active"] & ((loaded["fault_reason"] != 0) | loaded["completed"] | (loaded["event_id"] < 1))
        invalid |= loaded["attempt_active"] & (~loaded["event_active"] | (loaded["mode"] != ModeCode.RECOVERY) | (loaded["attempt_id"] < 1) | (loaded["active_context_id"] < 0))
        invalid |= (loaded["mode"] == ModeCode.RECOVERY) & ~loaded["attempt_active"] & (loaded["fault_reason"] == 0) & ~loaded["completed"]
        invalid |= loaded["completed"] & (~loaded["standalone"] | (loaded["fault_reason"] != 0) | loaded["event_active"] | loaded["attempt_active"] | (loaded["mode"] != ModeCode.RECOVERY))
        invalid |= loaded["standalone"] != self.standalone
        invalid |= loaded["initial_recovery"] != self.initial_recovery
        if bool(invalid.any()):
            raise ValueError("Invalid batched supervisor checkpoint state")
        for name, value in loaded.items():
            setattr(self, name, value)
        self.initial_update = self._packet(self._blank())
