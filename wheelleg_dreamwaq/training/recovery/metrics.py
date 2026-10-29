from __future__ import annotations

from array import array
from collections.abc import Sequence
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from math import ceil, isfinite
from statistics import median
from typing import Any, Iterable

from .metrics_archive import MetricsArchive


class ObservationEndKind(str, Enum):
    COMPLETED = "completed"
    TASK_FAILURE = "task_failure"
    CENSORED = "censored"


@dataclass(frozen=True)
class AttemptContext:
    initial_category: str = "unclassified"
    motion_kind: str = "unknown"
    randomization_profile: str = "unspecified"
    stage: str = "combined"

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in asdict(self).values()):
            raise ValueError("Attempt context fields must be nonempty strings")


@dataclass(frozen=True)
class AttemptStep:
    step: int
    frame_id: int
    physical: bool
    action_compatible: bool
    physical_hold_steps: int
    joint_confirm_steps: int
    blockers: tuple[str, ...]
    confirmation_reset_reasons: tuple[str, ...]


class CompactAttemptSteps(Sequence):
    def __init__(self) -> None:
        self.frame_ids = array("Q")
        self.physical_holds = array("H")
        self.joint_holds = array("H")
        self.blocker_masks = array("Q")
        self.flags = array("B")
        self.labels: list[str] = []
        self._label_indices: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self))) ]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        mask = self.blocker_masks[index]
        blockers = tuple(sorted(label for position, label in enumerate(self.labels) if mask & (1 << position)))
        flags = self.flags[index]
        previous_joint = index > 0 and self.flags[index - 1] == 3
        return AttemptStep(index + 1, self.frame_ids[index], bool(flags & 1), bool(flags & 2),
                           self.physical_holds[index], self.joint_holds[index], blockers,
                           blockers if previous_joint and flags != 3 else ())

    def append(self, record: AttemptStep) -> None:
        mask = 0
        for label in record.blockers:
            if label not in self._label_indices:
                if len(self.labels) >= 64:
                    raise ValueError("An attempt supports at most 64 distinct blocker labels")
                self._label_indices[label] = len(self.labels)
                self.labels.append(label)
            mask |= 1 << self._label_indices[label]
        self.frame_ids.append(record.frame_id)
        self.physical_holds.append(record.physical_hold_steps)
        self.joint_holds.append(record.joint_confirm_steps)
        self.blocker_masks.append(mask)
        self.flags.append(int(record.physical) | (int(record.action_compatible) << 1))

    @property
    def payload_bytes(self) -> int:
        return sum(len(value) * value.itemsize for value in (
            self.frame_ids, self.physical_holds, self.joint_holds, self.blocker_masks, self.flags
        ))

    def to_payload(self) -> dict:
        return {"schema": "RecoveryCompactAttemptStepsV1", "labels": list(self.labels),
                "frame_ids": self.frame_ids.tolist(), "physical_holds": self.physical_holds.tolist(),
                "joint_holds": self.joint_holds.tolist(), "blocker_masks": self.blocker_masks.tolist(),
                "flags": self.flags.tolist()}

    @classmethod
    def from_payload(cls, payload: dict) -> CompactAttemptSteps:
        expected = {"schema", "labels", "frame_ids", "physical_holds", "joint_holds", "blocker_masks", "flags"}
        if set(payload) != expected or payload["schema"] != "RecoveryCompactAttemptStepsV1":
            raise ValueError("Unsupported compact attempt step schema")
        size = len(payload["frame_ids"])
        if any(len(payload[name]) != size for name in expected - {"schema", "labels"}):
            raise ValueError("Compact attempt columns have different lengths")
        labels = payload["labels"]
        if len(labels) > 64 or len(set(labels)) != len(labels) or any(not isinstance(label, str) or not label for label in labels):
            raise ValueError("Compact attempt blocker labels are invalid")
        result = cls()
        result.labels = list(labels)
        result._label_indices = {label: index for index, label in enumerate(labels)}
        for name, typecode in (("frame_ids", "Q"), ("physical_holds", "H"), ("joint_holds", "H"),
                               ("blocker_masks", "Q"), ("flags", "B")):
            setattr(result, name, array(typecode, payload[name]))
        if any(value > 3 for value in result.flags) or any(mask >> len(labels) for mask in result.blocker_masks):
            raise ValueError("Compact attempt flag or blocker mask is invalid")
        return result


@dataclass
class AttemptStatistics:
    event_id: int
    attempt_id: int
    context: AttemptContext
    dt: float
    ready_steps: int
    first_physical_step: int | None = None
    physical_ready_step: int | None = None
    joint_ready_step: int | None = None
    handoff_step: int | None = None
    physical_without_compatibility_steps: int = 0
    max_physical_hold_steps: int = 0
    max_joint_confirm_steps: int = 0
    physical_break_count: int = 0
    compatibility_break_count: int = 0
    physical_hold_steps: int = 0
    joint_confirm_steps: int = 0
    end_step: int | None = None
    end_reason: str | None = None
    observation_end_kind: ObservationEndKind | None = None
    confirmation_delay_s: float | None = None
    observed_after_physical_ready_s: float | None = None
    blocker_steps: dict[str, int] = field(default_factory=dict)
    steps: CompactAttemptSteps = field(default_factory=CompactAttemptSteps)
    _previous_physical: bool = field(default=False, repr=False)
    _previous_joint: bool = field(default=False, repr=False)

    def observe(
        self,
        *,
        frame_id: int,
        physical: bool,
        action_compatible: bool,
        blockers: Iterable[str],
    ) -> AttemptStep:
        if self.observation_end_kind is not None:
            raise RuntimeError("Cannot update a finished recovery attempt")
        if self.steps and frame_id <= self.steps[-1].frame_id:
            raise ValueError("Attempt frame IDs must strictly increase")
        blocker_tuple = tuple(sorted(set(blockers)))
        if any(not isinstance(value, str) or not value for value in blocker_tuple):
            raise ValueError("Blocker labels must be nonempty strings")
        step = len(self.steps) + 1
        joint = physical and action_compatible
        if physical and self.first_physical_step is None:
            self.first_physical_step = step
        if self._previous_physical and not physical:
            self.physical_break_count += 1
        if self._previous_joint and not action_compatible:
            self.compatibility_break_count += 1
        if physical and not action_compatible:
            self.physical_without_compatibility_steps += 1
        self.physical_hold_steps = self.physical_hold_steps + 1 if physical else 0
        self.joint_confirm_steps = self.joint_confirm_steps + 1 if joint else 0
        self.max_physical_hold_steps = max(self.max_physical_hold_steps, self.physical_hold_steps)
        self.max_joint_confirm_steps = max(self.max_joint_confirm_steps, self.joint_confirm_steps)
        if self.physical_hold_steps >= self.ready_steps and self.physical_ready_step is None:
            self.physical_ready_step = step
        if self.joint_confirm_steps >= self.ready_steps and self.joint_ready_step is None:
            self.joint_ready_step = step
        for label in blocker_tuple:
            self.blocker_steps[label] = self.blocker_steps.get(label, 0) + 1
        reset_reasons = blocker_tuple if self._previous_joint and not joint else ()
        record = AttemptStep(
            step, frame_id, physical, action_compatible,
            self.physical_hold_steps, self.joint_confirm_steps,
            blocker_tuple, reset_reasons,
        )
        self.steps.append(record)
        self._previous_physical = physical
        self._previous_joint = joint
        return record

    def finish(
        self,
        reason: str,
        kind: ObservationEndKind,
        *,
        accepted_handoff: bool = False,
    ) -> None:
        if self.observation_end_kind is not None:
            raise RuntimeError("Recovery attempt has already finished")
        if not isinstance(kind, ObservationEndKind) or not reason:
            raise ValueError("A valid end kind and reason are required")
        if accepted_handoff and (
            kind is not ObservationEndKind.COMPLETED or self.joint_ready_step is None
        ):
            raise ValueError("Accepted handoff requires completed joint confirmation")
        self.end_step = len(self.steps)
        self.end_reason = reason
        self.observation_end_kind = kind
        if self.physical_ready_step is not None:
            self.observed_after_physical_ready_s = (
                self.end_step - self.physical_ready_step
            ) * self.dt
        if accepted_handoff:
            self.handoff_step = self.end_step
            self.confirmation_delay_s = self.observed_after_physical_ready_s

    def to_dict(self, *, compact_steps: bool = False) -> dict[str, Any]:
        result = {item.name: getattr(self, item.name) for item in fields(self)
                  if item.name not in ("steps", "_previous_physical", "_previous_joint")}
        result["context"] = asdict(self.context)
        result["blocker_steps"] = dict(self.blocker_steps)
        result["steps"] = self.steps.to_payload() if compact_steps else [asdict(record) for record in self.steps]
        if self.observation_end_kind is not None:
            result["observation_end_kind"] = self.observation_end_kind.value
        return result


@dataclass
class EventStatistics:
    event_id: int
    stage: str
    event_steps: int = 0
    recovery_steps: int = 0
    zero_stand_break_count: int = 0
    recurrent_fall_count: int = 0
    end_reason: str | None = None
    observation_end_kind: ObservationEndKind | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.observation_end_kind is not None:
            result["observation_end_kind"] = self.observation_end_kind.value
        return result


class RecoveryMetrics:
    def __init__(self, *, dt: float = 0.02, ready_steps: int = 50,
                 archive: MetricsArchive | None = None, stream_id: str = "0") -> None:
        if not isfinite(dt) or dt <= 0 or ready_steps <= 0:
            raise ValueError("Metrics require positive finite dt and ready_steps")
        self.dt = dt
        self.ready_steps = ready_steps
        self.attempts: list[AttemptStatistics] = []
        self.events: list[EventStatistics] = []
        self.archive = archive
        self.stream_id = str(stream_id)
        self.last_event: EventStatistics | None = None
        self._next_event_id = 1
        self._next_attempt_id = 1
        self._aggregates: dict[AttemptContext, dict] = {}
        self._checkpoint_censor_sequences: list[int] = []

    def begin_event(self, stage: str) -> EventStatistics:
        if self.events and self.events[-1].observation_end_kind is None:
            raise RuntimeError("An event is already active")
        event = EventStatistics(event_id=self._next_event_id, stage=stage)
        self._next_event_id += 1
        self.events.append(event)
        self.last_event = event
        return event

    def begin_attempt(self, event: EventStatistics, context: AttemptContext) -> AttemptStatistics:
        if event not in self.events or event.observation_end_kind is not None:
            raise RuntimeError("A live registered event is required")
        if self.attempts and self.attempts[-1].observation_end_kind is None:
            raise RuntimeError("An attempt is already active")
        attempt = AttemptStatistics(
            event.event_id, self._next_attempt_id, context, self.dt, self.ready_steps,
        )
        self._next_attempt_id += 1
        self.attempts.append(attempt)
        return attempt

    def archive_attempt(self, attempt: AttemptStatistics) -> None:
        if self.archive is None:
            return
        if attempt.observation_end_kind is None or attempt not in self.attempts:
            raise RuntimeError("Only a registered completed attempt can be archived")
        self.archive.append("attempt", self.stream_id, attempt.to_dict(compact_steps=True))
        _aggregate_attempt(self._aggregates, attempt)
        self.attempts.remove(attempt)

    def archive_event(self, event: EventStatistics) -> None:
        self.last_event = event
        if self.archive is None:
            return
        if event.observation_end_kind is None or event not in self.events:
            raise RuntimeError("Only a registered completed event can be archived")
        self.archive.append("event", self.stream_id, event.to_dict())
        self.events.remove(event)

    def checkpoint_state(self, *, attempt: AttemptStatistics | None, event: EventStatistics | None,
                         reason: str = "checkpoint_resume", seal_archive: bool = True) -> dict:
        if self.archive is None:
            clone = deepcopy(self)
            if attempt is not None:
                active = clone.attempts[-1]
                active.finish(reason, ObservationEndKind.CENSORED)
            if event is not None:
                active_event = clone.events[-1]
                active_event.end_reason = reason
                active_event.observation_end_kind = ObservationEndKind.CENSORED
            return clone.state_dict()
        aggregates = deepcopy(self._aggregates)
        snapshot_sequences = list(self._checkpoint_censor_sequences)
        if attempt is not None:
            censored_attempt = copy(attempt)
            censored_attempt.finish(reason, ObservationEndKind.CENSORED)
            sequence = self.archive.append("checkpoint_attempt", self.stream_id,
                                           censored_attempt.to_dict(compact_steps=True))
            snapshot_sequences.append(sequence)
            _aggregate_attempt(aggregates, censored_attempt)
        if event is not None:
            censored_event = copy(event)
            censored_event.end_reason = reason
            censored_event.observation_end_kind = ObservationEndKind.CENSORED
            self.archive.append("checkpoint_event", self.stream_id, censored_event.to_dict())
        return self._archive_state(aggregates=aggregates, checkpoint_sequences=snapshot_sequences,
                                   include_active=False, seal_archive=seal_archive)

    def _archive_state(self, *, aggregates=None, checkpoint_sequences=None,
                       include_active=True, seal_archive=True) -> dict:
        result = {
            "schema": "RecoveryMetricsArchiveV2", "dt": self.dt, "ready_steps": self.ready_steps,
            "archive_id": self.archive.archive_id, "stream_id": self.stream_id,
            "record_cutoff_sequence": self.archive.last_sequence,
            "next_event_id": self._next_event_id, "next_attempt_id": self._next_attempt_id,
            "aggregates": deepcopy(list((self._aggregates if aggregates is None else aggregates).values())),
            "checkpoint_censor_sequences": list(self._checkpoint_censor_sequences if checkpoint_sequences is None else checkpoint_sequences),
            "events": [event.to_dict() for event in self.events] if include_active else [],
            "attempts": [attempt.to_dict(compact_steps=True) for attempt in self.attempts] if include_active else [],
        }
        if seal_archive:
            result["archive_index"] = self.archive.snapshot()
        return result

    def state_dict(self, *, seal_archive: bool = True) -> dict[str, Any]:
        if self.archive is not None:
            return self._archive_state(seal_archive=seal_archive)
        return {
            "schema": "RecoveryMetricsV1",
            "dt": self.dt,
            "ready_steps": self.ready_steps,
            "events": [event.to_dict() for event in self.events],
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], *, archive: MetricsArchive | None = None) -> RecoveryMetrics:
        if state.get("schema") == "RecoveryMetricsArchiveV2":
            return cls._from_archive_state(state, archive=archive)
        if set(state) != {"schema", "dt", "ready_steps", "events", "attempts"}:
            raise ValueError("Recovery metrics state has missing or unknown fields")
        if state["schema"] != "RecoveryMetricsV1":
            raise ValueError("Unsupported recovery metrics schema")
        metrics = cls(dt=state["dt"], ready_steps=state["ready_steps"])
        event_fields = set(EventStatistics.__dataclass_fields__)
        for index, saved in enumerate(state["events"], start=1):
            if set(saved) != event_fields or saved["event_id"] != index:
                raise ValueError("Event state fields or event ordering are invalid")
            values = dict(saved)
            values["observation_end_kind"] = (
                ObservationEndKind(values["observation_end_kind"])
                if values["observation_end_kind"] is not None else None
            )
            for name in ("event_steps", "recovery_steps", "zero_stand_break_count", "recurrent_fall_count"):
                _require_nonnegative_integer(values[name], name)
            if values["recovery_steps"] > values["event_steps"]:
                raise ValueError("Recovery time cannot exceed event time")
            if (values["end_reason"] is None) != (values["observation_end_kind"] is None):
                raise ValueError("Event end reason and kind must be present together")
            if metrics.events and metrics.events[-1].observation_end_kind is None:
                raise ValueError("Only the last event can remain active")
            metrics.events.append(EventStatistics(**values))
        for index, saved in enumerate(state["attempts"], start=1):
            if saved["attempt_id"] != index:
                raise ValueError("Attempt IDs must be ordered and consecutive")
            event_id = saved["event_id"]
            if not isinstance(event_id, int) or not 1 <= event_id <= len(metrics.events):
                raise ValueError("Attempt references an unknown event")
            if saved["dt"] != metrics.dt or saved["ready_steps"] != metrics.ready_steps:
                raise ValueError("Attempt timing does not match metrics timing")
            context = AttemptContext(**saved["context"])
            event = metrics.events[event_id - 1]
            if context.stage != event.stage:
                raise ValueError("Attempt stage does not match its event")
            if metrics.attempts and metrics.attempts[-1].observation_end_kind is None:
                raise ValueError("Only the last attempt can remain active")
            attempt = AttemptStatistics(event_id, index, context, metrics.dt, metrics.ready_steps)
            for record in saved["steps"]:
                attempt.observe(
                    frame_id=record["frame_id"],
                    physical=record["physical"],
                    action_compatible=record["action_compatible"],
                    blockers=record["blockers"],
                )
            if saved["observation_end_kind"] is not None:
                attempt.finish(
                    saved["end_reason"],
                    ObservationEndKind(saved["observation_end_kind"]),
                    accepted_handoff=saved["handoff_step"] is not None,
                )
            elif event.observation_end_kind is not None:
                raise ValueError("A finished event cannot contain an active attempt")
            # JSON 反序列化将元组转成列表，比较时只统一这两种容器表示。
            if _json_container_form(attempt.to_dict()) != _json_container_form(saved):
                raise ValueError("Saved attempt statistics disagree with its step records")
            metrics.attempts.append(attempt)
        for event in metrics.events:
            attempts = [attempt for attempt in metrics.attempts if attempt.event_id == event.event_id]
            if event.recovery_steps != sum(len(attempt.steps) for attempt in attempts):
                raise ValueError("Event recovery count disagrees with attempt step records")
            if not attempts:
                raise ValueError("Every recovery event requires an attempt")
        metrics._next_event_id = len(metrics.events) + 1
        metrics._next_attempt_id = len(metrics.attempts) + 1
        metrics.last_event = metrics.events[-1] if metrics.events else None
        return metrics

    @classmethod
    def _from_archive_state(cls, state: dict, *, archive: MetricsArchive | None) -> RecoveryMetrics:
        required = {"schema", "dt", "ready_steps", "archive_id", "stream_id", "next_event_id",
                    "next_attempt_id", "aggregates", "checkpoint_censor_sequences", "events", "attempts",
                    "record_cutoff_sequence"}
        if set(state) not in (required, required | {"archive_index"}):
            raise ValueError("Archived metrics state fields are invalid")
        if archive is None or not archive.accepts_archive_id(state["archive_id"]):
            raise ValueError("Archived metrics restore requires the verified shared archive lineage")
        _require_nonnegative_integer(state["record_cutoff_sequence"], "record_cutoff_sequence")
        if state["record_cutoff_sequence"] > archive.last_sequence:
            raise ValueError("Metrics state references records beyond the restored archive")
        if "archive_index" in state:
            MetricsArchive.verify_index(state["archive_index"])
        metrics = cls(dt=state["dt"], ready_steps=state["ready_steps"], archive=archive,
                      stream_id=state["stream_id"])
        for name in ("next_event_id", "next_attempt_id"):
            _require_nonnegative_integer(state[name], name)
            if state[name] < 1:
                raise ValueError("Next metrics IDs must be positive")
        if state["attempts"] or state["events"]:
            raise ValueError("Strict training resume requires checkpoint-censored active records")
        metrics._next_event_id = state["next_event_id"]
        metrics._next_attempt_id = state["next_attempt_id"]
        metrics._checkpoint_censor_sequences = list(state["checkpoint_censor_sequences"])
        for row in state["aggregates"]:
            context = AttemptContext(**row["context"])
            if context in metrics._aggregates:
                raise ValueError("Archived metrics aggregate context is repeated")
            metrics._aggregates[context] = deepcopy(row)
        # 每个环境只写小型 lineage 记录；原始逐步记录留在已核验的不可变分段中。
        archive.append("resume_stream", metrics.stream_id, {"source_state": state})
        return metrics

    def summary(self) -> list[dict[str, Any]]:
        if self.archive is not None:
            aggregates = deepcopy(self._aggregates)
            for attempt in self.attempts:
                _aggregate_attempt(aggregates, attempt)
            return [_aggregate_summary(row, self.dt, self.archive.archive_id) for row in aggregates.values()]
        groups: dict[AttemptContext, list[AttemptStatistics]] = {}
        for attempt in self.attempts:
            groups.setdefault(attempt.context, []).append(attempt)
        summaries: list[dict[str, Any]] = []
        for context, attempts in groups.items():
            successes = [item for item in attempts if item.handoff_step is not None]
            delays = sorted(item.confirmation_delay_s for item in successes)
            censored = sum(
                item.observation_end_kind is ObservationEndKind.CENSORED for item in attempts
            )
            active = sum(item.observation_end_kind is None for item in attempts)
            task_failures = sum(
                item.observation_end_kind is ObservationEndKind.TASK_FAILURE for item in attempts
            )
            observed = len(attempts) - censored - active
            no_handoff_ready = [
                item for item in attempts
                if item.physical_ready_step is not None and item.handoff_step is None
            ]
            summaries.append({
                "context": asdict(context),
                "legal_initial_count": len(attempts),
                "ever_physical_count": sum(item.first_physical_step is not None for item in attempts),
                "physical_ready_count": sum(item.physical_ready_step is not None for item in attempts),
                "accepted_handoff_count": len(successes),
                "task_failure_count": task_failures,
                "censored_count": censored,
                "active_count": active,
                "conclusive_outcome_count": observed,
                "success_fraction_of_all_legal": len(successes) / len(attempts),
                "success_fraction_of_conclusive": len(successes) / observed if observed else None,
                "delay_success_sample_count": len(delays),
                "confirmation_delay_median_s": median(delays) if delays else None,
                "confirmation_delay_p90_s": _nearest_rank(delays, 0.90),
                "confirmation_delay_p95_s": _nearest_rank(delays, 0.95),
                "confirmation_delay_max_s": max(delays) if delays else None,
                "physical_ready_without_handoff_count": len(no_handoff_ready),
                "physical_ready_without_handoff": [
                    {
                        "attempt_id": item.attempt_id,
                        "end_reason": item.end_reason,
                        "observation_end_kind": (
                            item.observation_end_kind.value
                            if item.observation_end_kind is not None else "active"
                        ),
                        "observed_after_physical_ready_s": (
                            item.observed_after_physical_ready_s
                            if item.end_step is not None
                            else (len(item.steps) - item.physical_ready_step) * item.dt
                        ),
                    }
                    for item in no_handoff_ready
                ],
            })
        return summaries


def _nearest_rank(values: list[float], quantile: float) -> float | None:
    return values[ceil(quantile * len(values)) - 1] if values else None


def _aggregate_attempt(groups: dict[AttemptContext, dict], attempt: AttemptStatistics) -> None:
    if attempt.context not in groups:
        groups[attempt.context] = {
            "context": asdict(attempt.context), "legal_initial_count": 0, "ever_physical_count": 0,
            "physical_ready_count": 0, "accepted_handoff_count": 0, "task_failure_count": 0,
            "censored_count": 0, "active_count": 0, "physical_ready_without_handoff_count": 0,
            "physical_without_compatibility_steps": 0, "physical_break_count": 0,
            "compatibility_break_count": 0, "delay_steps_histogram": {}, "blocker_steps": {},
            "physical_ready_without_handoff_end_counts": {},
            "physical_ready_without_handoff_observation_steps_histogram": {},
        }
    row = groups[attempt.context]
    row["legal_initial_count"] += 1
    row["ever_physical_count"] += int(attempt.first_physical_step is not None)
    row["physical_ready_count"] += int(attempt.physical_ready_step is not None)
    row["accepted_handoff_count"] += int(attempt.handoff_step is not None)
    row["task_failure_count"] += int(attempt.observation_end_kind is ObservationEndKind.TASK_FAILURE)
    row["censored_count"] += int(attempt.observation_end_kind is ObservationEndKind.CENSORED)
    row["active_count"] += int(attempt.observation_end_kind is None)
    for field_name in ("physical_without_compatibility_steps", "physical_break_count", "compatibility_break_count"):
        row[field_name] += getattr(attempt, field_name)
    for label, count in attempt.blocker_steps.items():
        row["blocker_steps"][label] = row["blocker_steps"].get(label, 0) + count
    if attempt.handoff_step is not None:
        delay = str(attempt.handoff_step - attempt.physical_ready_step)
        row["delay_steps_histogram"][delay] = row["delay_steps_histogram"].get(delay, 0) + 1
    elif attempt.physical_ready_step is not None:
        row["physical_ready_without_handoff_count"] += 1
        kind = attempt.observation_end_kind.value if attempt.observation_end_kind is not None else "active"
        outcome = f"{kind}:{attempt.end_reason or 'active'}"
        counts = row["physical_ready_without_handoff_end_counts"]
        counts[outcome] = counts.get(outcome, 0) + 1
        elapsed = str((attempt.end_step if attempt.end_step is not None else len(attempt.steps)) - attempt.physical_ready_step)
        histogram = row["physical_ready_without_handoff_observation_steps_histogram"].setdefault(outcome, {})
        histogram[elapsed] = histogram.get(elapsed, 0) + 1


def _histogram_rank(histogram: dict, rank: int) -> int | None:
    cumulative = 0
    for steps, count in sorted((int(steps), count) for steps, count in histogram.items()):
        cumulative += count
        if cumulative >= rank:
            return steps
    return None


def _aggregate_summary(row: dict, dt: float, archive_id: str) -> dict:
    result = deepcopy(row)
    count = sum(row["delay_steps_histogram"].values())
    observed = row["legal_initial_count"] - row["censored_count"] - row["active_count"]
    median_steps = None
    if count:
        median_steps = (_histogram_rank(row["delay_steps_histogram"], (count + 1) // 2)
                        + _histogram_rank(row["delay_steps_histogram"], (count + 2) // 2)) / 2
    result.update({
        "conclusive_outcome_count": observed,
        "success_fraction_of_all_legal": row["accepted_handoff_count"] / row["legal_initial_count"],
        "success_fraction_of_conclusive": row["accepted_handoff_count"] / observed if observed else None,
        "delay_success_sample_count": count,
        "confirmation_delay_median_s": median_steps * dt if count else None,
        "confirmation_delay_p90_s": _histogram_rank(row["delay_steps_histogram"], ceil(0.9 * count)) * dt if count else None,
        "confirmation_delay_p95_s": _histogram_rank(row["delay_steps_histogram"], ceil(0.95 * count)) * dt if count else None,
        "confirmation_delay_max_s": max(int(steps) for steps in row["delay_steps_histogram"]) * dt if count else None,
        "physical_ready_without_handoff": None,
        "detailed_attempt_records": {"storage": "gzip_jsonl", "archive_id": archive_id},
    })
    return result


def restore_archived_attempt(payload: dict) -> AttemptStatistics:
    steps = CompactAttemptSteps.from_payload(payload["steps"])
    attempt = AttemptStatistics(payload["event_id"], payload["attempt_id"], AttemptContext(**payload["context"]),
                                payload["dt"], payload["ready_steps"])
    for record in steps:
        attempt.observe(frame_id=record.frame_id, physical=record.physical,
                        action_compatible=record.action_compatible, blockers=record.blockers)
    if payload["observation_end_kind"] is not None:
        attempt.finish(payload["end_reason"], ObservationEndKind(payload["observation_end_kind"]),
                       accepted_handoff=payload["handoff_step"] is not None)
    if _json_container_form(attempt.to_dict(compact_steps=True)) != _json_container_form(payload):
        raise ValueError("Archived attempt counters disagree with its lossless step columns")
    return attempt


def recompute_archived_aggregates(state: dict, archive: MetricsArchive, *, archive_index: dict | None = None) -> list[dict]:
    descriptor = archive_index if archive_index is not None else state.get("archive_index")
    if descriptor is None:
        raise ValueError("Independent metrics recomputation requires a sealed archive index")
    MetricsArchive.verify_index(descriptor)
    selected_snapshots = set(state["checkpoint_censor_sequences"])
    groups: dict[AttemptContext, dict] = {}
    for record in archive.iter_records(stream_id=state["stream_id"]):
        if record["sequence"] > min(descriptor["last_sequence"], state["record_cutoff_sequence"]):
            break
        included = record["kind"] == "attempt" or (
            record["kind"] == "checkpoint_attempt" and record["sequence"] in selected_snapshots
        )
        if included:
            _aggregate_attempt(groups, restore_archived_attempt(record["payload"]))
    rows = list(groups.values())
    if _json_container_form(rows) != _json_container_form(state["aggregates"]):
        raise ValueError("Metrics aggregate checkpoint disagrees with archived raw attempts")
    return rows


def verify_archived_metrics_states(states: list[dict], archive: MetricsArchive, descriptor: dict) -> None:
    MetricsArchive.verify_index(descriptor)
    streams = {state["stream_id"]: state for state in states}
    if len(streams) != len(states):
        raise ValueError("恢复统计stream重复")
    groups = {stream: {} for stream in streams}
    selected = {stream: set(state["checkpoint_censor_sequences"]) for stream, state in streams.items()}
    found = {stream: set() for stream in streams}
    for record in archive.iter_records():
        stream = record["stream_id"]
        if stream not in streams:
            continue
        state = streams[stream]
        if record["sequence"] > min(descriptor["last_sequence"], state["record_cutoff_sequence"]):
            continue
        snapshot = record["kind"] == "checkpoint_attempt" and record["sequence"] in selected[stream]
        if record["kind"] == "attempt" or snapshot:
            _aggregate_attempt(groups[stream], restore_archived_attempt(record["payload"]))
            if snapshot:
                found[stream].add(record["sequence"])
    for stream, state in streams.items():
        if found[stream] != selected[stream]:
            raise ValueError("恢复统计引用的删失副本缺失或超过截止序号")
        if _json_container_form(list(groups[stream].values())) != _json_container_form(state["aggregates"]):
            raise ValueError("恢复统计与选定checkpoint原始记录不一致")


def _require_nonnegative_integer(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _json_container_form(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_container_form(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_container_form(item) for item in value]
    return value
