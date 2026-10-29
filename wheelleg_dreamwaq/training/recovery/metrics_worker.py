from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from multiprocessing.connection import Client
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from time import perf_counter

import numpy as np

from .contracts import file_sha256, stable_hash
from .metrics import AttemptContext, ObservationEndKind, RecoveryMetrics, verify_archived_metrics_states
from .metrics_archive import MetricsArchive
from .metrics_protocol import LOG_COLUMNS, PACKET_KINDS, PROTOCOL_VERSION, normalize_contexts


END_KINDS = (None, ObservationEndKind.COMPLETED, ObservationEndKind.TASK_FAILURE, ObservationEndKind.CENSORED)
COLUMN = {name: index for index, name in enumerate(LOG_COLUMNS)}


def write_json_durable(path: Path, payload: dict) -> dict:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path), "sha256": file_sha256(path)}


class TensorMetricsConsumer:
    def __init__(self, configuration: dict):
        self.configuration = configuration
        self.directory = Path(configuration["directory"])
        self.num_envs = configuration["num_envs"]
        self.contexts = {int(key): AttemptContext(**value) for key, value in normalize_contexts(configuration["contexts"]).items()}
        self.protocol = configuration["protocol"]
        self.reasons = self.protocol["reason_labels"]
        self.blockers = self.protocol["blocker_labels"]
        self.frames = np.full(self.num_envs, -1, dtype=np.int64)
        self.generations = np.full(self.num_envs, -1, dtype=np.int64)
        self.packet_sequence = 0
        self.snapshot_sequence = 0
        self.processed_packets = 0
        self.processing_seconds = 0.0
        restored = configuration.get("restore_path")
        if restored:
            state = json.loads(Path(restored).read_text(encoding="utf-8"))
            self.archive = MetricsArchive.from_index(state["metrics_archive"], self.directory / "archive")
            verify_archived_metrics_states(state["metrics"], self.archive, state["metrics_archive"])
            if len(state["metrics"]) != self.num_envs:
                raise ValueError("恢复统计环境数量与日志协议不同")
            self.metrics = [RecoveryMetrics.from_state_dict(item, archive=self.archive) for item in state["metrics"]]
            self.frames[:] = state["frame_ids"]
            if "logging_barrier" in state:
                self.generations[:] = state["logging_barrier"]["generations"]
        else:
            self.archive = MetricsArchive(self.directory / "archive", segment_records=configuration["segment_records"])
            self.metrics = [RecoveryMetrics(dt=configuration["dt"], ready_steps=configuration["ready_steps"],
                archive=self.archive, stream_id=str(index)) for index in range(self.num_envs)]
        self.archive.append("tensor_log_protocol", "all", {"protocol": self.protocol, "contexts": configuration["contexts"],
                            "slot_budget": configuration["slot_budget"]})

    def next_ids(self) -> dict:
        return {"next_event_id": [metric._next_event_id for metric in self.metrics],
                "next_attempt_id": [metric._next_attempt_id for metric in self.metrics],
                "frame_ids": self.frames.tolist(), "generations": self.generations.tolist()}

    def register_contexts(self, incoming: dict) -> None:
        contexts = normalize_contexts(incoming)
        combined = {str(key): asdict(value) for key, value in self.contexts.items()}
        for key, value in contexts.items():
            if key in combined and combined[key] != value:
                raise ValueError("日志上下文ID不可重新绑定")
            combined[key] = value
        normalize_contexts(combined)
        self.contexts = {int(key): AttemptContext(**value) for key, value in combined.items()}
        self.archive.append("tensor_log_contexts", "all", {"contexts": contexts})

    def consume(self, data: np.ndarray, headers: list[dict]) -> None:
        started = perf_counter()
        if data.shape != (len(headers), self.num_envs, len(LOG_COLUMNS)) or data.dtype != np.int64:
            raise ValueError("原始日志共享槽shape或dtype不符合协议")
        boolean_columns = ("halt_requested", "capture_heading", "zero_command", "completed", "success", "failure",
            "event_active", "attempt_active", "attempt_recorded", "physical", "action_compatible", "attempt_started",
            "event_started", "attempt_ended", "event_finished")
        for name in boolean_columns:
            values = data[..., COLUMN[name]]
            if np.any((values < 0) | (values > 1)):
                raise ValueError(f"原始日志布尔列越界: {name}")
        ranges = {"mode": 2, "transition": 3, "fault_reason": len(self.reasons),
                  "attempt_end_reason": len(self.reasons), "event_end_reason": len(self.reasons),
                  "attempt_end_kind": len(END_KINDS), "event_end_kind": len(END_KINDS)}
        for name, upper in ranges.items():
            values = data[..., COLUMN[name]]
            if np.any((values < 0) | (values >= upper)):
                raise ValueError(f"原始日志枚举列越界: {name}")
        for packet, header in zip(data, headers, strict=True):
            if header["sequence"] != self.packet_sequence + 1:
                raise ValueError("原始日志packet序号不连续")
            kind = header["kind"]
            if kind not in PACKET_KINDS:
                raise ValueError("原始日志packet类型无效")
            for index, row in enumerate(packet):
                self._consume_row(index, row, kind)
            self.packet_sequence = header["sequence"]
        self.archive.append_tensor_packets(data, headers, self.protocol)
        self.processed_packets += len(headers)
        self.processing_seconds += perf_counter() - started

    def _consume_row(self, index: int, row: np.ndarray, kind: str) -> None:
        value = lambda name: int(row[COLUMN[name]])
        active = any(value(name) for name in ("attempt_recorded", "attempt_started", "event_started", "attempt_ended", "event_finished"))
        if kind != "step" and not active:
            return
        frame, generation = value("frame_id"), value("generation")
        if frame < -1 or (kind == "step" and frame < 0) or generation < 0:
            raise ValueError("日志物理帧必须非负，初始化begin可使用-1")
        if kind == "step" and self.frames[index] >= 0 and frame != self.frames[index] + 1:
            raise ValueError("日志控制帧必须连续且每帧恰好一次")
        if kind != "step" and self.frames[index] >= 0 and frame != self.frames[index]:
            raise ValueError("reset/interrupt必须引用同一已提交物理帧")
        if generation < self.generations[index] or (kind == "step" and self.generations[index] >= 0 and generation != self.generations[index]):
            raise ValueError("日志generation倒退或控制步未经reset改变")
        metric = self.metrics[index]
        if value("event_started"):
            context = self.contexts[value("context_id")]
            event = metric.begin_event(context.stage)
            if event.event_id != value("event_id"):
                raise ValueError("GPU与归档event ID不同")
        event = metric.events[-1] if metric.events else None
        if value("attempt_started"):
            if event is None:
                raise ValueError("开始attempt时缺少活动event")
            attempt = metric.begin_attempt(event, self.contexts[value("context_id")])
            if attempt.attempt_id != value("attempt_id"):
                raise ValueError("GPU与归档attempt ID不同")
        attempt = metric.attempts[-1] if metric.attempts else None
        if event is not None:
            if event.event_id != value("event_id"):
                raise ValueError("日志引用活动event ID错误")
            for name in ("event_steps", "recovery_steps", "zero_stand_break_count", "recurrent_fall_count"):
                count = value(name)
                if count < getattr(event, name):
                    raise ValueError("活动event计数倒退")
                setattr(event, name, count)
        if attempt is not None and attempt.attempt_id != value("attempt_id"):
            raise ValueError("日志引用活动attempt ID错误")
        if value("attempt_recorded"):
            if attempt is None:
                raise ValueError("逐步日志缺少活动attempt")
            mask = value("blocker_mask")
            if mask < 0 or mask >> len(self.blockers):
                raise ValueError("阻断位图超出固定字典")
            labels = tuple(label for bit, label in enumerate(self.blockers) if mask & (1 << bit))
            record = attempt.observe(frame_id=frame, physical=bool(value("physical")),
                                     action_compatible=bool(value("action_compatible")), blockers=labels)
            if (record.step != value("attempt_steps") or record.physical_hold_steps != value("physical_hold_steps")
                    or record.joint_confirm_steps != value("joint_confirm_steps")):
                raise ValueError("GPU逐步计数与原始统计重算不同")
        if value("attempt_ended"):
            if attempt is None:
                raise ValueError("结束attempt时缺少活动记录")
            reason = self.reasons[value("attempt_end_reason")]
            end_kind = END_KINDS[value("attempt_end_kind")]
            attempt.finish(reason, end_kind, accepted_handoff=reason in ("standalone_ready", "handoff"))
            metric.archive_attempt(attempt)
        if value("event_finished"):
            if event is None:
                raise ValueError("结束event时缺少活动记录")
            event.end_reason = self.reasons[value("event_end_reason")]
            event.observation_end_kind = END_KINDS[value("event_end_kind")]
            if event.end_reason == "none" or event.observation_end_kind is None:
                raise ValueError("结束event必须提供原因和种类")
            metric.archive_event(event)
        if bool(metric.events) != bool(value("event_active")) or bool(metric.attempts) != bool(value("attempt_active")):
            raise ValueError("GPU与归档活动event/attempt状态不同")
        self.frames[index], self.generations[index] = frame, generation

    def snapshot(self, frame_ids: list[int] | None, *, reason: str, finalize: bool) -> dict:
        if frame_ids is not None and frame_ids != self.frames.tolist():
            raise ValueError("checkpoint帧屏障与已消费原始日志不同")
        states = []
        for metric in self.metrics:
            attempt = metric.attempts[-1] if metric.attempts else None
            event = metric.events[-1] if metric.events else None
            if finalize:
                if attempt is not None:
                    attempt.finish(reason, ObservationEndKind.CENSORED)
                    metric.archive_attempt(attempt)
                if event is not None:
                    event.end_reason, event.observation_end_kind = reason, ObservationEndKind.CENSORED
                    metric.archive_event(event)
                states.append(metric.state_dict(seal_archive=False))
            else:
                states.append(metric.checkpoint_state(attempt=attempt, event=event, reason=reason, seal_archive=False))
        descriptor = self.archive.snapshot()
        self.snapshot_sequence += 1
        frame_list = self.frames.tolist()
        result = {"metrics": states, "metrics_archive": descriptor,
                  "logging_barrier": {"schema_version": "RecoveryLoggingBarrierV1", "protocol_sha256": self.protocol["sha256"],
                      "submitted_packet_sequence": self.packet_sequence, "consumed_packet_sequence": self.packet_sequence,
                      "frame_ids_sha256": stable_hash(frame_list), "frame_ids": frame_list,
                      "generations": self.generations.tolist(), "archive_last_sequence": descriptor["last_sequence"],
                      "stream_cutoffs": [state["record_cutoff_sequence"] for state in states],
                      "pending_packets": 0, "finalized": finalize, "reason": reason,
                      "processed_packets": self.processed_packets, "worker_processing_seconds": self.processing_seconds}}
        path = self.directory / f"snapshot_{self.snapshot_sequence:07d}.json"
        return write_json_durable(path, result)


def worker(configuration_path: str) -> None:
    configuration = json.loads(Path(configuration_path).read_text(encoding="utf-8"))
    connection = Client(tuple(configuration["address"]), authkey=bytes.fromhex(configuration["authkey"]))
    consumer = TensorMetricsConsumer(configuration)
    slots = [SharedMemory(name=name) for name in configuration["shared_names"]]
    arrays = [np.ndarray((configuration["packets_per_slot"], configuration["num_envs"], len(LOG_COLUMNS)),
                         dtype=np.int64, buffer=slot.buf) for slot in slots]
    connection.send({"kind": "ready", "next_ids": consumer.next_ids(), "pid": os.getpid()})
    while True:
        message = connection.recv()
        kind = message["kind"]
        if kind == "batch":
            slot = message["slot"]
            consumer.consume(arrays[slot][:len(message["headers"])], message["headers"])
            connection.send({"kind": "ack", "slot": slot, "sequence": consumer.packet_sequence})
        elif kind == "contexts":
            consumer.register_contexts(message["contexts"])
            connection.send({"kind": "contexts_ack", "request": message["request"]})
        elif kind in ("snapshot", "finalize"):
            descriptor = consumer.snapshot(message["frame_ids"], reason=message["reason"], finalize=kind == "finalize")
            connection.send({"kind": "snapshot_ack", "request": message["request"], "descriptor": descriptor})
        elif kind == "close":
            consumer.archive.close()
            connection.send({"kind": "close_ack"})
            break
        else:
            raise ValueError("日志worker收到未知控制消息")
    arrays.clear()
    for slot in slots:
        slot.close()
    connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", required=True)
    worker(parser.parse_args().configuration)
