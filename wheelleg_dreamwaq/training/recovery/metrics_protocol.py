from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .contracts import stable_hash
from .metrics import AttemptContext


LOG_COLUMNS = (
    "frame_id", "event_id", "attempt_id", "context_id", "generation",
    "mode", "transition", "fault_reason", "halt_requested", "capture_heading", "zero_command",
    "completed", "success", "failure", "event_active", "attempt_active",
    "event_steps", "recovery_steps", "attempt_steps", "physical_hold_steps", "joint_confirm_steps",
    "zero_stand_hold_steps", "fall_hold_steps", "zero_stand_break_count", "recurrent_fall_count",
    "attempt_recorded", "physical", "action_compatible", "blocker_mask", "attempt_started", "event_started",
    "attempt_ended", "event_finished", "attempt_end_reason", "attempt_end_kind", "event_end_reason", "event_end_kind",
)
PACKET_KINDS = ("step", "reset", "interrupt")
PROTOCOL_VERSION = "RecoveryMetricsTensorLogV1"
SLOT_COUNT = 2
PACKETS_PER_SLOT = 2
MAX_PINNED_BYTES = 16 * 1024 * 1024
MAX_CONTEXTS = 4096
MAX_CONTEXT_TEXT_BYTES = 128


def slot_budget(num_envs: int) -> dict:
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError("日志环境数量必须为正整数")
    per_packet = num_envs * len(LOG_COLUMNS) * np.dtype(np.int64).itemsize
    per_slot = PACKETS_PER_SLOT * per_packet
    total = SLOT_COUNT * per_slot
    rounded_allocator_bytes = SLOT_COUNT * (1 << (per_slot - 1).bit_length())
    if rounded_allocator_bytes > MAX_PINNED_BYTES:
        raise ValueError("完整日志双槽超过16MiB pinned预算")
    return {"protocol": PROTOCOL_VERSION, "num_envs": num_envs, "columns": list(LOG_COLUMNS),
            "dtype": "int64", "slots": SLOT_COUNT, "packets_per_slot": PACKETS_PER_SLOT,
            "bytes_per_packet": per_packet, "bytes_per_slot": per_slot, "pinned_bytes": total,
            "pinned_allocator_rounded_budget_bytes": rounded_allocator_bytes,
            "shared_memory_bytes": total, "maximum_contexts": MAX_CONTEXTS,
            "context_text_maximum_utf8_bytes": MAX_CONTEXT_TEXT_BYTES,
            "worst_case_lifecycle": "每个环境每步均可提交step、interrupt、reset独立完整packet；槽满背压"}


def normalize_contexts(contexts: dict) -> dict[str, dict]:
    if not isinstance(contexts, dict) or not contexts or len(contexts) > MAX_CONTEXTS:
        raise ValueError("日志上下文字典为空或超过固定容量")
    normalized = {}
    for key, value in contexts.items():
        index = int(key)
        if index < 0 or str(index) in normalized:
            raise ValueError("日志上下文ID必须唯一且非负")
        record = asdict(value) if isinstance(value, AttemptContext) else dict(value)
        context = AttemptContext(**record)
        if any(len(text.encode("utf-8")) > MAX_CONTEXT_TEXT_BYTES for text in asdict(context).values()):
            raise ValueError("日志上下文字符串超过固定UTF8容量")
        normalized[str(index)] = asdict(context)
    return normalized


def protocol_identity(reason_labels, blocker_labels) -> dict:
    labels = list(blocker_labels)
    if len(labels) > 63 or len(set(labels)) != len(labels):
        raise ValueError("日志阻断位图超出int64表示范围或包含重复标签")
    result = {"version": PROTOCOL_VERSION, "columns": list(LOG_COLUMNS), "dtype": "int64",
              "reason_labels": list(reason_labels), "blocker_labels": labels,
              "slot_count": SLOT_COUNT, "packets_per_slot": PACKETS_PER_SLOT}
    result["sha256"] = stable_hash(result)
    return result
