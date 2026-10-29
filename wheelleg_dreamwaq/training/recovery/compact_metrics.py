from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from time import time_ns


def _encoded(payload):
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


class CompactStandingMetrics:
    schema_version = "IndependentStandingCompactMetricsV1"

    def __init__(self, directory, resume_state=None, diagnostic_quota_bytes=512 * 1024 * 1024):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.diagnostic_quota_bytes = int(diagnostic_quota_bytes)
        if self.diagnostic_quota_bytes <= 0:
            raise ValueError("诊断轨迹配额必须为正整数")
        self.sequence = 0
        self.frame_hash = "0" * 64
        self.submitted = 0
        self.consumed = 0
        self.last_update = 0
        self.episode_count = 0
        self.closed = False
        self._streams = {}
        if resume_state is not None:
            self._validate_resume(resume_state)
            self.sequence = resume_state["last_sequence"]
            self.frame_hash = resume_state["last_frame_hash"]
            self.submitted = resume_state["submitted"]
            self.consumed = resume_state["consumed"]
            self.last_update = resume_state["last_update"]
            self.episode_count = resume_state["episode_count"]
        for name in ("updates", "episodes"):
            self._streams[name] = (self.directory / (name + ".jsonl")).open("ab" if resume_state is not None else "xb")

    def _validate_resume(self, state):
        if state.get("schema_version") != self.schema_version or state.get("pending") != 0 or state.get("submitted") != state.get("consumed"):
            raise ValueError("紧凑日志恢复快照不完整")
        if state.get("last_sequence") != state["consumed"] or state.get("last_update", -1) < 0:
            raise ValueError("紧凑日志恢复序列或更新无效")
        frame_hash = state.get("last_frame_hash", "")
        if len(frame_hash) != 64 or any(letter not in "0123456789ABCDEF" for letter in frame_hash):
            raise ValueError("紧凑日志恢复 frame hash 无效")
        if state.get("finalized") or state.get("worker_state") != "synchronous":
            raise ValueError("日志恢复要求未结束的同步日志快照")
        for name in ("updates", "episodes"):
            path = self.directory / (name + ".jsonl")
            cutoff = state["stream_cutoffs"][name]
            if path.stat().st_size != cutoff["bytes"]:
                raise ValueError("当前日志长度与 checkpoint cutoff 不一致")
            with path.open("rb") as stream:
                stream.seek(max(0, cutoff["bytes"] - 4096))
                tail_hash = hashlib.sha256(stream.read()).hexdigest().upper()
            if tail_hash != cutoff["tail_sha256"]:
                raise ValueError("日志尾部与 checkpoint 快照不一致")

    def _append(self, name, payload):
        if self.closed:
            raise RuntimeError("日志已经结束")
        body = dict(payload)
        body["sequence"] = self.sequence + 1
        body["previous_frame_hash"] = self.frame_hash
        frame_hash = hashlib.sha256(_encoded(body)).hexdigest().upper()
        body["frame_hash"] = frame_hash
        data = _encoded(body)
        self.submitted += 1
        stream = self._streams[name]
        written = stream.write(data)
        if written != len(data):
            raise OSError("同步日志写入不完整")
        stream.flush()
        os.fsync(stream.fileno())
        self.sequence += 1
        self.frame_hash = frame_hash
        self.consumed += 1

    def append_update(self, record):
        if record["update"] != self.last_update + 1:
            raise ValueError("更新日志必须连续")
        self._append("updates", record)
        self.last_update = int(record["update"])
        state = self.snapshot()
        with (self.directory / "metrics_latest.json").open("w", encoding="utf-8") as stream:
            json.dump({"schema_version": self.schema_version, "logging_barrier": state}, stream,
                      ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def append_episodes(self, episodes, update):
        for episode in episodes:
            if not {"reward", "length", "source", "success", "reason"}.issubset(episode):
                raise ValueError("回合日志字段不完整")
            self._append("episodes", {"update": update, **episode})
            self.episode_count += 1

    def snapshot(self, *, finalized=False, reason=None):
        cutoffs = {}
        for name in ("updates", "episodes"):
            stream = self._streams[name]
            if not stream.closed:
                stream.flush()
                os.fsync(stream.fileno())
            path = self.directory / (name + ".jsonl")
            count = path.stat().st_size
            with path.open("rb") as reader:
                reader.seek(max(0, count - 4096))
                tail_hash = hashlib.sha256(reader.read()).hexdigest().upper()
            cutoffs[name] = {"bytes": count, "tail_sha256": tail_hash}
        return {"schema_version": self.schema_version, "last_sequence": self.sequence,
                "last_frame_hash": self.frame_hash, "submitted": self.submitted, "consumed": self.consumed,
                "pending": self.submitted - self.consumed, "last_update": self.last_update,
                "episode_count": self.episode_count, "worker_state": "synchronous", "stream_cutoffs": cutoffs,
                "finalized": finalized, "reason": reason}

    def append_diagnostic(self, record):
        data = _encoded(record)
        if len(data) > self.diagnostic_quota_bytes:
            raise ValueError("单条诊断轨迹超过配额")
        directory = self.directory / "diagnostic_trajectories"
        directory.mkdir(exist_ok=True)
        files = sorted(directory.glob("trajectory_*.json"))
        total = sum(path.stat().st_size for path in files)
        while files and total + len(data) > self.diagnostic_quota_bytes:
            oldest = files.pop(0)
            total -= oldest.stat().st_size
            oldest.unlink()
        path = directory / f"trajectory_{time_ns():020d}_{self.sequence:012d}.json"
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return path

    def close(self, reason="training_run_end"):
        if self.closed:
            raise RuntimeError("日志不能重复结束")
        state = self.snapshot(finalized=True, reason=reason)
        for stream in self._streams.values():
            stream.close()
        self.closed = True
        with (self.directory / "metrics.json").open("w", encoding="utf-8") as stream:
            json.dump({"schema_version": self.schema_version, "logging_barrier": state}, stream,
                      ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return state
