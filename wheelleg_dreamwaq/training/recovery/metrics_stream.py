from __future__ import annotations

from dataclasses import asdict
import json
from multiprocessing.connection import Listener
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread
from time import perf_counter, sleep

import numpy as np
import torch

from .batched_supervisor import BLOCKER_LABELS, REASON_LABELS
from .contracts import PROJECT_ROOT, file_sha256
from .metrics_protocol import LOG_COLUMNS, PACKET_KINDS, PACKETS_PER_SLOT, SLOT_COUNT, normalize_contexts, protocol_identity, slot_budget
from .supervisor import SupervisorConfig
from .worker_lifecycle import WindowsWorkerHandle


class RecoveryMetricsStream:
    def __init__(self, num_envs: int, device: str | torch.device, directory: str | Path, contexts: dict,
                 supervisor_config: SupervisorConfig = SupervisorConfig(), *, restore_state: dict | None = None,
                 timeout_s: float = 600.0, segment_records: int = 256) -> None:
        if timeout_s <= 0:
            raise ValueError("日志worker超时必须为正数")
        self.num_envs, self.device = num_envs, torch.empty(0, device=device).device
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.timeout_s = timeout_s
        self.budget = slot_budget(num_envs)
        self.protocol = protocol_identity(REASON_LABELS, BLOCKER_LABELS)
        self.contexts = normalize_contexts(contexts)
        self._closed = False
        self._sequence = self._request = self._ack_sequence = 0
        self._fill_slot: int | None = None
        self._connection = self._process = None
        self._listener = None
        self._worker_handle = None
        self._log = None
        self._responses = {}
        self.statistics = {"submitted_packets": 0, "acknowledged_packets": 0,
                           "backpressure_seconds": 0.0, "maximum_owned_slots": 0,
                           "pinned_bytes": self.budget["pinned_bytes"] if self.device.type == "cuda" else 0,
                           "shared_memory_bytes": self.budget["shared_memory_bytes"]}
        self._slots = []
        try:
            shape = (PACKETS_PER_SLOT, num_envs, len(LOG_COLUMNS))
            for _ in range(SLOT_COUNT):
                shared = SharedMemory(create=True, size=self.budget["bytes_per_slot"])
                slot = {"shared": shared, "array": None, "host": None,
                        "headers": [], "state": "free", "event": None}
                self._slots.append(slot)
                slot["array"] = np.ndarray(shape, dtype=np.int64, buffer=shared.buf)
                slot["host"] = torch.empty(shape, dtype=torch.int64, pin_memory=self.device.type == "cuda")
            restore_path = None
            if restore_state is not None:
                frames = restore_state["frame_ids"]
                frames = frames.detach().cpu().tolist() if isinstance(frames, torch.Tensor) else list(frames)
                restoration = {"metrics": restore_state["metrics"], "metrics_archive": restore_state["metrics_archive"], "frame_ids": frames}
                if "logging_barrier" in restore_state:
                    restoration["logging_barrier"] = restore_state["logging_barrier"]
                restore_path = str(self.directory / "restore_metrics.json")
                Path(restore_path).write_text(json.dumps(restoration, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            self._start_worker(restore_path, supervisor_config, segment_records)
        except BaseException:
            self.abort()
            raise

    def _start_worker(self, restore_path, supervisor_config, segment_records):
        authkey = os.urandom(32)
        listener = Listener(("127.0.0.1", 0), authkey=authkey)
        self._listener = listener
        configuration = {"directory": str(self.directory), "num_envs": self.num_envs, "contexts": self.contexts,
            "protocol": self.protocol, "slot_budget": self.budget, "segment_records": segment_records,
            "dt": supervisor_config.dt, "ready_steps": round(supervisor_config.ready_hold_s / supervisor_config.dt),
            "shared_names": [slot["shared"].name for slot in self._slots], "packets_per_slot": PACKETS_PER_SLOT,
            "restore_path": restore_path, "address": list(listener.address), "authkey": authkey.hex()}
        path = self.directory / "worker_configuration.json"
        path.write_text(json.dumps(configuration, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        self._log = (self.directory / "worker_stderr.log").open("w", encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._process = subprocess.Popen([sys.executable, "-u", "-m", "recovery.metrics_worker", "--configuration", str(path)],
            cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=self._log, creationflags=flags)
        accepted = []
        def accept():
            accepted.append(listener.accept())
        thread = Thread(target=accept, daemon=True)
        thread.start()
        started = perf_counter()
        while not accepted:
            self._check_process()
            if perf_counter() - started > self.timeout_s:
                raise TimeoutError("日志worker连接超时")
            sleep(.005)
        self._connection = accepted[0]
        listener.close()
        self._listener = None
        while not self._connection.poll(.01):
            self._check_process()
            if perf_counter() - started > self.timeout_s:
                raise TimeoutError("日志worker初始化超时")
        ready = self._connection.recv()
        if ready["kind"] != "ready":
            raise RuntimeError("日志worker初始化ACK无效")
        self._next_ids = ready["next_ids"]
        self.worker_pid = ready["pid"]
        if os.name == "nt":
            self._worker_handle = WindowsWorkerHandle(self.worker_pid)

    def _check_process(self) -> None:
        if self._process is not None and self._process.poll() is not None:
            self._log.flush()
            details = (self.directory / "worker_stderr.log").read_text(encoding="utf-8")[-12000:]
            raise RuntimeError(f"完整原始日志worker退出(exit={self._process.returncode}): {details}")

    def next_ids(self) -> dict:
        return {name: list(values) for name, values in self._next_ids.items()}

    def poll(self) -> None:
        if self._closed:
            raise RuntimeError("日志stream已经关闭")
        self._check_process()
        copying = sorted(((index, slot) for index, slot in enumerate(self._slots) if slot["state"] == "copying"),
                         key=lambda item: item[1]["headers"][0]["sequence"])
        for index, slot in copying:
            if slot["event"] is not None and not slot["event"].query():
                break
            if slot["state"] == "copying" and (slot["event"] is None or slot["event"].query()):
                count = len(slot["headers"])
                np.copyto(slot["array"][:count], slot["host"][:count].numpy())
                self._connection.send({"kind": "batch", "slot": index, "headers": list(slot["headers"])})
                slot["state"] = "processing"
        while self._connection.poll():
            message = self._connection.recv()
            if message["kind"] == "ack":
                slot = self._slots[message["slot"]]
                if slot["state"] != "processing" or message["sequence"] != slot["headers"][-1]["sequence"]:
                    raise RuntimeError("日志worker ACK与槽位所有权不同")
                self._ack_sequence = message["sequence"]
                self.statistics["acknowledged_packets"] = self._ack_sequence
                slot["state"], slot["headers"], slot["event"] = "free", [], None
            else:
                self._responses[message.get("request", message["kind"])] = message

    def _acquire(self) -> int:
        started = perf_counter()
        while True:
            self.poll()
            for index, slot in enumerate(self._slots):
                if slot["state"] == "free":
                    slot["state"] = "filling"
                    self.statistics["backpressure_seconds"] += perf_counter() - started
                    owned = sum(item["state"] != "free" for item in self._slots)
                    self.statistics["maximum_owned_slots"] = max(self.statistics["maximum_owned_slots"], owned)
                    return index
            if perf_counter() - started > self.timeout_s:
                raise TimeoutError("原始日志背压等待超时")
            sleep(.001)

    def submit(self, update, *, kind: str = "step") -> None:
        if kind not in PACKET_KINDS:
            raise ValueError("未知原始日志packet类型")
        self.poll()
        if self._fill_slot is None:
            self._fill_slot = self._acquire()
        slot = self._slots[self._fill_slot]
        columns = [getattr(update, name) for name in LOG_COLUMNS]
        if any(value.shape != (self.num_envs,) or value.device != self.device for value in columns):
            raise ValueError("原始日志列shape/device必须符合固定协议")
        packed = torch.stack([value.to(dtype=torch.int64) for value in columns], dim=-1)
        slot["host"][len(slot["headers"])].copy_(packed, non_blocking=self.device.type == "cuda")
        self._sequence += 1
        slot["headers"].append({"sequence": self._sequence, "kind": kind})
        self.statistics["submitted_packets"] = self._sequence
        if len(slot["headers"]) == PACKETS_PER_SLOT:
            self._seal()

    def _seal(self) -> None:
        if self._fill_slot is None:
            return
        slot = self._slots[self._fill_slot]
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            slot["event"] = event
        slot["state"] = "copying"
        self._fill_slot = None
        self.poll()

    def submit_rollout(self) -> None:
        self._seal()
        self.poll()

    def flush(self, frame_ids=None) -> None:
        self._seal()
        started = perf_counter()
        while any(slot["state"] != "free" for slot in self._slots):
            self.poll()
            if perf_counter() - started > self.timeout_s:
                raise TimeoutError("原始日志flush ACK等待超时")
            sleep(.001)
        if self._ack_sequence != self._sequence:
            raise RuntimeError("原始日志flush丢失packet ACK")

    def _request_response(self, kind: str, **payload) -> dict:
        self._request += 1
        request = self._request
        self._connection.send({"kind": kind, "request": request, **payload})
        started = perf_counter()
        while request not in self._responses:
            self.poll()
            if perf_counter() - started > self.timeout_s:
                raise TimeoutError("原始日志控制消息ACK等待超时")
            sleep(.001)
        return self._responses.pop(request)

    def register_contexts(self, contexts: dict) -> None:
        self.flush()
        incoming = normalize_contexts(contexts)
        combined = dict(self.contexts)
        for key, value in incoming.items():
            if key in combined and combined[key] != value:
                raise ValueError("日志上下文ID不可重新绑定")
            combined[key] = value
        normalize_contexts(combined)
        self._request_response("contexts", contexts=incoming)
        self.contexts = combined

    def _snapshot(self, frame_ids, reason: str, *, finalize: bool) -> dict:
        self.flush()
        frames = frame_ids.detach().cpu().tolist() if isinstance(frame_ids, torch.Tensor) else frame_ids
        result = self._request_response("finalize" if finalize else "snapshot", frame_ids=frames, reason=reason)
        descriptor = result["descriptor"]
        if file_sha256(descriptor["path"]) != descriptor["sha256"]:
            raise RuntimeError("原始日志snapshot文件SHA不一致")
        payload = json.loads(Path(descriptor["path"]).read_text(encoding="utf-8"))
        barrier = payload["logging_barrier"]
        if barrier["consumed_packet_sequence"] != self._sequence or barrier["pending_packets"]:
            raise RuntimeError("checkpoint未到达原始日志消费帧屏障")
        return payload

    def checkpoint(self, frame_ids, *, reason: str = "checkpoint_resume") -> dict:
        return self._snapshot(frame_ids, reason, finalize=False)

    def close(self, frame_ids=None, *, reason: str = "training_run_end") -> dict | None:
        if self._closed:
            return None
        try:
            payload = self._snapshot(frame_ids, reason, finalize=True)
            self._connection.send({"kind": "close"})
            started = perf_counter()
            while "close_ack" not in self._responses:
                if self._connection.poll(.01):
                    message = self._connection.recv()
                    self._responses[message["kind"]] = message
                elif self._process.poll() is not None:
                    self._check_process()
                if perf_counter() - started > self.timeout_s:
                    raise TimeoutError("日志worker关闭ACK超时")
            self._process.wait(timeout=10)
            if self._process.returncode != 0:
                self._check_process()
            return payload
        finally:
            self.abort()

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._worker_handle is not None:
            self._worker_handle.terminate_and_wait()
            self._worker_handle.close()
            self._worker_handle = None
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=10)
        if self._connection is not None:
            self._connection.close()
        if self._listener is not None:
            self._listener.close()
        for slot in self._slots:
            slot["array"] = None
            slot["host"] = None
            slot["shared"].close()
            slot["shared"].unlink()
        if self._log is not None:
            self._log.close()
