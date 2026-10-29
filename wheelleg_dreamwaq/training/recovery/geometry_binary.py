from __future__ import annotations

from collections import deque
import json
from multiprocessing import shared_memory
from pathlib import Path
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np

from .contracts import PROJECT_ROOT
from .worker_lifecycle import WindowsWorkerHandle


PROTOCOL = "RecoveryGeometryBinaryV1"
CAPACITY = 256
SLOTS = 2
MAX_BODIES = 27
INPUT_BYTES = CAPACITY * MAX_BODIES * 7 * 8
RESULT_DTYPE = np.dtype([("penetration", "<f8"), ("clearance", "<f8"),
                         ("first_geom", "<i4"), ("second_geom", "<i4"),
                         ("checked_pairs", "<i4"), ("violation", "i1"), ("failure", "i1")], align=True)
SLOT_BYTES = INPUT_BYTES + CAPACITY * RESULT_DTYPE.itemsize
FAILURE_CODES = {None: 0, "nonallowed_body_clearance": 1, "excessive_ground_penetration": 2}


def _read_acks(stream, target):
    try:
        for line in stream:
            target.put(json.loads(line))
    except BaseException as error:
        target.put({"status": "reader_error", "error": repr(error)})
    finally:
        target.put({"status": "closed"})


class GeometryBinaryClient:
    def __init__(self, config, directory):
        self.config = config
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.segments = []
        self.pending = {}
        self.free = deque(range(SLOTS))
        self.sequence = 0
        self.responses = queue.Queue(maxsize=SLOTS + 2)
        self.acks = {}
        self.closed = False
        self.worker_handle = None
        self.initialized = False
        self.log = None
        self.process = None
        self.reader = None
        try:
            for _ in range(SLOTS):
                self.segments.append(shared_memory.SharedMemory(create=True, size=SLOT_BYTES))
            self._initialize_worker()
        except BaseException:
            self.close()
            raise

    def _initialize_worker(self):
        self.log = (self.directory / "geometry_worker.log").open("a", encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.process = subprocess.Popen([self.config["python_executable"], "-u", "-m", "recovery.geometry_guard", "--worker"],
            cwd=PROJECT_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding="utf-8", creationflags=flags)
        self.reader = threading.Thread(target=_read_acks, args=(self.process.stdout, self.responses), daemon=True)
        self.reader.start()
        self._send({"kind": "initialize_binary", "sequence": 0, "protocol": PROTOCOL,
                    "segments": [segment.name for segment in self.segments], "slot_bytes": SLOT_BYTES,
                    "config": self.config, "directory": str(self.directory.resolve())})
        response = self._wait_ack(0)
        if os.name == "nt":
            self.worker_handle = WindowsWorkerHandle(response["worker_pid"])
        self.initialized = True
        self.constants = response["constants"]
        self.statistics = {"submitted_requests": 0, "completed_requests": 0,
                           "maximum_pending": 0, "input_payload_bytes": 0, "worker_elapsed_s": 0.0,
                           "shared_slot_bytes": SLOT_BYTES, "shared_slots": SLOTS}

    def _send(self, message):
        if self.process.poll() is not None or (self.worker_handle is not None and self.worker_handle.wait(0)):
            raise RuntimeError("几何worker已经退出，参见geometry_worker.log")
        self.process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
        self.process.stdin.flush()

    def check_health(self):
        if self.closed:
            raise RuntimeError("几何worker已经关闭")
        while True:
            try:
                response = self.responses.get_nowait()
            except queue.Empty:
                break
            self._acknowledge(response)
        if self.process.poll() is not None or (self.worker_handle is not None and self.worker_handle.wait(0)):
            raise RuntimeError("几何worker已经退出，参见geometry_worker.log")

    def _acknowledge(self, response):
        if response.get("status") != "ok":
            raise RuntimeError(f"几何worker通信失败: {response}")
        sequence = response["sequence"]
        initialization = sequence == 0 and not self.initialized
        if sequence in self.acks or (not initialization and sequence not in self.pending):
            raise RuntimeError("几何worker收到重复或未知请求ACK")
        self.acks[sequence] = response

    def _wait_ack(self, sequence):
        deadline = time.monotonic() + 120.0
        while True:
            self.check_health()
            if sequence in self.acks:
                return self.acks.pop(sequence)
            if time.monotonic() >= deadline:
                raise TimeoutError("几何worker ACK超过120秒")
            try:
                response = self.responses.get(timeout=0.05)
            except queue.Empty:
                continue
            self._acknowledge(response)

    def submit(self, kind, body_names, positions, quaternions, *, metadata=None):
        self.check_health()
        if not self.free:
            raise BufferError("几何双槽位已满，必须等待ACK并消费结果")
        positions, quaternions = np.asarray(positions), np.asarray(quaternions)
        count, bodies = positions.shape[:2]
        if (not 0 < count <= CAPACITY or not 0 < bodies <= MAX_BODIES
                or positions.shape != (count, len(body_names), 3)
                or quaternions.shape != (count, bodies, 4)
                or positions.dtype != quaternions.dtype
                or positions.dtype not in (np.dtype("float32"), np.dtype("float64"))):
            raise ValueError("几何二进制请求容量、形状或dtype无效")
        slot = self.free.popleft()
        inputs = np.ndarray((count, bodies, 7), dtype=positions.dtype, buffer=self.segments[slot].buf)
        inputs[..., :3], inputs[..., 3:] = positions, quaternions
        self.sequence += 1
        request = {"kind": kind, "sequence": self.sequence, "slot": slot, "count": count,
                   "body_names": list(body_names), "dtype": positions.dtype.str,
                   "metadata": {} if metadata is None else metadata}
        self.pending[self.sequence] = request
        self._send(request)
        stats = self.statistics
        stats["submitted_requests"] += 1
        stats["input_payload_bytes"] += inputs.nbytes
        stats["maximum_pending"] = max(stats["maximum_pending"], len(self.pending))
        return self.sequence

    def collect(self, sequence, *, wait=True):
        if sequence not in self.pending:
            raise ValueError("几何结果请求不在待处理集合")
        self.check_health()
        if not wait and sequence not in self.acks:
            return None
        response = self._wait_ack(sequence)
        request = self.pending.pop(sequence)
        slot = request["slot"]
        if response["slot"] != slot or response["count"] != request["count"]:
            raise RuntimeError("几何ACK容量或槽位错误")
        result = np.ndarray((request["count"],), dtype=RESULT_DTYPE,
                            buffer=self.segments[slot].buf, offset=INPUT_BYTES).copy()
        self.free.append(slot)
        self.statistics["completed_requests"] += 1
        self.statistics["worker_elapsed_s"] += response["worker_elapsed_s"]
        return {"sequence": sequence, "kind": request["kind"], "metadata": request["metadata"], "values": result}

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process is not None and self.process.poll() is None:
            self.process.stdin.close()
            if self.worker_handle is not None and not self.worker_handle.wait(5000):
                self.worker_handle.terminate_and_wait()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
        elif self.worker_handle is not None and not self.worker_handle.wait(0):
            self.worker_handle.terminate_and_wait()
        if self.worker_handle is not None:
            self.worker_handle.close()
        if self.reader is not None:
            self.reader.join(timeout=1)
        if self.process is not None:
            self.process.stdin.close()
            self.process.stdout.close()
        if self.log is not None:
            self.log.close()
        for segment in self.segments:
            segment.close()
            segment.unlink()
