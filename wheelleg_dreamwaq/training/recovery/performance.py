from __future__ import annotations

from contextlib import contextmanager
import ctypes
from functools import wraps
import os
from time import perf_counter

import psutil
import torch


def profile_phase(name):
    def decorate(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            with self._performance.measure(name):
                return function(self, *args, **kwargs)
        return wrapped
    return decorate


class _PerformanceInformation(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                ("HandleCount", ctypes.c_ulong), ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong)]


def system_commit():
    if os.name != "nt":
        return None
    info = _PerformanceInformation()
    info.cb = ctypes.sizeof(info)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
        raise OSError("读取Windows系统提交内存失败")
    return {"used_bytes": info.CommitTotal * info.PageSize, "limit_bytes": info.CommitLimit * info.PageSize,
            "available_bytes": (info.CommitLimit - info.CommitTotal) * info.PageSize,
            "peak_bytes": info.CommitPeak * info.PageSize}


class RecoveryPerformance:
    def __init__(self, device) -> None:
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.cpu_seconds = {}
        self.events = []
        self.physics_start = None

    def begin_update(self) -> None:
        self.cpu_seconds.clear()
        self.events.clear()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    @contextmanager
    def measure(self, name):
        start = perf_counter()
        pair = None
        if self.cuda:
            pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            pair[0].record(torch.cuda.current_stream(self.device))
        try:
            yield
        finally:
            if pair is not None:
                pair[1].record(torch.cuda.current_stream(self.device))
                self.events.append((name, *pair))
            self.cpu_seconds[name] = self.cpu_seconds.get(name, 0.0) + perf_counter() - start

    def begin_physics_interval(self):
        if self.physics_start is not None:
            raise RuntimeError("物理阶段重复开始")
        event = torch.cuda.Event(enable_timing=True) if self.cuda else None
        if event is not None:
            event.record(torch.cuda.current_stream(self.device))
        self.physics_start = (perf_counter(), event)

    def end_physics_interval(self):
        if self.physics_start is None:
            return
        start, event = self.physics_start
        self.physics_start = None
        name = "physics_and_scene_update"
        self.cpu_seconds[name] = self.cpu_seconds.get(name, 0.0) + perf_counter() - start
        if event is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream(self.device))
            self.events.append((name, event, end))

    def snapshot(self) -> dict:
        gpu_ms = {}
        if self.cuda:
            torch.cuda.synchronize(self.device)
            for name, start, end in self.events:
                gpu_ms[name] = gpu_ms.get(name, 0.0) + start.elapsed_time(end)
        processes = []
        parent = psutil.Process()
        for process in [parent, *parent.children(recursive=True)]:
            if process.is_running():
                memory = process.memory_info()
                processes.append({"pid": process.pid, "rss_bytes": memory.rss,
                                  "private_bytes": getattr(memory, "private", memory.vms)})
        memory = psutil.virtual_memory()
        return {"cpu_wall_seconds": dict(self.cpu_seconds), "cuda_event_ms": gpu_ms,
                "phase_times_overlap": True,
                "physics_interval_scope": "sim.step and scene.update between action applications",
                "process_memory": processes, "system_available_physical_bytes": memory.available,
                "system_commit": system_commit(),
                "cuda": None if not self.cuda else {
                    "allocated_bytes": torch.cuda.memory_allocated(self.device),
                    "reserved_bytes": torch.cuda.memory_reserved(self.device),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device)}}
