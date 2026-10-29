from __future__ import annotations

import ctypes
import os


class WindowsWorkerHandle:
    def __init__(self, pid: int):
        if os.name != "nt" or pid <= 0:
            raise ValueError("WindowsWorkerHandle需要Windows上的实际worker PID")
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        kernel.TerminateProcess.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        self.handle = kernel.OpenProcess(0x00100000 | 0x0001, False, pid)
        if not self.handle:
            raise ctypes.WinError()
        self.pid = pid

    def wait(self, timeout_ms: int = 0) -> bool:
        if self.handle is None:
            raise RuntimeError("worker HANDLE已经关闭")
        result = ctypes.windll.kernel32.WaitForSingleObject(self.handle, timeout_ms)
        if result == 0:
            return True
        if result == 0x00000102:
            return False
        raise ctypes.WinError()

    def terminate_and_wait(self, timeout_ms: int = 10000) -> None:
        if not self.wait():
            if not ctypes.windll.kernel32.TerminateProcess(self.handle, 1):
                error = ctypes.WinError()
                if not self.wait(100):
                    raise error
        if not self.wait(timeout_ms):
            raise TimeoutError("实际worker退出等待超时")

    def close(self) -> None:
        if self.handle is not None:
            if not ctypes.windll.kernel32.CloseHandle(self.handle):
                raise ctypes.WinError()
            self.handle = None
