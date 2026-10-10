from __future__ import annotations

import ctypes
from ctypes import wintypes
import logging
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Any, Callable


MUTATION_EVENTS = {
    "os.mkdir",
    "os.remove",
    "os.rmdir",
    "os.rename",
    "os.link",
    "os.symlink",
    "os.truncate",
    "os.chmod",
    "os.chown",
    "os.utime",
}
WRITE_FLAG_NAMES = (
    "O_WRONLY",
    "O_RDWR",
    "O_CREAT",
    "O_TRUNC",
    "O_APPEND",
    "O_EXCL",
    "O_TMPFILE",
    "O_TEMPORARY",
)


class PythonWriteGuard:
    def __init__(
        self, run_root: str | Path, *, wrap_file_handlers: bool = True
    ) -> None:
        self.run_root = Path(run_root).resolve()
        self.wrap_file_handlers = bool(wrap_file_handlers)
        self.ledger: list[dict[str, Any]] = []
        self.created_lifetime_handlers: list[dict[str, Any]] = []
        self.pre_handlers: list[dict[str, Any]] = []
        self.installed = False
        self.probe_count = 0
        self._probe_nonce: str | None = None
        self._original_file_handler_init: Callable[..., Any] | None = None
        self._registered_pipes: set[int] = set()
        self.capability_manifest = self._capabilities()

    @staticmethod
    def _capabilities() -> dict[str, Any]:
        flags = {
            name: {"available": hasattr(os, name), "value": getattr(os, name, None)}
            for name in WRITE_FLAG_NAMES
        }
        functions = {
            name: getattr(os, name, None)
            for name in (
                "mkdir", "remove", "rmdir", "rename", "link", "symlink",
                "truncate", "chmod", "chown", "utime",
            )
        }
        supports_dir_fd = getattr(os, "supports_dir_fd", set())
        return {
            "platform": sys.platform,
            "flags": flags,
            "apis": {
                name: "available" if function is not None else "unavailable_on_platform"
                for name, function in functions.items()
            },
            "dir_fd": {
                name: (
                    "supported"
                    if function is not None and function in supports_dir_fd
                    else "unavailable_on_platform"
                )
                for name, function in functions.items()
            },
        }

    @property
    def _write_mask(self) -> int:
        mask = 0
        for name in WRITE_FLAG_NAMES:
            if hasattr(os, name):
                mask |= int(getattr(os, name))
        return mask

    def register_pipe(self, descriptor: int) -> None:
        if not stat.S_ISFIFO(os.fstat(descriptor).st_mode):
            raise ValueError(f"fd {descriptor} is not an anonymous pipe")
        self._registered_pipes.add(int(descriptor))

    @staticmethod
    def _handler_snapshot() -> list[dict[str, Any]]:
        loggers: list[tuple[str, logging.Logger]] = [("root", logging.getLogger())]
        for name, value in logging.Logger.manager.loggerDict.items():
            if isinstance(value, logging.Logger):
                loggers.append((name, value))
        rows: list[dict[str, Any]] = []
        for logger_name, logger in loggers:
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    rows.append(
                        {
                            "logger": logger_name,
                            "baseFilename": str(Path(handler.baseFilename).resolve()),
                            "mode": handler.mode,
                            "encoding": handler.encoding,
                            "delay": handler.delay,
                        }
                    )
        return rows

    def post_handlers(self) -> list[dict[str, Any]]:
        return self._handler_snapshot()

    def _inside(self, path: Path) -> bool:
        try:
            path.relative_to(self.run_root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _strip_windows_device_prefix(value: str) -> str:
        if value.startswith("\\\\?\\UNC\\"):
            return "\\\\" + value[8:]
        if value.startswith("\\\\?\\"):
            return value[4:]
        return value

    def _resolve_fd(self, descriptor: int) -> Path | None:
        descriptor = int(descriptor)
        if descriptor in (0, 1, 2) or descriptor in self._registered_pipes:
            return None
        if os.name == "nt":
            import msvcrt

            handle = msvcrt.get_osfhandle(descriptor)
            buffer = ctypes.create_unicode_buffer(32768)
            result = ctypes.windll.kernel32.GetFinalPathNameByHandleW(
                wintypes.HANDLE(handle), buffer, len(buffer), 0
            )
            if result == 0 or result >= len(buffer):
                raise PermissionError(f"Cannot resolve writable fd {descriptor}")
            return Path(self._strip_windows_device_prefix(buffer.value)).resolve()
        proc_path = Path(f"/proc/self/fd/{descriptor}")
        if proc_path.exists():
            return Path(os.readlink(proc_path)).resolve()
        raise PermissionError(f"Cannot resolve writable fd {descriptor}")

    def _resolve_path(self, raw: Any, *, dir_fd: int | None = None) -> Path | None:
        if isinstance(raw, int):
            return self._resolve_fd(raw)
        value = os.fsdecode(raw)
        if value.lower() in {os.devnull.lower(), "nul", "\\.\\nul"}:
            return None
        candidate = Path(value)
        if not candidate.is_absolute():
            if dir_fd not in (None, -1):
                base = self._resolve_fd(int(dir_fd))
                if base is None:
                    raise PermissionError(f"Cannot use stream/pipe fd {dir_fd} as a directory")
                candidate = base / candidate
            else:
                candidate = Path.cwd() / candidate
        if candidate.exists():
            return candidate.resolve(strict=True)
        return candidate.parent.resolve(strict=True) / candidate.name

    def _require_allowed(self, raw: Any, *, dir_fd: int | None = None) -> Path | None:
        path = self._resolve_path(raw, dir_fd=dir_fd)
        if path is not None and not self._inside(path):
            raise PermissionError(f"Python filesystem mutation escapes run root: {path}")
        return path

    def _is_write_open(self, mode: Any, flags: Any) -> bool:
        mode_text = "" if mode is None else str(mode)
        return any(character in mode_text for character in "wax+") or (
            isinstance(flags, int) and bool(flags & self._write_mask)
        )

    def _record(self, event: str, args: tuple[Any, ...], paths: list[Path | None]) -> None:
        self.ledger.append(
            {
                "event": event,
                "args": [repr(argument) for argument in args],
                "paths": [None if path is None else str(path) for path in paths],
                "decision": "allow",
            }
        )

    def _audit_hook(self, event: str, args: tuple[Any, ...]) -> None:
        if event == "wheelleg.root_cause_suite.guard_probe":
            if self._probe_nonce is not None and args == (self._probe_nonce,):
                self.probe_count += 1
                self._record(event, args, [])
            return
        if event == "open":
            path, mode, flags = args
            if self._is_write_open(mode, flags):
                resolved = self._require_allowed(path)
                self._record(event, args, [resolved])
            return
        if event not in MUTATION_EVENTS:
            return
        paths: list[Path | None] = []
        if event in {"os.rename", "os.link"}:
            source, destination = args[0], args[1]
            source_fd = args[2] if len(args) > 2 else None
            destination_fd = args[3] if len(args) > 3 else None
            paths.extend(
                [
                    self._require_allowed(source, dir_fd=source_fd),
                    self._require_allowed(destination, dir_fd=destination_fd),
                ]
            )
        elif event == "os.symlink":
            source, destination = args[0], args[1]
            destination_fd = args[2] if len(args) > 2 else None
            paths.extend(
                [self._require_allowed(source), self._require_allowed(destination, dir_fd=destination_fd)]
            )
        else:
            path = args[0]
            dir_fd = None
            if event == "os.mkdir" and len(args) >= 3:
                dir_fd = args[2]
            elif event in {"os.remove", "os.rmdir"} and len(args) >= 2:
                dir_fd = args[1]
            elif event == "os.chmod" and len(args) >= 3:
                dir_fd = args[2]
            elif event in {"os.chown", "os.utime"} and len(args) >= 4:
                dir_fd = args[3]
            paths.append(self._require_allowed(path, dir_fd=dir_fd))
        self._record(event, args, paths)

    def _install_file_handler_wrapper(self) -> None:
        if self._original_file_handler_init is not None:
            return
        original = logging.FileHandler.__init__
        self._original_file_handler_init = original
        guard = self

        def guarded_init(handler: logging.FileHandler, filename: Any, *args: Any, **kwargs: Any) -> None:
            resolved = guard._require_allowed(filename)
            original(handler, filename, *args, **kwargs)
            guard.created_lifetime_handlers.append(
                {
                    "baseFilename": str(resolved) if resolved is not None else str(filename),
                    "mode": handler.mode,
                    "encoding": handler.encoding,
                    "delay": handler.delay,
                }
            )

        logging.FileHandler.__init__ = guarded_init  # type: ignore[assignment]

    def install(self) -> "PythonWriteGuard":
        if self.installed:
            return self
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.pre_handlers = self._handler_snapshot()
        for handler in self.pre_handlers:
            if not self._inside(Path(handler["baseFilename"])):
                raise PermissionError(f"Pre-existing FileHandler escapes run root: {handler}")
        self._probe_nonce = secrets.token_hex(16)
        sys.addaudithook(self._audit_hook)
        sys.audit("wheelleg.root_cause_suite.guard_probe", self._probe_nonce)
        if self.probe_count != 1:
            raise RuntimeError("Python audit hook installation self-test failed")
        if self.wrap_file_handlers:
            self._install_file_handler_wrapper()
        self.installed = True
        return self

    def restore_file_handler(self) -> None:
        if self._original_file_handler_init is not None:
            logging.FileHandler.__init__ = self._original_file_handler_init  # type: ignore[assignment]
            self._original_file_handler_init = None

    def __enter__(self) -> "PythonWriteGuard":
        return self.install()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore_file_handler()
