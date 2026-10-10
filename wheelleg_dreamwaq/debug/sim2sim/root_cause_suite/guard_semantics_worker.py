from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Callable

from .bootstrap_runtime import current_bootstrap_guard
from .contracts import canonical_json_bytes, require_path_within, sha256_file
from .python_write_guard import PythonWriteGuard


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RootCauseSuite guard semantics worker")
    parser.add_argument("command", choices=("run", "child"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("disabled", "enabled"))
    parser.add_argument("--outside-root", type=Path)
    parser.add_argument("--preopened-fd", type=int)
    return parser


def _file_semantics(root: Path) -> dict[str, Any]:
    raw_path = root / "raw.bin"
    with raw_path.open("wb") as stream:
        stream.write(b"alpha\x00omega")

    eager_path = root / "eager.log"
    eager_logger = logging.Logger("root-cause-eager", level=logging.INFO)
    eager_handler = logging.FileHandler(
        eager_path, mode="w", encoding="utf-8", delay=False
    )
    eager_handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    eager_logger.addHandler(eager_handler)
    eager_logger.warning("same-record")
    eager_handler.close()
    eager_logger.removeHandler(eager_handler)

    delayed_path = root / "delayed.log"
    delayed_logger = logging.Logger("root-cause-delayed", level=logging.INFO)
    delayed_handler = logging.FileHandler(
        delayed_path, mode="w", encoding="utf-8", delay=True
    )
    delayed_before_emit = delayed_path.exists()
    delayed_handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    delayed_logger.addHandler(delayed_handler)
    delayed_logger.error("same-delayed-record")
    delayed_handler.close()
    delayed_logger.removeHandler(delayed_handler)

    exclusive_path = root / "exclusive.log"
    exclusive_path.write_bytes(b"keep")
    exception: dict[str, Any]
    try:
        logging.FileHandler(exclusive_path, mode="x", encoding="utf-8", delay=False)
    except Exception as error:
        exception = {
            "type": type(error).__name__,
            "errno": getattr(error, "errno", None),
            "exclusive_bytes_unchanged": exclusive_path.read_bytes() == b"keep",
        }
    else:
        exception = {
            "type": None,
            "errno": None,
            "exclusive_bytes_unchanged": False,
        }
    return {
        "raw_hex": raw_path.read_bytes().hex(),
        "eager_hex": eager_path.read_bytes().hex(),
        "delayed_hex": delayed_path.read_bytes().hex(),
        "delayed_exists_before_emit": delayed_before_emit,
        "exception": exception,
    }


def _attempt(
    operation: Callable[[], None],
    unchanged: Callable[[], bool],
) -> dict[str, Any]:
    try:
        operation()
    except Exception as error:
        return {
            "passed": isinstance(error, PermissionError) and unchanged(),
            "error_type": type(error).__name__,
            "error": str(error),
            "unchanged": unchanged(),
        }
    return {
        "passed": False,
        "error_type": None,
        "error": None,
        "unchanged": unchanged(),
    }


def _inside_mutations(root: Path, guard: PythonWriteGuard) -> dict[str, Any]:
    results: dict[str, Any] = {}

    mkdir_path = root / "mkdir"
    os.mkdir(mkdir_path)
    results["mkdir"] = {"passed": mkdir_path.is_dir(), "status": "tested"}

    remove_path = root / "remove.txt"
    remove_path.write_bytes(b"remove")
    os.remove(remove_path)
    results["remove"] = {"passed": not remove_path.exists(), "status": "tested"}

    rmdir_path = root / "rmdir"
    os.mkdir(rmdir_path)
    os.rmdir(rmdir_path)
    results["rmdir"] = {"passed": not rmdir_path.exists(), "status": "tested"}

    rename_source = root / "rename-source.txt"
    rename_destination = root / "rename-destination.txt"
    rename_source.write_bytes(b"rename")
    os.rename(rename_source, rename_destination)
    results["rename"] = {
        "passed": not rename_source.exists()
        and rename_destination.read_bytes() == b"rename",
        "status": "tested",
    }

    link_source = root / "link-source.txt"
    link_destination = root / "link-destination.txt"
    link_source.write_bytes(b"link")
    os.link(link_source, link_destination)
    results["link"] = {
        "passed": link_destination.read_bytes() == b"link",
        "status": "tested",
    }

    symlink_source = root / "symlink-source.txt"
    symlink_destination = root / "symlink-destination.txt"
    symlink_source.write_bytes(b"symlink")
    try:
        os.symlink(symlink_source, symlink_destination)
    except OSError as error:
        results["symlink"] = {
            "passed": True,
            "status": "unavailable_at_runtime",
            "error_type": type(error).__name__,
        }
    else:
        results["symlink"] = {
            "passed": symlink_destination.read_bytes() == b"symlink",
            "status": "tested",
        }

    truncate_path = root / "truncate.txt"
    truncate_path.write_bytes(b"truncate")
    os.truncate(truncate_path, 3)
    results["truncate"] = {
        "passed": truncate_path.read_bytes() == b"tru",
        "status": "tested",
    }

    chmod_path = root / "chmod.txt"
    chmod_path.write_bytes(b"chmod")
    os.chmod(chmod_path, stat.S_IREAD | stat.S_IWRITE)
    results["chmod"] = {"passed": chmod_path.is_file(), "status": "tested"}

    chown = getattr(os, "chown", None)
    if chown is None:
        results["chown"] = {
            "passed": True,
            "status": "unavailable_on_platform",
        }
    else:
        chown_path = root / "chown.txt"
        chown_path.write_bytes(b"chown")
        chown(chown_path, os.getuid(), os.getgid())
        results["chown"] = {"passed": chown_path.is_file(), "status": "tested"}

    utime_path = root / "utime.txt"
    utime_path.write_bytes(b"utime")
    os.utime(utime_path, ns=(1_000_000_000, 1_000_000_000))
    results["utime"] = {
        "passed": utime_path.stat().st_mtime_ns == 1_000_000_000,
        "status": "tested",
    }

    temporary_flag = getattr(os, "O_TEMPORARY", None)
    if temporary_flag is None:
        results["O_TEMPORARY"] = {
            "passed": True,
            "status": "unavailable_on_platform",
        }
    else:
        temporary_path = root / "temporary.txt"
        temporary_path.write_bytes(b"temporary")
        descriptor = os.open(temporary_path, os.O_RDONLY | temporary_flag)
        os.close(descriptor)
        results["O_TEMPORARY"] = {
            "passed": not temporary_path.exists(),
            "status": "tested",
        }

    fd_path = root / "fd.txt"
    descriptor = os.open(fd_path, os.O_CREAT | os.O_RDWR)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write("fd-ok")
    results["integer_fd"] = {
        "passed": fd_path.read_text(encoding="utf-8") == "fd-ok",
        "status": "tested",
    }

    read_fd, write_fd = os.pipe()
    try:
        guard.register_pipe(write_fd)
        with os.fdopen(write_fd, "wb") as stream:
            stream.write(b"pipe-ok")
        write_fd = -1
        pipe_bytes = os.read(read_fd, 64)
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)
    results["anonymous_pipe"] = {
        "passed": pipe_bytes == b"pipe-ok",
        "status": "tested",
    }
    return results


def _prepare_outside(outside: Path) -> None:
    outside.mkdir(parents=True, exist_ok=False)
    (outside / "remove.txt").write_bytes(b"remove")
    (outside / "rmdir").mkdir()
    (outside / "rename-source.txt").write_bytes(b"rename")
    (outside / "link-source.txt").write_bytes(b"link")
    (outside / "symlink-source.txt").write_bytes(b"symlink")
    (outside / "truncate.txt").write_bytes(b"truncate")
    (outside / "chmod.txt").write_bytes(b"chmod")
    (outside / "chown.txt").write_bytes(b"chown")
    (outside / "utime.txt").write_bytes(b"utime")
    (outside / "temporary.txt").write_bytes(b"temporary")
    (outside / "fd.txt").write_bytes(b"fd-outside")


def _outside_rejections(outside: Path, preopened_fd: int) -> dict[str, Any]:
    results = {
        "mkdir": _attempt(
            lambda: os.mkdir(outside / "mkdir"),
            lambda: not (outside / "mkdir").exists(),
        ),
        "remove": _attempt(
            lambda: os.remove(outside / "remove.txt"),
            lambda: (outside / "remove.txt").read_bytes() == b"remove",
        ),
        "rmdir": _attempt(
            lambda: os.rmdir(outside / "rmdir"),
            lambda: (outside / "rmdir").is_dir(),
        ),
        "rename": _attempt(
            lambda: os.rename(
                outside / "rename-source.txt", outside / "rename-destination.txt"
            ),
            lambda: (outside / "rename-source.txt").read_bytes() == b"rename"
            and not (outside / "rename-destination.txt").exists(),
        ),
        "link": _attempt(
            lambda: os.link(
                outside / "link-source.txt", outside / "link-destination.txt"
            ),
            lambda: (outside / "link-source.txt").read_bytes() == b"link"
            and not (outside / "link-destination.txt").exists(),
        ),
        "symlink": _attempt(
            lambda: os.symlink(
                outside / "symlink-source.txt",
                outside / "symlink-destination.txt",
            ),
            lambda: (outside / "symlink-source.txt").read_bytes() == b"symlink"
            and not (outside / "symlink-destination.txt").exists(),
        ),
        "truncate": _attempt(
            lambda: os.truncate(outside / "truncate.txt", 2),
            lambda: (outside / "truncate.txt").read_bytes() == b"truncate",
        ),
        "chmod": _attempt(
            lambda: os.chmod(outside / "chmod.txt", stat.S_IREAD),
            lambda: (outside / "chmod.txt").is_file(),
        ),
        "utime": _attempt(
            lambda: os.utime(
                outside / "utime.txt", ns=(1_000_000_000, 1_000_000_000)
            ),
            lambda: (outside / "utime.txt").stat().st_mtime_ns != 1_000_000_000,
        ),
        "FileHandler_delay": _attempt(
            lambda: logging.FileHandler(outside / "handler.log", delay=True),
            lambda: not (outside / "handler.log").exists(),
        ),
    }
    chown = getattr(os, "chown", None)
    if chown is None:
        results["chown"] = {
            "passed": True,
            "status": "unavailable_on_platform",
        }
    else:
        results["chown"] = _attempt(
            lambda: chown(outside / "chown.txt", os.getuid(), os.getgid()),
            lambda: (outside / "chown.txt").is_file(),
        )
    temporary_flag = getattr(os, "O_TEMPORARY", None)
    if temporary_flag is None:
        results["O_TEMPORARY"] = {
            "passed": True,
            "status": "unavailable_on_platform",
        }
    else:
        results["O_TEMPORARY"] = _attempt(
            lambda: os.close(
                os.open(
                    outside / "temporary.txt",
                    os.O_RDONLY | temporary_flag,
                )
            ),
            lambda: (outside / "temporary.txt").read_bytes() == b"temporary",
        )

    duplicated = os.dup(int(preopened_fd))
    try:
        results["integer_fd"] = _attempt(
            lambda: os.fdopen(duplicated, "w", encoding="utf-8"),
            lambda: (outside / "fd.txt").read_bytes() == b"fd-outside",
        )
    finally:
        try:
            os.close(duplicated)
        except OSError:
            pass
    return results


def _run_child(
    output: Path,
    run_root: Path,
    mode: str,
    *,
    outside_root: Path,
    preopened_fd: int,
) -> dict[str, Any]:
    output = require_path_within(output.resolve(), run_root, label="guard child output")
    outside = require_path_within(
        outside_root.resolve(strict=True), run_root, label="guard child outside root"
    )
    if not output.is_dir() or any(output.iterdir()):
        raise RuntimeError("Guard child output must be a pre-created empty directory")
    if output == outside or output in outside.parents or outside in output.parents:
        raise RuntimeError("Guard child allowed and outside roots must be disjoint")
    allowed = output
    guard = current_bootstrap_guard()
    if not isinstance(guard, PythonWriteGuard) or guard.run_root != allowed.resolve():
        raise RuntimeError("Guard child is not bound to its exact allowed root")

    if mode == "disabled":
        if guard.wrap_file_handlers:
            raise RuntimeError("Disabled FileHandler comparison unexpectedly enabled the wrapper")
        file_semantics = _file_semantics(allowed)
        payload = {
            "schema_version": "RootCauseGuardSemanticsChildV1",
            "mode": mode,
            "file_semantics": file_semantics,
            "guard": {
                "installed": guard.installed,
                "probe_count": guard.probe_count,
                "file_handler_wrapper": guard.wrap_file_handlers,
            },
            "passed": bool(guard.installed and guard.probe_count == 1),
        }
        (allowed / "result.json").write_bytes(canonical_json_bytes(payload) + b"\n")
        return payload

    if not guard.wrap_file_handlers:
        raise RuntimeError("Enabled FileHandler comparison is missing the wrapper")
    file_semantics = _file_semantics(allowed)
    inside = _inside_mutations(allowed, guard)
    outside_results = _outside_rejections(outside, preopened_fd)
    payload = {
        "schema_version": "RootCauseGuardSemanticsChildV1",
        "mode": mode,
        "file_semantics": file_semantics,
        "inside_mutations": inside,
        "outside_rejections": outside_results,
        "guard": {
            "installed": guard.installed,
            "probe_count": guard.probe_count,
            "file_handler_wrapper": guard.wrap_file_handlers,
            "created_lifetime_handlers": guard.created_lifetime_handlers,
            "capability_manifest": guard.capability_manifest,
            "ledger_event_names": sorted({row["event"] for row in guard.ledger}),
        },
        "passed": bool(
            all(item["passed"] for item in inside.values())
            and all(item["passed"] for item in outside_results.values())
            and guard.installed
            and guard.probe_count == 1
        ),
    }
    (allowed / "result.json").write_bytes(canonical_json_bytes(payload) + b"\n")
    return payload


def _run_parent(output: Path, run_root: Path) -> dict[str, Any]:
    output = require_path_within(output.resolve(), run_root, label="guard probe output")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    children: dict[str, dict[str, Any]] = {}
    for mode in ("disabled", "enabled"):
        child_output = output / f"{mode}-child"
        allowed = child_output / "allowed"
        outside = child_output / "outside"
        allowed.mkdir(parents=True)
        _prepare_outside(outside)
        command = [
            sys.executable,
            "-B",
            "-m",
            "debug.sim2sim.root_cause_suite.worker_bootstrap",
            "--guard-root",
            str(allowed),
            "--worker-module",
            "debug.sim2sim.root_cause_suite.guard_semantics_worker",
            "--file-handler-wrapper",
            mode,
            "--preopen-path",
            str(outside / "fd.txt"),
            "--",
            "child",
            "--output",
            str(allowed),
            "--run-root",
            str(run_root),
            "--mode",
            mode,
            "--outside-root",
            str(outside),
        ]
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPYCACHEPREFIX"] = str(allowed / "pycache")
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[3],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        (output / f"{mode}.stdout.txt").write_text(
            completed.stdout, encoding="utf-8", newline="\n"
        )
        (output / f"{mode}.stderr.txt").write_text(
            completed.stderr, encoding="utf-8", newline="\n"
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Guard semantics {mode} child failed with {completed.returncode}"
            )
        result_path = allowed / "result.json"
        children[mode] = json.loads(result_path.read_text(encoding="utf-8"))
        bootstrap = json.loads(
            (allowed / "bootstrap_guard.json").read_text(encoding="utf-8")
        )
        if bootstrap.get("file_handler_wrapper") != mode:
            raise RuntimeError(f"Guard semantics {mode} bootstrap mode drifted")

    disabled = children["disabled"]
    enabled = children["enabled"]
    semantics_equal = disabled["file_semantics"] == enabled["file_semantics"]
    payload = {
        "schema_version": "RootCauseGuardSemanticsV1",
        "file_handler_semantics_equal": semantics_equal,
        "disabled": disabled,
        "enabled": enabled,
        "child_result_sha256": {
            mode: sha256_file(output / f"{mode}-child" / "allowed" / "result.json")
            for mode in ("disabled", "enabled")
        },
        "passed": bool(
            disabled.get("passed") is True
            and enabled.get("passed") is True
            and semantics_equal
        ),
    }
    (output / "result.json").write_bytes(canonical_json_bytes(payload) + b"\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_root = args.run_root.resolve(strict=True)
    if args.command == "child":
        if args.mode is None:
            raise ValueError("guard child requires --mode")
        if args.outside_root is None or args.preopened_fd is None:
            raise ValueError("guard child requires --outside-root and --preopened-fd")
        payload = _run_child(
            args.output,
            run_root,
            args.mode,
            outside_root=args.outside_root,
            preopened_fd=args.preopened_fd,
        )
    else:
        payload = _run_parent(args.output, run_root)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0 if payload.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
