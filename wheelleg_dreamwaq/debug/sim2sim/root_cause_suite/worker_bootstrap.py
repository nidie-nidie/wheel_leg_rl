from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from .bootstrap_runtime import (
    bootstrap_finalized,
    clear_bootstrap_finalizer,
    register_bootstrap_guard,
    finalize_bootstrap,
    register_bootstrap_finalizer,
)
from .contracts import (
    BootstrapError,
    SUITE_ROOT,
    canonical_json_bytes,
    require_path_within,
)
from .python_write_guard import PythonWriteGuard


def _split_arguments(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    values = list(argv)
    if "--" not in values:
        raise BootstrapError("Worker bootstrap requires a -- argument separator")
    index = values.index("--")
    return values[:index], values[index + 1 :]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RootCauseSuite guarded worker bootstrap")
    parser.add_argument("--guard-root", type=Path, required=True)
    parser.add_argument("--worker-module", required=True)
    parser.add_argument(
        "--file-handler-wrapper",
        choices=("enabled", "disabled"),
        default="enabled",
    )
    parser.add_argument("--preopen-path", type=Path)
    return parser


def _argument_path(arguments: Sequence[str], flag: str) -> Path:
    values = list(arguments)
    try:
        index = values.index(flag)
    except ValueError as error:
        raise BootstrapError(f"Guarded worker is missing {flag}") from error
    if index + 1 >= len(values):
        raise BootstrapError(f"Guarded worker {flag} has no value")
    return Path(values[index + 1])


def main(argv: list[str] | None = None) -> int:
    if not sys.flags.dont_write_bytecode:
        raise BootstrapError("Guarded worker must be launched with -B")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise BootstrapError("Guarded worker requires PYTHONNOUSERSITE=1")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise BootstrapError("Guarded worker requires PYTHONDONTWRITEBYTECODE=1")
    bootstrap_args, worker_args = _split_arguments(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(bootstrap_args)
    guard_root = require_path_within(
        args.guard_root.resolve(strict=True), SUITE_ROOT, label="worker guard root"
    )
    output = require_path_within(
        _argument_path(worker_args, "--output"), guard_root, label="worker output"
    )
    prefix = os.environ.get("PYTHONPYCACHEPREFIX")
    if not prefix:
        raise BootstrapError("Guarded worker PYTHONPYCACHEPREFIX is missing")
    require_path_within(prefix, guard_root, label="worker pycache prefix")
    if not args.worker_module.startswith("debug.sim2sim.root_cause_suite."):
        raise BootstrapError(f"Worker module is outside the suite: {args.worker_module}")

    preopened_fd: int | None = None
    if args.preopen_path is not None:
        if args.worker_module != "debug.sim2sim.root_cause_suite.guard_semantics_worker":
            raise BootstrapError("Pre-opened descriptors are restricted to the guard probe")
        preopen_path = require_path_within(
            args.preopen_path.resolve(strict=True), SUITE_ROOT, label="pre-open path"
        )
        preopened_fd = os.open(preopen_path, os.O_RDWR)
        worker_args.extend(("--preopened-fd", str(preopened_fd)))

    guard = PythonWriteGuard(
        guard_root,
        wrap_file_handlers=args.file_handler_wrapper == "enabled",
    )
    original_argv = list(sys.argv)
    sys.argv = [args.worker_module, *worker_args]
    try:
        with guard:
            register_bootstrap_guard(guard)
            def write_bootstrap_evidence(return_code: int) -> None:
                output.mkdir(parents=True, exist_ok=True)
                payload = {
                    "schema_version": "RootCauseWorkerBootstrapV1",
                    "worker_module": args.worker_module,
                    "argv": worker_args,
                    "process_argv": list(sys.argv),
                    "guard_root": str(guard_root),
                    "output": str(output),
                    "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
                    "file_handler_wrapper": args.file_handler_wrapper,
                    "environment": {
                        key: os.environ.get(key)
                        for key in (
                            "PYTHONNOUSERSITE",
                            "PYTHONDONTWRITEBYTECODE",
                            "PYTHONPYCACHEPREFIX",
                        )
                    },
                    "write_guard": {
                        "installed": guard.installed,
                        "probe_count": guard.probe_count,
                        "pre_handlers": guard.pre_handlers,
                        "created_lifetime_handlers": guard.created_lifetime_handlers,
                        "post_handlers": guard.post_handlers(),
                        "capability_manifest": guard.capability_manifest,
                        "ledger": guard.ledger,
                        "ledger_cutoff": (
                            "immediately_before_bootstrap_identity_write"
                        ),
                    },
                    "return_code": int(return_code),
                }
                (output / "bootstrap_guard.json").write_bytes(
                    canonical_json_bytes(payload) + b"\n"
                )

            register_bootstrap_finalizer(write_bootstrap_evidence)
            module = importlib.import_module(args.worker_module)
            worker_main = getattr(module, "main", None)
            if not callable(worker_main):
                raise BootstrapError(
                    f"Worker module has no callable main(): {args.worker_module}"
                )
            return_code = int(worker_main(worker_args))
            if not bootstrap_finalized():
                finalize_bootstrap(return_code)
            return return_code
    finally:
        clear_bootstrap_finalizer()
        sys.argv = original_argv
        if preopened_fd is not None:
            try:
                os.close(preopened_fd)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
