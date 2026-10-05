"""Isolated-process capacity qualification for the A1 PACE task."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import shlex
from typing import Any, Mapping, Sequence, cast


PASS = "PASS"
FAIL = "FAIL"


class CapacitySmokeError(ValueError):
    """Raised when capacity evidence is incomplete or no candidate passes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CapacitySmokeError(message)


def select_formal_num_envs(
    candidates: Sequence[int],
    records: Sequence[Mapping[str, Any]],
    repeats: int,
) -> int:
    """Select the largest candidate for which every isolated repeat passed."""

    _require(type(repeats) is int and repeats > 0, "repeats must be a positive integer")
    normalized = tuple(candidates)
    _require(bool(normalized), "capacity candidates must not be empty")
    _require(
        all(type(value) is int and value > 0 for value in normalized),
        "capacity candidates must be positive integers",
    )
    _require(
        len(normalized) == len(set(normalized)), "capacity candidates must be unique"
    )

    expected = {
        (candidate, repeat) for candidate in normalized for repeat in range(repeats)
    }
    indexed: dict[tuple[int, int], str] = {}
    for record in records:
        candidate = record.get("candidate")
        repeat = record.get("repeat")
        status = record.get("status")
        _require(
            type(candidate) is int and type(repeat) is int,
            "capacity record candidate/repeat must be integers",
        )
        key = (candidate, repeat)
        _require(key in expected, "capacity record has an unexpected candidate/repeat")
        _require(key not in indexed, "capacity records contain a duplicate repeat")
        _require(status in (PASS, FAIL), "capacity record status must be PASS or FAIL")
        typed_key = cast(tuple[int, int], key)
        indexed[typed_key] = str(status)
    _require(set(indexed) == expected, "capacity records are incomplete")

    passing = [
        candidate
        for candidate in normalized
        if all(indexed[(candidate, repeat)] == PASS for repeat in range(repeats))
    ]
    _require(bool(passing), "no candidate passed every repeat")
    return max(passing)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _a1_extension_hashes(repository_root: Path) -> dict[str, str]:
    relative_paths = (
        "source/pace_sim2real/pace_sim2real/tasks/manager_based/__init__.py",
        "source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/a1_mean.py",
        "source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/a1_pace_env_cfg.py",
        "source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/a1_replay.py",
    )
    return {
        relative: _sha256_path(repository_root / relative)
        for relative in relative_paths
    }


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def _load_worker_record(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise CapacitySmokeError(f"worker did not produce valid JSON: {exc}") from exc
    _require(isinstance(value, dict), "worker JSON root must be an object")
    return value


def _tail(value: str, limit: int = 8000) -> str:
    return value[-limit:]


def _worker_command(
    *,
    isaaclab_root: Path,
    script_path: Path,
    task: str,
    candidate: int,
    repeat: int,
    steps: int,
    output: Path,
    device: str,
    headless: bool,
) -> list[str]:
    command = [
        str(isaaclab_root / "isaaclab.sh"),
        "-p",
        str(script_path),
        "--_worker",
        "--task",
        task,
        "--candidate",
        str(candidate),
        "--repeat",
        str(repeat),
        "--steps",
        str(steps),
        "--worker-output",
        str(output),
        "--device",
        device,
    ]
    if headless:
        command.append("--headless")
    return command


def _run_parent(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    _require(
        args.task == "Isaac-Pace-A1-v0", "capacity smoke accepts only Isaac-Pace-A1-v0"
    )
    _require(
        type(args.repeats) is int and args.repeats >= 2,
        "capacity qualification requires at least two repeats",
    )
    _require(type(args.steps) is int and args.steps > 0, "smoke steps must be positive")
    _require(
        math.isfinite(args.timeout_s) and args.timeout_s > 0,
        "worker timeout must be finite and positive",
    )
    candidates = tuple(args.candidates)
    _require(bool(candidates), "capacity candidates must not be empty")
    _require(
        all(type(value) is int and value > 0 for value in candidates),
        "capacity candidates must be positive integers",
    )
    _require(
        len(candidates) == len(set(candidates)), "capacity candidates must be unique"
    )

    repository_root = args.repository_root.resolve()
    isaaclab_root = args.isaaclab_root.resolve()
    launcher = isaaclab_root / "isaaclab.sh"
    script_path = Path(__file__).resolve()
    _require(launcher.is_file(), f"Isaac Lab launcher does not exist: {launcher}")
    _require(
        repository_root.is_dir(), f"PACE repository does not exist: {repository_root}"
    )
    output = args.output.resolve()
    records: list[dict[str, Any]] = []
    environment = os.environ.copy()
    environment["PACE_ROOT"] = str(repository_root)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".a1-capacity-", dir=output.parent
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        for candidate in candidates:
            for repeat in range(args.repeats):
                worker_output = temporary_root / f"{candidate}-{repeat}.json"
                command = _worker_command(
                    isaaclab_root=isaaclab_root,
                    script_path=script_path,
                    task=args.task,
                    candidate=candidate,
                    repeat=repeat,
                    steps=args.steps,
                    output=worker_output,
                    device=args.device,
                    headless=args.headless,
                )
                started = time.monotonic()
                try:
                    child = subprocess.run(
                        command,
                        cwd=repository_root,
                        env=environment,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=args.timeout_s,
                    )
                    duration_s = time.monotonic() - started
                    if worker_output.is_file():
                        record = _load_worker_record(worker_output)
                    else:
                        record = {
                            "candidate": candidate,
                            "repeat": repeat,
                            "status": FAIL,
                            "error": "worker produced no result artifact",
                        }
                    if child.returncode != 0:
                        record["status"] = FAIL
                        record.setdefault(
                            "error", f"worker exited with code {child.returncode}"
                        )
                    record["exit_code"] = child.returncode
                    record["stdout_tail"] = _tail(child.stdout)
                    record["stderr_tail"] = _tail(child.stderr)
                except subprocess.TimeoutExpired as exc:
                    duration_s = time.monotonic() - started
                    record = {
                        "candidate": candidate,
                        "repeat": repeat,
                        "status": FAIL,
                        "exit_code": None,
                        "error": f"worker exceeded {args.timeout_s:.3f} s timeout",
                        "stdout_tail": _tail(
                            exc.stdout.decode(errors="replace")
                            if isinstance(exc.stdout, bytes)
                            else (exc.stdout or "")
                        ),
                        "stderr_tail": _tail(
                            exc.stderr.decode(errors="replace")
                            if isinstance(exc.stderr, bytes)
                            else (exc.stderr or "")
                        ),
                    }
                record["candidate"] = candidate
                record["repeat"] = repeat
                record["duration_s"] = duration_s
                record["command"] = command
                records.append(record)

    report: dict[str, Any] = {
        "schema_version": "a1_pace_capacity/v1",
        "exact_command": shlex.join(sys.argv),
        "capacity_script_sha256": _sha256_path(script_path),
        "repository_revision": _git_head(repository_root),
        "a1_extension_source_sha256": _a1_extension_hashes(repository_root),
        "task": args.task,
        "candidates": list(candidates),
        "repeats": args.repeats,
        "steps_per_repeat": args.steps,
        "device": args.device,
        "worker_timeout_s": args.timeout_s,
        "fit_buffer_contract": {
            "sample_count": 10_000,
            "joint_count": 12,
            "parameter_count": 49,
            "save_optimization_process": True,
        },
        "records": records,
    }
    try:
        formal = select_formal_num_envs(candidates, records, args.repeats)
    except CapacitySmokeError as exc:
        report.update({"status": FAIL, "formal_num_envs": None, "error": str(exc)})
        return 2, report
    report.update({"status": PASS, "formal_num_envs": formal})
    return 0, report


def _run_worker(args: argparse.Namespace) -> int:
    _require(args.worker_output is not None, "worker mode requires --worker-output")
    _require(
        type(args.candidate) is int and args.candidate > 0,
        "worker candidate must be positive",
    )
    _require(
        type(args.repeat) is int and args.repeat >= 0,
        "worker repeat must be non-negative",
    )
    _require(
        type(args.steps) is int and args.steps > 0, "worker steps must be positive"
    )

    record: dict[str, Any] = {
        "candidate": args.candidate,
        "repeat": args.repeat,
        "status": FAIL,
        "steps_completed": 0,
    }
    simulation_app = None
    env = None
    try:
        # App launch is intentionally confined to this disposable child process.
        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(headless=args.headless).app
        import gymnasium as gym
        import torch

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        import pace_sim2real.tasks  # noqa: F401

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            minimum_free_bytes = int(free_bytes)
            record["device_total_bytes"] = int(total_bytes)
            record["device_free_before_bytes"] = int(free_bytes)
            record["gpu_name"] = torch.cuda.get_device_name()
            record["torch_version"] = torch.__version__
            record["torch_cuda_version"] = torch.version.cuda
        cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.candidate)
        cfg.seed = 1701 + args.repeat
        env = gym.make(args.task, cfg=cfg)
        env.reset(seed=1701 + args.repeat)
        unwrapped = cast(Any, env.unwrapped)
        fit_buffers: list[Any] = []
        if torch.cuda.is_available():
            parameter_count = int(cfg.sim2real.bounds_params.shape[0])
            max_iteration = int(cfg.sim2real.cmaes.max_iteration)
            fit_buffers = [
                torch.empty(
                    (args.candidate, 10_000, len(cfg.sim2real.joint_order)),
                    device=unwrapped.device,
                    dtype=torch.float32,
                ),
                torch.empty(
                    (max_iteration, args.candidate, parameter_count),
                    device=unwrapped.device,
                    dtype=torch.float32,
                ),
                torch.empty(
                    (max_iteration, args.candidate),
                    device=unwrapped.device,
                    dtype=torch.float32,
                ),
                torch.empty(
                    (args.candidate, parameter_count * 2 + 1),
                    device=unwrapped.device,
                    dtype=torch.float32,
                ),
            ]
            record["reserved_fit_buffer_bytes"] = int(
                sum(value.numel() * value.element_size() for value in fit_buffers)
            )
            free_bytes, _ = torch.cuda.mem_get_info()
            minimum_free_bytes = min(minimum_free_bytes, int(free_bytes))
        for step in range(args.steps):
            action_shape = cast(Sequence[int], env.action_space.shape)
            actions = torch.zeros(
                action_shape,
                device=unwrapped.device,
                dtype=torch.float32,
            )
            env.step(actions)
            record["steps_completed"] = step + 1
            if torch.cuda.is_available():
                free_bytes, _ = torch.cuda.mem_get_info()
                minimum_free_bytes = min(minimum_free_bytes, int(free_bytes))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            record["peak_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            record["peak_gpu_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            minimum_free_bytes = min(minimum_free_bytes, int(free_bytes))
            record["device_free_after_bytes"] = int(free_bytes)
            record["peak_whole_device_used_bytes"] = int(
                total_bytes - minimum_free_bytes
            )
        else:
            record["peak_gpu_allocated_bytes"] = 0
            record["peak_gpu_reserved_bytes"] = 0
            record["reserved_fit_buffer_bytes"] = 0
            record["peak_whole_device_used_bytes"] = 0
        record["status"] = PASS
    except Exception as exc:  # worker must leave machine-readable failure evidence
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                record["status"] = FAIL
                record["close_error"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(args.worker_output.resolve(), record)
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:
                pass
    return 0 if record["status"] == PASS else 2


def _build_parser() -> argparse.ArgumentParser:
    repository_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Qualify an A1 PACE CMA-ES population in isolated processes"
    )
    parser.add_argument("--task", default="Isaac-Pace-A1-v0")
    parser.add_argument(
        "--candidates", type=int, nargs="+", default=[4096, 2048, 1024, 512, 256]
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repository-root", type=Path, default=repository_default)
    parser.add_argument(
        "--isaaclab-root",
        type=Path,
        default=Path(os.environ.get("ISAACLAB_ROOT", "/home/changba01/IsaacLab")),
    )
    parser.add_argument("--headless", action="store_true")

    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--candidate", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args._worker:
            return _run_worker(args)
        _require(args.output is not None, "parent mode requires --output")
        return_code, report = _run_parent(args)
        _atomic_json(args.output.resolve(), report)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return return_code
    except (CapacitySmokeError, OSError, subprocess.SubprocessError) as exc:
        print(f"[A1 PACE CAPACITY] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
