from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import subprocess
import sys
import threading
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIM2SIM_PROJECT = PROJECT_ROOT / "sim2sim" / "mujoco"
FORMAL_RUN_COUNT = 4
FORMAL_ITERATIONS = 1000
SUITE_SOURCE_SCHEMA_VERSION = "TrainingSuiteSourceV1"
TRAINING_SUITE_SCHEMA_VERSION = "TrainingSuiteV3"
TRAINING_SUITE_ARTIFACT_DIR = "phase1_randomized_v3"
TRAINING_RUN_PREFIX = "phase1r-v3"
RUN_DIR_PATTERN = re.compile(r"^\[INFO\] Run directory: (.+)$")
MONITOR_TAGS = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Loss/value_function",
    "Loss/surrogate",
    "Loss/entropy",
    "Loss/learning_rate",
    "Policy/mean_noise_std",
    "Action/saturation_fraction",
    "Actuator/effort_saturation_fraction",
    "Tracking/vx_abs_error",
    "Tracking/yaw_rate_abs_error",
    "Tracking/base_height_abs_error",
    "Metric/phi0_delta_abs_rad",
    "Metric/loop_closure_error_max_m",
    "Termination/invalid",
    "Termination/height_terminated",
    "Termination/tilt_terminated",
    "Termination/root_linear_terminated",
    "Termination/root_angular_terminated",
    "Termination/joint_velocity_terminated",
)


def _stable_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest().upper()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _validate_suite_mode(mode: str, *, runs: int, iterations: int) -> None:
    if mode == "formal" and (runs != FORMAL_RUN_COUNT or iterations != FORMAL_ITERATIONS):
        raise ValueError("Formal suite requires exactly 4 runs x 1000 iterations")
    if mode not in {"formal", "smoke"}:
        raise ValueError(f"Unsupported suite mode: {mode}")


def _suite_source_fingerprint() -> tuple[dict, str]:
    files = []
    for root in (PROJECT_ROOT / "source", PROJECT_ROOT / "scripts", PROJECT_ROOT / "configs"):
        files.extend(path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    files.extend(
        PROJECT_ROOT / name
        for name in ("pyproject.toml", "dependency-manifest.toml", "uv.lock")
    )
    files.extend(path for path in (SIM2SIM_PROJECT / "wheelleg_mujoco").rglob("*.py") if path.is_file())
    files.extend(
        (
            SIM2SIM_PROJECT / "pyproject.toml",
            SIM2SIM_PROJECT / "uv.lock",
            SIM2SIM_PROJECT / "model_manifest.json",
        )
    )
    unique_files = sorted(set(files), key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())
    payload = {
        "schema_version": SUITE_SOURCE_SCHEMA_VERSION,
        "files": {
            path.relative_to(PROJECT_ROOT).as_posix(): _sha256_file(path)
            for path in unique_files
        },
    }
    return payload, _stable_hash(payload)


def _assert_suite_source_fingerprint(expected_hash: str) -> None:
    _, actual_hash = _suite_source_fingerprint()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Suite source fingerprint changed during execution: {actual_hash} != {expected_hash}"
        )


def _training_fingerprint(run_manifest: dict) -> tuple[dict, str]:
    payload = json.loads(json.dumps(run_manifest))
    for key in ("command", "created_at", "seed", "resume"):
        payload.pop(key, None)
    payload.pop("randomization_audit", None)
    payload.pop("closed_chain_reset_cache", None)
    randomness = payload.get("randomness", {})
    randomness.pop("master_seed", None)
    randomness.pop("effective_process_seed", None)
    randomness.pop("environment_streams", None)
    # The raw YAML files retain per-run seed, run name, and log directory for provenance.
    # Phase1RandomizedContractV3 is the canonical seed-independent semantic record.
    resolved_hashes = payload.get("resolved_config_sha256", {})
    resolved_hashes.pop("environment", None)
    resolved_hashes.pop("agent", None)
    return payload, _stable_hash(payload)


def _window_mean(values: list[float], count: int = 20) -> float:
    window = values[-min(count, len(values)) :]
    return sum(window) / len(window)


def _classify_monitor_values(
    tag_values: dict[str, list[float]],
    *,
    missing_tags: list[str],
    event_file_missing: bool,
    reason: str,
) -> tuple[list[str], list[str]]:
    warnings = []
    hard_anomalies = []
    if event_file_missing:
        target = hard_anomalies if reason == "post_run" else warnings
        target.append("event_file_missing")
    if missing_tags:
        target = hard_anomalies if reason == "post_run" else warnings
        target.extend(f"missing_tag:{tag}" for tag in missing_tags)
    for tag, values in tag_values.items():
        if any(not math.isfinite(value) for value in values):
            hard_anomalies.append(f"non_finite:{tag}")

    value_loss = tag_values.get("Loss/value_function", [])
    if value_loss and max(value_loss) > 1.0e6:
        hard_anomalies.append("value_loss_above_1e6")
    loop_closure = tag_values.get("Metric/loop_closure_error_max_m", [])
    if loop_closure and max(loop_closure) > 5.0e-3:
        hard_anomalies.append("loop_closure_above_5mm")
    invalid = tag_values.get("Termination/invalid", [])
    if invalid and max(invalid) > 0.0:
        hard_anomalies.append("invalid_termination_nonzero")

    for tag, label in (
        ("Action/saturation_fraction", "action_saturation_sustained_above_95_percent"),
        ("Actuator/effort_saturation_fraction", "effort_saturation_sustained_above_95_percent"),
    ):
        values = tag_values.get(tag, [])
        if values and _window_mean(values) > 0.95:
            warnings.append(label)
    episode_length = tag_values.get("Train/mean_episode_length", [])
    if len(episode_length) >= 20 and episode_length[-1] < 0.25 * max(episode_length[-20:]):
        warnings.append("recent_episode_length_collapse")
    entropy = tag_values.get("Loss/entropy", [])
    if len(entropy) >= 20 and entropy[-1] < 0.1 * max(entropy[:10]):
        warnings.append("entropy_collapsed_below_10_percent_of_initial")

    trend_limits = {
        "Tracking/vx_abs_error": (0.75, "vx_tracking_degraded"),
        "Tracking/yaw_rate_abs_error": (0.75, "yaw_tracking_degraded"),
        "Tracking/base_height_abs_error": (0.04, "base_height_tracking_degraded"),
        "Metric/phi0_delta_abs_rad": (0.20, "phi0_symmetry_degraded"),
    }
    for tag, (absolute_limit, label) in trend_limits.items():
        values = tag_values.get(tag, [])
        if len(values) >= 40:
            first = sum(values[:20]) / 20.0
            last = _window_mean(values)
            if last > absolute_limit and last > 1.5 * max(first, 1.0e-9):
                warnings.append(label)
    for tag in (
        "Termination/height_terminated",
        "Termination/tilt_terminated",
        "Termination/root_linear_terminated",
        "Termination/root_angular_terminated",
        "Termination/joint_velocity_terminated",
    ):
        values = tag_values.get(tag, [])
        if values and _window_mean(values) > 0.25:
            warnings.append(f"termination_rate_above_25_percent:{tag}")
    return sorted(set(warnings)), sorted(set(hard_anomalies))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _tensorboard_snapshot(run_dir: Path, output: Path, *, reason: str) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda path: path.stat().st_mtime)
    snapshot = {
        "captured_at": datetime.now().astimezone().isoformat(),
        "reason": reason,
        "run_dir": str(run_dir.resolve()),
        "event_file": None,
        "latest": {},
        "series": {},
        "warnings": [],
        "hard_anomalies": [],
        "anomalies": [],
    }
    if not event_files:
        warnings, hard_anomalies = _classify_monitor_values(
            {},
            missing_tags=[],
            event_file_missing=True,
            reason=reason,
        )
        snapshot.update(
            {
                "warnings": warnings,
                "hard_anomalies": hard_anomalies,
                "anomalies": warnings + hard_anomalies,
                "status": "hard_anomaly" if hard_anomalies else "warning",
            }
        )
        _write_json(output, snapshot)
        return snapshot
    event_file = event_files[-1]
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags()["scalars"])
    snapshot["event_file"] = str(event_file.resolve())
    tag_values = {}
    missing_tags = []
    for tag in MONITOR_TAGS:
        if tag not in scalar_tags:
            missing_tags.append(tag)
            continue
        events = accumulator.Scalars(tag)
        if not events:
            missing_tags.append(tag)
            continue
        latest = events[-1]
        value = float(latest.value)
        snapshot["latest"][tag] = {"step": int(latest.step), "value": value}
        values = [float(event.value) for event in events]
        tag_values[tag] = values
        finite_values = [item for item in values if math.isfinite(item)]
        snapshot["series"][tag] = {
            "count": len(values),
            "first_step": int(events[0].step),
            "last_step": int(events[-1].step),
            "first_value": values[0],
            "last_value": values[-1],
            "minimum": min(finite_values) if finite_values else None,
            "maximum": max(finite_values) if finite_values else None,
            "non_finite_count": len(values) - len(finite_values),
        }
    warnings, hard_anomalies = _classify_monitor_values(
        tag_values,
        missing_tags=missing_tags,
        event_file_missing=False,
        reason=reason,
    )
    snapshot["warnings"] = warnings
    snapshot["hard_anomalies"] = hard_anomalies
    snapshot["anomalies"] = warnings + hard_anomalies
    snapshot["status"] = "hard_anomaly" if hard_anomalies else "warning" if warnings else "ok"
    _write_json(output, snapshot)
    return snapshot


def _monitor_report(monitor_dir: Path) -> dict:
    snapshots = []
    warnings = set()
    hard_anomalies = set()
    for path in sorted(monitor_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        snapshots.append(str(path.resolve()))
        payload_warnings = payload.get("warnings", [])
        payload_hard_anomalies = payload.get("hard_anomalies", [])
        if payload.get("reason") != "post_run":
            payload_warnings = [
                item
                for item in payload_warnings
                if item != "event_file_missing" and not item.startswith("missing_tag:")
            ]
            payload_hard_anomalies = [
                item
                for item in payload_hard_anomalies
                if item != "event_file_missing" and not item.startswith("missing_tag:")
            ]
        warnings.update(payload_warnings)
        hard_anomalies.update(payload_hard_anomalies)
    return {
        "status": "hard_anomaly" if hard_anomalies else "warning" if warnings else "ok",
        "warnings": sorted(warnings),
        "hard_anomalies": sorted(hard_anomalies),
        "snapshots": snapshots,
    }


def _stream_process(command: list[str], log_path: Path, *, monitor_dir: Path, monitor_interval_s: float) -> tuple[int, Path | None]:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    messages: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            messages.put(line)
        messages.put(None)

    threading.Thread(target=reader, daemon=True).start()
    run_dir = None
    next_monitor = time.monotonic() + monitor_interval_s
    initial_snapshot_done = False
    with log_path.open("w", encoding="utf-8") as log:
        stream_finished = False
        while not stream_finished or process.poll() is None:
            try:
                message = messages.get(timeout=1.0)
            except queue.Empty:
                message = ""
            if message is None:
                stream_finished = True
            elif message:
                log.write(message)
                log.flush()
                print(message, end="", flush=True)
                match = RUN_DIR_PATTERN.match(message.strip())
                if match:
                    run_dir = Path(match.group(1)).resolve()
            now = time.monotonic()
            if run_dir is not None and not initial_snapshot_done and list(run_dir.glob("events.out.tfevents.*")):
                initial = _tensorboard_snapshot(run_dir, monitor_dir / "initial.json", reason="initial")
                initial_snapshot_done = bool(initial["latest"])
            if run_dir is not None and now >= next_monitor:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                _tensorboard_snapshot(run_dir, monitor_dir / f"interval-{timestamp}.json", reason="30_minute_interval")
                next_monitor = now + monitor_interval_s
        return_code = process.wait()
    if run_dir is not None:
        _tensorboard_snapshot(run_dir, monitor_dir / "final.json", reason="post_run")
    return return_code, run_dir


def _run_checked(command: list[str], log_path: Path) -> None:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a formal or smoke WheelLeg PPO/MuJoCo suite.")
    parser.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--profile", choices=("portable", "rtx4060", "rtx5070"), default="rtx5070")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--monitor-interval-seconds", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.iterations <= 0 or args.runs <= 0 or args.monitor_interval_seconds <= 0.0:
        raise ValueError("iterations, runs, and monitor interval must be positive")
    _validate_suite_mode(args.mode, runs=args.runs, iterations=args.iterations)

    suite_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    suite_root = PROJECT_ROOT / "artifacts" / TRAINING_SUITE_ARTIFACT_DIR
    suite_dir = (args.output or suite_root / f"training-suite-{suite_id}").resolve()
    suite_dir.mkdir(parents=True, exist_ok=False)
    suite_source_payload, suite_source_hash = _suite_source_fingerprint()
    _write_json(suite_dir / "suite-source-fingerprint.json", suite_source_payload)
    random_source = secrets.SystemRandom()
    seeds = []
    while len(seeds) < args.runs:
        candidate = random_source.randrange(1, 2**31)
        if candidate not in seeds:
            seeds.append(candidate)
    suite_manifest_path = suite_dir / "training-suite-manifest.json"
    suite_manifest = {
        "schema_version": TRAINING_SUITE_SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "suite_id": suite_id,
        "suite_mode": args.mode,
        "suite_source_fingerprint_hash": suite_source_hash,
        "profile": args.profile,
        "iterations_per_run": args.iterations,
        "run_count": args.runs,
        "fresh_start": True,
        "seeds": seeds,
        "monitor_interval_seconds": args.monitor_interval_seconds,
        "runs": [],
    }
    _write_json(suite_manifest_path, suite_manifest)

    evaluation_summaries = []
    reference_fingerprint = None
    evaluation_context_path = None
    suite_warnings = set()
    suite_hard_anomalies = set()
    for index, seed in enumerate(seeds, start=1):
        _assert_suite_source_fingerprint(suite_source_hash)
        run_name = f"{TRAINING_RUN_PREFIX}-suite-{suite_id}-run{index:02d}-seed{seed}"
        record = {"index": index, "seed": seed, "run_name": run_name, "status": "training"}
        suite_manifest["runs"].append(record)
        _write_json(suite_manifest_path, suite_manifest)
        train_command = [
            sys.executable,
            "scripts/train_ppo.py",
            "--profile",
            args.profile,
            "--max-iterations",
            str(args.iterations),
            "--seed",
            str(seed),
            "--run-name",
            run_name,
            "--randomization-profile",
            "fudan-v1",
            "--headless",
        ]
        return_code, run_dir = _stream_process(
            train_command,
            suite_dir / f"run-{index:02d}-training.log",
            monitor_dir=suite_dir / "monitor" / f"run-{index:02d}",
            monitor_interval_s=args.monitor_interval_seconds,
        )
        if return_code != 0 or run_dir is None:
            record.update({"status": "training_failed", "return_code": return_code})
            _write_json(suite_manifest_path, suite_manifest)
            continue
        _assert_suite_source_fingerprint(suite_source_hash)
        training_summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
        if training_summary["completed_iterations"] != args.iterations or training_summary["resumed_from"] is not None:
            record.update(
                {
                    "status": "training_summary_rejected",
                    "run_dir": str(run_dir),
                    "training_summary": str(run_dir / "training_summary.json"),
                }
            )
            _write_json(suite_manifest_path, suite_manifest)
            continue
        run_manifest_path = run_dir / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if (
            run_manifest.get("seed") != seed
            or run_manifest.get("resume") is not None
            or run_manifest.get("profile_name") != args.profile
            or run_manifest.get("max_iterations") != args.iterations
        ):
            record.update({"status": "run_manifest_rejected", "run_manifest": str(run_manifest_path)})
            _write_json(suite_manifest_path, suite_manifest)
            continue
        fingerprint_payload, fingerprint_hash = _training_fingerprint(run_manifest)
        if reference_fingerprint is None:
            reference_fingerprint = (fingerprint_payload, fingerprint_hash)
            suite_manifest["training_fingerprint_hash"] = fingerprint_hash
            _write_json(suite_dir / "training-fingerprint.json", fingerprint_payload)
            if args.mode == "formal":
                evaluation_context = {
                    "schema_version": "TrainingEvaluationContextV1",
                    "suite_mode": "formal",
                    "suite_id": suite_id,
                    "run_count": args.runs,
                    "iterations_per_run": args.iterations,
                    "training_fingerprint_hash": fingerprint_hash,
                    "suite_source_fingerprint_hash": suite_source_hash,
                }
                evaluation_context["context_hash"] = _stable_hash(evaluation_context)
                evaluation_context_path = suite_dir / "evaluation-suite-context.json"
                _write_json(evaluation_context_path, evaluation_context)
        elif fingerprint_hash != reference_fingerprint[1] or fingerprint_payload != reference_fingerprint[0]:
            record.update(
                {
                    "status": "training_fingerprint_mismatch",
                    "training_fingerprint_hash": fingerprint_hash,
                    "run_manifest": str(run_manifest_path),
                }
            )
            _write_json(suite_manifest_path, suite_manifest)
            continue
        monitor_report = _monitor_report(suite_dir / "monitor" / f"run-{index:02d}")
        suite_warnings.update(monitor_report["warnings"])
        suite_hard_anomalies.update(monitor_report["hard_anomalies"])
        checkpoint = Path(training_summary["final_checkpoint"])
        record.update(
            {
                "status": "trained_monitor_anomaly" if monitor_report["hard_anomalies"] else "trained",
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "run_manifest": str(run_manifest_path),
                "training_fingerprint_hash": fingerprint_hash,
                "monitor": monitor_report,
            }
        )
        _write_json(suite_manifest_path, suite_manifest)

    for record in suite_manifest["runs"]:
        checkpoint_value = record.get("checkpoint")
        if checkpoint_value is None:
            continue
        index = int(record["index"])
        checkpoint = Path(checkpoint_value)
        training_summary = json.loads(
            (Path(record["run_dir"]) / "training_summary.json").read_text(encoding="utf-8")
        )
        export_dir = suite_dir / "exports" / f"run-{index:02d}"
        export_command = [
            sys.executable,
            "scripts/export_ppo_actor.py",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(export_dir),
        ]
        try:
            _assert_suite_source_fingerprint(suite_source_hash)
            _run_checked(export_command, suite_dir / f"run-{index:02d}-export.log")
            _assert_suite_source_fingerprint(suite_source_hash)
        except Exception as error:
            record.update({"status": "export_failed", "export_error": f"{type(error).__name__}: {error}"})
            _write_json(suite_manifest_path, suite_manifest)
            continue

        evaluation_dir = suite_dir / "evaluation" / f"run-{index:02d}"
        evaluation_command = [
            "uv",
            "run",
            "--project",
            str(SIM2SIM_PROJECT),
            "python",
            "scripts/evaluate_mujoco.py",
            "--policy",
            str(export_dir / "actor.ts"),
            "--manifest",
            str(export_dir / "policy_manifest.json"),
            "--output",
            str(evaluation_dir),
        ]
        if args.mode == "smoke":
            evaluation_command.append("--smoke")
        else:
            assert evaluation_context_path is not None
            evaluation_command.extend(
                (
                    "--suite-context",
                    str(evaluation_context_path),
                    "--run-index",
                    str(index),
                    "--completed-iterations",
                    str(training_summary["completed_iterations"]),
                )
            )
        try:
            _assert_suite_source_fingerprint(suite_source_hash)
            _run_checked(evaluation_command, suite_dir / f"run-{index:02d}-evaluation.log")
            _assert_suite_source_fingerprint(suite_source_hash)
        except Exception as error:
            record.update(
                {
                    "status": "evaluation_failed",
                    "export_dir": str(export_dir),
                    "evaluation_error": f"{type(error).__name__}: {error}",
                }
            )
            _write_json(suite_manifest_path, suite_manifest)
            continue
        evaluation_summary = evaluation_dir / "summary.json"
        evaluation_summaries.append(evaluation_summary)
        record.update(
            {
                "status": "completed_monitor_anomaly" if record["monitor"]["hard_anomalies"] else "completed",
                "export_dir": str(export_dir),
                "evaluation_summary": str(evaluation_summary),
            }
        )
        _write_json(suite_manifest_path, suite_manifest)

    ranking_path = suite_dir / "training-suite-summary.json"
    monitor_gate = {
        "status": "hard_anomaly" if suite_hard_anomalies else "warning" if suite_warnings else "ok",
        "warnings": sorted(suite_warnings),
        "hard_anomalies": sorted(suite_hard_anomalies),
    }
    completed_evaluations = len(evaluation_summaries)
    incomplete_records = [
        record
        for record in suite_manifest["runs"]
        if record.get("status") not in {"completed", "completed_monitor_anomaly"}
    ]
    if args.mode == "formal" and completed_evaluations == args.runs:
        _assert_suite_source_fingerprint(suite_source_hash)
        ranking_command = [
            "uv",
            "run",
            "--project",
            str(SIM2SIM_PROJECT),
            "python",
            "scripts/rank_mujoco_runs.py",
        ]
        for path in evaluation_summaries:
            ranking_command.extend(("--evaluation", str(path)))
        ranking_command.extend(("--output", str(ranking_path)))
        _run_checked(ranking_command, suite_dir / "ranking.log")
        _assert_suite_source_fingerprint(suite_source_hash)
        ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
        ranking["training_monitor_gate"] = monitor_gate
    elif args.mode == "formal":
        ranking = {
            "schema_version": "TrainingSuiteIncompleteV1",
            "suite_mode": "formal",
            "evaluation_summaries": [str(path.resolve()) for path in evaluation_summaries],
            "expected_evaluation_count": args.runs,
            "completed_evaluation_count": completed_evaluations,
            "training_monitor_gate": monitor_gate,
            "phase1_qualified": False,
            "qualification_status": "incomplete_training_or_evaluation",
            "failed_runs": [
                {"index": record["index"], "status": record.get("status")}
                for record in incomplete_records
            ],
            "selected": None,
        }
    else:
        ranking = {
            "schema_version": "TrainingSuiteSmokeV1",
            "suite_mode": "smoke",
            "evaluation_summaries": [str(path.resolve()) for path in evaluation_summaries],
            "training_monitor_gate": monitor_gate,
            "has_full_survival_candidate": False,
            "phase1_qualified": False,
            "qualification_status": "smoke_not_eligible",
            "selected": None,
            "best_candidate": None,
            "diagnostic_candidate": None,
        }
    _write_json(ranking_path, ranking)
    suite_manifest["status"] = (
        "failed"
        if incomplete_records
        else "review_required"
        if suite_hard_anomalies
        else "completed_with_warnings"
        if suite_warnings
        else "completed"
    )
    suite_manifest["training_monitor_gate"] = monitor_gate
    suite_manifest["ranking_summary"] = str(ranking_path)
    _write_json(suite_manifest_path, suite_manifest)
    latest_pointer = "latest_training_suite.txt" if args.mode == "formal" else "latest_training_smoke.txt"
    suite_root.mkdir(parents=True, exist_ok=True)
    (suite_root / latest_pointer).write_text(
        str(suite_dir), encoding="utf-8"
    )
    print(json.dumps({"suite_dir": str(suite_dir), "ranking_summary": str(ranking_path)}, indent=2), flush=True)
    if incomplete_records:
        raise RuntimeError(
            f"Training suite completed with {len(incomplete_records)} incomplete run(s); see {suite_manifest_path}"
        )


if __name__ == "__main__":
    main()
