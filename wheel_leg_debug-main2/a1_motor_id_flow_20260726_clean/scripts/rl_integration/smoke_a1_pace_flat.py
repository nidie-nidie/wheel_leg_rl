"""Run a reproducible smoke training job for the fitted A1 PACE flat task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


SMOKE_TASK = "Gogo-A1-PACE-Flat-v0"
SMOKE_NUM_ENVS = 64
SMOKE_MAX_ITERATIONS = 2
SMOKE_SAVE_INTERVAL = 1
SMOKE_SEED = 42
EXPERIMENT_NAME = "a1_pace_flat_smoke"
LOG_ROOT_PATTERN = re.compile(
    r"^\[INFO\] Logging experiment in directory: (?P<root>.+)$", re.MULTILINE
)
RUN_DIRECTORY_PATTERN = re.compile(
    r"^Exact experiment name requested from command line: (?P<name>[^\r\n]+)$",
    re.MULTILINE,
)


def build_train_command(
    *, isaaclab_sh: Path, training_script: Path, num_envs: int
) -> list[str]:
    return [
        str(isaaclab_sh),
        "-p",
        str(training_script),
        "--task",
        SMOKE_TASK,
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        str(SMOKE_MAX_ITERATIONS),
        "--seed",
        str(SMOKE_SEED),
        "--headless",
    ]


def parse_log_directory(stdout: str, *, run_name: str) -> Path:
    roots = LOG_ROOT_PATTERN.findall(stdout)
    run_directories = RUN_DIRECTORY_PATTERN.findall(stdout)
    if len(roots) != 1 or len(run_directories) != 1:
        raise ValueError("training output did not print one exact log directory")
    root = Path(roots[0])
    if not run_directories[0] or Path(run_directories[0]).name != run_directories[0]:
        raise ValueError("training output log directory is not a safe relative path")
    return root / f"{run_directories[0]}_{run_name}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fit_evidence(fit_manifest: Path, pace_repository_root: Path) -> dict[str, str]:
    fit_manifest = fit_manifest.resolve()
    pace_repository_root = pace_repository_root.resolve()
    manifest = json.loads(fit_manifest.read_text(encoding="ascii"))
    if manifest.get("schema_version") != "a1_pace_fit_manifest/v1":
        raise ValueError("fit manifest schema is not a1_pace_fit_manifest/v1")
    if manifest.get("status") != "PASS":
        raise ValueError("fit manifest status is not PASS")
    if manifest.get("task") != "Isaac-Pace-A1-v0":
        raise ValueError("fit manifest task is not Isaac-Pace-A1-v0")
    run_basename = manifest.get("run_basename")
    mean_basename = manifest.get("mean_basename")
    mean_sha256 = manifest.get("mean_sha256")
    if (
        not isinstance(run_basename, str)
        or Path(run_basename).name != run_basename
        or not isinstance(mean_basename, str)
        or Path(mean_basename).name != mean_basename
        or not isinstance(mean_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", mean_sha256)
    ):
        raise ValueError("fit manifest run/mean binding is invalid")
    mean_path = pace_repository_root / "logs" / "pace" / "a1" / run_basename / mean_basename
    if not mean_path.is_file() or _sha256(mean_path) != mean_sha256:
        raise ValueError("fit manifest mean SHA does not match the selected file")
    return {
        "fit_manifest_path": str(fit_manifest),
        "fit_manifest_sha256": _sha256(fit_manifest),
        "mean_sha256": mean_sha256,
    }


def validate_smoke_artifacts(
    log_directory: Path, *, fit_manifest_sha256: str, mean_sha256: str
) -> None:
    checkpoints = [
        path
        for path in log_directory.glob("model_*.pt")
        if path.is_file() and path.stat().st_size > 0
    ]
    if not checkpoints:
        raise ValueError("smoke training did not produce a non-empty checkpoint")
    env_config = log_directory / "params" / "env.yaml"
    agent_config = log_directory / "params" / "agent.yaml"
    if not env_config.is_file() or not agent_config.is_file():
        raise ValueError("smoke training configuration dump is incomplete")
    environment_text = env_config.read_text(encoding="utf-8")
    required_values = (
        fit_manifest_sha256,
        mean_sha256,
        "0.002",
        "10",
        "2.0",
    )
    if any(value not in environment_text for value in required_values):
        raise ValueError("smoke env configuration does not bind fitted PACE evidence")


def run_smoke(
    *, fit_manifest: Path, pace_repository_root: Path, isaaclab_root: Path, gogo_root: Path
) -> Path:
    fit_manifest = fit_manifest.resolve()
    pace_repository_root = pace_repository_root.resolve()
    isaaclab_root = isaaclab_root.resolve()
    gogo_root = gogo_root.resolve()
    evidence = load_fit_evidence(fit_manifest, pace_repository_root)
    isaaclab_sh = isaaclab_root / "isaaclab.sh"
    training_script = gogo_root / "training" / "isaaclab_train.py"
    if not isaaclab_sh.is_file() or not training_script.is_file():
        raise ValueError("IsaacLab launcher or Gogo training entry point is unavailable")

    run_name = f"a1_pace_flat_smoke_{evidence['fit_manifest_sha256'][:12]}"
    environment = os.environ.copy()
    environment.update(
        {
            "A1_PACE_FIT_MANIFEST": evidence["fit_manifest_path"],
            "A1_PACE_REPOSITORY_ROOT": str(pace_repository_root),
            "GOGO_TASK": SMOKE_TASK,
            "GOGO_NUM_ENVS": str(SMOKE_NUM_ENVS),
            "GOGO_MAX_ITERATIONS": str(SMOKE_MAX_ITERATIONS),
            "GOGO_SAVE_INTERVAL": str(SMOKE_SAVE_INTERVAL),
            "GOGO_SEED": str(SMOKE_SEED),
            "GOGO_EXPERIMENT_NAME": EXPERIMENT_NAME,
            "GOGO_RUN_NAME": run_name,
            "GOGO_A1_LATENCY_MODEL": "0",
        }
    )
    result = subprocess.run(
        build_train_command(
            isaaclab_sh=isaaclab_sh,
            training_script=training_script,
            num_envs=SMOKE_NUM_ENVS,
        ),
        cwd=gogo_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise ValueError(f"A1 PACE smoke training exited with code {result.returncode}")
    relative_log_directory = parse_log_directory(result.stdout, run_name=run_name)
    log_directory = (gogo_root / relative_log_directory).resolve()
    try:
        log_directory.relative_to(gogo_root)
    except ValueError as exc:
        raise ValueError("training log directory escapes the Gogo repository") from exc
    validate_smoke_artifacts(
        log_directory,
        fit_manifest_sha256=evidence["fit_manifest_sha256"],
        mean_sha256=evidence["mean_sha256"],
    )
    return log_directory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the two-iteration fitted A1 PACE flat training smoke"
    )
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--pace-repository-root", type=Path, required=True)
    parser.add_argument(
        "--isaaclab-root", type=Path, default=Path("/home/changba01/IsaacLab")
    )
    parser.add_argument(
        "--gogo-root",
        type=Path,
        default=Path("/home/changba01/worktrees/gogo-learn-a1-pace"),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        log_directory = run_smoke(
            fit_manifest=args.fit_manifest,
            pace_repository_root=args.pace_repository_root,
            isaaclab_root=args.isaaclab_root,
            gogo_root=args.gogo_root,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"[A1 PACE TRAIN SMOKE] FAIL: {exc}", file=sys.stderr)
        return 2
    print(f"[A1 PACE TRAIN SMOKE] PASS: {log_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
