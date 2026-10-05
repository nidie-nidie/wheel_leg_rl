"""Load one hash-bound A1 PACE actuator from a successful fit manifest."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from pace_sim2real.tasks.manager_based.pace.a1_mean import (
    A1_PACE_JOINT_ORDER,
    load_a1_mean,
)
from pace_sim2real.tasks.manager_based.pace.a1_replay import (
    make_a1_pace_actuator_cfg,
)
from pace_sim2real.utils import PaceDCMotorCfg


FIT_MANIFEST_SCHEMA = "a1_pace_fit_manifest/v1"
FIT_TASK = "Isaac-Pace-A1-v0"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RUN_BASENAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MEAN_BASENAME_PATTERN = re.compile(r"^mean_[0-9]+\.pt$")


@dataclass(frozen=True)
class A1PaceActuatorEvidence:
    fit_manifest_path: Path
    fit_manifest_sha256: str
    mean_path: Path
    mean_sha256: str
    joint_order: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate fit manifest key: {key}")
        result[key] = value
    return result


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"fit manifest must be a regular file: {path}")
    try:
        manifest = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite fit manifest constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read strict fit manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("fit manifest root must be an object")
    return manifest


def _required_string(manifest: Mapping[str, Any], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"fit manifest requires non-empty {key}")
    return value


def _safe_basename(value: str, *, label: str, pattern: re.Pattern[str]) -> str:
    if not pattern.fullmatch(value) or Path(value).name != value:
        raise ValueError(f"fit manifest {label} is not a safe basename")
    return value


def _validate_manifest(manifest: Mapping[str, Any]) -> tuple[str, str, str]:
    if manifest.get("schema_version") != FIT_MANIFEST_SCHEMA:
        raise ValueError("fit manifest schema version is not supported")
    if manifest.get("status") != "PASS":
        raise ValueError("fit manifest status must be PASS")
    if manifest.get("task") != FIT_TASK:
        raise ValueError("fit manifest task is not the A1 PACE task")

    run_basename = _safe_basename(
        _required_string(manifest, "run_basename"),
        label="run_basename",
        pattern=RUN_BASENAME_PATTERN,
    )
    mean_basename = _safe_basename(
        _required_string(manifest, "mean_basename"),
        label="mean_basename",
        pattern=MEAN_BASENAME_PATTERN,
    )
    mean_sha256 = _required_string(manifest, "mean_sha256")
    if not SHA256_PATTERN.fullmatch(mean_sha256):
        raise ValueError("fit manifest mean_sha256 is invalid")

    environment_contract = manifest.get("environment_contract")
    if not isinstance(environment_contract, Mapping):
        raise ValueError("fit manifest environment contract is missing")
    if tuple(environment_contract.get("joint_order", ())) != A1_PACE_JOINT_ORDER:
        raise ValueError("fit manifest A1 joint order does not match")

    revisions = manifest.get("repository_revisions")
    if not isinstance(revisions, Mapping):
        raise ValueError("fit manifest repository revisions are missing")
    for name in ("pace", "a1_base", "gogo_learn"):
        revision = revisions.get(name)
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40,64}", revision):
            raise ValueError(f"fit manifest repository revision is invalid: {name}")
    return run_basename, mean_basename, mean_sha256


def load_fitted_a1_actuator(
    fit_manifest_path: Path, *, pace_repository_root: Path
) -> tuple[PaceDCMotorCfg, A1PaceActuatorEvidence]:
    """Load the exact successful mean named and hashed by a fit manifest."""

    manifest_path = Path(fit_manifest_path).resolve()
    pace_root = Path(pace_repository_root).resolve()
    manifest = _load_manifest(manifest_path)
    run_basename, mean_basename, expected_mean_sha256 = _validate_manifest(manifest)

    run_directory = pace_root / "logs" / "pace" / "a1" / run_basename
    mean_path = run_directory / mean_basename
    if not mean_path.is_file() or mean_path.is_symlink():
        raise ValueError(f"fit mean must be a regular file: {mean_path}")
    resolved_run_directory = run_directory.resolve()
    resolved_mean_path = mean_path.resolve()
    try:
        resolved_mean_path.relative_to(resolved_run_directory)
    except ValueError as exc:
        raise ValueError("fit mean escapes its manifest-selected run directory") from exc
    observed_mean_sha256 = _sha256(resolved_mean_path)
    if observed_mean_sha256 != expected_mean_sha256:
        raise ValueError("fit mean SHA does not match the manifest")

    mean = load_a1_mean(resolved_mean_path)
    actuator = make_a1_pace_actuator_cfg(mean)
    evidence = A1PaceActuatorEvidence(
        fit_manifest_path=manifest_path,
        fit_manifest_sha256=_sha256(manifest_path),
        mean_path=resolved_mean_path,
        mean_sha256=observed_mean_sha256,
        joint_order=A1_PACE_JOINT_ORDER,
    )
    return actuator, evidence
