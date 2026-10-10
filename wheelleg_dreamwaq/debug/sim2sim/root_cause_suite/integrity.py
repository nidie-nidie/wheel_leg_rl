from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from .contracts import (
    FROZEN_POLICIES,
    FROZEN_REPLAY_SOURCE,
    IdentityDriftError,
    PROJECT_ROOT,
    SUITE_ROOT,
    WORKSPACE_ROOT,
    canonical_json_bytes,
    sha256_file,
    stable_hash,
)


FORMAL_FILES: Mapping[str, tuple[Path, str]] = {
    "architecture": (
        WORKSPACE_ROOT / "docs/2026-10-03-wheelleg-dreamwaq-architecture.md",
        "5C7AF59F56750B50BA3CF14D731DF619489C0BBDBCCDA620B885B610087BB3AE",
    ),
    "mujoco_xml": (
        PROJECT_ROOT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml",
        "691C607271ED85EFFA388237E00A7B8F8F41CB66F6BA9C49EA64ABD186D803D1",
    ),
    "mujoco_manifest": (
        PROJECT_ROOT / "sim2sim/mujoco/model_manifest.json",
        "C3DB4FE2797F6AC3628F0A9CB5E0B166FFD0CE802776A335922C474A045F7AE0",
    ),
    "mujoco_runner": (
        PROJECT_ROOT / "sim2sim/mujoco/wheelleg_mujoco/runner.py",
        "B0C973A5335856E3991ECCF6014BA9B5CF65E638A8BD4701FC07EED1A446EDBB",
    ),
    "mujoco_control": (
        PROJECT_ROOT / "sim2sim/mujoco/wheelleg_mujoco/control.py",
        "83080BC57A0399B11610CE5E06B4F607ADA1C0FB995344E23E531658EBBEFD62",
    ),
    "mujoco_observation": (
        PROJECT_ROOT / "sim2sim/mujoco/wheelleg_mujoco/observation.py",
        "DD7F972D77FCCA25378BF403F61A67755C8AB4BA28738C97ABF2BBDD733970BD",
    ),
    "formal_env": (
        PROJECT_ROOT
        / "source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py",
        "5399B29B92B54851EB0A5FA880EA9386C82F202A5C337F19ABE4EDF300C2969C",
    ),
    "formal_env_cfg": (
        PROJECT_ROOT
        / "source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env_cfg.py",
        "AB7BAA693A73EFBB0F933A605EF02F94C0BDBCF95E468D3CF400E590F59101F5",
    ),
    "asset_manifest": (
        PROJECT_ROOT / "artifacts/phase1_v4/asset-bundle-v2-manifest.json",
        "CC4BF6006F103E9520E522BD8636BCDDAF69B2F121BFA73E65D984A9CFAA903B",
    ),
    "source_audit": (
        PROJECT_ROOT / "artifacts/phase1_v4/mujoco-usd-data.json",
        "411B3913D03F4523132177E42B416AFA736A026E7ADB567A81AD28ECB75D53C5",
    ),
    "usd_anchor_audit": (
        PROJECT_ROOT / "artifacts/phase1_v4/usd-anchor-audit.json",
        "6231C16A830401B46516880821D3E8C4225BF8D2963367619D0F7FBC9D69BB5F",
    ),
    "reset_cache": (
        FROZEN_REPLAY_SOURCE["reset_cache"].absolute_path(),
        FROZEN_REPLAY_SOURCE["reset_cache"].sha256,
    ),
    "evaluation_summary": (
        FROZEN_REPLAY_SOURCE["evaluation_summary"].absolute_path(),
        FROZEN_REPLAY_SOURCE["evaluation_summary"].sha256,
    ),
}


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(resolved),
    }


def formal_snapshot(*, validate_expected: bool = True) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for name, (path, expected) in FORMAL_FILES.items():
        record = _file_record(path)
        record["expected_sha256"] = expected
        if validate_expected and record["sha256"] != expected:
            raise IdentityDriftError(
                f"Frozen formal file drifted: {name}: {record['sha256']} != {expected}"
            )
        rows[name] = record
    for policy_name, records in FROZEN_POLICIES.items():
        for kind, frozen in records.items():
            name = f"policy:{policy_name}:{kind}"
            record = _file_record(frozen.absolute_path())
            record["expected_sha256"] = frozen.sha256
            if validate_expected and record["sha256"] != frozen.sha256:
                raise IdentityDriftError(f"Frozen policy drifted: {policy_name}/{kind}")
            rows[name] = record
    return rows


def _selected_files(root: Path, names: Iterable[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return rows
    wanted = set(names)
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().lower()):
        if path.is_file() and path.name in wanted:
            rows[path.relative_to(root).as_posix()] = _file_record(path)
    return rows


def interpreter_snapshot(interpreter: Path, *, config_root: Path) -> dict[str, Any]:
    executable = interpreter.resolve(strict=True)
    venv_root = executable.parents[1]
    files: dict[str, dict[str, Any]] = {
        "python.exe": _file_record(executable),
        "pyvenv.cfg": _file_record(venv_root / "pyvenv.cfg"),
    }
    for name in ("pyproject.toml", "uv.lock"):
        path = config_root / name
        if path.is_file():
            files[name] = _file_record(path)
    metadata_root = venv_root / "Lib/site-packages"
    metadata = _selected_files(metadata_root, ("METADATA", "RECORD", "direct_url.json"))
    return {
        "executable": str(executable),
        "version": sys.version if executable == Path(sys.executable).resolve() else None,
        "files": files,
        "distribution_metadata": metadata,
        "identity_hash": stable_hash({"files": files, "distribution_metadata": metadata}),
    }


def suite_source_snapshot() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(SUITE_ROOT.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not path.is_file() or "runs" in path.relative_to(SUITE_ROOT).parts:
            continue
        if "__pycache__" in path.parts or path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        if path.suffix.lower() not in {".py", ".ps1", ".json"}:
            continue
        rows[path.relative_to(SUITE_ROOT).as_posix()] = _file_record(path)
    for path in (
        PROJECT_ROOT / "debug/sim2sim/dreamwaq_debug_contract.py",
        PROJECT_ROOT / "debug/sim2sim/isaac_debug_env.py",
    ):
        rows[f"external:{path.relative_to(PROJECT_ROOT).as_posix()}"] = _file_record(path)
    return rows


def catalog_snapshot() -> dict[str, Any]:
    from .scenario_catalog import catalog

    scenarios = json.loads(
        canonical_json_bytes([asdict(item) for item in catalog()]).decode("ascii")
    )
    return {
        "scenarios": scenarios,
        "identity_hash": stable_hash(scenarios),
    }


def controlled_project_snapshot() -> dict[str, dict[str, Any]]:
    candidates: set[Path] = set()
    search_roots = (
        PROJECT_ROOT / "debug",
        PROJECT_ROOT / "source",
        PROJECT_ROOT / "sim2sim",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "apps",
        PROJECT_ROOT / "dependencies/IsaacLab-v2.3.2/source",
    )
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for current, directories, files in os.walk(search_root):
            current_path = Path(current)
            if current_path == SUITE_ROOT or SUITE_ROOT in current_path.parents:
                directories[:] = []
                continue
            candidates.update(current_path / name for name in files)
            directories[:] = [
                name
                for name in directories
                if name not in {".venv", "artifacts", "runs"}
            ]
    candidates.update(path for path in PROJECT_ROOT.iterdir() if path.is_file())
    root_pytest = PROJECT_ROOT / ".pytest_cache"
    if root_pytest.is_dir():
        candidates.update(path for path in root_pytest.rglob("*") if path.is_file())
    kit_root = PROJECT_ROOT / ".venv/Lib/site-packages/isaacsim/kit"
    for name in ("logs", "data", "cache"):
        directory = kit_root / name
        if directory.is_dir():
            candidates.update(path for path in directory.rglob("*") if path.is_file())
    return {
        path.absolute().relative_to(PROJECT_ROOT.absolute()).as_posix(): _file_record(path)
        for path in sorted(candidates, key=lambda item: item.as_posix().lower())
    }


def build_run_identity(
    *,
    project_python: Path,
    mujoco_python: Path,
    parent_argv: list[str],
    environment: Mapping[str, str | None],
) -> dict[str, Any]:
    payload = {
        "formal_before": formal_snapshot(),
        "suite_sources": suite_source_snapshot(),
        "catalog": catalog_snapshot(),
        "project_interpreter": interpreter_snapshot(
            project_python, config_root=PROJECT_ROOT
        ),
        "mujoco_interpreter": interpreter_snapshot(
            mujoco_python, config_root=PROJECT_ROOT / "sim2sim/mujoco"
        ),
        "controlled_project_before": controlled_project_snapshot(),
        "environment": dict(environment),
        "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
        "initial_parent_argv": list(parent_argv),
        "initial_parent_argv_hash": stable_hash(list(parent_argv)),
    }
    payload["identity_hash"] = stable_hash(payload)
    return payload


def verify_run_identity(
    payload: Mapping[str, Any], *, verify_environment: bool = True
) -> None:
    stored = dict(payload)
    expected_hash = stored.pop("identity_hash", None)
    if expected_hash != stable_hash(stored):
        raise IdentityDriftError("Run identity payload hash is invalid")
    if payload.get("initial_parent_argv_hash") != stable_hash(
        payload.get("initial_parent_argv", [])
    ):
        raise IdentityDriftError("Initial parent argv identity is invalid")
    current_sources = suite_source_snapshot()
    if current_sources != payload.get("suite_sources"):
        raise IdentityDriftError("Suite source identity changed since the run began")
    if catalog_snapshot() != payload.get("catalog"):
        raise IdentityDriftError("Scenario catalog identity changed since the run began")
    current_formal = formal_snapshot()
    if current_formal != payload.get("formal_before"):
        raise IdentityDriftError("Frozen formal inputs changed since the run began")
    current_controlled = controlled_project_snapshot()
    if current_controlled != payload.get("controlled_project_before"):
        raise IdentityDriftError(
            "Project-controlled file or Isaac Kit tree changed outside the suite"
        )
    current_project = interpreter_snapshot(
        PROJECT_ROOT / ".venv/Scripts/python.exe", config_root=PROJECT_ROOT
    )
    if current_project != payload.get("project_interpreter"):
        raise IdentityDriftError("Project interpreter identity changed since the run began")
    current_mujoco = interpreter_snapshot(
        PROJECT_ROOT / "sim2sim/mujoco/.venv/Scripts/python.exe",
        config_root=PROJECT_ROOT / "sim2sim/mujoco",
    )
    if current_mujoco != payload.get("mujoco_interpreter"):
        raise IdentityDriftError("MuJoCo interpreter identity changed since the run began")
    if verify_environment and environment_contract() != payload.get("environment"):
        raise IdentityDriftError("Bootstrap environment changed since the run began")


def environment_contract() -> dict[str, str | None]:
    return {
        key: os.environ.get(key)
        for key in (
            "PYTHONNOUSERSITE",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONPYCACHEPREFIX",
            "WHEELLEG_ROOT_CAUSE_BOOTSTRAP",
            "WHEELLEG_ROOT_CAUSE_RUN_ID",
            "WHEELLEG_ROOT_CAUSE_RUN_ROOT",
        )
    }
