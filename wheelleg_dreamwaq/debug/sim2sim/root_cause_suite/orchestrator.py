from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping

from .contracts import (
    DESIGN_SHA256,
    DESIGN_VERSION,
    EvidenceIntegrityError,
    IdentityDriftError,
    PROJECT_ROOT,
    StageExecutionError,
    SUITE_ROOT,
    canonical_json_bytes,
    require_path_within,
    sha256_file,
    stable_hash,
    validate_run_id,
)
from .integrity import (
    build_run_identity,
    environment_contract,
    suite_source_snapshot,
    verify_run_identity,
)
from .scenario_catalog import (
    FACTOR_PATH_ALLOWLIST_PATH,
    factor_path_allowlist_artifact,
)


WORKER_BOOTSTRAP = "debug.sim2sim.root_cause_suite.worker_bootstrap"
ADAPTER_TERMINAL_STAGE_IDS = (
    "G00_integrity",
    "G01_repeatability",
    "G02_adapter",
    "G03_instrumentation",
)
FRESH_BOOTSTRAP_CACHE_DIRS = ("pycache", "pytest-cache", "pytest-tmp")
RUN_MANIFEST_NAME = "run_manifest.json"
INITIALIZATION_CLAIM_SCHEMA = "RootCauseRunInitializationClaimV1"


@dataclass(frozen=True)
class WorkerInvocation:
    interpreter: Path
    module: str
    arguments: tuple[str, ...] = ()

    def command(self, output: Path, *, guard_root: Path) -> list[str]:
        if not self.module or self.module.startswith("-"):
            raise ValueError("Worker module is invalid")
        return [
            str(self.interpreter),
            "-B",
            "-m",
            WORKER_BOOTSTRAP,
            "--guard-root",
            str(guard_root),
            "--worker-module",
            self.module,
            "--",
            *self.arguments,
            "--output",
            str(output),
        ]


@dataclass(frozen=True)
class StageDefinition:
    stage_id: str
    dependencies: tuple[str, ...]
    worker: WorkerInvocation


@dataclass(frozen=True)
class StageOutcome:
    stage_id: str
    status: str
    reason: str | None
    attempt_dir: Path | None


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _is_link_or_reparse(path: Path, *, file_stat: os.stat_result | None = None) -> bool:
    value = path.lstat() if file_stat is None else file_stat
    attributes = int(getattr(value, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(value.st_mode) or bool(attributes & reparse_flag)


def _plain_directory(
    path: Path,
    *,
    expected: Path,
    label: str,
    error_type: type[Exception],
) -> Path:
    try:
        file_stat = path.lstat()
    except FileNotFoundError as error:
        raise error_type(f"Missing {label}: {path}") from error
    if _is_link_or_reparse(path, file_stat=file_stat):
        raise error_type(f"{label} is a link or reparse point: {path}")
    if not stat.S_ISDIR(file_stat.st_mode):
        raise error_type(f"{label} is not a directory: {path}")
    resolved = path.resolve(strict=True)
    if resolved != expected:
        raise error_type(f"{label} resolves unexpectedly: {resolved} != {expected}")
    return resolved


def _plain_file(
    path: Path,
    *,
    expected: Path,
    label: str,
    error_type: type[Exception],
) -> Path:
    try:
        file_stat = path.lstat()
    except FileNotFoundError as error:
        raise error_type(f"Missing {label}: {path}") from error
    if _is_link_or_reparse(path, file_stat=file_stat):
        raise error_type(f"{label} is a link or reparse point: {path}")
    if not stat.S_ISREG(file_stat.st_mode):
        raise error_type(f"{label} is not a regular file: {path}")
    resolved = path.resolve(strict=True)
    if resolved != expected:
        raise error_type(f"{label} resolves unexpectedly: {resolved} != {expected}")
    return resolved


def _validate_run_root_entry(
    run_root: Path, *, error_type: type[Exception]
) -> Path:
    runs_root = run_root.parent.resolve(strict=True)
    return _plain_directory(
        run_root,
        expected=runs_root / run_root.name,
        label="run root",
        error_type=error_type,
    )


def _validate_run_cache_layout(
    run_root: Path,
    *,
    error_type: type[Exception],
    expected_top_level: set[str] | None,
    require_empty_bootstrap_caches: bool,
    require_manifest: bool,
) -> None:
    resolved_root = _validate_run_root_entry(run_root, error_type=error_type)
    top_level = {entry.name: entry for entry in run_root.iterdir()}
    if expected_top_level is not None and set(top_level) != expected_top_level:
        raise error_type(
            f"Unexpected run-root entries: {sorted(top_level)} != "
            f"{sorted(expected_top_level)}"
        )
    runtime_cache = run_root / "runtime_cache"
    resolved_cache = _plain_directory(
        runtime_cache,
        expected=resolved_root / "runtime_cache",
        label="runtime cache",
        error_type=error_type,
    )
    cache_entries = {entry.name: entry for entry in runtime_cache.iterdir()}
    if require_empty_bootstrap_caches and set(cache_entries) != set(
        FRESH_BOOTSTRAP_CACHE_DIRS
    ):
        raise error_type(
            f"Unexpected bootstrap cache entries: {sorted(cache_entries)}"
        )
    for entry in cache_entries.values():
        file_stat = entry.lstat()
        if _is_link_or_reparse(entry, file_stat=file_stat):
            raise error_type(f"Runtime-cache entry is a link or reparse point: {entry}")
    for name in FRESH_BOOTSTRAP_CACHE_DIRS:
        child = runtime_cache / name
        _plain_directory(
            child,
            expected=resolved_cache / name,
            label=f"runtime cache child {name}",
            error_type=error_type,
        )
        if require_empty_bootstrap_caches and any(child.iterdir()):
            raise error_type(f"Bootstrap cache is not empty: {child}")
    manifest_path = run_root / RUN_MANIFEST_NAME
    if require_manifest:
        _plain_file(
            manifest_path,
            expected=resolved_root / RUN_MANIFEST_NAME,
            label="run manifest",
            error_type=error_type,
        )
    elif _path_entry_exists(manifest_path):
        _plain_file(
            manifest_path,
            expected=resolved_root / RUN_MANIFEST_NAME,
            label="run manifest",
            error_type=error_type,
        )


def _validate_fresh_bootstrap_skeleton(run_root: Path) -> None:
    _validate_run_cache_layout(
        run_root,
        error_type=FileExistsError,
        expected_top_level={"runtime_cache"},
        require_empty_bootstrap_caches=True,
        require_manifest=False,
    )


def _validate_fresh_claimed_layout(run_root: Path) -> None:
    _validate_run_cache_layout(
        run_root,
        error_type=FileExistsError,
        expected_top_level={"runtime_cache", RUN_MANIFEST_NAME},
        require_empty_bootstrap_caches=True,
        require_manifest=True,
    )


def _validate_existing_run_layout(
    run_root: Path, *, require_manifest: bool
) -> None:
    _validate_run_cache_layout(
        run_root,
        error_type=EvidenceIntegrityError,
        expected_top_level=None,
        require_empty_bootstrap_caches=False,
        require_manifest=require_manifest,
    )


def _ensure_runtime_cache_tree(run_root: Path) -> None:
    resolved_root = _validate_run_root_entry(
        run_root, error_type=EvidenceIntegrityError
    )
    runtime_cache = run_root / "runtime_cache"
    if not _path_entry_exists(runtime_cache):
        try:
            runtime_cache.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            pass
    resolved_cache = _plain_directory(
        runtime_cache,
        expected=resolved_root / "runtime_cache",
        label="runtime cache",
        error_type=EvidenceIntegrityError,
    )
    for name in FRESH_BOOTSTRAP_CACHE_DIRS:
        child = runtime_cache / name
        if not _path_entry_exists(child):
            try:
                child.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                pass
        _plain_directory(
            child,
            expected=resolved_cache / name,
            label=f"runtime cache child {name}",
            error_type=EvidenceIntegrityError,
        )


def _transitive_dependents(
    definitions: Iterable[StageDefinition], source_stage: str
) -> set[str]:
    items = tuple(definitions)
    affected = {source_stage}
    changed = True
    while changed:
        changed = False
        for definition in items:
            if definition.stage_id in affected:
                continue
            if any(dependency in affected for dependency in definition.dependencies):
                affected.add(definition.stage_id)
                changed = True
    affected.discard(source_stage)
    return affected


def _replace_g01_selection_and_seal(
    manifest: dict[str, object],
    definitions: Iterable[StageDefinition],
    *,
    selection: Mapping[str, str],
    seal: Mapping[str, Any],
) -> list[str]:
    selections = manifest.get("selected_attempts")
    if not isinstance(selections, dict):
        raise EvidenceIntegrityError("Run selected_attempts is invalid")
    invalidated = sorted(
        stage
        for stage in _transitive_dependents(definitions, "G01_repeatability")
        if stage in selections
    )
    for stage in invalidated:
        selections.pop(stage, None)
    selections["G01_repeatability"] = dict(selection)
    manifest["replay_source_seal"] = dict(seal)
    return invalidated


def _replace_terminal_g02_selection(
    manifest: dict[str, object],
    definitions: Iterable[StageDefinition],
    *,
    selection: Mapping[str, str],
) -> list[str]:
    selections = manifest.get("selected_attempts")
    if not isinstance(selections, dict):
        raise EvidenceIntegrityError("Run selected_attempts is invalid")
    invalidated = sorted(
        stage
        for stage in _transitive_dependents(definitions, "G02_adapter")
        if stage in selections
    )
    for stage in invalidated:
        selections.pop(stage, None)
    selections["G02_adapter"] = dict(selection)
    return invalidated


def _attempt_has_terminal_adapter_bug(attempt: Path | None) -> bool:
    if attempt is None:
        return False
    result_path = attempt / "worker" / "result.json"
    if not result_path.is_file():
        return False
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    gate = payload.get("adapter_gate")
    return bool(
        payload.get("stage") == "G02_adapter"
        and payload.get("passed") is True
        and payload.get("terminal_adapter_bug") is True
        and isinstance(gate, dict)
        and gate.get("evidence_valid") is True
        and gate.get("digital_chain_passed") is False
        and gate.get("terminal_adapter_bug") is True
    )


def _attempt_has_unusable_repeatability(attempt: Path | None) -> bool:
    if attempt is None:
        return False
    result_path = attempt / "worker" / "result.json"
    if not result_path.is_file():
        return False
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(
        payload.get("stage") == "G01_repeatability"
        and payload.get("passed") is True
        and payload.get("exact_executed_coverage") is True
        and payload.get("repeatability_usable") is False
    )


def _manifest_has_terminal_adapter_bug(
    run_root: Path, manifest: Mapping[str, object]
) -> bool:
    selections = manifest.get("selected_attempts")
    if not isinstance(selections, dict):
        return False
    record = selections.get("G02_adapter")
    if not isinstance(record, dict) or not isinstance(record.get("attempt"), str):
        return False
    attempt = require_path_within(
        run_root / "stages" / "G02_adapter" / record["attempt"],
        run_root,
        label="selected G02 attempt",
    )
    return _attempt_has_terminal_adapter_bug(attempt)


def _manifest_has_unusable_repeatability(
    run_root: Path, manifest: Mapping[str, object]
) -> bool:
    selections = manifest.get("selected_attempts")
    if not isinstance(selections, dict):
        return False
    record = selections.get("G01_repeatability")
    if not isinstance(record, dict) or not isinstance(record.get("attempt"), str):
        return False
    attempt = require_path_within(
        run_root / "stages" / "G01_repeatability" / record["attempt"],
        run_root,
        label="selected G01 attempt",
    )
    return _attempt_has_unusable_repeatability(attempt)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _open_new_binary(path: Path) -> int:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOINHERIT", 0))
    return os.open(str(path), flags, 0o600)


def _write_new_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = _open_new_binary(path)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    try:
        _write_new_json(temporary, payload)
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _initialization_claim_payload(run_id: str) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": INITIALIZATION_CLAIM_SCHEMA,
        "run_id": run_id,
        "created_utc": _utc_now(),
        "process_id": os.getpid(),
        "nonce": secrets.token_hex(16),
    }
    return {
        **unsigned,
        "claim_identity_hash": stable_hash(unsigned),
    }


def _validate_initialization_claim(
    claim: object, *, expected_run_id: str
) -> dict[str, object]:
    if not isinstance(claim, dict):
        raise EvidenceIntegrityError("Run initialization claim is missing")
    unsigned = dict(claim)
    identity_hash = unsigned.pop("claim_identity_hash", None)
    if unsigned.get("schema_version") != INITIALIZATION_CLAIM_SCHEMA:
        raise EvidenceIntegrityError("Run initialization claim schema is invalid")
    if unsigned.get("run_id") != expected_run_id:
        raise EvidenceIntegrityError("Run initialization claim id is invalid")
    if not isinstance(unsigned.get("nonce"), str) or not unsigned["nonce"]:
        raise EvidenceIntegrityError("Run initialization claim nonce is invalid")
    if identity_hash != stable_hash(unsigned):
        raise EvidenceIntegrityError("Run initialization claim hash is invalid")
    return dict(claim)


def _create_initialization_claim(path: Path, *, run_id: str) -> dict[str, object]:
    claim = _initialization_claim_payload(run_id)
    _write_new_json(path, claim)
    return claim


def _read_initialization_claim(path: Path, *, run_id: str) -> dict[str, object]:
    try:
        claim = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise EvidenceIntegrityError("Run initialization claim is unreadable") from error
    return _validate_initialization_claim(claim, expected_run_id=run_id)


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and path.name != "stage_state.json"
    }


def _manifest_identity_payload(manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        key: manifest[key]
        for key in (
            "schema_version",
            "run_id",
            "scope",
            "design_version",
            "design_sha256",
            "stage_ids",
            "definitions",
            "definitions_hash",
            "factor_path_allowlist_path",
            "factor_path_allowlist_sha256",
            "run_identity",
            "initialization_claim",
            "created_utc",
            "launch_history",
            "selected_attempts",
            "replay_source_seal",
        )
    }


def _read_manifest_payload(run_root: Path) -> dict[str, object]:
    root = Path(run_root)
    _validate_existing_run_layout(root, require_manifest=True)
    path = root / RUN_MANIFEST_NAME
    if not path.is_file():
        raise EvidenceIntegrityError(f"Missing run manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise EvidenceIntegrityError(f"Run manifest is unreadable: {path}") from error
    if not isinstance(manifest, dict):
        raise EvidenceIntegrityError("Run manifest is not an object")
    try:
        expected_identity = stable_hash(_manifest_identity_payload(manifest))
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceIntegrityError("Run manifest identity payload is incomplete") from error
    if manifest.get("manifest_identity_hash") != expected_identity:
        raise EvidenceIntegrityError("Run manifest identity hash mismatch")
    if manifest.get("run_id") != root.name:
        raise EvidenceIntegrityError("Run manifest id does not match its directory")
    _validate_initialization_claim(
        manifest.get("initialization_claim"), expected_run_id=root.name
    )
    if manifest.get("design_sha256") != DESIGN_SHA256:
        raise IdentityDriftError("Run design hash mismatch")
    definitions = manifest.get("definitions")
    stage_ids = manifest.get("stage_ids")
    if not isinstance(definitions, list) or not isinstance(stage_ids, list):
        raise EvidenceIntegrityError("Run definitions or stage ids are invalid")
    if manifest.get("definitions_hash") != stable_hash(definitions):
        raise EvidenceIntegrityError("Run definition hash mismatch")
    factor_path_allowlist_artifact()
    if manifest.get("factor_path_allowlist_path") != FACTOR_PATH_ALLOWLIST_PATH.name:
        raise IdentityDriftError("Run factor-path allow-list path mismatch")
    if manifest.get("factor_path_allowlist_sha256") != sha256_file(
        FACTOR_PATH_ALLOWLIST_PATH
    ):
        raise IdentityDriftError("Run factor-path allow-list hash mismatch")
    definition_ids = [
        item.get("stage_id") for item in definitions if isinstance(item, dict)
    ]
    if definition_ids != stage_ids:
        raise EvidenceIntegrityError("Run definition order does not match stage ids")
    return manifest


def _atomic_manifest(path: Path, manifest: dict[str, object]) -> None:
    manifest["manifest_identity_hash"] = stable_hash(
        _manifest_identity_payload(manifest)
    )
    _atomic_json(path, manifest)


def _compute_replay_source_seal(
    run_root: Path, *, source_result_path: Path, actions_path: Path
) -> dict[str, Any]:
    root = require_path_within(
        Path(run_root).resolve(strict=True), SUITE_ROOT, label="replay run root"
    )
    result_path = require_path_within(
        Path(source_result_path).resolve(strict=True), root, label="replay source result"
    )
    action_path = require_path_within(
        Path(actions_path).resolve(strict=True), root, label="replay action sequence"
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "RootCauseIsaacReplaySourceV1":
        raise EvidenceIntegrityError("Replay source result schema is invalid")
    replay_identity = payload.get("replay_source_identity")
    replay_identity_hash = payload.get("replay_source_identity_hash")
    if not isinstance(replay_identity, dict) or stable_hash(replay_identity) != replay_identity_hash:
        raise EvidenceIntegrityError("Replay source identity hash is invalid")
    expected_action_path = require_path_within(
        result_path.parent / str(payload.get("action_sequence_file")),
        root,
        label="replay action result binding",
    )
    if action_path != expected_action_path:
        raise EvidenceIntegrityError("Replay source action path is not result-bound")
    action_file_hash = sha256_file(action_path)
    if action_file_hash != payload.get("clipped_action_file_sha256"):
        raise EvidenceIntegrityError("Replay source action file hash mismatch")
    import hashlib

    import numpy as np

    with np.load(action_path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "action_sequence",
            "environment_action_sequence",
        }:
            raise EvidenceIntegrityError("Replay source action archive fields are invalid")
        action_sequence = np.array(archive["action_sequence"], copy=True)
        environment_action_sequence = np.array(
            archive["environment_action_sequence"], copy=True
        )
    if action_sequence.dtype != np.float32 or action_sequence.ndim != 2:
        raise EvidenceIntegrityError("Replay source action tensor layout is invalid")
    action_tensor_hash = hashlib.sha256(
        np.ascontiguousarray(action_sequence).tobytes()
    ).hexdigest().upper()
    if action_tensor_hash != payload.get("clipped_action_sequence_sha256"):
        raise EvidenceIntegrityError("Replay source action tensor hash mismatch")
    trace_metadata_path = result_path.parent / "trace" / "trace.json"
    trace_data_path = result_path.parent / "trace" / "trace.npz"
    trace_metadata = json.loads(trace_metadata_path.read_text(encoding="utf-8"))
    trace_hash = sha256_file(trace_data_path)
    if trace_metadata.get("trace_sha256") != trace_hash:
        raise EvidenceIntegrityError("Replay source trace sidecar hash mismatch")
    if payload.get("source_trace_sha256") != trace_hash:
        raise EvidenceIntegrityError("Replay source result trace hash mismatch")
    if payload.get("trace_metadata_sha256") != sha256_file(trace_metadata_path):
        raise EvidenceIntegrityError("Replay source trace metadata hash mismatch")
    action_count = int(payload.get("action_count", -1))
    horizon = int(replay_identity.get("horizon", -2))
    if action_count != horizon or action_count <= 0:
        raise EvidenceIntegrityError("Replay source action count is not the frozen horizon")
    environment_count = int(replay_identity.get("environment_count", -1))
    if action_sequence.shape != (action_count, 6):
        raise EvidenceIntegrityError("Replay source row-zero action shape is invalid")
    if environment_action_sequence.shape != (action_count, environment_count, 6):
        raise EvidenceIntegrityError("Replay source environment action shape is invalid")
    if int(trace_metadata.get("row_count", -1)) != action_count:
        raise EvidenceIntegrityError("Replay source trace length differs from action count")
    first_action = payload.get("first_action")
    if not isinstance(first_action, list) or not np.array_equal(
        np.asarray(first_action, dtype=np.float32), action_sequence[0]
    ):
        raise EvidenceIntegrityError("Replay source first action is not tensor-bound")
    seal = {
        "schema_version": "RootCauseReplaySourceSealV1",
        "replay_source_identity_hash": replay_identity_hash,
        "source_result_sha256": sha256_file(result_path),
        "source_trace_sha256": trace_hash,
        "clipped_action_file_sha256": action_file_hash,
        "clipped_action_sequence_sha256": action_tensor_hash,
        "action_count": action_count,
    }
    seal["seal_identity_hash"] = stable_hash(seal)
    return seal


def _g01_replay_source_seal(run_root: Path, attempt: Path) -> dict[str, Any]:
    root = Path(run_root).resolve(strict=True)
    selected_attempt = require_path_within(
        Path(attempt).resolve(strict=True), root, label="G01 selected attempt"
    )
    worker = selected_attempt / "worker"
    result_path = worker / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("stage") != "G01_repeatability":
        raise EvidenceIntegrityError("Replay source is not owned by G01")
    source = payload.get("replay_source")
    if not isinstance(source, dict) or not isinstance(
        source.get("relative_path"), str
    ):
        raise EvidenceIntegrityError("G01 replay source index is missing")
    source_root = require_path_within(
        worker / Path(source["relative_path"]),
        worker,
        label="G01 replay source",
    ).resolve(strict=True)
    source_result = source_root / "result.json"
    if sha256_file(source_result) != source.get("result_sha256"):
        raise EvidenceIntegrityError("G01 replay source result hash mismatch")
    actions_file = source.get("actions_file")
    if not isinstance(actions_file, str) or Path(actions_file).name != actions_file:
        raise EvidenceIntegrityError("G01 replay source action path is invalid")
    actions = (source_root / actions_file).resolve(strict=True)
    if sha256_file(actions) != source.get("actions_sha256"):
        raise EvidenceIntegrityError("G01 replay source action hash mismatch")
    seal = _compute_replay_source_seal(
        root,
        source_result_path=source_result,
        actions_path=actions,
    )
    seal["g01_attempt"] = selected_attempt.name
    seal["g01_stage_state_sha256"] = sha256_file(
        selected_attempt / "stage_state.json"
    )
    seal["seal_identity_hash"] = stable_hash(
        {key: value for key, value in seal.items() if key != "seal_identity_hash"}
    )
    return seal


def verified_replay_source_seal(run_root: Path) -> dict[str, Any]:
    root = Path(run_root).resolve(strict=True)
    manifest = _read_manifest_payload(root)
    seal = manifest.get("replay_source_seal")
    if not isinstance(seal, dict):
        raise EvidenceIntegrityError("Replay source seal is missing")
    unsigned = dict(seal)
    expected = unsigned.pop("seal_identity_hash", None)
    if expected != stable_hash(unsigned):
        raise EvidenceIntegrityError("Replay source seal identity hash mismatch")
    selected = manifest.get("selected_attempts")
    g01 = selected.get("G01_repeatability") if isinstance(selected, dict) else None
    if not isinstance(g01, dict) or not isinstance(g01.get("attempt"), str):
        raise EvidenceIntegrityError("Replay source seal has no selected G01 attempt")
    attempt = require_path_within(
        root / "stages" / "G01_repeatability" / g01["attempt"],
        root,
        label="selected G01 replay attempt",
    ).resolve(strict=True)
    recomputed = _g01_replay_source_seal(root, attempt)
    if recomputed != seal:
        raise EvidenceIntegrityError("Replay source seal differs from selected G01 evidence")
    return dict(seal)


def _validate_p60_replay_binding(
    p60_result: Mapping[str, Any], seal: Mapping[str, Any]
) -> None:
    unsigned_seal = dict(seal)
    seal_identity_hash = unsigned_seal.pop("seal_identity_hash", None)
    if seal_identity_hash != stable_hash(unsigned_seal):
        raise EvidenceIntegrityError("P60 replay source seal identity is invalid")
    if p60_result.get("replay_source_seal") != dict(seal):
        raise EvidenceIntegrityError("P60 result does not embed the selected replay seal")
    if p60_result.get("replay_source_seal_identity_hash") != seal_identity_hash:
        raise EvidenceIntegrityError("P60 result does not bind the replay seal")
    if p60_result.get("replay_source_identity_hash") != seal.get(
        "replay_source_identity_hash"
    ):
        raise EvidenceIntegrityError("P60 result replay identity is not sealed")
    replay_identity = p60_result.get("replay_source_identity")
    if not isinstance(replay_identity, Mapping) or stable_hash(replay_identity) != seal.get(
        "replay_source_identity_hash"
    ):
        raise EvidenceIntegrityError("P60 replay source identity payload is invalid")
    evidence = p60_result.get("replay_evidence")
    if not isinstance(evidence, Mapping):
        raise EvidenceIntegrityError("P60 replay evidence index is missing")
    if evidence.get("source_result_sha256") != seal.get("source_result_sha256"):
        raise EvidenceIntegrityError("P60 replay source result hash is not sealed")
    if evidence.get("actions_sha256") != seal.get("clipped_action_file_sha256"):
        raise EvidenceIntegrityError("P60 replay action file hash is not sealed")


class RootCauseOrchestrator:
    def __init__(self, suite_root: Path, *, runs_root: Path | None = None) -> None:
        self.suite_root = Path(suite_root).resolve()
        if self.suite_root != SUITE_ROOT.resolve():
            raise ValueError(f"Unexpected suite root: {self.suite_root}")
        self.project_root = PROJECT_ROOT.resolve()
        candidate = Path(runs_root) if runs_root is not None else self.suite_root / "runs"
        self.runs_root = require_path_within(candidate, self.suite_root, label="runs root")
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.run_identity_accepted = False

    def _run_root(self, run_id: str) -> Path:
        selected = validate_run_id(run_id)
        candidate = self.runs_root / selected
        if candidate.parent != self.runs_root:
            raise EvidenceIntegrityError(f"Run root escapes runs directory: {candidate}")
        return candidate

    @staticmethod
    def _next_attempt(stage_root: Path) -> int:
        numbers: list[int] = []
        if stage_root.exists():
            for path in stage_root.iterdir():
                name = path.name.removesuffix(".incomplete")
                if name.startswith("attempt-") and name[8:].isdigit():
                    numbers.append(int(name[8:]))
        return max(numbers, default=0) + 1

    @staticmethod
    def verify_attempt(
        attempt: Path,
        *,
        expected_stage_id: str | None = None,
        expected_input_identity_hash: str | None = None,
        expected_worker_module: str | None = None,
        expected_worker: WorkerInvocation | None = None,
        expected_guard_root: Path | None = None,
        expected_project_root: Path | None = None,
    ) -> dict[str, object]:
        state_path = attempt / "stage_state.json"
        if not state_path.is_file():
            raise EvidenceIntegrityError(f"Missing stage state: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") != "complete":
            raise EvidenceIntegrityError(f"Stage attempt is not complete: {attempt}")
        if expected_stage_id is not None and state.get("stage_id") != expected_stage_id:
            raise EvidenceIntegrityError(f"Stage identity mismatch: {attempt}")
        if (
            expected_input_identity_hash is not None
            and state.get("input_identity_hash") != expected_input_identity_hash
        ):
            raise EvidenceIntegrityError(f"Stage input identity changed: {attempt}")
        expected = state.get("artifact_hashes")
        if not isinstance(expected, dict):
            raise EvidenceIntegrityError(f"Stage artifact hashes are missing: {attempt}")
        actual = _artifact_hashes(attempt)
        if set(actual) != set(expected):
            raise EvidenceIntegrityError(f"Stage artifact set mismatch: {attempt}")
        mismatches = {
            name: {"expected": expected[name], "actual": actual[name]}
            for name in actual
            if actual[name] != expected[name]
        }
        if mismatches:
            raise EvidenceIntegrityError(f"Stage artifact hash mismatch: {mismatches}")
        bootstrap_path = attempt / "worker" / "bootstrap_guard.json"
        if not bootstrap_path.is_file():
            raise EvidenceIntegrityError(
                f"Stage worker bootstrap evidence is missing: {attempt}"
            )
        bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        if bootstrap.get("schema_version") != "RootCauseWorkerBootstrapV1":
            raise EvidenceIntegrityError("Stage worker bootstrap schema is invalid")
        if (
            expected_worker_module is not None
            and bootstrap.get("worker_module") != expected_worker_module
        ):
            raise EvidenceIntegrityError("Stage worker module identity mismatch")
        if bootstrap.get("return_code") != 0:
            raise EvidenceIntegrityError("Stage worker bootstrap did not return success")
        if bootstrap.get("dont_write_bytecode") is not True:
            raise EvidenceIntegrityError("Stage worker bytecode guard was not active")
        environment = bootstrap.get("environment")
        if not isinstance(environment, dict) or any(
            environment.get(name) != "1"
            for name in ("PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE")
        ):
            raise EvidenceIntegrityError("Stage worker environment contract is invalid")
        write_guard = bootstrap.get("write_guard")
        if not isinstance(write_guard, dict):
            raise EvidenceIntegrityError("Stage worker write guard evidence is missing")
        if write_guard.get("installed") is not True or write_guard.get("probe_count") != 1:
            raise EvidenceIntegrityError("Stage worker write guard installation is invalid")
        expected_output = (attempt / "worker").resolve(strict=True)
        committed_from = (
            attempt.with_name(attempt.name + ".incomplete") / "worker"
        ).resolve()
        if Path(str(bootstrap.get("output"))).resolve() != committed_from:
            raise EvidenceIntegrityError("Stage worker bootstrap output path mismatch")
        if expected_guard_root is not None:
            expected_root = Path(expected_guard_root).resolve(strict=True)
            if Path(str(bootstrap.get("guard_root"))).resolve() != expected_root:
                raise EvidenceIntegrityError("Stage worker guard root mismatch")
            require_path_within(committed_from, expected_root, label="stage worker output")
        if expected_worker is not None:
            if expected_guard_root is None or expected_stage_id is None:
                raise ValueError("Exact worker verification requires guard root and stage id")
            project_root = (
                PROJECT_ROOT.resolve()
                if expected_project_root is None
                else Path(expected_project_root).resolve(strict=True)
            )
            expected_root = Path(expected_guard_root).resolve(strict=True)
            expected_command = expected_worker.command(
                committed_from, guard_root=expected_root
            )
            command_path = attempt / "commands" / "worker.json"
            if not command_path.is_file():
                raise EvidenceIntegrityError("Stage command evidence is missing")
            command_record = json.loads(command_path.read_text(encoding="utf-8"))
            source = RootCauseOrchestrator._worker_source(expected_worker.module)
            bootstrap_source = RootCauseOrchestrator._worker_source(WORKER_BOOTSTRAP)
            expected_environment = {
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX": str(
                    expected_root / "runtime_cache" / "pycache" / expected_stage_id
                ),
            }
            exact_command_fields = {
                "argv": expected_command,
                "argv_hash": stable_hash(expected_command),
                "cwd": str(project_root),
                "environment": expected_environment,
                "return_code": 0,
                "interpreter": str(expected_worker.interpreter.resolve(strict=True)),
                "interpreter_sha256": sha256_file(expected_worker.interpreter),
                "worker_module": expected_worker.module,
                "worker_source": str(source),
                "worker_source_sha256": sha256_file(source),
                "bootstrap_source": str(bootstrap_source),
                "bootstrap_source_sha256": sha256_file(bootstrap_source),
                "input_identity_hash": expected_input_identity_hash,
            }
            for field, expected_value in exact_command_fields.items():
                if command_record.get(field) != expected_value:
                    raise EvidenceIntegrityError(
                        f"Stage command field mismatch: {field}"
                    )
            separator = expected_command.index("--")
            expected_worker_argv = expected_command[separator + 1 :]
            if bootstrap.get("argv") != expected_worker_argv:
                raise EvidenceIntegrityError("Bootstrap worker argv mismatch")
            if bootstrap.get("process_argv") != [
                expected_worker.module,
                *expected_worker_argv,
            ]:
                raise EvidenceIntegrityError("Bootstrap process argv mismatch")
            if bootstrap.get("environment") != expected_environment:
                raise EvidenceIntegrityError("Bootstrap environment differs from command")
        return state

    @staticmethod
    def _worker_source(module: str) -> Path:
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise ValueError(f"Cannot resolve worker module: {module}")
        return Path(spec.origin).resolve()

    def _definition_payload(self, definition: StageDefinition) -> dict[str, object]:
        source = self._worker_source(definition.worker.module)
        bootstrap = self._worker_source(WORKER_BOOTSTRAP)
        return {
            "stage_id": definition.stage_id,
            "dependencies": list(definition.dependencies),
            "interpreter": str(definition.worker.interpreter.resolve(strict=True)),
            "interpreter_sha256": sha256_file(definition.worker.interpreter),
            "worker_module": definition.worker.module,
            "worker_source": str(source),
            "worker_source_sha256": sha256_file(source),
            "bootstrap_module": WORKER_BOOTSTRAP,
            "bootstrap_source": str(bootstrap),
            "bootstrap_source_sha256": sha256_file(bootstrap),
            "arguments": list(definition.worker.arguments),
        }

    def _definitions_payload(
        self, definitions: tuple[StageDefinition, ...]
    ) -> list[dict[str, object]]:
        return [self._definition_payload(item) for item in definitions]

    def _new_run_identity(self, *, full_identity: bool) -> dict[str, object]:
        if full_identity:
            return build_run_identity(
                project_python=self.project_root / ".venv/Scripts/python.exe",
                mujoco_python=self.project_root / "sim2sim/mujoco/.venv/Scripts/python.exe",
                parent_argv=[sys.executable, *sys.argv],
                environment=environment_contract(),
            )
        sources = suite_source_snapshot()
        return {
            "suite_sources": sources,
            "identity_hash": stable_hash({"suite_sources": sources}),
        }

    @staticmethod
    def _verify_light_identity(identity: Mapping[str, object]) -> None:
        current = suite_source_snapshot()
        if identity.get("suite_sources") != current:
            raise IdentityDriftError("Suite source identity changed since the run began")
        if identity.get("identity_hash") != stable_hash({"suite_sources": current}):
            raise IdentityDriftError("Run identity hash is invalid")

    def _write_manifest(self, path: Path, manifest: dict[str, object]) -> None:
        _atomic_manifest(path, manifest)

    def _load_manifest(self, run_root: Path) -> dict[str, object]:
        manifest = _read_manifest_payload(run_root)
        identity = manifest.get("run_identity")
        if not isinstance(identity, dict):
            raise EvidenceIntegrityError("Run identity is missing")
        if manifest.get("scope") == "core_v1":
            verify_run_identity(identity)
        else:
            self._verify_light_identity(identity)
        return manifest

    def _stage_input_identity(
        self,
        definition: StageDefinition,
        manifest: Mapping[str, object],
    ) -> tuple[str, dict[str, str]]:
        selections = manifest.get("selected_attempts", {})
        if not isinstance(selections, dict):
            raise EvidenceIntegrityError("Run selected_attempts is invalid")
        dependency_hashes: dict[str, str] = {}
        for dependency in definition.dependencies:
            record = selections.get(dependency)
            if not isinstance(record, dict) or not isinstance(
                record.get("stage_state_sha256"), str
            ):
                raise EvidenceIntegrityError(
                    f"Selected dependency state is missing: {dependency}"
                )
            dependency_hashes[dependency] = record["stage_state_sha256"]
        definitions = manifest.get("definitions")
        if not isinstance(definitions, list):
            raise EvidenceIntegrityError("Run definitions are missing")
        definition_payload = next(
            (
                item
                for item in definitions
                if isinstance(item, dict) and item.get("stage_id") == definition.stage_id
            ),
            None,
        )
        if definition_payload is None:
            raise EvidenceIntegrityError(f"Stage definition is missing: {definition.stage_id}")
        payload = {
            "run_identity_hash": manifest["run_identity"]["identity_hash"],
            "definition": definition_payload,
            "dependency_stage_state_sha256": dependency_hashes,
        }
        if definition.stage_id in {"P60_full_robot", "C70_checkpoint"}:
            seal = manifest.get("replay_source_seal")
            if not isinstance(seal, dict):
                raise EvidenceIntegrityError(
                    f"{definition.stage_id} requires a frozen replay source seal"
                )
            unsigned = dict(seal)
            seal_identity_hash = unsigned.pop("seal_identity_hash", None)
            if seal_identity_hash != stable_hash(unsigned):
                raise EvidenceIntegrityError("C70 replay source seal is invalid")
            payload["replay_source_seal_identity_hash"] = seal_identity_hash
        return stable_hash(payload), dependency_hashes

    def _execute_stage(
        self,
        run_root: Path,
        definition: StageDefinition,
        *,
        input_identity_hash: str,
        dependency_state_hashes: Mapping[str, str],
    ) -> StageOutcome:
        stage_root = require_path_within(
            run_root / "stages" / definition.stage_id,
            run_root,
            label="stage root",
        )
        stage_root.mkdir(parents=True, exist_ok=True)
        number = self._next_attempt(stage_root)
        incomplete = stage_root / f"attempt-{number:04d}.incomplete"
        final = stage_root / f"attempt-{number:04d}"
        incomplete.mkdir(parents=False, exist_ok=False)
        worker_output = incomplete / "worker"
        command = definition.worker.command(worker_output, guard_root=run_root)
        if command[1:3] != ["-B", "-m"] or command[3] != WORKER_BOOTSTRAP:
            raise ValueError("Worker command does not use the guarded -B bootstrap")
        started = _utc_now()
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPYCACHEPREFIX"] = str(
            run_root / "runtime_cache" / "pycache" / definition.stage_id
        )
        completed = subprocess.run(
            command,
            cwd=self.project_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        ended = _utc_now()
        commands = incomplete / "commands"
        commands.mkdir(parents=True, exist_ok=False)
        source = self._worker_source(definition.worker.module)
        bootstrap = self._worker_source(WORKER_BOOTSTRAP)
        command_payload = {
            "argv": command,
            "argv_hash": stable_hash(command),
            "cwd": str(self.project_root),
            "environment": {
                key: environment.get(key)
                for key in (
                    "PYTHONNOUSERSITE",
                    "PYTHONDONTWRITEBYTECODE",
                    "PYTHONPYCACHEPREFIX",
                )
            },
            "return_code": completed.returncode,
            "interpreter": str(definition.worker.interpreter.resolve(strict=True)),
            "interpreter_sha256": sha256_file(definition.worker.interpreter),
            "worker_module": definition.worker.module,
            "started_utc": started,
            "ended_utc": ended,
            "worker_source": str(source),
            "worker_source_sha256": sha256_file(source),
            "bootstrap_source": str(bootstrap),
            "bootstrap_source_sha256": sha256_file(bootstrap),
            "input_identity_hash": input_identity_hash,
        }
        _atomic_json(commands / "worker.json", command_payload)
        (commands / "stdout.txt").write_text(
            completed.stdout, encoding="utf-8", newline="\n"
        )
        (commands / "stderr.txt").write_text(
            completed.stderr, encoding="utf-8", newline="\n"
        )
        if completed.returncode != 0:
            _atomic_json(
                incomplete / "stage_state.json",
                {
                    "schema_version": "RootCauseStageV1",
                    "stage_id": definition.stage_id,
                    "status": "failed",
                    "reason": f"worker_exit_{completed.returncode}",
                    "input_identity_hash": input_identity_hash,
                    "dependency_stage_state_sha256": dict(dependency_state_hashes),
                    "artifact_hashes": _artifact_hashes(incomplete),
                },
            )
            return StageOutcome(
                definition.stage_id,
                "failed",
                f"worker_exit_{completed.returncode}",
                incomplete,
            )
        state = {
            "schema_version": "RootCauseStageV1",
            "stage_id": definition.stage_id,
            "status": "complete",
            "dependencies": list(definition.dependencies),
            "dependency_stage_state_sha256": dict(dependency_state_hashes),
            "input_identity_hash": input_identity_hash,
            "artifact_hashes": _artifact_hashes(incomplete),
        }
        _atomic_json(incomplete / "stage_state.json", state)
        os.replace(incomplete, final)
        self.verify_attempt(
            final,
            expected_stage_id=definition.stage_id,
            expected_input_identity_hash=input_identity_hash,
            expected_worker_module=definition.worker.module,
            expected_worker=definition.worker,
            expected_guard_root=run_root,
            expected_project_root=self.project_root,
        )
        return StageOutcome(definition.stage_id, "complete", None, final)

    def run_definitions(
        self,
        run_id: str,
        definitions: Iterable[StageDefinition],
        *,
        resume: bool = False,
        full_identity: bool = False,
        scope: str = "test",
    ) -> dict[str, StageOutcome]:
        self.run_identity_accepted = False
        run_root = self._run_root(run_id)
        manifest_path = run_root / RUN_MANIFEST_NAME
        if resume:
            if not _path_entry_exists(run_root):
                raise EvidenceIntegrityError(
                    f"Cannot resume missing run root: {run_root}"
                )
            _validate_run_root_entry(
                run_root, error_type=EvidenceIntegrityError
            )
            if not _path_entry_exists(manifest_path):
                raise EvidenceIntegrityError(
                    f"Cannot resume without run manifest: {manifest_path}"
                )
            _validate_existing_run_layout(run_root, require_manifest=True)
        else:
            created_run_root = False
            if not _path_entry_exists(run_root):
                try:
                    run_root.mkdir(parents=False, exist_ok=False)
                except FileExistsError:
                    pass
                else:
                    created_run_root = True
            _validate_run_root_entry(run_root, error_type=FileExistsError)
            if not created_run_root:
                _validate_fresh_bootstrap_skeleton(run_root)
            _ensure_runtime_cache_tree(run_root)
            _validate_fresh_bootstrap_skeleton(run_root)
        items = tuple(definitions)
        ids = [item.stage_id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate stage definition")
        known = set(ids)
        for item in items:
            unknown = set(item.dependencies) - known
            if unknown:
                raise ValueError(
                    f"Unknown dependencies for {item.stage_id}: {sorted(unknown)}"
                )
        definitions_payload = self._definitions_payload(items)
        definitions_hash = stable_hash(definitions_payload)
        if resume:
            manifest = self._load_manifest(run_root)
            if manifest.get("stage_ids") != ids:
                raise IdentityDriftError("Resume stage set changed")
            if manifest.get("definitions_hash") != definitions_hash:
                raise IdentityDriftError("Resume worker definition identity changed")
            if manifest.get("scope") != scope:
                raise IdentityDriftError("Resume run scope changed")
            self.run_identity_accepted = True
        else:
            claim = _create_initialization_claim(manifest_path, run_id=run_id)
            _validate_fresh_claimed_layout(run_root)
            persisted_claim = _read_initialization_claim(
                manifest_path, run_id=run_id
            )
            if persisted_claim != claim:
                raise EvidenceIntegrityError(
                    "Run initialization claim changed after exclusive creation"
                )
            manifest = {
                "schema_version": "RootCauseRunManifestV2",
                "run_id": run_id,
                "scope": scope,
                "design_version": DESIGN_VERSION,
                "design_sha256": DESIGN_SHA256,
                "stage_ids": ids,
                "definitions": definitions_payload,
                "definitions_hash": definitions_hash,
                "factor_path_allowlist_path": FACTOR_PATH_ALLOWLIST_PATH.name,
                "factor_path_allowlist_sha256": sha256_file(
                    FACTOR_PATH_ALLOWLIST_PATH
                ),
                "run_identity": self._new_run_identity(full_identity=full_identity),
                "initialization_claim": claim,
                "created_utc": _utc_now(),
                "launch_history": [],
                "selected_attempts": {},
                "replay_source_seal": None,
            }
        launches = manifest.setdefault("launch_history", [])
        if not isinstance(launches, list):
            raise EvidenceIntegrityError("Run launch history is invalid")
        launches.append(
            {
                "utc": _utc_now(),
                "argv": [sys.executable, *sys.argv],
                "environment": environment_contract(),
                "resume": bool(resume),
            }
        )
        self._write_manifest(manifest_path, manifest)
        if not resume:
            manifest = self._load_manifest(run_root)
            self.run_identity_accepted = True

        outcomes: dict[str, StageOutcome] = {}
        for definition in items:
            g02_outcome = outcomes.get("G02_adapter")
            if (
                "G02_adapter" in definition.dependencies
                and g02_outcome is not None
                and g02_outcome.status == "complete"
                and _attempt_has_terminal_adapter_bug(g02_outcome.attempt_dir)
            ):
                outcomes[definition.stage_id] = StageOutcome(
                    definition.stage_id,
                    "blocked",
                    "terminal:G02_adapter",
                    None,
                )
                continue
            g01_outcome = outcomes.get("G01_repeatability")
            if (
                "G01_repeatability" in definition.dependencies
                and g01_outcome is not None
                and g01_outcome.status == "complete"
                and _attempt_has_unusable_repeatability(g01_outcome.attempt_dir)
            ):
                outcomes[definition.stage_id] = StageOutcome(
                    definition.stage_id,
                    "blocked",
                    "terminal:G01_repeatability",
                    None,
                )
                continue
            failed_dependencies = [
                dependency
                for dependency in definition.dependencies
                if outcomes.get(dependency) is None
                or outcomes[dependency].status != "complete"
            ]
            if failed_dependencies:
                outcomes[definition.stage_id] = StageOutcome(
                    definition.stage_id,
                    "blocked",
                    "dependencies:" + ",".join(failed_dependencies),
                    None,
                )
                continue
            input_hash, dependency_hashes = self._stage_input_identity(
                definition, manifest
            )
            stage_root = run_root / "stages" / definition.stage_id
            selections = manifest["selected_attempts"]
            if not isinstance(selections, dict):
                raise EvidenceIntegrityError("Run selected_attempts is invalid")
            selected = selections.get(definition.stage_id)
            previous = None
            if isinstance(selected, dict) and isinstance(selected.get("attempt"), str):
                previous = stage_root / selected["attempt"]
            if resume and previous is not None:
                try:
                    state = self.verify_attempt(
                        previous,
                        expected_stage_id=definition.stage_id,
                        expected_input_identity_hash=input_hash,
                        expected_worker_module=definition.worker.module,
                        expected_worker=definition.worker,
                        expected_guard_root=run_root,
                        expected_project_root=self.project_root,
                    )
                except (EvidenceIntegrityError, json.JSONDecodeError):
                    previous = None
                else:
                    selection = {
                        "attempt": previous.name,
                        "stage_state_sha256": sha256_file(previous / "stage_state.json"),
                        "input_identity_hash": state["input_identity_hash"],
                    }
                    if (
                        definition.stage_id == "G02_adapter"
                        and _attempt_has_terminal_adapter_bug(previous)
                    ):
                        _replace_terminal_g02_selection(
                            manifest,
                            items,
                            selection=selection,
                        )
                    else:
                        selections[definition.stage_id] = selection
                    if definition.stage_id == "G01_repeatability":
                        seal = _g01_replay_source_seal(run_root, previous)
                        existing = manifest.get("replay_source_seal")
                        if existing is not None and existing != seal:
                            raise EvidenceIntegrityError(
                                "Selected G01 replay seal differs from the manifest"
                            )
                        manifest["replay_source_seal"] = seal
                    self._write_manifest(manifest_path, manifest)
                    outcomes[definition.stage_id] = StageOutcome(
                        definition.stage_id, "complete", None, previous
                    )
                    continue
            outcome = self._execute_stage(
                run_root,
                definition,
                input_identity_hash=input_hash,
                dependency_state_hashes=dependency_hashes,
            )
            outcomes[definition.stage_id] = outcome
            if outcome.status == "complete" and outcome.attempt_dir is not None:
                manifest = _read_manifest_payload(run_root)
                selections = manifest.get("selected_attempts")
                if not isinstance(selections, dict):
                    raise EvidenceIntegrityError("Run selected_attempts is invalid")
                selection = {
                    "attempt": outcome.attempt_dir.name,
                    "stage_state_sha256": sha256_file(
                        outcome.attempt_dir / "stage_state.json"
                    ),
                    "input_identity_hash": input_hash,
                }
                if definition.stage_id == "G01_repeatability":
                    seal = _g01_replay_source_seal(
                        run_root, outcome.attempt_dir
                    )
                    _replace_g01_selection_and_seal(
                        manifest,
                        items,
                        selection=selection,
                        seal=seal,
                    )
                elif (
                    definition.stage_id == "G02_adapter"
                    and _attempt_has_terminal_adapter_bug(outcome.attempt_dir)
                ):
                    _replace_terminal_g02_selection(
                        manifest,
                        items,
                        selection=selection,
                    )
                else:
                    selections[definition.stage_id] = selection
                self._write_manifest(manifest_path, manifest)
        _atomic_json(
            run_root / "run_state.json",
            {
                "schema_version": "RootCauseRunStateV1",
                "run_id": run_id,
                "stages": {
                    name: {
                        "status": outcome.status,
                        "reason": outcome.reason,
                        "attempt": (
                            None
                            if outcome.attempt_dir is None
                            else outcome.attempt_dir.name
                        ),
                    }
                    for name, outcome in outcomes.items()
                },
            },
        )
        return outcomes

    def run(self, *, run_id: str | None, resume: bool) -> dict[str, StageOutcome]:
        selected = run_id or f"root-cause-core-v1-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run_root = self._run_root(selected)
        outcomes = self.run_definitions(
            selected,
            self.core_definitions(run_root),
            resume=resume,
            full_identity=True,
            scope="core_v1",
        )
        manifest = _read_manifest_payload(run_root)
        terminal_adapter_bug = _manifest_has_terminal_adapter_bug(run_root, manifest)
        repeatability_unavailable = _manifest_has_unusable_repeatability(
            run_root, manifest
        )
        if terminal_adapter_bug or repeatability_unavailable:
            incomplete_gates = {
                stage: outcomes[stage].status
                for stage in ADAPTER_TERMINAL_STAGE_IDS
                if outcomes.get(stage) is None
                or outcomes[stage].status != "complete"
            }
            if incomplete_gates:
                raise StageExecutionError(
                    "Terminal adapter run did not complete all gate stages: "
                    f"{incomplete_gates}"
                )
            verified = self.verify_stage_evidence(selected, require_core=True)
            from .finalize import (
                finalize_adapter_terminal,
                finalize_repeatability_terminal,
                verify_final_state,
            )

            if terminal_adapter_bug:
                finalize_adapter_terminal(run_root, verified_attempts=verified)
            else:
                finalize_repeatability_terminal(
                    run_root, verified_attempts=verified
                )
            verify_final_state(run_root, verified_attempts=verified)
            return outcomes
        incomplete = {
            name: outcome.status
            for name, outcome in outcomes.items()
            if outcome.status != "complete"
        }
        if incomplete:
            raise StageExecutionError(f"Core run did not complete: {incomplete}")
        verified = self.verify_stage_evidence(selected, require_core=True)
        from .finalize import finalize_run, verify_final_state

        finalize_run(run_root, verified_attempts=verified)
        verify_final_state(run_root, verified_attempts=verified)
        return outcomes

    def core_definitions(self, run_root: Path) -> tuple[StageDefinition, ...]:
        project_python = self.project_root / ".venv/Scripts/python.exe"

        def stage(stage_id: str, dependencies: tuple[str, ...]) -> StageDefinition:
            return StageDefinition(
                stage_id,
                dependencies,
                WorkerInvocation(
                    project_python,
                    "debug.sim2sim.root_cause_suite.core_stage_worker",
                    ("--stage", stage_id, "--run-root", str(run_root)),
                ),
            )

        return (
            stage("G00_integrity", ()),
            stage("G01_repeatability", ("G00_integrity",)),
            stage("G02_adapter", ("G00_integrity",)),
            stage("G03_instrumentation", ("G00_integrity",)),
            stage(
                "P10_rest",
                ("G01_repeatability", "G02_adapter", "G03_instrumentation"),
            ),
            stage(
                "P20_static_properties",
                ("G01_repeatability", "G02_adapter", "G03_instrumentation"),
            ),
            stage("P30_actuator", ("P10_rest", "P20_static_properties")),
            stage("P40_closure", ("P10_rest", "P20_static_properties")),
            stage("P50_contact", ("P10_rest", "P20_static_properties")),
            stage(
                "P60_full_robot",
                ("P30_actuator", "P40_closure", "P50_contact"),
            ),
            stage("C70_checkpoint", ("P60_full_robot",)),
        )

    def verify_stage_evidence(
        self,
        run_id: str,
        *,
        require_core: bool,
    ) -> dict[str, Path]:
        run_root = self._run_root(run_id)
        manifest = self._load_manifest(run_root)
        stage_ids = manifest.get("stage_ids")
        if not isinstance(stage_ids, list) or not all(
            isinstance(item, str) for item in stage_ids
        ):
            raise EvidenceIntegrityError("Run stage_ids are invalid")
        if require_core:
            if manifest.get("scope") != "core_v1":
                raise IdentityDriftError("Core run scope changed")
            expected_definitions = self.core_definitions(run_root)
            expected_ids = [item.stage_id for item in expected_definitions]
            expected_payload = self._definitions_payload(expected_definitions)
            if stage_ids != expected_ids:
                raise EvidenceIntegrityError("Core run stage set is not exact")
            if manifest.get("definitions") != expected_payload:
                raise IdentityDriftError("Core run definitions changed")
            if manifest.get("definitions_hash") != stable_hash(expected_payload):
                raise EvidenceIntegrityError("Core definition hash mismatch")
        self.run_identity_accepted = True
        terminal_adapter_bug = _manifest_has_terminal_adapter_bug(run_root, manifest)
        repeatability_unavailable = _manifest_has_unusable_repeatability(
            run_root, manifest
        )
        verification_stage_ids = list(stage_ids)
        if terminal_adapter_bug or repeatability_unavailable:
            if tuple(stage_ids[: len(ADAPTER_TERMINAL_STAGE_IDS)]) != (
                ADAPTER_TERMINAL_STAGE_IDS
            ):
                raise EvidenceIntegrityError(
                    "Terminal adapter run does not have the frozen gate-stage prefix"
                )
            verification_stage_ids = list(ADAPTER_TERMINAL_STAGE_IDS)
        selections = manifest.get("selected_attempts")
        if not isinstance(selections, dict) or set(selections) != set(
            verification_stage_ids
        ):
            raise EvidenceIntegrityError("Selected attempt set is incomplete or contains extras")
        verified: dict[str, Path] = {}
        selected_state_hashes: dict[str, str] = {}
        definitions_by_id = {
            item["stage_id"]: item
            for item in manifest["definitions"]
            if isinstance(item, dict)
        }
        run_identity = manifest.get("run_identity")
        if not isinstance(run_identity, dict) or not isinstance(
            run_identity.get("identity_hash"), str
        ):
            raise EvidenceIntegrityError("Run identity hash is missing")
        for stage_id in verification_stage_ids:
            record = selections.get(stage_id)
            if not isinstance(record, dict):
                raise EvidenceIntegrityError(f"Selected attempt is invalid: {stage_id}")
            attempt_name = record.get("attempt")
            if not isinstance(attempt_name, str):
                raise EvidenceIntegrityError(f"Selected attempt name is invalid: {stage_id}")
            attempt = require_path_within(
                run_root / "stages" / stage_id / attempt_name,
                run_root,
                label="selected attempt",
            )
            definition_payload = definitions_by_id.get(stage_id)
            if not isinstance(definition_payload, dict):
                raise EvidenceIntegrityError(f"Stage definition is missing: {stage_id}")
            dependencies = definition_payload.get("dependencies")
            if not isinstance(dependencies, list) or not all(
                isinstance(name, str) for name in dependencies
            ):
                raise EvidenceIntegrityError(
                    f"Stage dependency declaration is invalid: {stage_id}"
                )
            expected_dependency_hashes = {
                name: selected_state_hashes[name] for name in dependencies
            }
            input_payload: dict[str, object] = {
                "run_identity_hash": run_identity["identity_hash"],
                "definition": definition_payload,
                "dependency_stage_state_sha256": expected_dependency_hashes,
            }
            if stage_id in {"P60_full_robot", "C70_checkpoint"}:
                seal = manifest.get("replay_source_seal")
                if not isinstance(seal, dict):
                    raise EvidenceIntegrityError(
                        f"{stage_id} replay source seal is missing"
                    )
                unsigned = dict(seal)
                seal_identity_hash = unsigned.pop("seal_identity_hash", None)
                if seal_identity_hash != stable_hash(unsigned):
                    raise EvidenceIntegrityError(
                        f"{stage_id} replay source seal is invalid"
                    )
                input_payload["replay_source_seal_identity_hash"] = seal_identity_hash
            expected_input = stable_hash(input_payload)
            if record.get("input_identity_hash") != expected_input:
                raise EvidenceIntegrityError(
                    f"Selected input identity mismatch: {stage_id}"
                )
            state = self.verify_attempt(
                attempt,
                expected_stage_id=stage_id,
                expected_input_identity_hash=expected_input,
                expected_worker_module=str(definition_payload.get("worker_module")),
                expected_worker=WorkerInvocation(
                    Path(str(definition_payload.get("interpreter"))),
                    str(definition_payload.get("worker_module")),
                    tuple(str(value) for value in definition_payload.get("arguments", [])),
                ),
                expected_guard_root=run_root,
                expected_project_root=self.project_root,
            )
            state_hash = sha256_file(attempt / "stage_state.json")
            if record.get("stage_state_sha256") != state_hash:
                raise EvidenceIntegrityError(
                    f"Selected stage_state hash mismatch: {stage_id}"
                )
            if state.get("dependency_stage_state_sha256") != expected_dependency_hashes:
                raise EvidenceIntegrityError(
                    f"Selected dependency binding mismatch: {stage_id}"
                )
            selected_state_hashes[stage_id] = state_hash
            verified[stage_id] = attempt
        if (
            require_core
            and not terminal_adapter_bug
            and not repeatability_unavailable
        ):
            seal = verified_replay_source_seal(run_root)
            p60_result_path = verified["P60_full_robot"] / "worker" / "result.json"
            p60_result = json.loads(p60_result_path.read_text(encoding="utf-8"))
            _validate_p60_replay_binding(p60_result, seal)
            c70_result = json.loads(
                (verified["C70_checkpoint"] / "worker" / "result.json").read_text(
                    encoding="utf-8"
                )
            )
            if c70_result.get("replay_source_seal_identity_hash") != seal.get(
                "seal_identity_hash"
            ):
                raise EvidenceIntegrityError("C70 result does not bind the replay seal")
            if c70_result.get("p60_stage_state_sha256") != selected_state_hashes.get(
                "P60_full_robot"
            ):
                raise EvidenceIntegrityError("C70 result does not bind selected P60 evidence")
        return verified

    def verify(self, run_id: str) -> dict[str, str]:
        run_root = self._run_root(run_id)
        verified = self.verify_stage_evidence(run_id, require_core=True)
        from .finalize import verify_final_state

        verify_final_state(run_root, verified_attempts=verified)
        result = {stage: "complete" for stage in verified}
        result["final_state"] = "complete"
        return result

    def report(self, run_id: str) -> None:
        run_root = self._run_root(run_id)
        verified = self.verify_stage_evidence(run_id, require_core=True)
        manifest = _read_manifest_payload(run_root)
        from .finalize import (
            finalize_adapter_terminal,
            finalize_repeatability_terminal,
            finalize_run,
            verify_final_state,
        )

        if _manifest_has_terminal_adapter_bug(run_root, manifest):
            finalize_adapter_terminal(run_root, verified_attempts=verified)
        elif _manifest_has_unusable_repeatability(run_root, manifest):
            finalize_repeatability_terminal(run_root, verified_attempts=verified)
        else:
            finalize_run(run_root, verified_attempts=verified)
        verify_final_state(run_root, verified_attempts=verified)
