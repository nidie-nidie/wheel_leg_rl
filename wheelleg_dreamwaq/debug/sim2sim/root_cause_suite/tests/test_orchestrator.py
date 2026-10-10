from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

import debug.sim2sim.root_cause_suite.orchestrator as orchestrator_module
from debug.sim2sim.root_cause_suite.contracts import (
    EvidenceIntegrityError,
    canonical_json_bytes,
    sha256_file,
    stable_hash,
)
from debug.sim2sim.root_cause_suite.orchestrator import (
    ADAPTER_TERMINAL_STAGE_IDS,
    RootCauseOrchestrator,
    StageDefinition,
    WorkerInvocation,
    _compute_replay_source_seal,
    _manifest_identity_payload,
    _validate_p60_replay_binding,
)
from debug.sim2sim.root_cause_suite.trace_contract import FieldSpec, write_trace


SUITE_ROOT = Path(__file__).resolve().parents[1]


def _worker(stage: str, *extra: str) -> WorkerInvocation:
    return WorkerInvocation(
        Path(sys.executable),
        "debug.sim2sim.root_cause_suite.fake_worker",
        ("--stage", stage, *extra),
    )


def test_worker_command_uses_argument_array_and_python_b() -> None:
    command = _worker("G00").command(Path("out"), guard_root=SUITE_ROOT)
    assert command[1:3] == ["-B", "-m"]
    assert command[3] == "debug.sim2sim.root_cause_suite.worker_bootstrap"
    assert command[4:8] == [
        "--guard-root",
        str(SUITE_ROOT),
        "--worker-module",
        "debug.sim2sim.root_cause_suite.fake_worker",
    ]
    assert "--output" in command


def test_dag_blocks_only_dependents_and_continues_independent_branch(tmp_path: Path) -> None:
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=tmp_path / "runs")
    definitions = (
        StageDefinition("A", (), _worker("A", "--fail")),
        StageDefinition("B", ("A",), _worker("B")),
        StageDefinition("C", (), _worker("C")),
    )
    outcomes = orchestrator.run_definitions("unit-dag", definitions)
    assert outcomes["A"].status == "failed"
    assert outcomes["B"].status == "blocked"
    assert outcomes["C"].status == "complete"


def test_fresh_run_accepts_canonical_bootstrap_skeleton(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = runs_root / "fresh-skeleton"
    for name in ("pycache", "pytest-cache", "pytest-tmp"):
        (run_root / "runtime_cache" / name).mkdir(parents=True)
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=runs_root)

    outcome = orchestrator.run_definitions(
        "fresh-skeleton", (StageDefinition("A", (), _worker("A")),)
    )["A"]

    assert outcome.status == "complete"
    assert (run_root / "run_manifest.json").is_file()
    manifest = json.loads(
        (run_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["initialization_claim"]["run_id"] == "fresh-skeleton"


def test_fresh_run_claim_allows_exactly_one_concurrent_owner(
    tmp_path: Path, monkeypatch,
) -> None:
    runs_root = tmp_path / "runs"
    run_root = runs_root / "concurrent-owner"
    for name in ("pycache", "pytest-cache", "pytest-tmp"):
        (run_root / "runtime_cache" / name).mkdir(parents=True)
    orchestrators = (
        RootCauseOrchestrator(SUITE_ROOT, runs_root=runs_root),
        RootCauseOrchestrator(SUITE_ROOT, runs_root=runs_root),
    )
    definition = StageDefinition("A", (), _worker("A"))
    barrier = threading.Barrier(2)
    original = orchestrator_module._create_initialization_claim

    def synchronized_claim(path: Path, *, run_id: str) -> dict[str, object]:
        barrier.wait(timeout=10)
        return original(path, run_id=run_id)

    monkeypatch.setattr(
        orchestrator_module, "_create_initialization_claim", synchronized_claim
    )

    def launch(orchestrator: RootCauseOrchestrator) -> object:
        try:
            return orchestrator.run_definitions(
                "concurrent-owner", (definition,)
            )
        except BaseException as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(launch, orchestrators))

    successes = [result for result in results if isinstance(result, dict)]
    failures = [result for result in results if isinstance(result, BaseException)]
    assert len(successes) == 1
    assert successes[0]["A"].status == "complete"
    assert len(failures) == 1
    assert isinstance(failures[0], FileExistsError)


def test_fresh_run_rejects_mutation_between_validation_and_claim(
    tmp_path: Path, monkeypatch,
) -> None:
    runs_root = tmp_path / "runs"
    run_root = runs_root / "claim-mutation"
    for name in ("pycache", "pytest-cache", "pytest-tmp"):
        (run_root / "runtime_cache" / name).mkdir(parents=True)
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=runs_root)
    original = orchestrator_module._create_initialization_claim
    identity_called = False

    def mutating_claim(path: Path, *, run_id: str) -> dict[str, object]:
        (path.parent / "foreign.txt").write_text("unexpected\n", encoding="utf-8")
        return original(path, run_id=run_id)

    def unexpected_identity(*, full_identity: bool) -> dict[str, object]:
        nonlocal identity_called
        identity_called = True
        return {"full_identity": full_identity}

    monkeypatch.setattr(
        orchestrator_module, "_create_initialization_claim", mutating_claim
    )
    monkeypatch.setattr(orchestrator, "_new_run_identity", unexpected_identity)

    with pytest.raises(FileExistsError, match="Unexpected run-root entries"):
        orchestrator.run_definitions(
            "claim-mutation", (StageDefinition("A", (), _worker("A")),)
        )

    assert identity_called is False
    assert not (run_root / "stages").exists()


def _directory_symlink(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Directory symlinks are unavailable: {error}")


def test_fresh_run_rejects_run_root_symlink_alias(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    target = runs_root / "alias-target"
    for name in ("pycache", "pytest-cache", "pytest-tmp"):
        (target / "runtime_cache" / name).mkdir(parents=True)
    alias = runs_root / "alias-entry"
    _directory_symlink(alias, target)
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=runs_root)

    with pytest.raises(FileExistsError, match="link or reparse point"):
        orchestrator.run_definitions(
            "alias-entry", (StageDefinition("A", (), _worker("A")),)
        )

    assert not (target / "run_manifest.json").exists()


def test_fresh_run_rejects_runtime_cache_symlink(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = runs_root / "runtime-cache-link"
    run_root.mkdir(parents=True)
    target = tmp_path / "runtime-cache-target"
    for name in ("pycache", "pytest-cache", "pytest-tmp"):
        (target / name).mkdir(parents=True)
    _directory_symlink(run_root / "runtime_cache", target)
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=runs_root)

    with pytest.raises(FileExistsError, match="link or reparse point"):
        orchestrator.run_definitions(
            "runtime-cache-link", (StageDefinition("A", (), _worker("A")),)
        )


def test_fresh_run_rejects_cache_child_symlink(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = runs_root / "cache-child-link"
    runtime_cache = run_root / "runtime_cache"
    (runtime_cache / "pytest-cache").mkdir(parents=True)
    (runtime_cache / "pytest-tmp").mkdir()
    target = tmp_path / "pycache-target"
    target.mkdir()
    _directory_symlink(runtime_cache / "pycache", target)
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=runs_root)

    with pytest.raises(FileExistsError, match="link or reparse point"):
        orchestrator.run_definitions(
            "cache-child-link", (StageDefinition("A", (), _worker("A")),)
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
def test_fresh_run_rejects_run_root_junction_alias(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    target = runs_root / "junction-alias-target"
    for name in ("pycache", "pytest-cache", "pytest-tmp"):
        (target / "runtime_cache" / name).mkdir(parents=True)
    junction = runs_root / "junction-alias"
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"Directory junctions are unavailable: {completed.stderr}")
    try:
        orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=runs_root)
        with pytest.raises(FileExistsError, match="link or reparse point"):
            orchestrator.run_definitions(
                "junction-alias", (StageDefinition("A", (), _worker("A")),)
            )
        assert not (target / "run_manifest.json").exists()
    finally:
        if os.path.lexists(junction):
            os.rmdir(junction)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
def test_fresh_run_rejects_runtime_cache_junction(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = runs_root / "runtime-cache-junction"
    run_root.mkdir(parents=True)
    target = tmp_path / "runtime-cache-junction-target"
    for name in ("pycache", "pytest-cache", "pytest-tmp"):
        (target / name).mkdir(parents=True)
    junction = run_root / "runtime_cache"
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"Directory junctions are unavailable: {completed.stderr}")
    try:
        orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=runs_root)
        with pytest.raises(FileExistsError, match="link or reparse point"):
            orchestrator.run_definitions(
                "runtime-cache-junction",
                (StageDefinition("A", (), _worker("A")),),
            )
    finally:
        if os.path.lexists(junction):
            os.rmdir(junction)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
def test_fresh_run_rejects_cache_child_junction(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = runs_root / "cache-child-junction"
    runtime_cache = run_root / "runtime_cache"
    (runtime_cache / "pytest-cache").mkdir(parents=True)
    (runtime_cache / "pytest-tmp").mkdir()
    target = tmp_path / "pycache-junction-target"
    target.mkdir()
    junction = runtime_cache / "pycache"
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"Directory junctions are unavailable: {completed.stderr}")
    try:
        orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=runs_root)
        with pytest.raises(FileExistsError, match="link or reparse point"):
            orchestrator.run_definitions(
                "cache-child-junction",
                (StageDefinition("A", (), _worker("A")),),
            )
    finally:
        if os.path.lexists(junction):
            os.rmdir(junction)


@pytest.mark.parametrize("pollution", ("foreign-file", "cache-file", "foreign-dir"))
def test_fresh_run_rejects_polluted_bootstrap_skeleton(
    tmp_path: Path, pollution: str
) -> None:
    runs_root = tmp_path / "runs"
    run_root = runs_root / "polluted-skeleton"
    for name in ("pycache", "pytest-cache", "pytest-tmp"):
        (run_root / "runtime_cache" / name).mkdir(parents=True)
    if pollution == "foreign-file":
        (run_root / "foreign.txt").write_text("unexpected\n", encoding="utf-8")
    elif pollution == "cache-file":
        (run_root / "runtime_cache" / "pycache" / "foreign.pyc").write_bytes(
            b"unexpected"
        )
    else:
        (run_root / "runtime_cache" / "foreign").mkdir()
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=runs_root)

    with pytest.raises(FileExistsError):
        orchestrator.run_definitions(
            "polluted-skeleton", (StageDefinition("A", (), _worker("A")),)
        )


def test_terminal_g02_adapter_bug_completes_gate_and_skips_physics(
    tmp_path: Path, monkeypatch,
) -> None:
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=tmp_path / "runs")
    definitions = (
        StageDefinition("G00_integrity", (), _worker("G00_integrity")),
        StageDefinition(
            "G01_repeatability",
            ("G00_integrity",),
            _worker("G01_repeatability", "--replay-source"),
        ),
        StageDefinition(
            "G02_adapter",
            ("G00_integrity",),
            _worker("G02_adapter", "--terminal-adapter-bug"),
        ),
        StageDefinition(
            "G03_instrumentation", ("G00_integrity",), _worker("G03_instrumentation")
        ),
        StageDefinition(
            "P10_rest",
            ("G01_repeatability", "G02_adapter", "G03_instrumentation"),
            _worker("P10_rest"),
        ),
    )

    outcomes = orchestrator.run_definitions(
        "terminal-g02",
        definitions,
        full_identity=True,
        scope="core_v1",
    )

    assert outcomes["G02_adapter"].status == "complete"
    assert outcomes["G03_instrumentation"].status == "complete"
    assert outcomes["P10_rest"].status == "blocked"
    assert outcomes["P10_rest"].reason == "terminal:G02_adapter"
    manifest = json.loads(
        (tmp_path / "runs" / "terminal-g02" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(manifest["selected_attempts"]) == {
        "G00_integrity",
        "G01_repeatability",
        "G02_adapter",
        "G03_instrumentation",
    }
    monkeypatch.setattr(orchestrator, "core_definitions", lambda _run_root: definitions)
    verified = orchestrator.verify_stage_evidence("terminal-g02", require_core=True)
    assert set(verified) == set(manifest["selected_attempts"])


def test_unusable_g01_completes_gates_and_skips_physics_as_inconclusive(
    tmp_path: Path, monkeypatch,
) -> None:
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=tmp_path / "runs")
    definitions = (
        StageDefinition("G00_integrity", (), _worker("G00_integrity")),
        StageDefinition(
            "G01_repeatability",
            ("G00_integrity",),
            _worker(
                "G01_repeatability",
                "--replay-source",
                "--repeatability-unavailable",
            ),
        ),
        StageDefinition("G02_adapter", ("G00_integrity",), _worker("G02_adapter")),
        StageDefinition(
            "G03_instrumentation", ("G00_integrity",), _worker("G03_instrumentation")
        ),
        StageDefinition(
            "P10_rest",
            ("G01_repeatability", "G02_adapter", "G03_instrumentation"),
            _worker("P10_rest"),
        ),
    )

    outcomes = orchestrator.run_definitions(
        "terminal-g01",
        definitions,
        full_identity=True,
        scope="core_v1",
    )

    assert outcomes["G01_repeatability"].status == "complete"
    assert outcomes["G02_adapter"].status == "complete"
    assert outcomes["G03_instrumentation"].status == "complete"
    assert outcomes["P10_rest"].status == "blocked"
    assert outcomes["P10_rest"].reason == "terminal:G01_repeatability"
    monkeypatch.setattr(orchestrator, "core_definitions", lambda _run_root: definitions)
    verified = orchestrator.verify_stage_evidence("terminal-g01", require_core=True)
    assert set(verified) == set(ADAPTER_TERMINAL_STAGE_IDS)


def test_p60_verification_reuses_replay_seal_input_identity(tmp_path: Path) -> None:
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=tmp_path / "runs")
    definitions = (
        StageDefinition(
            "G01_repeatability", (), _worker("G01_repeatability", "--replay-source")
        ),
        StageDefinition(
            "P60_full_robot", ("G01_repeatability",), _worker("P60_full_robot")
        ),
    )

    outcomes = orchestrator.run_definitions("p60-seal-verify", definitions)
    assert outcomes["P60_full_robot"].status == "complete"

    verified = orchestrator.verify_stage_evidence(
        "p60-seal-verify", require_core=False
    )
    assert set(verified) == {"G01_repeatability", "P60_full_robot"}


def test_resume_terminal_g02_invalidates_stale_downstream_selection(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "terminal.marker"
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=tmp_path / "runs")
    definitions = (
        StageDefinition("G00_integrity", (), _worker("G00_integrity")),
        StageDefinition(
            "G01_repeatability",
            ("G00_integrity",),
            _worker("G01_repeatability", "--replay-source"),
        ),
        StageDefinition(
            "G02_adapter",
            ("G00_integrity",),
            _worker("G02_adapter", "--terminal-adapter-marker", str(marker)),
        ),
        StageDefinition(
            "G03_instrumentation", ("G00_integrity",), _worker("G03_instrumentation")
        ),
        StageDefinition(
            "P10_rest",
            ("G01_repeatability", "G02_adapter", "G03_instrumentation"),
            _worker("P10_rest"),
        ),
    )
    first = orchestrator.run_definitions("resume-terminal-g02", definitions)
    assert first["P10_rest"].status == "complete"
    first_g02 = first["G02_adapter"].attempt_dir
    assert first_g02 is not None

    marker.write_text("terminal\n", encoding="utf-8")
    (first_g02 / "worker" / "result.json").write_text(
        "tampered\n", encoding="utf-8"
    )
    resumed = orchestrator.run_definitions(
        "resume-terminal-g02", definitions, resume=True
    )

    assert resumed["G02_adapter"].status == "complete"
    assert resumed["G02_adapter"].attempt_dir is not None
    assert resumed["G02_adapter"].attempt_dir.name == "attempt-0002"
    assert resumed["P10_rest"].status == "blocked"
    assert resumed["P10_rest"].reason == "terminal:G02_adapter"
    manifest = json.loads(
        (
            tmp_path
            / "runs"
            / "resume-terminal-g02"
            / "run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert set(manifest["selected_attempts"]) == set(ADAPTER_TERMINAL_STAGE_IDS)


def test_resume_reuses_only_hash_verified_complete_attempt(tmp_path: Path) -> None:
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=tmp_path / "runs")
    definition = StageDefinition("A", (), _worker("A"))
    first = orchestrator.run_definitions("resume", (definition,))
    attempt = first["A"].attempt_dir
    assert attempt is not None
    second = orchestrator.run_definitions("resume", (definition,), resume=True)
    assert second["A"].attempt_dir == attempt
    (attempt / "worker" / "result.json").write_text("tampered", encoding="utf-8")
    third = orchestrator.run_definitions("resume", (definition,), resume=True)
    assert third["A"].status == "complete"
    assert third["A"].attempt_dir is not None
    assert third["A"].attempt_dir.name == "attempt-0002"
    assert attempt.exists()


def test_resume_reselects_g01_seal_and_invalidates_only_transitive_downstream(
    tmp_path: Path,
) -> None:
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=tmp_path / "runs")
    definitions = (
        StageDefinition(
            "G01_repeatability", (), _worker("G01_repeatability", "--replay-source")
        ),
        StageDefinition("independent", (), _worker("independent")),
        StageDefinition("dependent", ("G01_repeatability",), _worker("dependent")),
        StageDefinition("transitive", ("dependent",), _worker("transitive")),
    )
    first = orchestrator.run_definitions("g01-reseal", definitions)
    first_g01 = first["G01_repeatability"].attempt_dir
    first_independent = first["independent"].attempt_dir
    first_dependent = first["dependent"].attempt_dir
    first_transitive = first["transitive"].attempt_dir
    assert all(
        path is not None
        for path in (
            first_g01,
            first_independent,
            first_dependent,
            first_transitive,
        )
    )

    assert first_g01 is not None
    (first_g01 / "worker" / "result.json").write_text(
        "tampered\n", encoding="utf-8"
    )
    resumed = orchestrator.run_definitions("g01-reseal", definitions, resume=True)

    assert resumed["G01_repeatability"].attempt_dir is not None
    assert resumed["G01_repeatability"].attempt_dir.name == "attempt-0002"
    assert resumed["independent"].attempt_dir == first_independent
    assert resumed["dependent"].attempt_dir is not None
    assert resumed["dependent"].attempt_dir.name == "attempt-0002"
    assert resumed["transitive"].attempt_dir is not None
    assert resumed["transitive"].attempt_dir.name == "attempt-0002"

    manifest = json.loads(
        (tmp_path / "runs" / "g01-reseal" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["selected_attempts"]["G01_repeatability"]["attempt"] == (
        "attempt-0002"
    )
    assert manifest["replay_source_seal"]["g01_attempt"] == "attempt-0002"
    assert manifest["selected_attempts"]["independent"]["attempt"] == "attempt-0001"
    assert manifest["selected_attempts"]["dependent"]["attempt"] == "attempt-0002"
    assert manifest["selected_attempts"]["transitive"]["attempt"] == "attempt-0002"


def test_resume_without_manifest_rejects_incomplete_attempt_debris(
    tmp_path: Path,
) -> None:
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=tmp_path / "runs")
    stage_root = tmp_path / "runs" / "interrupted" / "stages" / "A"
    incomplete = stage_root / "attempt-0001.incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "partial.txt").write_text("partial", encoding="utf-8")

    with pytest.raises(EvidenceIntegrityError, match="without run manifest"):
        orchestrator.run_definitions(
            "interrupted", (StageDefinition("A", (), _worker("A")),), resume=True
        )

    assert not (tmp_path / "runs" / "interrupted" / "run_manifest.json").exists()
    assert not (tmp_path / "runs" / "interrupted" / "runtime_cache").exists()
    assert (incomplete / "partial.txt").read_text(encoding="utf-8") == "partial"


def test_resume_missing_run_does_not_create_directory(tmp_path: Path) -> None:
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=tmp_path / "runs")
    run_root = tmp_path / "runs" / "missing-resume"

    with pytest.raises(EvidenceIntegrityError, match="missing run root"):
        orchestrator.run_definitions(
            "missing-resume", (StageDefinition("A", (), _worker("A")),), resume=True
        )

    assert not run_root.exists()


def test_verify_fails_on_unlisted_extra_artifact(tmp_path: Path) -> None:
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=tmp_path / "runs")
    outcome = orchestrator.run_definitions(
        "verify-extra", (StageDefinition("A", (), _worker("A")),)
    )["A"]
    assert outcome.attempt_dir is not None
    (outcome.attempt_dir / "extra.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="artifact set"):
        orchestrator.verify_stage_evidence("verify-extra", require_core=False)


def test_run_manifest_records_worker_argv_and_source_hash(tmp_path: Path) -> None:
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=tmp_path / "runs")
    outcome = orchestrator.run_definitions(
        "command", (StageDefinition("A", (), _worker("A")),)
    )["A"]
    assert outcome.attempt_dir is not None
    command = json.loads((outcome.attempt_dir / "commands" / "worker.json").read_text(encoding="utf-8"))
    assert command["argv"][1:3] == ["-B", "-m"]
    assert len(command["worker_source_sha256"]) == 64
    bootstrap = json.loads(
        (outcome.attempt_dir / "worker" / "bootstrap_guard.json").read_text(
            encoding="utf-8"
        )
    )
    assert bootstrap["worker_module"] == "debug.sim2sim.root_cause_suite.fake_worker"
    assert bootstrap["process_argv"][0] == bootstrap["worker_module"]
    assert "--" not in bootstrap["process_argv"]
    assert bootstrap["process_argv"][1:] == bootstrap["argv"]
    assert bootstrap["write_guard"]["installed"] is True
    assert bootstrap["write_guard"]["probe_count"] == 1


def _rehash_attempt(attempt: Path) -> None:
    state_path = attempt / "stage_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["artifact_hashes"] = {
        path.relative_to(attempt).as_posix(): sha256_file(path)
        for path in sorted(attempt.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and path.name != "stage_state.json"
    }
    state_path.write_bytes(canonical_json_bytes(state) + b"\n")


@pytest.mark.parametrize("target", ("command", "bootstrap"))
def test_verify_rejects_semantically_changed_worker_command(
    tmp_path: Path, target: str
) -> None:
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=tmp_path / "runs")
    definition = StageDefinition("A", (), _worker("A"))
    outcome = orchestrator.run_definitions("command-tamper", (definition,))["A"]
    assert outcome.attempt_dir is not None
    attempt = outcome.attempt_dir
    if target == "command":
        path = attempt / "commands" / "worker.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["argv"][-1] = "wrong-output"
    else:
        path = attempt / "worker" / "bootstrap_guard.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["argv"][0:0] = ["--unexpected"]
    path.write_bytes(canonical_json_bytes(payload) + b"\n")
    _rehash_attempt(attempt)
    manifest_path = tmp_path / "runs" / "command-tamper" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selected_attempts"]["A"]["stage_state_sha256"] = sha256_file(
        attempt / "stage_state.json"
    )
    manifest["manifest_identity_hash"] = stable_hash(
        _manifest_identity_payload(manifest)
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    with pytest.raises(ValueError, match="command|argv"):
        orchestrator.verify_stage_evidence("command-tamper", require_core=False)


def test_fast_exit_worker_finalizes_bootstrap_evidence(tmp_path: Path) -> None:
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=tmp_path / "runs")
    outcome = orchestrator.run_definitions(
        "fast-exit",
        (StageDefinition("A", (), _worker("A", "--fast-exit")),),
    )["A"]
    assert outcome.status == "complete"
    assert outcome.attempt_dir is not None
    bootstrap = json.loads(
        (outcome.attempt_dir / "worker" / "bootstrap_guard.json").read_text(
            encoding="utf-8"
        )
    )
    assert bootstrap["return_code"] == 0
    assert bootstrap["write_guard"]["installed"] is True


def test_selected_attempt_manifest_identity_detects_tampering(tmp_path: Path) -> None:
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=tmp_path / "runs")
    orchestrator.run_definitions(
        "selection-binding", (StageDefinition("A", (), _worker("A")),)
    )
    manifest_path = tmp_path / "runs" / "selection-binding" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selected_attempts"]["A"]["attempt"] = "attempt-9999"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest identity"):
        orchestrator.verify_stage_evidence("selection-binding", require_core=False)


def test_rehashed_manifest_cannot_replace_recomputed_input_identity(
    tmp_path: Path,
) -> None:
    orchestrator = RootCauseOrchestrator(SUITE_ROOT, runs_root=tmp_path / "runs")
    definitions = (
        StageDefinition("A", (), _worker("A")),
        StageDefinition("B", ("A",), _worker("B")),
    )
    orchestrator.run_definitions("rehash-input", definitions)
    manifest_path = tmp_path / "runs" / "rehash-input" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selected_attempts"]["B"]["input_identity_hash"] = "0" * 64
    manifest["manifest_identity_hash"] = stable_hash(
        _manifest_identity_payload(manifest)
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    with pytest.raises(ValueError, match="Selected input identity mismatch"):
        orchestrator.verify_stage_evidence("rehash-input", require_core=False)


def _make_replay_source(root: Path) -> tuple[Path, Path]:
    import hashlib

    source = root / "source"
    source.mkdir(parents=True)
    actions = np.arange(18, dtype=np.float32).reshape(3, 6) / 100.0
    action_path = source / "actions.npz"
    np.savez(
        action_path,
        action_sequence=actions,
        environment_action_sequence=actions[:, None, :],
    )
    trace = write_trace(
        source / "trace",
        {
            "time_s": np.asarray((0.0, 0.02, 0.04), dtype=np.float64),
            "value": np.asarray((1.0, 2.0, 3.0), dtype=np.float64),
        },
        {
            "time_s": FieldSpec("s", "simulation", "pre_step", "time"),
            "value": FieldSpec("1", "canonical", "pre_step", "test"),
        },
    )
    replay_identity = {"horizon": 3, "environment_count": 1}
    payload = {
        "schema_version": "RootCauseIsaacReplaySourceV1",
        "replay_source_identity": replay_identity,
        "replay_source_identity_hash": stable_hash(replay_identity),
        "action_sequence_file": "actions.npz",
        "clipped_action_file_sha256": sha256_file(action_path),
        "clipped_action_sequence_sha256": hashlib.sha256(
            np.ascontiguousarray(actions).tobytes()
        ).hexdigest().upper(),
        "action_count": 3,
        "first_action": actions[0].tolist(),
        "source_trace_sha256": trace.trace_sha256,
        "trace_metadata_sha256": trace.metadata_sha256,
    }
    result_path = source / "result.json"
    result_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    return result_path, action_path


def test_replay_source_seal_binds_file_tensor_trace_and_identity(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baseline"
    baseline_root.mkdir()
    result_path, action_path = _make_replay_source(baseline_root)
    seal = _compute_replay_source_seal(
        baseline_root,
        source_result_path=result_path,
        actions_path=action_path,
    )
    assert seal["action_count"] == 3

    action_root = tmp_path / "action-tamper"
    action_root.mkdir()
    result_path, action_path = _make_replay_source(action_root)
    with np.load(action_path, allow_pickle=False) as archive:
        actions = np.array(archive["action_sequence"], copy=True)
        environment = np.array(archive["environment_action_sequence"], copy=True)
    actions[0, 0] += 1.0
    environment[0, 0, 0] += 1.0
    np.savez(
        action_path,
        action_sequence=actions,
        environment_action_sequence=environment,
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["clipped_action_file_sha256"] = sha256_file(action_path)
    result_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(ValueError, match="action tensor hash"):
        _compute_replay_source_seal(
            action_root,
            source_result_path=result_path,
            actions_path=action_path,
        )

    trace_root = tmp_path / "trace-tamper"
    trace_root.mkdir()
    result_path, action_path = _make_replay_source(trace_root)
    trace_path = result_path.parent / "trace" / "trace.npz"
    np.savez(
        trace_path,
        time_s=np.asarray((0.0, 0.02, 0.04), dtype=np.float64),
        value=np.asarray((9.0, 9.0, 9.0), dtype=np.float64),
    )
    metadata_path = result_path.parent / "trace" / "trace.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["trace_sha256"] = sha256_file(trace_path)
    metadata_path.write_bytes(canonical_json_bytes(metadata) + b"\n")
    with pytest.raises(ValueError, match="result trace hash"):
        _compute_replay_source_seal(
            trace_root,
            source_result_path=result_path,
            actions_path=action_path,
        )

    identity_root = tmp_path / "identity-tamper"
    identity_root.mkdir()
    result_path, action_path = _make_replay_source(identity_root)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["replay_source_identity"]["horizon"] = 4
    result_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(ValueError, match="identity hash"):
        _compute_replay_source_seal(
            identity_root,
            source_result_path=result_path,
            actions_path=action_path,
        )


def test_p60_result_must_bind_the_selected_replay_seal() -> None:
    replay_identity = {
        "source_scenario_id": "P60_D_REPLAY_SOURCE_DREAMWAQ_RUN01",
        "horizon": 3,
        "environment_count": 1,
    }
    seal = {
        "schema_version": "RootCauseReplaySourceSealV1",
        "replay_source_identity_hash": stable_hash(replay_identity),
        "source_result_sha256": "A" * 64,
        "source_trace_sha256": "B" * 64,
        "clipped_action_file_sha256": "C" * 64,
        "clipped_action_sequence_sha256": "D" * 64,
        "action_count": 3,
        "g01_attempt": "attempt-0001",
        "g01_stage_state_sha256": "E" * 64,
    }
    seal["seal_identity_hash"] = stable_hash(seal)
    result = {
        "replay_source_identity_hash": stable_hash(replay_identity),
        "replay_source_identity": replay_identity,
        "replay_source_seal_identity_hash": seal["seal_identity_hash"],
        "replay_source_seal": seal,
        "replay_evidence": {
            "source_result_sha256": seal["source_result_sha256"],
            "actions_sha256": seal["clipped_action_file_sha256"],
        },
    }
    _validate_p60_replay_binding(result, seal)

    tampered = deepcopy(result)
    tampered["replay_evidence"]["actions_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="action file hash"):
        _validate_p60_replay_binding(tampered, seal)

    tampered = deepcopy(result)
    tampered["replay_source_identity"]["horizon"] = 4
    with pytest.raises(ValueError, match="identity payload"):
        _validate_p60_replay_binding(tampered, seal)
