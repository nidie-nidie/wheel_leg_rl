import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import pytest

from debug.sim2sim.root_cause_suite.contracts import (
    canonical_json_bytes,
    stable_hash,
)
from debug.sim2sim.root_cause_suite.orchestrator import (
    RootCauseOrchestrator,
    _manifest_identity_payload,
)


SUITE_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = SUITE_ROOT / "runs"
LAUNCHER = SUITE_ROOT / "run_suite.ps1"


def _shell_command(shell: str, *arguments: str) -> list[str]:
    command = [shell, "-NoProfile"]
    if Path(shell).name.lower() == "powershell.exe":
        command.extend(("-ExecutionPolicy", "Bypass"))
    command.extend(("-File", str(LAUNCHER), *arguments))
    return command


def _invoke_launcher(shell: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _shell_command(shell, *arguments),
        cwd=SUITE_ROOT.parents[2],
        capture_output=True,
        text=True,
        check=False,
    )


def _unique_run_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _safe_regular_run_root(path: Path) -> Path:
    runs_root = RUNS_ROOT.resolve(strict=True)
    assert path.parent.resolve(strict=True) == runs_root
    assert path.name and path.name not in {".", ".."}
    return path


def _remove_regular_run_root(path: Path) -> None:
    if os.path.lexists(path):
        shutil.rmtree(_safe_regular_run_root(path))


def _tree_snapshot(root: Path) -> dict[str, object]:
    directories: list[str] = []
    files: dict[str, bytes] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            directories.append(relative)
        elif path.is_file():
            files[relative] = path.read_bytes()
    return {"directories": directories, "files": files}


def _create_junction(link: Path, target: Path) -> None:
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"Directory junctions are unavailable: {completed.stderr}")


def _remove_junction(path: Path) -> None:
    if os.path.lexists(path):
        os.rmdir(path)


def _launcher_environment(run_id: str, run_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "WHEELLEG_ROOT_CAUSE_BOOTSTRAP": "1",
            "WHEELLEG_ROOT_CAUSE_RUN_ID": run_id,
            "WHEELLEG_ROOT_CAUSE_RUN_ROOT": str(run_root.resolve()),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(
                run_root.resolve() / "runtime_cache" / "pycache"
            ),
        }
    )
    return environment


def _create_empty_run(
    run_id: str, *, scope: str, full_identity: bool
) -> tuple[RootCauseOrchestrator, Path]:
    run_root = RUNS_ROOT / run_id
    environment = _launcher_environment(run_id, run_root)
    names = (
        "WHEELLEG_ROOT_CAUSE_BOOTSTRAP",
        "WHEELLEG_ROOT_CAUSE_RUN_ID",
        "WHEELLEG_ROOT_CAUSE_RUN_ROOT",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPYCACHEPREFIX",
    )
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update({name: environment[name] for name in names})
        orchestrator = RootCauseOrchestrator(SUITE_ROOT)
        orchestrator.run_definitions(
            run_id, (), full_identity=full_identity, scope=scope
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return orchestrator, run_root


def _rewrite_manifest_for_command_drift(
    orchestrator: RootCauseOrchestrator,
    run_root: Path,
    drift: str,
) -> None:
    path = run_root / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    definitions = orchestrator._definitions_payload(
        orchestrator.core_definitions(run_root)
    )
    stage_ids = [item["stage_id"] for item in definitions]
    if drift == "definition":
        definitions = json.loads(json.dumps(definitions))
        definitions[0]["arguments"].append("--identity-drift")
    if drift in {"definition", "scope"}:
        manifest["stage_ids"] = stage_ids
        manifest["definitions"] = definitions
        manifest["definitions_hash"] = stable_hash(definitions)
    manifest["manifest_identity_hash"] = stable_hash(
        _manifest_identity_payload(manifest)
    )
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")


def _normalization_block(script: str) -> str:
    start = script.index("$forwardArgs = @()")
    end = script.index("$suiteRoot =", start)
    return script[start:end]


def test_run_suite_uses_untyped_forward_args_for_all_downstream_access() -> None:
    script = LAUNCHER.read_text(encoding="utf-8")
    normalization = "$forwardArgs = @()"

    assert normalization in script
    assert script.index(normalization) < script.index("[Array]::IndexOf")
    assert "$RemainingArgs = @($RemainingArgs)" not in script
    assert "[Array]::IndexOf($forwardArgs, '--resume')" in script
    assert "$forwardArgs.Count" in script
    assert "+ $forwardArgs" in script
    assert "if ($isFreshRun)" in script
    assert script.index("if ($isFreshRun)") < script.index("New-Item -ItemType Directory")


@pytest.mark.parametrize("shell_name", ("powershell.exe", "pwsh.exe"))
@pytest.mark.parametrize(
    ("arguments", "expected_count", "expected_index", "expected_joined"),
    (
        ((), 0, -1, ""),
        (("verify-id",), 1, -1, "verify-id"),
        (("--resume", "run-id"), 2, 0, "--resume|run-id"),
    ),
)
def test_forward_arg_normalization_runs_in_both_powershell_engines(
    tmp_path: Path,
    shell_name: str,
    arguments: tuple[str, ...],
    expected_count: int,
    expected_index: int,
    expected_joined: str,
) -> None:
    shell = shutil.which(shell_name)
    if shell is None:
        pytest.skip(f"{shell_name} is unavailable")
    script = LAUNCHER.read_text(encoding="utf-8")
    probe = tmp_path / f"forward-args-{shell_name}.ps1"
    probe.write_text(
        "\n".join(
            (
                "param(",
                "    [Parameter(ValueFromRemainingArguments = $true)]",
                "    [string[]]$RemainingArgs",
                ")",
                _normalization_block(script),
                "$payload = [ordered]@{",
                "    count = $forwardArgs.Count",
                "    resume_index = [Array]::IndexOf($forwardArgs, '--resume')",
                "    joined = [string]::Join('|', $forwardArgs)",
                "}",
                "$payload | ConvertTo-Json -Compress",
            )
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(probe), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip())

    assert payload == {
        "count": expected_count,
        "resume_index": expected_index,
        "joined": expected_joined,
    }


@pytest.mark.parametrize("shell_name", ("powershell.exe", "pwsh.exe"))
@pytest.mark.parametrize(
    "arguments",
    (
        (),
        ("invalid-command",),
        ("run", "--resume"),
        ("verify",),
        ("verify", "../invalid"),
    ),
)
def test_official_launcher_configuration_errors_return_code_2(
    shell_name: str, arguments: tuple[str, ...]
) -> None:
    shell = shutil.which(shell_name)
    if shell is None:
        pytest.skip(f"{shell_name} is unavailable")
    completed = _invoke_launcher(shell, *arguments)
    assert completed.returncode == 2


@pytest.mark.parametrize("shell_name", ("powershell.exe", "pwsh.exe"))
@pytest.mark.parametrize("command", ("resume", "verify", "report"))
def test_official_launcher_missing_run_is_read_only(
    shell_name: str, command: str
) -> None:
    shell = shutil.which(shell_name)
    if shell is None:
        pytest.skip(f"{shell_name} is unavailable")
    run_id = _unique_run_id(f"launcher-missing-{command}")
    run_root = RUNS_ROOT / run_id
    arguments = (
        ("run", "--resume", run_id)
        if command == "resume"
        else (command, run_id)
    )
    try:
        completed = _invoke_launcher(shell, *arguments)
        assert completed.returncode == 5
        assert "Missing run root" in completed.stderr
        assert not os.path.lexists(run_root)
    finally:
        _remove_regular_run_root(run_root)


@pytest.mark.parametrize("shell_name", ("powershell.exe", "pwsh.exe"))
@pytest.mark.parametrize("command", ("resume", "verify", "report"))
def test_official_launcher_manifestless_debris_is_unchanged(
    shell_name: str, command: str
) -> None:
    shell = shutil.which(shell_name)
    if shell is None:
        pytest.skip(f"{shell_name} is unavailable")
    run_id = _unique_run_id(f"launcher-debris-{command}")
    run_root = RUNS_ROOT / run_id
    partial = run_root / "stages" / "A" / "attempt-0001.incomplete" / "partial.txt"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"unowned debris\n")
    before = _tree_snapshot(run_root)
    arguments = (
        ("run", "--resume", run_id)
        if command == "resume"
        else (command, run_id)
    )
    try:
        completed = _invoke_launcher(shell, *arguments)
        assert completed.returncode == 5
        assert "Missing run manifest" in completed.stderr
        assert _tree_snapshot(run_root) == before
        assert not (run_root / "runtime_cache").exists()
        assert not (run_root / "parent_guard.json").exists()
    finally:
        _remove_regular_run_root(run_root)


@pytest.mark.parametrize("shell_name", ("powershell.exe", "pwsh.exe"))
@pytest.mark.parametrize("manifest_kind", ("claim-only", "corrupt"))
def test_official_launcher_invalid_manifest_writes_no_parent_guard(
    shell_name: str, manifest_kind: str
) -> None:
    shell = shutil.which(shell_name)
    if shell is None:
        pytest.skip(f"{shell_name} is unavailable")
    run_id = _unique_run_id(f"launcher-{manifest_kind}")
    run_root = RUNS_ROOT / run_id
    for name in ("pycache", "pytest-cache", "pytest-tmp"):
        (run_root / "runtime_cache" / name).mkdir(parents=True)
    manifest_path = run_root / "run_manifest.json"
    if manifest_kind == "claim-only":
        unsigned = {
            "schema_version": "RootCauseRunInitializationClaimV1",
            "run_id": run_id,
            "created_utc": "2026-10-08T00:00:00+00:00",
            "process_id": 1,
            "nonce": "test-claim",
        }
        payload = {**unsigned, "claim_identity_hash": stable_hash(unsigned)}
        manifest_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    else:
        manifest_path.write_bytes(b"{broken\n")
    before = _tree_snapshot(run_root)
    try:
        completed = _invoke_launcher(shell, "run", "--resume", run_id)
        assert completed.returncode == 5
        assert _tree_snapshot(run_root) == before
        assert not (run_root / "parent_guard.json").exists()
    finally:
        _remove_regular_run_root(run_root)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
@pytest.mark.parametrize("shell_name", ("powershell.exe", "pwsh.exe"))
@pytest.mark.parametrize(
    "junction_position", ("run-root", "runtime-cache", "cache-child")
)
def test_official_launcher_rejects_junction_before_any_redirected_write(
    shell_name: str, junction_position: str
) -> None:
    shell = shutil.which(shell_name)
    if shell is None:
        pytest.skip(f"{shell_name} is unavailable")
    run_id = _unique_run_id(f"launcher-junction-{junction_position}")
    run_root = RUNS_ROOT / run_id
    target = RUNS_ROOT / _unique_run_id("launcher-junction-target")
    target.mkdir(parents=True)
    (target / "marker.txt").write_bytes(b"unchanged\n")
    junction: Path
    if junction_position == "run-root":
        junction = run_root
        _create_junction(junction, target)
    else:
        run_root.mkdir()
        (run_root / "run_manifest.json").write_bytes(b"{not-used\n")
        if junction_position == "runtime-cache":
            junction = run_root / "runtime_cache"
            _create_junction(junction, target)
        else:
            runtime_cache = run_root / "runtime_cache"
            (runtime_cache / "pytest-cache").mkdir(parents=True)
            (runtime_cache / "pytest-tmp").mkdir()
            junction = runtime_cache / "pycache"
            _create_junction(junction, target)
    target_before = _tree_snapshot(target)
    try:
        completed = _invoke_launcher(shell, "run", "--resume", run_id)
        assert completed.returncode == 5
        assert "link or reparse point" in completed.stderr
        assert _tree_snapshot(target) == target_before
        assert not (target / "parent_guard.json").exists()
    finally:
        _remove_junction(junction)
        _remove_regular_run_root(run_root)
        _remove_regular_run_root(target)


@pytest.mark.parametrize("drift", ("stage-set", "definition", "scope"))
def test_official_launcher_identity_drift_writes_no_parent_guard(
    drift: str,
) -> None:
    run_id = _unique_run_id(f"launcher-identity-{drift}")
    run_root = RUNS_ROOT / run_id
    scope = "test" if drift == "scope" else "core_v1"
    try:
        orchestrator, run_root = _create_empty_run(
            run_id,
            scope=scope,
            full_identity=drift != "scope",
        )
        _rewrite_manifest_for_command_drift(orchestrator, run_root, drift)
        before = _tree_snapshot(run_root)
        executed = 0
        for shell_name in ("powershell.exe", "pwsh.exe"):
            shell = shutil.which(shell_name)
            if shell is None:
                continue
            executed += 1
            completed = _invoke_launcher(shell, "run", "--resume", run_id)
            assert completed.returncode == 3
            assert _tree_snapshot(run_root) == before
            assert not (run_root / "parent_guard.json").exists()
        if executed == 0:
            pytest.skip("No PowerShell engine is available")
    finally:
        _remove_regular_run_root(run_root)


def test_parent_guard_is_written_after_accepted_stage_failure() -> None:
    run_id = _unique_run_id("launcher-accepted-failure")
    run_root = RUNS_ROOT / run_id
    code = "\n".join(
        (
            "import sys",
            "from debug.sim2sim.root_cause_suite.contracts import StageExecutionError",
            "from debug.sim2sim.root_cause_suite.orchestrator import RootCauseOrchestrator",
            "def fail_after_acceptance(self, *, run_id, resume):",
            "    self.run_definitions(run_id, (), resume=True, full_identity=False, scope='test')",
            "    raise StageExecutionError('expected accepted-stage failure')",
            "RootCauseOrchestrator.run = fail_after_acceptance",
            f"sys.argv = ['root-cause-suite', 'run', '--resume', {run_id!r}]",
            "from debug.sim2sim.root_cause_suite.__main__ import main",
            "raise SystemExit(main())",
        )
    )
    try:
        _, run_root = _create_empty_run(
            run_id,
            scope="test",
            full_identity=False,
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=SUITE_ROOT.parents[2],
            env=_launcher_environment(run_id, run_root),
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 4
        parent_guard = json.loads(
            (run_root / "parent_guard.json").read_text(encoding="utf-8")
        )
        assert parent_guard["schema_version"] == "RootCauseParentBootstrapV1"
        assert parent_guard["return_code"] == 4
        assert parent_guard["write_guard"]["installed"] is True
        assert parent_guard["write_guard"]["probe_count"] == 1
    finally:
        _remove_regular_run_root(run_root)
