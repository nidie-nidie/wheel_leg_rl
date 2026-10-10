from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from .adapter_gate import evaluate_adapter_gate
from .analysis import (
    analyze_c70_checkpoint_sensitivity,
    compare_p10_trace_directories,
    compare_p30_trace_directories,
    compare_p40_trace_directories,
    compare_p50_impact_trace_directories,
    compare_p50_slide_trace_directories,
    compare_p60_replay_trace_directories,
    compare_p60_robot_trace_directories,
    response_mismatch_metrics,
)
from .causal_contract import load_result, validate_ratio_pair, validate_result_identity
from .contracts import (
    DESIGN_SHA256,
    DESIGN_VERSION,
    FROZEN_POLICIES,
    FROZEN_REPLAY_SOURCE,
    PROJECT_ROOT,
    StageName,
    WORKSPACE_ROOT,
    canonical_json_bytes,
    require_path_within,
    sha256_file,
    stable_hash,
    validate_design_identity,
)
from .integrity import verify_run_identity
from .orchestrator import (
    _read_manifest_payload,
    verified_replay_source_seal,
)
from .repeatability import (
    build_threshold_snapshot,
    frozen_repeat_envelope,
    validate_threshold_snapshot,
)
from .scenario_catalog import (
    FACTOR_PATH_ALLOWLIST_PATH,
    catalog,
    factor_path_allowlist_artifact,
)
from .trace_contract import load_verified_trace


PROJECT_PYTHON = PROJECT_ROOT / ".venv/Scripts/python.exe"
MUJOCO_PYTHON = PROJECT_ROOT / "sim2sim/mujoco/.venv/Scripts/python.exe"
MODEL_MANIFEST = PROJECT_ROOT / "sim2sim/mujoco/model_manifest.json"
ISAAC_MODULE = "debug.sim2sim.root_cause_suite.isaac_worker"
MUJOCO_MODULE = "debug.sim2sim.root_cause_suite.mujoco_worker"
GUARD_SEMANTICS_MODULE = "debug.sim2sim.root_cause_suite.guard_semantics_worker"
WORKER_BOOTSTRAP = "debug.sim2sim.root_cause_suite.worker_bootstrap"

FORMAL_FILES = {
    "architecture": (
        WORKSPACE_ROOT / "docs/2026-10-03-wheelleg-dreamwaq-architecture.md",
        "5C7AF59F56750B50BA3CF14D731DF619489C0BBDBCCDA620B885B610087BB3AE",
    ),
    "mujoco_xml": (
        PROJECT_ROOT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml",
        "691C607271ED85EFFA388237E00A7B8F8F41CB66F6BA9C49EA64ABD186D803D1",
    ),
    "mujoco_manifest": (
        MODEL_MANIFEST,
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RootCauseSuite real stage worker")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


class StageRunner:
    def __init__(self, stage: str, run_root: Path, output: Path) -> None:
        self.stage = stage
        self.run_root = run_root.resolve(strict=True)
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.commands = self.output / "commands"
        self.commands.mkdir()

    def _child(
        self,
        name: str,
        interpreter: Path,
        module: str,
        arguments: list[str],
        *,
        output: Path,
        extra_environment: dict[str, str] | None = None,
    ) -> Path:
        output = output.resolve()
        if output.exists():
            raise FileExistsError(output)
        command = [
            str(interpreter),
            "-B",
            "-m",
            WORKER_BOOTSTRAP,
            "--guard-root",
            str(self.run_root),
            "--worker-module",
            module,
            "--",
            *arguments,
            "--output",
            str(output),
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX": str(
                    self.run_root / "runtime_cache" / "pycache" / self.stage / name
                ),
            }
        )
        if extra_environment:
            environment.update(extra_environment)
        started = _utc_now()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        ended = _utc_now()
        (self.commands / f"{name}.stdout.txt").write_text(
            completed.stdout, encoding="utf-8", newline="\n"
        )
        (self.commands / f"{name}.stderr.txt").write_text(
            completed.stderr, encoding="utf-8", newline="\n"
        )
        _write_json(
            self.commands / f"{name}.json",
            {
                "argv": command,
                "cwd": str(PROJECT_ROOT),
                "started_utc": started,
                "ended_utc": ended,
                "return_code": completed.returncode,
                "interpreter_sha256": sha256_file(interpreter),
                "module": module,
                "bootstrap_module": WORKER_BOOTSTRAP,
                "argv_hash": stable_hash(command),
            },
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{self.stage}/{name} failed with exit {completed.returncode}; "
                f"see {self.commands / f'{name}.stderr.txt'}"
            )
        bootstrap_path = output / "bootstrap_guard.json"
        if not bootstrap_path.is_file():
            raise RuntimeError(f"{self.stage}/{name} has no bootstrap guard evidence")
        bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        guard = bootstrap.get("write_guard")
        if (
            bootstrap.get("schema_version") != "RootCauseWorkerBootstrapV1"
            or bootstrap.get("worker_module") != module
            or bootstrap.get("return_code") != 0
            or bootstrap.get("dont_write_bytecode") is not True
            or Path(str(bootstrap.get("guard_root"))).resolve() != self.run_root
            or Path(str(bootstrap.get("output"))).resolve() != output
            or not isinstance(guard, dict)
            or guard.get("installed") is not True
            or guard.get("probe_count") != 1
        ):
            raise RuntimeError(f"{self.stage}/{name} bootstrap guard evidence is invalid")
        return output

    def mujoco(self, name: str, command: str, *arguments: object) -> Path:
        python_path = os.pathsep.join(
            (str(PROJECT_ROOT), str(PROJECT_ROOT / "sim2sim/mujoco"))
        )
        return self._child(
            name,
            MUJOCO_PYTHON,
            MUJOCO_MODULE,
            [command, *(str(value) for value in arguments)],
            output=self.output / "mujoco" / name,
            extra_environment={"PYTHONPATH": python_path},
        )

    def isaac(self, name: str, command: str, *arguments: object) -> Path:
        kit_root = self.run_root / "runtime_cache" / "kit" / self.stage / name
        return self._child(
            name,
            PROJECT_PYTHON,
            ISAAC_MODULE,
            [
                command,
                "--run-root",
                str(self.run_root),
                "--kit-root",
                str(kit_root),
                "--device",
                "cuda:0",
                *(str(value) for value in arguments),
            ],
            output=self.output / "isaac" / name,
        )

    def guard_semantics(self) -> Path:
        return self._child(
            "python_guard_semantics",
            PROJECT_PYTHON,
            GUARD_SEMANTICS_MODULE,
            ["run", "--run-root", str(self.run_root)],
            output=self.output / "guard" / "python_semantics",
        )

    def write_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = {
            "schema_version": "RootCauseCoreStageResultV1",
            "stage": self.stage,
            "design_version": DESIGN_VERSION,
            "design_sha256": DESIGN_SHA256,
            **payload,
        }
        _write_json(self.output / "result.json", result)
        return result


def _hash_snapshot() -> dict[str, dict[str, str]]:
    snapshot: dict[str, dict[str, str]] = {}
    for name, (path, expected) in FORMAL_FILES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Frozen formal file drifted: {name}: {actual} != {expected}")
        snapshot[name] = {
            "path": str(path.resolve()),
            "expected_sha256": expected,
            "actual_sha256": actual,
        }
    for policy_name, records in FROZEN_POLICIES.items():
        for kind, record in records.items():
            actual = sha256_file(record.absolute_path())
            if actual != record.sha256:
                raise ValueError(f"Frozen policy drifted: {policy_name}/{kind}")
            snapshot[f"policy:{policy_name}:{kind}"] = {
                "path": str(record.absolute_path()),
                "expected_sha256": record.sha256,
                "actual_sha256": actual,
            }
    return snapshot


def _run_g00(runner: StageRunner) -> dict[str, Any]:
    validate_design_identity()
    scenarios = catalog()
    factor_allowlist = factor_path_allowlist_artifact()
    manifest = _read_manifest_payload(runner.run_root)
    identity = manifest.get("run_identity")
    if not isinstance(identity, dict):
        raise RuntimeError("G00 run identity is missing")
    verify_run_identity(identity, verify_environment=False)
    pycache_prefix = os.environ.get("PYTHONPYCACHEPREFIX")
    if not pycache_prefix:
        raise RuntimeError("G00 worker PYTHONPYCACHEPREFIX is missing")
    require_path_within(pycache_prefix, runner.run_root, label="G00 worker pycache")
    gates = {
        "scope_is_core_v1": manifest.get("scope") == "core_v1",
        "full_stage_set": manifest.get("stage_ids")
        == [item.value for item in (
            # Keep this explicit so a catalog or enum expansion cannot silently
            # add work outside the frozen Core graph.
            StageName.G00_INTEGRITY,
            StageName.G01_REPEATABILITY,
            StageName.G02_ADAPTER,
            StageName.G03_INSTRUMENTATION,
            StageName.P10_REST,
            StageName.P20_STATIC_PROPERTIES,
            StageName.P30_ACTUATOR,
            StageName.P40_CLOSURE,
            StageName.P50_CONTACT,
            StageName.P60_FULL_ROBOT,
            StageName.C70_CHECKPOINT,
        )],
        "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
        "no_user_site": os.environ.get("PYTHONNOUSERSITE") == "1",
        "pycache_inside_run": True,
        "run_identity_static_verified": True,
        "catalog_core_only": len(scenarios) == len({item.scenario_id for item in scenarios}),
        "factor_path_allowlist_bound": manifest.get(
            "factor_path_allowlist_sha256"
        )
        == sha256_file(FACTOR_PATH_ALLOWLIST_PATH),
    }
    return runner.write_result(
        {
            "passed": all(gates.values()),
            "gates": gates,
            "formal_hashes": _hash_snapshot(),
            "catalog_scenario_ids": [item.scenario_id for item in scenarios],
            "factor_path_allowlist": factor_allowlist,
            "factor_path_allowlist_sha256": sha256_file(
                FACTOR_PATH_ALLOWLIST_PATH
            ),
            "run_manifest_identity_hash": manifest["manifest_identity_hash"],
            "run_identity_hash": identity["identity_hash"],
            "definitions_hash": manifest["definitions_hash"],
            "worker_argv": list(sys.argv),
            "worker_environment": {
                key: os.environ.get(key)
                for key in (
                    "PYTHONNOUSERSITE",
                    "PYTHONDONTWRITEBYTECODE",
                    "PYTHONPYCACHEPREFIX",
                )
            },
        }
    )


def _run_g01(runner: StageRunner) -> dict[str, Any]:
    reset_cache = FROZEN_REPLAY_SOURCE["reset_cache"].absolute_path()
    zero = [0.0] * 6
    source = runner.isaac("replay_source", "replay-source")
    source_result = load_result(source / "result.json")
    actions = source / str(source_result["action_sequence_file"])
    first_action = [float(value) for value in source_result["first_action"]]

    executions: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, str]] = {}

    def register(engine: str, label: str, path: Path) -> None:
        result_path = path / "result.json"
        result = load_result(result_path)
        trace_path = path / "trace"
        executions.append(
            {
                "result": result,
                "result_sha256": sha256_file(result_path),
                "trace_path": trace_path,
            }
        )
        artifacts[f"{engine}:{label}"] = {
            "relative_path": path.relative_to(runner.output).as_posix(),
            "result_sha256": sha256_file(result_path),
            "trace_sha256": str(result["trace_sha256"]),
        }

    robot_cases = (
        ("p10_a", 5, None, None),
        ("p10_b", 5, None, None),
        ("p10_c", 5, None, None),
        ("p30_open_direct", 5, None, None),
        ("p30_open_target", 5, None, None),
        ("p30_closed_direct", 5, None, None),
        ("p30_closed_target", 5, None, None),
        ("p40_on", 5, None, None),
        ("p40_off", 5, None, None),
        ("p60_b", 5, zero, None),
        ("p60_c", 5, first_action, None),
        ("p60_c", 20, zero, "reset_nominal_zero"),
    )
    for scenario, control_ticks, vector, family_override in robot_cases:
        label = (
            "nominal_zero_action"
            if family_override == "reset_nominal_zero"
            else scenario
        )
        common: list[object] = [
            "--scenario",
            scenario,
            "--control-ticks",
            control_ticks,
            "--repetitions",
            3,
        ]
        if vector is not None:
            common.extend(("--input-vector", *vector))
        if family_override is not None:
            common.extend(("--repeatability-family", family_override))
        isaac_arguments = list(common)
        if scenario.startswith("p60"):
            isaac_arguments.extend(("--reset-cache", reset_cache))
        isaac_path = runner.isaac(label, "robot-probe", *isaac_arguments)
        mujoco_path = runner.mujoco(label, "robot-batch", *common)
        register("isaac", label, isaac_path)
        register("mujoco", label, mujoco_path)

    for label, mode, friction, duration in (
        ("impact", "impact", 0.0, 0.5),
        ("slide_zero", "slide", 0.0, 0.4),
        ("slide_nominal", "slide", 1.0, 0.4),
    ):
        isaac_path = runner.isaac(
            label,
            "sphere-probe",
            "--sphere-mode",
            mode,
            "--friction",
            friction,
            "--repetitions",
            3,
            "--duration",
            duration,
        )
        mujoco_path = runner.mujoco(
            label,
            "sphere-batch",
            "--mode",
            mode,
            "--friction",
            friction,
            "--repetitions",
            3,
            "--duration",
            duration,
        )
        register("isaac", label, isaac_path)
        register("mujoco", label, mujoco_path)

    fresh_replay_equivalence: dict[str, Any] = {}
    for repetition in range(3):
        label = f"replay_{repetition}"
        isaac_path = runner.isaac(
            f"{label}_isaac",
            "replay",
            "--actions",
            actions,
            "--source-result",
            source / "result.json",
            "--repetition",
            repetition,
        )
        isaac_result = load_result(isaac_path / "result.json")
        equivalence = isaac_result.get("fresh_replay_equivalence")
        if not isinstance(equivalence, dict) or equivalence.get("passed") is not True:
            raise RuntimeError(
                f"G01 fresh Isaac replay equivalence failed: repetition={repetition}"
            )
        fresh_replay_equivalence[str(repetition)] = equivalence
        mujoco_path = runner.mujoco(
            f"{label}_mujoco",
            "replay",
            "--actions",
            actions,
            "--source-result",
            source / "result.json",
            "--command",
            *FROZEN_REPLAY_SOURCE["command"],
            "--repetition",
            repetition,
        )
        register("isaac", label, isaac_path)
        register("mujoco", label, mujoco_path)

    threshold_snapshot = build_threshold_snapshot(executions)
    validate_threshold_snapshot(threshold_snapshot)
    snapshot_path = runner.output / "threshold_snapshot.json"
    _write_json(snapshot_path, threshold_snapshot)

    expected_family_key_counts = {
        "reset_nominal_zero": 1,
        "p10_rest_closure_on": 1,
        "p10_rest_closure_off": 1,
        "p10_freefall_closure_on": 1,
        "p30_open_direct": 18,
        "p30_open_target": 18,
        "p30_closed_direct": 18,
        "p30_closed_target": 18,
        "p40_closure_on_pulse": 3,
        "p40_closure_off_pulse": 3,
        "p50_sphere_impact": 6,
        "p50_sphere_slide_zero_friction": 3,
        "p50_sphere_slide_nominal_friction": 3,
        "p60_zero_action_drive_off": 1,
        "p60_first_action_formal_drive": 1,
        "p60_open_loop_replay": 1,
    }
    actual_family_key_counts: dict[str, dict[str, int]] = {
        engine: {} for engine in ("isaac", "mujoco")
    }
    for frozen in threshold_snapshot["keys"].values():
        engine = str(frozen["engine"])
        family = str(frozen["repeatability_family"])
        actual_family_key_counts[engine][family] = (
            actual_family_key_counts[engine].get(family, 0) + 1
        )
    exact_coverage = all(
        actual_family_key_counts[engine] == expected_family_key_counts
        for engine in ("isaac", "mujoco")
    )
    repeatability_usable = all(
        bool(record["usable"])
        for record in threshold_snapshot["keys"].values()
    )
    return runner.write_result(
        {
            "passed": exact_coverage,
            "repeatability_usable": repeatability_usable,
            "exact_executed_coverage": exact_coverage,
            "expected_family_key_counts_per_engine": expected_family_key_counts,
            "actual_family_key_counts_per_engine": actual_family_key_counts,
            "minimum_repetitions": 3,
            "threshold_snapshot": threshold_snapshot,
            "threshold_snapshot_file": snapshot_path.name,
            "threshold_snapshot_sha256": sha256_file(snapshot_path),
            "executed_key_count": len(threshold_snapshot["keys"]),
            "executed_sample_count": sum(
                int(record["sample_count"])
                for record in threshold_snapshot["keys"].values()
            ),
            "probe_artifacts": artifacts,
            "replay_source": {
                "relative_path": source.relative_to(runner.output).as_posix(),
                "result_sha256": sha256_file(source / "result.json"),
                "actions_file": actions.name,
                "actions_sha256": sha256_file(actions),
                "replay_source_identity_hash": source_result[
                    "replay_source_identity_hash"
                ],
                "first_action": first_action,
            },
            "fresh_replay_equivalence": fresh_replay_equivalence,
        }
    )


def _run_g02(runner: StageRunner) -> dict[str, Any]:
    import torch

    from debug.sim2sim.dreamwaq_debug_contract import DebugPolicyAdapter

    pulse_amplitude = 1.0e-3
    isaac_path = runner.isaac(
        "adapter_probe",
        "adapter-probe",
        "--pulse-amplitude",
        pulse_amplitude,
    )
    mujoco_path = runner.mujoco(
        "adapter_probe",
        "adapter-probe",
        "--pulse-amplitude",
        pulse_amplitude,
    )
    isaac_result_path = isaac_path / "result.json"
    mujoco_result_path = mujoco_path / "result.json"
    isaac_result = load_result(isaac_result_path)
    mujoco_result = load_result(mujoco_result_path)
    isaac_current = np.asarray(
        isaac_result["reset_phases"]["returned_policy"]["actor_obs_current"],
        dtype=np.float32,
    )
    mujoco_current = np.asarray(
        mujoco_result["reset_phases"]["returned_policy"]["actor_obs_current"],
        dtype=np.float32,
    )
    isaac_history = np.asarray(
        isaac_result["reset_phases"]["returned_policy"]["policy_input"],
        dtype=np.float32,
    )
    mujoco_history = np.asarray(
        mujoco_result["reset_phases"]["returned_policy"]["policy_input"],
        dtype=np.float32,
    )
    if isaac_current.shape != (25,) or mujoco_current.shape != (25,):
        raise RuntimeError("G02 worker ActorObsV1 evidence is not 25D")
    if isaac_history.shape != (125,) or mujoco_history.shape != (125,):
        raise RuntimeError("G02 worker DreamWaQ history evidence is not 125D")

    records: dict[str, Any] = {}
    for name, policy in FROZEN_POLICIES.items():
        adapter = DebugPolicyAdapter(
            actor_path=policy["actor"].absolute_path(),
            manifest_path=policy["manifest"].absolute_path(),
            model_manifest_path=MODEL_MANIFEST,
            load_mujoco_contract=False,
        )
        independently_loaded = DebugPolicyAdapter(
            actor_path=policy["actor"].absolute_path(),
            manifest_path=policy["manifest"].absolute_path(),
            model_manifest_path=MODEL_MANIFEST,
            load_mujoco_contract=False,
        )
        anchor_inputs = (
            {"isaac": isaac_history, "mujoco": mujoco_history}
            if adapter.policy_kind == "dreamwaq"
            else {"isaac": isaac_current, "mujoco": mujoco_current}
        )
        anchor_records: dict[str, Any] = {}
        maximum_abs_error = 0.0
        for anchor_name, policy_input in anchor_inputs.items():
            first_output = adapter.infer(policy_input.copy())
            second_output = independently_loaded.infer(policy_input.copy())
            field_errors = {
                "raw_action": float(
                    np.max(
                        np.abs(first_output.raw_action - second_output.raw_action),
                        initial=0.0,
                    )
                )
            }
            if adapter.policy_kind == "dreamwaq":
                field_errors.update(
                    {
                        "estimated_velocity": float(
                            np.max(
                                np.abs(
                                    first_output.estimated_velocity
                                    - second_output.estimated_velocity
                                ),
                                initial=0.0,
                            )
                        ),
                        "context_mu": float(
                            np.max(
                                np.abs(
                                    first_output.context_mu
                                    - second_output.context_mu
                                ),
                                initial=0.0,
                            )
                        ),
                        "context_logvar": float(
                            np.max(
                                np.abs(
                                    first_output.context_logvar
                                    - second_output.context_logvar
                                ),
                                initial=0.0,
                            )
                        ),
                    }
                )
            maximum_abs_error = max(maximum_abs_error, *field_errors.values())
            anchor_records[anchor_name] = {
                "policy_input_sha256": stable_hash(policy_input.tolist()),
                "field_max_abs_errors": field_errors,
            }
        record: dict[str, Any] = {
            "policy_kind": adapter.policy_kind,
            "input_dimension": adapter.input_dimension,
            "maximum_abs_error": maximum_abs_error,
            "anchors": anchor_records,
        }
        if adapter.policy_kind == "dreamwaq":
            golden_path = policy["manifest"].absolute_path().parent / "golden_vectors.pt"
            manifest = json.loads(policy["manifest"].absolute_path().read_text(encoding="utf-8"))
            if sha256_file(golden_path) != manifest["golden_vectors_sha256"]:
                raise ValueError("DreamWaQ golden-vector hash drifted")
            golden = torch.load(golden_path, map_location="cpu", weights_only=False)
            with torch.inference_mode():
                actual = adapter.actor(golden["history"]).detach().cpu()
            record["golden_vector_max_abs"] = float(
                torch.max(torch.abs(actual - golden["expected_action_mean"])).item()
            )
        records[name] = record
    adapter_gate = evaluate_adapter_gate(
        isaac_result,
        mujoco_result,
        same_input_actor_checks=records,
    )
    return runner.write_result(
        {
            "passed": True,
            "diagnostic_gate_passed": bool(adapter_gate["digital_chain_passed"]),
            "terminal_adapter_bug": bool(adapter_gate["terminal_adapter_bug"]),
            "gates": adapter_gate["gates"],
            "adapter_gate": adapter_gate,
            "same_input_actor_checks": records,
            "evidence": {
                "isaac": {
                    "relative_path": isaac_path.relative_to(runner.output).as_posix(),
                    "result_sha256": sha256_file(isaac_result_path),
                    "worker_source_sha256": isaac_result["worker_source_sha256"],
                },
                "mujoco": {
                    "relative_path": mujoco_path.relative_to(runner.output).as_posix(),
                    "result_sha256": sha256_file(mujoco_result_path),
                    "worker_source_sha256": mujoco_result["worker_source_sha256"],
                },
            },
        }
    )


def _run_g03(runner: StageRunner) -> dict[str, Any]:
    repetitions = 5
    ticks = 10
    guard_path = runner.guard_semantics()
    guard_result = json.loads(
        (guard_path / "result.json").read_text(encoding="utf-8")
    )
    isaac_paths: dict[str, Path] = {}
    contact_error: dict[str, str] | None = None
    for mode in ("formal", "debug", "system_observer"):
        isaac_paths[mode] = runner.isaac(
            mode,
            "instrumentation-probe",
            "--instrumentation-mode",
            mode,
            "--instrumentation-repetitions",
            repetitions,
            "--instrumentation-ticks",
            ticks,
        )
    try:
        isaac_paths["contact"] = runner.isaac(
            "contact",
            "instrumentation-probe",
            "--instrumentation-mode",
            "contact",
            "--instrumentation-repetitions",
            repetitions,
            "--instrumentation-ticks",
            ticks,
        )
    except Exception as error:
        contact_error = {"type": type(error).__name__, "message": str(error)}

    mujoco_path = runner.mujoco(
        "observer",
        "instrumentation",
        "--repetitions",
        repetitions,
        "--ticks",
        ticks,
    )
    mujoco_result = json.loads(
        (mujoco_path / "result.json").read_text(encoding="utf-8")
    )

    float_floors = {
        "time_s": 1.0e-12,
        "actor_observation": 1.0e-6,
        "controlled_position_canonical": 1.0e-7,
        "controlled_velocity_canonical": 1.0e-6,
        "base_linear_velocity_control": 1.0e-6,
        "base_angular_velocity_control": 1.0e-6,
    }
    discrete_fields = (
        "repetition",
        "control_tick",
        "episode_length",
        "common_step_counter",
        "sim_step_counter",
        "terminated",
        "truncated",
    )

    def load_isaac(mode: str) -> tuple[dict[str, Any], Any]:
        path = isaac_paths[mode]
        payload = json.loads(
            (path / "instrumentation.json").read_text(encoding="utf-8")
        )
        if payload.get("mode") != mode:
            raise RuntimeError(f"G03 Isaac mode identity drifted: {mode}")
        trace = load_verified_trace(path / "trace")
        expected_rows = repetitions * (ticks + 1)
        if trace.identity.row_count != expected_rows:
            raise RuntimeError(
                f"G03 {mode} trace has {trace.identity.row_count} rows, expected {expected_rows}"
            )
        return payload, trace

    loaded = {mode: load_isaac(mode) for mode in isaac_paths}
    reference_payload, reference_trace = loaded["formal"]

    def repeat_envelope(values: np.ndarray) -> float:
        shaped = np.asarray(values).reshape(
            repetitions, ticks + 1, *values.shape[1:]
        )
        return max(
            float(np.max(np.abs(shaped[left] - shaped[right])))
            for left in range(repetitions)
            for right in range(left + 1, repetitions)
        )

    def compare_mode(mode: str) -> dict[str, Any]:
        payload, trace = loaded[mode]
        if set(trace.arrays) != set(reference_trace.arrays):
            raise RuntimeError(f"G03 trace schema differs for {mode}")
        continuous: dict[str, dict[str, Any]] = {}
        for field, floor in float_floors.items():
            reference = np.asarray(reference_trace.arrays[field], dtype=np.float64)
            candidate = np.asarray(trace.arrays[field], dtype=np.float64)
            reference_envelope = repeat_envelope(reference)
            candidate_envelope = repeat_envelope(candidate)
            error = float(np.max(np.abs(reference - candidate)))
            bitwise_equal = bool(np.array_equal(reference, candidate))
            continuous[field] = {
                "maximum_absolute_error": error,
                "formal_repeat_envelope": reference_envelope,
                "candidate_repeat_envelope": candidate_envelope,
                "reporting_floor": floor,
                "bitwise_equal": bitwise_equal,
                "passed": bitwise_equal,
            }
        discrete = {
            field: bool(
                np.array_equal(
                    np.asarray(reference_trace.arrays[field]),
                    np.asarray(trace.arrays[field]),
                )
            )
            for field in discrete_fields
        }
        compiled_properties_equal = bool(
            payload.get("compiled_properties_hash")
            == reference_payload.get("compiled_properties_hash")
        )
        formal_config_unchanged = bool(
            payload.get("formal_config_hash_before")
            == payload.get("formal_config_hash_after")
            == reference_payload.get("formal_config_hash_before")
        )
        return {
            "mode": mode,
            "continuous": continuous,
            "discrete": discrete,
            "compiled_properties_equal": compiled_properties_equal,
            "formal_config_unchanged": formal_config_unchanged,
            "read_rng_unchanged": bool(payload.get("read_rng_unchanged")),
            "passed": bool(
                all(item["passed"] for item in continuous.values())
                and all(discrete.values())
                and compiled_properties_equal
                and formal_config_unchanged
                and payload.get("read_rng_unchanged") is True
            ),
        }

    comparisons = {
        mode: compare_mode(mode)
        for mode in ("debug", "system_observer", "contact")
        if mode in loaded
    }

    counter_gates: dict[str, dict[str, bool]] = {}
    for mode, (_, trace) in loaded.items():
        episode = np.asarray(trace.arrays["episode_length"]).reshape(
            repetitions, ticks + 1
        )
        common = np.asarray(trace.arrays["common_step_counter"]).reshape(
            repetitions, ticks + 1
        )
        sim = np.asarray(trace.arrays["sim_step_counter"]).reshape(
            repetitions, ticks + 1
        )
        counter_gates[mode] = {
            "episode_ticks_exact": bool(
                np.array_equal(
                    episode,
                    np.tile(np.arange(ticks + 1), (repetitions, 1)),
                )
            ),
            "one_common_step_per_action": bool(
                np.all(np.diff(common, axis=1) == 1)
            ),
            "four_physics_steps_per_action": bool(
                np.all(np.diff(sim, axis=1) == 4)
            ),
            "no_terminal_or_truncation": bool(
                not np.any(trace.arrays["terminated"])
                and not np.any(trace.arrays["truncated"])
            ),
        }

    def bootstrap_gate(path: Path) -> dict[str, Any]:
        evidence_path = path / "bootstrap_guard.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        guard = evidence["write_guard"]
        handler_paths = [
            Path(row["baseFilename"]).resolve()
            for key in (
                "pre_handlers",
                "created_lifetime_handlers",
                "post_handlers",
            )
            for row in guard.get(key, [])
        ]
        handlers_inside_run = all(
            path == runner.run_root or runner.run_root in path.parents
            for path in handler_paths
        )
        capability = guard.get("capability_manifest", {})
        required_apis = {
            "mkdir",
            "remove",
            "rmdir",
            "rename",
            "link",
            "symlink",
            "truncate",
            "chmod",
            "chown",
            "utime",
        }
        capability_complete = bool(
            set(capability.get("apis", {})) == required_apis
            and set(capability.get("dir_fd", {})) == required_apis
            and set(capability.get("flags", {}))
            == {
                "O_WRONLY",
                "O_RDWR",
                "O_CREAT",
                "O_TRUNC",
                "O_APPEND",
                "O_EXCL",
                "O_TMPFILE",
                "O_TEMPORARY",
            }
        )
        passed = bool(
            guard.get("installed") is True
            and guard.get("probe_count") == 1
            and handlers_inside_run
            and capability_complete
        )
        return {
            "passed": passed,
            "evidence_file": str(evidence_path),
            "evidence_sha256": sha256_file(evidence_path),
            "handlers_inside_run": handlers_inside_run,
            "handler_count": len(handler_paths),
            "capability_complete": capability_complete,
            "ledger_event_count": len(guard.get("ledger", [])),
        }

    bootstrap_gates = {
        f"isaac_{mode}": bootstrap_gate(path)
        for mode, path in isaac_paths.items()
    }
    bootstrap_gates["mujoco_observer"] = bootstrap_gate(mujoco_path)
    bootstrap_gates["python_guard_semantics"] = bootstrap_gate(guard_path)

    mandatory_passed = bool(
        all(
            comparisons[mode]["passed"]
            for mode in ("debug", "system_observer")
        )
        and all(
            all(counter_gates[mode].values())
            for mode in ("formal", "debug", "system_observer")
        )
        and all(item["passed"] for item in bootstrap_gates.values())
        and guard_result.get("passed") is True
        and mujoco_result.get("passed") is True
        and reference_payload.get("read_rng_unchanged") is True
        and reference_payload.get("formal_config_hash_before")
        == reference_payload.get("formal_config_hash_after")
    )
    contact_observer_usable = bool(
        contact_error is None
        and comparisons.get("contact", {}).get("passed") is True
        and all(counter_gates.get("contact", {}).values())
    )
    return runner.write_result(
        {
            "passed": mandatory_passed,
            "invalid_evidence_pipeline": not mandatory_passed,
            "contact_observer_usable": contact_observer_usable,
            "contact_observer_error": contact_error,
            "isaac_comparisons": comparisons,
            "isaac_counter_gates": counter_gates,
            "mujoco_result": mujoco_result,
            "python_guard_semantics": guard_result,
            "bootstrap_gates": bootstrap_gates,
            "probe_parameters": {
                "fresh_resets": repetitions,
                "control_ticks_per_reset": ticks,
                "float_tolerance_rule": (
                    "bitwise_equal; repeat_envelopes_are_diagnostic_only"
                ),
                "float_material_floors": float_floors,
            },
            "result_files": {
                **{
                    f"isaac_{mode}": str(path / "instrumentation.json")
                    for mode, path in isaac_paths.items()
                },
                "mujoco": str(mujoco_path / "result.json"),
                "python_guard_semantics": str(guard_path / "result.json"),
            },
        }
    )


def _g01_evidence(
    runner: StageRunner,
) -> tuple[Path, dict[str, Any], dict[str, Any], str]:
    worker, stage_state_hash = _selected_stage_worker(
        runner.run_root, "G01_repeatability"
    )
    result = load_result(worker / "result.json")
    if result.get("exact_executed_coverage") is not True:
        raise RuntimeError("Selected G01 evidence has incomplete exact-key coverage")
    snapshot_name = result.get("threshold_snapshot_file")
    if not isinstance(snapshot_name, str) or Path(snapshot_name).name != snapshot_name:
        raise RuntimeError("Selected G01 threshold snapshot path is invalid")
    snapshot_path = require_path_within(
        worker / snapshot_name, worker, label="G01 threshold snapshot"
    ).resolve(strict=True)
    if sha256_file(snapshot_path) != result.get("threshold_snapshot_sha256"):
        raise RuntimeError("Selected G01 threshold snapshot hash is invalid")
    snapshot = load_result(snapshot_path)
    validate_threshold_snapshot(snapshot)
    embedded = result.get("threshold_snapshot")
    if not isinstance(embedded, dict) or embedded.get("identity_hash") != snapshot.get(
        "identity_hash"
    ):
        raise RuntimeError("Selected G01 embedded threshold identity is invalid")
    return worker, result, snapshot, stage_state_hash


def _g01_probe(
    worker: Path,
    g01_result: dict[str, Any],
    *,
    engine: str,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    artifacts = g01_result.get("probe_artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("Selected G01 probe artifact index is missing")
    record = artifacts.get(f"{engine}:{label}")
    if not isinstance(record, dict):
        raise RuntimeError(f"Selected G01 probe is missing: {engine}:{label}")
    relative = record.get("relative_path")
    if not isinstance(relative, str):
        raise RuntimeError("Selected G01 probe relative path is invalid")
    path = require_path_within(
        worker / Path(relative), worker, label=f"G01 {engine}:{label}"
    ).resolve(strict=True)
    result_path = path / "result.json"
    if sha256_file(result_path) != record.get("result_sha256"):
        raise RuntimeError(f"Selected G01 result hash drifted: {engine}:{label}")
    result = load_result(result_path)
    trace = load_verified_trace(path / "trace")
    if (
        result.get("trace_sha256") != trace.identity.trace_sha256
        or record.get("trace_sha256") != trace.identity.trace_sha256
    ):
        raise RuntimeError(f"Selected G01 trace hash drifted: {engine}:{label}")
    return path, result


def _g01_replay_source(
    worker: Path, g01_result: dict[str, Any]
) -> tuple[Path, dict[str, Any], Path]:
    record = g01_result.get("replay_source")
    if not isinstance(record, dict) or not isinstance(
        record.get("relative_path"), str
    ):
        raise RuntimeError("Selected G01 replay source index is invalid")
    path = require_path_within(
        worker / Path(record["relative_path"]),
        worker,
        label="G01 replay source",
    ).resolve(strict=True)
    result_path = path / "result.json"
    if sha256_file(result_path) != record.get("result_sha256"):
        raise RuntimeError("Selected G01 replay source result hash drifted")
    result = load_result(result_path)
    actions_file = record.get("actions_file")
    if not isinstance(actions_file, str) or Path(actions_file).name != actions_file:
        raise RuntimeError("Selected G01 replay action path is invalid")
    actions = (path / actions_file).resolve(strict=True)
    if sha256_file(actions) != record.get("actions_sha256"):
        raise RuntimeError("Selected G01 replay action hash drifted")
    if result.get("replay_source_identity_hash") != record.get(
        "replay_source_identity_hash"
    ):
        raise RuntimeError("Selected G01 replay source identity drifted")
    return path, result, actions


def _contact_exclusion_summary(
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, result in sorted(results.items()):
        validate_result_identity(result)
        semantics = result["configuration_semantics"]
        environment = semantics.get("environment")
        contact = semantics.get("contact")
        proof = contact.get("contact_exclusion") if isinstance(contact, dict) else None
        ground_absent = bool(
            isinstance(environment, dict)
            and environment.get("ground_enabled") is False
        )
        contact_free = bool(
            isinstance(proof, dict)
            and proof.get("contact_free") is True
            and ground_absent
        )
        records[name] = {
            "engine": result["engine"],
            "scenario_id": result["scenario_id"],
            "configuration_semantics_hash": result[
                "configuration_semantics_hash"
            ],
            "ground_absent": ground_absent,
            "contact_free": contact_free,
            "proof": proof,
        }
    return {
        "passed": bool(records) and all(
            record["contact_free"] for record in records.values()
        ),
        "records": records,
    }


def _run_p10(runner: StageRunner) -> dict[str, Any]:
    g01_worker, g01_result, snapshot, g01_state_hash = _g01_evidence(runner)
    analyses: dict[str, Any] = {}
    results: dict[str, dict[str, Any]] = {}
    for scenario in ("p10_a", "p10_b", "p10_c"):
        isaac, isaac_result = _g01_probe(
            g01_worker, g01_result, engine="isaac", label=scenario
        )
        mujoco, mujoco_result = _g01_probe(
            g01_worker, g01_result, engine="mujoco", label=scenario
        )
        results[f"isaac_{scenario}"] = isaac_result
        results[f"mujoco_{scenario}"] = mujoco_result
        analysis = compare_p10_trace_directories(
            isaac / "trace",
            mujoco / "trace",
            repeatability_snapshot=snapshot,
            isaac_result=isaac_result,
            mujoco_result=mujoco_result,
        )
        analyses[scenario] = analysis
        _write_json(runner.output / "analysis" / f"{scenario}.json", analysis)
    return runner.write_result(
        {
            "passed": True,
            "g01_stage_state_sha256": g01_state_hash,
            "threshold_snapshot_identity_hash": snapshot["identity_hash"],
            "analyses": analyses,
            "contact_exclusion": _contact_exclusion_summary(results),
        }
    )


def _run_p20(runner: StageRunner) -> dict[str, Any]:
    paths = {
        "isaac_properties": runner.isaac("properties", "properties"),
        "isaac_golden": runner.isaac("golden", "golden"),
        "mujoco_properties": runner.mujoco("properties", "properties"),
        "mujoco_golden": runner.mujoco("golden", "golden"),
    }
    payloads = {
        "isaac_properties": json.loads((paths["isaac_properties"] / "properties.json").read_text(encoding="utf-8")),
        "isaac_golden": json.loads((paths["isaac_golden"] / "golden.json").read_text(encoding="utf-8")),
        "mujoco_properties": json.loads((paths["mujoco_properties"] / "properties.json").read_text(encoding="utf-8")),
        "mujoco_golden": json.loads((paths["mujoco_golden"] / "golden.json").read_text(encoding="utf-8")),
    }
    property_audits = (
        payloads["isaac_properties"],
        payloads["mujoco_properties"],
    )
    golden_audits = (payloads["isaac_golden"], payloads["mujoco_golden"])
    audit_valid = all(bool(payload.get("audit_valid")) for payload in property_audits)
    golden_valid = all(bool(payload.get("passed")) for payload in golden_audits)
    mismatch_detected = any(
        bool(payload.get("mismatch_detected")) for payload in property_audits
    )
    return runner.write_result(
        {
            "passed": audit_valid and golden_valid,
            "audit_valid": audit_valid,
            "golden_valid": golden_valid,
            "mismatch_detected": mismatch_detected,
            "audits": payloads,
        }
    )


def _run_p30(runner: StageRunner) -> dict[str, Any]:
    g01_worker, g01_result, snapshot, g01_state_hash = _g01_evidence(runner)
    analyses: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    results: dict[str, dict[str, Any]] = {}
    for scenario in (
        "p30_open_direct",
        "p30_open_target",
        "p30_closed_direct",
        "p30_closed_target",
    ):
        isaac, isaac_result = _g01_probe(
            g01_worker, g01_result, engine="isaac", label=scenario
        )
        mujoco, mujoco_result = _g01_probe(
            g01_worker, g01_result, engine="mujoco", label=scenario
        )
        paths[f"isaac_{scenario}"] = isaac
        paths[f"mujoco_{scenario}"] = mujoco
        results[f"isaac_{scenario}"] = isaac_result
        results[f"mujoco_{scenario}"] = mujoco_result
        analysis = compare_p30_trace_directories(
            isaac / "trace",
            mujoco / "trace",
            repeatability_snapshot=snapshot,
            isaac_result=isaac_result,
            mujoco_result=mujoco_result,
        )
        analyses[scenario] = analysis
        _write_json(runner.output / "analysis" / f"{scenario}.json", analysis)
    causal_pairs: dict[str, Any] = {}
    for engine in ("isaac", "mujoco"):
        for label, pair_id in (
            ("open", "P30_OPEN_TARGET_TO_DIRECT"),
            ("closed", "P30_CLOSED_TARGET_TO_DIRECT"),
        ):
            causal_pairs[f"{engine}_{label}"] = validate_ratio_pair(
                results[f"{engine}_p30_{label}_target"],
                results[f"{engine}_p30_{label}_direct"],
                pair_id=pair_id,
            )
    _write_json(runner.output / "analysis" / "causal_pairs.json", causal_pairs)
    return runner.write_result(
        {
            "passed": True,
            "g01_stage_state_sha256": g01_state_hash,
            "threshold_snapshot_identity_hash": snapshot["identity_hash"],
            "analyses": analyses,
            "causal_pairs": causal_pairs,
            "contact_exclusion": _contact_exclusion_summary(results),
        }
    )


def _run_p40(runner: StageRunner) -> dict[str, Any]:
    g01_worker, g01_result, snapshot, g01_state_hash = _g01_evidence(runner)
    paths: dict[str, Path] = {}
    results: dict[str, dict[str, Any]] = {}
    for scenario in ("p40_on", "p40_off"):
        paths[f"isaac_{scenario}"], results[f"isaac_{scenario}"] = _g01_probe(
            g01_worker, g01_result, engine="isaac", label=scenario
        )
        paths[f"mujoco_{scenario}"], results[f"mujoco_{scenario}"] = _g01_probe(
            g01_worker, g01_result, engine="mujoco", label=scenario
        )
    analysis = compare_p40_trace_directories(
        paths["isaac_p40_on"] / "trace",
        paths["isaac_p40_off"] / "trace",
        paths["mujoco_p40_on"] / "trace",
        paths["mujoco_p40_off"] / "trace",
        repeatability_snapshot=snapshot,
        results={
            "isaac_on": results["isaac_p40_on"],
            "isaac_off": results["isaac_p40_off"],
            "mujoco_on": results["mujoco_p40_on"],
            "mujoco_off": results["mujoco_p40_off"],
        },
    )
    causal_pairs = {
        engine: validate_ratio_pair(
            results[f"{engine}_p40_on"],
            results[f"{engine}_p40_off"],
            pair_id="P40_CLOSURE_ON_TO_OFF",
        )
        for engine in ("isaac", "mujoco")
    }
    _write_json(runner.output / "analysis" / "p40.json", analysis)
    _write_json(runner.output / "analysis" / "causal_pairs.json", causal_pairs)
    return runner.write_result(
        {
            "passed": True,
            "g01_stage_state_sha256": g01_state_hash,
            "threshold_snapshot_identity_hash": snapshot["identity_hash"],
            "analysis": analysis,
            "causal_pairs": causal_pairs,
            "contact_exclusion": _contact_exclusion_summary(results),
        }
    )


def p50_normal_gate_decision(
    *,
    impact_evidence_valid: bool,
    raw_normal_primary_supported: bool,
    normal_impulse_evidence_valid: bool,
    normal_impulse_material: bool,
) -> dict[str, bool]:
    impact_branch = bool(impact_evidence_valid and not raw_normal_primary_supported)
    impulse_branch = bool(
        normal_impulse_evidence_valid and not normal_impulse_material
    )
    return {
        "impact_evidence_valid": bool(impact_evidence_valid),
        "raw_normal_not_primary": not bool(raw_normal_primary_supported),
        "normal_impulse_evidence_valid": bool(normal_impulse_evidence_valid),
        "normal_impulse_within_tolerance": not bool(normal_impulse_material),
        "impact_branch_passed": impact_branch,
        "impulse_branch_passed": impulse_branch,
        "passed": bool(impact_branch or impulse_branch),
    }


def _run_p50(runner: StageRunner) -> dict[str, Any]:
    g01_worker, g01_result, snapshot, g01_state_hash = _g01_evidence(runner)
    g03_worker, g03_state_hash = _selected_stage_worker(
        runner.run_root, "G03_instrumentation"
    )
    g03_result = json.loads(
        (g03_worker / "result.json").read_text(encoding="utf-8")
    )
    contact_instrumentation_gate = bool(
        g03_result.get("contact_observer_usable") is True
    )
    isaac_impact, isaac_impact_result = _g01_probe(
        g01_worker, g01_result, engine="isaac", label="impact"
    )
    mujoco_impact, mujoco_impact_result = _g01_probe(
        g01_worker, g01_result, engine="mujoco", label="impact"
    )
    slide: dict[str, Path] = {}
    slide_results_flat: dict[str, dict[str, Any]] = {}
    for label in ("zero", "nominal"):
        artifact_label = f"slide_{label}"
        slide[f"isaac_{label}"], slide_results_flat[f"isaac_{label}"] = _g01_probe(
            g01_worker,
            g01_result,
            engine="isaac",
            label=artifact_label,
        )
        slide[f"mujoco_{label}"], slide_results_flat[f"mujoco_{label}"] = _g01_probe(
            g01_worker,
            g01_result,
            engine="mujoco",
            label=artifact_label,
        )
    impact = compare_p50_impact_trace_directories(
        isaac_impact / "trace",
        mujoco_impact / "trace",
        repeatability_snapshot=snapshot,
        isaac_result=isaac_impact_result,
        mujoco_result=mujoco_impact_result,
    )
    slide_analysis = compare_p50_slide_trace_directories(
        slide["isaac_nominal"] / "trace",
        slide["isaac_zero"] / "trace",
        slide["mujoco_nominal"] / "trace",
        slide["mujoco_zero"] / "trace",
        repeatability_snapshot=snapshot,
        results=slide_results_flat,
    )
    impact_results = {
        "isaac": isaac_impact_result,
        "mujoco": mujoco_impact_result,
    }
    slide_results = {
        engine: {
            label: slide_results_flat[f"{engine}_{label}"]
            for label in ("nominal", "zero")
        }
        for engine in ("isaac", "mujoco")
    }
    causal_pairs = {
        engine: validate_ratio_pair(
            slide_results[engine]["nominal"],
            slide_results[engine]["zero"],
            pair_id="P50_NOMINAL_TO_ZERO_FRICTION",
        )
        for engine in ("isaac", "mujoco")
    }

    impact_traces = {
        "isaac": load_verified_trace(isaac_impact / "trace"),
        "mujoco": load_verified_trace(mujoco_impact / "trace"),
    }
    property_gate = all(
        bool(payload.get("compiled_property_passed"))
        for payload in impact_results.values()
    )
    clearance_gate = all(
        min(float(value) for value in payload.get("initial_clearance_m", [])) > 0.0
        for payload in impact_results.values()
    )
    t0_contact_gate = all(
        np.count_nonzero(np.asarray(trace.arrays["contact_count"])[0]) == 0
        and float(np.max(np.abs(np.asarray(trace.arrays["normal_force_n"])[0])))
        <= 1.0e-9
        for trace in impact_traces.values()
    )
    valid_contact_gate = all(
        int(payload.get("valid_contact_count", -1))
        == int(payload.get("profile_count", -2))
        for payload in impact_results.values()
    )
    freefall_and_event_gate = bool(
        impact["valid_condition_count"] == 6
        and all(
            condition["valid"]
            and not bool(condition["precontact"]["material"])
            and bool(condition["contact_event_guard"])
            for condition in impact["conditions"]
        )
    )
    impact_gate_status = {
        "contact_instrumentation_neutral": contact_instrumentation_gate,
        "coupon_property": property_gate,
        "positive_initial_clearance": clearance_gate,
        "filtered_t0_contact_absent": t0_contact_gate,
        "all_profiles_contact": valid_contact_gate,
        "precontact_freefall_and_event_alignment": freefall_and_event_gate,
    }
    impact["raw_normal_primary_supported"] = bool(
        impact["normal_primary_supported"]
    )
    impact["gate_status"] = impact_gate_status
    impact["normal_primary_supported"] = bool(
        impact["raw_normal_primary_supported"]
        and all(impact_gate_status.values())
    )

    nominal_traces = {
        "isaac": load_verified_trace(slide["isaac_nominal"] / "trace"),
        "mujoco": load_verified_trace(slide["mujoco_nominal"] / "trace"),
    }

    def final_impulse_and_repeat(trace: Any) -> tuple[np.ndarray, float]:
        impulse = np.asarray(trace.arrays["normal_impulse_ns"], dtype=np.float64)[-1]
        signs = np.asarray(trace.arrays["profile_sign"])[0]
        envelopes = [
            float(np.ptp(impulse[signs == sign]))
            for sign in (-1, 0, 1)
            if np.count_nonzero(signs == sign) >= 3
        ]
        return impulse, max(envelopes, default=float("inf"))

    isaac_impulse, isaac_repeat = final_impulse_and_repeat(nominal_traces["isaac"])
    mujoco_impulse, mujoco_repeat = final_impulse_and_repeat(nominal_traces["mujoco"])
    isaac_frozen_repeat = frozen_repeat_envelope(
        snapshot,
        slide_results["isaac"]["nominal"]["repeatability_records"],
        "normal_impulse_ns",
    )
    mujoco_frozen_repeat = frozen_repeat_envelope(
        snapshot,
        slide_results["mujoco"]["nominal"]["repeatability_records"],
        "normal_impulse_ns",
    )
    normal_impulse = response_mismatch_metrics(
        isaac_impulse,
        mujoco_impulse,
        left_repeat_envelope=isaac_frozen_repeat,
        right_repeat_envelope=mujoco_frozen_repeat,
        material_floor=0.02,
    )
    normal_impulse["isaac_observed_repeat_envelope"] = isaac_repeat
    normal_impulse["mujoco_observed_repeat_envelope"] = mujoco_repeat
    slide_property_gate = all(
        bool(payload.get("compiled_property_passed"))
        for engine in slide_results.values()
        for payload in engine.values()
    )
    slide_contact_gate = all(
        int(payload.get("valid_contact_count", -1))
        == int(payload.get("profile_count", -2))
        for engine in slide_results.values()
        for payload in engine.values()
    )
    normal_gate_detail = p50_normal_gate_decision(
        impact_evidence_valid=all(impact_gate_status.values()),
        raw_normal_primary_supported=impact["raw_normal_primary_supported"],
        normal_impulse_evidence_valid=bool(
            contact_instrumentation_gate
            and slide_property_gate
            and slide_contact_gate
            and np.isfinite(float(normal_impulse["rmse"]))
            and np.isfinite(float(normal_impulse["effective_tolerance"]))
        ),
        normal_impulse_material=bool(normal_impulse["material"]),
    )
    slide_gate_status = {
        "contact_instrumentation_neutral": contact_instrumentation_gate,
        "coupon_property": slide_property_gate,
        "all_profiles_contact": slide_contact_gate,
        "normal_gate_or_impulse_within_tolerance": normal_gate_detail["passed"],
    }
    slide_analysis["normal_impulse"] = normal_impulse
    slide_analysis["normal_gate_detail"] = normal_gate_detail
    slide_analysis["gate_status"] = slide_gate_status
    slide_analysis["tangential_primary_supported"] = bool(
        slide_analysis["tangential_primary_supported_before_normal_gate"]
        and all(slide_gate_status.values())
    )
    _write_json(runner.output / "analysis" / "impact.json", impact)
    _write_json(runner.output / "analysis" / "slide.json", slide_analysis)
    _write_json(runner.output / "analysis" / "causal_pairs.json", causal_pairs)
    return runner.write_result(
        {
            "passed": True,
            "g01_stage_state_sha256": g01_state_hash,
            "threshold_snapshot_identity_hash": snapshot["identity_hash"],
            "impact": impact,
            "slide": slide_analysis,
            "causal_pairs": causal_pairs,
            "g03_stage_state_sha256": g03_state_hash,
            "contact_instrumentation_usable": contact_instrumentation_gate,
            "raw_t0_contact_evidence": {
                engine: impact_results[engine].get("raw_initial_normal_force_n")
                for engine in ("isaac", "mujoco")
            },
        }
    )


def _run_p60(runner: StageRunner) -> dict[str, Any]:
    g01_worker, g01_result, snapshot, g01_state_hash = _g01_evidence(runner)
    source, source_result, actions = _g01_replay_source(g01_worker, g01_result)
    replay_seal = verified_replay_source_seal(runner.run_root)
    if replay_seal.get("replay_source_identity_hash") != source_result.get(
        "replay_source_identity_hash"
    ):
        raise RuntimeError("P60 replay seal is not bound to selected G01 source")
    fresh, fresh_result = _g01_probe(
        g01_worker, g01_result, engine="isaac", label="replay_0"
    )
    if not fresh_result["fresh_replay_equivalence"]["passed"]:
        raise RuntimeError("Fresh Isaac replay equivalence gate failed")
    first_action = [float(value) for value in source_result["first_action"]]
    robot_paths: dict[str, Path] = {}
    robot_results: dict[str, dict[str, Any]] = {}
    for scenario in ("p60_b", "p60_c"):
        robot_paths[f"isaac_{scenario}"], robot_results[f"isaac_{scenario}"] = _g01_probe(
            g01_worker, g01_result, engine="isaac", label=scenario
        )
        robot_paths[f"mujoco_{scenario}"], robot_results[f"mujoco_{scenario}"] = _g01_probe(
            g01_worker, g01_result, engine="mujoco", label=scenario
        )
    mujoco_replay, mujoco_replay_result = _g01_probe(
        g01_worker, g01_result, engine="mujoco", label="replay_0"
    )
    analyses = {
        "p60_b": compare_p60_robot_trace_directories(
            robot_paths["isaac_p60_b"] / "trace",
            robot_paths["mujoco_p60_b"] / "trace",
            scenario="p60_b",
            repeatability_snapshot=snapshot,
            isaac_result=robot_results["isaac_p60_b"],
            mujoco_result=robot_results["mujoco_p60_b"],
        ),
        "p60_c": compare_p60_robot_trace_directories(
            robot_paths["isaac_p60_c"] / "trace",
            robot_paths["mujoco_p60_c"] / "trace",
            scenario="p60_c",
            repeatability_snapshot=snapshot,
            isaac_result=robot_results["isaac_p60_c"],
            mujoco_result=robot_results["mujoco_p60_c"],
        ),
        "p60_d": compare_p60_replay_trace_directories(
            source / "trace",
            mujoco_replay / "trace",
            replay_source_identity_hash=source_result["replay_source_identity_hash"],
            repeatability_snapshot=snapshot,
            isaac_result=fresh_result,
            mujoco_result=mujoco_replay_result,
        ),
    }
    for name, analysis in analyses.items():
        _write_json(runner.output / "analysis" / f"{name}.json", analysis)

    frozen_cache = FROZEN_REPLAY_SOURCE["reset_cache"].absolute_path()
    import torch

    from wheelleg_dreamwaq.schemas.randomization import (
        validate_closed_chain_reset_cache_artifact,
    )

    cache_artifact = validate_closed_chain_reset_cache_artifact(
        torch.load(frozen_cache, map_location="cpu", weights_only=False)
    )
    cache_identity = {
        "schema_version": cache_artifact["schema_version"],
        "relaxation_algorithm_version": cache_artifact["algorithm_version"],
        "root_height_algorithm_version": cache_artifact[
            "root_height_algorithm_version"
        ],
        "tensor_sha256": cache_artifact["tensor_sha256"],
        "environment_count": int(cache_artifact["q_reset_projected_env"].shape[0]),
    }
    expected_cache_identity = {
        "schema_version": FROZEN_REPLAY_SOURCE["reset_cache_schema"],
        "relaxation_algorithm_version": FROZEN_REPLAY_SOURCE[
            "reset_cache_relaxation_algorithm"
        ],
        "root_height_algorithm_version": FROZEN_REPLAY_SOURCE[
            "reset_cache_root_height_algorithm"
        ],
        "tensor_sha256": FROZEN_REPLAY_SOURCE["reset_cache_tensor_sha256"],
        "environment_count": FROZEN_REPLAY_SOURCE["environment_count"],
    }
    reset_cache_gate: dict[str, Any] = {
        "file_sha256": sha256_file(frozen_cache),
        "expected_file_sha256": FROZEN_REPLAY_SOURCE["reset_cache"].sha256,
        "cache_identity": cache_identity,
        "expected_cache_identity": expected_cache_identity,
        "probes": {},
    }
    for scenario in ("p60_b", "p60_c"):
        result = robot_results[f"isaac_{scenario}"]
        reset_cache_gate["probes"][scenario] = {
            "path": result.get("reset_cache_path"),
            "file_sha256": result.get("reset_cache_sha256"),
            "environment_rows": result.get("reset_cache_environment_rows"),
            "expected_environment_row": FROZEN_REPLAY_SOURCE["environment_index"],
            "passed": bool(
                Path(str(result.get("reset_cache_path"))).resolve() == frozen_cache.resolve()
                and result.get("reset_cache_sha256")
                == FROZEN_REPLAY_SOURCE["reset_cache"].sha256
                and set(result.get("reset_cache_environment_rows") or [])
                == {FROZEN_REPLAY_SOURCE["environment_index"]}
            ),
        }
    reset_cache_gate["passed"] = bool(
        reset_cache_gate["file_sha256"] == reset_cache_gate["expected_file_sha256"]
        and cache_identity == expected_cache_identity
        and all(row["passed"] for row in reset_cache_gate["probes"].values())
    )
    if not reset_cache_gate["passed"]:
        raise RuntimeError("P60-B/C reset-cache source or row identity drifted")
    return runner.write_result(
        {
            "passed": True,
            "g01_stage_state_sha256": g01_state_hash,
            "threshold_snapshot_identity_hash": snapshot["identity_hash"],
            "replay_source_identity_hash": source_result["replay_source_identity_hash"],
            "replay_source_identity": source_result["replay_source_identity"],
            "replay_source_seal_identity_hash": replay_seal["seal_identity_hash"],
            "replay_source_seal": replay_seal,
            "replay_evidence": {
                "source_artifact": "replay_source",
                "isaac_artifact": "isaac:replay_0",
                "mujoco_artifact": "mujoco:replay_0",
                "source_result_sha256": sha256_file(source / "result.json"),
                "isaac_result_sha256": sha256_file(fresh / "result.json"),
                "mujoco_result_sha256": sha256_file(
                    mujoco_replay / "result.json"
                ),
                "actions_sha256": sha256_file(actions),
            },
            "fresh_replay_equivalence": fresh_result["fresh_replay_equivalence"],
            "first_action": first_action,
            "reset_cache_gate": reset_cache_gate,
            "analyses": analyses,
        }
    )


def _selected_stage_worker(run_root: Path, stage: str) -> tuple[Path, str]:
    manifest = _read_manifest_payload(run_root)
    selected = manifest.get("selected_attempts")
    if not isinstance(selected, dict) or not isinstance(selected.get(stage), dict):
        raise RuntimeError(f"Selected dependency stage is missing: {stage}")
    record = selected[stage]
    attempt_name = record.get("attempt")
    if not isinstance(attempt_name, str):
        raise RuntimeError(f"Selected dependency attempt is invalid: {stage}")
    attempt = (run_root / "stages" / stage / attempt_name).resolve(strict=True)
    state_path = attempt / "stage_state.json"
    state_hash = sha256_file(state_path)
    if record.get("stage_state_sha256") != state_hash:
        raise RuntimeError(f"Selected dependency state hash is invalid: {stage}")
    return attempt / "worker", state_hash


def _run_c70(runner: StageRunner) -> dict[str, Any]:
    p60, p60_state_hash = _selected_stage_worker(runner.run_root, "P60_full_robot")
    p60_result = json.loads((p60 / "result.json").read_text(encoding="utf-8"))
    g01_worker, g01_result, snapshot, g01_state_hash = _g01_evidence(runner)
    source, source_result, _ = _g01_replay_source(g01_worker, g01_result)
    mujoco_replay, mujoco_replay_result = _g01_probe(
        g01_worker, g01_result, engine="mujoco", label="replay_0"
    )
    if p60_result.get("g01_stage_state_sha256") != g01_state_hash:
        raise RuntimeError("C70 selected P60 result is not bound to selected G01")
    replay_seal = verified_replay_source_seal(runner.run_root)
    if p60_result.get("replay_source_seal_identity_hash") != replay_seal.get(
        "seal_identity_hash"
    ):
        raise RuntimeError("C70 selected P60 result is not bound to the replay seal")
    material_plant_failure = any(
        bool(value.get("material")) for value in p60_result["analyses"].values()
    )
    if material_plant_failure:
        analysis = analyze_c70_checkpoint_sensitivity(
            source / "result.json",
            mujoco_replay / "result.json",
        )
    else:
        analysis = {
            "schema_version": "RootCauseC70CheckpointSensitivityV1",
            "replay_source_identity_hash": replay_seal[
                "replay_source_identity_hash"
            ],
            "anchors": {},
            "checkpoint_role": "not_evaluated",
            "reason": "no_material_plant_failure",
        }
    _write_json(runner.output / "analysis" / "c70.json", analysis)
    return runner.write_result(
        {
            "passed": True,
            "material_plant_failure": material_plant_failure,
            "p60_stage_state_sha256": p60_state_hash,
            "g01_stage_state_sha256": g01_state_hash,
            "threshold_snapshot_identity_hash": snapshot["identity_hash"],
            "replay_source_seal_identity_hash": replay_seal["seal_identity_hash"],
            "replay_source_identity_hash": source_result[
                "replay_source_identity_hash"
            ],
            "mujoco_replay_result_sha256": sha256_file(
                mujoco_replay / "result.json"
            ),
            "analysis": analysis,
        }
    )


STAGE_FUNCTIONS = {
    "G00_integrity": _run_g00,
    "G01_repeatability": _run_g01,
    "G02_adapter": _run_g02,
    "G03_instrumentation": _run_g03,
    "P10_rest": _run_p10,
    "P20_static_properties": _run_p20,
    "P30_actuator": _run_p30,
    "P40_closure": _run_p40,
    "P50_contact": _run_p50,
    "P60_full_robot": _run_p60,
    "C70_checkpoint": _run_c70,
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage not in STAGE_FUNCTIONS:
        raise ValueError(f"Unknown Core stage: {args.stage}")
    runner = StageRunner(args.stage, args.run_root, args.output)
    result = STAGE_FUNCTIONS[args.stage](runner)
    if not result.get("passed"):
        raise RuntimeError(f"Core stage did not pass: {args.stage}")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
