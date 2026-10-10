from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from .contract import sha256_file
from .physics import PHYSICS_SCHEMA_VERSION, unrestricted_velocity_policy, validate_policy_physics
from .model_semantics import stable_hash


EVALUATION_SCHEMA_VERSION = "MujocoEvaluationV1"
EVALUATION_CONTRACT_VERSION = "MujocoEvaluationContractV3"
# A diagnostic failure threshold, never a physics speed cap or braking rule.
JOINT_SPEED_FAILURE_THRESHOLD_RAD_S = 80.0
RANKING_VERSION = "MujocoRankingV1"
EVALUATION_SOURCE_VERSION = "MujocoEvaluationSourceV1"
FORMAL_RUN_COUNT = 4
FORMAL_ITERATIONS = 1000
FORMAL_SCENARIOS = (
    ("nominal_stand", (0.0, 0.0, 0.20)),
    ("low_stand", (0.0, 0.0, 0.17)),
    ("high_stand", (0.0, 0.0, 0.23)),
    ("forward", (1.0, 0.0, 0.20)),
    ("reverse", (-1.0, 0.0, 0.20)),
    ("left_turn", (0.0, 0.6, 0.20)),
    ("right_turn", (0.0, -0.6, 0.20)),
    ("combined", (0.8, 0.5, 0.20)),
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SIM2SIM_ROOT = PROJECT_ROOT / "sim2sim" / "mujoco"


def build_evaluation_source_fingerprint() -> dict:
    files = sorted(
        [path for path in (SIM2SIM_ROOT / "wheelleg_mujoco").rglob("*.py") if path.is_file()]
        + [
            PROJECT_ROOT / "scripts" / "evaluate_mujoco.py",
            PROJECT_ROOT / "scripts" / "rank_mujoco_runs.py",
            SIM2SIM_ROOT / "pyproject.toml",
            SIM2SIM_ROOT / "uv.lock",
        ],
        key=lambda path: path.relative_to(PROJECT_ROOT).as_posix(),
    )
    payload = {
        "schema_version": EVALUATION_SOURCE_VERSION,
        "files": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
            for path in files
        },
    }
    payload["hash"] = stable_hash(payload)
    return payload


def unwrap_yaw(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.unwrap(values)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _mean_available(scenarios: list[dict], key: str) -> float | None:
    values = [float(scenario[key]) for scenario in scenarios if scenario.get(key) is not None]
    return fmean(values) if values else None


def _score(
    *,
    vx_mae: float,
    yaw_rate_mae: float | None,
    height_mae: float,
    roll_rms: float,
    pitch_rms: float,
    yaw_drift_rms: float | None,
    phi0_delta_rms: float,
) -> float:
    score = (
        vx_mae / 1.5
        + height_mae / 0.04
        + roll_rms / math.radians(10.0)
        + pitch_rms / math.radians(10.0)
        + phi0_delta_rms / math.radians(10.0)
    )
    if yaw_rate_mae is not None:
        score += yaw_rate_mae / 1.0
    if yaw_drift_rms is not None:
        score += yaw_drift_rms / math.radians(15.0)
    return score


def summarize_scenario(
    name: str,
    command: tuple[float, float, float],
    rows: list[dict[str, float]],
    *,
    expected_ticks: int,
    failure_reason: str | None = None,
    initial_yaw_rad: float = 0.0,
) -> dict:
    if expected_ticks <= 0:
        raise ValueError("expected_ticks must be positive")
    if not rows:
        return {
            "name": name,
            "command": list(command),
            "ticks": 0,
            "expected_ticks": expected_ticks,
            "completed": False,
            "survival_fraction": 0.0,
            "failure_reason": failure_reason or "no_valid_tick",
            "initial_yaw_rad": initial_yaw_rad,
            "vx_mae": None,
            "yaw_rate_mae": None,
            "height_mae": None,
            "height_max_abs_error": None,
            "roll_rms": None,
            "roll_max_abs": None,
            "pitch_rms": None,
            "pitch_max_abs": None,
            "yaw_drift_rms": None,
            "yaw_drift_max_abs": None,
            "phi0_delta_rms": None,
            "phi0_delta_max_abs": None,
            "action_saturation_fraction": None,
            "effort_saturation_fraction": None,
            "max_loop_closure_error_m": None,
            "max_tilt_rad": None,
            "prefix_score": math.inf,
        }

    vx = np.asarray([row["vx_mps"] for row in rows], dtype=np.float64)
    yaw_rate = np.asarray([row["yaw_rate_rad_s"] for row in rows], dtype=np.float64)
    height = np.asarray([row["base_height_m"] for row in rows], dtype=np.float64)
    roll = np.asarray([row["roll_rad"] for row in rows], dtype=np.float64)
    pitch = np.asarray([row["pitch_rad"] for row in rows], dtype=np.float64)
    phi0_delta = np.asarray([row["phi0_delta_abs_rad"] for row in rows], dtype=np.float64)
    yaw = unwrap_yaw(np.asarray([row["yaw_rad"] for row in rows], dtype=np.float64))
    height_error = np.abs(height - command[2])
    zero_yaw_command = abs(command[1]) < 1.0e-12
    yaw_drift = yaw - initial_yaw_rad if zero_yaw_command else None
    yaw_rate_mae = None if zero_yaw_command else float(np.mean(np.abs(yaw_rate - command[1])))
    yaw_drift_rms = _rms(yaw_drift) if yaw_drift is not None else None

    summary = {
        "name": name,
        "command": list(command),
        "ticks": len(rows),
        "expected_ticks": expected_ticks,
        "completed": len(rows) == expected_ticks and failure_reason is None,
        "survival_fraction": min(len(rows) / expected_ticks, 1.0),
        "failure_reason": failure_reason,
        "initial_yaw_rad": initial_yaw_rad,
        "vx_mae": float(np.mean(np.abs(vx - command[0]))),
        "yaw_rate_mae": yaw_rate_mae,
        "height_mae": float(np.mean(height_error)),
        "height_max_abs_error": float(np.max(height_error)),
        "roll_rms": _rms(roll),
        "roll_max_abs": float(np.max(np.abs(roll))),
        "pitch_rms": _rms(pitch),
        "pitch_max_abs": float(np.max(np.abs(pitch))),
        "yaw_drift_rms": yaw_drift_rms,
        "yaw_drift_max_abs": float(np.max(np.abs(yaw_drift))) if yaw_drift is not None else None,
        "phi0_delta_rms": _rms(phi0_delta),
        "phi0_delta_max_abs": float(np.max(phi0_delta)),
        "action_saturation_fraction": float(
            np.mean([row["action_saturation_fraction"] for row in rows])
        ),
        "effort_saturation_fraction": float(
            np.mean([row["effort_saturation_fraction"] for row in rows])
        ),
        "max_loop_closure_error_m": float(
            np.max([row["max_loop_closure_error_m"] for row in rows])
        ),
        "max_tilt_rad": float(np.max([row["tilt_rad"] for row in rows])),
    }
    summary["prefix_score"] = _score(
        vx_mae=summary["vx_mae"],
        yaw_rate_mae=summary["yaw_rate_mae"],
        height_mae=summary["height_mae"],
        roll_rms=summary["roll_rms"],
        pitch_rms=summary["pitch_rms"],
        yaw_drift_rms=summary["yaw_drift_rms"],
        phi0_delta_rms=summary["phi0_delta_rms"],
    )
    return summary


def aggregate_run(run_id: str, scenarios: list[dict]) -> dict:
    if not scenarios:
        raise ValueError("A run must contain at least one scenario summary")
    metric_keys = (
        "vx_mae",
        "yaw_rate_mae",
        "height_mae",
        "roll_rms",
        "pitch_rms",
        "yaw_drift_rms",
        "phi0_delta_rms",
        "action_saturation_fraction",
        "effort_saturation_fraction",
        "max_tilt_rad",
    )
    metrics = {key: _mean_available(scenarios, key) for key in metric_keys}
    metrics["max_loop_closure_error_m"] = max(
        (float(value) for scenario in scenarios if (value := scenario.get("max_loop_closure_error_m")) is not None),
        default=math.inf,
    )
    required = ("vx_mae", "height_mae", "roll_rms", "pitch_rms", "phi0_delta_rms")
    score = math.inf
    if all(metrics[key] is not None for key in required) and not any(
        math.isinf(float(scenario["prefix_score"])) for scenario in scenarios
    ):
        score = _score(
            vx_mae=float(metrics["vx_mae"]),
            yaw_rate_mae=metrics["yaw_rate_mae"],
            height_mae=float(metrics["height_mae"]),
            roll_rms=float(metrics["roll_rms"]),
            pitch_rms=float(metrics["pitch_rms"]),
            yaw_drift_rms=metrics["yaw_drift_rms"],
            phi0_delta_rms=float(metrics["phi0_delta_rms"]),
        )
    completed_scenarios = sum(bool(scenario["completed"]) for scenario in scenarios)
    return {
        "run_id": run_id,
        "scenario_count": len(scenarios),
        "completed_scenarios": completed_scenarios,
        "full_survival": completed_scenarios == len(scenarios),
        "survival_fraction": fmean(float(scenario["survival_fraction"]) for scenario in scenarios),
        "score": score,
        "metrics": metrics,
        "scenarios": scenarios,
    }


def build_evaluation_contract(
    *,
    policy_manifest: dict,
    model_manifest: dict,
    scenarios: tuple[tuple[str, tuple[float, float, float]], ...],
    expected_ticks: int,
    smoke: bool,
    training_suite: dict | None = None,
) -> dict:
    validate_policy_physics(policy_manifest)
    policy_schema_version = policy_manifest["schema_version"]
    if policy_schema_version == "DreamWaQPolicyExportV1":
        task_contract_version = policy_manifest["base_task_contract_version"]
        task_contract_hash = policy_manifest["base_task_contract_hash"]
    else:
        task_contract_version = policy_manifest["phase1_contract_version"]
        task_contract_hash = policy_manifest["phase1_contract_hash"]
    return {
        "schema_version": EVALUATION_CONTRACT_VERSION,
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "ranking_version": RANKING_VERSION,
        "smoke": smoke,
        "training_suite": training_suite,
        "evaluation_implementation": build_evaluation_source_fingerprint(),
        "policy_export_schema_version": policy_schema_version,
        "task_contract_version": task_contract_version,
        "task_contract_hash": task_contract_hash,
        "phase1_contract_hash": policy_manifest.get("phase1_contract_hash"),
        "base_task_contract_hash": policy_manifest.get("base_task_contract_hash"),
        "control_dt_s": policy_manifest["timing"]["control_dt_s"],
        "physics_dt_s": policy_manifest["timing"]["mujoco_physics_dt_s"],
        "physics_steps_per_action": policy_manifest["timing"]["mujoco_physics_steps_per_action"],
        "physics_schema_version": PHYSICS_SCHEMA_VERSION,
        "velocity_limit_policy": unrestricted_velocity_policy(),
        "expected_ticks_per_scenario": expected_ticks,
        "scenario_duration_s": expected_ticks * policy_manifest["timing"]["control_dt_s"],
        "scenarios": [
            {"name": name, "command": list(command)} for name, command in scenarios
        ],
        "model_version": model_manifest["model_version"],
        "model_xml_sha256": model_manifest["model_xml"]["sha256"],
        "model_manifest_sha256": policy_manifest["mujoco_model"]["model_manifest_sha256"],
        "dynamics_semantics_hash": model_manifest["dynamics_semantics_hash"],
        "reset_initial_yaw_rad": 0.0,
        "adapter_versions": {
            name: adapter["version"] for name, adapter in model_manifest["adapters"].items()
        },
        "failure_limits": {
            "base_height_m": [0.10, 0.40],
            "tilt_rad": 0.80,
            "root_linear_speed_mps": 20.0,
            "root_angular_speed_rad_s": 35.0,
            "joint_speed_rad_s": JOINT_SPEED_FAILURE_THRESHOLD_RAD_S,
            "loop_closure_error_m": 5.0e-3,
            "virtual_leg_min_length_m": 0.05,
        },
    }


def _assert_equivalent(expected: Any, actual: Any, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise ValueError(f"{path} keys differ")
        for key in sorted(expected):
            _assert_equivalent(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"{path} list shape differs")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            _assert_equivalent(expected_item, actual_item, f"{path}[{index}]")
        return
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            raise ValueError(f"{path} differs: {expected!r} != {actual!r}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        expected_float = float(expected)
        actual_float = float(actual)
        if math.isnan(expected_float) or math.isnan(actual_float):
            raise ValueError(f"{path} contains NaN")
        if math.isinf(expected_float) or math.isinf(actual_float):
            if expected_float != actual_float:
                raise ValueError(f"{path} differs: {expected!r} != {actual!r}")
        elif not math.isclose(expected_float, actual_float, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError(f"{path} differs: {expected!r} != {actual!r}")
        return
    if expected != actual:
        raise ValueError(f"{path} differs: {expected!r} != {actual!r}")


def _load_bound_policy_manifest(report: dict, training_run: dict) -> dict:
    actor_path = Path(report["policy"]).resolve()
    manifest_path = Path(report["policy_manifest"]).resolve()
    checkpoint_path = Path(report["source_checkpoint"]).resolve()
    for label, path in (("actor", actor_path), ("policy manifest", manifest_path), ("checkpoint", checkpoint_path)):
        if not path.is_file():
            raise ValueError(f"MuJoCo ranking {label} is missing: {path}")
    if sha256_file(actor_path) != report.get("policy_sha256"):
        raise ValueError("MuJoCo ranking actor hash differs from the evaluation report")
    if sha256_file(manifest_path) != report.get("policy_manifest_sha256"):
        raise ValueError("MuJoCo ranking policy-manifest file hash differs from the evaluation report")
    if sha256_file(checkpoint_path) != report.get("source_checkpoint_sha256"):
        raise ValueError("MuJoCo ranking checkpoint hash differs from the evaluation report")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedded_hash = manifest.get("manifest_hash")
    calculated_hash = stable_hash({key: value for key, value in manifest.items() if key != "manifest_hash"})
    if embedded_hash != calculated_hash or embedded_hash != report.get("policy_manifest_hash"):
        raise ValueError("MuJoCo ranking policy-manifest embedded hash is invalid")
    if manifest.get("actor_sha256") != report["policy_sha256"]:
        raise ValueError("MuJoCo ranking actor is not bound to the policy manifest")
    if manifest.get("source_checkpoint_sha256") != report["source_checkpoint_sha256"]:
        raise ValueError("MuJoCo ranking checkpoint is not bound to the policy manifest")
    if int(manifest.get("source_completed_iterations", -1)) != int(training_run["completed_iterations"]):
        raise ValueError("MuJoCo ranking checkpoint iteration count differs from the training context")
    return manifest


def _summary_from_csv(
    report: dict,
    scenario_summary: dict,
    expected: dict,
    evaluation_contract: dict,
) -> dict:
    scenario_files = report.get("scenario_files", {})
    file_record = scenario_files.get(expected["name"])
    if not isinstance(file_record, dict):
        raise ValueError(f"MuJoCo evaluation is missing CSV identity for {expected['name']}")
    csv_path = Path(file_record["path"]).resolve()
    if not csv_path.is_file() or sha256_file(csv_path) != file_record.get("sha256"):
        raise ValueError(f"MuJoCo evaluation CSV hash differs for {expected['name']}")

    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    valid_rows = []
    failure_reason = None
    metric_fields = (
        "base_height_m",
        "roll_rad",
        "pitch_rad",
        "yaw_rad",
        "vx_mps",
        "yaw_rate_rad_s",
        "phi0_delta_abs_rad",
        "action_saturation_fraction",
        "effort_saturation_fraction",
        "max_loop_closure_error_m",
        "tilt_rad",
    )
    for index, row in enumerate(rows, start=1):
        if int(row["tick"]) != index:
            raise ValueError(f"MuJoCo evaluation CSV tick order differs for {expected['name']}")
        targets = (
            float(row["target_vx_mps"]),
            float(row["target_yaw_rate_rad_s"]),
            float(row["target_base_height_m"]),
        )
        if not np.allclose(targets, expected["command"], rtol=0.0, atol=1.0e-12):
            raise ValueError(f"MuJoCo evaluation CSV command differs for {expected['name']}")
        survived = int(row["survived"])
        if survived == 1:
            if failure_reason is not None:
                raise ValueError(f"MuJoCo evaluation CSV resumes after failure for {expected['name']}")
            valid_rows.append({field: float(row[field]) for field in metric_fields})
        elif survived == 0:
            if failure_reason is not None or index != len(rows):
                raise ValueError(f"MuJoCo evaluation CSV failure position differs for {expected['name']}")
            failure_reason = row["failure_reason"] or "unspecified_failure"
        else:
            raise ValueError(f"MuJoCo evaluation CSV has invalid survived flag for {expected['name']}")

    return summarize_scenario(
        expected["name"],
        tuple(expected["command"]),
        valid_rows,
        expected_ticks=int(evaluation_contract["expected_ticks_per_scenario"]),
        failure_reason=failure_reason,
        initial_yaw_rad=float(evaluation_contract["reset_initial_yaw_rad"]),
    )


def validate_evaluation_reports_for_ranking(reports: list[dict]) -> dict:
    if len(reports) != FORMAL_RUN_COUNT:
        raise ValueError("Formal MuJoCo ranking requires exactly four evaluation reports")
    expected_scenarios = [
        {"name": name, "command": list(command)} for name, command in FORMAL_SCENARIOS
    ]
    common_contract = None
    checkpoint_hashes = set()
    run_indices = set()
    current_implementation = build_evaluation_source_fingerprint()
    for report in reports:
        embedded_report_hash = report.get("report_hash")
        calculated_report_hash = stable_hash(
            {key: value for key, value in report.items() if key != "report_hash"}
        )
        if embedded_report_hash != calculated_report_hash:
            raise ValueError("MuJoCo evaluation report hash is invalid")
        if report.get("evaluation_schema_version") != EVALUATION_SCHEMA_VERSION:
            raise ValueError("Unsupported MuJoCo evaluation schema")
        if report.get("smoke") is not False:
            raise ValueError("Smoke evaluations cannot enter formal ranking")
        contract = report.get("evaluation_contract")
        if not isinstance(contract, dict) or report.get("evaluation_contract_hash") != stable_hash(contract):
            raise ValueError("MuJoCo evaluation contract hash is invalid")
        if contract.get("schema_version") != EVALUATION_CONTRACT_VERSION:
            raise ValueError("Unsupported MuJoCo evaluation contract")
        if (
            contract.get("physics_schema_version") != PHYSICS_SCHEMA_VERSION
            or contract.get("velocity_limit_policy") != unrestricted_velocity_policy()
            or "rigid_body_angular_limit" in contract
        ):
            raise ValueError("MuJoCo velocity-limit policy differs from the current runtime")
        if contract.get("evaluation_implementation") != current_implementation:
            raise ValueError("MuJoCo evaluation implementation fingerprint differs from the current runtime")
        if contract.get("smoke") is not False or contract.get("scenarios") != expected_scenarios:
            raise ValueError("Formal MuJoCo scenario contract differs from the frozen suite")
        if contract.get("expected_ticks_per_scenario") != 500 or contract.get("scenario_duration_s") != 10.0:
            raise ValueError("Formal MuJoCo evaluation must use 500 ticks over 10 seconds")
        training_suite = contract.get("training_suite")
        if not isinstance(training_suite, dict):
            raise ValueError("Formal MuJoCo evaluation is missing the training-suite context")
        suite_schema = training_suite.get("schema_version")
        expected_run_schema = {
            "TrainingEvaluationContextV1": "TrainingEvaluationRunV1",
            "DreamWaQTrainingEvaluationContextV1": "DreamWaQTrainingEvaluationRunV1",
        }.get(suite_schema)
        if (
            expected_run_schema is None
            or training_suite.get("suite_mode") != "formal"
            or training_suite.get("run_count") != FORMAL_RUN_COUNT
            or training_suite.get("iterations_per_run") != FORMAL_ITERATIONS
        ):
            raise ValueError("Formal MuJoCo evaluation training-suite context is invalid")
        context_hash = training_suite.get("context_hash")
        calculated_context_hash = stable_hash(
            {key: value for key, value in training_suite.items() if key != "context_hash"}
        )
        if context_hash != calculated_context_hash:
            raise ValueError("Formal MuJoCo evaluation training-suite context hash is invalid")
        if common_contract is None:
            common_contract = contract
        elif contract != common_contract:
            raise ValueError("MuJoCo evaluation reports use different contracts")
        training_run = report.get("training_run")
        if not isinstance(training_run, dict):
            raise ValueError("Formal MuJoCo evaluation is missing its training-run context")
        if (
            training_run.get("schema_version") != expected_run_schema
            or training_run.get("suite_id") != training_suite.get("suite_id")
            or training_run.get("run_count") != FORMAL_RUN_COUNT
            or training_run.get("completed_iterations") != FORMAL_ITERATIONS
            or training_run.get("training_fingerprint_hash") != training_suite.get("training_fingerprint_hash")
            or training_run.get("suite_source_fingerprint_hash")
            != training_suite.get("suite_source_fingerprint_hash")
        ):
            raise ValueError("Formal MuJoCo evaluation training-run context is invalid")
        run_index = training_run.get("run_index")
        if not isinstance(run_index, int) or not 1 <= run_index <= FORMAL_RUN_COUNT or run_index in run_indices:
            raise ValueError("Formal MuJoCo evaluation run indices are invalid or duplicated")
        run_indices.add(run_index)
        checkpoint_hash = report.get("source_checkpoint_sha256")
        if not checkpoint_hash or checkpoint_hash in checkpoint_hashes:
            raise ValueError("MuJoCo ranking contains a missing or duplicate source checkpoint")
        checkpoint_hashes.add(checkpoint_hash)
        policy_manifest = _load_bound_policy_manifest(report, training_run)
        validate_policy_physics(policy_manifest)
        manifest_task_hash = policy_manifest.get(
            "base_task_contract_hash",
            policy_manifest.get("phase1_contract_hash"),
        )
        contract_task_hash = contract.get("task_contract_hash", contract.get("phase1_contract_hash"))
        if manifest_task_hash != contract_task_hash:
            raise ValueError("MuJoCo ranking policy manifest uses a different task contract")
        aggregate = report.get("aggregate", {})
        scenarios = aggregate.get("scenarios", [])
        if aggregate.get("scenario_count") != len(FORMAL_SCENARIOS) or len(scenarios) != len(FORMAL_SCENARIOS):
            raise ValueError("MuJoCo evaluation report does not contain all formal scenarios")
        recomputed_scenarios = []
        for summary, expected in zip(scenarios, expected_scenarios, strict=True):
            if (
                summary.get("name") != expected["name"]
                or summary.get("command") != expected["command"]
                or summary.get("expected_ticks") != 500
            ):
                raise ValueError("MuJoCo scenario summary differs from the evaluation contract")
            recomputed = _summary_from_csv(report, summary, expected, contract)
            _assert_equivalent(recomputed, summary, f"scenario[{expected['name']}]")
            recomputed_scenarios.append(recomputed)
        recomputed_aggregate = aggregate_run(checkpoint_hash[:12], recomputed_scenarios)
        try:
            _assert_equivalent(recomputed_aggregate, aggregate, "aggregate")
        except ValueError as error:
            raise ValueError(f"MuJoCo evaluation aggregate is inconsistent: {error}") from error
    if run_indices != set(range(1, FORMAL_RUN_COUNT + 1)):
        raise ValueError("Formal MuJoCo evaluation run index set is incomplete")
    assert common_contract is not None
    return common_contract


def rank_runs(runs: list[dict]) -> list[dict]:
    def finite_or_inf(value) -> float:
        if value is None:
            return math.inf
        converted = float(value)
        return converted if math.isfinite(converted) else math.inf

    def key(run: dict) -> tuple:
        metrics = run["metrics"]
        score = finite_or_inf(run.get("score"))
        rounded_score = round(score, 6) if math.isfinite(score) else math.inf
        tie_breakers = (
            finite_or_inf(metrics.get("effort_saturation_fraction")),
            finite_or_inf(metrics.get("action_saturation_fraction")),
            finite_or_inf(metrics.get("max_tilt_rad")),
        )
        if run["full_survival"]:
            return (0, rounded_score, *tie_breakers)
        return (
            1,
            -int(run["completed_scenarios"]),
            -float(run["survival_fraction"]),
            rounded_score,
            *tie_breakers,
        )

    return sorted(runs, key=key)


def build_ranking_summary(reports: list[dict], evaluation_paths: list[str]) -> dict:
    if len(reports) != len(evaluation_paths):
        raise ValueError("Evaluation report/path counts differ")
    evaluation_contract = validate_evaluation_reports_for_ranking(reports)
    candidates = []
    for report, evaluation_path in zip(reports, evaluation_paths, strict=True):
        candidate = dict(report["aggregate"])
        candidate.update(
            {
                "evaluation_summary": str(Path(evaluation_path).resolve()),
                "source_checkpoint": report["source_checkpoint"],
                "source_checkpoint_sha256": report["source_checkpoint_sha256"],
                "policy": report["policy"],
                "policy_manifest": report["policy_manifest"],
            }
        )
        candidates.append(candidate)
    ranked = rank_runs(candidates)
    has_full_survival_candidate = any(candidate["full_survival"] for candidate in ranked)
    best_candidate = ranked[0]
    training_suite = evaluation_contract.get("training_suite") or {}
    dreamwaq = training_suite.get("schema_version") == "DreamWaQTrainingEvaluationContextV1"
    return {
        "schema_version": "MujocoDreamWaQTrainingSuiteRankingV1" if dreamwaq else "MujocoTrainingSuiteRankingV2",
        "evaluation_contract": evaluation_contract,
        "evaluation_contract_hash": reports[0]["evaluation_contract_hash"],
        "has_full_survival_candidate": has_full_survival_candidate,
        "phase1_qualified": False if not dreamwaq else None,
        "phase2_acceptance_status": "pending_isaac_and_context_gates" if dreamwaq else None,
        "qualification_status": "pending_g08" if not dreamwaq else "mujoco_evidence_complete",
        "selected": None,
        "best_candidate": best_candidate,
        "diagnostic_candidate": None if has_full_survival_candidate else best_candidate,
        "ranked_runs": ranked,
    }
