from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .trace_schema import (
    load_npz,
    sha256_file,
    stable_payload_hash,
    validate_control_trace,
    validate_substep_trace,
    verify_file_hashes,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_INPUT_HASHES = {
    "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml": "691C607271ED85EFFA388237E00A7B8F8F41CB66F6BA9C49EA64ABD186D803D1",
    "sim2sim/mujoco/model_manifest.json": "C3DB4FE2797F6AC3628F0A9CB5E0B166FFD0CE802776A335922C474A045F7AE0",
    "sim2sim/mujoco/wheelleg_mujoco/runner.py": "B0C973A5335856E3991ECCF6014BA9B5CF65E638A8BD4701FC07EED1A446EDBB",
    "sim2sim/mujoco/wheelleg_mujoco/control.py": "83080BC57A0399B11610CE5E06B4F607ADA1C0FB995344E23E531658EBBEFD62",
    "sim2sim/mujoco/wheelleg_mujoco/observation.py": "DD7F972D77FCCA25378BF403F61A67755C8AB4BA28738C97ABF2BBDD733970BD",
    "source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py": "5399B29B92B54851EB0A5FA880EA9386C82F202A5C337F19ABE4EDF300C2969C",
    "source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env_cfg.py": "AB7BAA693A73EFBB0F933A605EF02F94C0BDBCF95E468D3CF400E590F59101F5",
}


CONTROL_THRESHOLDS = {
    "actor_obs_current_pre_step": 0.01,
    "actor_obs_policy_pre_step": 0.01,
    "cenet_estimated_velocity": 0.02,
    "cenet_context_mu": 0.02,
    "cenet_context_logvar": 0.02,
    "actor_output_raw": 0.05,
    "action_clipped": 0.05,
    "target_command_canonical": 0.50,
    "active_joint_position_canonical_post_step_pre_reset": 0.01,
    "active_joint_velocity_canonical_post_step_pre_reset": 0.20,
    "base_linear_velocity_control_post_step_pre_reset": 0.10,
    "base_angular_velocity_control_post_step_pre_reset": 0.20,
    "projected_gravity_post_step_pre_reset": 0.02,
    "base_height_post_step_pre_reset": 0.005,
    "virtual_leg_length_post_step_pre_reset": 0.005,
    "virtual_leg_phi0_post_step_pre_reset": 0.02,
    "loop_closure_error_post_step_pre_reset": 5.0e-4,
    "pd_torque_effort_clipped_canonical_mean": 0.50,
}

SUBSTEP_THRESHOLDS = {
    "active_joint_position_canonical_post_step": 0.01,
    "active_joint_velocity_canonical_post_step": 0.20,
    "base_linear_velocity_control_post_step": 0.10,
    "base_angular_velocity_control_post_step": 0.20,
    "projected_gravity_post_step": 0.02,
    "pd_torque_effort_clipped_canonical": 0.50,
}


def _first_true(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(values)
    return None if len(indices) == 0 else int(indices[0])


def _first_persistent(values: np.ndarray, width: int = 3) -> int | None:
    if len(values) < width:
        return None
    for start in range(len(values) - width + 1):
        if bool(np.all(values[start : start + width])):
            return start
    return None


def signal_metrics(
    left: np.ndarray,
    right: np.ndarray,
    *,
    material_threshold: float,
    numeric_threshold: float = 1.0e-6,
) -> dict[str, Any]:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.shape[0] == 0:
        raise ValueError(f"Signal shapes must be equal and non-empty: {lhs.shape} != {rhs.shape}")
    delta = np.abs(lhs - rhs).reshape(lhs.shape[0], -1)
    finite = np.isfinite(delta)
    per_row = np.full(lhs.shape[0], np.nan)
    for index in range(lhs.shape[0]):
        if finite[index].any():
            per_row[index] = float(np.max(delta[index][finite[index]]))
    finite_rows = np.isfinite(per_row)
    numeric = finite_rows & (per_row > numeric_threshold)
    material = finite_rows & (per_row > material_threshold)
    return {
        "material_threshold": material_threshold,
        "numeric_threshold": numeric_threshold,
        "max_abs_error": float(np.nanmax(per_row)),
        "rms_error": float(np.sqrt(np.nanmean(np.square(delta)))),
        "first_numeric_tick": _first_true(numeric),
        "first_material_tick": _first_true(material),
        "first_persistent_material_tick": _first_persistent(material),
        "per_tick_max_abs_error": per_row.tolist(),
    }


def _load_run(path: Path) -> dict[str, Any]:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    reset = json.loads((path / "reset_snapshot.json").read_text(encoding="utf-8"))
    expected_hashes = json.loads((path / "file_hashes.json").read_text(encoding="utf-8"))
    engine = metadata.get("engine")
    collector_name = (
        "collect_dreamwaq_isaac_trace.py"
        if engine == "isaac_sim"
        else "collect_dreamwaq_mujoco_trace.py"
    )
    artifact_paths = {
        "collector": Path(__file__).with_name(collector_name),
        "debug_contract": Path(__file__).with_name("dreamwaq_debug_contract.py"),
        "metadata": path / "metadata.json",
        "reset_snapshot": path / "reset_snapshot.json",
        "control_trace": path / "control_trace.npz",
        "substep_trace": path / "substep_trace.npz",
    }
    if engine == "isaac_sim":
        artifact_paths["debug_environment"] = Path(__file__).with_name("isaac_debug_env.py")
    unknown_hashes = sorted(set(expected_hashes) - set(artifact_paths))
    if unknown_hashes:
        raise ValueError(f"Unsupported trace hash entries in {path}: {unknown_hashes}")
    verify_file_hashes(
        {name: artifact_paths[name] for name in expected_hashes},
        expected_hashes,
    )
    control = load_npz(path / "control_trace.npz")
    substep = load_npz(path / "substep_trace.npz")
    policy_dimension = int(metadata["policy_input_dimension"])
    steps_per_action = int(metadata["timing_profile"]["physics_steps_per_action"])
    validate_control_trace(control, policy_observation_dimension=policy_dimension)
    validate_substep_trace(
        substep,
        physics_steps_per_action=steps_per_action,
        continuity_atol=1.0e-6 if engine == "isaac_sim" else 1.0e-12,
    )
    completed_ticks = int(metadata["completed_control_ticks"])
    if len(control["control_tick"]) != completed_ticks:
        raise ValueError(f"Control row count differs from metadata in {path}")
    if len(substep["control_tick"]) != completed_ticks * steps_per_action:
        raise ValueError(f"Substep row count differs from metadata in {path}")
    if metadata.get("schema_version") != "DreamWaQSim2SimTraceV1":
        raise ValueError(f"Unsupported trace schema in {path}")
    if metadata.get("debug_only") is not True or metadata.get("formal_ranking_eligible") is not False:
        raise ValueError(f"Trace is not marked debug-only in {path}")
    return {
        "path": str(path.resolve()),
        "metadata": metadata,
        "reset": reset,
        "control": control,
        "substep": substep,
        "integrity": {
            "file_hashes_verified": True,
            "trace_dimensions_verified": True,
            "control_rows": completed_ticks,
            "substep_rows": completed_ticks * steps_per_action,
        },
        "hashes": {
            "metadata": sha256_file(path / "metadata.json"),
            "reset": sha256_file(path / "reset_snapshot.json"),
            "control": sha256_file(path / "control_trace.npz"),
            "substep": sha256_file(path / "substep_trace.npz"),
        },
    }


def _reset_alignment(isaac: dict[str, Any], mujoco: dict[str, Any]) -> dict[str, float]:
    fields = ("current_actor_observation", "policy_input", "active_joint_position_canonical")
    return {
        field: float(
            np.max(
                np.abs(
                    np.asarray(isaac["reset"][field], dtype=np.float64)
                    - np.asarray(mujoco["reset"][field], dtype=np.float64)
                )
            )
        )
        for field in fields
    }


def compare_control_runs(
    isaac: dict[str, Any],
    mujoco: dict[str, Any],
    *,
    require_identical_applied_actions: bool,
) -> dict[str, Any]:
    count = min(
        len(isaac["control"]["control_tick"]),
        len(mujoco["control"]["control_tick"]),
    )
    signals = {}
    first_material: tuple[int, str] | None = None
    for name, threshold in CONTROL_THRESHOLDS.items():
        metrics = signal_metrics(
            isaac["control"][name][:count],
            mujoco["control"][name][:count],
            material_threshold=threshold,
        )
        signals[name] = metrics
        tick = metrics["first_material_tick"]
        if tick is not None and (first_material is None or (tick, name) < first_material):
            first_material = (tick, name)
    action_error = float(
        np.max(
            np.abs(
                np.asarray(isaac["control"]["action_clipped"][:count], dtype=np.float64)
                - np.asarray(mujoco["control"]["action_clipped"][:count], dtype=np.float64)
            )
        )
    )
    if require_identical_applied_actions and action_error > 1.0e-6:
        raise ValueError(f"Applied action replay is not identical: {action_error}")
    return {
        "compared_ticks": count,
        "reset_alignment": _reset_alignment(isaac, mujoco),
        "applied_action_max_abs_error": action_error,
        "applied_actions_identical": action_error <= 1.0e-6,
        "first_material_divergence": None
        if first_material is None
        else {"control_tick": first_material[0], "signal": first_material[1]},
        "signals": signals,
        "inputs": {"isaac": isaac["path"], "mujoco": mujoco["path"]},
        "input_hashes": {"isaac": isaac["hashes"], "mujoco": mujoco["hashes"]},
    }


def compare_single_action_substeps(
    isaac: dict[str, Any],
    mujoco: dict[str, Any],
) -> dict[str, Any]:
    isaac_indices = np.asarray((0, 1, 2, 3))
    mujoco_indices = np.asarray((4, 9, 14, 19))
    signals = {
        name: signal_metrics(
            isaac["substep"][name][isaac_indices],
            mujoco["substep"][name][mujoco_indices],
            material_threshold=threshold,
        )
        for name, threshold in SUBSTEP_THRESHOLDS.items()
    }
    initial_fields = (
        "target_command_canonical",
        "active_joint_position_canonical_pre_step",
        "active_joint_velocity_canonical_pre_step",
        "pd_torque_effort_clipped_canonical",
    )
    initial_alignment = {
        name: float(
            np.max(
                np.abs(
                    np.asarray(isaac["substep"][name][0], dtype=np.float64)
                    - np.asarray(mujoco["substep"][name][0], dtype=np.float64)
                )
            )
        )
        for name in initial_fields
    }
    first_material: tuple[int, str] | None = None
    for name, metrics in signals.items():
        sample = metrics["first_material_tick"]
        if sample is not None and (first_material is None or (sample, name) < first_material):
            first_material = (sample, name)
    return {
        "common_elapsed_time_ms": [5, 10, 15, 20],
        "initial_pre_step_alignment": initial_alignment,
        "first_material_divergence": None
        if first_material is None
        else {
            "elapsed_time_ms": [5, 10, 15, 20][first_material[0]],
            "signal": first_material[1],
        },
        "signals": signals,
    }


def _load_evaluation(path: Path) -> dict[str, Any]:
    report = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    expected_report_hash = report.get("report_hash")
    actual_report_hash = stable_payload_hash(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    if expected_report_hash != actual_report_hash:
        raise ValueError(f"Evaluation report hash mismatch: {path}")
    for scenario, identity in report["scenario_files"].items():
        scenario_path = Path(identity["path"])
        if sha256_file(scenario_path) != identity["sha256"]:
            raise ValueError(f"Evaluation scenario hash mismatch: {scenario} in {path}")
    if report.get("debug_only") is not True or report.get("formal_ranking_eligible") is not False:
        raise ValueError(f"Evaluation is not marked debug-only: {path}")
    if report["formal_model_sha256_before"] != report["formal_model_sha256_after"]:
        raise ValueError(f"Formal model changed during evaluation: {path}")
    aggregate = report["aggregate"]
    return {
        "path": str((path / "summary.json").resolve()),
        "sha256": sha256_file(path / "summary.json"),
        "timing_profile": report["timing_profile"],
        "completed_scenarios": aggregate["completed_scenarios"],
        "survival_fraction": aggregate["survival_fraction"],
        "action_saturation_fraction": aggregate["metrics"]["action_saturation_fraction"],
        "effort_saturation_fraction": aggregate["metrics"]["effort_saturation_fraction"],
        "artifact_hashes_verified": True,
    }


def analyze(root: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    diagnosis = Path(root).resolve()
    traces = diagnosis / "traces"
    closed_isaac = _load_run(traces / "closed-loop" / "isaac")
    zero_isaac = _load_run(traces / "zero-action" / "isaac")
    single_isaac = _load_run(traces / "single-action" / "isaac")
    closed_mujoco = _load_run(traces / "closed-loop" / "mujoco-formal-v2")
    zero_mujoco = _load_run(traces / "zero-action" / "mujoco-formal-v2")
    replay_mujoco = _load_run(traces / "open-loop-replay" / "mujoco-formal-v2")
    single_mujoco = _load_run(traces / "single-action" / "mujoco-formal-v2")

    comparisons = {
        "zero_action_free_response": compare_control_runs(
            zero_isaac,
            zero_mujoco,
            require_identical_applied_actions=True,
        ),
        "single_identical_action": compare_control_runs(
            single_isaac,
            single_mujoco,
            require_identical_applied_actions=True,
        ),
        "single_identical_action_substeps": compare_single_action_substeps(
            single_isaac,
            single_mujoco,
        ),
        "isaac_action_open_loop_replay": compare_control_runs(
            closed_isaac,
            replay_mujoco,
            require_identical_applied_actions=True,
        ),
        "closed_loop_first_20_ticks": compare_control_runs(
            closed_isaac,
            closed_mujoco,
            require_identical_applied_actions=False,
        ),
    }
    timing = {
        "dreamwaq_formal_1ms": _load_evaluation(diagnosis / "timing-formal-1ms"),
        "dreamwaq_hold_5ms": _load_evaluation(diagnosis / "timing-hold-5ms"),
        "dreamwaq_sync_5ms": _load_evaluation(diagnosis / "timing-gate-5ms"),
        "phase1r_run03_formal_1ms": _load_evaluation(diagnosis / "phase1r-run03-formal-1ms"),
        "phase1r_run03_sync_5ms": _load_evaluation(diagnosis / "phase1r-run03-sync-5ms"),
    }
    project = Path(project_root).resolve()
    formal_hashes = {
        relative: {
            "expected": expected,
            "actual": sha256_file(project / relative),
            "unchanged": sha256_file(project / relative) == expected,
        }
        for relative, expected in FORMAL_INPUT_HASHES.items()
    }
    single_substep = comparisons["single_identical_action_substeps"]
    open_loop = comparisons["isaac_action_open_loop_replay"]
    closed_loop = comparisons["closed_loop_first_20_ticks"]
    zero_action = comparisons["zero_action_free_response"]
    conclusions = {
        "timing_sync_solved_failure": timing["dreamwaq_sync_5ms"]["completed_scenarios"] == 8,
        "torque_refresh_cadence_solved_failure": timing["dreamwaq_hold_5ms"]["completed_scenarios"] == 8,
        "reset_history_actor_input_aligned": max(
            zero_action["reset_alignment"]["current_actor_observation"],
            zero_action["reset_alignment"]["policy_input"],
        ) < 1.0e-6,
        "first_action_and_pd_reference_aligned": max(
            single_substep["initial_pre_step_alignment"].values()
        ) < 1.0e-4,
        "plant_diverges_before_second_policy_inference": single_substep[
            "first_material_divergence"
        ]
        is not None,
        "plant_diverges_with_zero_applied_action": zero_action["first_material_divergence"]
        is not None,
        "plant_diverges_under_identical_isaac_actions": open_loop["first_material_divergence"]
        is not None
        and open_loop["applied_actions_identical"],
        "closed_loop_policy_feedback_amplifies_divergence": closed_loop["signals"][
            "action_clipped"
        ]["max_abs_error"]
        > 0.10,
        "phase1r_timing_sync_is_not_a_general_improvement": timing[
            "phase1r_run03_sync_5ms"
        ]["survival_fraction"]
        < timing["phase1r_run03_formal_1ms"]["survival_fraction"],
        "trace_artifact_hashes_and_dimensions_verified": True,
        "evaluation_artifact_hashes_verified": True,
        "formal_inputs_unchanged": all(item["unchanged"] for item in formal_hashes.values()),
        "evidence_ranked_attribution": [
            "cross_engine plant/actuator/contact/constraint dynamics gap appears within the first physical interval",
            "DreamWaQ history, CENet, and actor outputs initially match and then respond to the divergent state",
            "closed-loop action scaling and saturation amplify the plant divergence, but do not originate it",
            "physics timestep alone is not the root cause and is not a valid formal fix",
        ],
    }
    report = {
        "schema_version": "DreamWaQBroadSim2SimDiagnosisV1",
        "debug_only": True,
        "formal_ranking_eligible": False,
        "diagnosis_root": str(diagnosis),
        "comparisons": comparisons,
        "timing_and_baseline": timing,
        "formal_input_hashes": formal_hashes,
        "conclusions": conclusions,
    }
    report["report_hash"] = stable_payload_hash(report)
    return report


def _markdown(report: dict[str, Any]) -> str:
    timing = report["timing_and_baseline"]
    comparisons = report["comparisons"]
    single = comparisons["single_identical_action_substeps"]
    zero = comparisons["zero_action_free_response"]
    replay = comparisons["isaac_action_open_loop_replay"]
    closed = comparisons["closed_loop_first_20_ticks"]
    lines = [
        "# DreamWaQ Run-01 Broad Sim2Sim Diagnosis",
        "",
        "> Debug-only evidence. Not eligible for formal evaluation or ranking.",
        "",
        "## Timing Gate",
        "",
        "| Policy / timing | Completed | Mean survival | Action saturation |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in (
        "dreamwaq_formal_1ms",
        "dreamwaq_hold_5ms",
        "dreamwaq_sync_5ms",
        "phase1r_run03_formal_1ms",
        "phase1r_run03_sync_5ms",
    ):
        item = timing[key]
        lines.append(
            f"| {key} | {item['completed_scenarios']}/8 | {item['survival_fraction']:.6f} | "
            f"{item['action_saturation_fraction']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Earliest Boundary",
            "",
            f"- Reset current/history max error: "
            f"{zero['reset_alignment']['policy_input']:.9g}.",
            f"- Initial identical-action target max error: "
            f"{single['initial_pre_step_alignment']['target_command_canonical']:.9g}.",
            f"- Initial PD reference max error: "
            f"{single['initial_pre_step_alignment']['pd_torque_effort_clipped_canonical']:.9g} Nm.",
            f"- First material single-action divergence: {single['first_material_divergence']}.",
            "",
            "## Isolation Results",
            "",
            f"- Zero action first material divergence: {zero['first_material_divergence']}.",
            f"- Open-loop replay action max error: {replay['applied_action_max_abs_error']:.9g}.",
            f"- Open-loop first material divergence: {replay['first_material_divergence']}.",
            f"- Closed-loop first material divergence: {closed['first_material_divergence']}.",
            f"- Closed-loop action max difference: "
            f"{closed['signals']['action_clipped']['max_abs_error']:.9g}.",
            "- Trace sidecar hashes, dimensions, row counts, and substep continuity: verified.",
            "- Timing/baseline report hashes and all scenario CSV hashes: verified.",
            "",
            "## Conclusion",
            "",
            "The reset, history, first actor output, action target, and initial PD reference align. "
            "A material state difference appears inside the first physical interval, before the "
            "second policy inference. Zero-action and identical-action replay retain the divergence, "
            "so CENet is responding to a plant mismatch rather than creating it. Closed-loop policy "
            "feedback then amplifies that mismatch. Matching the MuJoCo timestep to Isaac does not "
            "solve the failure and substantially degrades Phase 1R run-03, so it must not replace the "
            "formal timing contract.",
            "",
            f"Report hash: `{report['report_hash']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the DreamWaQ broad sim2sim diagnosis.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", report)
    (output / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
