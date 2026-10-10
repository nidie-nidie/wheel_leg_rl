from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .analysis import P50_IMPACT_OFFSETS_MS, explanation_ratio, first_persistent_divergence
from .contracts import (
    CandidateStatus,
    EvidenceIntegrityError,
    canonical_json_bytes,
    require_path_within,
    sha256_file,
    stable_hash,
)
from .report import write_report_bundle
from .repeatability import MATERIAL_FLOORS, validate_threshold_snapshot
from .scenario_catalog import catalog
from .verdict import build_verdict


CORE_STAGES = (
    "G00_integrity",
    "G01_repeatability",
    "G02_adapter",
    "G03_instrumentation",
    "P10_rest",
    "P20_static_properties",
    "P30_actuator",
    "P40_closure",
    "P50_contact",
    "P60_full_robot",
    "C70_checkpoint",
)
EXTENDED_PROBES = ["P20-D", "P40-C", "P50-B", "P50-D", "P60-A", "P60-E"]
ADAPTER_TERMINAL_STAGES = CORE_STAGES[:4]


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(payload) + b"\n")
    os.replace(temporary, path)


def _load_stages(
    verified_attempts: Mapping[str, Path],
) -> dict[str, tuple[Path, dict[str, Any]]]:
    if set(verified_attempts) != set(CORE_STAGES):
        raise EvidenceIntegrityError("Finalizer did not receive the exact Core stage set")
    stages: dict[str, tuple[Path, dict[str, Any]]] = {}
    for stage in CORE_STAGES:
        attempt = Path(verified_attempts[stage]).resolve(strict=True)
        worker = attempt / "worker"
        result_path = worker / "result.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != "RootCauseCoreStageResultV1"
            or payload.get("stage") != stage
            or payload.get("passed") is not True
        ):
            raise EvidenceIntegrityError(f"Selected stage result is invalid: {stage}")
        stages[stage] = (worker, payload)
    return stages


def _selected_identity(verified_attempts: Mapping[str, Path]) -> dict[str, Any]:
    rows = {
        stage: {
            "attempt": Path(attempt).name,
            "stage_state_sha256": sha256_file(Path(attempt) / "stage_state.json"),
            "result_sha256": sha256_file(Path(attempt) / "worker" / "result.json"),
        }
        for stage, attempt in sorted(verified_attempts.items())
    }
    return {"stages": rows, "identity_hash": stable_hash(rows)}


def _load_stage_subset(
    verified_attempts: Mapping[str, Path], expected_stages: tuple[str, ...]
) -> dict[str, tuple[Path, dict[str, Any]]]:
    if set(verified_attempts) != set(expected_stages):
        raise EvidenceIntegrityError("Finalizer stage subset is not exact")
    stages: dict[str, tuple[Path, dict[str, Any]]] = {}
    for stage in expected_stages:
        attempt = Path(verified_attempts[stage]).resolve(strict=True)
        worker = attempt / "worker"
        payload = json.loads((worker / "result.json").read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != "RootCauseCoreStageResultV1"
            or payload.get("stage") != stage
            or payload.get("passed") is not True
        ):
            raise EvidenceIntegrityError(f"Selected stage result is invalid: {stage}")
        stages[stage] = (worker, payload)
    return stages


def finalize_adapter_terminal(
    run_root: Path, *, verified_attempts: Mapping[str, Path]
) -> dict[str, Any]:
    root = Path(run_root).resolve(strict=True)
    stages = _load_stage_subset(verified_attempts, ADAPTER_TERMINAL_STAGES)
    selected_identity = _selected_identity(verified_attempts)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    manifest_identity_hash = manifest.get("manifest_identity_hash")
    if not isinstance(manifest_identity_hash, str):
        raise EvidenceIntegrityError("Run manifest identity hash is missing")

    g00_worker, g00 = stages["G00_integrity"]
    g02_worker, g02 = stages["G02_adapter"]
    adapter_gate = g02.get("adapter_gate")
    if not isinstance(adapter_gate, Mapping) or not (
        g02.get("terminal_adapter_bug") is True
        and adapter_gate.get("evidence_valid") is True
        and adapter_gate.get("digital_chain_passed") is False
        and adapter_gate.get("terminal_adapter_bug") is True
    ):
        raise EvidenceIntegrityError("G02 result is not a terminal adapter diagnosis")

    formal_after = {
        name: sha256_file(Path(record["path"]))
        for name, record in g00["formal_hashes"].items()
    }
    formal_unchanged = all(
        formal_after[name] == record["actual_sha256"] == record["expected_sha256"]
        for name, record in g00["formal_hashes"].items()
    )
    g00_passed = bool(all(g00["gates"].values()) and formal_unchanged)
    if not g00_passed:
        raise EvidenceIntegrityError(
            "Terminal adapter diagnosis requires a valid G00 identity boundary"
        )

    evidence = [
        _evidence_row(
            stage="G00_integrity",
            probe="G00_IDENTITY_AND_SCOPE",
            signal="formal_identity_scope_and_write_boundary",
            status="pass",
            artifact=g00_worker / "result.json",
            signal_path="$.gates",
            observed_value=g00["gates"],
            threshold={
                "all_stage_gates": True,
                "formal_file_hashes_unchanged": True,
            },
            supports="valid_evidence_pipeline",
            guard_status={
                "stage_gates": g00["gates"],
                "formal_file_hashes_unchanged": formal_unchanged,
            },
        ),
        _evidence_row(
            stage="G02_adapter",
            probe="G02_DUAL_ENGINE_ADAPTER_CHAIN",
            signal="reset_history_actor_target_clock_and_polarity",
            status=str(adapter_gate["classification"]),
            artifact=g02_worker / "result.json",
            signal_path="$.adapter_gate.classification",
            observed_value=adapter_gate["classification"],
            threshold={"digital_chain_passed": True},
            supports="SIM2SIM_ADAPTER_BUG",
            guard_status={
                "failures": adapter_gate.get("failures", []),
                "deferred_failures": adapter_gate.get("deferred_failures", []),
                "gate_groups": adapter_gate.get("gate_groups", {}),
            },
        ),
    ]
    evidence_refs = _build_evidence_refs(root, evidence, selected_identity)
    public_evidence = [_public_evidence_row(row) for row in evidence]
    analysis = {
        "schema_version": "RootCauseCoreAnalysisV2",
        "run_id": root.name,
        "manifest_identity_hash": manifest_identity_hash,
        "selected_attempts_identity": selected_identity,
        "failed_gates": ["G02_adapter_digital_chain"],
        "candidates": {},
        "confidence": "high",
        "first_material_divergence": {
            "stage": "G02_adapter",
            "classification": adapter_gate["classification"],
        },
        "executed_core_probes": [],
        "unexecuted_extended_probes": EXTENDED_PROBES,
        "checkpoint_role": "not_evaluated",
        "evidence_refs": evidence_refs,
        "remaining_uncertainty": [
            "physical_core_probes_not_run_after_terminal_adapter_failure"
        ],
        "formal_file_hashes_unchanged": formal_unchanged,
        "formal_file_hashes_before": {
            name: record["actual_sha256"]
            for name, record in g00["formal_hashes"].items()
        },
        "formal_file_hashes_after": formal_after,
        "key_findings": [
            {
                "finding": "terminal_adapter_bug",
                "classification": adapter_gate["classification"],
                "failures": adapter_gate.get("failures", []),
            }
        ],
        "threshold_snapshot": {},
        "decision_thresholds": {"digital_chain_passed": True},
        "stage_result_sha256": {
            stage: sha256_file(worker / "result.json")
            for stage, (worker, _) in stages.items()
        },
    }
    analysis_root = root / "analysis"
    analysis_path = analysis_root / "analysis.json"
    evidence_path = analysis_root / "evidence.json"
    _atomic_json(analysis_path, analysis)
    _atomic_json(evidence_path, public_evidence)
    verdict = build_verdict(analysis)
    bundle = write_report_bundle(root / "report", verdict, public_evidence)
    artifacts = {
        "analysis/analysis.json": sha256_file(analysis_path),
        "analysis/evidence.json": sha256_file(evidence_path),
        "report/verdict.json": sha256_file(bundle.verdict),
        "report/evidence.csv": sha256_file(bundle.evidence),
        "report/report.md": sha256_file(bundle.report),
        "report/file_hashes.json": sha256_file(bundle.hashes),
    }
    final_state = {
        "schema_version": "RootCauseFinalStateV2",
        "run_id": root.name,
        "manifest_identity_hash": manifest_identity_hash,
        "selected_attempts_identity": selected_identity,
        "verdict": verdict["primary_classification"],
        "checkpoint_role": "not_evaluated",
        "formal_file_hashes_unchanged": formal_unchanged,
        "artifact_hashes": artifacts,
    }
    final_state["final_state_identity_hash"] = stable_hash(final_state)
    _atomic_json(root / "final_state.json", final_state)
    return final_state


def finalize_repeatability_terminal(
    run_root: Path, *, verified_attempts: Mapping[str, Path]
) -> dict[str, Any]:
    root = Path(run_root).resolve(strict=True)
    stages = _load_stage_subset(verified_attempts, ADAPTER_TERMINAL_STAGES)
    selected_identity = _selected_identity(verified_attempts)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    manifest_identity_hash = manifest.get("manifest_identity_hash")
    if not isinstance(manifest_identity_hash, str):
        raise EvidenceIntegrityError("Run manifest identity hash is missing")

    g00_worker, g00 = stages["G00_integrity"]
    g01_worker, g01 = stages["G01_repeatability"]
    if not (
        g01.get("exact_executed_coverage") is True
        and g01.get("repeatability_usable") is False
    ):
        raise EvidenceIntegrityError(
            "G01 result is not a valid unavailable-repeatability terminal"
        )
    formal_after = {
        name: sha256_file(Path(record["path"]))
        for name, record in g00["formal_hashes"].items()
    }
    formal_unchanged = all(
        formal_after[name] == record["actual_sha256"] == record["expected_sha256"]
        for name, record in g00["formal_hashes"].items()
    )
    g00_passed = bool(all(g00["gates"].values()) and formal_unchanged)
    if not g00_passed:
        raise EvidenceIntegrityError(
            "Repeatability terminal requires a valid G00 identity boundary"
        )

    _, snapshot = _load_g01_threshold_snapshot(g01_worker, g01)
    unusable_reasons = {
        str(key): list(record["unusable_reasons"])
        for key, record in snapshot["keys"].items()
        if record["usable"] is False
    }
    if not unusable_reasons:
        raise EvidenceIntegrityError(
            "G01 unavailable-repeatability terminal has no unusable snapshot key"
        )

    evidence = [
        _evidence_row(
            stage="G00_integrity",
            probe="G00_IDENTITY_AND_SCOPE",
            signal="formal_identity_scope_and_write_boundary",
            status="pass",
            artifact=g00_worker / "result.json",
            signal_path="$.gates",
            observed_value=g00["gates"],
            threshold={
                "all_stage_gates": True,
                "formal_file_hashes_unchanged": True,
            },
            supports="valid_evidence_pipeline",
            guard_status={
                "stage_gates": g00["gates"],
                "formal_file_hashes_unchanged": formal_unchanged,
            },
        ),
        _evidence_row(
            stage="G01_repeatability",
            probe="G01_FROZEN_THRESHOLD_SNAPSHOT",
            signal="per_key_repeatability_envelopes",
            status="unavailable",
            artifact=g01_worker / "result.json",
            signal_path="$.repeatability_usable",
            observed_value=False,
            threshold=True,
            supports="nondeterministic_evidence",
            guard_status={
                "exact_executed_coverage": True,
                "unusable_reasons": unusable_reasons,
            },
        ),
    ]
    evidence_refs = _build_evidence_refs(root, evidence, selected_identity)
    public_evidence = [_public_evidence_row(row) for row in evidence]
    analysis = {
        "schema_version": "RootCauseCoreAnalysisV2",
        "run_id": root.name,
        "manifest_identity_hash": manifest_identity_hash,
        "selected_attempts_identity": selected_identity,
        "failed_gates": ["G01_repeatability_unavailable"],
        "candidates": {},
        "confidence": "low",
        "first_material_divergence": {},
        "executed_core_probes": [],
        "unexecuted_extended_probes": EXTENDED_PROBES,
        "checkpoint_role": "not_evaluated",
        "evidence_refs": evidence_refs,
        "remaining_uncertainty": [
            "physical_core_probes_not_run_without_usable_repeatability_thresholds"
        ],
        "formal_file_hashes_unchanged": formal_unchanged,
        "formal_file_hashes_before": {
            name: record["actual_sha256"]
            for name, record in g00["formal_hashes"].items()
        },
        "formal_file_hashes_after": formal_after,
        "key_findings": [
            {
                "finding": "repeatability_unavailable",
                "unusable_reasons": unusable_reasons,
            }
        ],
        "threshold_snapshot": {
            "identity_hash": snapshot["identity_hash"],
            "repeatability_usable": False,
            "unusable_reasons": unusable_reasons,
        },
        "decision_thresholds": {
            "all_verdict_repeatability_keys_usable": True
        },
        "stage_result_sha256": {
            stage: sha256_file(worker / "result.json")
            for stage, (worker, _) in stages.items()
        },
    }
    analysis_root = root / "analysis"
    analysis_path = analysis_root / "analysis.json"
    evidence_path = analysis_root / "evidence.json"
    _atomic_json(analysis_path, analysis)
    _atomic_json(evidence_path, public_evidence)
    verdict = build_verdict(analysis)
    bundle = write_report_bundle(root / "report", verdict, public_evidence)
    artifacts = {
        "analysis/analysis.json": sha256_file(analysis_path),
        "analysis/evidence.json": sha256_file(evidence_path),
        "report/verdict.json": sha256_file(bundle.verdict),
        "report/evidence.csv": sha256_file(bundle.evidence),
        "report/report.md": sha256_file(bundle.report),
        "report/file_hashes.json": sha256_file(bundle.hashes),
    }
    final_state = {
        "schema_version": "RootCauseFinalStateV2",
        "run_id": root.name,
        "manifest_identity_hash": manifest_identity_hash,
        "selected_attempts_identity": selected_identity,
        "verdict": verdict["primary_classification"],
        "checkpoint_role": "not_evaluated",
        "formal_file_hashes_unchanged": formal_unchanged,
        "artifact_hashes": artifacts,
    }
    final_state["final_state_identity_hash"] = stable_hash(final_state)
    _atomic_json(root / "final_state.json", final_state)
    return final_state


def _p30_status(analyses: Mapping[str, Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    pairs: dict[str, Any] = {}
    supported = False
    inconclusive = False
    for label in ("open", "closed"):
        direct = analyses[f"p30_{label}_direct"]
        target = analyses[f"p30_{label}_target"]
        ratio = explanation_ratio(
            float(target["normalized_rmse"]),
            float(direct["normalized_rmse"]),
            tolerance=float(target["normalized_effective_tolerance"]),
        )
        pairs[label] = {
            "direct": direct,
            "target": target,
            "explanation_ratio": ratio,
        }
        if bool(target["material"]) and not bool(direct["material"]) and ratio is not None:
            if ratio >= 0.70:
                supported = True
            elif ratio >= 0.30:
                inconclusive = True
        elif bool(target["material"]) or bool(direct["material"]):
            inconclusive = True
    if supported:
        status = CandidateStatus.SUPPORTED_PRIMARY.value
    elif inconclusive:
        status = CandidateStatus.INCONCLUSIVE.value
    else:
        status = CandidateStatus.NOT_SUPPORTED.value
    return status, pairs


def _strict_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _p10_closure_material(
    p10: Mapping[str, Any], scenario: str
) -> tuple[bool | None, dict[str, Any]]:
    try:
        signal = p10["analyses"][scenario]["signals"]["all_hinge_velocity"]
        mismatch = float(signal["max_abs"])
        repeat_envelope = max(
            float(signal["isaac_repeat_envelope"]),
            float(signal["mujoco_repeat_envelope"]),
        )
    except (KeyError, TypeError, ValueError):
        return None, {"available": False}
    if not np.isfinite(mismatch) or not np.isfinite(repeat_envelope):
        return None, {"available": False}
    effective_tolerance = max(
        float(MATERIAL_FLOORS["all_hinge_velocity"]),
        5.0 * repeat_envelope,
    )
    material = bool(mismatch > effective_tolerance)
    return material, {
        "available": True,
        "metric": "max_abs",
        "value": mismatch,
        "effective_tolerance": effective_tolerance,
        "material": material,
    }


def _closure_status(
    p10: Mapping[str, Any],
    p20: Mapping[str, Any],
    p30: Mapping[str, Any],
    p40: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    p10_on, p10_on_record = _p10_closure_material(p10, "p10_a")
    p10_off, p10_off_record = _p10_closure_material(p10, "p10_c")
    p10_available = p10_on is not None and p10_off is not None
    p10_support = bool(p10_available and p10_on and not p10_off)

    p30_analyses = p30.get("analyses")
    p30_modes: dict[str, Any] = {}
    p30_available = isinstance(p30_analyses, Mapping)
    p30_support = False
    for mode in ("direct", "target"):
        open_material = None
        closed_material = None
        if isinstance(p30_analyses, Mapping):
            open_record = p30_analyses.get(f"p30_open_{mode}")
            closed_record = p30_analyses.get(f"p30_closed_{mode}")
            if isinstance(open_record, Mapping):
                open_material = _strict_bool(open_record.get("material"))
            if isinstance(closed_record, Mapping):
                closed_material = _strict_bool(closed_record.get("material"))
        mode_available = open_material is not None and closed_material is not None
        mode_support = bool(
            mode_available and closed_material and not open_material
        )
        p30_available = bool(p30_available and mode_available)
        p30_support = bool(p30_support or mode_support)
        p30_modes[mode] = {
            "available": mode_available,
            "open_material": open_material,
            "closed_material": closed_material,
            "supports_closure": mode_support,
        }

    p40_analysis = p40.get("analysis")
    p40_on = None
    p40_off = None
    ratio: float | None = None
    if isinstance(p40_analysis, Mapping):
        on_record = p40_analysis.get("closure_on")
        off_record = p40_analysis.get("closure_off")
        if isinstance(on_record, Mapping):
            p40_on = _strict_bool(on_record.get("material"))
        if isinstance(off_record, Mapping):
            p40_off = _strict_bool(off_record.get("material"))
        raw_ratio = p40_analysis.get("explanation_ratio")
        if isinstance(raw_ratio, (int, float)) and not isinstance(raw_ratio, bool):
            candidate_ratio = float(raw_ratio)
            if np.isfinite(candidate_ratio):
                ratio = candidate_ratio
    p40_available = p40_on is not None and p40_off is not None
    p40_support = bool(p40_available and p40_on and not p40_off)
    p40_strong = bool(p40_support and ratio is not None and ratio >= 0.70)

    probe_support = {
        "P10_rest_closure_on_to_off": p10_support,
        "P30_closed_to_open": p30_support,
        "P40_closure_on_to_off": p40_support,
    }
    support_count = sum(probe_support.values())
    contact_free = {
        "P10_rest": bool(p10.get("contact_exclusion", {}).get("passed") is True),
        "P30_actuator": bool(
            p30.get("contact_exclusion", {}).get("passed") is True
        ),
        "P40_closure": bool(
            p40.get("contact_exclusion", {}).get("passed") is True
        ),
    }
    p20_audit_valid = p20.get("audit_valid") is True
    p20_golden_valid = p20.get("golden_valid") is True
    p20_no_static_hard_failure = p20.get("mismatch_detected") is False
    evidence_available = bool(p10_available and p30_available and p40_available)
    diagnostics = {
        "probe_support": probe_support,
        "support_count": support_count,
        "p10": {
            "available": p10_available,
            "closure_on": p10_on_record,
            "closure_off": p10_off_record,
        },
        "p30": {"available": p30_available, "modes": p30_modes},
        "p40": {
            "available": p40_available,
            "closure_on_material": p40_on,
            "closure_off_material": p40_off,
            "explanation_ratio": ratio,
            "strong_ratio_support": p40_strong,
        },
        "preconditions": {
            "evidence_available": evidence_available,
            "p20_static_audit_valid": p20_audit_valid,
            "p20_golden_valid": p20_golden_valid,
            "p20_no_static_hard_failure": p20_no_static_hard_failure,
            "contact_free": contact_free,
            "all_contact_free": all(contact_free.values()),
        },
    }

    if (
        not evidence_available
        or not p20_audit_valid
        or not p20_golden_valid
        or not all(contact_free.values())
    ):
        return CandidateStatus.INCONCLUSIVE.value, diagnostics
    if not p20_no_static_hard_failure:
        if support_count:
            return CandidateStatus.SUPPORTED_CONTRIBUTOR.value, diagnostics
        return CandidateStatus.NOT_SUPPORTED.value, diagnostics
    if p40_strong or support_count >= 2:
        return CandidateStatus.SUPPORTED_PRIMARY.value, diagnostics
    if support_count == 1:
        return CandidateStatus.SUPPORTED_CONTRIBUTOR.value, diagnostics
    return CandidateStatus.NOT_SUPPORTED.value, diagnostics


def _gate_mapping_passed(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value
        and all(item is True for item in value.values())
    )


def _p50_normal_status(impact: Mapping[str, Any]) -> str:
    if not _gate_mapping_passed(impact.get("gate_status")):
        return CandidateStatus.INCONCLUSIVE.value
    valid_count = impact.get("valid_condition_count")
    material_count = impact.get("material_condition_count")
    if (
        not isinstance(valid_count, int)
        or isinstance(valid_count, bool)
        or not isinstance(material_count, int)
        or isinstance(material_count, bool)
        or valid_count != 6
        or material_count < 0
        or material_count > valid_count
    ):
        return CandidateStatus.INCONCLUSIVE.value
    if material_count >= 2:
        return CandidateStatus.SUPPORTED_PRIMARY.value
    if material_count == 1:
        return CandidateStatus.SUPPORTED_CONTRIBUTOR.value
    return CandidateStatus.NOT_SUPPORTED.value


def _p50_tangential_status(slide: Mapping[str, Any]) -> str:
    if not _gate_mapping_passed(slide.get("gate_status")):
        return CandidateStatus.INCONCLUSIVE.value
    nominal = slide.get("nominal")
    zero = slide.get("zero_friction")
    if not isinstance(nominal, Mapping) or not isinstance(zero, Mapping):
        return CandidateStatus.INCONCLUSIVE.value
    nominal_material = nominal.get("material")
    zero_material = zero.get("material")
    if not isinstance(nominal_material, bool) or not isinstance(zero_material, bool):
        return CandidateStatus.INCONCLUSIVE.value
    if not nominal_material:
        return CandidateStatus.NOT_SUPPORTED.value
    ratio = slide.get("explanation_ratio")
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
        return CandidateStatus.INCONCLUSIVE.value
    ratio_value = float(ratio)
    if not np.isfinite(ratio_value):
        return CandidateStatus.INCONCLUSIVE.value
    if ratio_value >= 0.70:
        return (
            CandidateStatus.SUPPORTED_PRIMARY.value
            if not zero_material
            else CandidateStatus.SUPPORTED_CONTRIBUTOR.value
        )
    if ratio_value >= 0.30:
        return CandidateStatus.SUPPORTED_CONTRIBUTOR.value
    return CandidateStatus.NOT_SUPPORTED.value


def _first_contact_divergence(impact: Mapping[str, Any]) -> dict[str, Any]:
    best: tuple[int, dict[str, Any]] | None = None
    post_mask = P50_IMPACT_OFFSETS_MS >= 0.0
    post_offsets = P50_IMPACT_OFFSETS_MS[post_mask]
    for condition in impact["conditions"]:
        if not condition["material_postcontact_failure"]:
            continue
        isaac = np.asarray(condition["isaac"]["median_velocity_z"], dtype=np.float64)[
            post_mask
        ]
        mujoco = np.asarray(condition["mujoco"]["median_velocity_z"], dtype=np.float64)[
            post_mask
        ]
        tolerance = float(condition["postcontact"]["effective_tolerance"])
        index = first_persistent_divergence(np.abs(isaac - mujoco) > tolerance, width=3)
        if index is None:
            continue
        row = {
            "stage": "P50_contact",
            "probe": "P50_A_SPHERE_IMPACT",
            "layer": "common_sphere_post_contact",
            "contact_relative_time_ms": int(post_offsets[index]),
            "height_m": condition["height_m"],
            "vertical_velocity_mps": condition["vertical_velocity_mps"],
            "effective_tolerance": tolerance,
        }
        if best is None or int(post_offsets[index]) < best[0]:
            best = (int(post_offsets[index]), row)
    return {} if best is None else best[1]


def _pair_columns(pair: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "baseline_scenario_id": pair.get("baseline_scenario_id"),
        "baseline_configuration_hash": pair.get("baseline_configuration_hash"),
        "baseline_configuration_semantics_hash": pair.get(
            "baseline_configuration_semantics_hash"
        ),
        "baseline_pre_forward_initial_condition_hash": pair.get(
            "baseline_pre_forward_initial_condition_hash"
        ),
        "baseline_post_forward_state_hash": pair.get("baseline_post_forward_state_hash"),
        "baseline_reset_returned_policy_hash": pair.get(
            "baseline_reset_returned_policy_hash"
        ),
        "baseline_excitation_hash": pair.get("baseline_excitation_hash"),
        "ablation_scenario_id": pair.get("ablation_scenario_id"),
        "ablation_configuration_hash": pair.get("ablation_configuration_hash"),
        "ablation_configuration_semantics_hash": pair.get(
            "ablation_configuration_semantics_hash"
        ),
        "ablation_pre_forward_initial_condition_hash": pair.get(
            "ablation_pre_forward_initial_condition_hash"
        ),
        "ablation_post_forward_state_hash": pair.get("ablation_post_forward_state_hash"),
        "ablation_reset_returned_policy_hash": pair.get(
            "ablation_reset_returned_policy_hash"
        ),
        "ablation_excitation_hash": pair.get("ablation_excitation_hash"),
        "comparison_profile_hash": pair.get("comparison_profile_hash"),
        "semantic_diff_paths_sha256": pair.get("semantic_diff_paths_sha256"),
        "allowed_ablation_factors": pair.get("allowed_ablation_factors"),
        "expected_improvement_direction": pair.get("expected_improvement_direction"),
    }


def _evidence_row(
    *,
    stage: str,
    probe: str,
    signal: str,
    status: str,
    artifact: Path,
    signal_path: str,
    observed_value: Any,
    threshold: Any,
    supports: str = "",
    excludes: str = "",
    time_window_ms: Any = "",
    effective_tolerance: float | None = None,
    explanation: float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    artifact_path = Path(artifact).resolve(strict=True)
    row = {
        "stage": stage,
        "probe": probe,
        "engine_pair": "Isaac-MuJoCo",
        "signal": signal,
        "time_window_ms": time_window_ms,
        "status": status,
        "supports": supports,
        "excludes": excludes,
        "effective_tolerance": effective_tolerance,
        "explanation_ratio": explanation,
        "availability": "available",
        "artifact_sha256": sha256_file(artifact_path),
        "_artifact_path": str(artifact_path),
        "_signal_path": signal_path,
        "_observed_value": observed_value,
        "_threshold": threshold,
    }
    row.update(extra)
    return row


def _closure_evidence_rows(
    *,
    closure_status: str,
    closure_diagnostics: Mapping[str, Any],
    p10_worker: Path,
    p10: Mapping[str, Any],
    p20_worker: Path,
    p20: Mapping[str, Any],
    p30_worker: Path,
    p30: Mapping[str, Any],
    p40_worker: Path,
    p40: Mapping[str, Any],
) -> list[dict[str, Any]]:
    claim = "CLOSED_CHAIN_CONSTRAINT_MISMATCH"
    final_supported = closure_status in {
        CandidateStatus.SUPPORTED_PRIMARY.value,
        CandidateStatus.SUPPORTED_CONTRIBUTOR.value,
    }
    final_excluded = closure_status == CandidateStatus.NOT_SUPPORTED.value

    def relation(*, contributes: bool, vote_input: bool = True) -> tuple[str, str]:
        supports = claim if final_supported and contributes else ""
        excludes = claim if final_excluded and vote_input else ""
        return supports, excludes

    rows: list[dict[str, Any]] = []
    probe_support = closure_diagnostics["probe_support"]
    p10_diagnostics = closure_diagnostics["p10"]
    for scenario, role in (("p10_a", "on"), ("p10_c", "off")):
        signal = p10["analyses"][scenario]["signals"]["all_hinge_velocity"]
        derived = p10_diagnostics[f"closure_{role}"]
        contributes = bool(probe_support["P10_rest_closure_on_to_off"])
        supports, excludes = relation(contributes=contributes)
        for field in (
            "max_abs",
            "isaac_repeat_envelope",
            "mujoco_repeat_envelope",
        ):
            is_mismatch = field == "max_abs"
            rows.append(
                _evidence_row(
                    stage="P10_rest",
                    probe=f"CLOSURE_AGGREGATE_P10_{role.upper()}",
                    signal=f"all_hinge_velocity_{field}",
                    status=(
                        "material"
                        if is_mismatch and derived["material"]
                        else "within_tolerance"
                        if is_mismatch
                        else "threshold_input"
                    ),
                    artifact=p10_worker / "analysis" / f"{scenario}.json",
                    signal_path=f"$.signals.all_hinge_velocity.{field}",
                    observed_value=signal[field],
                    threshold=(
                        derived["effective_tolerance"]
                        if is_mismatch
                        else {
                            "rule": "max(material_floor,5*max_repeat_envelope)",
                            "material_floor": MATERIAL_FLOORS[
                                "all_hinge_velocity"
                            ],
                        }
                    ),
                    supports=supports,
                    excludes=excludes,
                    time_window_ms=p10["analyses"][scenario]["common_times_ms"],
                    effective_tolerance=(
                        float(derived["effective_tolerance"])
                        if is_mismatch
                        else None
                    ),
                    scalar_metric=field,
                    channel_reduction="max_abs",
                    guard_status={
                        "aggregate_status": closure_status,
                        "derived_material": derived["material"],
                        "probe_support": contributes,
                    },
                )
            )

    p30_diagnostics = closure_diagnostics["p30"]["modes"]
    for mode in ("direct", "target"):
        contributes = bool(p30_diagnostics[mode]["supports_closure"])
        supports, excludes = relation(contributes=contributes)
        for closure_state in ("open", "closed"):
            name = f"p30_{closure_state}_{mode}"
            item = p30["analyses"][name]
            rows.append(
                _evidence_row(
                    stage="P30_actuator",
                    probe=(
                        f"CLOSURE_AGGREGATE_P30_{mode.upper()}_"
                        f"{closure_state.upper()}"
                    ),
                    signal="material",
                    status="material" if item["material"] else "within_tolerance",
                    artifact=p30_worker / "analysis" / f"{name}.json",
                    signal_path="$.material",
                    observed_value=item["material"],
                    threshold={
                        "closure_support_rule": "closed_material_and_open_not_material",
                        "mode": mode,
                    },
                    supports=supports,
                    excludes=excludes,
                    time_window_ms=item.get("common_times_ms", ""),
                    guard_status={
                        "aggregate_status": closure_status,
                        "mode_support": contributes,
                    },
                )
            )

    p40_analysis = p40["analysis"]
    p40_support = bool(probe_support["P40_closure_on_to_off"])
    p40_supports, p40_excludes = relation(contributes=p40_support)
    for closure_state in ("on", "off"):
        material = p40_analysis[f"closure_{closure_state}"]["material"]
        rows.append(
            _evidence_row(
                stage="P40_closure",
                probe=f"CLOSURE_AGGREGATE_P40_{closure_state.upper()}",
                signal="material",
                status="material" if material else "within_tolerance",
                artifact=p40_worker / "analysis" / "p40.json",
                signal_path=f"$.closure_{closure_state}.material",
                observed_value=material,
                threshold={
                    "closure_support_rule": "closure_on_material_and_off_not_material"
                },
                supports=p40_supports,
                excludes=p40_excludes,
                time_window_ms=p40_analysis.get("common_times_ms", ""),
                guard_status={
                    "aggregate_status": closure_status,
                    "probe_support": p40_support,
                },
            )
        )
    ratio = p40_analysis.get("explanation_ratio")
    rows.append(
        _evidence_row(
            stage="P40_closure",
            probe="CLOSURE_AGGREGATE_P40_RATIO",
            signal="explanation_ratio",
            status=(
                "strong"
                if isinstance(ratio, (int, float))
                and not isinstance(ratio, bool)
                and float(ratio) >= 0.70
                else "below_strong_threshold"
                if ratio is not None
                else "unavailable"
            ),
            artifact=p40_worker / "analysis" / "p40.json",
            signal_path="$.explanation_ratio",
            observed_value=ratio,
            threshold=0.70,
            supports=p40_supports,
            excludes=p40_excludes,
            time_window_ms=p40_analysis.get("common_times_ms", ""),
            explanation=(
                float(ratio)
                if isinstance(ratio, (int, float)) and not isinstance(ratio, bool)
                else None
            ),
            guard_status={
                "aggregate_status": closure_status,
                "strong_ratio_support": closure_diagnostics["p40"][
                    "strong_ratio_support"
                ],
            },
        )
    )

    preconditions = (
        ("audit_valid", True, "static_property_audit"),
        ("golden_valid", True, "static_property_golden"),
        ("mismatch_detected", False, "no_static_hard_failure"),
    )
    for field, expected, signal in preconditions:
        observed = p20[field]
        contributes = bool(final_supported and observed is expected)
        supports, _ = relation(contributes=contributes, vote_input=False)
        rows.append(
            _evidence_row(
                stage="P20_static_properties",
                probe=f"CLOSURE_AGGREGATE_P20_{field.upper()}",
                signal=signal,
                status="pass" if observed is expected else "fail",
                artifact=p20_worker / "result.json",
                signal_path=f"$.{field}",
                observed_value=observed,
                threshold=expected,
                supports=supports,
                guard_status={"aggregate_status": closure_status},
            )
        )

    contact_inputs = (
        ("P10_rest", p10_worker, p10),
        ("P30_actuator", p30_worker, p30),
        ("P40_closure", p40_worker, p40),
    )
    for stage, worker, payload in contact_inputs:
        passed = payload["contact_exclusion"]["passed"]
        contributes = bool(final_supported and passed is True)
        supports, _ = relation(contributes=contributes, vote_input=False)
        rows.append(
            _evidence_row(
                stage=stage,
                probe=f"CLOSURE_AGGREGATE_{stage.upper()}_CONTACT_EXCLUSION",
                signal="contact_exclusion_passed",
                status="pass" if passed is True else "fail",
                artifact=worker / "result.json",
                signal_path="$.contact_exclusion.passed",
                observed_value=passed,
                threshold=True,
                supports=supports,
                guard_status={"aggregate_status": closure_status},
            )
        )
    return rows


def _public_evidence_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _resolve_json_signal_path(payload: Any, signal_path: str) -> Any:
    if signal_path == "$":
        return payload
    if not signal_path.startswith("$."):
        raise EvidenceIntegrityError(f"Evidence signal path is invalid: {signal_path}")
    current = payload
    for token in signal_path[2:].split("."):
        if not token or not isinstance(current, Mapping) or token not in current:
            raise EvidenceIntegrityError(
                f"Evidence signal path does not exist: {signal_path}"
            )
        current = current[token]
    return current


def _build_evidence_refs(
    root: Path,
    evidence: list[Mapping[str, Any]],
    selected_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    root = Path(root).resolve(strict=True)
    references: list[dict[str, Any]] = []
    stage_identities = selected_identity.get("stages")
    if not isinstance(stage_identities, Mapping):
        raise EvidenceIntegrityError("Selected attempt identity has no stage map")
    for row in evidence:
        stage = str(row.get("stage"))
        selected = stage_identities.get(stage)
        if not isinstance(selected, Mapping) or not isinstance(
            selected.get("attempt"), str
        ):
            raise EvidenceIntegrityError(
                f"Evidence reference has no selected attempt: {stage}"
            )
        attempt_name = str(selected["attempt"])
        if Path(attempt_name).name != attempt_name:
            raise EvidenceIntegrityError(
                f"Evidence selected attempt name is invalid: {stage}/{attempt_name}"
            )
        artifact_value = row.get("_artifact_path")
        if not isinstance(artifact_value, str):
            raise EvidenceIntegrityError("Evidence reference has no artifact path")
        try:
            artifact = require_path_within(
                Path(artifact_value), root, label="Evidence artifact"
            ).resolve(strict=True)
            relative = artifact.relative_to(root).as_posix()
        except (FileNotFoundError, ValueError) as error:
            raise EvidenceIntegrityError(
                f"Evidence artifact is unavailable: {artifact_value}"
            ) from error
        try:
            selected_worker = require_path_within(
                root / "stages" / stage / attempt_name / "worker",
                root,
                label="Evidence selected attempt worker",
            ).resolve(strict=True)
            artifact.relative_to(selected_worker)
        except (FileNotFoundError, ValueError) as error:
            raise EvidenceIntegrityError(
                "Evidence artifact is not owned by the selected attempt: "
                f"{stage}/{attempt_name}/{relative}"
            ) from error
        actual_sha256 = sha256_file(artifact)
        if row.get("artifact_sha256") != actual_sha256:
            raise EvidenceIntegrityError(
                f"Evidence artifact hash drifted: {relative}"
            )
        signal_path = row.get("_signal_path")
        if not isinstance(signal_path, str) or not signal_path.startswith("$"):
            raise EvidenceIntegrityError(
                f"Evidence signal path is invalid: {stage}/{row.get('probe')}"
            )
        if "_observed_value" not in row or "_threshold" not in row:
            raise EvidenceIntegrityError(
                f"Evidence value or threshold is missing: {stage}/{row.get('probe')}"
            )
        try:
            artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EvidenceIntegrityError(
                f"Evidence artifact is not valid JSON: {relative}"
            ) from error
        extracted_value = _resolve_json_signal_path(artifact_payload, signal_path)
        if canonical_json_bytes(extracted_value) != canonical_json_bytes(
            row["_observed_value"]
        ):
            raise EvidenceIntegrityError(
                "Evidence observed value does not match artifact: "
                f"{stage}/{row.get('probe')}/{signal_path}"
            )
        reference = {
            "stage": stage,
            "attempt": selected["attempt"],
            "probe": str(row.get("probe")),
            "artifact": relative,
            "artifact_sha256": actual_sha256,
            "signal": str(row.get("signal")),
            "signal_path": signal_path,
            "time_window_ms": (
                row.get("time_window_ms")
                if row.get("time_window_ms") not in (None, "")
                else "not_applicable"
            ),
            "threshold": row["_threshold"],
            "observed_value": row["_observed_value"],
            "status": str(row.get("status")),
            "supports": str(row.get("supports", "")),
            "excludes": str(row.get("excludes", "")),
        }
        reference["reference_identity_hash"] = stable_hash(reference)
        references.append(reference)
    return references


def _load_g01_threshold_snapshot(
    worker: Path, result: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    snapshot_name = result.get("threshold_snapshot_file")
    if not isinstance(snapshot_name, str) or Path(snapshot_name).name != snapshot_name:
        raise EvidenceIntegrityError("G01 threshold snapshot path is invalid")
    snapshot_path = require_path_within(
        worker / snapshot_name, worker, label="G01 threshold snapshot"
    ).resolve(strict=True)
    if sha256_file(snapshot_path) != result.get("threshold_snapshot_sha256"):
        raise EvidenceIntegrityError("G01 threshold snapshot hash drifted")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    validate_threshold_snapshot(snapshot)
    keys = snapshot["keys"]
    for key, record in keys.items():
        if not isinstance(key, str) or not isinstance(record, Mapping):
            raise EvidenceIntegrityError("G01 threshold snapshot key record is invalid")
        sample_count = record.get("sample_count")
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count <= 0
        ):
            raise EvidenceIntegrityError(
                "G01 threshold snapshot sample count is invalid"
            )
        usable = record.get("usable")
        if not isinstance(usable, bool):
            raise EvidenceIntegrityError("G01 threshold snapshot usable flag is invalid")
        reasons = record.get("unusable_reasons")
        if not isinstance(reasons, list) or not all(
            isinstance(reason, str) and reason for reason in reasons
        ):
            raise EvidenceIntegrityError(
                "G01 threshold snapshot unusable reasons are invalid"
            )
        if not usable and not reasons:
            raise EvidenceIntegrityError(
                "G01 threshold snapshot unusable key has no reason"
            )
    executed_key_count = len(keys)
    executed_sample_count = sum(int(record["sample_count"]) for record in keys.values())
    repeatability_usable = all(bool(record["usable"]) for record in keys.values())
    embedded = result.get("threshold_snapshot")
    if (
        result.get("executed_key_count") != executed_key_count
        or result.get("executed_sample_count") != executed_sample_count
        or result.get("repeatability_usable") is not repeatability_usable
        or not isinstance(embedded, Mapping)
        or embedded.get("identity_hash") != snapshot.get("identity_hash")
    ):
        raise EvidenceIntegrityError("G01 threshold snapshot metadata is inconsistent")
    return snapshot_path, snapshot


def _failed_gates(
    stages: Mapping[str, tuple[Path, Mapping[str, Any]]],
    *,
    formal_unchanged: bool,
) -> list[str]:
    failures: list[str] = []
    for stage in CORE_STAGES[:4]:
        if stages[stage][1].get("passed") is not True:
            failures.append(stage)
    g02 = stages["G02_adapter"][1]
    adapter_gate = g02.get("adapter_gate")
    if not isinstance(adapter_gate, Mapping) or adapter_gate.get(
        "digital_chain_passed"
    ) is not True:
        failures.append("G02_adapter_digital_chain")
    if not formal_unchanged:
        failures.append("G00_formal_file_hashes_changed")
    p20 = stages["P20_static_properties"][1]
    if not bool(p20.get("audit_valid")):
        failures.append("P20_property_audit_invalid")
    if not bool(p20.get("golden_valid")):
        failures.append("P20_golden_invalid")
    p50 = stages["P50_contact"][1]
    for group_name in ("impact", "slide"):
        for name, passed in p50[group_name].get("gate_status", {}).items():
            if not bool(passed):
                failures.append(f"P50_{group_name}_{name}")
    p60 = stages["P60_full_robot"][1]
    if not bool(p60.get("fresh_replay_equivalence", {}).get("passed")):
        failures.append("P60_fresh_replay_equivalence")
    if not bool(p60.get("reset_cache_gate", {}).get("passed")):
        failures.append("P60_reset_cache_identity")
    return sorted(set(failures))


def _replay_columns(p60: Mapping[str, Any]) -> dict[str, Any]:
    identity = p60.get("replay_source_identity")
    if not isinstance(identity, Mapping) or stable_hash(identity) != p60.get(
        "replay_source_identity_hash"
    ):
        raise EvidenceIntegrityError("P60 replay source identity is invalid")
    seal = p60.get("replay_source_seal")
    if not isinstance(seal, Mapping):
        raise EvidenceIntegrityError("P60 replay source seal is missing")
    unsigned_seal = dict(seal)
    seal_identity_hash = unsigned_seal.pop("seal_identity_hash", None)
    if seal_identity_hash != stable_hash(unsigned_seal):
        raise EvidenceIntegrityError("P60 replay source seal is invalid")
    replay_evidence = p60.get("replay_evidence")
    if not isinstance(replay_evidence, Mapping):
        raise EvidenceIntegrityError("P60 replay evidence index is missing")
    if (
        replay_evidence.get("source_result_sha256")
        != seal.get("source_result_sha256")
        or replay_evidence.get("actions_sha256")
        != seal.get("clipped_action_file_sha256")
    ):
        raise EvidenceIntegrityError("P60 replay evidence is not seal-bound")
    return {
        "replay_source_identity_hash": p60["replay_source_identity_hash"],
        "replay_generator_actor_sha256": identity.get("generator_actor_sha256"),
        "replay_generator_manifest_sha256": identity.get("generator_manifest_sha256"),
        "replay_source_scenario_id": identity.get("source_scenario_id"),
        "replay_source_seed": identity.get("seed"),
        "replay_source_repetition_index": identity.get("repetition_index"),
        "replay_environment_count": identity.get("environment_count"),
        "replay_environment_index": identity.get("environment_index"),
        "replay_environment_scenario_name": identity.get("scenario_name"),
        "replay_evaluation_contract_hash": identity.get("evaluation_contract_hash"),
        "replay_reset_cache_file_sha256": identity.get("reset_cache_file_sha256"),
        "replay_reset_cache_tensor_sha256": identity.get("reset_cache_tensor_sha256"),
        "replay_reset_cache_identity_hash": identity.get("reset_cache_identity_hash"),
        "replay_reset_cache_schema": identity.get("cache_schema_version"),
        "replay_reset_cache_root_height_algorithm": identity.get(
            "root_height_algorithm_version"
        ),
        "replay_reset_cache_relaxation_algorithm": identity.get(
            "relaxation_algorithm_version"
        ),
        "replay_action_sequence_sha256": seal.get(
            "clipped_action_sequence_sha256"
        ),
    }


def finalize_run(
    run_root: Path, *, verified_attempts: Mapping[str, Path]
) -> dict[str, Any]:
    root = Path(run_root).resolve(strict=True)
    stages = _load_stages(verified_attempts)
    selected_identity = _selected_identity(verified_attempts)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    manifest_identity_hash = manifest.get("manifest_identity_hash")
    if not isinstance(manifest_identity_hash, str):
        raise EvidenceIntegrityError("Run manifest identity hash is missing")

    g00_worker, g00 = stages["G00_integrity"]
    formal_after = {
        name: sha256_file(Path(record["path"]))
        for name, record in g00["formal_hashes"].items()
    }
    formal_unchanged = all(
        formal_after[name] == record["actual_sha256"] == record["expected_sha256"]
        for name, record in g00["formal_hashes"].items()
    )

    g01_worker, g01 = stages["G01_repeatability"]
    g02_worker, g02 = stages["G02_adapter"]
    g03_worker, g03 = stages["G03_instrumentation"]
    p10_worker, p10 = stages["P10_rest"]
    p20_worker, p20 = stages["P20_static_properties"]
    p30_worker, p30 = stages["P30_actuator"]
    p40_worker, p40 = stages["P40_closure"]
    p50_worker, p50 = stages["P50_contact"]
    p60_worker, p60 = stages["P60_full_robot"]
    c70_worker, c70 = stages["C70_checkpoint"]
    threshold_snapshot_path, threshold_snapshot = _load_g01_threshold_snapshot(
        g01_worker, g01
    )

    failed_gates = _failed_gates(stages, formal_unchanged=formal_unchanged)
    actuator_status, actuator_pairs = _p30_status(p30["analyses"])
    closure_status, closure_diagnostics = _closure_status(p10, p20, p30, p40)
    normal_status = _p50_normal_status(p50["impact"])
    normal_primary = normal_status == CandidateStatus.SUPPORTED_PRIMARY.value
    slide = p50["slide"]
    tangential_status = _p50_tangential_status(slide)
    candidates = {
        "MASS_OR_INERTIA_MISMATCH": (
            CandidateStatus.SUPPORTED_PRIMARY.value
            if p20["mismatch_detected"]
            else CandidateStatus.NOT_SUPPORTED.value
        ),
        "ACTUATOR_INTEGRATION_MISMATCH": actuator_status,
        "CLOSED_CHAIN_CONSTRAINT_MISMATCH": closure_status,
        "CONTACT_OR_FRICTION_MISMATCH:normal": normal_status,
        "CONTACT_OR_FRICTION_MISMATCH:tangential": tangential_status,
    }
    checkpoint_role = str(c70["analysis"]["checkpoint_role"])
    p60_material = {
        name: bool(value["material"]) for name, value in p60["analyses"].items()
    }
    first_divergence = _first_contact_divergence(p50["impact"])
    key_findings = [
        {
            "finding": "common_sphere_normal_contact_mismatch",
            "material_conditions": int(p50["impact"]["material_condition_count"]),
            "valid_conditions": int(p50["impact"]["valid_condition_count"]),
            "gate_status": p50["impact"].get("gate_status", {}),
        },
        {
            "finding": "tangential_friction_ablation",
            "nominal_normalized_rmse": float(slide["nominal"]["normalized_rmse"]),
            "zero_friction_normalized_rmse": float(
                slide["zero_friction"]["normalized_rmse"]
            ),
            "explanation_ratio": slide.get("explanation_ratio"),
            "gate_status": slide.get("gate_status", {}),
        },
        {
            "finding": "closed_chain_constraint_aggregation",
            "status": closure_status,
            "diagnostics": closure_diagnostics,
        },
        {"finding": "full_robot_same_action_diverges", "p60_material": p60_material},
        {
            "finding": "checkpoint_role",
            "role": checkpoint_role,
            "reason": c70["analysis"].get("reason"),
            "anchors": {
                name: value["classification"]
                for name, value in c70["analysis"].get("anchors", {}).items()
            },
        },
    ]
    confidence = (
        "low"
        if failed_gates
        else "high"
        if normal_primary and p60_material["p60_b"] and p60_material["p60_d"]
        else "medium"
        if normal_primary
        else "low"
    )

    evidence: list[dict[str, Any]] = []
    g00_passed = bool(all(g00["gates"].values()) and formal_unchanged)
    evidence.append(
        _evidence_row(
            stage="G00_integrity",
            probe="G00_IDENTITY_AND_SCOPE",
            signal="formal_identity_scope_and_write_boundary",
            status="pass" if g00_passed else "fail",
            artifact=g00_worker / "result.json",
            signal_path="$.gates",
            observed_value=g00["gates"],
            threshold={
                "all_stage_gates": True,
                "formal_file_hashes_unchanged": True,
            },
            supports="valid_evidence_pipeline" if g00_passed else "",
            excludes="identity_or_scope_drift" if g00_passed else "",
            guard_status={
                "stage_gates": g00["gates"],
                "formal_file_hashes_unchanged": formal_unchanged,
            },
        )
    )

    threshold_keys = threshold_snapshot["keys"]
    usable_key_count = sum(bool(record["usable"]) for record in threshold_keys.values())
    evidence.append(
        _evidence_row(
            stage="G01_repeatability",
            probe="G01_FROZEN_THRESHOLD_SNAPSHOT",
            signal="per_key_repeatability_envelopes",
            status="usable" if g01["repeatability_usable"] else "unusable",
            artifact=threshold_snapshot_path,
            signal_path="$.identity_hash",
            observed_value=threshold_snapshot["identity_hash"],
            threshold={
                "minimum_repetitions": threshold_snapshot["minimum_repetitions"],
                "effective_tolerance_rule": threshold_snapshot[
                    "effective_tolerance_rule"
                ],
                "high_noise_rule": threshold_snapshot["high_noise_rule"],
                "all_verdict_keys_usable": True,
            },
            supports=(
                "repeatability_evidence_usable"
                if g01["repeatability_usable"]
                else ""
            ),
            excludes=(
                "nondeterministic_evidence"
                if g01["repeatability_usable"]
                else ""
            ),
            guard_status={
                "exact_executed_coverage": g01["exact_executed_coverage"],
                "repeatability_usable": g01["repeatability_usable"],
                "executed_key_count": g01["executed_key_count"],
                "executed_sample_count": g01["executed_sample_count"],
                "usable_key_count": usable_key_count,
                "unusable_key_count": len(threshold_keys) - usable_key_count,
            },
        )
    )

    adapter_gate = g02["adapter_gate"]
    adapter_metric_values = {
        name: value.get("value") if isinstance(value, Mapping) else value
        for name, value in adapter_gate["metrics"].items()
    }
    evidence.append(
        _evidence_row(
            stage="G02_adapter",
            probe="G02_DUAL_ENGINE_ADAPTER_CHAIN",
            signal="reset_history_actor_target_clock_and_polarity",
            status=str(adapter_gate["classification"]),
            artifact=g02_worker / "result.json",
            signal_path="$.adapter_gate.classification",
            observed_value=adapter_gate["classification"],
            threshold=adapter_gate["thresholds"],
            supports=(
                "SIM2SIM_ADAPTER_BUG"
                if not adapter_gate["digital_chain_passed"]
                else ""
            ),
            excludes=(
                "SIM2SIM_ADAPTER_BUG"
                if adapter_gate["digital_chain_passed"]
                else ""
            ),
            guard_status={
                "gates": adapter_gate["gates"],
                "failures": adapter_gate["failures"],
                "deferred_failures": adapter_gate["deferred_failures"],
                "gate_groups": adapter_gate["gate_groups"],
                "metrics": adapter_metric_values,
                "isaac_evidence_hash": adapter_gate["isaac_evidence_hash"],
                "mujoco_evidence_hash": adapter_gate["mujoco_evidence_hash"],
            },
        )
    )

    instrumentation_observed = {
        mode: {
            "passed": comparison["passed"],
            "maximum_absolute_error": max(
                (
                    float(field["maximum_absolute_error"])
                    for field in comparison["continuous"].values()
                ),
                default=0.0,
            ),
        }
        for mode, comparison in g03["isaac_comparisons"].items()
    }
    instrumentation_observed["contact_observer_usable"] = g03[
        "contact_observer_usable"
    ]
    evidence.append(
        _evidence_row(
            stage="G03_instrumentation",
            probe="G03_OBSERVER_NEUTRALITY",
            signal="formal_debug_contact_and_system_observer_equivalence",
            status="neutral" if g03["passed"] else "perturbing",
            artifact=g03_worker / "result.json",
            signal_path="$.passed",
            observed_value=bool(g03["passed"]),
            threshold={
                "continuous_fields": "bitwise_equal",
                "discrete_fields": "exact_equal",
                "step_counters": "exact",
            },
            supports="instrumentation_perturbation" if not g03["passed"] else "",
            excludes="instrumentation_perturbation" if g03["passed"] else "",
            guard_status=instrumentation_observed,
        )
    )

    for scenario, comparison in p10["analyses"].items():
        for field, signal_record in comparison["signals"].items():
            evidence.append(
                _evidence_row(
                    stage="P10_rest",
                    probe=scenario,
                    signal=field,
                    status="measured",
                    artifact=p10_worker / "analysis" / f"{scenario}.json",
                    signal_path=f"$.signals.{field}.rmse",
                    observed_value=float(signal_record["rmse"]),
                    threshold={
                        "isaac_repeat_envelope": float(
                            signal_record["isaac_repeat_envelope"]
                        ),
                        "mujoco_repeat_envelope": float(
                            signal_record["mujoco_repeat_envelope"]
                        ),
                        "interpretation": "diagnostic_only_no_standalone_candidate_vote",
                    },
                    time_window_ms=comparison["common_times_ms"],
                    scalar_metric="rmse",
                    channel_reduction="l2_or_named_channel",
                )
            )

    p20_artifact = p20_worker / "result.json"
    evidence.append(
        _evidence_row(
            stage="P20_static_properties",
            probe="P20_S_STATIC_PROPERTIES",
            signal="compiled_body_properties_and_goldens",
            status="mismatch" if p20["mismatch_detected"] else "within_tolerance",
            artifact=p20_artifact,
            signal_path="$.mismatch_detected",
            observed_value=bool(p20["mismatch_detected"]),
            threshold={
                "mismatch_detected": False,
                "audit_valid": True,
                "golden_valid": True,
            },
            supports="MASS_OR_INERTIA_MISMATCH" if p20["mismatch_detected"] else "",
            excludes="MASS_OR_INERTIA_MISMATCH" if not p20["mismatch_detected"] else "",
            guard_status={
                "audit_valid": p20["audit_valid"],
                "golden_valid": p20["golden_valid"],
            },
        )
    )
    for name, item in p30["analyses"].items():
        label = "open" if "open" in name else "closed"
        pair = p30["causal_pairs"][f"isaac_{label}"]
        evidence.append(
            _evidence_row(
                stage="P30_actuator",
                probe=name,
                signal="canonical_joint_velocity_odd_response",
                status="material" if item["material"] else "within_tolerance",
                artifact=p30_worker / "analysis" / f"{name}.json",
                signal_path="$.normalized_rmse",
                observed_value=float(item["normalized_rmse"]),
                threshold=float(item["normalized_effective_tolerance"]),
                time_window_ms=item.get("common_times_ms", ""),
                supports="ACTUATOR_INTEGRATION_MISMATCH" if item["material"] else "",
                excludes="ACTUATOR_INTEGRATION_MISMATCH" if not item["material"] else "",
                effective_tolerance=float(item["normalized_effective_tolerance"]),
                scalar_metric="normalized_rmse",
                channel_reduction="l2",
                **_pair_columns(pair),
            )
        )
    p40_pair = p40["causal_pairs"]["isaac"]
    p40_on = p40["analysis"]["closure_on"]
    evidence.append(
        _evidence_row(
            stage="P40_closure",
            probe="P40_CLOSURE_ON_TO_OFF",
            signal="all_hinge_velocity_odd_response",
            status="material" if p40_on["material"] else "within_tolerance",
            artifact=p40_worker / "analysis" / "p40.json",
            signal_path="$.closure_on.normalized_rmse",
            observed_value=float(p40_on["normalized_rmse"]),
            threshold=float(p40_on["normalized_effective_tolerance"]),
            time_window_ms=p40["analysis"].get("common_times_ms", ""),
            effective_tolerance=float(p40_on["normalized_effective_tolerance"]),
            explanation=p40["analysis"].get("explanation_ratio"),
            scalar_metric="normalized_rmse",
            channel_reduction="l2",
            guard_status={
                "closure_on_material": p40_on["material"],
                "aggregate_status": closure_status,
            },
            **_pair_columns(p40_pair),
        )
    )
    evidence.extend(
        _closure_evidence_rows(
            closure_status=closure_status,
            closure_diagnostics=closure_diagnostics,
            p10_worker=p10_worker,
            p10=p10,
            p20_worker=p20_worker,
            p20=p20,
            p30_worker=p30_worker,
            p30=p30,
            p40_worker=p40_worker,
            p40=p40,
        )
    )
    evidence.extend(
        [
            _evidence_row(
                stage="P50_contact",
                probe="P50_A_SPHERE_IMPACT",
                signal="contact_aligned_com_velocity_z",
                status=normal_status,
                artifact=p50_worker / "analysis" / "impact.json",
                signal_path="$.material_condition_count",
                observed_value=int(p50["impact"]["material_condition_count"]),
                threshold={
                    "condition_effective_tolerances": [
                        condition["postcontact"].get("effective_tolerance")
                        for condition in p50["impact"]["conditions"]
                    ],
                    "persistent_samples": 3,
                },
                time_window_ms=P50_IMPACT_OFFSETS_MS.astype(int).tolist(),
                supports=(
                    "CONTACT_OR_FRICTION_MISMATCH:normal"
                    if normal_status
                    in {
                        CandidateStatus.SUPPORTED_PRIMARY.value,
                        CandidateStatus.SUPPORTED_CONTRIBUTOR.value,
                    }
                    else ""
                ),
                excludes=(
                    "CONTACT_OR_FRICTION_MISMATCH:normal"
                    if normal_status == CandidateStatus.NOT_SUPPORTED.value
                    else ""
                ),
                guard_status={
                    "gate_status": p50["impact"].get("gate_status"),
                    "valid_condition_count": int(
                        p50["impact"]["valid_condition_count"]
                    ),
                },
                scalar_metric="rmse",
                channel_reduction="named_channel",
            ),
            _evidence_row(
                stage="P50_contact",
                probe="P50_C_NOMINAL_TO_ZERO_FRICTION",
                signal="com_velocity_x_odd_response",
                status=tangential_status,
                artifact=p50_worker / "analysis" / "slide.json",
                signal_path="$.nominal.normalized_rmse",
                observed_value=float(slide["nominal"]["normalized_rmse"]),
                threshold=float(slide["nominal"]["normalized_effective_tolerance"]),
                time_window_ms=slide.get("common_times_ms", ""),
                supports=(
                    "CONTACT_OR_FRICTION_MISMATCH:tangential"
                    if tangential_status
                    in {
                        CandidateStatus.SUPPORTED_PRIMARY.value,
                        CandidateStatus.SUPPORTED_CONTRIBUTOR.value,
                    }
                    else ""
                ),
                excludes=(
                    "CONTACT_OR_FRICTION_MISMATCH:tangential"
                    if tangential_status == CandidateStatus.NOT_SUPPORTED.value
                    else ""
                ),
                effective_tolerance=float(
                    slide["nominal"]["normalized_effective_tolerance"]
                ),
                explanation=slide.get("explanation_ratio"),
                guard_status=slide.get("gate_status"),
                scalar_metric="normalized_rmse",
                channel_reduction="named_channel",
                **_pair_columns(p50["causal_pairs"]["isaac"]),
            ),
        ]
    )
    replay_columns = _replay_columns(p60)
    for name, item in p60["analyses"].items():
        evidence.append(
            _evidence_row(
                stage="P60_full_robot",
                probe=name,
                signal=item["field"],
                status="material" if item["material"] else "within_tolerance",
                artifact=p60_worker / "analysis" / f"{name}.json",
                signal_path="$.normalized_rmse",
                observed_value=float(item["normalized_rmse"]),
                threshold=float(item["normalized_effective_tolerance"]),
                time_window_ms=item.get("common_times_ms", ""),
                supports="plant_material_failure" if item["material"] else "",
                effective_tolerance=float(item["normalized_effective_tolerance"]),
                guard_status={
                    "fresh_replay": p60["fresh_replay_equivalence"]["passed"],
                    "reset_cache": p60["reset_cache_gate"]["passed"],
                },
                **(replay_columns if name == "p60_d" else {}),
            )
        )
    c70_artifact = c70_worker / "analysis" / "c70.json"
    anchors = c70["analysis"].get("anchors", {})
    if not anchors:
        evidence.append(
            _evidence_row(
                stage="C70_checkpoint",
                probe="C70_CHECKPOINT_SENSITIVITY",
                signal="raw_action_rms_gain",
                status="not_evaluated",
                artifact=c70_artifact,
                signal_path="$.reason",
                observed_value=c70["analysis"].get("reason"),
                threshold="material_plant_failure_required",
                supports=f"checkpoint_role={checkpoint_role}",
                guard_status=c70["analysis"].get("reason"),
                **replay_columns,
            )
        )
    else:
        for anchor, anchor_row in anchors.items():
            for reduction_key in ("full_history", "latest_frame_only"):
                reduction = anchor_row[reduction_key]
                for group, group_row in reduction["groups"].items():
                    for sign, sign_row in group_row["signs"].items():
                        evidence.append(
                            _evidence_row(
                                stage="C70_checkpoint",
                                probe="C70_CHECKPOINT_SENSITIVITY",
                                signal=f"{group}:{sign}:raw_action_rms_gain",
                                status=(
                                    reduction["classification"]
                                    if sign_row["valid_tick_count"] >= 3
                                    else "ineligible"
                                ),
                                artifact=c70_artifact,
                                signal_path=(
                                    f"$.anchors.{anchor}.{reduction_key}.groups."
                                    f"{group}.signs.{sign}"
                                ),
                                observed_value=sign_row,
                                threshold={
                                    "minimum_valid_tick_count": 3,
                                    "amplifier_ratio": 1.50,
                                    "amplifier_difference": 0.25,
                                    "tie_ratio": [0.80, 1.25],
                                    "tie_difference": 0.10,
                                },
                                supports=f"checkpoint_role={checkpoint_role}",
                                baseline_value=sign_row.get("phase1r_gain"),
                                ablation_value=sign_row.get("dreamwaq_gain"),
                                channel_reduction=reduction_key,
                                baseline_engine=anchor,
                                baseline_trace_sha256=anchor_row["trace_sha256"],
                                baseline_observation_sha256=anchor_row[
                                    "observation_sequence_sha256"
                                ],
                                guard_status={
                                    "eligible_group": group_row["eligible"],
                                    "valid_tick_count": sign_row["valid_tick_count"],
                                },
                                **replay_columns,
                            )
                        )

    evidence_refs = _build_evidence_refs(root, evidence, selected_identity)
    public_evidence = [_public_evidence_row(row) for row in evidence]
    threshold_snapshot_metadata = {
        "schema_version": threshold_snapshot["schema_version"],
        "identity_hash": threshold_snapshot["identity_hash"],
        "artifact": threshold_snapshot_path.relative_to(root).as_posix(),
        "artifact_sha256": sha256_file(threshold_snapshot_path),
        "frozen_before_cross_engine_physics": threshold_snapshot[
            "frozen_before_cross_engine_physics"
        ],
        "minimum_repetitions": threshold_snapshot["minimum_repetitions"],
        "effective_tolerance_rule": threshold_snapshot[
            "effective_tolerance_rule"
        ],
        "high_noise_rule": threshold_snapshot["high_noise_rule"],
        "executed_key_count": g01["executed_key_count"],
        "executed_sample_count": g01["executed_sample_count"],
        "usable_key_count": usable_key_count,
        "unusable_key_count": len(threshold_keys) - usable_key_count,
        "repeatability_usable": g01["repeatability_usable"],
    }
    analysis = {
        "schema_version": "RootCauseCoreAnalysisV2",
        "run_id": root.name,
        "manifest_identity_hash": manifest_identity_hash,
        "selected_attempts_identity": selected_identity,
        "failed_gates": failed_gates,
        "candidates": candidates,
        "closure_diagnostics": closure_diagnostics,
        "confidence": confidence,
        "first_material_divergence": first_divergence,
        "executed_core_probes": [item.scenario_id for item in catalog()],
        "unexecuted_extended_probes": EXTENDED_PROBES,
        "checkpoint_role": checkpoint_role,
        "evidence_refs": evidence_refs,
        "remaining_uncertainty": [
            "dynamic_effective_inertia_not_directly_identified",
            "full_robot_geometry_contact_coupling_requires_P60_A_or_P60_E",
            "simulator_closeness_to_hardware_requires_system_identification_data",
        ],
        "formal_file_hashes_unchanged": formal_unchanged,
        "formal_file_hashes_before": {
            name: record["actual_sha256"] for name, record in g00["formal_hashes"].items()
        },
        "formal_file_hashes_after": formal_after,
        "key_findings": key_findings,
        "threshold_snapshot": threshold_snapshot_metadata,
        "decision_thresholds": {
            "explanation_primary": 0.70,
            "explanation_contributor": 0.30,
            "c70_amplifier_ratio": 1.50,
            "c70_amplifier_difference": 0.25,
            "c70_tie_ratio": [0.80, 1.25],
            "c70_tie_difference": 0.10,
        },
        "actuator_pairs": actuator_pairs,
        "stage_result_sha256": {
            stage: sha256_file(worker / "result.json")
            for stage, (worker, _) in stages.items()
        },
    }
    analysis_root = root / "analysis"
    analysis_path = analysis_root / "analysis.json"
    evidence_path = analysis_root / "evidence.json"
    _atomic_json(analysis_path, analysis)
    _atomic_json(evidence_path, public_evidence)
    verdict = build_verdict(analysis)
    bundle = write_report_bundle(root / "report", verdict, public_evidence)
    artifacts = {
        "analysis/analysis.json": sha256_file(analysis_path),
        "analysis/evidence.json": sha256_file(evidence_path),
        "report/verdict.json": sha256_file(bundle.verdict),
        "report/evidence.csv": sha256_file(bundle.evidence),
        "report/report.md": sha256_file(bundle.report),
        "report/file_hashes.json": sha256_file(bundle.hashes),
    }
    final_state = {
        "schema_version": "RootCauseFinalStateV2",
        "run_id": root.name,
        "manifest_identity_hash": manifest_identity_hash,
        "selected_attempts_identity": selected_identity,
        "verdict": verdict["primary_classification"],
        "checkpoint_role": checkpoint_role,
        "formal_file_hashes_unchanged": formal_unchanged,
        "artifact_hashes": artifacts,
    }
    final_state["final_state_identity_hash"] = stable_hash(final_state)
    _atomic_json(root / "final_state.json", final_state)
    return final_state


def verify_final_state(
    run_root: Path, *, verified_attempts: Mapping[str, Path]
) -> dict[str, Any]:
    root = Path(run_root).resolve(strict=True)
    from .orchestrator import _read_manifest_payload

    manifest = _read_manifest_payload(root)
    state_path = root / "final_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    unsigned = dict(state)
    expected_identity = unsigned.pop("final_state_identity_hash", None)
    if expected_identity != stable_hash(unsigned):
        raise EvidenceIntegrityError("Final state identity hash mismatch")
    if state.get("run_id") != root.name:
        raise EvidenceIntegrityError("Final state run id mismatch")
    if state.get("manifest_identity_hash") != manifest.get("manifest_identity_hash"):
        raise EvidenceIntegrityError("Final state manifest identity mismatch")
    selected_identity = _selected_identity(verified_attempts)
    if state.get("selected_attempts_identity") != selected_identity:
        raise EvidenceIntegrityError("Final state selected attempt identity mismatch")
    artifacts = state.get("artifact_hashes")
    if not isinstance(artifacts, dict):
        raise EvidenceIntegrityError("Final state artifact hashes are missing")
    for relative, expected in artifacts.items():
        actual = sha256_file(root / relative)
        if actual != expected:
            raise EvidenceIntegrityError(f"Final artifact hash mismatch: {relative}")
    report_hashes = json.loads(
        (root / "report" / "file_hashes.json").read_text(encoding="utf-8")
    )
    for name in ("verdict.json", "evidence.csv", "report.md"):
        if report_hashes.get(name) != sha256_file(root / "report" / name):
            raise EvidenceIntegrityError(f"Report bundle hash mismatch: {name}")
    analysis = json.loads((root / "analysis" / "analysis.json").read_text(encoding="utf-8"))
    if analysis.get("manifest_identity_hash") != manifest.get("manifest_identity_hash"):
        raise EvidenceIntegrityError("Analysis manifest identity mismatch")
    if analysis.get("selected_attempts_identity") != selected_identity:
        raise EvidenceIntegrityError("Analysis selected attempt identity mismatch")
    return state
