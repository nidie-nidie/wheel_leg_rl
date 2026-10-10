from __future__ import annotations

from typing import Any, Mapping

from .contracts import CandidateStatus


CHECKPOINT_CLASS = "CHECKPOINT_ROBUSTNESS_FAILURE"


def _gate_name(value: object) -> str:
    return str(value).split("_", 1)[0].upper()


def build_verdict(analysis: Mapping[str, Any]) -> dict[str, Any]:
    failed_gates = [str(item) for item in analysis.get("failed_gates", [])]
    gate_names = {_gate_name(item) for item in failed_gates}
    candidates = {str(key): str(value) for key, value in dict(analysis.get("candidates", {})).items()}
    candidates.pop(CHECKPOINT_CLASS, None)

    base: dict[str, Any] = {
        "schema_version": "RootCauseVerdictV1",
        "scope": "core_v1",
        "run_id": str(analysis.get("run_id", "unknown")),
        "run_valid": not bool(failed_gates),
        "conclusion_status": "inconclusive",
        "primary_classification": "INCONCLUSIVE",
        "confidence": str(analysis.get("confidence", "low")),
        "supported_primary": [],
        "supported_contributors": [],
        "excluded_candidates": [],
        "inconclusive_candidates": [],
        "first_material_divergence": dict(analysis.get("first_material_divergence", {})),
        "failed_gates": failed_gates,
        "executed_core_probes": list(analysis.get("executed_core_probes", [])),
        "unexecuted_extended_probes": list(analysis.get("unexecuted_extended_probes", [])),
        "checkpoint_role": str(analysis.get("checkpoint_role", "not_evaluated")),
        "evidence_refs": list(analysis.get("evidence_refs", [])),
        "remaining_uncertainty": list(analysis.get("remaining_uncertainty", [])),
        "formal_file_hashes_unchanged": bool(
            analysis.get("formal_file_hashes_unchanged", False)
        ),
        "formal_file_hashes_before": dict(
            analysis.get("formal_file_hashes_before", {})
        ),
        "formal_file_hashes_after": dict(
            analysis.get("formal_file_hashes_after", {})
        ),
        "key_findings": list(analysis.get("key_findings", [])),
        "threshold_snapshot": dict(analysis.get("threshold_snapshot", {})),
        "decision_thresholds": dict(analysis.get("decision_thresholds", {})),
    }

    if "G00" in gate_names:
        base.update(
            run_valid=False,
            conclusion_status="invalid",
            primary_classification="INVALID_EVIDENCE_PIPELINE",
            confidence="low",
        )
        return base
    if "G02" in gate_names:
        base.update(
            run_valid=False,
            conclusion_status="identified",
            primary_classification="SIM2SIM_ADAPTER_BUG",
            confidence="high",
            supported_primary=["SIM2SIM_ADAPTER_BUG"],
        )
        return base
    if "G03" in gate_names or any(name.startswith("P20") and "golden" in name.lower() for name in failed_gates):
        base.update(
            run_valid=False,
            conclusion_status="invalid",
            primary_classification="INVALID_EVIDENCE_PIPELINE",
            confidence="low",
        )
        return base
    if failed_gates:
        base.update(run_valid=False, confidence="low")
        base["remaining_uncertainty"] = sorted(
            set(base["remaining_uncertainty"]) | {"repeatability_or_gate_failure"}
        )
        return base

    primary = sorted(
        candidate
        for candidate, status in candidates.items()
        if status == CandidateStatus.SUPPORTED_PRIMARY.value
    )
    contributors = sorted(
        candidate
        for candidate, status in candidates.items()
        if status == CandidateStatus.SUPPORTED_CONTRIBUTOR.value
    )
    excluded = sorted(
        candidate
        for candidate, status in candidates.items()
        if status == CandidateStatus.NOT_SUPPORTED.value
    )
    inconclusive = sorted(
        candidate
        for candidate, status in candidates.items()
        if status == CandidateStatus.INCONCLUSIVE.value
    )
    unknown_statuses = sorted(
        candidate
        for candidate, status in candidates.items()
        if status not in {member.value for member in CandidateStatus}
    )
    if unknown_statuses:
        raise ValueError(f"Unknown candidate status for: {unknown_statuses}")

    base["supported_primary"] = primary
    base["supported_contributors"] = contributors
    base["excluded_candidates"] = excluded
    base["inconclusive_candidates"] = inconclusive
    if len(primary) == 1:
        base.update(conclusion_status="identified", primary_classification=primary[0])
    elif len(primary) > 1:
        base.update(
            conclusion_status="multiple",
            primary_classification="MULTIPLE_PHYSICS_MISMATCHES",
        )
    else:
        base.update(conclusion_status="inconclusive", primary_classification="INCONCLUSIVE")
        if not contributors and not inconclusive and not base["unexecuted_extended_probes"]:
            base["primary_classification"] = "CORE_NO_MATERIAL_DIVERGENCE"
        if base["unexecuted_extended_probes"]:
            base["remaining_uncertainty"] = sorted(
                set(base["remaining_uncertainty"]) | {"extended_scope_required"}
            )
    return base
