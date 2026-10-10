from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import io
from typing import Any, Iterable, Mapping

from .contracts import canonical_json_bytes, sha256_file


EVIDENCE_COLUMNS = (
    "stage", "probe", "engine_pair", "signal", "time_window_ms", "scalar_metric",
    "channel_reduction", "repeatability_family", "baseline_value", "ablation_value",
    "scenario_id", "configuration_hash", "configuration_semantics_hash",
    "pre_forward_initial_condition_hash", "post_forward_state_hash",
    "reset_returned_policy_hash", "excitation_hash", "comparison_profile_hash",
    "baseline_scenario_id", "baseline_configuration_hash", "baseline_configuration_semantics_hash",
    "baseline_pre_forward_initial_condition_hash", "baseline_post_forward_state_hash",
    "baseline_reset_returned_policy_hash", "baseline_excitation_hash",
    "ablation_scenario_id", "ablation_configuration_hash", "ablation_configuration_semantics_hash",
    "ablation_pre_forward_initial_condition_hash", "ablation_post_forward_state_hash",
    "ablation_reset_returned_policy_hash", "ablation_excitation_hash",
    "semantic_diff_paths_sha256", "allowed_ablation_factors",
    "baseline_engine", "baseline_trace_sha256", "baseline_observation_sha256",
    "replay_source_identity_hash", "replay_generator_actor_sha256",
    "replay_generator_manifest_sha256", "replay_source_scenario_id", "replay_source_seed",
    "replay_source_repetition_index", "replay_environment_count", "replay_environment_index",
    "replay_environment_scenario_name", "replay_evaluation_contract_hash",
    "replay_reset_cache_file_sha256", "replay_reset_cache_tensor_sha256",
    "replay_reset_cache_identity_hash", "replay_reset_cache_schema",
    "replay_reset_cache_root_height_algorithm", "replay_reset_cache_relaxation_algorithm",
    "replay_action_sequence_sha256",
    "expected_improvement_direction", "guard_status",
    "effective_tolerance", "explanation_ratio", "availability", "status", "supports",
    "excludes", "artifact_sha256",
)


@dataclass(frozen=True)
class ReportBundle:
    verdict: Path
    evidence: Path
    report: Path
    hashes: Path


def _cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return value


def _inline_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _claim_evidence_lines(
    claims: Iterable[str],
    references: Iterable[Mapping[str, Any]],
    *,
    relation: str,
) -> list[str]:
    refs = list(references)
    lines: list[str] = []
    for claim in claims:
        matches = [row for row in refs if row.get(relation) == claim]
        if not matches:
            lines.append(f"- `{claim}`: no direct evidence reference")
            continue
        for row in matches:
            lines.append(
                "- "
                f"`{claim}` via `{row.get('artifact')}` "
                f"`{row.get('signal_path')}`: observed "
                f"`{_inline_json(row.get('observed_value'))}`, threshold "
                f"`{_inline_json(row.get('threshold'))}`, status "
                f"`{row.get('status')}`."
            )
    if not lines:
        lines.append("- None.")
    return lines


def _next_allowed_contract(primary: str, status: str) -> str:
    if status in {"invalid", "inconclusive", "multiple"}:
        return "No formal contract change is authorized; run the listed extended probes first."
    mapping = {
        "SIM2SIM_ADAPTER_BUG": "Adapter/reset/history/action serialization contract.",
        "MASS_OR_INERTIA_MISMATCH": "Mass, center-of-mass, and inertia contract.",
        "ACTUATOR_INTEGRATION_MISMATCH": "Actuator drive and integration contract.",
        "CLOSED_CHAIN_CONSTRAINT_MISMATCH": "Closed-chain constraint contract.",
        "CONTACT_OR_FRICTION_MISMATCH:normal": "Normal-contact contract.",
        "CONTACT_OR_FRICTION_MISMATCH:tangential": "Tangential friction contract.",
    }
    return mapping.get(primary, "No formal contract change is authorized from this run.")


def _render_report(verdict: Mapping[str, Any]) -> str:
    status = verdict.get("conclusion_status", "inconclusive")
    primary = verdict.get("primary_classification", "INCONCLUSIVE")
    confidence = verdict.get("confidence", "low")
    lines = [
        "# WheelLeg Sim2Sim Root-Cause Report",
        "",
        f"- Run: `{verdict.get('run_id', 'unknown')}`",
        f"- Conclusion: `{primary}`",
        f"- Status: `{status}`",
        f"- Confidence: `{confidence}`",
        f"- Evidence pipeline valid: `{bool(verdict.get('run_valid', False))}`",
        f"- Formal files unchanged: `{bool(verdict.get('formal_file_hashes_unchanged', False))}`",
        "",
    ]
    if status == "inconclusive":
        lines.extend(["## 结论", "", "本轮没有找到可认证根因。", ""])
    elif status == "invalid":
        lines.extend(["## 结论", "", "本轮证据链无效，不能作物理根因判断。", ""])
    else:
        lines.extend(["## 结论", "", f"认证分类：`{primary}`。", ""])
    references = list(verdict.get("evidence_refs", []))
    primary_claims = list(verdict.get("supported_primary", []))
    contributor_claims = list(verdict.get("supported_contributors", []))
    excluded_claims = list(verdict.get("excluded_candidates", []))
    inconclusive_claims = list(verdict.get("inconclusive_candidates", []))
    lines.extend(
        [
            "## Run Validity",
            "",
            f"- Failed gates: `{_inline_json(verdict.get('failed_gates', []))}`",
            f"- Formal hashes before: `{_inline_json(verdict.get('formal_file_hashes_before', {}))}`",
            f"- Formal hashes after: `{_inline_json(verdict.get('formal_file_hashes_after', {}))}`",
            "",
            "## First Material Divergence",
            "",
            f"`{_inline_json(verdict.get('first_material_divergence', {}))}`",
            "",
            "## Positive Evidence",
            "",
            *_claim_evidence_lines(primary_claims, references, relation="supports"),
            "",
            "## Contributor Evidence",
            "",
            *_claim_evidence_lines(contributor_claims, references, relation="supports"),
            "",
            "## Excluded Candidates",
            "",
            *_claim_evidence_lines(excluded_claims, references, relation="excludes"),
            "",
            "## Inconclusive Candidates",
            "",
            *_claim_evidence_lines(inconclusive_claims, references, relation="supports"),
            "",
            "## Checkpoint Role",
            "",
            f"- Role: `{verdict.get('checkpoint_role', 'not_evaluated')}`",
            "",
            "## Key Findings",
            "",
            f"`{_inline_json(verdict.get('key_findings', []))}`",
            "",
            "## Frozen Repeatability Snapshot",
            "",
            f"`{_inline_json(verdict.get('threshold_snapshot', {}))}`",
            "",
            "## Decision Thresholds",
            "",
            f"`{_inline_json(verdict.get('decision_thresholds', {}))}`",
            "",
            "## Remaining Uncertainty",
            "",
            f"`{_inline_json(verdict.get('remaining_uncertainty', []))}`",
            "",
            "## Next Allowed Contract",
            "",
            _next_allowed_contract(str(primary), str(status)),
            "",
            f"The evidence index contains {len(references)} hash-verified references; evidence.csv records each probe finding and exclusion.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report_bundle(
    directory: Path,
    verdict: Mapping[str, Any],
    evidence: Iterable[Mapping[str, Any]],
) -> ReportBundle:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    verdict_path = destination / "verdict.json"
    evidence_path = destination / "evidence.csv"
    report_path = destination / "report.md"
    hashes_path = destination / "file_hashes.json"

    def atomic_bytes(path: Path, payload: bytes) -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)

    atomic_bytes(verdict_path, canonical_json_bytes(dict(verdict)) + b"\n")
    rows = [dict(row) for row in evidence]
    unknown = sorted({key for row in rows for key in row if key not in EVIDENCE_COLUMNS})
    if unknown:
        raise ValueError(f"Unknown evidence columns: {unknown}")
    rows.sort(key=lambda row: tuple(str(_cell(row.get(column))) for column in EVIDENCE_COLUMNS))
    stream = io.StringIO(newline="")
    try:
        writer = csv.DictWriter(stream, fieldnames=EVIDENCE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _cell(row.get(column)) for column in EVIDENCE_COLUMNS})
        evidence_payload = stream.getvalue().encode("utf-8")
    finally:
        stream.close()
    atomic_bytes(evidence_path, evidence_payload)
    atomic_bytes(report_path, _render_report(verdict).encode("utf-8"))

    hashes = {
        path.name: sha256_file(path)
        for path in (verdict_path, evidence_path, report_path)
    }
    atomic_bytes(hashes_path, canonical_json_bytes(hashes) + b"\n")
    return ReportBundle(verdict_path, evidence_path, report_path, hashes_path)
