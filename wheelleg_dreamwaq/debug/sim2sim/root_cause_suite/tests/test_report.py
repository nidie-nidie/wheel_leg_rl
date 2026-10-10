from __future__ import annotations

import csv
import json
from pathlib import Path

from debug.sim2sim.root_cause_suite.report import write_report_bundle


def test_report_bundle_is_deterministic_and_auditable(tmp_path: Path) -> None:
    verdict = {
        "schema_version": "RootCauseVerdictV1",
        "scope": "core_v1",
        "run_id": "unit",
        "run_valid": True,
        "conclusion_status": "inconclusive",
        "primary_classification": "INCONCLUSIVE",
        "confidence": "low",
        "supported_primary": [],
        "supported_contributors": [],
        "excluded_candidates": [],
        "inconclusive_candidates": ["complex_geometry"],
        "first_material_divergence": {},
        "failed_gates": [],
        "executed_core_probes": ["G00"],
        "unexecuted_extended_probes": ["P50-B"],
        "checkpoint_role": "not_evaluated",
        "evidence_refs": [],
        "remaining_uncertainty": ["complex_geometry"],
    }
    evidence = [{"stage": "G00", "probe": "identity", "status": "pass", "artifact_sha256": "A" * 64}]
    result = write_report_bundle(tmp_path, verdict, evidence)
    assert json.loads(result.verdict.read_text(encoding="utf-8"))["run_id"] == "unit"
    with result.evidence.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["artifact_sha256"] == "A" * 64
    assert "本轮没有找到可认证根因" in result.report.read_text(encoding="utf-8")


def test_report_renders_claim_evidence_thresholds_and_next_contract(
    tmp_path: Path,
) -> None:
    verdict = {
        "schema_version": "RootCauseVerdictV1",
        "scope": "core_v1",
        "run_id": "unit-identified",
        "run_valid": True,
        "conclusion_status": "identified",
        "primary_classification": "ACTUATOR_INTEGRATION_MISMATCH",
        "confidence": "high",
        "supported_primary": ["ACTUATOR_INTEGRATION_MISMATCH"],
        "supported_contributors": [],
        "excluded_candidates": ["MASS_OR_INERTIA_MISMATCH"],
        "inconclusive_candidates": [],
        "first_material_divergence": {"stage": "P30_actuator"},
        "failed_gates": [],
        "executed_core_probes": ["P30"],
        "unexecuted_extended_probes": [],
        "checkpoint_role": "not_distinguished",
        "formal_file_hashes_unchanged": True,
        "formal_file_hashes_before": {"formal": "A" * 64},
        "formal_file_hashes_after": {"formal": "A" * 64},
        "evidence_refs": [
            {
                "artifact": "stages/P30/attempt-0001/worker/analysis/p30.json",
                "signal_path": "$.normalized_rmse",
                "observed_value": 0.4,
                "threshold": 0.1,
                "status": "material",
                "supports": "ACTUATOR_INTEGRATION_MISMATCH",
                "excludes": "",
            },
            {
                "artifact": "stages/P20/attempt-0001/worker/result.json",
                "signal_path": "$.mismatch_detected",
                "observed_value": False,
                "threshold": False,
                "status": "within_tolerance",
                "supports": "",
                "excludes": "MASS_OR_INERTIA_MISMATCH",
            },
        ],
        "threshold_snapshot": {"identity_hash": "B" * 64},
        "decision_thresholds": {"explanation_primary": 0.7},
        "remaining_uncertainty": [],
        "key_findings": [],
    }
    result = write_report_bundle(tmp_path, verdict, [])
    rendered = result.report.read_text(encoding="utf-8")
    assert "## Positive Evidence" in rendered
    assert "$.normalized_rmse" in rendered
    assert "## Excluded Candidates" in rendered
    assert "## Frozen Repeatability Snapshot" in rendered
    assert "Actuator drive and integration contract" in rendered
