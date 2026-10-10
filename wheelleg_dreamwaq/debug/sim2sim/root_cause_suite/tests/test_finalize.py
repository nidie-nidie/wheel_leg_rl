from __future__ import annotations

import json
from pathlib import Path

import pytest

from debug.sim2sim.root_cause_suite import finalize as finalize_module
from debug.sim2sim.root_cause_suite.contracts import (
    CandidateStatus,
    EvidenceIntegrityError,
    canonical_json_bytes,
    sha256_file,
    stable_hash,
)
from debug.sim2sim.root_cause_suite.finalize import (
    CORE_STAGES,
    _build_evidence_refs,
    _evidence_row,
    _public_evidence_row,
    finalize_adapter_terminal,
    finalize_repeatability_terminal,
    finalize_run,
)
from debug.sim2sim.root_cause_suite.repeatability import SNAPSHOT_SCHEMA_VERSION


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _stage(
    root: Path,
    stage: str,
    payload: dict,
    *,
    artifacts: dict[str, object] | None = None,
) -> Path:
    attempt = root / "stages" / stage / "attempt-0001"
    worker = attempt / "worker"
    worker.mkdir(parents=True)
    result = {
        "schema_version": "RootCauseCoreStageResultV1",
        "stage": stage,
        "passed": True,
        **payload,
    }
    _write_json(worker / "result.json", result)
    for relative, artifact in (artifacts or {}).items():
        _write_json(worker / relative, artifact)
    _write_json(attempt / "stage_state.json", {"stage": stage, "complete": True})
    return attempt


def _unusable_threshold_snapshot() -> dict:
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "frozen_before_cross_engine_physics": True,
        "minimum_repetitions": 3,
        "effective_tolerance_rule": "max(material_floor,5*repeat_envelope)",
        "high_noise_rule": (
            "mark_key_unusable_when_repeat_envelope_exceeds_material_floor"
        ),
        "material_floors": {},
        "executions": [],
        "keys": {
            "fixture-key": {
                "sample_count": 3,
                "usable": False,
                "unusable_reasons": ["nondeterministic:p30_open_direct"],
            }
        },
        "index": {},
    }
    snapshot["identity_hash"] = stable_hash(snapshot)
    return snapshot


def test_evidence_reference_binds_path_signal_value_threshold_and_hash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    artifact = (
        root
        / "stages"
        / "G00_integrity"
        / "attempt-0001"
        / "worker"
        / "result.json"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"passed":true}\n', encoding="utf-8")
    row = _evidence_row(
        stage="G00_integrity",
        probe="G00_IDENTITY_AND_SCOPE",
        signal="formal_identity_scope_and_write_boundary",
        status="pass",
        artifact=artifact,
        signal_path="$.passed",
        observed_value=True,
        threshold=True,
    )
    selected = {
        "stages": {
            "G00_integrity": {
                "attempt": "attempt-0001",
            }
        }
    }
    references = _build_evidence_refs(root, [row], selected)
    assert references[0]["artifact"] == (
        "stages/G00_integrity/attempt-0001/worker/result.json"
    )
    assert references[0]["signal_path"] == "$.passed"
    assert references[0]["observed_value"] is True
    assert references[0]["threshold"] is True
    assert len(references[0]["reference_identity_hash"]) == 64
    assert not any(key.startswith("_") for key in _public_evidence_row(row))

    artifact.write_text('{"passed":false}\n', encoding="utf-8")
    with pytest.raises(EvidenceIntegrityError, match="hash drifted"):
        _build_evidence_refs(root, [row], selected)


def test_evidence_reference_rejects_artifact_outside_run(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    artifact = tmp_path / "outside.json"
    artifact.write_text("{}\n", encoding="utf-8")
    row = _evidence_row(
        stage="G00_integrity",
        probe="G00",
        signal="identity",
        status="pass",
        artifact=artifact,
        signal_path="$.passed",
        observed_value=True,
        threshold=True,
    )
    selected = {"stages": {"G00_integrity": {"attempt": "attempt-0001"}}}
    with pytest.raises(EvidenceIntegrityError, match="unavailable"):
        _build_evidence_refs(root, [row], selected)


def test_evidence_reference_rejects_unselected_attempt_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    selected_worker = (
        root / "stages" / "G00_integrity" / "attempt-0001" / "worker"
    )
    selected_worker.mkdir(parents=True)
    _write_json(selected_worker / "result.json", {"passed": True})
    unselected_artifact = (
        root
        / "stages"
        / "G00_integrity"
        / "attempt-9999"
        / "worker"
        / "result.json"
    )
    _write_json(unselected_artifact, {"passed": True})
    row = _evidence_row(
        stage="G00_integrity",
        probe="G00",
        signal="identity",
        status="pass",
        artifact=unselected_artifact,
        signal_path="$.passed",
        observed_value=True,
        threshold=True,
    )
    selected = {"stages": {"G00_integrity": {"attempt": "attempt-0001"}}}

    with pytest.raises(EvidenceIntegrityError, match="selected attempt"):
        _build_evidence_refs(root, [row], selected)


def test_evidence_reference_rejects_missing_signal_path(tmp_path: Path) -> None:
    root = tmp_path / "run"
    artifact = (
        root
        / "stages"
        / "G00_integrity"
        / "attempt-0001"
        / "worker"
        / "result.json"
    )
    _write_json(artifact, {"passed": True})
    row = _evidence_row(
        stage="G00_integrity",
        probe="G00",
        signal="identity",
        status="pass",
        artifact=artifact,
        signal_path="$.missing",
        observed_value=True,
        threshold=True,
    )
    selected = {"stages": {"G00_integrity": {"attempt": "attempt-0001"}}}

    with pytest.raises(EvidenceIntegrityError, match="signal path"):
        _build_evidence_refs(root, [row], selected)


def test_evidence_reference_rejects_wrong_observed_value(tmp_path: Path) -> None:
    root = tmp_path / "run"
    artifact = (
        root
        / "stages"
        / "G00_integrity"
        / "attempt-0001"
        / "worker"
        / "result.json"
    )
    _write_json(artifact, {"passed": True})
    row = _evidence_row(
        stage="G00_integrity",
        probe="G00",
        signal="identity",
        status="pass",
        artifact=artifact,
        signal_path="$.passed",
        observed_value=False,
        threshold=True,
    )
    selected = {"stages": {"G00_integrity": {"attempt": "attempt-0001"}}}

    with pytest.raises(EvidenceIntegrityError, match="observed value"):
        _build_evidence_refs(root, [row], selected)


@pytest.mark.parametrize(
    ("material_count", "gates_passed", "expected"),
    (
        (2, True, CandidateStatus.SUPPORTED_PRIMARY.value),
        (1, True, CandidateStatus.SUPPORTED_CONTRIBUTOR.value),
        (0, True, CandidateStatus.NOT_SUPPORTED.value),
        (0, False, CandidateStatus.INCONCLUSIVE.value),
    ),
)
def test_p50_normal_status_preserves_four_state_semantics(
    material_count: int, gates_passed: bool, expected: str
) -> None:
    impact = {
        "valid_condition_count": 6 if gates_passed else 5,
        "material_condition_count": material_count,
        "gate_status": {
            "contact_instrumentation_neutral": gates_passed,
            "coupon_property": True,
            "positive_initial_clearance": True,
            "filtered_t0_contact_absent": True,
            "all_profiles_contact": gates_passed,
            "precontact_freefall_and_event_alignment": gates_passed,
        },
    }

    assert finalize_module._p50_normal_status(impact) == expected


@pytest.mark.parametrize(
    ("nominal_material", "zero_material", "ratio", "gates_passed", "expected"),
    (
        (True, False, 0.70, True, CandidateStatus.SUPPORTED_PRIMARY.value),
        (True, False, 0.30, True, CandidateStatus.SUPPORTED_CONTRIBUTOR.value),
        (True, False, 0.299, True, CandidateStatus.NOT_SUPPORTED.value),
        (True, True, 0.80, True, CandidateStatus.SUPPORTED_CONTRIBUTOR.value),
        (False, False, None, True, CandidateStatus.NOT_SUPPORTED.value),
        (True, False, 0.80, False, CandidateStatus.INCONCLUSIVE.value),
        (True, False, None, True, CandidateStatus.INCONCLUSIVE.value),
    ),
)
def test_p50_tangential_status_preserves_ratio_and_gate_semantics(
    nominal_material: bool,
    zero_material: bool,
    ratio: float | None,
    gates_passed: bool,
    expected: str,
) -> None:
    slide = {
        "nominal": {"material": nominal_material},
        "zero_friction": {"material": zero_material},
        "explanation_ratio": ratio,
        "gate_status": {
            "contact_instrumentation_neutral": gates_passed,
            "coupon_property": True,
            "all_profiles_contact": gates_passed,
            "normal_gate_or_impulse_within_tolerance": gates_passed,
        },
    }

    assert finalize_module._p50_tangential_status(slide) == expected


def _closure_inputs(
    *,
    p10_vote: bool = False,
    p30_vote: bool = False,
    p40_vote: bool = False,
    p40_ratio: float | None = 0.0,
    p20_mismatch: bool = False,
    contact_free: bool = True,
) -> tuple[dict, dict, dict, dict]:
    p10 = {
        "contact_exclusion": {"passed": contact_free},
        "analyses": {
            "p10_a": {
                "signals": {
                    "all_hinge_velocity": {
                        "max_abs": 0.02 if p10_vote else 0.0,
                        "isaac_repeat_envelope": 0.0,
                        "mujoco_repeat_envelope": 0.0,
                    }
                }
            },
            "p10_c": {
                "signals": {
                    "all_hinge_velocity": {
                        "max_abs": 0.0,
                        "isaac_repeat_envelope": 0.0,
                        "mujoco_repeat_envelope": 0.0,
                    }
                }
            },
        },
    }
    p20 = {
        "audit_valid": True,
        "golden_valid": True,
        "mismatch_detected": p20_mismatch,
    }
    p30 = {
        "contact_exclusion": {"passed": contact_free},
        "analyses": {
            "p30_open_direct": {"material": False},
            "p30_closed_direct": {"material": p30_vote},
            "p30_open_target": {"material": False},
            "p30_closed_target": {"material": False},
        },
    }
    p40 = {
        "contact_exclusion": {"passed": contact_free},
        "analysis": {
            "closure_on": {"material": p40_vote},
            "closure_off": {"material": False},
            "explanation_ratio": p40_ratio,
        },
    }
    return p10, p20, p30, p40


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    (
        (
            {"p40_vote": True, "p40_ratio": 0.70},
            CandidateStatus.SUPPORTED_PRIMARY.value,
        ),
        (
            {"p10_vote": True, "p30_vote": True},
            CandidateStatus.SUPPORTED_PRIMARY.value,
        ),
        (
            {"p10_vote": True},
            CandidateStatus.SUPPORTED_CONTRIBUTOR.value,
        ),
        ({}, CandidateStatus.NOT_SUPPORTED.value),
        (
            {"p40_vote": True, "p40_ratio": 0.80, "p20_mismatch": True},
            CandidateStatus.SUPPORTED_CONTRIBUTOR.value,
        ),
        (
            {"p40_vote": True, "p40_ratio": 0.80, "contact_free": False},
            CandidateStatus.INCONCLUSIVE.value,
        ),
    ),
)
def test_closure_status_enforces_frozen_multi_probe_and_precondition_rules(
    kwargs: dict[str, object], expected: str
) -> None:
    p10, p20, p30, p40 = _closure_inputs(**kwargs)

    status, diagnostics = finalize_module._closure_status(p10, p20, p30, p40)

    assert status == expected
    assert diagnostics["support_count"] == sum(diagnostics["probe_support"].values())


def test_closure_status_is_inconclusive_when_p20_audit_is_invalid() -> None:
    p10, p20, p30, p40 = _closure_inputs(p40_vote=True, p40_ratio=0.90)
    p20["audit_valid"] = False

    status, diagnostics = finalize_module._closure_status(p10, p20, p30, p40)

    assert status == CandidateStatus.INCONCLUSIVE.value
    assert diagnostics["preconditions"]["p20_static_audit_valid"] is False


def test_terminal_adapter_bug_writes_hash_bound_partial_report(tmp_path: Path) -> None:
    root = tmp_path / "adapter-terminal"
    root.mkdir()
    formal = root / "formal.txt"
    formal.write_text("frozen\n", encoding="utf-8")
    formal_hash = sha256_file(formal)
    _write_json(
        root / "run_manifest.json",
        {"manifest_identity_hash": "M" * 64},
    )
    attempts = {
        "G00_integrity": _stage(
            root,
            "G00_integrity",
            {
                "gates": {"scope": True},
                "formal_hashes": {
                    "formal": {
                        "path": str(formal),
                        "actual_sha256": formal_hash,
                        "expected_sha256": formal_hash,
                    }
                },
            },
        ),
        "G01_repeatability": _stage(
            root,
            "G01_repeatability",
            {"repeatability_usable": True},
        ),
        "G02_adapter": _stage(
            root,
            "G02_adapter",
            {
                "terminal_adapter_bug": True,
                "diagnostic_gate_passed": False,
                "adapter_gate": {
                    "classification": "SIM2SIM_ADAPTER_BUG/adapter_chain",
                    "evidence_valid": True,
                    "digital_chain_passed": False,
                    "terminal_adapter_bug": True,
                    "failures": ["mujoco_history_five_current_frames"],
                    "deferred_failures": [],
                    "gates": {"mujoco_history_five_current_frames": False},
                    "gate_groups": {
                        "digital_chain": {
                            "passed": False,
                            "failures": ["mujoco_history_five_current_frames"],
                            "gates": {
                                "mujoco_history_five_current_frames": False
                            },
                        }
                    },
                },
            },
        ),
        "G03_instrumentation": _stage(
            root,
            "G03_instrumentation",
            {"contact_observer_usable": True},
        ),
    }

    final_state = finalize_adapter_terminal(root, verified_attempts=attempts)
    verdict = json.loads((root / "report" / "verdict.json").read_text())
    analysis = json.loads((root / "analysis" / "analysis.json").read_text())

    assert final_state["verdict"] == "SIM2SIM_ADAPTER_BUG"
    assert verdict["primary_classification"] == "SIM2SIM_ADAPTER_BUG"
    assert verdict["conclusion_status"] == "identified"
    assert analysis["failed_gates"] == ["G02_adapter_digital_chain"]
    assert set(analysis["stage_result_sha256"]) == set(attempts)
    assert len(analysis["evidence_refs"]) == 2


def test_unusable_repeatability_writes_inconclusive_partial_report(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repeatability-terminal"
    root.mkdir()
    formal = root / "formal.txt"
    formal.write_text("frozen\n", encoding="utf-8")
    formal_hash = sha256_file(formal)
    _write_json(root / "run_manifest.json", {"manifest_identity_hash": "M" * 64})
    snapshot = _unusable_threshold_snapshot()
    attempts = {
        "G00_integrity": _stage(
            root,
            "G00_integrity",
            {
                "gates": {"scope": True},
                "formal_hashes": {
                    "formal": {
                        "path": str(formal),
                        "actual_sha256": formal_hash,
                        "expected_sha256": formal_hash,
                    }
                },
            },
        ),
        "G01_repeatability": _stage(
            root,
            "G01_repeatability",
            {
                "repeatability_usable": False,
                "exact_executed_coverage": True,
                "threshold_snapshot": snapshot,
                "threshold_snapshot_file": "threshold_snapshot.json",
                "threshold_snapshot_sha256": "pending",
                "executed_key_count": 1,
                "executed_sample_count": 3,
            },
            artifacts={"threshold_snapshot.json": snapshot},
        ),
        "G02_adapter": _stage(
            root,
            "G02_adapter",
            {"terminal_adapter_bug": False},
        ),
        "G03_instrumentation": _stage(
            root,
            "G03_instrumentation",
            {"contact_observer_usable": True},
        ),
    }
    g01_result_path = attempts["G01_repeatability"] / "worker" / "result.json"
    g01_result = json.loads(g01_result_path.read_text(encoding="utf-8"))
    g01_result["threshold_snapshot_sha256"] = sha256_file(
        attempts["G01_repeatability"] / "worker" / "threshold_snapshot.json"
    )
    _write_json(g01_result_path, g01_result)

    final_state = finalize_repeatability_terminal(root, verified_attempts=attempts)
    verdict = json.loads((root / "report" / "verdict.json").read_text())
    analysis = json.loads((root / "analysis" / "analysis.json").read_text())

    assert final_state["verdict"] == "INCONCLUSIVE"
    assert verdict["primary_classification"] == "INCONCLUSIVE"
    assert verdict["conclusion_status"] == "inconclusive"
    assert analysis["failed_gates"] == ["G01_repeatability_unavailable"]
    assert len(analysis["evidence_refs"]) == 2
    assert analysis["threshold_snapshot"]["identity_hash"] == snapshot["identity_hash"]


def test_repeatability_terminal_rejects_missing_threshold_snapshot(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repeatability-terminal-missing-snapshot"
    root.mkdir()
    formal = root / "formal.txt"
    formal.write_text("frozen\n", encoding="utf-8")
    formal_hash = sha256_file(formal)
    _write_json(root / "run_manifest.json", {"manifest_identity_hash": "M" * 64})
    attempts = {
        "G00_integrity": _stage(
            root,
            "G00_integrity",
            {
                "gates": {"scope": True},
                "formal_hashes": {
                    "formal": {
                        "path": str(formal),
                        "actual_sha256": formal_hash,
                        "expected_sha256": formal_hash,
                    }
                },
            },
        ),
        "G01_repeatability": _stage(
            root,
            "G01_repeatability",
            {
                "repeatability_usable": False,
                "exact_executed_coverage": True,
                "unusable_reasons": ["nondeterministic:unbound"],
            },
        ),
        "G02_adapter": _stage(
            root,
            "G02_adapter",
            {"terminal_adapter_bug": False},
        ),
        "G03_instrumentation": _stage(
            root,
            "G03_instrumentation",
            {"contact_observer_usable": True},
        ),
    }

    with pytest.raises(EvidenceIntegrityError, match="threshold snapshot path"):
        finalize_repeatability_terminal(root, verified_attempts=attempts)


def test_finalize_run_builds_hash_bound_core_report(tmp_path: Path) -> None:
    root = tmp_path / "core-unit"
    root.mkdir()
    formal = root / "formal.txt"
    formal.write_text("frozen\n", encoding="utf-8")
    formal_hash = sha256_file(formal)
    _write_json(
        root / "run_manifest.json",
        {"manifest_identity_hash": "M" * 64},
    )

    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "frozen_before_cross_engine_physics": True,
        "minimum_repetitions": 3,
        "effective_tolerance_rule": "max(material_floor,5*repeat_envelope)",
        "high_noise_rule": "mark_key_unusable_when_repeat_envelope_exceeds_material_floor",
        "material_floors": {},
        "executions": [],
        "keys": {},
        "index": {},
    }
    snapshot["identity_hash"] = stable_hash(snapshot)

    attempts: dict[str, Path] = {}
    attempts["G00_integrity"] = _stage(
        root,
        "G00_integrity",
        {
            "gates": {"scope": True},
            "formal_hashes": {
                "formal": {
                    "path": str(formal),
                    "actual_sha256": formal_hash,
                    "expected_sha256": formal_hash,
                }
            },
        },
    )
    attempts["G01_repeatability"] = _stage(
        root,
        "G01_repeatability",
        {
            "repeatability_usable": True,
            "exact_executed_coverage": True,
            "threshold_snapshot": snapshot,
            "threshold_snapshot_file": "threshold_snapshot.json",
            "threshold_snapshot_sha256": "pending",
            "executed_key_count": 0,
            "executed_sample_count": 0,
        },
        artifacts={"threshold_snapshot.json": snapshot},
    )
    g01_result_path = attempts["G01_repeatability"] / "worker" / "result.json"
    g01_result = json.loads(g01_result_path.read_text(encoding="utf-8"))
    g01_result["threshold_snapshot_sha256"] = sha256_file(
        attempts["G01_repeatability"] / "worker" / "threshold_snapshot.json"
    )
    _write_json(g01_result_path, g01_result)

    adapter_gate = {
        "passed": True,
        "evidence_valid": True,
        "digital_chain_passed": True,
        "post_forward_projection_equivalent": True,
        "plant_response_equivalent": True,
        "terminal_adapter_bug": False,
        "classification": "adapter_chain_equivalent",
        "failures": [],
        "deferred_failures": [],
        "all_failures": [],
        "gates": {"contract_identity": True},
        "gate_groups": {
            "digital_chain": {
                "passed": True,
                "failures": [],
                "gates": {"contract_identity": True},
            },
            "post_forward_projection": {
                "passed": True,
                "failures": [],
                "gates": {},
            },
            "plant_response": {"passed": True, "failures": [], "gates": {}},
        },
        "metrics": {},
        "thresholds": {},
        "isaac_evidence_hash": "I" * 64,
        "mujoco_evidence_hash": "J" * 64,
    }
    attempts["G02_adapter"] = _stage(
        root,
        "G02_adapter",
        {"adapter_gate": adapter_gate},
    )
    attempts["G03_instrumentation"] = _stage(
        root,
        "G03_instrumentation",
        {
            "contact_observer_usable": True,
            "isaac_comparisons": {
                "debug": {"passed": True, "continuous": {}},
                "system_observer": {"passed": True, "continuous": {}},
                "contact": {"passed": True, "continuous": {}},
            },
        },
    )

    p10_analyses = {}
    p10_artifacts = {}
    for scenario in ("p10_a", "p10_b", "p10_c"):
        closure_max_abs = 0.2 if scenario == "p10_a" else 0.0
        comparison = {
            "common_times_ms": [0, 20],
            "signals": {
                "all_hinge_velocity": {
                    "rmse": closure_max_abs,
                    "max_abs": closure_max_abs,
                    "isaac_repeat_envelope": 0.0,
                    "mujoco_repeat_envelope": 0.0,
                },
                "kinetic_energy_j": {
                    "rmse": 0.0,
                    "isaac_repeat_envelope": 0.0,
                    "mujoco_repeat_envelope": 0.0,
                }
            },
        }
        p10_analyses[scenario] = comparison
        p10_artifacts[f"analysis/{scenario}.json"] = comparison
    attempts["P10_rest"] = _stage(
        root,
        "P10_rest",
        {"analyses": p10_analyses, "contact_exclusion": {"passed": True}},
        artifacts=p10_artifacts,
    )
    attempts["P20_static_properties"] = _stage(
        root,
        "P20_static_properties",
        {
            "audit_valid": True,
            "golden_valid": True,
            "mismatch_detected": False,
        },
    )

    p30_analyses = {
        name: {
            "material": name == "p30_closed_direct",
            "normalized_rmse": (
                0.2 if name == "p30_closed_direct" else 0.01
            ),
            "normalized_effective_tolerance": 0.1,
            "common_times_ms": [0, 20],
        }
        for name in (
            "p30_open_direct",
            "p30_open_target",
            "p30_closed_direct",
            "p30_closed_target",
        )
    }
    attempts["P30_actuator"] = _stage(
        root,
        "P30_actuator",
        {
            "analyses": p30_analyses,
            "causal_pairs": {"isaac_open": {}, "isaac_closed": {}},
            "contact_exclusion": {"passed": True},
        },
        artifacts={
            f"analysis/{name}.json": value for name, value in p30_analyses.items()
        },
    )
    p40_analysis = {
        "common_times_ms": [0, 20],
        "closure_on": {
            "material": False,
            "normalized_rmse": 0.01,
            "normalized_effective_tolerance": 0.1,
        },
        "closure_off": {
            "material": False,
            "normalized_rmse": 0.01,
            "normalized_effective_tolerance": 0.1,
        },
        "explanation_ratio": 0.0,
    }
    attempts["P40_closure"] = _stage(
        root,
        "P40_closure",
        {
            "analysis": p40_analysis,
            "causal_pairs": {"isaac": {}},
            "contact_exclusion": {"passed": True},
        },
        artifacts={"analysis/p40.json": p40_analysis},
    )
    impact = {
        "normal_primary_supported": False,
        "material_condition_count": 0,
        "valid_condition_count": 1,
        "conditions": [],
        "gate_status": {"valid": True},
    }
    slide = {
        "tangential_primary_supported": False,
        "nominal": {
            "normalized_rmse": 0.01,
            "normalized_effective_tolerance": 0.1,
        },
        "zero_friction": {"normalized_rmse": 0.01},
        "explanation_ratio": 0.0,
        "gate_status": {"valid": True},
        "common_times_ms": [0, 20],
    }
    attempts["P50_contact"] = _stage(
        root,
        "P50_contact",
        {"impact": impact, "slide": slide, "causal_pairs": {"isaac": {}}},
        artifacts={"analysis/impact.json": impact, "analysis/slide.json": slide},
    )

    replay_identity = {
        "source_scenario_id": "P60_D_REPLAY_SOURCE_DREAMWAQ_RUN01",
        "horizon": 3,
        "environment_count": 1,
    }
    replay_identity_hash = stable_hash(replay_identity)
    seal = {
        "schema_version": "RootCauseReplaySourceSealV1",
        "replay_source_identity_hash": replay_identity_hash,
        "source_result_sha256": "A" * 64,
        "source_trace_sha256": "B" * 64,
        "clipped_action_file_sha256": "C" * 64,
        "clipped_action_sequence_sha256": "D" * 64,
        "action_count": 3,
        "g01_attempt": "attempt-0001",
        "g01_stage_state_sha256": "E" * 64,
    }
    seal["seal_identity_hash"] = stable_hash(seal)
    p60_analyses = {
        name: {
            "field": "base_height",
            "material": False,
            "normalized_rmse": 0.01,
            "normalized_effective_tolerance": 0.1,
            "common_times_ms": [0, 20],
        }
        for name in ("p60_b", "p60_c", "p60_d")
    }
    attempts["P60_full_robot"] = _stage(
        root,
        "P60_full_robot",
        {
            "fresh_replay_equivalence": {"passed": True},
            "reset_cache_gate": {"passed": True},
            "replay_source_identity_hash": replay_identity_hash,
            "replay_source_identity": replay_identity,
            "replay_source_seal_identity_hash": seal["seal_identity_hash"],
            "replay_source_seal": seal,
            "replay_evidence": {
                "source_result_sha256": seal["source_result_sha256"],
                "actions_sha256": seal["clipped_action_file_sha256"],
            },
            "analyses": p60_analyses,
        },
        artifacts={
            f"analysis/{name}.json": value for name, value in p60_analyses.items()
        },
    )
    c70 = {
        "anchors": {},
        "checkpoint_role": "not_evaluated",
        "reason": "no_material_plant_failure",
    }
    attempts["C70_checkpoint"] = _stage(
        root,
        "C70_checkpoint",
        {"analysis": c70},
        artifacts={"analysis/c70.json": c70},
    )
    assert set(attempts) == set(CORE_STAGES)

    final_state = finalize_run(root, verified_attempts=attempts)
    analysis = json.loads((root / "analysis" / "analysis.json").read_text())
    verdict = json.loads((root / "report" / "verdict.json").read_text())
    assert final_state["verdict"] == "CLOSED_CHAIN_CONSTRAINT_MISMATCH"
    assert analysis["threshold_snapshot"]["identity_hash"] == snapshot["identity_hash"]
    assert len(analysis["evidence_refs"]) >= 15
    assert all("artifact" in row and "signal_path" in row for row in analysis["evidence_refs"])
    closure_refs = [
        row
        for row in analysis["evidence_refs"]
        if row["supports"] == "CLOSED_CHAIN_CONSTRAINT_MISMATCH"
    ]
    assert {row["stage"] for row in closure_refs} >= {
        "P10_rest",
        "P20_static_properties",
        "P30_actuator",
    }
    all_closure_inputs = {
        (row["stage"], row["signal_path"])
        for row in analysis["evidence_refs"]
        if row["probe"].startswith("CLOSURE_AGGREGATE_")
    }
    assert all_closure_inputs >= {
        ("P10_rest", "$.signals.all_hinge_velocity.max_abs"),
        ("P10_rest", "$.signals.all_hinge_velocity.isaac_repeat_envelope"),
        ("P10_rest", "$.signals.all_hinge_velocity.mujoco_repeat_envelope"),
        ("P30_actuator", "$.material"),
        ("P40_closure", "$.closure_on.material"),
        ("P40_closure", "$.closure_off.material"),
        ("P40_closure", "$.explanation_ratio"),
        ("P20_static_properties", "$.audit_valid"),
        ("P20_static_properties", "$.golden_valid"),
        ("P20_static_properties", "$.mismatch_detected"),
        ("P10_rest", "$.contact_exclusion.passed"),
        ("P30_actuator", "$.contact_exclusion.passed"),
        ("P40_closure", "$.contact_exclusion.passed"),
    }
    assert not any(
        row["stage"] == "P40_closure"
        and row["signal_path"] == "$.closure_on.normalized_rmse"
        and row["supports"] == "CLOSED_CHAIN_CONSTRAINT_MISMATCH"
        for row in analysis["evidence_refs"]
    )
    assert verdict["formal_file_hashes_unchanged"] is True
    assert (root / "report" / "report.md").is_file()
