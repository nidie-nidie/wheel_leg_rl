from __future__ import annotations

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

from debug.sim2sim.root_cause_suite import core_stage_worker
from debug.sim2sim.root_cause_suite.contracts import sha256_file, stable_hash
from debug.sim2sim.root_cause_suite.core_stage_worker import p50_normal_gate_decision


def test_p50_normal_gate_truth_table_is_fail_closed() -> None:
    for impact_valid, raw_primary, impulse_valid, impulse_material in itertools.product(
        (False, True), repeat=4
    ):
        decision = p50_normal_gate_decision(
            impact_evidence_valid=impact_valid,
            raw_normal_primary_supported=raw_primary,
            normal_impulse_evidence_valid=impulse_valid,
            normal_impulse_material=impulse_material,
        )
        expected = (impact_valid and not raw_primary) or (
            impulse_valid and not impulse_material
        )
        assert decision["passed"] is expected


def test_p50_invalid_impact_cannot_hide_material_normal_impulse() -> None:
    decision = p50_normal_gate_decision(
        impact_evidence_valid=False,
        raw_normal_primary_supported=False,
        normal_impulse_evidence_valid=True,
        normal_impulse_material=True,
    )
    assert decision["passed"] is False


def test_run_p60_passes_the_actual_mujoco_replay_result_to_analysis(
    tmp_path: Path, monkeypatch,
) -> None:
    run_root = tmp_path / "run"
    output = run_root / "worker"
    source = run_root / "g01" / "replay_source"
    fresh = run_root / "g01" / "isaac_replay"
    mujoco_replay = run_root / "g01" / "mujoco_replay"
    actions = source / "actions.npz"
    cache = run_root / "cache.pt"
    for directory in (output, source, fresh, mujoco_replay):
        directory.mkdir(parents=True, exist_ok=True)
    for path in (
        source / "result.json",
        fresh / "result.json",
        mujoco_replay / "result.json",
        actions,
        cache,
    ):
        path.write_bytes(b"fixture\n")

    replay_identity = {"source": "unit", "horizon": 3}
    replay_identity_hash = stable_hash(replay_identity)
    source_result = {
        "first_action": [0.0] * 6,
        "replay_source_identity": replay_identity,
        "replay_source_identity_hash": replay_identity_hash,
    }
    replay_seal = {
        "replay_source_identity_hash": replay_identity_hash,
        "seal_identity_hash": "S" * 64,
    }
    fresh_result = {"fresh_replay_equivalence": {"passed": True}}
    actual_mujoco_replay_result = {"engine": "mujoco", "marker": "actual"}
    cache_sha256 = sha256_file(cache)
    probe_results = {}
    for scenario in ("p60_b", "p60_c"):
        for engine in ("isaac", "mujoco"):
            probe = run_root / "g01" / f"{engine}_{scenario}"
            probe.mkdir(parents=True)
            result = {"engine": engine, "scenario": scenario}
            if engine == "isaac":
                result.update(
                    {
                        "reset_cache_path": str(cache),
                        "reset_cache_sha256": cache_sha256,
                        "reset_cache_environment_rows": [0],
                    }
                )
            probe_results[(engine, scenario)] = (probe, result)
    probe_results[("isaac", "replay_0")] = (fresh, fresh_result)
    probe_results[("mujoco", "replay_0")] = (
        mujoco_replay,
        actual_mujoco_replay_result,
    )

    monkeypatch.setattr(
        core_stage_worker,
        "_g01_evidence",
        lambda _runner: (run_root / "g01", {}, {"identity_hash": "T" * 64}, "G" * 64),
    )
    monkeypatch.setattr(
        core_stage_worker,
        "_g01_replay_source",
        lambda _worker, _result: (source, source_result, actions),
    )
    monkeypatch.setattr(
        core_stage_worker, "verified_replay_source_seal", lambda _root: replay_seal
    )
    monkeypatch.setattr(
        core_stage_worker,
        "_g01_probe",
        lambda _worker, _result, *, engine, label: probe_results[(engine, label)],
    )
    monkeypatch.setattr(
        core_stage_worker,
        "compare_p60_robot_trace_directories",
        lambda *_args, scenario, **_kwargs: {
            "scenario": scenario,
            "field": "base_height",
            "material": False,
        },
    )

    captured = {}

    def compare_replay(*_args, mujoco_result, **_kwargs):
        captured["mujoco_result"] = mujoco_result
        return {"field": "base_height", "material": False}

    monkeypatch.setattr(
        core_stage_worker, "compare_p60_replay_trace_directories", compare_replay
    )

    fake_tensor = SimpleNamespace(shape=(8, 6))
    monkeypatch.setattr(
        core_stage_worker,
        "FROZEN_REPLAY_SOURCE",
        {
            "reset_cache": SimpleNamespace(
                absolute_path=lambda: cache, sha256=cache_sha256
            ),
            "reset_cache_schema": "cache-v1",
            "reset_cache_relaxation_algorithm": "relax-v1",
            "reset_cache_root_height_algorithm": "height-v1",
            "environment_count": 8,
            "environment_index": 0,
        },
    )
    import torch
    import wheelleg_dreamwaq.schemas.randomization as randomization

    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        randomization,
        "validate_closed_chain_reset_cache_artifact",
        lambda _payload: {
            "schema_version": "cache-v1",
            "algorithm_version": "relax-v1",
            "root_height_algorithm_version": "height-v1",
            "tensor_sha256": "tensor-hash",
            "q_reset_projected_env": fake_tensor,
        },
    )
    core_stage_worker.FROZEN_REPLAY_SOURCE["reset_cache_tensor_sha256"] = (
        "tensor-hash"
    )

    class Runner:
        stage = "P60_full_robot"

        def __init__(self) -> None:
            self.run_root = run_root
            self.output = output

        def write_result(self, payload):
            (output / "result.json").write_text(json.dumps(payload), encoding="utf-8")
            return payload

    result = core_stage_worker._run_p60(Runner())

    assert captured["mujoco_result"] is actual_mujoco_replay_result
    assert result["passed"] is True
