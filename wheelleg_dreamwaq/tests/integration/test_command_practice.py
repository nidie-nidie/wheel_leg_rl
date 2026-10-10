from __future__ import annotations
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import torch

from conftest import PROJECT_ROOT, run_project_script
from test_dreamwaq_startup import _load_run, _load_resume_trace, _run_training


@pytest.mark.integration
def test_real_physx_command_reward_and_history_timing(tmp_path):
    output = tmp_path / "practice-probe.json"
    run_project_script(["tests/integration/probes/command_practice.py", "--output", str(output), "--headless"], timeout=900)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"], [check for check in report["checks"] if not check["passed"]]
    assert len(report["checks"]) >= 150


@pytest.mark.integration
def test_profile_fresh_resume_fixed_play_export_and_current_evaluation(tmp_path):
    seed, suffix = 20261011, uuid.uuid4().hex[:8]
    source_dir = _run_training(["--profile", "portable", "--task-profile", "stop_reverse_v1",
                               "--max-iterations", "2", "--seed", str(seed),
                               "--run-name", f"practice-source-{suffix}"])
    source = _load_run(source_dir)
    assert source[0]["completed_iterations"] == 2
    assert source[1]["task_profile"] == "stop_reverse_v1"
    assert source[1]["base_task_contract"]["task"]["reward_weights"]["tracking_vx"] == 2.
    checkpoint = source[2]
    for count in (32, 16):
        target_dir = _run_training(["--profile", "portable", "--num-envs", str(count), "--max-iterations", "3",
                                   "--seed", str(seed), "--run-name", f"practice-resume{count}-{suffix}",
                                   "--resume", str(source[2])])
        summary, manifest, checkpoint, payload = _load_run(target_dir)
        assert manifest["task_profile"] == "stop_reverse_v1"  # inferred without CLI override
        assert summary["starting_completed_iterations"] == 2 and summary["iterations_executed"] == 1
        assert summary["completed_iterations"] == 3
        assert payload["infos"]["estimator_monitor_state"]["completed_rollouts"] == 3
        assert manifest["resume_provenance"]["same_num_envs"] is (count == 32)
        _load_resume_trace(target_dir, summary, num_envs=count)
    environment = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    denied = subprocess.run([sys.executable, "scripts/train_dreamwaq.py", "--resume", str(source[2]),
                             "--profile", "portable", "--task-profile", "legacy_v1", "--max-iterations", "3",
                             "--seed", str(seed), "--headless"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, encoding="utf-8", errors="replace",
                             env=environment, timeout=900)
    assert denied.returncode != 0
    assert "profile" in denied.stderr.lower() or "profile" in denied.stdout.lower()
    play_path = tmp_path / "play.json"
    run_project_script(["scripts/play_dreamwaq.py", "--checkpoint", str(checkpoint), "--num-envs", "2",
                        "--steps", "160", "--fixed-command", "0", "0", ".2",
                        "--output", str(play_path), "--headless"], timeout=900)
    play = json.loads(play_path.read_text(encoding="utf-8"))
    assert play["runtime_command_practice"] == "disabled" and play["task_profile"] == "stop_reverse_v1"
    assert play["completed_steps"] == 160 and play["finite"]
    export = tmp_path / "export"
    run_project_script(["scripts/export_dreamwaq_actor.py", "--checkpoint", str(checkpoint), "--output", str(export)])
    manifest = json.loads((export / "policy_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_reward_weights"]["tracking_vx"] == manifest["training_reward_weights"]["tracking_vx_enhance"] == 2.
    assert manifest["command_sampling"]["practice_contract"]["stage_end_control_steps"] == [100, 150, 250, 300, 400, 500]
    golden = torch.load(export / "golden_vectors.pt", map_location="cpu", weights_only=False)
    actor = torch.jit.load(str(export / "actor.ts"), map_location="cpu").eval()
    with torch.inference_mode():
        actual = actor(golden["history"])
        assert actor(golden["history"][:1]).shape == (1, 6)
        assert actor(golden["history"][:3]).shape == (3, 6)
    assert float((actual - golden["expected_action_mean"]).abs().max()) <= 1.e-7
    isaac = tmp_path / "isaac"
    run_project_script(["scripts/evaluate_isaac.py", "--checkpoint", str(checkpoint), "--candidate-only",
                        "--reset-cache", str(tmp_path / "eval-cache.pt"), "--output", str(isaac), "--headless"], timeout=900)
    report = json.loads((isaac / "summary.json").read_text(encoding="utf-8"))
    assert report["evaluation_schema_version"] == "IsaacEvaluationV2"
    assert report["baseline_comparison"] is None
    assert report["baseline_status"] == "not_comparable"
    assert report["evaluation_contract"]["evaluation_reward_weights"]["tracking_vx"] == 1.
    assert report["evaluation_contract"]["evaluation_commands"]["practice_contract"] is None
    assert len(report["aggregate"]["scenarios"]) == 8
    assert all("vx_mae" in row for row in report["aggregate"]["scenarios"])
    dynamic = tmp_path / "dynamic"
    result = subprocess.run(
        [str(PROJECT_ROOT / "sim2sim/mujoco/.venv/Scripts/python.exe"),
         "scripts/evaluate_command_practice_mujoco.py", "--policy", str(export / "actor.ts"),
         "--manifest", str(export / "policy_manifest.json"), "--output", str(dynamic), "--smoke"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    dynamic_report = json.loads((dynamic / "summary.json").read_text(encoding="utf-8"))
    assert dynamic_report["smoke"] and dynamic_report["performance_accepted"] is False
    assert len(dynamic_report["scenarios"]) == 4
    evidence = {"source": str(source_dir), "last_resume": str(checkpoint.parent), "export": str(export),
                "play": str(play_path), "isaac_evaluation": str(isaac / "summary.json"),
                "dynamic_evaluation": str(dynamic / "summary.json")}
    (tmp_path / "acceptance-evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
