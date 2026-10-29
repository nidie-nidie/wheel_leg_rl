import json
import subprocess
import uuid

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from conftest import PROJECT_ROOT, run_project_script
from test_dreamwaq_startup import _run_training, _load_run


@pytest.mark.integration
def test_split_fresh_resume_export_and_real_mujoco_loader(tmp_path):
    suffix = uuid.uuid4().hex[:8]
    source = _run_training(["--profile", "portable", "--task-profile", "stop_reverse_split_smooth_v1",
                            "--max-iterations", "2", "--seed", "20261013", "--run-name", "split-smoke-" + suffix])
    summary, manifest, checkpoint, _ = _load_run(source)
    assert summary["completed_iterations"] == 2
    assert manifest["base_task_contract"]["schemas"]["reward"] == "RewardSchemaV3SplitSmooth"
    assert "leg_acceleration" not in manifest["base_task_contract"]["task"]
    resumed = _run_training(["--profile", "portable", "--resume", str(checkpoint),
                             "--max-iterations", "3", "--seed", "20261013", "--run-name", "split-resume-" + suffix])
    summary, saved, checkpoint, _ = _load_run(resumed)
    assert summary["starting_completed_iterations"] == 2 and summary["completed_iterations"] == 3
    assert saved["task_profile"] == "stop_reverse_split_smooth_v1"
    tags = EventAccumulator(str(source)).Reload().Tags()["scalars"]
    assert "Reward/action_smooth_leg" in tags and "Reward/action_smooth_wheel" in tags
    assert "Reward/action_smooth" not in tags
    export = tmp_path / "export"
    run_project_script(["training/scripts/export_dreamwaq_actor.py", "--checkpoint", str(checkpoint), "--output", str(export)])
    actor = torch.jit.load(str(export / "actor.ts")).eval()
    golden = torch.load(export / "golden_vectors.pt", weights_only=False, map_location="cpu")
    with torch.inference_mode():
        actual = actor(golden["history"])
        assert actual.shape == (32, 6)
        assert (actual - golden["expected_action_mean"]).abs().max() <= 1e-7
        assert actor(golden["history"][:1]).shape == (1, 6)
        assert actor(golden["history"][:3]).shape == (3, 6)
    output = tmp_path / "mujoco-loader-smoke"
    subprocess.run([str(PROJECT_ROOT / "sim2sim/mujoco/.venv/Scripts/python.exe"),
                    "training/scripts/evaluate_command_practice_mujoco.py", "--policy", str(export / "actor.ts"),
                    "--manifest", str(export / "policy_manifest.json"), "--output", str(output), "--smoke"],
                   cwd=PROJECT_ROOT, check=True, timeout=180, capture_output=True)
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["smoke"] and not report["performance_accepted"]
    evidence = {"fresh": str(source), "resume": str(resumed), "export": str(export),
                "mujoco_loader_smoke": str(output), "golden_batches": [1, 3, 32]}
    (tmp_path / "acceptance-evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
