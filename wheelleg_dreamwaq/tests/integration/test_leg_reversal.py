from __future__ import annotations
import json
import sys
import uuid
from pathlib import Path

import pytest
import torch

from conftest import run_project_script, PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from export_dreamwaq_actor import _build_policy
from wheelleg_dreamwaq.deployment.dreamwaq_inference import DreamWaQInferenceActorV1
from test_dreamwaq_startup import _load_run, _run_training
from wheelleg_dreamwaq.schemas.randomization import canonical_tensor_sha256


@pytest.mark.integration
def test_real_reversal_control_tick_partial_reset_and_repeated_reads(tmp_path):
    output = tmp_path / "probe.json"
    run_project_script(["tests/integration/probes/leg_reversal.py",
                        "--output", str(output), "--headless"], timeout=900)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] and report["counted_control_steps"] == 72
    assert report["rollouts"] == 3


@pytest.mark.integration
def test_real_weights_initialization_resume_play_and_golden(tmp_path):
    seed = 20261011
    suffix = uuid.uuid4().hex[:8]
    source_dir = _run_training(["--profile", "portable", "--task-profile", "stop_reverse_v1",
        "--seed", str(seed), "--max-iterations", "2", "--run-name", f"reversal-source-{suffix}"])
    source = _load_run(source_dir)
    initialized_dir = _run_training(["--profile", "portable", "--task-profile", "stop_reverse_reversal12_v1",
        "--seed", str(seed), "--max-iterations", "3", "--initialize-from", str(source[2]),
        "--run-name", f"reversal-init-{suffix}"])
    summary, manifest, cp, payload = _load_run(initialized_dir)
    assert summary["starting_completed_iterations"] == 0 and summary["resumed_from"] is None
    assert summary["completed_iterations"] == 3
    assert summary["initial_model_tensor_sha256"] == canonical_tensor_sha256(source[3]["model_state_dict"])
    assert payload["infos"]["leg_reversal_gain_state"]["completed_rollouts"] == 3
    assert manifest["weights_initialization"]["source_completed_iterations"] == 2
    resumed_dir = _run_training(["--profile", "portable", "--seed", str(seed), "--max-iterations", "4",
        "--resume", str(cp), "--run-name", f"reversal-resume-{suffix}"])
    resumed = _load_run(resumed_dir)
    assert resumed[0]["starting_completed_iterations"] == 3
    assert resumed[0]["completed_iterations"] == 4
    assert resumed[3]["infos"]["leg_reversal_gain_state"]["completed_rollouts"] == 4
    assert resumed[1]["weights_initialization"] == manifest["weights_initialization"]
    play = tmp_path / "play.json"
    run_project_script(["scripts/play_dreamwaq.py", "--checkpoint", str(resumed[2]),
        "--num-envs", "2", "--steps", "8", "--fixed-command", "0", "0", ".2",
        "--output", str(play), "--headless"], timeout=900)
    assert json.loads(play.read_text(encoding="utf-8"))["finite"]
    export = tmp_path / "export"
    run_project_script(["scripts/export_dreamwaq_actor.py", "--checkpoint", str(resumed[2]),
        "--output", str(export)], timeout=900)
    actor = torch.jit.load(str(export / "actor.ts"), map_location="cpu").eval()
    golden = torch.load(export / "golden_vectors.pt", map_location="cpu", weights_only=False)
    python_actor = DreamWaQInferenceActorV1.from_policy(
        _build_policy(resumed[3]["model_state_dict"])).eval()
    with torch.inference_mode():
        assert torch.allclose(actor(golden["history"]), golden["expected_action_mean"], atol=1e-7, rtol=0)
        for batch in (1, 3, 32):
            actual = actor(golden["history"][:batch])
            assert actual.shape == (batch, 6)
            assert torch.allclose(actual, python_actor(golden["history"][:batch]), atol=1e-7, rtol=0)
    (tmp_path / "evidence.json").write_text(json.dumps({
        "source": str(source_dir), "initialized": str(initialized_dir),
        "resumed": str(resumed_dir), "export": str(export)}, indent=2), encoding="utf-8")
