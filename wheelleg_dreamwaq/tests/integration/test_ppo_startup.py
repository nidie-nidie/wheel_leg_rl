from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
import torch

from conftest import PROJECT_ROOT, run_project_script
from wheelleg_dreamwaq.training.checkpoint import CHECKPOINT_METADATA_VERSION, sha256_file


@pytest.mark.integration
def test_rtx5070_startup_checkpoint_can_resume_across_hardware_profiles() -> None:
    run_name = f"pytest-{uuid.uuid4().hex[:8]}"
    run_project_script(
        [
            "scripts/train_ppo.py",
            "--profile",
            "rtx5070",
            "--max-iterations",
            "2",
            "--run-name",
            run_name,
            "--headless",
        ],
        timeout=300,
    )

    latest_path = PROJECT_ROOT / "artifacts" / "phase1" / "latest_run.txt"
    run_dir = Path(latest_path.read_text(encoding="utf-8").strip())
    assert run_name in run_dir.name
    summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    checkpoint = Path(summary["final_checkpoint"])
    manifest_path = run_dir / "run_manifest.json"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload["infos"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_optimizer_lr = float(payload["optimizer_state_dict"]["param_groups"][0]["lr"])

    assert summary["iterations_requested"] == 2
    assert summary["starting_completed_iterations"] == 0
    assert summary["iterations_executed"] == 2
    assert summary["completed_iterations"] == 2
    assert summary["runner_final_iteration_index"] == 1
    assert checkpoint.is_file()
    assert list(run_dir.glob("events.out.tfevents.*"))
    assert metadata["metadata_version"] == CHECKPOINT_METADATA_VERSION
    assert metadata["runner_iteration"] == 1
    assert metadata["completed_iterations"] == 2
    assert metadata["run_manifest_sha256"] == sha256_file(manifest_path)
    assert metadata["contract_hash"] == summary["contract_hash"]
    assert metadata["algorithm_learning_rate"] == pytest.approx(source_optimizer_lr)
    assert metadata["optimizer_learning_rates"] == pytest.approx([source_optimizer_lr])
    assert set(metadata["rng_state"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    assert manifest["asset_bundle_version"] == "AssetBundleV2"
    assert manifest["contract"]["manifest_version"] == "Phase1ContractV4"
    assert manifest["contract"]["asset_bundle_version"] == "AssetBundleV2"
    assert manifest["contract"]["runtime"]["collision_policy"] == "WheelOnlyCollisionV2"
    assert manifest["contract"]["schemas"]["reward"] == "RewardSchemaV2"
    assert manifest["contract"]["schemas"]["virtual_leg_kinematics"] == "VirtualLegKinematicsV1"
    assert manifest["contract"]["task"]["physics"] == {
        "sim_dt": 0.005,
        "decimation": 4,
        "control_dt": 0.02,
        "solver_position_iterations": 96,
        "solver_velocity_iterations": 4,
    }
    assert manifest["contract"]["runtime"]["robot_except_asset_absolute_path"]["spawn"]["func"]
    ground = manifest["contract"]["runtime"]["ground"]
    assert ground["prim_path"] == "/World/Ground"
    assert ground["spawn"]["func"]
    assert ground["spawn"]["collision_props"]["collision_enabled"] is True
    assert ground["spawn"]["physics_material"]["friction_combine_mode"] == "multiply"
    assert ground["spawn"]["physics_material"]["restitution_combine_mode"] == "multiply"

    resume_name = f"pytest-resume-{uuid.uuid4().hex[:8]}"
    run_project_script(
        [
            "scripts/train_ppo.py",
            "--profile",
            "portable",
            "--max-iterations",
            "4",
            "--run-name",
            resume_name,
            "--resume",
            str(checkpoint),
            "--headless",
        ],
        timeout=300,
    )
    resumed_run_dir = Path(latest_path.read_text(encoding="utf-8").strip())
    resumed_summary = json.loads((resumed_run_dir / "training_summary.json").read_text(encoding="utf-8"))
    resumed_checkpoint = Path(resumed_summary["final_checkpoint"])
    resumed_payload = torch.load(resumed_checkpoint, map_location="cpu", weights_only=False)
    resumed_manifest = json.loads((resumed_run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert resume_name in resumed_run_dir.name
    assert resumed_summary["iterations_requested"] == 4
    assert resumed_summary["starting_completed_iterations"] == 2
    assert resumed_summary["iterations_executed"] == 2
    assert resumed_summary["completed_iterations"] == 4
    assert resumed_summary["runner_final_iteration_index"] == 3
    assert resumed_summary["restored_learning_rate"] == pytest.approx(source_optimizer_lr)
    assert resumed_payload["iter"] == 3
    assert resumed_payload["infos"]["completed_iterations"] == 4
    resumed_optimizer_lr = float(resumed_payload["optimizer_state_dict"]["param_groups"][0]["lr"])
    assert resumed_payload["infos"]["algorithm_learning_rate"] == pytest.approx(resumed_optimizer_lr)
    assert resumed_manifest["resume"]["source_completed_iterations"] == 2
    assert resumed_manifest["resume"]["mode"] == "weights_optimizer_rng_new_environment"

    run_project_script(
        [
            "scripts/play.py",
            "--checkpoint",
            str(resumed_checkpoint),
            "--num-envs",
            "2",
            "--steps",
            "10",
            "--headless",
        ],
        timeout=240,
    )
