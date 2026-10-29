import argparse
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from wheelleg_dreamwaq.schemas.dreamwaq_manifest import stable_contract_hash
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.training_profiles import (
    RAW_BASELINE_YAW1_NO_HEADING_PROFILE, resolve_task_profile,
)
from wheelleg_dreamwaq.training.checkpoint import sha256_file
from wheelleg_dreamwaq.training.dreamwaq_checkpoint import load_dreamwaq_checkpoint_artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", type=Path, required=True)
    args = parser.parse_args()
    folder = args.launch.resolve()
    launch = json.loads((folder / "launch.json").read_text(encoding="utf-8-sig"))
    run = Path(launch["run_dir"])
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    reference = json.loads(Path(launch["comparison_manifest"]).read_text(encoding="utf-8"))
    assert manifest["num_envs"] == launch["num_envs"]
    assert manifest["max_iterations"] == launch["target_iterations"]
    assert manifest["seed"] == 41882943 and manifest["num_steps_per_env"] == 24
    assert manifest["num_mini_batches"] == 8 and manifest["device"] == "cuda:0"
    assert manifest["task_profile"] == RAW_BASELINE_YAW1_NO_HEADING_PROFILE
    assert manifest["resume_provenance"]["mode"] == "fresh"
    assert "weights_initialization" not in manifest
    assert "--resume" not in manifest["command"] and "--initialize-from" not in manifest["command"]
    base = manifest["base_task_contract"]
    expected = deepcopy(reference["base_task_contract"])
    expected.pop("contract_hash")
    assert expected["task"]["heading_hold"]["enabled"] is False
    for key in ("tracking_yaw_rate", "tracking_yaw_rate_enhance"):
        assert expected["task"]["reward_weights"][key] == 2.
        expected["task"]["reward_weights"][key] = 1.
    expected["contract_hash"] = stable_contract_hash(expected)
    assert base == expected, "任务配置存在两个偏航权重之外的变化"
    assert manifest["dreamwaq_algorithm_contract"] == reference["dreamwaq_algorithm_contract"]
    assert manifest["dreamwaq_export_contract"] == reference["dreamwaq_export_contract"]
    assert resolve_task_profile(None, base) == RAW_BASELINE_YAW1_NO_HEADING_PROFILE
    frozen = json.loads((folder / "source_before.json").read_text(encoding="utf-8-sig"))
    assert frozen and len(frozen) == launch["frozen_file_count"]
    changed = [item["path"] for item in frozen if sha256_file(Path(item["path"])) != item["sha256"]]
    assert not changed, changed
    events = EventAccumulator(str(run), size_guidance={"scalars": 2}).Reload()
    latest = {tag: events.Scalars(tag)[-1] for tag in events.Tags()["scalars"] if events.Scalars(tag)}
    assert "Loss/total" in latest
    assert all(math.isfinite(event.value) for event in latest.values())
    checkpoints = sorted(run.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[-1]))
    assert checkpoints and checkpoints[0].name == "model_0.pt"
    first, _, first_info = load_dreamwaq_checkpoint_artifact(checkpoints[0])
    steps = sorted({float(value["step"]) for value in first["optimizer_state_dict"]["state"].values()})
    assert first_info["completed_iterations"] == 1 and steps == [40.]
    summary_path = run / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    if summary:
        assert summary["completed_iterations"] == summary["iterations_executed"] == launch["target_iterations"]
        assert summary["starting_completed_iterations"] == 0 and summary["resumed_from"] is None
        assert summary["estimator_monitor_state"]["completed_rollouts"] == launch["target_iterations"]
        assert checkpoints[-1].name == f"model_{launch['target_iterations'] - 1}.pt"
    report = {
        "observed_at": datetime.now().astimezone().isoformat(), "run_dir": str(run),
        "state": "completed" if summary else "running", "validation_only": launch["validation_only"],
        "completed_iterations": summary["completed_iterations"] if summary else latest["Loss/total"].step + 1,
        "target_iterations": launch["target_iterations"], "num_envs": manifest["num_envs"],
        "seed": manifest["seed"], "initialization": "fresh", "task_profile": manifest["task_profile"],
        "heading": base["task"]["heading_hold"], "reward_weights": base["task"]["reward_weights"],
        "only_effective_task_config_differences": ["reward_weights.tracking_yaw_rate", "reward_weights.tracking_yaw_rate_enhance"],
        "algorithm_contract_exact_equal": True, "export_contract_exact_equal": True,
        "frozen_files_verified": len(frozen), "latest_scalars_finite": True,
        "first_checkpoint_completed_iterations": 1, "first_checkpoint_optimizer_steps": steps,
        "latest_checkpoint": str(checkpoints[-1]),
        "latest_scalars": {key: {"step": value.step, "value": value.value} for key, value in latest.items()},
    }
    (folder / "monitor_snapshot.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "latest_scalars"}, indent=2))


if __name__ == "__main__":
    main()
