from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery.contracts import PROJECT_ROOT, file_sha256, stable_hash, write_json
from recovery.independent_config import default_independent_config, validate_independent_config
from recovery.independent_poses import IndependentPoseBank


def prepare_config(pose_bank, calibration, geometry, run_dir, *, formal=False):
    bank = IndependentPoseBank.load(pose_bank)
    config = default_independent_config()
    ids = bank.split_ids()
    config["initialization"].update(train_ids=ids["train"], test_ids=ids["test"])
    config["environment"].update(pose_bank_path=str(Path(pose_bank).resolve()), run_dir=str(Path(run_dir).resolve()))
    config["geometry_guard"] = geometry
    config["assessment"].update(calibration_path=str(Path(calibration).resolve()), calibration_sha256=file_sha256(calibration))
    if formal:
        config["environment"].update(num_envs=4096, task_type="formal_training")
        config["ppo"]["max_iterations"] = 4000
        config["environment"]["physx_buffers"] = {"gpu_max_rigid_contact_count": 1048576,
            "gpu_max_rigid_patch_count": 524288, "gpu_found_lost_pairs_capacity": 1048576,
            "gpu_found_lost_aggregate_pairs_capacity": 1048576, "gpu_total_aggregate_pairs_capacity": 1048576}
    else:
        config["ppo"]["max_iterations"] = 4
    validate_independent_config(config)
    bank.validate_split_ids(config["initialization"])
    return config


def main():
    parser = argparse.ArgumentParser(description="生成独立站立新版本的完整配置，正式训练须另外通过准入")
    parser.add_argument("--pose-bank", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    geometry = json.loads(args.geometry_config.read_text(encoding="utf-8"))
    config = prepare_config(args.pose_bank, args.calibration, geometry, args.run_dir, formal=args.formal)
    write_json(args.output, config)
    print(json.dumps({"config": str(args.output.resolve()), "config_sha256": stable_hash(config),
                      "training_started": False, "formal_preflight_executed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
