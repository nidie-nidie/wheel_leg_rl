from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from recovery.contracts import PROJECT_ROOT, write_json
from recovery.mujoco_runtime import MujocoRecoveryRuntime
from recovery.pose_bank import PoseBank
from recovery.randomization import RecoveryRandomization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42001)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    parameters = RecoveryRandomization(arguments.num_envs, arguments.seed)
    contact = PROJECT_ROOT / "recovery/artifacts/mujoco_verification/model/contact_manifest.json"
    walker = PROJECT_ROOT / "logs/rsl_rl/wheelleg_flat_dreamwaq/2026-10-17_17-56-29_rtx5070_fudan-v1_baseline-yaw1-headingoff-env4096-fresh701-20261017/export"
    records = []
    for index, plan in enumerate(parameters.plans):
        runtime = MujocoRecoveryRuntime(contact, walker, actual_parameters=asdict(plan))
        if runtime.parameter_sha256 != plan.sha256:
            raise ValueError("MuJoCo 实际参数和 Isaac 计划身份不同")
        for split in ("train", "test"):
            runtime.reset_nominal()
            for step in range(9 if split == "train" else 11):
                runtime.step(runtime.walker_action(), locomotion=True)
                if step < 5:
                    continue
                records.append(runtime.pose_record(record_id=f"dr-{arguments.num_envs}-{index}-{split}-{step}",
                    trajectory_id=f"dr-{arguments.num_envs}-{index}-{split}", split=split,
                    category="near_upright", motion_kind="dynamic", metadata={"origin": "interface_validation"}))
    bank = PoseBank(records, {"purpose": "interface_validation", "num_envs": arguments.num_envs,
                             "master_seed": arguments.seed, "randomization": parameters.manifest()})
    bank.save(output / "interface_pose_bank.json")
    write_json(output / "collection.json", {"record_count": len(records), "pose_bank_sha256": bank.sha256,
               "randomization": parameters.manifest(), "fall_recovery_coverage_validated": False})
    print(f"REAL_DR_INTERFACE_POSES={len(records)} BANK={output / 'interface_pose_bank.json'}", flush=True)


if __name__ == "__main__":
    main()
