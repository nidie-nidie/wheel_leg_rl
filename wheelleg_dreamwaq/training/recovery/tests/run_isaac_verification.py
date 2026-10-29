from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import subprocess
import sys

from recovery.contracts import PROJECT_ROOT, default_resolved_config, write_json
from recovery.randomization import RecoveryRandomizationV1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose-bank", type=Path, default=PROJECT_ROOT / "recovery/artifacts/mujoco_verification/interface_pose_bank.json")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--randomization", action="store_true")
    parser.add_argument("--geometry-config", type=Path)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = default_resolved_config()
    config["environment"].update({
        "pose_bank_path": str(arguments.pose_bank.resolve()),
        "walker_export_dir": str(PROJECT_ROOT / "logs/rsl_rl/wheelleg_flat_dreamwaq/2026-10-17_17-56-29_rtx5070_fudan-v1_baseline-yaw1-headingoff-env4096-fresh701-20261017/export"),
        "run_dir": str(output / "run"), "num_envs": arguments.num_envs,
    })
    if not arguments.randomization:
        config["randomization"] = asdict(RecoveryRandomizationV1(name="RecoveryNominalEvaluationV1", enabled=False))
    if arguments.geometry_config is not None:
        config["readiness"]["geometry_guard"] = json.loads(arguments.geometry_config.read_text(encoding="utf-8"))
    config["assessment"]["updates"] = 1
    config_path = output / "interface_config.json"
    write_json(config_path, config)
    command = [sys.executable, "-u", "-m", "recovery.train", "--config", str(config_path),
               "--headless", "--device", config["environment"]["device"], "--verify-interface", "--verify-resume"]
    with (output / "isaac_stdout.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
    print(f"ISAAC_VERIFICATION_EXIT={result.returncode} LOG={output / 'isaac_stdout.log'}", flush=True)
    if result.returncode:
        raise RuntimeError(f"真实 Isaac 验证失败，请检查 {output / 'isaac_stdout.log'}")
    verification_path = output / "run/training_verification.json"
    if not verification_path.is_file():
        raise RuntimeError("Isaac 进程未产生真实训练验证报告，退出码不足以证明通过")
    report = json.loads(verification_path.read_text(encoding="utf-8"))
    if not report["parameters_changed"] or not report["resume_and_continued_optimization_verified"]:
        raise RuntimeError("真实 PPO 更新或 checkpoint 继续优化尚未通过")


if __name__ == "__main__":
    main()
