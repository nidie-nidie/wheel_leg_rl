from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from debug.sim2sim.collect_recovery_stand_reference_20261029 import (
    COMMAND,
    SIM,
    load_actor,
)
from wheelleg_mujoco.runner import WheelLegMujocoRuntime


PROJECT = Path(__file__).resolve().parents[2]
ARTIFACT = PROJECT / "recovery/artifacts/standing-reference401-20261029"


def main() -> None:
    started = perf_counter()
    torch.set_num_threads(1)
    actor, contract, model_manifest, identity = load_actor()
    runtime = WheelLegMujocoRuntime(
        SIM / "models" / model_manifest["model_xml"]["path"], contract, model_manifest
    )
    runtime.heading.enabled = False
    runtime.reset(COMMAND)
    reference_path = ARTIFACT / "formal401_00.npz"
    ticks = 300
    comparisons = {
        name: {"exact_equal": True, "max_abs_difference": 0.0, "first_difference_tick": None}
        for name in ("observation", "action_raw", "qpos_mujoco", "qvel_mujoco")
    }
    with np.load(reference_path, allow_pickle=False) as reference:
        if len(reference["time_s"]) < ticks:
            raise ValueError("名义采集记录不足300步")
        for tick in range(ticks):
            observation = runtime.prepare_policy_observation(COMMAND)
            with torch.inference_mode():
                action = actor(torch.from_numpy(observation).view(1, -1)).numpy()[0].copy()
            runtime.step(action, COMMAND)
            actual = {
                "observation": observation,
                "action_raw": action,
                "qpos_mujoco": runtime.data.qpos,
                "qvel_mujoco": runtime.data.qvel,
            }
            for name, value in actual.items():
                expected = reference[name][tick]
                if value.shape != expected.shape or not np.isfinite(value).all():
                    raise ValueError(f"字段形状或有限性错误: {name}")
                result = comparisons[name]
                difference = float(np.max(np.abs(value - expected)))
                result["max_abs_difference"] = max(result["max_abs_difference"], difference)
                if not np.array_equal(value, expected):
                    result["exact_equal"] = False
                    if result["first_difference_tick"] is None:
                        result["first_difference_tick"] = tick + 1
    passed = all(value["exact_equal"] for value in comparisons.values())
    report = {
        "schema": "StandingReference401RuntimeVerificationV1",
        "passed": passed,
        "comparison": "unwrapped_formal_runtime_against_collector_with_copy_forward_snapshots",
        "identity": identity,
        "reference": str(reference_path),
        "device": "cpu",
        "torch_threads": 1,
        "ticks": ticks,
        "simulated_s": ticks * contract.control_dt_s,
        "comparisons": comparisons,
        "policy_updates": 0,
        "optimizer_updates": 0,
        "wall_time_s": perf_counter() - started,
        "limitations": ["nominal yaw0 first 300 control steps only; not proof for all scenarios"],
    }
    output = ARTIFACT / "runtime_verification.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
