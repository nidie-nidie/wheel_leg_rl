from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from recovery.contracts import write_json
from recovery.mujoco_runtime import MujocoRecoveryRuntime
from recovery.pose_bank import PoseBank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads((arguments.run / "resolved_config.json").read_text(encoding="utf-8"))
    actual = json.loads((arguments.run / "initial_body_pose.json").read_text(encoding="utf-8"))
    bank = PoseBank.load(config["environment"]["pose_bank_path"])
    record = next(item for item in bank.records if item.record_id == actual["record_id"])
    contact = Path(__file__).resolve().parents[3] / "recovery/artifacts/mujoco_verification/model/contact_manifest.json"
    runtime = MujocoRecoveryRuntime(contact, config["environment"]["walker_export_dir"],
                                   actual_parameters=record.metadata["actual_parameters"])
    runtime.reset_nominal()
    runtime.restore_pose(record)
    reports = []
    bodies = sorted({int(runtime.model.geom_bodyid[index]) for index in range(runtime.model.ngeom)
                     if runtime.model.geom_contype[index] == 2})
    for body in bodies:
        name = mujoco.mj_id2name(runtime.model, mujoco.mjtObj.mjOBJ_BODY, body)
        usd_name = "base_link" if name == "base" else name
        if usd_name not in actual["body_names"]:
            raise ValueError(f"Isaac 缺少同源真实碰撞刚体: {usd_name}")
        index = actual["body_names"].index(usd_name)
        position_error = float(np.max(np.abs(np.asarray(actual["positions"][0][index]) - runtime.data.xpos[body])))
        quaternion = np.asarray(actual["quaternions_wxyz"][0][index])
        reference = runtime.data.xquat[body]
        quaternion_error = float(min(np.max(np.abs(quaternion - reference)), np.max(np.abs(quaternion + reference))))
        reports.append({"body_name": name, "position_max_abs_error_m": position_error,
                        "quaternion_max_abs_error": quaternion_error})
    result = {"record_id": record.record_id, "bodies": reports,
              "maximum_body_position_error_m": max(item["position_max_abs_error_m"] for item in reports),
              "maximum_quaternion_error": max(item["quaternion_max_abs_error"] for item in reports)}
    result["passed"] = result["maximum_body_position_error_m"] <= 1e-5 and result["maximum_quaternion_error"] <= 1e-5
    write_json(arguments.run / "body_pose_cross_sim_validation.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not result["passed"]:
        raise ValueError("真实 Isaac 与源 MuJoCo 刚体坐标系不一致，几何审查不能直接复用")


if __name__ == "__main__":
    main()
