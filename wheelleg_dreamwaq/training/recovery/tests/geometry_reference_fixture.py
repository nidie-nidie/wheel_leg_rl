import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from recovery.contact_model import build_contact_model
from recovery.geometry_guard import GeometryInspector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--bank")
    args = parser.parse_args()
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    xml = build_contact_model(directory / "model")
    config = {"python_executable": __import__("sys").executable, "contact_manifest": str(xml.with_name("contact_manifest.json")),
              "allowed_body_pairs": [], "maximum_ground_penetration_m": .005,
              "minimum_clearance_m": 0.0, "distance_limit_m": .05, "internal_clearance_mode": "diagnostic",
              "diagnostic_envs_per_substep": 32}
    inspector = GeometryInspector(config)
    model, data = inspector.model, inspector.data
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) for index in range(1, model.nbody)]
    positions, quaternions, labels = [], [], []

    def capture(label):
        positions.append(data.xpos[1:].copy().astype(np.float32))
        quaternions.append(data.xquat[1:].copy().astype(np.float32))
        labels.append(label)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    capture("nominal")
    if args.bank:
        records = json.loads(Path(args.bank).read_text(encoding="utf-8"))["records"]
        free = next(index for index in range(model.njnt) if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE)
        qa = int(model.jnt_qposadr[free])
        for index, record in enumerate(records):
            snapshot = record["metadata"]["mujoco_snapshot"]
            mujoco.mj_resetData(model, data)
            data.qpos[qa:qa + 3] = snapshot["root_position"]
            data.qpos[qa + 3:qa + 7] = snapshot["root_quaternion_wxyz"]
            for name, position, velocity in zip(snapshot["joint_names"], snapshot["joint_positions"], snapshot["joint_velocities"], strict=True):
                joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                data.qpos[model.jnt_qposadr[joint]] = position
                data.qvel[model.jnt_dofadr[joint]] = velocity
            mujoco.mj_forward(model, data)
            capture(record["record_id"])
            if index % 10 == 0:
                for step in range(20):
                    mujoco.mj_step(model, data)
                    capture(f"physical:{index}:{step}")
    p, q = np.asarray(positions), np.asarray(quaternions)
    reports = inspector.inspect(names, p, q)
    native = np.asarray([report["maximum_ground_penetration_m"] for report in reports])
    np.savez(directory / "reference.npz", positions=p, quaternions=q, native_penetration=native)
    (directory / "reference.json").write_text(json.dumps({"config": config, "body_names": names, "labels": labels}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
