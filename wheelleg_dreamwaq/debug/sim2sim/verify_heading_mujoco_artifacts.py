from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
import torch

from wheelleg_mujoco.contract import load_policy_contract, sha256_file
from wheelleg_mujoco.model_semantics import stable_hash
from wheelleg_mujoco.runner import WheelLegMujocoRuntime


PROJECT = Path(__file__).resolve().parents[2]
SIM = PROJECT / "sim2sim/mujoco"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    contract, policy_manifest, model = load_policy_contract(args.export / "policy_manifest.json", args.export / "actor.ts",
                                                         SIM / "model_manifest.json")
    rejected_cases = {}
    for case in ("missing_heading", "kp", "missing_geometry", "clipped_action"):
        directory = args.output / ("invalid_" + case)
        shutil.copytree(args.export, directory)
        changed = deepcopy(policy_manifest)
        if case == "missing_heading":
            del changed["heading_hold"]
        elif case == "kp":
            changed["heading_hold"]["kp"] = 3
        elif case == "missing_geometry":
            del changed["runtime_geometry"]
        else:
            changed["action"]["clip_actions"] = 1.
        changed["manifest_hash"] = stable_hash({k: v for k, v in changed.items() if k != "manifest_hash"})
        (directory / "policy_manifest.json").write_text(json.dumps(changed), encoding="utf-8")
        try:
            load_policy_contract(directory / "policy_manifest.json", directory / "actor.ts", SIM / "model_manifest.json")
        except ValueError:
            rejected_cases["deployment_" + case] = True
        else:
            raise AssertionError("篡改的部署契约被接受")
    runtime = WheelLegMujocoRuntime(SIM / "models" / model["model_xml"]["path"], contract, model)
    vectors = []
    for start, current, source_yaw in ((0., .1, 0.), (3.13, -3.13, 0.), (0., .3, .6), (0., -.3, -.6)):
        source = np.array([1., source_yaw, .2])
        runtime.reset(source)
        runtime.data.qpos[3:7] = [np.cos(start / 2), 0, 0, np.sin(start / 2)]
        mujoco.mj_forward(runtime.model, runtime.data)
        runtime.heading.reset(runtime._heading_yaw())
        runtime.data.qpos[3:7] = [np.cos(current / 2), 0, 0, np.sin(current / 2)]
        mujoco.mj_forward(runtime.model, runtime.data)
        prepared = runtime.prepare_policy_observation(source)
        expected = source_yaw if source_yaw != 0 else np.clip(9 * np.arctan2(
            np.sin(start - current), np.cos(start - current)), -1, 1)
        assert abs(runtime.effective_command[1] - expected) < 1e-12
        vectors.append({"rotation": runtime.data.xmat[runtime.model_map.base_body_id].copy(),
                        "source": source.copy(), "reference": runtime.heading.reference.clone(),
                        "effective": runtime.effective_command.copy(), "history": prepared.copy()})
        source[1] = 0
        prepared = runtime.prepare_policy_observation(source)
        if source_yaw != 0:
            assert abs(runtime.effective_command[1]) < 1e-12
        vectors.append({"source": source.copy(), "reference": runtime.heading.reference.clone(),
                        "effective": runtime.effective_command.copy(), "history": prepared.copy()})
    path = args.output / "heading_vectors.pt"
    vectors = [{key: torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value
                for key, value in vector.items()} for vector in vectors]
    torch.save(vectors, path)
    report = {"rejected_contract_cases": rejected_cases,
              "heading_vectors": {"count": len(vectors), "sha256": sha256_file(path)}}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
