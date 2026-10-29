from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from debug.no_action_clip.mujoco_adapter import load_runtime, RawActionRuntime
from wheelleg_mujoco.contract import load_policy_contract, sha256_file
from wheelleg_mujoco.evaluation import FORMAL_SCENARIOS
from wheelleg_mujoco.runner import WheelLegMujocoRuntime


PROJECT = Path(__file__).resolve().parents[2]
SIM = PROJECT / "sim2sim/mujoco"
EXPORTS = {
    "debug701": PROJECT / "debug/no_action_clip/runs/fresh1000-split-straight-lambda01-env4096-20261016/export",
    "formal": PROJECT / "artifacts/phase2_dreamwaq/training-suite-20261011-stop-reverse-v1/exports/run-03",
}


def capture(output: Path, reference: Path | None = None) -> None:
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    for name, export in EXPORTS.items():
        if name == "debug701":
            if reference is None:
                runtime = load_runtime(export)
            else:
                # 兼容记录生成前使用已保存身份逐项验证，生产加载器仍执行原严格检查。
                identity = json.loads((reference / "identity.json").read_text(encoding="utf-8"))
                for filename, expected in identity[name]["files"].items():
                    if sha256_file(export / filename) != expected:
                        raise ValueError("修改前保存的导出身份发生变化")
                metadata = json.loads((export / "experiment_export.json").read_text(encoding="utf-8"))
                if sha256_file(Path(metadata["source_checkpoint"])) != metadata["source_checkpoint_sha256"]:
                    raise ValueError("修改前 checkpoint 身份发生变化")
                baseline = Path(metadata["baseline_export"]["directory"]) / "policy_manifest.json"
                if sha256_file(baseline) != metadata["baseline_export"]["manifest_sha256"]:
                    raise ValueError("修改前物理契约身份发生变化")
                model_path = SIM / "model_manifest.json"
                if sha256_file(model_path) != identity["model_manifest_sha256"]:
                    raise ValueError("物理模型 manifest 发生变化")
                model = json.loads(model_path.read_text(encoding="utf-8"))
                xml = SIM / "models" / model["model_xml"]["path"]
                if sha256_file(xml) != model["model_xml"]["sha256"]:
                    raise ValueError("物理模型 XML 发生变化")
                from wheelleg_mujoco.contract import AdapterContract
                runtime = RawActionRuntime(xml, AdapterContract.from_policy_manifest(
                    json.loads(baseline.read_text(encoding="utf-8"))), model)
        else:
            contract, _, model = load_policy_contract(
                export / "policy_manifest.json", export / "actor.ts", SIM / "model_manifest.json")
            runtime = WheelLegMujocoRuntime(SIM / "models" / model["model_xml"]["path"], contract, model)
        actor = torch.jit.load(str(export / "actor.ts"), map_location="cpu").eval()
        arrays = {}
        for scenario, values in FORMAL_SCENARIOS:
            command = np.asarray(values, dtype=np.float64)
            observation = runtime.reset(command)
            rows = [np.concatenate((observation, runtime.data.qpos, runtime.data.qvel))]
            steps = []
            for _ in range(32):
                with torch.inference_mode():
                    action = actor(torch.from_numpy(observation).view(1, -1)).numpy()[0]
                target = runtime.controller.prepare(action)
                result = runtime.step(action, command)
                steps.append(np.concatenate((action, target.leg_position_target_mujoco,
                                             target.wheel_velocity_target_mujoco,
                                             result.applied_torque)))
                observation = result.observation
                rows.append(np.concatenate((observation, runtime.data.qpos, runtime.data.qvel)))
            arrays[scenario + "_states"] = np.asarray(rows)
            arrays[scenario + "_controls"] = np.asarray(steps)
        np.savez_compressed(output / (name + ".npz"), **arrays)
        records[name] = {"export": str(export), "files": {
            p.name: sha256_file(p) for p in export.iterdir() if p.is_file()}}
    records["runtime_files"] = {str(p.relative_to(PROJECT)): sha256_file(p)
                                for p in (SIM / "wheelleg_mujoco").glob("*.py")}
    records["model_manifest_sha256"] = sha256_file(SIM / "model_manifest.json")
    (output / "identity.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def compare(before: Path, after: Path, report: Path) -> None:
    results = {}
    for name in EXPORTS:
        with np.load(before / (name + ".npz")) as old, np.load(after / (name + ".npz")) as new:
            if set(old.files) != set(new.files):
                raise ValueError("兼容测试数组名称不同")
            results[name] = {key: float(np.max(np.abs(old[key] - new[key]))) for key in old.files}
            if any(value != 0.0 for value in results[name].values()):
                raise ValueError(f"旧模型运行路径发生变化：{results[name]}")
    payload = {"schema_version": "BaselineRuntimeCompatibilityV1", "exact_equal": True,
               "before": str(before), "after": str(after), "max_abs_difference": results,
               "before_identity": json.loads((before / "identity.json").read_text(encoding="utf-8")),
               "after_identity": json.loads((after / "identity.json").read_text(encoding="utf-8"))}
    metadata = json.loads((EXPORTS["debug701"] / "experiment_export.json").read_text(encoding="utf-8"))
    model = json.loads((SIM / "model_manifest.json").read_text(encoding="utf-8"))
    payload["verified_checkpoint"] = {"path": metadata["source_checkpoint"],
                                      "sha256": metadata["source_checkpoint_sha256"]}
    payload["physics_identity"] = {"model_manifest_sha256": sha256_file(SIM / "model_manifest.json"),
                                   "xml_sha256": model["model_xml"]["sha256"],
                                   "dynamics_semantics_hash": model["dynamics_semantics_hash"]}
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        capture(args.output, args.reference)
    else:
        compare(args.before, args.after, args.report)
