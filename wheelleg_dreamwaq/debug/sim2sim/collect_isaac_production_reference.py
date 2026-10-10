from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import tomllib
import uuid
from pathlib import Path
from typing import Any

import h5py  # noqa: F401
import numpy as np
import tensordict  # noqa: F401
import torch

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTOR_RELATIVE = Path("artifacts/phase1_v4/formal-4x1000-20261005/exports/run-01/actor.ts")
POLICY_MANIFEST_RELATIVE = Path(
    "artifacts/phase1_v4/formal-4x1000-20261005/exports/run-01/policy_manifest.json"
)
MODEL_MANIFEST_RELATIVE = Path("sim2sim/mujoco/model_manifest.json")

from isaaclab.app import AppLauncher

from wheelleg_dreamwaq.training.runtime import validate_runtime


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array).copy()


def _rows_to_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows[1:]):
        raise ValueError("Production-reference rows do not share one schema")
    return {
        key: np.stack([_to_numpy(row[key]) for row in rows])
        if _to_numpy(rows[0][key]).ndim
        else np.asarray([_to_numpy(row[key]) for row in rows])
        for key in keys
    }


def _distribution_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect an unmodified Isaac production-step reference.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-ticks", type=int, default=10)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 1 <= args.control_ticks <= 10:
        raise ValueError("The production-equivalence reference is limited to 1..10 control ticks")

    root = args.project_root.resolve()
    validate_runtime(root, device=args.device)
    launcher = AppLauncher(args)
    simulation_app = launcher.app

    from isaaclab.utils import math as math_utils

    from debug.sim2sim.isaac_debug_env import make_debug_env_cfg
    from debug.sim2sim.stand_scenario import StandScenarioV1, verify_frozen_files
    from debug.sim2sim.trace_schema import (
        SCHEMA_VERSION,
        build_file_hashes,
        closed_loop_action_identity,
        load_json,
        save_npz,
        sha256_file,
        write_json,
    )
    from wheelleg_dreamwaq.schemas.frames import R_CONTROL_FROM_USD
    from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    actor_path = root / ACTOR_RELATIVE
    policy_manifest_path = root / POLICY_MANIFEST_RELATIVE
    model_manifest_path = root / MODEL_MANIFEST_RELATIVE
    policy_manifest = load_json(policy_manifest_path)
    model_manifest = load_json(model_manifest_path)
    dependency_manifest = tomllib.loads((root / "dependency-manifest.toml").read_text(encoding="utf-8"))
    all_hinge_names = list(model_manifest["joint_order"])
    scenario = StandScenarioV1()
    frozen_files = verify_frozen_files(root)

    env = WheelLegFlatEnv(make_debug_env_cfg(device=args.device, enable_contact_sensors=False))
    joint_ids, resolved = env.robot.find_joints(all_hinge_names, preserve_order=True)
    if resolved != all_hinge_names or len(set(joint_ids)) != 26:
        raise RuntimeError(f"Production reference failed to resolve the 26 hinge order: {resolved}")
    policy = torch.jit.load(str(actor_path), map_location=env.device).eval()
    observations, _ = env.reset(seed=scenario.random_seed)
    observation = observations["policy"]
    initial_com = env._current_state().root_com_pos_w.detach().clone()
    rows: list[dict[str, Any]] = []

    try:
        for tick in range(args.control_ticks):
            previous_action = env._canonical_action.detach().clone()
            with torch.inference_mode():
                raw_action = policy(observation)
            next_observations, _, terminated, truncated, _ = env.step(raw_action)
            if bool(terminated[0].item()) or bool(truncated[0].item()):
                raise RuntimeError("Production equivalence run terminated inside the 10-tick gate")
            state = env._current_state()
            loop_each = state.loop_closure_position_error
            if loop_each.shape[1] != 4:
                raise RuntimeError(f"Expected four production loop residuals, got {loop_each.shape[1]}")
            loop_by_side = torch.stack(
                (loop_each[:, :2].amax(dim=-1), loop_each[:, 2:].amax(dim=-1)), dim=-1
            )
            rotation_world_from_body = math_utils.matrix_from_quat(state.root_link_quat_w)
            rotation = R_CONTROL_FROM_USD.to(device=env.device, dtype=state.root_link_quat_w.dtype)
            orientation = math_utils.quat_from_matrix(
                rotation.unsqueeze(0) @ rotation_world_from_body @ rotation.T.unsqueeze(0)
            )
            orientation = torch.where(orientation[:, :1] < 0.0, -orientation, orientation)
            position_diag = torch.empty_like(state.root_com_pos_w)
            position_diag[:, :2] = (
                state.root_com_pos_w[:, :2] - initial_com[:, :2]
            ) @ rotation[:2, :2].T
            position_diag[:, 2:3] = state.base_height
            row = {
                    "control_tick": np.int64(tick),
                    "actor_obs_policy_pre_step": observation,
                    "actor_output_raw": raw_action,
                    "action_clipped": env._canonical_action,
                    "active_joint_position_canonical_post_step_pre_reset": state.joint_position,
                    "active_joint_velocity_canonical_post_step_pre_reset": state.joint_velocity,
                    "all_hinge_position_named_post_step_pre_reset": env.robot.data.joint_pos[:, joint_ids],
                    "all_hinge_velocity_named_post_step_pre_reset": env.robot.data.joint_vel[:, joint_ids],
                    "base_com_position_engine_world_post_step_pre_reset": state.root_com_pos_w,
                    "base_com_position_diag_post_step_pre_reset": position_diag,
                    "base_orientation_control_wxyz_post_step_pre_reset": orientation,
                    "base_linear_velocity_control_post_step_pre_reset": state.root_com_linear_velocity,
                    "base_angular_velocity_control_post_step_pre_reset": state.root_angular_velocity,
                    "projected_gravity_post_step_pre_reset": state.projected_gravity,
                    "base_height_post_step_pre_reset": state.base_height[:, 0],
                    "virtual_leg_length_post_step_pre_reset": state.virtual_leg_length_true,
                    "virtual_leg_phi0_post_step_pre_reset": state.virtual_leg_phi0_true,
                    "loop_closure_error_post_step_pre_reset": loop_by_side,
                    "native_terminated_int8": terminated.to(torch.int8),
                    "native_truncated_int8": truncated.to(torch.int8),
                    "next_actor_obs_policy_returned": next_observations["policy"],
                    "previous_action_before_inference": previous_action,
                }
            rows.append({name: _to_numpy(value) for name, value in row.items()})
            observation = next_observations["policy"]

        trace = _rows_to_arrays(rows)
        save_npz(output / "reference_trace.npz", trace)
        articulation_props = env.cfg.robot_cfg.spawn.articulation_props
        formal_task_root = root / "source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat"
        schema_root = root / "source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas"
        formal_sources = {
            "formal_environment": formal_task_root / "env.py",
            "formal_environment_config": formal_task_root / "env_cfg.py",
            "formal_observation_adapter": formal_task_root / "observations.py",
            "formal_action_adapter": formal_task_root / "control.py",
            "formal_action_schema": schema_root / "action.py",
            "formal_frame_schema": schema_root / "frames.py",
            "formal_normalization_schema": schema_root / "normalization.py",
        }
        metadata = {
            "schema_version": "IsaacProductionEquivalenceReferenceV1",
            "collection_id": uuid.uuid4().hex,
            "trace_schema_version": SCHEMA_VERSION,
            "engine": "isaac_sim",
            "engine_version": _distribution_version("isaacsim"),
            "isaac_lab_version": dependency_manifest["isaac_lab"]["tag"].removeprefix("v"),
            "isaac_lab_extension_version": _distribution_version("isaaclab"),
            "isaac_lab_commit": dependency_manifest["isaac_lab"]["commit"],
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "scenario_hash": scenario.payload_hash,
            "frozen_files": frozen_files,
            "scenario_variant": "closed_loop",
            "action_sequence_identity": closed_loop_action_identity(),
            "command": list(scenario.command),
            "random_seed": scenario.random_seed,
            "actor_sha256": sha256_file(actor_path),
            "policy_manifest_sha256": sha256_file(policy_manifest_path),
            "model_manifest_sha256": sha256_file(model_manifest_path),
            "model_xml_sha256": model_manifest["model_xml"]["sha256"],
            "requested_control_ticks": args.control_ticks,
            "completed_control_ticks": len(rows),
            "control_dt_s": env.step_dt,
            "physics_dt_s": env.physics_dt,
            "physics_steps_per_action": env.cfg.decimation,
            "inference_device": str(env.device),
            "inference_dtype": "float32",
            "simulation_device": str(env.cfg.sim.device),
            "physics_pipeline": "gpu" if str(env.cfg.sim.device).startswith("cuda") else "cpu",
            "canonical_joint_order": list(policy_manifest["action"]["canonical_joint_order"]),
            "canonical_from_engine_native": [1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            "all_hinge_order": all_hinge_names,
            "r_diag_from_engine_world": policy_manifest["frames"]["r_control_from_usd"],
            "solver_position_iterations": articulation_props.solver_position_iteration_count,
            "solver_velocity_iterations": articulation_props.solver_velocity_iteration_count,
            "clone_in_fabric": env.cfg.scene.clone_in_fabric,
            "use_fabric": env.cfg.sim.use_fabric,
            "render_mode": "headless" if args.headless else "interactive",
            "contact_observation_mode": "disabled",
            "contact_processing_state": "baseline_without_contact_report_observer",
            "formal_source_sha256": build_file_hashes(formal_sources),
            "environment_class": f"{WheelLegFlatEnv.__module__}.{WheelLegFlatEnv.__name__}",
            "observer_mode": "production_unmodified_step",
            "step_override": False,
        }
        write_json(output / "metadata.json", metadata)
        source_paths = {
            "collector": Path(__file__),
            "debug_environment": Path(__file__).with_name("isaac_debug_env.py"),
            "trace_schema": Path(__file__).with_name("trace_schema.py"),
            "stand_scenario": Path(__file__).with_name("stand_scenario.py"),
            "equivalence_comparison": Path(__file__).with_name("compare_isaac_equivalence.py"),
            **formal_sources,
            "metadata": output / "metadata.json",
            "reference_trace": output / "reference_trace.npz",
        }
        write_json(output / "file_hashes.json", build_file_hashes(source_paths))
        print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
