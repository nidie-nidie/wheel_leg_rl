from __future__ import annotations

import pathlib
import sys
from typing import Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from pace_raw.normalize import NormalizedDataset

from .common import STAGE_DM_FIT, fit_and_validation_roles, require_dm_baseline, select_rows
from .model_manifest import replace_motor_result, validate_model_manifest


def pace_parameter_bounds(joint_count: int = 4) -> np.ndarray:
    if joint_count <= 0:
        raise ValueError("joint_count must be positive")
    bounds = np.zeros((joint_count * 4 + 1, 2), dtype=np.float32)
    bounds[0:joint_count, 0] = 1.0e-5
    bounds[0:joint_count, 1] = 1.0
    bounds[joint_count : 2 * joint_count, 1] = 7.0
    bounds[2 * joint_count : 3 * joint_count, 1] = 0.5
    bounds[3 * joint_count : 4 * joint_count, 0] = -0.1
    bounds[3 * joint_count : 4 * joint_count, 1] = 0.1
    bounds[4 * joint_count, 1] = 10.0
    return bounds


def parse_pace_parameter_vector(
    values: Sequence[float], joint_count: int = 4
) -> Tuple[list, int]:
    vector = np.asarray(values, dtype=np.float64)
    expected = joint_count * 4 + 1
    if vector.shape != (expected,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"PACE parameter vector must contain {expected} finite values")
    lower, upper = pace_parameter_bounds(joint_count).T
    if np.any(vector < lower) or np.any(vector > upper):
        raise ValueError("PACE parameter vector is outside the official bounds")
    delay_steps = int(vector[-1])
    parameters = []
    for index in range(joint_count):
        friction = float(vector[2 * joint_count + index])
        parameters.append(
            {
                "armature": float(vector[index]),
                "viscous_friction": float(vector[joint_count + index]),
                "static_friction": friction,
                "dynamic_friction": friction,
                "encoder_bias_rad": float(vector[3 * joint_count + index]),
                "friction_transition_velocity_rad_s": 0.05,
                "delay_steps": delay_steps,
            }
        )
    return parameters, delay_steps


def export_pace_dataset(dataset: NormalizedDataset, output: pathlib.Path) -> None:
    header = dataset.decoded.header
    if header is None:
        raise ValueError("PACE export requires a session header")
    fit_role, _ = fit_and_validation_roles(header.session_type)
    rows = select_rows(dataset.rows, STAGE_DM_FIT, fit_role)
    for index in range(4):
        require_dm_baseline(rows, index)
    time = np.asarray([row["time_s"] for row in rows], dtype=np.float32)
    dof_pos = np.column_stack(
        [np.asarray([row[f"m{index}_fb_q_rad"] for row in rows]) for index in range(4)]
    ).astype(np.float32)
    des_dof_pos = np.column_stack(
        [np.asarray([row[f"m{index}_cmd_q_rad"] for row in rows]) for index in range(4)]
    ).astype(np.float32)
    output = pathlib.Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".npz":
        np.savez_compressed(
            str(output), time=time, dof_pos=dof_pos, des_dof_pos=des_dof_pos
        )
        return
    if output.suffix.lower() != ".pt":
        raise ValueError("PACE dataset output must use .npz or .pt")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for the formal PACE .pt contract") from error
    torch.save(
        {
            "time": torch.from_numpy(time),
            "dof_pos": torch.from_numpy(dof_pos),
            "des_dof_pos": torch.from_numpy(des_dof_pos),
        },
        output,
    )


def load_reference_optimizer(reference_package_root: Optional[pathlib.Path] = None):
    if reference_package_root is None:
        workspace = pathlib.Path(__file__).resolve().parents[2]
        reference_package_root = (
            workspace
            / "a1_motor_id_flow_20260726_clean"
            / "pace-sim2real"
            / "source"
            / "pace_sim2real"
        )
    root = pathlib.Path(reference_package_root).resolve()
    if not (root / "pace_sim2real" / "optim" / "cma_es.py").is_file():
        raise FileNotFoundError("the generic PACE optimizer package was not found")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from pace_sim2real import CMAESOptimizer
    except ImportError as error:
        raise RuntimeError(
            "formal PACE optimization requires the Isaac environment, PyTorch, and cmaes"
        ) from error
    return CMAESOptimizer


def apply_pace_parameter_vector(
    model: MutableMapping[str, object],
    values: Sequence[float],
    fit_metadata: Mapping[str, object],
    validation_metrics: Sequence[Mapping[str, object]],
) -> MutableMapping[str, object]:
    validate_model_manifest(model, require_complete=False)
    parameters, delay_steps = parse_pace_parameter_vector(values, 4)
    if len(validation_metrics) != 4:
        raise ValueError("formal PACE import requires four validation metric sets")
    for index, (parameters_for_motor, metrics) in enumerate(
        zip(parameters, validation_metrics)
    ):
        actuator_model = {
            "type": "pace_dm_pd_v1",
            **parameters_for_motor,
            "controller": {
                "kp": 20.0,
                "kd": 0.6,
                "dq_des_rad_s": 0.0,
                "torque_feedforward_nm": 0.0,
            },
            "effort_limit_nm": 54.0,
            "parameter_semantics": "isaac_pace_cmaes",
        }
        replace_motor_result(
            model,
            index,
            actuator_model,
            dict(fit_metadata),
            dict(metrics),
        )
    model["fit_backend"]["leg_dm"] = {
        "name": "pace_sim2real.CMAESOptimizer",
        "parameter_layout": "armature[4], viscous[4], static_dynamic_friction[4], encoder_bias[4], global_delay[1]",
        "selected_global_delay_steps": delay_steps,
    }
    return model
