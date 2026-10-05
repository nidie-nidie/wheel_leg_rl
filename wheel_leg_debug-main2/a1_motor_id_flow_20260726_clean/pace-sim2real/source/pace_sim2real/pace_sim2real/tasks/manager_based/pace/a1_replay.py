"""Thin final-mean replay adapter around the official PACE actuator."""

from copy import deepcopy
import math
from typing import Any, Mapping

import torch
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG

from pace_sim2real.utils import PaceDCMotorCfg

from .a1_mean import A1_PACE_JOINT_ORDER, A1PaceMean


def make_a1_pace_actuator_cfg(mean: A1PaceMean) -> PaceDCMotorCfg:
    """Assemble the official actuator config from one parsed physical mean."""

    stiffness = {name: 25.0 for name in A1_PACE_JOINT_ORDER}
    damping = {name: 2.0 for name in A1_PACE_JOINT_ORDER}
    return PaceDCMotorCfg(
        joint_names_expr=list(A1_PACE_JOINT_ORDER),
        saturation_effort=33.5,
        effort_limit=33.5,
        velocity_limit=21.0,
        stiffness=stiffness,
        damping=damping,
        armature=mean.armature,
        encoder_bias=mean.encoder_bias,
        friction=mean.coulomb_friction,
        dynamic_friction=mean.coulomb_friction,
        viscous_friction=mean.viscous_friction,
        max_delay=mean.delay_steps,
    )


def inject_a1_actuator_before_make(env_cfg: Any, actuator_cfg: PaceDCMotorCfg) -> Any:
    """Inject a fresh official actuator config before the caller uses gym.make."""

    env_cfg.scene.robot.actuators = {"base_legs": actuator_cfg}
    return env_cfg


def inject_a1_stock_actuator_before_make(env_cfg: Any) -> Any:
    """Inject an independent copy of Isaac Lab's stock A1 leg actuator."""

    try:
        stock_actuator = UNITREE_A1_CFG.actuators["base_legs"]
    except KeyError as exc:
        raise ValueError("UNITREE_A1_CFG has no base_legs actuator") from exc
    if isinstance(stock_actuator, PaceDCMotorCfg):
        raise ValueError("UNITREE_A1_CFG base_legs actuator is not stock DCMotorCfg")
    env_cfg.scene.robot.actuators = {"base_legs": deepcopy(stock_actuator)}
    return env_cfg


def validate_a1_pace_data(data: dict[str, torch.Tensor]) -> None:
    """Validate the shared three-tensor A1 PACE data contract."""

    expected_keys = {"time", "dof_pos", "des_dof_pos"}
    if set(data) != expected_keys:
        raise KeyError(f"PACE data keys must be exactly {sorted(expected_keys)}")

    time = data["time"]
    dof_pos = data["dof_pos"]
    des_dof_pos = data["des_dof_pos"]
    if not all(
        isinstance(value, torch.Tensor) for value in (time, dof_pos, des_dof_pos)
    ):
        raise TypeError("PACE data values must be torch.Tensor objects")
    if time.ndim != 1 or time.numel() == 0:
        raise ValueError("PACE time must have non-empty shape (T,)")
    expected_shape = (time.shape[0], len(A1_PACE_JOINT_ORDER))
    if dof_pos.shape != expected_shape or des_dof_pos.shape != expected_shape:
        raise ValueError(f"PACE positions must both have shape {expected_shape}")
    if not all(
        torch.is_floating_point(value) for value in (time, dof_pos, des_dof_pos)
    ):
        raise TypeError("PACE data tensors must use floating dtypes")
    if time.dtype != torch.float64:
        raise ValueError("PACE time must use torch.float64")
    if dof_pos.dtype != torch.float32 or des_dof_pos.dtype != torch.float32:
        raise ValueError("PACE position tensors must use torch.float32")
    if not all(
        value.device.type == "cpu" and value.is_contiguous()
        for value in (time, dof_pos, des_dof_pos)
    ):
        raise ValueError("PACE data tensors must be contiguous CPU tensors")
    if not all(torch.isfinite(value).all() for value in (time, dof_pos, des_dof_pos)):
        raise ValueError("PACE data tensors must contain only finite values")
    if not torch.isclose(time[0], torch.zeros_like(time[0])):
        raise ValueError("PACE time must start at zero")
    if time.numel() > 1:
        expected_dt = torch.full_like(time[1:], 0.002)
        if not torch.allclose(torch.diff(time), expected_dt, atol=1.0e-7, rtol=0.0):
            raise ValueError("PACE A1 time steps must be exactly 0.002 s")


_validate_data = validate_a1_pace_data


def zero_a1_encoder_bias_reference() -> dict[str, float]:
    """Return the encoder-frame reference for the stock A1 actuator."""

    return {name: 0.0 for name in A1_PACE_JOINT_ORDER}


def _joint_ids(articulation: Any, device: str | torch.device) -> torch.Tensor:
    try:
        indices = [articulation.joint_names.index(name) for name in A1_PACE_JOINT_ORDER]
    except ValueError as exc:
        raise ValueError(
            "A1 articulation does not contain the complete PACE joint order"
        ) from exc
    return torch.tensor(indices, device=device, dtype=torch.long)


def _bias_tensor(
    encoder_bias_reference: Mapping[str, float],
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if set(encoder_bias_reference) != set(A1_PACE_JOINT_ORDER):
        raise ValueError("encoder bias reference must contain the exact A1 joint set")
    values = [encoder_bias_reference[name] for name in A1_PACE_JOINT_ORDER]
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("encoder bias reference must contain finite numbers")
    return torch.tensor(
        values,
        device=device,
        dtype=dtype,
    )


def to_encoder_frame(
    joint_position: torch.Tensor,
    encoder_bias_reference: Mapping[str, float],
) -> torch.Tensor:
    """Convert physical positions through one explicit encoder reference."""

    bias = _bias_tensor(
        encoder_bias_reference,
        device=joint_position.device,
        dtype=joint_position.dtype,
    )
    return joint_position - bias


def replay_absolute_targets(
    env: Any,
    data: dict[str, torch.Tensor],
    encoder_bias_reference: Mapping[str, float],
) -> torch.Tensor:
    """Replay absolute encoder targets with official fit.py sample ordering."""

    validate_a1_pace_data(data)
    unwrapped = env.unwrapped
    if unwrapped.num_envs != 1:
        raise ValueError("A1 final-mean replay requires exactly one environment")

    articulation = unwrapped.scene["robot"]
    joint_ids = _joint_ids(articulation, unwrapped.device)
    device = torch.device(unwrapped.device)
    dtype = articulation.data.joint_pos.dtype
    measured = data["dof_pos"].to(device=device, dtype=dtype)
    targets = data["des_dof_pos"].to(device=device, dtype=dtype)
    bias = _bias_tensor(encoder_bias_reference, device=device, dtype=dtype)

    env.reset()
    initial_position = (measured[0] + bias).unsqueeze(0)
    articulation.write_joint_position_to_sim(initial_position, joint_ids=joint_ids)
    articulation.write_joint_velocity_to_sim(
        torch.zeros_like(initial_position), joint_ids=joint_ids
    )

    samples = torch.empty_like(measured)
    for index in range(measured.shape[0]):
        samples[index] = articulation.data.joint_pos[0, joint_ids]
        actions = torch.zeros(env.action_space.shape, device=device, dtype=dtype)
        actions[:, joint_ids] = targets[index].unsqueeze(0)
        env.step(actions)

    return samples.detach().cpu()
