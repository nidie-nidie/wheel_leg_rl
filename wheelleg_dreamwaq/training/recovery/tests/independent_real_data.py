from pathlib import Path

import numpy as np
import torch

from recovery.standing_reference import StandingReference


REFERENCE_DIR = Path(__file__).resolve().parents[3] / "recovery/artifacts/standing-reference401-20261029"
REFERENCE_SHA256 = "E093A55A98F91B6DC0D35DDBDE68905DB78B82E273807766F404083A5DDFA5DE"


def real_reference():
    return StandingReference.load(REFERENCE_DIR / "manifest.json",
        REFERENCE_DIR / "recovery_contact_paired_reference.npz", REFERENCE_SHA256)


def real_standing_state(reference, *, count=1, index=100):
    data = reference.arrays
    indices = (np.arange(count) + index) % len(reference.q)
    tensor = lambda name: torch.as_tensor(data[name][indices], dtype=torch.float32)
    return {"q": tensor("q_control_rad")[:, :4], "dq": tensor("dq_control_rad_s"),
            "omega": tensor("angular_velocity_control_rad_s"), "gravity": tensor("projected_gravity"),
            "linear": tensor("linear_velocity_control_m_s"), "height": tensor("height_m"),
            "wheel_forces": tensor("wheel_normal_force_n"),
            "nonwheel_force": tensor("nonwheel_normal_force_n"),
            "world_xy": tensor("base_com_world_m")[:, :2],
            "physical_failure": torch.zeros(count, dtype=torch.bool)}


def real_reward_state(reference, count=1, index=100):
    state = real_standing_state(reference, count=count, index=index)
    indices = (np.arange(count) + index) % len(reference.q)
    data = reference.arrays
    action = torch.cat((torch.tensor(data["leg_target_rad"][indices], dtype=torch.float32),
                        torch.tensor(data["wheel_target_control_rad_s"][indices], dtype=torch.float32) / 30.), dim=1)
    state.pop("world_xy")
    state["wheel_forces_mean"] = state.pop("wheel_forces")
    state["nonwheel_force_mean"] = state.pop("nonwheel_force")
    state.update(torque=torch.tensor(data["motor_generalized_torque_nm"][indices], dtype=torch.float32),
                 actions=action, previous_actions=action.clone(), previous_previous_actions=action.clone(),
                 reference=reference, weight_n=reference.identity["weight_n"],
                 success=torch.zeros(count, dtype=torch.bool), timeout=torch.zeros(count, dtype=torch.bool))
    return state
