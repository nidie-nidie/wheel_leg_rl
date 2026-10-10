from __future__ import annotations

from copy import deepcopy

import pytest

from wheelleg_mujoco.physics import (
    PHYSICS_SCHEMA_VERSION,
    UNRESTRICTED_SIM_VELOCITY,
    unrestricted_velocity_policy,
    validate_policy_physics,
)


def _payload() -> dict:
    return {
        "schemas": {"physics": PHYSICS_SCHEMA_VERSION},
        "velocity_limit_policy": unrestricted_velocity_policy(),
        "rigid_body_properties": {
            "max_linear_velocity": UNRESTRICTED_SIM_VELOCITY,
            "max_angular_velocity": UNRESTRICTED_SIM_VELOCITY,
        },
        "actuators": {
            name: {"velocity_limit_sim": UNRESTRICTED_SIM_VELOCITY}
            for name in ("legs", "wheels", "passive")
        },
    }


def test_accepts_current_unrestricted_policy() -> None:
    validate_policy_physics(_payload())


@pytest.mark.parametrize("mutation", ("old_schema", "missing", "torque", "linear", "angular", "active", "passive"))
def test_rejects_old_physics_or_policies_retaining_speed_limits(mutation: str) -> None:
    payload = deepcopy(_payload())
    if mutation == "old_schema":
        payload["schemas"]["physics"] = "PhysicsV4"
    elif mutation == "missing":
        del payload["velocity_limit_policy"]
    elif mutation == "torque":
        payload["velocity_limit_policy"]["external_speed_limit_torque"] = True
    elif mutation in ("linear", "angular"):
        payload["rigid_body_properties"][f"max_{mutation}_velocity"] = 100.0
    else:
        name = "legs" if mutation == "active" else "passive"
        payload["actuators"][name]["velocity_limit_sim"] = 45.0 if mutation == "active" else 80.0
    with pytest.raises(ValueError, match="physics|velocity-limit|speed limit"):
        validate_policy_physics(payload)
