"""PhysicsV5 runtime identity; this module has no Isaac dependency."""

PHYSICS_SCHEMA_VERSION = "PhysicsV5"
UNRESTRICTED_SIM_VELOCITY = float.fromhex("0x1.fffffep+127")


def unrestricted_velocity_policy() -> dict:
    return {
        "version": "UnrestrictedVelocityPolicyV1",
        "rigid_body_linear_speed_limit": "disabled",
        "rigid_body_angular_speed_limit": "disabled",
        "joint_speed_limit": "disabled",
        "external_speed_limit_force": False,
        "external_speed_limit_torque": False,
        "runtime_velocity_write": False,
        "isaac_velocity_sentinel": UNRESTRICTED_SIM_VELOCITY,
    }


def validate_policy_physics(payload: dict) -> None:
    if payload.get("schemas", {}).get("physics") != PHYSICS_SCHEMA_VERSION:
        raise ValueError("Policy physics differs from the current PhysicsV5 runtime; old policies require historical runtime")
    if payload.get("velocity_limit_policy") != unrestricted_velocity_policy():
        raise ValueError("Policy velocity-limit policy differs from the unrestricted runtime")
    rigid = payload.get("rigid_body_properties", {})
    for field in ("max_linear_velocity", "max_angular_velocity"):
        if rigid.get(field) != UNRESTRICTED_SIM_VELOCITY:
            raise ValueError(f"Policy retains a rigid-body speed limit: {field}")
    for name in ("legs", "wheels", "passive"):
        if payload.get("actuators", {}).get(name, {}).get("velocity_limit_sim") != UNRESTRICTED_SIM_VELOCITY:
            raise ValueError(f"Policy retains a joint speed limit: {name}")
