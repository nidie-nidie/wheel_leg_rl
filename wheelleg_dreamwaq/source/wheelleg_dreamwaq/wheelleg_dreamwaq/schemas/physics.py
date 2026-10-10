from __future__ import annotations


PHYSICS_SCHEMA_VERSION = "PhysicsV5"
# Explicitly override inherited USD/PhysX caps without introducing non-finite
# configuration or a replacement operational speed limit.
UNRESTRICTED_SIM_VELOCITY = float.fromhex("0x1.fffffep+127")
SIM_DT_S = 0.005
DECIMATION = 4
CONTROL_DT_S = 0.02
SOLVER_POSITION_ITERATIONS = 96
SOLVER_VELOCITY_ITERATIONS = 4


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


def validate_phase1_physics(
    *,
    sim_dt: float,
    decimation: int,
    solver_position_iterations: int,
    solver_velocity_iterations: int,
) -> None:
    actual = (
        float(sim_dt),
        int(decimation),
        int(solver_position_iterations),
        int(solver_velocity_iterations),
    )
    expected = (
        SIM_DT_S,
        DECIMATION,
        SOLVER_POSITION_ITERATIONS,
        SOLVER_VELOCITY_ITERATIONS,
    )
    if actual != expected:
        raise ValueError(
            "PhysicsV5 physics mismatch: "
            f"actual={actual}, expected={expected}"
        )
    if abs(float(sim_dt) * int(decimation) - CONTROL_DT_S) > 1.0e-12:
        raise ValueError("PhysicsV5 control period must be exactly 0.02 s")
