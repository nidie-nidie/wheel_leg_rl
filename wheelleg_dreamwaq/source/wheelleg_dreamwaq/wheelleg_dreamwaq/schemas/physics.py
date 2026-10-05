from __future__ import annotations


PHYSICS_SCHEMA_VERSION = "PhysicsV4"
SIM_DT_S = 0.005
DECIMATION = 4
CONTROL_DT_S = 0.02
SOLVER_POSITION_ITERATIONS = 96
SOLVER_VELOCITY_ITERATIONS = 4


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
            "Phase1ContractV4 physics mismatch: "
            f"actual={actual}, expected={expected}"
        )
    if abs(float(sim_dt) * int(decimation) - CONTROL_DT_S) > 1.0e-12:
        raise ValueError("Phase1ContractV4 control period must be exactly 0.02 s")
