from __future__ import annotations

import pytest

from wheelleg_dreamwaq.schemas.physics import (
    CONTROL_DT_S,
    DECIMATION,
    SIM_DT_S,
    SOLVER_POSITION_ITERATIONS,
    SOLVER_VELOCITY_ITERATIONS,
    validate_phase1_physics,
)


def test_phase1_v4_physics_constants_are_frozen() -> None:
    assert SIM_DT_S == 0.005
    assert DECIMATION == 4
    assert CONTROL_DT_S == 0.02
    assert SOLVER_POSITION_ITERATIONS == 96
    assert SOLVER_VELOCITY_ITERATIONS == 4
    validate_phase1_physics(
        sim_dt=SIM_DT_S,
        decimation=DECIMATION,
        solver_position_iterations=SOLVER_POSITION_ITERATIONS,
        solver_velocity_iterations=SOLVER_VELOCITY_ITERATIONS,
    )


@pytest.mark.parametrize(
    ("sim_dt", "decimation", "position", "velocity"),
    [
        (0.01, 2, 96, 4),
        (0.005, 4, 32, 4),
        (0.005, 4, 96, 2),
    ],
)
def test_phase1_v4_rejects_historical_or_partial_physics(
    sim_dt: float,
    decimation: int,
    position: int,
    velocity: int,
) -> None:
    with pytest.raises(ValueError, match="physics mismatch"):
        validate_phase1_physics(
            sim_dt=sim_dt,
            decimation=decimation,
            solver_position_iterations=position,
            solver_velocity_iterations=velocity,
        )
