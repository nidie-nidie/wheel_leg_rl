from __future__ import annotations

import torch

from wheelleg_dreamwaq.kinematics.virtual_leg import (
    MAX_FK_LENGTH_ERROR_M, MAX_FK_PHI0_ERROR_RAD, MAX_FK_WHEEL_ERROR_M,
    MAX_LOOP_CLOSURE_ERROR_M, MIN_VIRTUAL_LEG_LENGTH_M, wrap_to_pi,
)


def verify_actual_geometry(virtual_leg, vector_body):
    for name, values in (("vector_body", vector_body),
                         ("length_true", virtual_leg["length_true"]),
                         ("phi0_true", virtual_leg["phi0_true"]),
                         ("loop_error", virtual_leg["loop_error"])):
        if not torch.isfinite(values).all():
            raise RuntimeError(f"实际腿部几何存在非有限数值：{name}")
    if (virtual_leg["length_true"] <= MIN_VIRTUAL_LEG_LENGTH_M).any():
        raise RuntimeError("实际虚拟腿长度未超过现有最小值 0.05m")
    if (virtual_leg["loop_error"] > MAX_LOOP_CLOSURE_ERROR_M).any():
        maximum = float(virtual_leg["loop_error"].max())
        raise RuntimeError(f"实际闭环锚点间隙超过现有 5mm：{maximum:.9g}m")


def fk_diagnostic_metrics(virtual_leg):
    length_error = (virtual_leg["length_fk"] - virtual_leg["length_true"]).abs()
    phi0_error = wrap_to_pi(virtual_leg["phi0_fk"] - virtual_leg["phi0_true"]).abs()
    invalid = ~virtual_leg["fk_valid"]
    disagreement = (invalid | (virtual_leg["wheel_error"] > MAX_FK_WHEEL_ERROR_M)
                    | (length_error > MAX_FK_LENGTH_ERROR_M)
                    | (phi0_error > MAX_FK_PHI0_ERROR_RAD))
    return {
        "Debug/fk_disagreement_fraction": disagreement.float().mean(),
        "Debug/fk_invalid_fraction": invalid.float().mean(),
        "Debug/fk_length_error_max_m": length_error.max(),
    }
