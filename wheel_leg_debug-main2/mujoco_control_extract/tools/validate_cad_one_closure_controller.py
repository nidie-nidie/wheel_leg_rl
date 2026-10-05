#!/usr/bin/env python3
"""Validate the CAD 4.3/5.5 controller FK, analytic J_H and virtual work.

This is an assertion-based check for the exact model implemented by
Components/Algorithm/Src/VMC_Calc.c when
MUJOCO_VMC_KINEMATICS_MODE=CAD_ONE_CLOSURE.
"""

from __future__ import annotations

import json
import math

from derive_offset_kinematics import (
    analytic_closed_chain,
    cad_one_closure,
    cad_one_closure_with_jacobian,
)


def wrapped_difference(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def finite_difference_jh(q: tuple[float, float], step: float) -> list[list[float]]:
    result = [[0.0, 0.0], [0.0, 0.0]]
    for column in range(2):
        q_plus = list(q)
        q_minus = list(q)
        q_plus[column] += step
        q_minus[column] -= step
        plus = cad_one_closure(q_plus)
        minus = cad_one_closure(q_minus)
        result[0][column] = (plus["L0"] - minus["L0"]) / (2.0 * step)
        result[1][column] = wrapped_difference(
            plus["phi0"], minus["phi0"]
        ) / (2.0 * step)
    return result


def matrix_vector(matrix: list[list[float]], vector: tuple[float, float]) -> tuple[float, float]:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    )


def transpose_vector(matrix: list[list[float]], vector: tuple[float, float]) -> tuple[float, float]:
    return (
        matrix[0][0] * vector[0] + matrix[1][0] * vector[1],
        matrix[0][1] * vector[0] + matrix[1][1] * vector[1],
    )


def dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def main() -> None:
    samples: list[tuple[float, float]] = [(0.0, 0.0)]
    # Physical-like family: differential angle changes extension, common angle
    # tilts the leg.  It covers and exceeds the normal standing neighborhood.
    for extension_index in range(11):
        extension = 0.05 + 0.05 * extension_index
        for tilt_index in range(11):
            tilt = -0.25 + 0.05 * tilt_index
            samples.append((tilt - extension, tilt + extension))

    max_jh_fd_error = 0.0
    max_virtual_power_error = 0.0
    max_finite_virtual_work_error = 0.0
    max_cad_full_l0_error_m = 0.0
    max_cad_full_phi0_error_rad = 0.0
    min_abs_closure_determinant_m2 = math.inf
    valid_samples = 0

    for q in samples:
        try:
            state = cad_one_closure_with_jacobian(q)
            full = analytic_closed_chain(q)
        except (ValueError, RuntimeError):
            continue

        valid_samples += 1
        jh = state["JH"]
        jh_fd = finite_difference_jh(q, 1.0e-6)
        max_jh_fd_error = max(
            max_jh_fd_error,
            max(abs(jh[row][column] - jh_fd[row][column]) for row in range(2) for column in range(2)),
        )
        min_abs_closure_determinant_m2 = min(
            min_abs_closure_determinant_m2,
            abs(state["circle_M_determinant_m2"]),
        )

        q_dot = (0.73, -0.41)
        virtual_force = (40.0, 3.0)
        y_dot = matrix_vector(jh, q_dot)
        joint_torque = transpose_vector(jh, virtual_force)
        max_virtual_power_error = max(
            max_virtual_power_error,
            abs(dot(joint_torque, q_dot) - dot(virtual_force, y_dot)),
        )

        displacement_scale = 1.0e-7
        delta_q = (displacement_scale * q_dot[0], displacement_scale * q_dot[1])
        moved = cad_one_closure((q[0] + delta_q[0], q[1] + delta_q[1]))
        delta_y = (
            moved["L0"] - state["L0"],
            wrapped_difference(moved["phi0"], state["phi0"]),
        )
        max_finite_virtual_work_error = max(
            max_finite_virtual_work_error,
            abs(dot(joint_torque, delta_q) - dot(virtual_force, delta_y)),
        )

        max_cad_full_l0_error_m = max(
            max_cad_full_l0_error_m, abs(state["L0"] - full["L0"])
        )
        max_cad_full_phi0_error_rad = max(
            max_cad_full_phi0_error_rad,
            abs(wrapped_difference(state["phi0"], full["phi0"])),
        )

    reference = cad_one_closure_with_jacobian((0.0, 0.0))
    assert abs(reference["L0"] - 0.12049298105058033) < 2.0e-10
    assert abs(reference["phi0"] - 1.571050093872754) < 2.0e-10
    assert valid_samples >= 100
    assert max_jh_fd_error < 1.0e-6
    assert max_virtual_power_error < 1.0e-12
    assert max_finite_virtual_work_error < 1.0e-11
    assert min_abs_closure_determinant_m2 > 1.0e-5

    print(
        json.dumps(
            {
                "model": "CAD_ONE_CLOSURE",
                "valid_samples": valid_samples,
                "max_abs_analytic_minus_fd_jh": max_jh_fd_error,
                "max_abs_virtual_power_residual_w": max_virtual_power_error,
                "max_abs_finite_virtual_work_residual_j": max_finite_virtual_work_error,
                "min_abs_circle_m_determinant_m2": min_abs_closure_determinant_m2,
                "cad_vs_full_xml_max_l0_error_mm": 1.0e3 * max_cad_full_l0_error_m,
                "cad_vs_full_xml_max_phi0_error_deg": math.degrees(max_cad_full_phi0_error_rad),
                "reference_q_zero": {
                    "L0_m": reference["L0"],
                    "phi0_rad": reference["phi0"],
                    "JH": reference["JH"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
