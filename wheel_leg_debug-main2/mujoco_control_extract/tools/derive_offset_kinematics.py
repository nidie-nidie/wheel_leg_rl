#!/usr/bin/env python3
"""Derive and validate the nominal left-leg kinematics from the MuJoCo XML.

This is an offline analysis tool.  It does not import MuJoCo and it does not
modify the controller.  The script evaluates the MJCF body/joint transforms,
enforces the two exported closed-loop connections, and compares the resulting
wheel-axis map with the ideal five-bar model used by VMC_Calc.c.

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]


def v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def v_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_norm(a: Vec3) -> float:
    return math.sqrt(v_dot(a, a))


def m_identity() -> Mat3:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def m_mul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


def m_vec(a: Mat3, v: Vec3) -> Vec3:
    return (
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    )


def quat_to_matrix(q: Sequence[float]) -> Mat3:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return m_identity()
    w, x, y, z = w / n, x / n, y / n, z / n
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def axis_angle(axis: Vec3, angle: float) -> Mat3:
    n = v_norm(axis)
    if n == 0.0:
        return m_identity()
    x, y, z = v_scale(axis, 1.0 / n)
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return (
        (t * x * x + c, t * x * y - s * z, t * x * z + s * y),
        (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
        (t * x * z - s * y, t * y * z + s * x, t * z * z + c),
    )


def parse_vec(text: str | None, default: Vec3 = (0.0, 0.0, 0.0)) -> Vec3:
    if not text:
        return default
    values = tuple(float(x) for x in text.split())
    if len(values) != 3:
        raise ValueError(f"Expected vec3, got {text!r}")
    return values  # type: ignore[return-value]


def parse_quat(text: str | None) -> tuple[float, float, float, float]:
    if not text:
        return (1.0, 0.0, 0.0, 0.0)
    values = tuple(float(x) for x in text.split())
    if len(values) != 4:
        raise ValueError(f"Expected quaternion, got {text!r}")
    return values  # type: ignore[return-value]


@dataclass
class Site:
    name: str
    pos: Vec3


@dataclass
class Body:
    name: str
    pos: Vec3
    quat: tuple[float, float, float, float]
    joint_name: str | None = None
    joint_axis: Vec3 = (0.0, 0.0, 1.0)
    sites: list[Site] = field(default_factory=list)
    children: list["Body"] = field(default_factory=list)


@dataclass
class Pose:
    p: Vec3
    r: Mat3


@dataclass
class EvalResult:
    bodies: dict[str, Pose]
    sites: dict[str, Vec3]


class MjcfTree:
    def __init__(self, xml_path: Path):
        root = ET.parse(xml_path).getroot()
        base_elem = root.find("./worldbody/body[@name='base']")
        if base_elem is None:
            raise RuntimeError("Could not find worldbody/base in XML")
        self.base = self._parse_body(base_elem)

    def _parse_body(self, elem: ET.Element) -> Body:
        joint = elem.find("joint")
        body = Body(
            name=elem.get("name", ""),
            pos=parse_vec(elem.get("pos")),
            quat=parse_quat(elem.get("quat")),
            joint_name=joint.get("name") if joint is not None else None,
            joint_axis=parse_vec(joint.get("axis"), (0.0, 0.0, 1.0)) if joint is not None else (0.0, 0.0, 1.0),
        )
        body.sites = [Site(s.get("name", ""), parse_vec(s.get("pos"))) for s in elem.findall("site")]
        body.children = [self._parse_body(child) for child in elem.findall("body")]
        return body

    def evaluate(self, q: dict[str, float]) -> EvalResult:
        bodies: dict[str, Pose] = {"base": Pose((0.0, 0.0, 0.0), m_identity())}
        sites: dict[str, Vec3] = {}

        def walk(body: Body, parent_pose: Pose, skip_base: bool = False) -> None:
            if skip_base:
                pose = parent_pose
            else:
                p = v_add(parent_pose.p, m_vec(parent_pose.r, body.pos))
                r0 = quat_to_matrix(body.quat)
                rj = axis_angle(body.joint_axis, q.get(body.joint_name or "", 0.0))
                # MJCF: body reference transform, followed by the hinge motion in
                # the body's local frame.  All exported hinges are at body origin.
                pose = Pose(p, m_mul(m_mul(parent_pose.r, r0), rj))
                bodies[body.name] = pose
                for site in body.sites:
                    sites[site.name] = v_add(pose.p, m_vec(pose.r, site.pos))
            for child in body.children:
                walk(child, pose)

        walk(self.base, bodies["base"], skip_base=True)
        return EvalResult(bodies, sites)


ACTIVE = ("jIJ", "jIO")
PASSIVE = ("jJM", "jMK", "jKN", "jOP")
CONNECTS = (
    ("site_mk_io_a1", "site_mk_io_b1"),
    ("site_mk_io_a2", "site_mk_io_b2"),
    ("site_kn_op_a1", "site_kn_op_b1"),
    ("site_kn_op_a2", "site_kn_op_b2"),
)


def closure_residual(tree: MjcfTree, active: Sequence[float], passive: Sequence[float]) -> list[float]:
    q = dict(zip(ACTIVE, active))
    q.update(zip(PASSIVE, passive))
    ev = tree.evaluate(q)
    residual: list[float] = []
    for a, b in CONNECTS:
        residual.extend(v_sub(ev.sites[a], ev.sites[b]))
    return residual


def solve_linear(a: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    n = len(b)
    aug = [list(a[i]) + [float(b[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-14:
            raise RuntimeError("Singular normal matrix")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [x / scale for x in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [aug[row][j] - factor * aug[col][j] for j in range(n + 1)]
    return [aug[i][-1] for i in range(n)]


def least_squares_step(j: Sequence[Sequence[float]], residual: Sequence[float], damping: float = 1e-12) -> list[float]:
    rows, cols = len(j), len(j[0])
    normal = [[sum(j[k][i] * j[k][m] for k in range(rows)) for m in range(cols)] for i in range(cols)]
    rhs = [-sum(j[k][i] * residual[k] for k in range(rows)) for i in range(cols)]
    for i in range(cols):
        normal[i][i] += damping
    return solve_linear(normal, rhs)


def numeric_jacobian(function, x: Sequence[float], step: float = 1e-6) -> list[list[float]]:
    f0 = function(x)
    jac = [[0.0 for _ in x] for _ in f0]
    for col in range(len(x)):
        xp, xm = list(x), list(x)
        xp[col] += step
        xm[col] -= step
        fp, fm = function(xp), function(xm)
        for row in range(len(f0)):
            jac[row][col] = (fp[row] - fm[row]) / (2.0 * step)
    return jac


def solve_passive(tree: MjcfTree, active: Sequence[float], seed: Sequence[float]) -> tuple[list[float], float, int]:
    x = list(seed)
    for iteration in range(60):
        residual = closure_residual(tree, active, x)
        rms = math.sqrt(sum(r * r for r in residual) / len(residual))
        if rms < 1e-10:
            return x, rms, iteration
        jac = numeric_jacobian(lambda p: closure_residual(tree, active, p), x)
        step = least_squares_step(jac, residual, damping=1e-11)
        step_norm = math.sqrt(sum(s * s for s in step))
        if step_norm > 0.35:
            step = [s * (0.35 / step_norm) for s in step]
        # A short line search keeps the solve on the current assembly branch.
        old_cost = sum(r * r for r in residual)
        accepted = False
        scale = 1.0
        for _ in range(12):
            candidate = [x[i] + scale * step[i] for i in range(len(x))]
            cost = sum(r * r for r in closure_residual(tree, active, candidate))
            if cost < old_cost:
                x = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
    residual = closure_residual(tree, active, x)
    rms = math.sqrt(sum(r * r for r in residual) / len(residual))
    return x, rms, 60


def wheel_output(tree: MjcfTree, active: Sequence[float], passive: Sequence[float]) -> tuple[float, float, float, float]:
    q = dict(zip(ACTIVE, active))
    q.update(zip(PASSIVE, passive))
    ev = tree.evaluate(q)
    hip = ev.bodies["jIO"].p
    wheel = ev.bodies["jwheel_left"].p
    delta = v_sub(wheel, hip)
    # The jIJ/jIO body reference quaternion rotates the exported local hinge
    # axis onto base +X.  Therefore the left leg moves in the base Y-Z plane;
    # the non-zero base-X component is only the lateral wheel/hip offset and
    # must be projected out of the virtual-leg length.
    # Signed wheel coordinates in the leg side-view plane.  ``s_w`` is the
    # projection along +base Y; ``d_w`` is the projection toward chassis down.
    # These are position components, not a vehicle-forward command or wheel
    # spin angle.
    s_w = delta[1]
    d_w = -delta[2]
    length = math.hypot(s_w, d_w)
    theta_leg = math.atan2(s_w, d_w)
    return length, theta_leg, s_w, d_w


def exact_map(tree: MjcfTree, active: Sequence[float], seed: Sequence[float]) -> tuple[list[float], list[float], float]:
    """Return controller virtual coordinates y=[L0, phi0].

    theta_leg is measured from chassis down toward base +Y.  The controller's
    phi0 is measured from the planar +x direction toward chassis down, hence
    phi0 = pi/2 - theta_leg.  Keeping this conversion here makes the Jacobian
    returned below directly conjugate to the controller's [F0, Tp] variables.
    """
    passive, closure_rms, _ = solve_passive(tree, active, seed)
    length, theta_leg, _, _ = wheel_output(tree, active, passive)
    phi0 = math.pi * 0.5 - theta_leg
    return [length, phi0], passive, closure_rms


def tree_jacobian_center_difference(
    tree: MjcfTree,
    active: Sequence[float],
    seed: Sequence[float],
) -> tuple[list[list[float]], list[float], float]:
    """Offline validation Jacobian from five full XML-tree solves.

    The production Jacobian is the closed-form implicit derivative implemented
    by :func:`analytic_closed_chain_with_jacobian`.  This finite difference is
    intentionally retained only as an independent numerical cross-check.
    """
    center, center_passive, rms = exact_map(tree, active, seed)
    step = 2e-5
    jac = [[0.0, 0.0], [0.0, 0.0]]
    for col in range(2):
        ap, am = list(active), list(active)
        ap[col] += step
        am[col] -= step
        pp, _, _ = exact_map(tree, ap, center_passive)
        pm, _, _ = exact_map(tree, am, center_passive)
        jac[0][col] = (pp[0] - pm[0]) / (2.0 * step)
        # Unwrap controller phi0 locally.  This row is d(phi0)/dq, not
        # d(theta_leg)/dq; the latter has the opposite sign.
        dphi0 = math.atan2(math.sin(pp[1] - pm[1]), math.cos(pp[1] - pm[1]))
        jac[1][col] = dphi0 / (2.0 * step)
    return jac, center_passive, rms


L1 = 0.215
L2 = 0.258
J0_OFFSET = -0.10301526
J1_OFFSET = 0.10301526 + math.pi

# Exact nominal planar vectors extracted from the left-leg MJCF.  Coordinates
# are in the common jIJ/jIO local X-Z plane before those two hip bodies' own
# +90 deg reference rotation about base Z (this is not the root/base pose).
V_IJ = (0.0967494, 0.00674587)
V_IL = (-0.0963553, 0.0109404)
V_IP = (-0.213571, 0.0245218)
V_JM = (-0.0979082, -0.060324)
V_ML = (-0.0951965, 0.0645185)
V_MK = (-0.132583, 0.116217)
V_KN = (-0.117216, 0.0135814)
V_PN = (-0.0373868, 0.0516985)
V_PW = (0.213571, -0.144746)


def v2_add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    return (a[0] + b[0], a[1] + b[1])


def v2_sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def v2_norm(a: Sequence[float]) -> float:
    return math.hypot(a[0], a[1])


def rotate2(angle: float, vector: Sequence[float]) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return (c * vector[0] - s * vector[1], s * vector[0] + c * vector[1])


def v2_scale(scale: float, vector: Sequence[float]) -> tuple[float, float]:
    return (scale * vector[0], scale * vector[1])


def similarity_parameters(
    source: Sequence[float], target: Sequence[float]
) -> tuple[float, float]:
    """Return the fixed scale and signed rotation mapping source to target."""

    source_norm = v2_norm(source)
    target_norm = v2_norm(target)
    if source_norm < 1e-16 or target_norm < 1e-16:
        raise RuntimeError("Degenerate vector in planar similarity transform")
    dot = source[0] * target[0] + source[1] * target[1]
    cross = source[0] * target[1] - source[1] * target[0]
    return target_norm / source_norm, math.atan2(cross, dot)


# The nominal L-K-N-P loop is a parallelogram.  It keeps the output rigid
# body P-N-W at the same orientation as the compound body M-L-K.  Therefore
# the local vector L->M maps to P->W through one fixed planar similarity.
V_LM = (-V_ML[0], -V_ML[1])
SIMPLIFIED_SCALE_B, SIMPLIFIED_ANGLE_B = similarity_parameters(V_LM, V_PW)
CAD_SCALE_A = v2_norm(V_IP) / v2_norm(V_IL)
CAD_SCALE_B = v2_norm(V_PW) / v2_norm(V_ML)


def apply_similarity(
    vector: Sequence[float], scale: float, angle: float
) -> tuple[float, float]:
    return v2_scale(scale, rotate2(angle, vector))


def circle_intersection(
    center_a: Sequence[float],
    radius_a: float,
    center_b: Sequence[float],
    radius_b: float,
    branch: float,
) -> tuple[float, float]:
    delta = v2_sub(center_b, center_a)
    distance = v2_norm(delta)
    if distance < 1e-12:
        raise RuntimeError("Coincident circle centers")
    along = (radius_a * radius_a - radius_b * radius_b + distance * distance) / (2.0 * distance)
    height_sq = radius_a * radius_a - along * along
    if height_sq < -1e-10:
        raise RuntimeError("No real closed-chain circle intersection")
    height = math.sqrt(max(0.0, height_sq))
    ex, ez = delta[0] / distance, delta[1] / distance
    return (
        center_a[0] + along * ex + branch * height * (-ez),
        center_a[1] + along * ez + branch * height * ex,
    )


def analytic_closed_chain(active: Sequence[float]) -> dict:
    """Exact two-circle-intersection reduction of the nominal XML geometry."""
    q_front, q_rear = active
    point_j = rotate2(q_front, V_IJ)
    point_l = rotate2(q_rear, V_IL)
    point_p = rotate2(q_rear, V_IP)
    point_m = circle_intersection(point_j, v2_norm(V_JM), point_l, v2_norm(V_ML), +1.0)
    alpha = math.atan2(point_l[1] - point_m[1], point_l[0] - point_m[0]) - math.atan2(V_ML[1], V_ML[0])
    point_k = v2_add(point_m, rotate2(alpha, V_MK))
    point_n = circle_intersection(point_k, v2_norm(V_KN), point_p, v2_norm(V_PN), -1.0)
    beta = math.atan2(point_n[1] - point_p[1], point_n[0] - point_p[0]) - math.atan2(V_PN[1], V_PN[0])
    point_w = v2_add(point_p, rotate2(beta, V_PW))
    length = math.hypot(point_w[0], point_w[1])
    theta_leg = math.atan2(point_w[0], -point_w[1])
    return {
        "L0": length,
        "theta_leg": theta_leg,
        "phi0": math.pi * 0.5 - theta_leg,
        "J": point_j,
        "L": point_l,
        "M": point_m,
        "K": point_k,
        "P": point_p,
        "N": point_n,
        "W": point_w,
        "alpha": alpha,
        "beta": beta,
    }


def simplified_closed_chain(active: Sequence[float]) -> dict:
    """Recommended nominal FK using one closure plus the parallelogram map.

    The complete K-N closure remains the independent geometry baseline.  This
    function removes K, N and the second circle intersection from the runtime
    path by using W = P + S_b (M - L).
    """

    q_front, q_rear = active
    point_j = rotate2(q_front, V_IJ)
    point_l = rotate2(q_rear, V_IL)
    point_p = rotate2(q_rear, V_IP)
    point_m = circle_intersection(
        point_j, v2_norm(V_JM), point_l, v2_norm(V_ML), +1.0
    )
    point_w = v2_add(
        point_p,
        apply_similarity(
            v2_sub(point_m, point_l),
            SIMPLIFIED_SCALE_B,
            SIMPLIFIED_ANGLE_B,
        ),
    )
    length = math.hypot(point_w[0], point_w[1])
    theta_leg = math.atan2(point_w[0], -point_w[1])
    return {
        "L0": length,
        "theta_leg": theta_leg,
        "phi0": math.pi * 0.5 - theta_leg,
        "J": point_j,
        "L": point_l,
        "M": point_m,
        "P": point_p,
        "W": point_w,
    }


def cad_one_closure(active: Sequence[float]) -> dict:
    """CAD 4.3 FK: I-L-P collinear and P-W parallel to L-M.

    Unlike ``simplified_closed_chain``, this proposed controller model drops
    the tiny exported XML direction residuals exactly as documented in the
    current HTML report:

        P = k_a L
        W = P + k_b (M - L)
    """

    q_front, q_rear = active
    point_j = rotate2(q_front, V_IJ)
    point_l = rotate2(q_rear, V_IL)
    point_p = (CAD_SCALE_A * point_l[0], CAD_SCALE_A * point_l[1])
    point_m = circle_intersection(
        point_j, v2_norm(V_JM), point_l, v2_norm(V_ML), +1.0
    )
    point_w = (
        point_p[0] + CAD_SCALE_B * (point_m[0] - point_l[0]),
        point_p[1] + CAD_SCALE_B * (point_m[1] - point_l[1]),
    )
    length = math.hypot(point_w[0], point_w[1])
    theta_leg = math.atan2(point_w[0], -point_w[1])
    return {
        "L0": length,
        "theta_leg": theta_leg,
        "phi0": math.pi * 0.5 - theta_leg,
        "J": point_j,
        "L": point_l,
        "M": point_m,
        "P": point_p,
        "W": point_w,
    }


def v2_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def point_jacobian_column(jacobian: Sequence[Sequence[float]], column: int) -> tuple[float, float]:
    return (jacobian[0][column], jacobian[1][column])


def circle_intersection_jacobian(
    point_x: Sequence[float],
    center_a: Sequence[float],
    jacobian_a: Sequence[Sequence[float]],
    center_b: Sequence[float],
    jacobian_b: Sequence[Sequence[float]],
) -> tuple[list[list[float]], float]:
    """Differentiate a selected two-circle intersection exactly.

    X is already the intersection selected by the forward-kinematics assembly
    branch.  The fixed-radius constraints are differentiated implicitly, so no
    input perturbation or finite-difference step is used here.
    """

    arm_a = v2_sub(point_x, center_a)
    arm_b = v2_sub(point_x, center_b)
    determinant = arm_a[0] * arm_b[1] - arm_a[1] * arm_b[0]
    if abs(determinant) < 1e-12:
        raise RuntimeError("Circle-intersection Jacobian is singular")

    jacobian_x = [[0.0, 0.0], [0.0, 0.0]]
    for column in range(2):
        derivative_a = point_jacobian_column(jacobian_a, column)
        derivative_b = point_jacobian_column(jacobian_b, column)
        rhs_a = v2_dot(arm_a, derivative_a)
        rhs_b = v2_dot(arm_b, derivative_b)
        jacobian_x[0][column] = (
            arm_b[1] * rhs_a - arm_a[1] * rhs_b
        ) / determinant
        jacobian_x[1][column] = (
            -arm_b[0] * rhs_a + arm_a[0] * rhs_b
        ) / determinant
    return jacobian_x, determinant


def rigid_target_jacobian(
    point_anchor: Sequence[float],
    jacobian_anchor: Sequence[Sequence[float]],
    point_direction: Sequence[float],
    jacobian_direction: Sequence[Sequence[float]],
    point_target: Sequence[float],
) -> tuple[list[list[float]], list[float]]:
    """Propagate derivatives through a planar rigid body's fixed geometry.

    ``anchor -> direction`` fixes the current body orientation, while
    ``anchor -> target`` is another body-fixed vector such as M->K or P->W.
    """

    direction = v2_sub(point_direction, point_anchor)
    target_arm = v2_sub(point_target, point_anchor)
    norm_sq = v2_dot(direction, direction)
    if norm_sq < 1e-16:
        raise RuntimeError("Rigid-body direction vector is degenerate")

    jacobian_target = [[0.0, 0.0], [0.0, 0.0]]
    angle_jacobian = [0.0, 0.0]
    for column in range(2):
        derivative_anchor = point_jacobian_column(jacobian_anchor, column)
        derivative_direction = point_jacobian_column(jacobian_direction, column)
        delta_direction = (
            derivative_direction[0] - derivative_anchor[0],
            derivative_direction[1] - derivative_anchor[1],
        )
        angle_rate = (
            direction[0] * delta_direction[1]
            - direction[1] * delta_direction[0]
        ) / norm_sq
        angle_jacobian[column] = angle_rate
        jacobian_target[0][column] = derivative_anchor[0] - angle_rate * target_arm[1]
        jacobian_target[1][column] = derivative_anchor[1] + angle_rate * target_arm[0]
    return jacobian_target, angle_jacobian


def analytic_closed_chain_with_jacobian(active: Sequence[float]) -> dict:
    """Return exact closed-chain FK and its analytic controller Jacobian.

    Each point carries a 2x2 derivative with columns [q_front, q_rear].
    The two circle intersections are differentiated by their fixed-distance
    constraints.  Runtime central differences are therefore unnecessary.
    """

    result = analytic_closed_chain(active)
    point_j = result["J"]
    point_l = result["L"]
    point_m = result["M"]
    point_k = result["K"]
    point_p = result["P"]
    point_n = result["N"]
    point_w = result["W"]

    jacobian_j = [[-point_j[1], 0.0], [point_j[0], 0.0]]
    jacobian_l = [[0.0, -point_l[1]], [0.0, point_l[0]]]
    jacobian_p = [[0.0, -point_p[1]], [0.0, point_p[0]]]

    jacobian_m, determinant_m = circle_intersection_jacobian(
        point_m, point_j, jacobian_j, point_l, jacobian_l
    )
    jacobian_k, alpha_jacobian = rigid_target_jacobian(
        point_m, jacobian_m, point_l, jacobian_l, point_k
    )
    jacobian_n, determinant_n = circle_intersection_jacobian(
        point_n, point_k, jacobian_k, point_p, jacobian_p
    )
    jacobian_w, beta_jacobian = rigid_target_jacobian(
        point_p, jacobian_p, point_n, jacobian_n, point_w
    )

    wheel_s, wheel_z = point_w
    length = result["L0"]
    length_sq = length * length
    if length_sq < 1e-16:
        raise RuntimeError("Virtual-leg polar Jacobian is singular at L0=0")
    jacobian_h = [[0.0, 0.0], [0.0, 0.0]]
    for column in range(2):
        wheel_s_derivative = jacobian_w[0][column]
        wheel_z_derivative = jacobian_w[1][column]
        jacobian_h[0][column] = (
            wheel_s * wheel_s_derivative + wheel_z * wheel_z_derivative
        ) / length
        # phi0 = atan2(-wheel_z, wheel_s)
        jacobian_h[1][column] = (
            wheel_z * wheel_s_derivative - wheel_s * wheel_z_derivative
        ) / length_sq

    result.update(
        {
            "JH": jacobian_h,
            "D_J": jacobian_j,
            "D_L": jacobian_l,
            "D_P": jacobian_p,
            "D_M": jacobian_m,
            "D_K": jacobian_k,
            "D_N": jacobian_n,
            "D_W": jacobian_w,
            "D_alpha": alpha_jacobian,
            "D_beta": beta_jacobian,
            "circle_M_determinant_m2": determinant_m,
            "circle_N_determinant_m2": determinant_n,
        }
    )
    return result


def simplified_closed_chain_with_jacobian(active: Sequence[float]) -> dict:
    """Recommended FK/JH path using only the first closure at M."""

    result = simplified_closed_chain(active)
    point_j = result["J"]
    point_l = result["L"]
    point_m = result["M"]
    point_p = result["P"]
    point_w = result["W"]

    jacobian_j = [[-point_j[1], 0.0], [point_j[0], 0.0]]
    jacobian_l = [[0.0, -point_l[1]], [0.0, point_l[0]]]
    jacobian_p = [[0.0, -point_p[1]], [0.0, point_p[0]]]
    jacobian_m, determinant_m = circle_intersection_jacobian(
        point_m, point_j, jacobian_j, point_l, jacobian_l
    )

    # W = P + S_b(M - L), hence D_W = D_P + S_b(D_M - D_L).
    jacobian_w = [[0.0, 0.0], [0.0, 0.0]]
    for column in range(2):
        derivative_lm = (
            jacobian_m[0][column] - jacobian_l[0][column],
            jacobian_m[1][column] - jacobian_l[1][column],
        )
        transformed = apply_similarity(
            derivative_lm, SIMPLIFIED_SCALE_B, SIMPLIFIED_ANGLE_B
        )
        jacobian_w[0][column] = jacobian_p[0][column] + transformed[0]
        jacobian_w[1][column] = jacobian_p[1][column] + transformed[1]

    wheel_s, wheel_z = point_w
    length = result["L0"]
    length_sq = length * length
    if length_sq < 1e-16:
        raise RuntimeError("Virtual-leg polar Jacobian is singular at L0=0")
    jacobian_h = [[0.0, 0.0], [0.0, 0.0]]
    for column in range(2):
        wheel_s_derivative = jacobian_w[0][column]
        wheel_z_derivative = jacobian_w[1][column]
        jacobian_h[0][column] = (
            wheel_s * wheel_s_derivative + wheel_z * wheel_z_derivative
        ) / length
        jacobian_h[1][column] = (
            wheel_z * wheel_s_derivative - wheel_s * wheel_z_derivative
        ) / length_sq

    result.update(
        {
            "JH": jacobian_h,
            "D_J": jacobian_j,
            "D_L": jacobian_l,
            "D_P": jacobian_p,
            "D_M": jacobian_m,
            "D_W": jacobian_w,
            "circle_M_determinant_m2": determinant_m,
        }
    )
    return result


def cad_one_closure_with_jacobian(active: Sequence[float]) -> dict:
    """Analytic J_H for the CAD 4.3 one-closure controller model."""

    result = cad_one_closure(active)
    point_j = result["J"]
    point_l = result["L"]
    point_m = result["M"]
    point_p = result["P"]
    point_w = result["W"]

    jacobian_j = [[-point_j[1], 0.0], [point_j[0], 0.0]]
    jacobian_l = [[0.0, -point_l[1]], [0.0, point_l[0]]]
    jacobian_p = [
        [CAD_SCALE_A * value for value in jacobian_l[0]],
        [CAD_SCALE_A * value for value in jacobian_l[1]],
    ]
    jacobian_m, determinant_m = circle_intersection_jacobian(
        point_m, point_j, jacobian_j, point_l, jacobian_l
    )
    jacobian_w = [[0.0, 0.0], [0.0, 0.0]]
    for column in range(2):
        jacobian_w[0][column] = (
            CAD_SCALE_B * jacobian_m[0][column]
            + (CAD_SCALE_A - CAD_SCALE_B) * jacobian_l[0][column]
        )
        jacobian_w[1][column] = (
            CAD_SCALE_B * jacobian_m[1][column]
            + (CAD_SCALE_A - CAD_SCALE_B) * jacobian_l[1][column]
        )

    wheel_s, wheel_z = point_w
    length = result["L0"]
    length_sq = length * length
    if length_sq < 1e-16:
        raise RuntimeError("Virtual-leg polar Jacobian is singular at L0=0")
    jacobian_h = [[0.0, 0.0], [0.0, 0.0]]
    for column in range(2):
        wheel_s_derivative = jacobian_w[0][column]
        wheel_z_derivative = jacobian_w[1][column]
        jacobian_h[0][column] = (
            wheel_s * wheel_s_derivative + wheel_z * wheel_z_derivative
        ) / length
        jacobian_h[1][column] = (
            wheel_z * wheel_s_derivative - wheel_s * wheel_z_derivative
        ) / length_sq

    result.update(
        {
            "JH": jacobian_h,
            "D_J": jacobian_j,
            "D_L": jacobian_l,
            "D_P": jacobian_p,
            "D_M": jacobian_m,
            "D_W": jacobian_w,
            "circle_M_determinant_m2": determinant_m,
        }
    )
    return result


def analytic_jacobian_center_difference(
    active: Sequence[float], step: float = 1e-6
) -> list[list[float]]:
    """Independent offline check of the analytic closed-chain Jacobian."""

    jacobian = [[0.0, 0.0], [0.0, 0.0]]
    for column in range(2):
        active_plus = list(active)
        active_minus = list(active)
        active_plus[column] += step
        active_minus[column] -= step
        plus = analytic_closed_chain(active_plus)
        minus = analytic_closed_chain(active_minus)
        jacobian[0][column] = (plus["L0"] - minus["L0"]) / (2.0 * step)
        phi_delta = wrap_pi(plus["phi0"] - minus["phi0"])
        jacobian[1][column] = phi_delta / (2.0 * step)
    return jacobian


def wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def simplified_jacobian_center_difference(
    active: Sequence[float], step: float = 1e-6
) -> list[list[float]]:
    """Offline finite-difference check of the recommended simplified JH."""

    jacobian = [[0.0, 0.0], [0.0, 0.0]]
    for column in range(2):
        active_plus = list(active)
        active_minus = list(active)
        active_plus[column] += step
        active_minus[column] -= step
        plus = simplified_closed_chain(active_plus)
        minus = simplified_closed_chain(active_minus)
        jacobian[0][column] = (plus["L0"] - minus["L0"]) / (2.0 * step)
        jacobian[1][column] = wrap_pi(plus["phi0"] - minus["phi0"]) / (
            2.0 * step
        )
    return jacobian


def ideal_fivebar(active: Sequence[float], branch_sign: float = 1.0) -> tuple[float, float, float, float]:
    """Current firmware's ideal L5=0 five-bar FK as [L0, phi0].

    The two active angles are converted exactly as sim_adapter.c converts the
    left MuJoCo joints into the controller's phi4/phi1 feedback.
    """
    q_front, q_rear = active
    phi4 = wrap_pi(-q_front + J0_OFFSET)
    phi1 = wrap_pi(-q_rear + J1_OFFSET)
    xb, yb = L1 * math.cos(phi1), L1 * math.sin(phi1)
    xd, yd = L1 * math.cos(phi4), L1 * math.sin(phi4)
    dx, dy = xd - xb, yd - yb
    bd = math.hypot(dx, dy)
    if bd < 1e-12 or bd > 2.0 * L2:
        return (math.nan, math.nan, math.nan, math.nan)
    mx, my = (xb + xd) * 0.5, (yb + yd) * 0.5
    h = math.sqrt(max(0.0, L2 * L2 - 0.25 * bd * bd))
    nx, ny = -dy / bd, dx / bd
    # In VMC_Calc.c the controller's +y direction represents chassis down, so
    # the physical wheel is the larger-y circle intersection on this assembly
    # branch.  (This is a control-frame convention, not the XML world Z sign.)
    candidates = [(mx + h * nx, my + h * ny), (mx - h * nx, my - h * ny)]
    xc, yc = max(candidates, key=lambda p: p[1]) if branch_sign >= 0 else min(candidates, key=lambda p: p[1])
    length = math.hypot(xc, yc)
    phi0 = math.atan2(yc, xc)
    return length, phi0, xc, yc


def inverse_exact(
    tree: MjcfTree,
    target: Sequence[float],
    active_seed: Sequence[float],
    passive_seed: Sequence[float],
) -> tuple[list[float], list[float], float, int]:
    active = list(active_seed)
    passive = list(passive_seed)
    for iteration in range(40):
        value, passive, closure_rms = exact_map(tree, active, passive)
        error = [value[0] - target[0], wrap_pi(value[1] - target[1])]
        scaled_error = [error[0], 0.20 * error[1]]
        if math.hypot(scaled_error[0], scaled_error[1]) < 2e-8 and closure_rms < 1e-9:
            return active, passive, math.hypot(scaled_error[0], scaled_error[1]), iteration
        jac = analytic_closed_chain_with_jacobian(active)["JH"]
        scaled_j = [[jac[0][0], jac[0][1]], [0.20 * jac[1][0], 0.20 * jac[1][1]]]
        step = solve_linear(scaled_j, [-scaled_error[0], -scaled_error[1]])
        norm = math.hypot(step[0], step[1])
        if norm > 0.20:
            step = [x * 0.20 / norm for x in step]
        active = [active[i] + step[i] for i in range(2)]
    value, passive, _ = exact_map(tree, active, passive)
    error = math.hypot(value[0] - target[0], 0.20 * wrap_pi(value[1] - target[1]))
    return active, passive, error, 40


def matrix_det_2(j: Sequence[Sequence[float]]) -> float:
    return j[0][0] * j[1][1] - j[0][1] * j[1][0]


def matrix_condition_2(j: Sequence[Sequence[float]]) -> float:
    # 2-norm condition number from eigenvalues of J^T J.
    a = j[0][0] ** 2 + j[1][0] ** 2
    b = j[0][0] * j[0][1] + j[1][0] * j[1][1]
    d = j[0][1] ** 2 + j[1][1] ** 2
    trace = a + d
    disc = math.sqrt(max(0.0, (a - d) ** 2 + 4.0 * b * b))
    lmax, lmin = 0.5 * (trace + disc), 0.5 * (trace - disc)
    return math.sqrt(lmax / lmin) if lmin > 1e-16 else math.inf


def run(xml_path: Path, json_path: Path, csv_path: Path, grid_csv_path: Path) -> dict:
    tree = MjcfTree(xml_path)

    default_residual = closure_residual(tree, (0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
    default_rms = math.sqrt(sum(x * x for x in default_residual) / len(default_residual))

    # Follow one continuous assembly branch from the exported q=0 assembly to
    # the useful 0.15--0.42 m leg-length range at a vertical virtual-leg target
    # (controller phi0=pi/2, equivalently theta_leg=0).
    passive = [0.0, 0.0, 0.0, 0.0]
    initial, passive, rms = exact_map(tree, (0.0, 0.0), passive)
    active = [0.0, 0.0]
    samples: list[dict] = []
    vertical_seeds: dict[float, tuple[list[float], list[float]]] = {}
    targets = [0.15 + i * 0.01 for i in range(28)]  # 0.15 ... 0.42 m
    # Start closer to the long-leg end and continue monotonically to preserve
    # the assembly branch, then report in ascending order.
    for length in reversed(targets):
        active, passive, inv_error, iterations = inverse_exact(
            tree, (length, math.pi * 0.5), active, passive
        )
        exact, passive, closure_rms = exact_map(tree, active, passive)
        ideal = ideal_fivebar(active)
        analytic = analytic_closed_chain_with_jacobian(active)
        jac = analytic["JH"]
        tree_fd_jac, passive, _ = tree_jacobian_center_difference(
            tree, active, passive
        )
        jacobian_tree_fd_error = max(
            abs(jac[row][column] - tree_fd_jac[row][column])
            for row in range(2)
            for column in range(2)
        )
        length_error = ideal[0] - exact[0]
        phi0_error = wrap_pi(ideal[1] - exact[1])
        theta_leg_error = -phi0_error
        theta_jac = [jac[0][:], [-jac[1][0], -jac[1][1]]]
        samples.append(
            {
                "target_l0_m": length,
                "q_front_rad": active[0],
                "q_rear_rad": active[1],
                "passive_rad": passive[:],
                "exact_l0_m": exact[0],
                "exact_phi0_rad": exact[1],
                "exact_theta_leg_rad": math.pi * 0.5 - exact[1],
                "ideal_l0_m": ideal[0],
                "ideal_phi0_rad": ideal[1],
                "ideal_theta_leg_rad": math.pi * 0.5 - ideal[1],
                "ideal_minus_exact_l0_mm": 1000.0 * length_error,
                "ideal_minus_exact_phi0_deg": math.degrees(phi0_error),
                "ideal_minus_exact_theta_leg_deg": math.degrees(theta_leg_error),
                "closure_rms_um": closure_rms * 1e6,
                "inverse_scaled_error": inv_error,
                "inverse_iterations": iterations,
                "jacobian_phi0": jac,
                "jacobian_theta_leg": theta_jac,
                "jacobian_det_phi0": matrix_det_2(jac),
                "jacobian_condition": matrix_condition_2(jac),
                "analytic_minus_tree_fd_jacobian_max_abs": jacobian_tree_fd_error,
                "circle_M_determinant_m2": analytic["circle_M_determinant_m2"],
                "circle_N_determinant_m2": analytic["circle_N_determinant_m2"],
                "analytic_minus_tree_l0_um": 1e6 * (analytic["L0"] - exact[0]),
                "analytic_minus_tree_theta_microdeg": 1e6
                * math.degrees(
                    wrap_pi(analytic["theta_leg"] - (math.pi * 0.5 - exact[1]))
                ),
            }
        )
        vertical_seeds[round(length, 2)] = (active[:], passive[:])
    samples.sort(key=lambda row: row["target_l0_m"])

    # A two-dimensional check covers phi0=70..110 deg, equivalently the
    # zero-pitch virtual-leg angle theta_leg=+20..-20 deg from vertical.
    grid_samples: list[dict] = []

    def append_grid(length: float, theta_deg: float, active_q: Sequence[float], passive_q: Sequence[float], inv_error: float) -> None:
        exact, _, closure_rms = exact_map(tree, active_q, passive_q)
        ideal = ideal_fivebar(active_q)
        analytic = analytic_closed_chain_with_jacobian(active_q)
        simplified = simplified_closed_chain_with_jacobian(active_q)
        finite_difference_jacobian = analytic_jacobian_center_difference(active_q)
        simplified_finite_difference_jacobian = (
            simplified_jacobian_center_difference(active_q)
        )
        analytic_fd_error = max(
            abs(analytic["JH"][row][column] - finite_difference_jacobian[row][column])
            for row in range(2)
            for column in range(2)
        )
        exact_theta_leg = math.pi * 0.5 - exact[1]
        ideal_theta_leg = math.pi * 0.5 - ideal[1]
        grid_samples.append(
            {
                "target_l0_m": length,
                "target_theta_leg_deg": theta_deg,
                "target_phi0_deg": 90.0 - theta_deg,
                "q_front_rad": active_q[0],
                "q_rear_rad": active_q[1],
                "exact_l0_m": exact[0],
                "exact_phi0_deg": math.degrees(exact[1]),
                "exact_theta_leg_deg": math.degrees(exact_theta_leg),
                "ideal_l0_m": ideal[0],
                "ideal_phi0_deg": math.degrees(ideal[1]),
                "ideal_theta_leg_deg": math.degrees(ideal_theta_leg),
                "ideal_minus_exact_l0_mm": 1000.0 * (ideal[0] - exact[0]),
                "ideal_minus_exact_phi0_deg": math.degrees(wrap_pi(ideal[1] - exact[1])),
                "ideal_minus_exact_theta_leg_deg": math.degrees(
                    wrap_pi(ideal_theta_leg - exact_theta_leg)
                ),
                "analytic_minus_tree_l0_um": 1e6 * (analytic["L0"] - exact[0]),
                "analytic_minus_tree_theta_microdeg": 1e6
                * math.degrees(wrap_pi(analytic["theta_leg"] - exact_theta_leg)),
                "analytic_minus_fd_jacobian_max_abs": analytic_fd_error,
                "simplified_minus_full_w_um": 1e6
                * v2_norm(v2_sub(simplified["W"], analytic["W"])),
                "simplified_minus_full_l0_um": 1e6
                * (simplified["L0"] - analytic["L0"]),
                "simplified_minus_full_phi0_microdeg": 1e6
                * math.degrees(wrap_pi(simplified["phi0"] - analytic["phi0"])),
                "simplified_minus_full_jacobian_max_abs": max(
                    abs(simplified["JH"][row][column] - analytic["JH"][row][column])
                    for row in range(2)
                    for column in range(2)
                ),
                "simplified_minus_fd_jacobian_max_abs": max(
                    abs(
                        simplified["JH"][row][column]
                        - simplified_finite_difference_jacobian[row][column]
                    )
                    for row in range(2)
                    for column in range(2)
                ),
                "circle_M_determinant_m2": analytic["circle_M_determinant_m2"],
                "circle_N_determinant_m2": analytic["circle_N_determinant_m2"],
                "closure_rms_um": closure_rms * 1e6,
                "inverse_scaled_error": inv_error,
            }
        )

    for length in targets:
        base_active, base_passive = vertical_seeds[round(length, 2)]
        append_grid(length, 0.0, base_active, base_passive, 0.0)
        for direction in (-1.0, +1.0):
            angle_active, angle_passive = base_active[:], base_passive[:]
            for magnitude in (5.0, 10.0, 15.0, 20.0):
                theta_deg = direction * magnitude
                angle_active, angle_passive, inv_error, _ = inverse_exact(
                    tree,
                    (length, math.pi * 0.5 - math.radians(theta_deg)),
                    angle_active,
                    angle_passive,
                )
                append_grid(length, theta_deg, angle_active, angle_passive, inv_error)
    grid_samples.sort(key=lambda row: (row["target_l0_m"], row["target_theta_leg_deg"]))

    finite = [s for s in samples if math.isfinite(s["ideal_l0_m"])]
    summary = {
        "xml": str(xml_path),
        "coordinate_definition": {
            "q_order": ["q_front_raw", "q_rear_raw"],
            "q_front_raw": "MuJoCo left front active joint jIJ raw qpos, right-hand positive about its hinge axis, rad",
            "q_rear_raw": "MuJoCo left rear active joint jIO raw qpos, right-hand positive about its hinge axis, rad",
            "left_q": ["jIJ = joint_pos[0]", "jIO = joint_pos[1]"],
            "right_q_for_same_front_rear_interface": [
                "jAB = joint_pos[3]",
                "jAG = joint_pos[2]",
            ],
            "front_rear_meaning": "anatomical/XML linkage branches, not vehicle driving direction",
            "raw_q_positive": "right-hand positive about local hinge axis (0,-1,0); in the local X-Z drawing, +X rotates toward +Z",
            "s_W": "signed wheel displacement along +base Y in the leg side-view plane; legacy report name x_H",
            "d_W": "signed wheel displacement toward chassis down, equal to -base Z",
            "controller_drive_x_relation": "main_mujoco controller x = -s_W direction for the audited nominal base orientation",
            "wheel_position_map_output": ["s_W", "d_W"],
            "L0": "distance from coaxial hip axis to wheel axis, m",
            "theta_leg": "atan2(s_W, d_W), positive from chassis down toward +base Y, rad",
            "phi0": "pi/2 - theta_leg = atan2(d_W, s_W), controller virtual-leg polar angle, rad",
            "H_output": ["L0", "phi0"],
            "J_phi0_rows": ["dL0", "dphi0"],
            "J_columns": ["dq_front_raw", "dq_rear_raw"],
        },
        "default_closure_rms_um": default_rms * 1e6,
        "default_exact_l0_m": initial[0],
        "default_exact_phi0_deg": math.degrees(initial[1]),
        "default_exact_theta_leg_deg": math.degrees(math.pi * 0.5 - initial[1]),
        "max_abs_ideal_l0_error_mm": max(abs(s["ideal_minus_exact_l0_mm"]) for s in finite),
        "max_abs_ideal_phi0_error_deg": max(abs(s["ideal_minus_exact_phi0_deg"]) for s in finite),
        "max_abs_ideal_theta_leg_error_deg": max(
            abs(s["ideal_minus_exact_theta_leg_deg"]) for s in finite
        ),
        "grid_domain": "L0=0.15..0.42 m in 0.01 m steps; phi0=70..110 deg in 5 deg steps",
        "grid_max_abs_ideal_l0_error_mm": max(abs(s["ideal_minus_exact_l0_mm"]) for s in grid_samples),
        "grid_max_abs_ideal_phi0_error_deg": max(
            abs(s["ideal_minus_exact_phi0_deg"]) for s in grid_samples
        ),
        "grid_max_abs_ideal_theta_leg_error_deg": max(
            abs(s["ideal_minus_exact_theta_leg_deg"]) for s in grid_samples
        ),
        "max_abs_analytic_minus_tree_l0_um": max(abs(s["analytic_minus_tree_l0_um"]) for s in grid_samples),
        "max_abs_analytic_minus_tree_theta_microdeg": max(
            abs(s["analytic_minus_tree_theta_microdeg"]) for s in grid_samples
        ),
        "max_abs_analytic_minus_fd_jacobian": max(
            s["analytic_minus_fd_jacobian_max_abs"] for s in grid_samples
        ),
        "max_abs_analytic_minus_tree_fd_jacobian": max(
            s["analytic_minus_tree_fd_jacobian_max_abs"] for s in samples
        ),
        "simplified_scale_b": SIMPLIFIED_SCALE_B,
        "simplified_angle_b_deg": math.degrees(SIMPLIFIED_ANGLE_B),
        "parallelogram_lk_pn_vector_mismatch_um": 1e6
        * v2_norm(v2_sub(v2_sub(V_MK, V_ML), V_PN)),
        "parallelogram_lp_kn_vector_mismatch_um": 1e6
        * v2_norm(v2_sub(v2_sub(V_IP, V_IL), V_KN)),
        "grid_max_simplified_minus_full_w_um": max(
            s["simplified_minus_full_w_um"] for s in grid_samples
        ),
        "grid_max_abs_simplified_minus_full_l0_um": max(
            abs(s["simplified_minus_full_l0_um"]) for s in grid_samples
        ),
        "grid_max_abs_simplified_minus_full_phi0_microdeg": max(
            abs(s["simplified_minus_full_phi0_microdeg"]) for s in grid_samples
        ),
        "grid_max_simplified_minus_full_jacobian": max(
            s["simplified_minus_full_jacobian_max_abs"] for s in grid_samples
        ),
        "grid_max_simplified_minus_fd_jacobian": max(
            s["simplified_minus_fd_jacobian_max_abs"] for s in grid_samples
        ),
        "circle_M_determinant_range_m2": [
            min(s["circle_M_determinant_m2"] for s in grid_samples),
            max(s["circle_M_determinant_m2"] for s in grid_samples),
        ],
        "circle_N_determinant_range_m2": [
            min(s["circle_N_determinant_m2"] for s in grid_samples),
            max(s["circle_N_determinant_m2"] for s in grid_samples),
        ],
        "max_closure_rms_um": max(s["closure_rms_um"] for s in samples),
        "max_inverse_scaled_error": max(s["inverse_scaled_error"] for s in samples),
        "samples": samples,
        "grid_samples": grid_samples,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "target_l0_m",
                "q_front_rad",
                "q_rear_rad",
                "exact_l0_m",
                "ideal_l0_m",
                "ideal_minus_exact_l0_mm",
                "exact_phi0_deg",
                "ideal_phi0_deg",
                "ideal_minus_exact_phi0_deg",
                "exact_theta_leg_deg",
                "ideal_theta_leg_deg",
                "ideal_minus_exact_theta_leg_deg",
                "closure_rms_um",
                "J_L_qfront_m_per_rad",
                "J_L_qrear_m_per_rad",
                "J_phi0_qfront_rad_per_rad",
                "J_phi0_qrear_rad_per_rad",
                "J_theta_leg_qfront_rad_per_rad",
                "J_theta_leg_qrear_rad_per_rad",
                "J_condition",
            ]
        )
        for s in samples:
            j_phi0 = s["jacobian_phi0"]
            j_theta = s["jacobian_theta_leg"]
            writer.writerow(
                [
                    s["target_l0_m"],
                    s["q_front_rad"],
                    s["q_rear_rad"],
                    s["exact_l0_m"],
                    s["ideal_l0_m"],
                    s["ideal_minus_exact_l0_mm"],
                    math.degrees(s["exact_phi0_rad"]),
                    math.degrees(s["ideal_phi0_rad"]),
                    s["ideal_minus_exact_phi0_deg"],
                    math.degrees(s["exact_theta_leg_rad"]),
                    math.degrees(s["ideal_theta_leg_rad"]),
                    s["ideal_minus_exact_theta_leg_deg"],
                    s["closure_rms_um"],
                    j_phi0[0][0],
                    j_phi0[0][1],
                    j_phi0[1][0],
                    j_phi0[1][1],
                    j_theta[1][0],
                    j_theta[1][1],
                    s["jacobian_condition"],
                ]
            )
    with grid_csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = list(grid_samples[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(grid_samples)
    return summary


def main() -> None:
    here = Path(__file__).resolve()
    root = here.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        type=Path,
        default=root / "sim" / "models" / "wheel_leg_urdf4_self_mesh_all.xml",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=root / "output" / "offset_kinematics_results.json",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "output" / "offset_kinematics_sweep.csv",
    )
    parser.add_argument(
        "--grid-csv",
        type=Path,
        default=root / "output" / "offset_kinematics_grid.csv",
    )
    args = parser.parse_args()
    result = run(args.xml.resolve(), args.json.resolve(), args.csv.resolve(), args.grid_csv.resolve())
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"samples", "grid_samples"}},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
