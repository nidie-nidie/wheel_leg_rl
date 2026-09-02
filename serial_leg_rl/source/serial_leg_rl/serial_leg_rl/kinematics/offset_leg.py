"""Offset closed-chain leg kinematics.

This module implements the section 4.3 CAD one-closure model documented in
``offset_closed_chain_kinematics_derivation.html``. It is intentionally scalar
and dependency-free so Phase 0 tests can run without Isaac Sim or PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


Vector2 = tuple[float, float]
Matrix2 = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class OffsetLegGeometry:
    """Geometry constants for one leg in the local ``(s, z)`` plane."""

    v_ij: Vector2 = (0.096749440646263, 0.00674587310762054)
    v_il: Vector2 = (-0.0963551866275421, 0.0109403315543567)
    v_ip: Vector2 = (-0.21357099037258, 0.0245217656620774)
    l_jm: float = 0.115
    l_lm: float = 0.115
    l_pw: float = 0.258
    sigma_phi: int = -1
    singular_epsilon: float = 1.0e-9

    @property
    def l_il(self) -> float:
        return norm(self.v_il)

    @property
    def l_ip(self) -> float:
        return norm(self.v_ip)

    @property
    def k_a(self) -> float:
        return self.l_ip / self.l_il

    @property
    def k_b(self) -> float:
        return self.l_pw / self.l_lm


@dataclass(frozen=True)
class LegState:
    """Forward kinematics result for one leg."""

    q_front: float
    q_rear: float
    j: Vector2
    l: Vector2
    p: Vector2
    m: Vector2
    w: Vector2
    leg_length: float
    phi0: float
    theta_leg: float
    jacobian: Matrix2
    delta_m: float
    phi2: float


class KinematicsError(ValueError):
    """Raised when the requested leg configuration is not valid."""


class OffsetLegKinematics:
    """CAD simplified offset closed-chain kinematics for one leg."""

    def __init__(self, geometry: OffsetLegGeometry | None = None):
        self.geometry = geometry or OffsetLegGeometry()

    def forward(self, q_front: float, q_rear: float) -> LegState:
        """Compute ``H(q) = [L0, phi0]`` and the analytic Jacobian."""

        g = self.geometry
        point_j = rotate(q_front, g.v_ij)
        point_l = rotate(q_rear, g.v_il)
        point_p = scale(g.k_a, point_l)
        point_m, phi2 = ideal_fivebar_point_m(point_l, g.l_lm, point_j, g.l_jm, g.sigma_phi)
        point_w = add(point_p, scale(g.k_b, sub(point_m, point_l)))

        leg_length = norm(point_w)
        if leg_length <= g.singular_epsilon:
            raise KinematicsError("virtual leg length is singular")

        s_w, z_w = point_w
        phi0 = math.atan2(-z_w, s_w)
        theta_leg = math.pi / 2.0 - phi0

        jacobian, delta_m = self._jacobian(point_j, point_l, point_m, point_w)
        return LegState(
            q_front=q_front,
            q_rear=q_rear,
            j=point_j,
            l=point_l,
            p=point_p,
            m=point_m,
            w=point_w,
            leg_length=leg_length,
            phi0=phi0,
            theta_leg=theta_leg,
            jacobian=jacobian,
            delta_m=delta_m,
            phi2=phi2,
        )

    def virtual_velocity(self, q_front: float, q_rear: float, dq_front: float, dq_rear: float) -> tuple[float, float]:
        """Return ``(dL0, dphi0)`` from joint velocities."""

        state = self.forward(q_front, q_rear)
        jh = state.jacobian
        d_l0 = jh[0][0] * dq_front + jh[0][1] * dq_rear
        d_phi0 = jh[1][0] * dq_front + jh[1][1] * dq_rear
        return d_l0, d_phi0

    def inverse(
        self,
        leg_length: float,
        phi0: float,
        seed: tuple[float, float] = (0.0, 0.0),
        max_iterations: int = 30,
        tolerance: float = 1.0e-10,
    ) -> tuple[float, float]:
        """Solve ``H(q) = [leg_length, phi0]`` with Newton iterations."""

        q_front, q_rear = seed
        for _ in range(max_iterations):
            state = self.forward(q_front, q_rear)
            err_l = leg_length - state.leg_length
            err_phi = wrap_to_pi(phi0 - state.phi0)
            if abs(err_l) < tolerance and abs(err_phi) < tolerance:
                return q_front, q_rear

            (a, b), (c, d) = state.jacobian
            det = a * d - b * c
            if abs(det) <= self.geometry.singular_epsilon:
                raise KinematicsError("Jacobian is singular during IK")

            dq_front = (d * err_l - b * err_phi) / det
            dq_rear = (-c * err_l + a * err_phi) / det
            q_front = wrap_to_pi(q_front + dq_front)
            q_rear = wrap_to_pi(q_rear + dq_rear)

        raise KinematicsError("IK did not converge")

    def _jacobian(self, point_j: Vector2, point_l: Vector2, point_m: Vector2, point_w: Vector2) -> tuple[Matrix2, float]:
        g = self.geometry

        j_s, j_z = point_j
        l_s, l_z = point_l
        m_s, m_z = point_m
        w_s, w_z = point_w
        leg_length = norm(point_w)

        u_s = m_s - j_s
        u_z = m_z - j_z
        v_s = m_s - l_s
        v_z = m_z - l_z
        delta_m = u_s * v_z - u_z * v_s
        if abs(delta_m) <= g.singular_epsilon:
            raise KinematicsError("first closed-chain branch is singular")

        b1_front = -u_s * j_z + u_z * j_s
        m_s_front = v_z * b1_front / delta_m
        m_z_front = -v_s * b1_front / delta_m

        b2_rear = -v_s * l_z + v_z * l_s
        m_s_rear = -u_z * b2_rear / delta_m
        m_z_rear = u_s * b2_rear / delta_m

        w_s_front = g.k_b * m_s_front
        w_z_front = g.k_b * m_z_front
        w_s_rear = g.k_b * m_s_rear - (g.k_a - g.k_b) * l_z
        w_z_rear = g.k_b * m_z_rear + (g.k_a - g.k_b) * l_s

        d_l_front = (w_s * w_s_front + w_z * w_z_front) / leg_length
        d_l_rear = (w_s * w_s_rear + w_z * w_z_rear) / leg_length
        d_phi_front = (w_z * w_s_front - w_s * w_z_front) / (leg_length * leg_length)
        d_phi_rear = (w_z * w_s_rear - w_s * w_z_rear) / (leg_length * leg_length)

        return ((d_l_front, d_l_rear), (d_phi_front, d_phi_rear)), delta_m


def rotate(angle: float, vector: Vector2) -> Vector2:
    c = math.cos(angle)
    s = math.sin(angle)
    x, z = vector
    return c * x - s * z, s * x + c * z


def ideal_fivebar_point_m(point_l: Vector2, length_lm: float, point_j: Vector2, length_jm: float, sigma_phi: int) -> tuple[Vector2, float]:
    """Return section 4.3's inner five-bar point ``M`` and passive angle ``phi2``.

    The point order follows the derivation: ``B=L``, ``D=J``, ``C=M``.
    ``sigma_phi=-1`` selects the physical assembly branch for the current CAD
    model.
    """

    if sigma_phi not in (-1, 1):
        raise ValueError("sigma_phi must be +1 or -1")

    delta_s = point_j[0] - point_l[0]
    delta_z = point_j[1] - point_l[1]
    rho = math.hypot(delta_s, delta_z)
    if rho <= 1.0e-12:
        raise KinematicsError("inner five-bar active endpoints coincide")
    if rho > length_lm + length_jm + 1.0e-12:
        raise KinematicsError("inner five-bar links are too far apart")
    if rho < abs(length_lm - length_jm) - 1.0e-12:
        raise KinematicsError("inner five-bar link triangle is invalid")

    a0 = 2.0 * length_lm * delta_s
    b0 = 2.0 * length_lm * delta_z
    c0 = length_lm * length_lm + rho * rho - length_jm * length_jm
    discriminant = a0 * a0 + b0 * b0 - c0 * c0
    if discriminant < -1.0e-12:
        raise KinematicsError("inner five-bar half-angle discriminant is negative")

    root = math.sqrt(max(0.0, discriminant))
    phi2 = wrap_to_pi(2.0 * math.atan2(b0 + float(sigma_phi) * root, a0 + c0))
    point_m = (
        point_l[0] + length_lm * math.cos(phi2),
        point_l[1] + length_lm * math.sin(phi2),
    )
    return point_m, phi2


def norm(vector: Vector2) -> float:
    return math.hypot(vector[0], vector[1])


def add(lhs: Vector2, rhs: Vector2) -> Vector2:
    return lhs[0] + rhs[0], lhs[1] + rhs[1]


def sub(lhs: Vector2, rhs: Vector2) -> Vector2:
    return lhs[0] - rhs[0], lhs[1] - rhs[1]


def scale(factor: float, vector: Vector2) -> Vector2:
    return factor * vector[0], factor * vector[1]


def wrap_to_pi(angle: float) -> float:
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped <= -math.pi:
        return wrapped + 2.0 * math.pi
    return wrapped
