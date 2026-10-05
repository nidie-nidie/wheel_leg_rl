from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path

import torch


VIRTUAL_LEG_KINEMATICS_VERSION = "VirtualLegKinematicsV1"
MIN_VIRTUAL_LEG_LENGTH_M = 0.05
MAX_LOOP_CLOSURE_ERROR_M = 5.0e-3
MAX_FK_WHEEL_ERROR_M = 5.0e-3
MAX_FK_LENGTH_ERROR_M = 5.0e-3
MAX_FK_PHI0_ERROR_RAD = math.radians(3.0)
DEFAULT_USD_ANCHOR_AUDIT = (
    Path(__file__).resolve().parents[4] / "artifacts" / "phase1_v4" / "usd-anchor-audit.json"
)


@dataclass(frozen=True)
class OffsetLegFkResult:
    wheel_vector_body: torch.Tensor
    length: torch.Tensor
    phi0: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class OffsetLegGeometry:
    v_ij: tuple[float, float]
    v_il: tuple[float, float]
    v_ip: tuple[float, float]
    v_jm: tuple[float, float]
    v_ml: tuple[float, float]
    v_mk: tuple[float, float]
    v_kn: tuple[float, float]
    v_pn: tuple[float, float]
    v_pw: tuple[float, float]


DEFAULT_OFFSET_LEG_GEOMETRY = OffsetLegGeometry(
    v_ij=(0.0967494, 0.00674587),
    v_il=(-0.0963553, 0.0109404),
    v_ip=(-0.213571, 0.0245218),
    v_jm=(-0.0979082, -0.060324),
    v_ml=(-0.0951965, 0.0645185),
    v_mk=(-0.132583, 0.116217),
    v_kn=(-0.117216, 0.0135814),
    v_pn=(-0.0373868, 0.0516985),
    v_pw=(0.213571, -0.144746),
)


def load_usd_anchor_audit(path: Path = DEFAULT_USD_ANCHOR_AUDIT) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Virtual-leg USD anchor audit is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != VIRTUAL_LEG_KINEMATICS_VERSION:
        raise ValueError(
            f"Expected {VIRTUAL_LEG_KINEMATICS_VERSION}, got {payload.get('schema_version')!r}"
        )
    if set(payload.get("legs", {})) != {"left", "right"}:
        raise ValueError("USD anchor audit must define exactly left and right legs")
    return payload


def build_virtual_leg_contract(path: Path = DEFAULT_USD_ANCHOR_AUDIT) -> dict:
    """Build a path-independent contract from the audited USD joint-axis geometry."""

    audit = load_usd_anchor_audit(path)

    def _anchor(payload: dict) -> dict:
        return {
            "joint_name": payload["joint_name"],
            "body_name": payload["body_name"],
            "side": int(payload["side"]),
            "local_anchor_m": [float(value) for value in payload["local_anchor_m"]],
            "axis_body": [float(value) for value in payload["axis_body"]],
        }

    legs = {}
    for side in ("left", "right"):
        leg = audit["legs"][side]
        legs[side] = {
            "hip": _anchor(leg["hip"]),
            "wheel": _anchor(leg["wheel"]),
            "hip_joint_names": list(leg["hip_joint_names"]),
            "wheel_joint_name": leg["wheel_joint_name"],
            "fk_point_joint_names": dict(leg["fk_geometry"]["point_joint_names"]),
            "fk_vectors_s_z_m": {
                name: [float(value) for value in vector]
                for name, vector in sorted(leg["fk_geometry"]["vectors_s_z_m"].items())
            },
        }

    loop_joints = []
    for joint in audit["loop_joints"]:
        loop_joints.append(
            {
                "joint_name": joint["joint_name"],
                "joint_path": joint["joint_path"],
                "sides": [
                    {
                        "body_name": side["body_name"],
                        "local_anchor_m": [float(value) for value in side["local_anchor_m"]],
                        "axis_body": [float(value) for value in side["axis_body"]],
                    }
                    for side in joint["sides"]
                ],
            }
        )

    contract = {
        "schema_version": VIRTUAL_LEG_KINEMATICS_VERSION,
        "asset_bundle_version": audit["asset_bundle_version"],
        "asset_bundle_hash": audit["asset_bundle_hash"],
        "coordinate_definition": dict(audit["coordinate_definition"]),
        "legs": legs,
        "loop_joints": loop_joints,
        "runtime_thresholds": {
            "min_virtual_leg_length_m": MIN_VIRTUAL_LEG_LENGTH_M,
            "max_loop_closure_error_m": MAX_LOOP_CLOSURE_ERROR_M,
            "max_fk_wheel_error_m": MAX_FK_WHEEL_ERROR_M,
            "max_fk_length_error_m": MAX_FK_LENGTH_ERROR_M,
            "max_fk_phi0_error_rad": MAX_FK_PHI0_ERROR_RAD,
        },
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    contract["semantic_sha256"] = hashlib.sha256(encoded).hexdigest().upper()
    return contract


def offset_geometry_from_audit(leg_payload: dict) -> OffsetLegGeometry:
    vectors = leg_payload.get("fk_geometry", {}).get("vectors_s_z_m")
    expected = {field.name for field in fields(OffsetLegGeometry)}
    if not isinstance(vectors, dict) or set(vectors) != expected:
        raise ValueError(f"USD anchor audit FK vectors must be {sorted(expected)}")
    return OffsetLegGeometry(**{name: tuple(float(value) for value in vectors[name]) for name in expected})


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi] without a discontinuous modulo branch."""

    return torch.atan2(torch.sin(angle), torch.cos(angle))


def virtual_leg_from_body_vector(vector_body: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return VMC ``(L0, phi0)`` from an I-to-W vector in the USD base frame.

    The audited leg plane uses ``s=+base Y`` and ``d=-base Z``. The lateral
    base-X component is intentionally excluded from the planar virtual leg.
    """

    if vector_body.shape[-1] != 3:
        raise ValueError(f"Expected body vectors [...,3], got {tuple(vector_body.shape)}")
    s_w = vector_body[..., 1]
    d_w = -vector_body[..., 2]
    length = torch.hypot(s_w, d_w)
    phi0 = torch.atan2(d_w, s_w)
    return length, phi0


def phi0_symmetry_error(phi0: torch.Tensor) -> torch.Tensor:
    if phi0.shape[-1] != 2:
        raise ValueError(f"Expected left/right phi0 [...,2], got {tuple(phi0.shape)}")
    return torch.square(wrap_to_pi(phi0[..., 0] - phi0[..., 1]))


def _rotate2(angle: torch.Tensor, vector: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    return cosine * vector[0] - sine * vector[1], sine * vector[0] + cosine * vector[1]


def _circle_intersection(
    center_a: torch.Tensor,
    radius_a: float,
    center_b: torch.Tensor,
    radius_b: float,
    branch: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = center_b - center_a
    distance = torch.linalg.vector_norm(delta, dim=-1)
    safe_distance = distance.clamp_min(1.0e-12)
    along = (radius_a * radius_a - radius_b * radius_b + distance.square()) / (2.0 * safe_distance)
    height_sq = radius_a * radius_a - along.square()
    unit = delta / safe_distance.unsqueeze(-1)
    perpendicular = torch.stack((-unit[..., 1], unit[..., 0]), dim=-1)
    point = (
        center_a
        + along.unsqueeze(-1) * unit
        + branch * torch.sqrt(height_sq.clamp_min(0.0)).unsqueeze(-1) * perpendicular
    )
    valid = (distance > 1.0e-12) & (height_sq >= -1.0e-10)
    return point, valid


def offset_leg_fk(
    active_joint_position: torch.Tensor,
    geometry: OffsetLegGeometry = DEFAULT_OFFSET_LEG_GEOMETRY,
) -> OffsetLegFkResult:
    """Diagnostic exact offset closed-chain FK for ``[q_front, q_rear]``.

    Reward truth comes from live USD joint-axis geometry. This compact FK is
    retained only as an independent encoder-side diagnostic for future
    real-hardware use.
    """

    if active_joint_position.shape[-1] != 2:
        raise ValueError(
            f"Expected active joint positions [...,2], got {tuple(active_joint_position.shape)}"
        )

    q_front = active_joint_position[..., 0]
    q_rear = active_joint_position[..., 1]
    j_s, j_z = _rotate2(q_front, geometry.v_ij)
    l_s, l_z = _rotate2(q_rear, geometry.v_il)
    p_s, p_z = _rotate2(q_rear, geometry.v_ip)
    point_j = torch.stack((j_s, j_z), dim=-1)
    point_l = torch.stack((l_s, l_z), dim=-1)
    point_p = torch.stack((p_s, p_z), dim=-1)
    point_m, valid_m = _circle_intersection(
        point_j,
        math.hypot(*geometry.v_jm),
        point_l,
        math.hypot(*geometry.v_ml),
        +1.0,
    )
    alpha = torch.atan2(point_l[..., 1] - point_m[..., 1], point_l[..., 0] - point_m[..., 0]) - math.atan2(
        geometry.v_ml[1], geometry.v_ml[0]
    )
    mk_s, mk_z = _rotate2(alpha, geometry.v_mk)
    point_k = point_m + torch.stack((mk_s, mk_z), dim=-1)
    point_n, valid_n = _circle_intersection(
        point_k,
        math.hypot(*geometry.v_kn),
        point_p,
        math.hypot(*geometry.v_pn),
        -1.0,
    )
    beta = torch.atan2(point_n[..., 1] - point_p[..., 1], point_n[..., 0] - point_p[..., 0]) - math.atan2(
        geometry.v_pn[1], geometry.v_pn[0]
    )
    pw_s, pw_z = _rotate2(beta, geometry.v_pw)
    w_s = p_s + pw_s
    w_z = p_z + pw_z
    wheel_vector = torch.stack((torch.zeros_like(w_s), w_s, w_z), dim=-1)
    length, phi0 = virtual_leg_from_body_vector(wheel_vector)
    valid = valid_m & valid_n & torch.isfinite(wheel_vector).all(dim=-1) & (length > MIN_VIRTUAL_LEG_LENGTH_M)
    return OffsetLegFkResult(wheel_vector, length, phi0, valid)
