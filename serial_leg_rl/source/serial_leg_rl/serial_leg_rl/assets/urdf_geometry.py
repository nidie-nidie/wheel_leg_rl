"""Extract kinematics constants from the source wheel-leg URDF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

from serial_leg_rl.kinematics.offset_leg import OffsetLegGeometry, norm


DEFAULT_SOURCE_URDF = Path(__file__).resolve().parents[5] / "wheel_leg_urdf4" / "urdf" / "wheel_leg_urdf4.urdf"


@dataclass(frozen=True)
class ExtractedLegGeometry:
    """Kinematic vectors extracted from one side of the URDF."""

    side: str
    v_ij: tuple[float, float]
    v_il: tuple[float, float]
    v_ip: tuple[float, float]
    v_jm: tuple[float, float]
    v_ml: tuple[float, float]
    v_pw: tuple[float, float]

    @property
    def offset_geometry(self) -> OffsetLegGeometry:
        return OffsetLegGeometry(
            v_ij=self.v_ij,
            v_il=self.v_il,
            v_ip=self.v_ip,
            l_jm=norm(self.v_jm),
            l_lm=norm(self.v_ml),
            l_pw=norm(self.v_pw),
        )


def extract_left_geometry(urdf_path: Path = DEFAULT_SOURCE_URDF) -> ExtractedLegGeometry:
    """Extract left leg geometry in the local ``(s, z)`` plane."""

    origins = _joint_origins(urdf_path)
    return ExtractedLegGeometry(
        side="left",
        v_ij=_xz(origins["jJM"]),
        v_il=_xz(origins["jIO_dummy_child_link1"]),
        v_ip=_xz(origins["jOP"]),
        v_jm=_xz(origins["jMK"]),
        v_ml=_xz(origins["jMK_dummy_child1"]),
        v_pw=_xz(origins["jwheel_left"]),
    )


def extract_right_geometry(urdf_path: Path = DEFAULT_SOURCE_URDF) -> ExtractedLegGeometry:
    """Extract right leg geometry in the local ``(s, z)`` plane."""

    origins = _joint_origins(urdf_path)
    return ExtractedLegGeometry(
        side="right",
        v_ij=_xz(origins["jBE"]),
        v_il=_xz(origins["jAG_dummy_child_link1"]),
        v_ip=_xz(origins["jGH"]),
        v_jm=_xz(origins["jEC"]),
        v_ml=_xz(origins["jEC_dummy_child_link1"]),
        v_pw=_xz(origins["jwheel_right"]),
    )


def compare_geometry(reference: OffsetLegGeometry, extracted: OffsetLegGeometry) -> dict[str, float]:
    """Return absolute differences between reference and extracted constants."""

    return {
        "v_ij_s": abs(reference.v_ij[0] - extracted.v_ij[0]),
        "v_ij_z": abs(reference.v_ij[1] - extracted.v_ij[1]),
        "v_il_s": abs(reference.v_il[0] - extracted.v_il[0]),
        "v_il_z": abs(reference.v_il[1] - extracted.v_il[1]),
        "v_ip_s_raw": abs(reference.v_ip[0] - extracted.v_ip[0]),
        "v_ip_z_raw": abs(reference.v_ip[1] - extracted.v_ip[1]),
        "l_jm": abs(reference.l_jm - extracted.l_jm),
        "l_lm": abs(reference.l_lm - extracted.l_lm),
        "l_pw": abs(reference.l_pw - extracted.l_pw),
        "k_a": abs(reference.k_a - extracted.k_a),
        "k_b": abs(reference.k_b - extracted.k_b),
    }


def _joint_origins(urdf_path: Path) -> dict[str, tuple[float, float, float]]:
    root = ET.parse(urdf_path).getroot()
    result = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        if origin is None:
            continue
        xyz = tuple(float(value) for value in origin.attrib["xyz"].split())
        if len(xyz) != 3:
            raise ValueError(f"joint {joint.attrib['name']} has invalid xyz")
        result[joint.attrib["name"]] = xyz
    return result


def _xz(xyz: tuple[float, float, float]) -> tuple[float, float]:
    # The exported joint frames are rotated so local x maps to leg-plane s.
    return xyz[0], xyz[2]

