from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .paths import asset_root, asset_root_v2


class AssetContractError(RuntimeError):
    """Raised when the frozen AssetBundleV1 identity does not match disk."""


@dataclass(frozen=True)
class AssetFileSpec:
    relative_path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class AssetBundleSpec:
    version: str
    files: tuple[AssetFileSpec, ...]
    entry_file: str
    default_prim: str
    articulation_root: str
    embedded_ground: str | None
    root_prims: tuple[str, ...]
    controlled_joints: tuple[str, ...]
    loop_joint_paths: tuple[str, ...]
    up_axis: str
    meters_per_unit: float
    rigid_body_count: int
    total_mass_kg: float
    dummy_body_count: int
    dummy_mass_kg: float
    tree_joint_count: int
    loop_joint_count: int

    @property
    def bundle_hash(self) -> str:
        payload = [
            {"path": item.relative_path, "size": item.size, "sha256": item.sha256}
            for item in self.files
        ]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(encoded).hexdigest().upper()


@dataclass(frozen=True)
class AssetFileReport:
    relative_path: str
    absolute_path: str
    expected_size: int
    actual_size: int | None
    expected_sha256: str
    actual_sha256: str | None
    exists: bool
    size_matches: bool
    sha256_matches: bool


@dataclass(frozen=True)
class AssetBundleReport:
    bundle_version: str
    root: str
    entry_file: str
    bundle_hash: str
    files: tuple[AssetFileReport, ...]


ASSET_BUNDLE_V1 = AssetBundleSpec(
    version="AssetBundleV1",
    files=(
        AssetFileSpec(
            "wheel_leg_urdf4.usd",
            18_571,
            "F371163FF638D3A151B303A202CE9EACD208E04751538FA1BCCBE4F9E5AA7F44",
        ),
        AssetFileSpec(
            "configuration/wheel_leg_urdf4_base.usd",
            35_535_228,
            "D980BE2205D0078AA7845AEE7C22B5C7F04FD5A05E79D327FC7E3ED28066C28E",
        ),
        AssetFileSpec(
            "configuration/wheel_leg_urdf4_physics.usd",
            5_371,
            "A8F67B6BDD1210D13AE4BD1254BE8BB87662312BC713046E5CB9478A8C0A2CE5",
        ),
        AssetFileSpec(
            "configuration/wheel_leg_urdf4_robot.usd",
            2_452,
            "81415C4FB375DAEC1A8DDFD0FE5CEAB6A1904FB77F29050149DED24C1F3436A1",
        ),
        AssetFileSpec(
            "configuration/wheel_leg_urdf4_sensor.usd",
            655,
            "D31E77E5C105470304DE4F5D8CAA9B3DB2A4D70ECC245510AD96C53F185A8314",
        ),
    ),
    entry_file="wheel_leg_urdf4.usd",
    default_prim="/wheel_leg_urdf4",
    articulation_root="/wheel_leg_urdf4/base_link",
    embedded_ground="/wheel_leg_urdf4/GroundPlane/CollisionPlane",
    root_prims=("/physicsScene", "/wheel_leg_urdf4", "/Render", "/World"),
    controlled_joints=("jIJ", "jIO", "jAB", "jAG", "jwheel_left", "jwheel_right"),
    loop_joint_paths=(
        "/wheel_leg_urdf4/jIO/jIO_loop_closure",
        "/wheel_leg_urdf4/jKN/jKN_loop_closure",
        "/wheel_leg_urdf4/jEC/jAG_loop_closure",
        "/wheel_leg_urdf4/jCF/jCF_revolute_joint",
    ),
    up_axis="Z",
    meters_per_unit=1.0,
    rigid_body_count=27,
    total_mass_kg=4.396253988146782,
    dummy_body_count=12,
    dummy_mass_kg=0.11999999731779099,
    tree_joint_count=26,
    loop_joint_count=4,
)

ASSET_BUNDLE_V2 = AssetBundleSpec(
    version="AssetBundleV2",
    files=(
        AssetFileSpec(
            "wheel_leg_urdf4.usd",
            19_268,
            "EFCA384A92B06CECE5ED2C5C0EC0A9077EE16D32D940473D3FFEFA8C378D4B0F",
        ),
        *ASSET_BUNDLE_V1.files[1:],
    ),
    entry_file=ASSET_BUNDLE_V1.entry_file,
    default_prim=ASSET_BUNDLE_V1.default_prim,
    articulation_root=ASSET_BUNDLE_V1.articulation_root,
    embedded_ground=None,
    root_prims=ASSET_BUNDLE_V1.root_prims,
    controlled_joints=ASSET_BUNDLE_V1.controlled_joints,
    loop_joint_paths=ASSET_BUNDLE_V1.loop_joint_paths,
    up_axis=ASSET_BUNDLE_V1.up_axis,
    meters_per_unit=ASSET_BUNDLE_V1.meters_per_unit,
    rigid_body_count=ASSET_BUNDLE_V1.rigid_body_count,
    total_mass_kg=ASSET_BUNDLE_V1.total_mass_kg,
    dummy_body_count=ASSET_BUNDLE_V1.dummy_body_count,
    dummy_mass_kg=ASSET_BUNDLE_V1.dummy_mass_kg,
    tree_joint_count=ASSET_BUNDLE_V1.tree_joint_count,
    loop_joint_count=ASSET_BUNDLE_V1.loop_joint_count,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_asset_bundle(
    root: Path | None = None,
    *,
    bundle: AssetBundleSpec = ASSET_BUNDLE_V1,
) -> AssetBundleReport:
    default_root = asset_root_v2() if bundle.version == ASSET_BUNDLE_V2.version else asset_root()
    resolved_root = (root or default_root).resolve()
    reports: list[AssetFileReport] = []
    errors: list[str] = []
    for spec in bundle.files:
        path = resolved_root / Path(spec.relative_path)
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_sha256 = _sha256(path) if exists else None
        size_matches = actual_size == spec.size
        sha256_matches = actual_sha256 == spec.sha256
        reports.append(
            AssetFileReport(
                relative_path=spec.relative_path,
                absolute_path=str(path),
                expected_size=spec.size,
                actual_size=actual_size,
                expected_sha256=spec.sha256,
                actual_sha256=actual_sha256,
                exists=exists,
                size_matches=size_matches,
                sha256_matches=sha256_matches,
            )
        )
        if not exists:
            errors.append(f"missing asset file: {path}")
        elif not size_matches or not sha256_matches:
            errors.append(f"asset identity mismatch: {path}")

    if errors:
        raise AssetContractError("; ".join(errors))

    return AssetBundleReport(
        bundle_version=bundle.version,
        root=str(resolved_root),
        entry_file=bundle.entry_file,
        bundle_hash=bundle.bundle_hash,
        files=tuple(reports),
    )
