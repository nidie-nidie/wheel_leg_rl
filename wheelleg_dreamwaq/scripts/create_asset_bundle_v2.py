from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import traceback
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = PROJECT_ROOT.parent / "wheel_leg_urdf4_usd_v2" / "wheel_leg_urdf4"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "phase1_v4"

parser = argparse.ArgumentParser(description="Create the immutable AssetBundleV2 copy without embedded ground.")
parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Sdf, Usd

from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V1, verify_asset_bundle
from wheelleg_dreamwaq.assets.paths import asset_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _relative_files(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in ASSET_BUNDLE_V1.files:
        path = root / spec.relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "relative_path": spec.relative_path,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _local_layer_path(identifier: str, bundle_root: Path) -> str:
    path = Path(identifier).resolve()
    try:
        return path.relative_to(bundle_root.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(f"GroundPlane is authored outside the copied bundle: {path}") from error


def _remove_ground(entry_path: Path, bundle_root: Path) -> dict[str, object]:
    stage = Usd.Stage.Open(str(entry_path))
    if stage is None:
        raise RuntimeError(f"Unable to open copied USD: {entry_path}")
    ground_path = Sdf.Path("/wheel_leg_urdf4/GroundPlane")
    prim = stage.GetPrimAtPath(ground_path)
    if not prim.IsValid():
        raise RuntimeError(f"Copied asset does not contain {ground_path}")

    stack = []
    defining_specs = []
    for spec in prim.GetPrimStack():
        row = {
            "layer": _local_layer_path(spec.layer.identifier, bundle_root),
            "path": str(spec.path),
            "specifier": str(spec.specifier),
        }
        stack.append(row)
        if spec.specifier == Sdf.SpecifierDef:
            defining_specs.append(spec)
    if len(defining_specs) != 1:
        raise RuntimeError(f"Expected one defining GroundPlane prim spec, found {stack}")

    defining_spec = defining_specs[0]
    layer = defining_spec.layer
    stage.SetEditTarget(Usd.EditTarget(layer))
    if not stage.RemovePrim(ground_path):
        raise RuntimeError(f"Failed to remove {ground_path} from {layer.identifier}")
    if not layer.Save():
        raise RuntimeError(f"Failed to save GroundPlane authoring layer: {layer.identifier}")

    reopened = Usd.Stage.Open(str(entry_path))
    if reopened is None:
        raise RuntimeError(f"Unable to reopen modified USD: {entry_path}")
    forbidden = (
        "/wheel_leg_urdf4/GroundPlane",
        "/wheel_leg_urdf4/GroundPlane/CollisionPlane",
    )
    surviving = [path for path in forbidden if reopened.GetPrimAtPath(path).IsValid()]
    if surviving:
        raise RuntimeError(f"GroundPlane prims survived source-layer deletion: {surviving}")
    return {
        "prim_path": str(ground_path),
        "prim_stack_before": stack,
        "authoring_layer": _local_layer_path(layer.identifier, bundle_root),
        "forbidden_paths_after": surviving,
    }


def main() -> None:
    source_root = asset_root().resolve()
    destination = args_cli.destination.resolve()
    expected_parent = (PROJECT_ROOT.parent / "wheel_leg_urdf4_usd_v2").resolve()
    if destination.parent != expected_parent or destination.name != "wheel_leg_urdf4":
        raise RuntimeError(f"Refusing unexpected AssetBundleV2 destination: {destination}")
    if destination.exists():
        raise FileExistsError(f"AssetBundleV2 destination already exists: {destination}")

    before = verify_asset_bundle(source_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, destination)
    ground_audit = _remove_ground(destination / ASSET_BUNDLE_V1.entry_file, destination)
    files = _relative_files(destination)
    after = verify_asset_bundle(source_root)
    if before.bundle_hash != after.bundle_hash:
        raise RuntimeError("AssetBundleV1 changed while creating AssetBundleV2")

    args_cli.artifact_dir.mkdir(parents=True, exist_ok=True)
    ground_audit_path = args_cli.artifact_dir / "groundplane-layer-audit.json"
    manifest_path = args_cli.artifact_dir / "asset-bundle-v2-manifest.json"
    ground_audit_path.write_text(json.dumps(ground_audit, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "version": "AssetBundleV2",
        "source_bundle_version": ASSET_BUNDLE_V1.version,
        "source_bundle_hash": ASSET_BUNDLE_V1.bundle_hash,
        "root": str(destination),
        "entry_file": ASSET_BUNDLE_V1.entry_file,
        "default_prim": ASSET_BUNDLE_V1.default_prim,
        "articulation_root": ASSET_BUNDLE_V1.articulation_root,
        "embedded_ground": None,
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print(f"[INFO] AssetBundleV2 created at: {destination}", flush=True)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        if exit_code == 0:
            simulation_app.close(skip_cleanup=True)
        else:
            os._exit(exit_code)
    raise SystemExit(exit_code)
