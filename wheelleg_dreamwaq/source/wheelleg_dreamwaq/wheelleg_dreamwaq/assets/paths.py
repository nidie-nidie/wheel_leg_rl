from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ASSET_ROOT_V1 = PROJECT_ROOT.parent / "wheel_leg_urdf4_usd (1)" / "wheel_leg_urdf4"
DEFAULT_ASSET_ROOT_V2 = PROJECT_ROOT.parent / "wheel_leg_urdf4_usd_v2" / "wheel_leg_urdf4"


def asset_root() -> Path:
    """Resolve the local AssetBundleV1 root without changing its contents."""

    override = os.environ.get("WHEELLEG_ASSET_ROOT")
    return Path(override).expanduser().resolve() if override else DEFAULT_ASSET_ROOT_V1.resolve()


def asset_root_v2() -> Path:
    """Resolve the generated AssetBundleV2 root."""

    override = os.environ.get("WHEELLEG_ASSET_ROOT_V2")
    return Path(override).expanduser().resolve() if override else DEFAULT_ASSET_ROOT_V2.resolve()
