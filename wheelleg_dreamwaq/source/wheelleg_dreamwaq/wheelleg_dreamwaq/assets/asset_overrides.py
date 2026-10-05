from __future__ import annotations


RUNTIME_COLLISION_POLICY_VERSION = "WheelOnlyCollisionV2"


def embedded_ground_path(robot_prim_path: str) -> str:
    return f"{robot_prim_path.rstrip('/')}/GroundPlane/CollisionPlane"


def assert_embedded_ground_absent(stage, robot_prim_path: str) -> None:
    root = robot_prim_path.rstrip("/")
    forbidden = (f"{root}/GroundPlane", f"{root}/GroundPlane/CollisionPlane")
    surviving = [path for path in forbidden if stage.GetPrimAtPath(path).IsValid()]
    if surviving:
        raise RuntimeError(f"AssetBundleV2 contains forbidden embedded ground prims: {surviving}")


def is_wheel_collision_path(path: str, robot_prim_path: str) -> bool:
    root = robot_prim_path.rstrip("/")
    return path.startswith(f"{root}/jwheel_left/") or path.startswith(f"{root}/jwheel_right/")


def disable_embedded_ground_collision(stage, robot_prim_path: str) -> str:
    """Author a session-layer collision override before physics initialization."""

    from pxr import UsdPhysics

    path = embedded_ground_path(robot_prim_path)
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Embedded ground prim is missing after reference load: {path}")
    collision_api = UsdPhysics.CollisionAPI(prim)
    if not collision_api:
        raise RuntimeError(f"Embedded ground prim has no CollisionAPI: {path}")
    collision_api.GetCollisionEnabledAttr().Set(False)
    if collision_api.GetCollisionEnabledAttr().Get() is not False:
        raise RuntimeError(f"Failed to disable embedded ground collision: {path}")
    return path


def configure_wheel_only_collisions(stage, robot_prim_path: str) -> tuple[list[str], list[str]]:
    """Keep wheel collision shapes enabled and disable every other robot collision shape."""

    from pxr import Usd, UsdPhysics

    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f"Robot prim is missing after reference load: {robot_prim_path}")

    disabled: list[str] = []
    kept: list[str] = []
    for prim in Usd.PrimRange(robot_prim):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        collision_api = UsdPhysics.CollisionAPI(prim)
        if is_wheel_collision_path(path, robot_prim_path):
            collision_api.GetCollisionEnabledAttr().Set(True)
            kept.append(path)
        else:
            collision_api.GetCollisionEnabledAttr().Set(False)
            disabled.append(path)

    if len(kept) != 2:
        raise RuntimeError(f"Expected exactly two wheel collision shapes, found {kept}")
    if not disabled:
        raise RuntimeError("No non-wheel robot collision shapes were disabled")
    return sorted(disabled), sorted(kept)
