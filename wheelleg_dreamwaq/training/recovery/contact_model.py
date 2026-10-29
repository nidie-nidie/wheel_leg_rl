from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_XML = PROJECT_ROOT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def payload_sha256(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest().upper()


def build_contact_model(output_dir: str | Path, *, friction: float = 1.0, source_xml: str | Path = SOURCE_XML) -> Path:
    import mujoco

    if not np.isfinite(friction) or friction <= 0:
        raise ValueError("对地滑动摩擦系数必须为正有限数值")
    source = Path(source_xml).resolve()
    output = Path(output_dir).resolve()
    if output == source.parent:
        raise ValueError("完整接触模型必须写入独立目录")
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "wheelleg_recovery_complete_contact_v1")
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("源模型缺少 compiler")
    mesh_root = (source.parent / compiler.get("meshdir", ".")).resolve()
    compiler.set("meshdir", str(mesh_root))
    meshes = []
    for mesh in root.findall("./asset/mesh"):
        path = mesh_root / mesh.attrib["file"]
        meshes.append({"name": mesh.attrib["name"], "path": str(path), "sha256": file_sha256(path)})
    contact = root.find("contact")
    if contact is not None:
        root.remove(contact)
    contact = ET.SubElement(root, "contact")
    floor = root.find("./worldbody/geom[@name='floor']")
    if floor is None:
        raise ValueError("源模型缺少命名地面")
    floor.set("contype", "1")
    floor.set("conaffinity", "2")
    shapes = []
    for body in root.findall("./worldbody/body//body") + root.findall("./worldbody/body"):
        for index, geom in enumerate(body.findall("geom")):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            if geom.get("type") != "mesh":
                continue
            name = "recovery_" + body.attrib["name"] + "_" + str(index)
            geom.set("name", name)
            geom.set("contype", "2")
            geom.set("conaffinity", "1")
            geom.set("group", "1")
            geom.set("friction", f"{friction:.17g} 0 0")
            ET.SubElement(contact, "pair", {
                "name": "floor_" + name, "geom1": "floor", "geom2": name,
                "condim": "3", "friction": f"{friction:.17g} {friction:.17g} 0 0 0",
                "solref": "0.004 1", "solimp": "0.95 0.99 0.001 0.5 2",
            })
            shapes.append({"name": name, "body": body.attrib["name"], "mesh": geom.attrib["mesh"]})
    if len(shapes) != len(meshes):
        raise ValueError("源网格与真实对地形状数量不同，需核查资产映射")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "complete_contact.xml"
    if destination.resolve() == source:
        raise ValueError("禁止覆盖源模型")
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(destination))
    assert_contact_filter(model)
    record = {
        "schema_version": "RecoveryContactModelV1", "source_xml": str(source),
        "source_xml_sha256": file_sha256(source), "model_xml": str(destination),
        "model_xml_sha256": file_sha256(destination), "meshes": meshes,
        "contact_shapes": shapes, "friction": friction,
        "geometry_representation": "MuJoCo_convex_hull_per_source_mesh",
        "geometry_validation": "required", "internal_contacts_enabled": False,
        "contact_parameters": {"condim": 3, "torsional_friction": 0.0, "rolling_friction": 0.0,
                               "solref": [0.004, 1.0], "solimp": [0.95, 0.99, 0.001, 0.5, 2.0]},
    }
    (output / "contact_manifest.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    return destination


def load_contact_model(manifest_path: str | Path) -> tuple[mujoco.MjModel, dict]:
    import mujoco

    record = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if record["schema_version"] != "RecoveryContactModelV1":
        raise ValueError("完整接触模型版本不支持")
    for path, expected in [(record["source_xml"], record["source_xml_sha256"]),
                           (record["model_xml"], record["model_xml_sha256"])]:
        if file_sha256(path) != expected:
            raise ValueError(f"接触模型源身份发生变化：{path}")
    for mesh in record["meshes"]:
        if file_sha256(mesh["path"]) != mesh["sha256"]:
            raise ValueError(f"网格身份发生变化：{mesh['name']}")
    model = mujoco.MjModel.from_xml_path(record["model_xml"])
    assert_contact_filter(model)
    return model, record


def assert_contact_filter(model: mujoco.MjModel) -> None:
    import mujoco

    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor < 0 or model.geom_contype[floor] != 1 or model.geom_conaffinity[floor] != 2:
        raise ValueError("地面接触掩码无效")
    ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] > 0 and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH]
    if len(ids) != 17:
        raise ValueError(f"真实网格数量无效：{len(ids)}")
    for geom_id in ids:
        if model.geom_contype[geom_id] != 2 or model.geom_conaffinity[geom_id] != 1:
            raise ValueError("真实部件未启用地面接触")
    for geom_id in range(model.ngeom):
        if geom_id != floor and geom_id not in ids and (model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0):
            raise ValueError("代理形状意外启用接触")
    for first, second in itertools.combinations(ids, 2):
        if (model.geom_contype[first] & model.geom_conaffinity[second]) or (model.geom_contype[second] & model.geom_conaffinity[first]):
            raise ValueError("检测到内部接触掩码")
    for first, second in zip(model.pair_geom1, model.pair_geom2):
        if floor not in (first, second):
            raise ValueError("显式接触对包含机器人内部部件")
    if model.npair != len(ids):
        raise ValueError("实际对地接触对未覆盖每个真实网格")
    if {int(second if first == floor else first) for first, second in zip(model.pair_geom1, model.pair_geom2)} != set(ids):
        raise ValueError("对地接触对的真实形状覆盖无效")


class GeometryAudit:
    def __init__(self, model: mujoco.MjModel, allowed_body_pairs: list[list[str]], distance_limit_m: float = 0.05):
        import mujoco

        self.model = model
        self.distance_limit_m = float(distance_limit_m)
        if self.distance_limit_m <= 0:
            raise ValueError("间隙查询范围必须大于零")
        allowed = {frozenset(pair) for pair in allowed_body_pairs}
        names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)}
        if any(len(pair) != 2 or not set(pair) <= names for pair in allowed_body_pairs):
            raise ValueError("允许部件关系包含未知部件")
        ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] > 0 and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH]
        self.pairs = []
        for first, second in itertools.combinations(ids, 2):
            first_body, second_body = int(model.geom_bodyid[first]), int(model.geom_bodyid[second])
            body_names = frozenset(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in (first_body, second_body))
            if first_body != second_body and body_names not in allowed:
                self.pairs.append((first, second))

    def inspect(self, data: mujoco.MjData) -> dict:
        import mujoco

        minimum = self.distance_limit_m
        closest = None
        for first, second in self.pairs:
            distance = float(mujoco.mj_geomDistance(self.model, data, first, second, self.distance_limit_m, None))
            if distance < minimum:
                minimum = distance
                closest = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in (first, second)]
        return {"minimum_nonallowed_clearance_m": minimum, "closest_pair": closest, "checked_pair_count": len(self.pairs),
                "geometry_representation": "MuJoCo_convex_hull_per_source_mesh"}


def configure_isaac_full_contact(stage, robot_prim_path: str) -> dict:
    from pxr import PhysxSchema, Usd, UsdPhysics

    robot = stage.GetPrimAtPath(robot_prim_path)
    if not robot.IsValid():
        raise ValueError(f"机器人 prim 不存在：{robot_prim_path}")
    enabled = []
    roots = []
    for prim in Usd.PrimRange(robot):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            path = str(prim.GetPath())
            if "/GroundPlane" in path:
                raise ValueError("机器人资产中包含嵌入地面")
            api = UsdPhysics.CollisionAPI(prim)
            api.CreateCollisionEnabledAttr().Set(True)
            enabled.append(path)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr().Set(False)
            roots.append(str(prim.GetPath()))
    if not roots:
        raise ValueError("机器人资产缺少 articulation root")
    result = assert_isaac_full_contact(stage, robot_prim_path)
    return {"schema_version": "RecoveryIsaacFullContactV1", "collision_paths": sorted(enabled),
            "articulation_roots": sorted(roots), "self_collision_enabled": False,
            "validation": result}


def assert_isaac_full_contact(stage, robot_prim_path: str) -> dict:
    from pxr import PhysxSchema, Usd, UsdPhysics

    robot = stage.GetPrimAtPath(robot_prim_path)
    if not robot.IsValid():
        raise ValueError("机器人 prim 不存在")
    paths = []
    roots = []
    for prim in Usd.PrimRange(robot):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True:
                raise ValueError(f"机器人对地形状未启用：{prim.GetPath()}")
            paths.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            if PhysxSchema.PhysxArticulationAPI(prim).GetEnabledSelfCollisionsAttr().Get() is not False:
                raise ValueError("起身机器人自碰撞配置无效")
            roots.append(str(prim.GetPath()))
    if len(paths) < 3 or not any("jwheel_left/" in path for path in paths) or not any("jwheel_right/" in path for path in paths):
        raise ValueError("完整接触形状缺少双轮或非轮部件")
    if not roots:
        raise ValueError("机器人资产缺少 articulation root")
    return {"collision_paths": sorted(paths), "articulation_roots": sorted(roots), "self_collision_enabled": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--friction", type=float, default=1.0)
    args = parser.parse_args()
    print(build_contact_model(args.output_dir, friction=args.friction))


if __name__ == "__main__":
    main()
