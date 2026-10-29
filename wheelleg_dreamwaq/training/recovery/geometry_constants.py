from pathlib import Path

import numpy as np

from .contracts import file_sha256, stable_hash, write_json


SCHEMA = "RecoveryCompiledPlaneGeometryV1"
BODY_MAPPING = "GeometryBodyMapV1_base_link_to_base"
NATIVE_SUPPORT = "MuJoCo_3.14.0_PlaneConvex_hillclimb_and_mesh_polygon_vertices"


def save_compiled_geometry(inspector, directory):
    import mujoco

    model, data = inspector.model, inspector.data
    floor, geoms = inspector.floor, inspector.geoms
    if (mujoco.__version__ != "3.14.0"
            or model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE
            or any(model.geom_type[index] != mujoco.mjtGeom.mjGEOM_MESH for index in geoms)
            or np.any(model.geom_margin[[floor, *geoms]] != 0)
            or np.any(model.geom_gap[[floor, *geoms]] != 0)):
        raise ValueError("GPU平面公式只接受已验证的零margin/gap平面与编译mesh")
    arrays = {"geom_pos": model.geom_pos[geoms].copy(),
              "local_rotations": np.asarray(inspector.local_rotations),
              "plane_position": data.geom_xpos[floor].copy(),
              "plane_normal": data.geom_xmat[floor].reshape(3, 3)[:, 2].copy()}
    counts = []
    for order, geom in enumerate(geoms):
        mesh = int(model.geom_dataid[geom])
        graph_address = int(model.mesh_graphadr[mesh])
        if graph_address < 0:
            raise ValueError("GPU平面公式需要当前MuJoCo编译的实际凸包")
        graph = model.mesh_graph[graph_address:]
        count = int(graph[0])
        indices = graph[2 + count:2 + 2 * count]
        start = int(model.mesh_vertadr[mesh])
        vertices = model.mesh_vert[start:start + int(model.mesh_vertnum[mesh])]
        if count < 4 or len(vertices) < 10 or np.any(indices < 0) or np.any(indices >= len(vertices)):
            raise ValueError("编译凸包索引无效")
        polygon_start = int(model.mesh_polyadr[mesh])
        polygon_count = int(model.mesh_polynum[mesh])
        native_vertices = set(int(index) for index in indices)
        for polygon in range(polygon_start, polygon_start + polygon_count):
            start_poly = int(model.mesh_polyvertadr[polygon])
            count_poly = int(model.mesh_polyvertnum[polygon])
            if not set(int(index) for index in model.mesh_polyvert[start_poly:start_poly + count_poly]) <= native_vertices:
                raise ValueError("原生PlaneConvex面顶点没有被GPU支持点集合覆盖")
        arrays[f"hull_{order}"] = vertices[indices].astype(np.float64)
        counts.append(count)
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    array_path = destination / "compiled_geometry.npz"
    np.savez(array_path, **arrays)
    record = {"schema_version": SCHEMA, "mujoco_version": mujoco.__version__,
              "native_support_selection": NATIVE_SUPPORT,
              "body_mapping_version": BODY_MAPPING, "body_names": inspector.body_names,
              "geom_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) for index in geoms],
              "hull_vertex_counts": counts, "arrays_file": str(array_path.resolve()),
              "arrays_sha256": file_sha256(array_path),
              "contact_manifest_sha256": file_sha256(inspector.config["contact_manifest"]),
              "geometry_representation": "MuJoCo_convex_hull_per_source_mesh", "margin_m": 0.0, "gap_m": 0.0}
    record["identity_sha256"] = stable_hash({key: value for key, value in record.items() if key != "arrays_file"})
    write_json(destination / "compiled_geometry.json", record)
    return record
