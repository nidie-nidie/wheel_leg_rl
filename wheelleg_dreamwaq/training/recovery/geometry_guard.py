from __future__ import annotations

import argparse
import json
from multiprocessing import shared_memory
import os
from pathlib import Path
import sys
import time

import numpy as np

from .contracts import PROJECT_ROOT, file_sha256, stable_hash
from .geometry_binary import (CAPACITY, FAILURE_CODES, INPUT_BYTES, MAX_BODIES, PROTOCOL,
                              RESULT_DTYPE, SLOT_BYTES, SLOTS, GeometryBinaryClient)
from .geometry_constants import save_compiled_geometry


def validate_guard_config(config: dict, *, check_paths: bool = True) -> None:
    fields = {"python_executable", "contact_manifest", "allowed_body_pairs",
              "maximum_ground_penetration_m", "minimum_clearance_m", "distance_limit_m", "internal_clearance_mode",
              "diagnostic_envs_per_substep"}
    if set(config) != fields:
        raise ValueError("逐子步几何检查配置字段不完整")
    limits = [config[key] for key in ("maximum_ground_penetration_m", "minimum_clearance_m", "distance_limit_m")]
    if not np.isfinite(limits).all() or min(limits[:2]) < 0 or limits[2] <= max(limits[:2]):
        raise ValueError("几何查询范围必须覆盖非负间隙及允许压缩量")
    if not isinstance(config["allowed_body_pairs"], list):
        raise ValueError("必须显式提供已验收的相邻部件关系")
    if config["internal_clearance_mode"] not in ("strict", "diagnostic"):
        raise ValueError("内部间隙处理必须显式选择strict或diagnostic")
    if not isinstance(config["diagnostic_envs_per_substep"], int) or config["diagnostic_envs_per_substep"] < 1:
        raise ValueError("内部间隙诊断采样数量必须为正整数")
    if check_paths:
        for key in ("python_executable", "contact_manifest"):
            if not Path(config[key]).is_file():
                raise FileNotFoundError(config[key])


class GeometryInspector:
    def __init__(self, config: dict):
        import mujoco

        from .contact_model import GeometryAudit, load_contact_model

        validate_guard_config(config)
        self.config = config
        self.diagnostic_offset = 0
        self.model, self.manifest = load_contact_model(config["contact_manifest"])
        self.data = mujoco.MjData(self.model)
        self.audit = GeometryAudit(self.model, config["allowed_body_pairs"], config["distance_limit_m"])
        self.geoms = [index for index in range(self.model.ngeom) if self.model.geom_contype[index] == 2]
        self.floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if self.floor < 0 or not self.audit.pairs:
            raise ValueError("几何检查需要地面和非允许部件查询对")
        self.body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[index]))
                           for index in self.geoms]
        self.local_rotations = []
        for index in self.geoms:
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, self.model.geom_quat[index])
            self.local_rotations.append(matrix.reshape(3, 3))
        mujoco.mj_forward(self.model, self.data)

    def inspect(self, body_names: list[str], positions, quaternions, *, kind="full", internal_indices=None) -> list[dict]:
        import mujoco

        positions, quaternions = np.asarray(positions, dtype=np.float64), np.asarray(quaternions, dtype=np.float64)
        if (len(set(body_names)) != len(body_names) or positions.ndim != 3
                or positions.shape[1:] != (len(body_names), 3)
                or quaternions.shape != (*positions.shape[:2], 4)
                or not np.isfinite(positions).all() or not np.isfinite(quaternions).all()
                or not np.allclose(np.linalg.norm(quaternions, axis=-1), 1, atol=1e-5)):
            raise ValueError("几何检查必须使用全部真实刚体的有限位姿和单位四元数")
        # 同源资产的USD根名为base_link，MJCF根名为base，参考系保持原生刚体坐标。
        mapped = [name if name in body_names else {"base": "base_link"}.get(name, name) for name in self.body_names]
        if not set(mapped) <= set(body_names):
            raise ValueError(f"Isaac刚体缺少同源几何: {set(mapped)-set(body_names)}")
        indices = [body_names.index(name) for name in mapped]
        quats = quaternions[:, indices]
        w, x, y, z = np.moveaxis(quats, -1, 0)
        rotations = np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                              2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                              2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape(*quats.shape[:2], 3, 3)
        world_positions = positions[:, indices] + np.einsum("ngij,gj->ngi", rotations, self.model.geom_pos[self.geoms])
        world_rotations = rotations @ np.asarray(self.local_rotations)
        if kind not in ("full", "floor", "internal", "validate"):
            raise ValueError("几何检查消息类型无效")
        if kind in ("floor", "validate"):
            checked_envs = set()
        elif kind == "internal":
            checked_envs = set(range(len(positions)))
        elif internal_indices is not None:
            checked_envs = set(internal_indices)
            if not checked_envs <= set(range(len(positions))):
                raise ValueError("内部检查索引越界")
        else:
            count = len(positions) if self.config["internal_clearance_mode"] == "strict" else min(
                len(positions), self.config["diagnostic_envs_per_substep"])
            checked_envs = {(self.diagnostic_offset + index) % len(positions) for index in range(count)}
            self.diagnostic_offset = (self.diagnostic_offset + count) % len(positions)
        reports = []
        for env in range(len(positions)):
            self.data.geom_xpos[self.geoms] = world_positions[env]
            self.data.geom_xmat[self.geoms] = world_rotations[env].reshape(-1, 9)
            clearance = (self.audit.inspect(self.data) if env in checked_envs else
                {"minimum_nonallowed_clearance_m": None, "closest_pair": None, "checked_pair_count": 0,
                 "geometry_representation": "MuJoCo_convex_hull_per_source_mesh", "observation": "not_sampled"})
            signed_floor = (min(float(mujoco.mj_geomDistance(self.model, self.data, self.floor, geom,
                               self.config["distance_limit_m"], None)) for geom in self.geoms)
                            if kind in ("floor", "full") else 0.0)
            penetration = max(0.0, -signed_floor)
            reason = None
            distance = clearance["minimum_nonallowed_clearance_m"]
            violation = None if distance is None else distance < self.config["minimum_clearance_m"]
            if violation and self.config["internal_clearance_mode"] == "strict":
                reason = "nonallowed_body_clearance"
            if penetration > self.config["maximum_ground_penetration_m"]:
                reason = "excessive_ground_penetration"
            reports.append({"failure_reason": reason, "maximum_ground_penetration_m": penetration,
                            "clearance": clearance, "internal_clearance_violation": violation,
                            "internal_clearance_mode": self.config["internal_clearance_mode"]})
        return reports


class GeometryGuardClient:
    def __init__(self, config: dict, run_dir: str | Path):
        validate_guard_config(config)
        self.config = config
        self.sequence = 0
        self.diagnostic_offset = 0
        self.transport = GeometryBinaryClient(config, run_dir)
        self.process = self.transport.process

    @property
    def identity(self) -> dict:
        return {"config_sha256": stable_hash(self.config),
                "contact_manifest_sha256": file_sha256(self.config["contact_manifest"]),
                "body_mapping_version": "GeometryBodyMapV1_base_link_to_base",
                "geometry_representation": "MuJoCo_convex_hull_per_source_mesh",
                "transport_version": PROTOCOL, "compiled_geometry_sha256": self.transport.constants["identity_sha256"]}

    def inspect(self, body_names, positions, quaternions) -> list[dict]:
        positions = np.asarray(positions, dtype=np.float64)
        quaternions = np.asarray(quaternions, dtype=np.float64)
        self.sequence += 1
        count = len(positions) if self.config["internal_clearance_mode"] == "strict" else min(
            len(positions), self.config["diagnostic_envs_per_substep"])
        selected = {(self.diagnostic_offset + index) % len(positions) for index in range(count)}
        self.diagnostic_offset = (self.diagnostic_offset + count) % len(positions)
        reports = []
        for start in range(0, len(positions), CAPACITY):
            stop = min(start + CAPACITY, len(positions))
            ticket = self.transport.submit("full", body_names, positions[start:stop], quaternions[start:stop],
                metadata={"internal_indices": [index - start for index in selected if start <= index < stop]})
            values = self.transport.collect(ticket)["values"]
            reports.extend(decode_reports(values, self.config, self.transport.constants["geom_names"]))
        return reports

    def close(self):
        self.transport.close()


def decode_reports(values, config, geom_names):
    reasons = {code: name for name, code in FAILURE_CODES.items()}
    reports = []
    for row in values:
        sampled = int(row["checked_pairs"]) > 0
        pair = ([geom_names[int(row["first_geom"])], geom_names[int(row["second_geom"])]]
                if row["first_geom"] >= 0 else None)
        clearance = {"minimum_nonallowed_clearance_m": float(row["clearance"]) if sampled else None,
                     "closest_pair": pair, "checked_pair_count": int(row["checked_pairs"]),
                     "geometry_representation": "MuJoCo_convex_hull_per_source_mesh"}
        if not sampled:
            clearance["observation"] = "not_sampled"
        reports.append({"failure_reason": reasons[int(row["failure"])],
                        "maximum_ground_penetration_m": float(row["penetration"]), "clearance": clearance,
                        "internal_clearance_violation": bool(row["violation"]) if sampled else None,
                        "internal_clearance_mode": config["internal_clearance_mode"]})
    return reports


def worker() -> None:
    inspector = None
    sequence = 0
    segments = []
    for line in sys.stdin:
        request = json.loads(line)
        if request["kind"] == "initialize_binary" and inspector is None:
            if (request["protocol"] != PROTOCOL or request["slot_bytes"] != SLOT_BYTES
                    or len(request["segments"]) != SLOTS):
                raise ValueError("几何二进制协议版本或槽位容量无效")
            inspector = GeometryInspector(request["config"])
            segments = [shared_memory.SharedMemory(name=name) for name in request["segments"]]
            constants = save_compiled_geometry(inspector, request["directory"])
            result = {"status": "ok", "sequence": 0, "constants": constants, "worker_pid": os.getpid()}
        elif request["kind"] in ("full", "floor", "internal", "validate") and inspector is not None and request["sequence"] == sequence + 1:
            sequence += 1
            count, bodies, slot = request["count"], len(request["body_names"]), request["slot"]
            dtype = np.dtype(request["dtype"])
            if (not 0 < count <= CAPACITY or not 0 < bodies <= MAX_BODIES or not 0 <= slot < SLOTS
                    or dtype not in (np.dtype("float32"), np.dtype("float64"))):
                raise ValueError("几何请求超出固定二进制槽位")
            start = time.perf_counter()
            inputs = np.ndarray((count, bodies, 7), dtype=dtype, buffer=segments[slot].buf)
            reports = inspector.inspect(request["body_names"], inputs[..., :3], inputs[..., 3:],
                kind=request["kind"], internal_indices=request["metadata"].get("internal_indices"))
            output = np.ndarray((count,), dtype=RESULT_DTYPE, buffer=segments[slot].buf, offset=INPUT_BYTES)
            output[:] = np.zeros(count, dtype=RESULT_DTYPE)
            geom_ids = {name: index for index, name in enumerate(constants["geom_names"])}
            for index, report in enumerate(reports):
                clearance = report["clearance"]
                output[index]["penetration"] = report["maximum_ground_penetration_m"]
                output[index]["clearance"] = np.nan if clearance["minimum_nonallowed_clearance_m"] is None else clearance["minimum_nonallowed_clearance_m"]
                pair = clearance["closest_pair"]
                output[index]["first_geom"] = -1 if pair is None else geom_ids[pair[0]]
                output[index]["second_geom"] = -1 if pair is None else geom_ids[pair[1]]
                output[index]["checked_pairs"] = clearance["checked_pair_count"]
                output[index]["violation"] = -1 if report["internal_clearance_violation"] is None else int(report["internal_clearance_violation"])
                output[index]["failure"] = FAILURE_CODES[report["failure_reason"]]
            result = {"status": "ok", "sequence": sequence, "slot": slot, "count": count,
                      "worker_elapsed_s": time.perf_counter() - start}
        else:
            raise ValueError("几何检查协议状态或顺序无效")
        print(json.dumps(result, allow_nan=False), flush=True)
    for segment in segments:
        segment.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.parse_args()
    worker()
