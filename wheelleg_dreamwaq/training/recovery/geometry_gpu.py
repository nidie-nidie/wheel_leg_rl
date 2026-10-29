from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from .contracts import file_sha256, stable_hash
from .geometry_binary import CAPACITY, PROTOCOL, GeometryBinaryClient
from .geometry_archive import prefix_sha256
from .geometry_constants import BODY_MAPPING, NATIVE_SUPPORT, SCHEMA


IMPLEMENTATION = "RecoveryPlaneSupportFP64SafeLowerBoundV2"
ROUNDING_OPERATIONS = 4096
UNIT_ROUNDOFF = 2.0 ** -53
ROUNDING_GAMMA = ROUNDING_OPERATIONS * UNIT_ROUNDOFF / (1 - ROUNDING_OPERATIONS * UNIT_ROUNDOFF)


@dataclass(frozen=True)
class GeometryBatchResult:
    maximum_ground_penetration_m: torch.Tensor
    ground_failure: torch.Tensor
    internal_failure: torch.Tensor
    absolute_distance_error_bound_m: torch.Tensor
    fallback_env_count: int


class PlaneSupportCalculator:
    def __init__(self, constants, device, body_names):
        if (constants["schema_version"] != SCHEMA or constants["body_mapping_version"] != BODY_MAPPING
                or constants.get("native_support_selection") != NATIVE_SUPPORT
                or file_sha256(constants["arrays_file"]) != constants["arrays_sha256"]
                or constants["identity_sha256"] != stable_hash({key: value for key, value in constants.items()
                    if key not in ("identity_sha256", "arrays_file")})
                or len(set(body_names)) != len(body_names) or not 0 < len(body_names) <= 27):
            raise ValueError("编译几何常量或真实body名称无效")
        self.constants = constants
        self.device = torch.device(device)
        self.body_names = tuple(body_names)
        mapped = [name if name in body_names else {"base": "base_link"}.get(name, name)
                  for name in constants["body_names"]]
        if not set(mapped) <= set(body_names):
            raise ValueError("真实body名称没有覆盖全部同源碰撞形状")
        self.indices = torch.tensor([body_names.index(name) for name in mapped], device=device)
        self.collision_indices = tuple(dict.fromkeys(int(index) for index in self.indices.cpu().tolist()))
        with np.load(constants["arrays_file"], allow_pickle=False) as arrays:
            self.geom_pos = torch.as_tensor(arrays["geom_pos"], dtype=torch.float64, device=device)
            self.local_rotations = torch.as_tensor(arrays["local_rotations"], dtype=torch.float64, device=device)
            self.plane_position = torch.as_tensor(arrays["plane_position"], dtype=torch.float64, device=device)
            self.plane_normal = torch.as_tensor(arrays["plane_normal"], dtype=torch.float64, device=device)
            self.hulls = [torch.as_tensor(arrays[f"hull_{index}"], dtype=torch.float64, device=device)
                          for index in range(len(constants["hull_vertex_counts"]))]
        if (len(self.hulls) != 17 or not torch.allclose(self.plane_normal.norm(), self.plane_normal.new_tensor(1.0), atol=1e-14, rtol=0)
                or not all(bool(torch.isfinite(value).all()) for value in
                           (self.geom_pos, self.local_rotations, self.plane_position, self.plane_normal, *self.hulls))):
            raise ValueError("GPU平面几何常量超出已验证契约")
        self.constant_envelope = (1 + self.geom_pos.abs().sum(dim=1).max().item()
                                  + self.local_rotations.abs().sum(dim=(1, 2)).max().item()
                                  + max(hull.abs().sum(dim=1).max().item() for hull in self.hulls)
                                  + self.plane_position.abs().sum().item()
                                  + self.plane_normal.abs().sum().item())

    def input_validation(self, positions, quaternions):
        if (positions.device != self.device or quaternions.device != self.device
                or positions.dtype != torch.float32 or positions.dtype != quaternions.dtype
                or positions.ndim != 3 or positions.shape[1:] != (len(self.body_names), 3)
                or quaternions.shape != (*positions.shape[:2], 4) or len(positions) == 0):
            raise ValueError("GPU几何输入必须包含全部真实body的同一device位姿")
        q = quaternions.double()
        squared = (q * q).sum(dim=-1)
        norms = squared.sqrt()
        norm_error = (16 * UNIT_ROUNDOFF) * (squared + 1)
        difference = (norms - 1).abs()
        tolerance = 1e-5 + 1e-5
        definitely_bad = difference - norm_error > tolerance
        finite = torch.isfinite(positions).all() & torch.isfinite(quaternions).all()
        if not bool(finite & ~definitely_bad.any()):
            raise ValueError("全部真实body位姿必须有限且四元数近似单位")
        return ((difference - tolerance).abs() <= norm_error).any(dim=1)

    def calculate(self, positions, quaternions, distance_limit):
        # FP64逐项公式与CPU生产公式一致；不使用autocast、TF32或世界顶点三维展开。
        p, q = positions.double(), quaternions.double()
        selected_q = q[:, self.indices]
        w, x, y, z = selected_q.unbind(dim=-1)
        rotation = torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
                                2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
                                2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)), dim=-1).reshape(-1, 17, 3, 3)
        world_position = p[:, self.indices] + (rotation * self.geom_pos[None, :, None, :]).sum(dim=-1)
        world_rotation = rotation @ self.local_rotations
        directions = (world_rotation * self.plane_normal[None, None, :, None]).sum(dim=-2)
        translation = ((world_position - self.plane_position) * self.plane_normal).sum(dim=-1)
        distances = []
        for index, hull in enumerate(self.hulls):
            projection = directions[:, index] @ hull.T
            distances.append(translation[:, index] + projection.amin(dim=1))
        signed = torch.stack(distances, dim=1).clamp_max(distance_limit)
        # 原始FP32输入提升double后，最大表达式深度不足128；4096u覆盖双路径与所有乘加传播。
        # 常量L1包络与(1+|q|_1)^8同时覆盖局部旋转、平面投影及乘法误差的放大。
        # 全部中间量无溢出/下溢：合法近单位q，FP32有限平移，模型常量初始化已验证。
        magnitude = (1 + p.abs().sum(dim=-1).amax(dim=1))
        magnitude *= (1 + q.abs().sum(dim=-1).amax(dim=1)).pow(8) * (1 + self.constant_envelope) ** 3
        error = (ROUNDING_GAMMA * magnitude + torch.finfo(torch.float64).tiny).nextafter(
            torch.full_like(magnitude, math.inf))
        return signed, error


class GeometryGpuGuard:
    def __init__(self, config, run_dir, device, body_names):
        from .geometry_guard import validate_guard_config

        validate_guard_config(config)
        self.config = config
        self.run_dir = Path(run_dir)
        self.transport = GeometryBinaryClient(config, self.run_dir)
        self.archive = None
        try:
            self._initialize_run(device, body_names)
        except BaseException:
            try:
                if self.archive is not None:
                    self.archive.close()
            finally:
                self.transport.close()
            raise

    def _initialize_run(self, device, body_names):
        self.calculator = PlaneSupportCalculator(self.transport.constants, device, body_names)
        self.body_names = tuple(body_names)
        self.device = torch.device(device)
        self.diagnostic_offset = 0
        self.pending_diagnostics = deque()
        self.completed = deque()
        self.archive_path = self.run_dir / "geometry_diagnostics.jsonl"
        archive_generation = 0
        while self.archive_path.exists():
            archive_generation += 1
            self.archive_path = self.run_dir / f"geometry_diagnostics_{archive_generation}.jsonl"
        self.archive = self.archive_path.open("x", encoding="utf-8")
        self.stream_id = self.archive_path.name
        self.last_source_frame = None
        self.last_source_substep = None
        self.archived_sequence = 0
        self.inspected_states = torch.zeros((), dtype=torch.int64, device=device)
        self.failure_states = torch.zeros((), dtype=torch.int64, device=device)
        self.maximum_penetration = torch.zeros((), dtype=torch.float64, device=device)
        self.fallback_states = 0
        self.fallback_batches = 0
        self.internal_inspected_states = 0
        self.internal_violation_states = 0
        self.minimum_clearance = None
        self.closed = False
        self._floor_cache = None
        self.floor_computed_states = 0
        self.floor_reused_states = 0

    @property
    def identity(self):
        return {"config_sha256": stable_hash(self.config),
                "contact_manifest_sha256": file_sha256(self.config["contact_manifest"]),
                "body_mapping_version": BODY_MAPPING, "geometry_representation": "MuJoCo_convex_hull_per_source_mesh",
                "implementation_version": IMPLEMENTATION, "transport_version": PROTOCOL,
                "compiled_geometry_sha256": self.transport.constants["identity_sha256"],
                "arithmetic": "float64_with_roundoff_interval_and_same_input_cpu_reference",
                "native_decision": "gpu_certifies_safe_only_native_cpu_rechecks_all_possible_violations",
                "penetration_semantics": "conservative_upper_bound_for_gpu_safe_states_native_exact_for_rechecked_states",
                "fallback_capacity": CAPACITY}

    def _record_diagnostic(self, result, *, notify=True):
        sequence = result["sequence"]
        values = result["values"]
        self.internal_inspected_states += len(values)
        self.internal_violation_states += int(np.count_nonzero(values["violation"] == 1))
        minimum = float(values["clearance"].min())
        self.minimum_clearance = minimum if self.minimum_clearance is None else min(self.minimum_clearance, minimum)
        record = {"schema_version": "RecoveryGeometryDiagnosticBatchV1", "sequence": sequence,
                  "stream_id": self.stream_id,
                  **result["metadata"], "minimum_nonallowed_clearance_m": values["clearance"].tolist(),
                  "first_geom": values["first_geom"].tolist(), "second_geom": values["second_geom"].tolist(),
                  "checked_pair_count": values["checked_pairs"].tolist(), "violation": values["violation"].tolist()}
        self.archive.write(json.dumps(record, allow_nan=False) + "\n")
        self.archived_sequence = sequence
        if notify:
            if len(self.completed) >= 32:
                raise BufferError("几何完成批次已满，调用方必须每控制步消费诊断")
            self.completed.append(record)

    def _consume(self, sequence, *, wait):
        result = self.transport.collect(sequence, wait=wait)
        if result is None:
            return False
        self.pending_diagnostics.remove(sequence)
        self._record_diagnostic(result)
        return True

    def _ready(self):
        self.transport.check_health()
        while self.pending_diagnostics:
            if not self._consume(self.pending_diagnostics[0], wait=False):
                break

    def _reserve(self):
        self._ready()
        if not self.transport.free:
            if not self.pending_diagnostics:
                raise RuntimeError("几何槽位所有权无效")
            self._consume(self.pending_diagnostics[0], wait=True)

    def check_health(self):
        self._ready()
        return self.drain_diagnostics()

    def drain_diagnostics(self):
        self._ready()
        result = list(self.completed)
        self.completed.clear()
        return result

    def _exact(self, kind, positions, quaternions, ids, *, metadata, env_ids, generations, archive=False):
        output = []
        for start in range(0, len(ids), CAPACITY):
            indices = ids[start:start + CAPACITY]
            self._reserve()
            source_ids = indices.cpu().tolist() if env_ids is None else torch.as_tensor(env_ids, device=self.device)[indices].cpu().tolist()
            source_generations = ([0] * len(indices) if generations is None else
                                  torch.as_tensor(generations, device=self.device)[indices].cpu().tolist())
            ticket = self.transport.submit(kind, self.body_names, positions[indices].detach().cpu().numpy(),
                quaternions[indices].detach().cpu().numpy(), metadata={**metadata, "env_ids": source_ids,
                                                                   "generations": source_generations})
            result = self.transport.collect(ticket)
            if archive:
                self._record_diagnostic(result, notify=False)
            output.append(result["values"])
        return np.concatenate(output) if output else np.empty(0)

    @torch.no_grad()
    def inspect(self, body_names, positions, quaternions, *, env_ids=None, generations=None,
                frame_id=0, substep_id=0, restoring=False, state_key=None):
        if tuple(body_names) != self.body_names:
            raise ValueError("运行时body映射发生变化")
        self._ready()
        metadata = {"frame_id": int(frame_id), "substep_id": int(substep_id),
                    "restoring": bool(restoring), "population_count": len(positions)}
        self.last_source_frame, self.last_source_substep = int(frame_id), int(substep_id)
        if state_key is not None and (not isinstance(state_key, tuple) or len(state_key) != 2
                or any(type(value) is not int or value < 0 for value in state_key)):
            raise ValueError("几何缓存键必须是物理步与写入代数")
        cache_key = None if state_key is None else (*state_key, len(positions))
        if cache_key is not None and self._floor_cache is not None and self._floor_cache[0] == cache_key:
            _, penetration, failure, error = self._floor_cache
            fallback_count = 0
            self.floor_reused_states += len(positions)
        else:
            ambiguous_validation = self.calculator.input_validation(positions, quaternions)
            validation_ids = ambiguous_validation.nonzero().flatten()
            if len(validation_ids):
                self._exact("validate", positions, quaternions, validation_ids, metadata=metadata,
                            env_ids=env_ids, generations=generations)
            signed, error = self.calculator.calculate(positions, quaternions, self.config["distance_limit_m"])
            minimum = signed.amin(dim=1)
            threshold = -self.config["maximum_ground_penetration_m"]
            # 全顶点下界只证明安全，所有可能越界状态使用同帧原生查询。
            failure = torch.zeros(len(positions), dtype=torch.bool, device=self.device)
            fallback_ids = (minimum - error < threshold).nonzero().flatten()
            fallback_count = len(fallback_ids)
            if fallback_count:
                exact = self._exact("floor", positions, quaternions, fallback_ids, metadata=metadata,
                                    env_ids=env_ids, generations=generations)
                minimum[fallback_ids] = -torch.as_tensor(exact["penetration"].copy(), device=self.device)
                failure[fallback_ids] = torch.as_tensor(exact["penetration"] > -threshold, device=self.device)
                self.fallback_states += fallback_count
                self.fallback_batches += (fallback_count + CAPACITY - 1) // CAPACITY
            penetration = (-minimum + error).clamp_min(0)
            if fallback_count:
                penetration[fallback_ids] = -minimum[fallback_ids]
            self.floor_computed_states += len(positions)
            self._floor_cache = None if cache_key is None else (cache_key, penetration, failure, error)
        internal_failure = torch.zeros(len(positions), dtype=torch.bool, device=self.device)
        if self.config["internal_clearance_mode"] == "strict":
            exact = self._exact("internal", positions, quaternions, torch.arange(len(positions), device=self.device),
                                metadata=metadata, env_ids=env_ids, generations=generations, archive=True)
            internal_failure.copy_(torch.as_tensor(exact["violation"] == 1, device=self.device))
        else:
            count = min(len(positions), self.config["diagnostic_envs_per_substep"])
            selected = [(self.diagnostic_offset + index) % len(positions) for index in range(count)]
            self.diagnostic_offset = (self.diagnostic_offset + count) % len(positions)
            collision_ids = self.calculator.collision_indices
            selected_tensor = torch.tensor(selected, device=self.device)
            p = positions[selected_tensor][:, collision_ids].detach().cpu().numpy()
            q = quaternions[selected_tensor][:, collision_ids].detach().cpu().numpy()
            source_ids = selected if env_ids is None else torch.as_tensor(env_ids, device=self.device)[selected_tensor].cpu().tolist()
            source_generations = ([0] * count if generations is None else
                                  torch.as_tensor(generations, device=self.device)[selected_tensor].cpu().tolist())
            for start in range(0, count, CAPACITY):
                self._reserve()
                ticket = self.transport.submit("internal", [self.body_names[index] for index in collision_ids],
                    p[start:start + CAPACITY], q[start:start + CAPACITY], metadata={"frame_id": int(frame_id),
                    "substep_id": int(substep_id), "restoring": bool(restoring), "population_count": len(positions),
                    "env_ids": source_ids[start:start + CAPACITY], "generations": source_generations[start:start + CAPACITY]})
                self.pending_diagnostics.append(ticket)
        self.inspected_states += len(positions)
        self.failure_states += (failure | internal_failure).sum()
        self.maximum_penetration.copy_(torch.maximum(self.maximum_penetration, penetration.max()))
        return GeometryBatchResult(penetration, failure, internal_failure, error, fallback_count)

    def statistics_snapshot(self):
        return {"inspected_states": int(self.inspected_states.item()),
                "failure_states": int(self.failure_states.item()),
                "maximum_ground_penetration_m": float(self.maximum_penetration.item()),
                "penetration_semantics": "conservative_upper_bound_for_gpu_safe_states_native_exact_for_rechecked_states",
                "minimum_nonallowed_clearance_m": self.minimum_clearance,
                "internal_clearance_inspected_states": self.internal_inspected_states,
                "internal_clearance_violation_states": self.internal_violation_states,
                "fallback_states": self.fallback_states, "fallback_batches": self.fallback_batches,
                "floor_computed_states": self.floor_computed_states, "floor_reused_states": self.floor_reused_states,
                "transport": dict(self.transport.statistics)}

    def flush(self):
        self._ready()
        while self.pending_diagnostics:
            self._consume(self.pending_diagnostics[0], wait=True)
        self.archive.flush()
        os.fsync(self.archive.fileno())
        byte_cutoff = self.archive_path.stat().st_size
        return {"diagnostics": self.drain_diagnostics(), "statistics": self.statistics_snapshot(),
                "barrier": {"schema_version": "RecoveryGeometryBarrierV1",
                            "submitted_sequence": self.transport.sequence, "archived_diagnostic_sequence": self.archived_sequence,
                            "stream_id": self.stream_id, "last_source_frame": self.last_source_frame,
                            "last_source_substep": self.last_source_substep,
                            "archive_path": str(self.archive_path.resolve()), "byte_cutoff": byte_cutoff,
                            "prefix_sha256": prefix_sha256(self.archive_path, byte_cutoff),
                            "pending_requests": len(self.transport.pending),
                            "all_requests_acknowledged": not self.transport.pending}}

    def close(self):
        if self.closed:
            return
        try:
            self.flush()
        finally:
            self.archive.close()
            self.transport.close()
            self.closed = True
