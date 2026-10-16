"""Observe the existing controller without changing its calculation or stepping."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import time

import numpy as np


class JointTelemetry:
    def __init__(self, runtime, session, policy_manifest):
        self.runtime, self.session = runtime, session
        self.names = runtime.model_map.joint_names
        self.episode_id = -1
        self.control_id = -1
        self.episode_step = 0
        self.started = time.monotonic()
        self.original_compute = runtime.controller.compute_torque
        self.pending = None
        self.physics_rows = []
        self.control_row = None
        self.last_control = None
        self.writers = {}
        session.mkdir(parents=True, exist_ok=True)
        self.streams = {name: (session / (name + ".csv")).open("w", encoding="utf-8", newline="")
                        for name in ("control", "physics")}
        self.events = (session / "events.jsonl").open("w", encoding="utf-8")
        manifest = {
            "schema": "MujocoJointTelemetryV1",
            "joint_order": list(self.names),
            "joint_labels": ["left_front", "left_rear", "right_front", "right_rear", "left_wheel", "right_wheel"],
            "joint_coordinates": "MuJoCo; right wheel control-to-MuJoCo sign = -1",
            "control_dt_s": runtime.contract.control_dt_s,
            "physics_dt_s": runtime.contract.physics_dt_s,
            "source_checkpoint": policy_manifest["source_checkpoint"],
            "source_completed_iterations": policy_manifest["source_completed_iterations"],
            "actor_sha256": policy_manifest["actor_sha256"],
            "source_seed": policy_manifest["source_seed"],
            "model_xml_sha256": policy_manifest["mujoco_model"]["model_xml_sha256"],
            "force_timestamp": "Forces/qacc produced by mj_step are evaluated at time_before_s; integrated qpos/qvel at time_after_s.",
            "constraint_force": "Total generalized constraint force includes closed-chain constraints and contacts.",
            "command_events": "Every processed key, resets, falls and M symptom markers; R retains the current command.",
        }
        (session / "telemetry_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        runtime.controller.compute_torque = self.compute_torque

    def vector(self, row, field, values, names=None):
        for name, value in zip(self.names if names is None else names, values):
            row[f"{name}_{field}"] = float(value)

    def clock(self):
        return {"episode_id": self.episode_id, "control_id": self.control_id,
                "episode_step": self.episode_step, "wall_elapsed_s": time.monotonic() - self.started}

    def event(self, kind, **details):
        row = {**self.clock(), "wall_utc": datetime.now(timezone.utc).isoformat(),
               "time_s": float(self.runtime.data.time), "event": kind, **details}
        self.events.write(json.dumps(row, allow_nan=False) + "\n")
        self.events.flush()

    def reset(self, command):
        if self.pending is not None or self.control_row is not None:
            raise RuntimeError("Reset during an unfinished telemetry control frame")
        self.episode_id += 1
        self.episode_step = 0
        self.last_control = None
        self.event("reset", command=command.tolist())

    def begin_control(self, action, command, observation, estimated_vx):
        if self.control_row is not None:
            raise RuntimeError("Unfinished telemetry control frame")
        self.control_id += 1
        self.episode_step += 1
        self.physics_rows = []
        row = {**self.clock(), "time_before_s": float(self.runtime.data.time),
               "command_vx_mps": float(command[0]), "command_yaw_rad_s": float(command[1]),
               "command_height_m": float(command[2]), "estimated_vx_before_action_mps": float(estimated_vx)}
        self.vector(row, "action_raw", action)
        self.vector(row, "position_before_rad", self.runtime.model_map.joint_position_mujoco(self.runtime.data))
        self.vector(row, "velocity_before_rad_s", self.runtime.model_map.joint_velocity_mujoco(self.runtime.data))
        for index, value in enumerate(observation):
            row[f"history_{index:03d}"] = float(value)
        self.control_row = row

    def add_targets(self, row, targets):
        self.vector(row, "action_clipped", targets.clipped_action)
        self.vector(row, "position_target_rad", targets.leg_position_target_mujoco, self.names[:4])
        self.vector(row, "velocity_target_rad_s", targets.wheel_velocity_target_mujoco, self.names[4:])

    def finish_physics(self, data):
        if self.pending is None:
            return
        row = self.pending
        row["time_after_s"] = float(data.time)
        self.vector(row, "position_after_rad", self.runtime.model_map.joint_position_mujoco(data))
        self.vector(row, "velocity_after_rad_s", self.runtime.model_map.joint_velocity_mujoco(data))
        self.vector(row, "actuator_force_nm", data.actuator_force[self.runtime.model_map.actuator_ids])
        for field in ("qfrc_actuator", "qfrc_passive", "qfrc_constraint", "qfrc_bias", "qfrc_applied", "qacc"):
            unit = "rad_s2" if field == "qacc" else "nm"
            self.vector(row, f"{field}_{unit}", getattr(data, field)[self.runtime.model_map.dof_addresses])
        row["contact_count"] = int(data.ncon)
        self.physics_rows.append(row)
        self.pending = None

    def compute_torque(self, data, targets):
        # On entry the preceding mj_step is finished. No extra mj_step/mj_forward is used.
        self.finish_physics(data)
        torque = self.original_compute(data, targets)
        if self.control_row is None:
            raise RuntimeError("Controller observed outside an active telemetry frame")
        row = {**self.clock(), "physics_substep": len(self.physics_rows),
               "time_before_s": float(data.time),
               "command_vx_mps": self.control_row["command_vx_mps"],
               "command_yaw_rad_s": self.control_row["command_yaw_rad_s"],
               "command_height_m": self.control_row["command_height_m"]}
        self.add_targets(row, targets)
        position = self.runtime.model_map.joint_position_mujoco(data)
        velocity = self.runtime.model_map.joint_velocity_mujoco(data)
        self.vector(row, "position_before_rad", position)
        self.vector(row, "velocity_before_rad_s", velocity)
        controller = self.runtime.controller
        unclipped = np.r_[controller.leg_kp * (targets.leg_position_target_mujoco - position[:4])
                          - controller.leg_kd * velocity[:4],
                          controller.wheel_kd * (targets.wheel_velocity_target_mujoco - velocity[4:])]
        self.vector(row, "torque_unclipped_nm", unclipped)
        self.vector(row, "torque_command_nm", torque)
        self.pending = row
        return torque

    def write_row(self, name, row):
        if name not in self.writers:
            self.writers[name] = csv.DictWriter(self.streams[name], fieldnames=list(row))
            self.writers[name].writeheader()
        self.writers[name].writerow(row)

    def end_control(self, result):
        self.finish_physics(self.runtime.data)
        if len(self.physics_rows) != self.runtime.contract.physics_steps_per_action:
            raise RuntimeError("Telemetry did not observe every physical step")
        row = self.control_row
        self.add_targets(row, self.runtime.controller.prepare(result.clipped_action))
        row["time_after_s"] = float(self.runtime.data.time)
        self.vector(row, "position_after_rad", self.runtime.model_map.joint_position_mujoco(self.runtime.data))
        self.vector(row, "velocity_after_rad_s", self.runtime.model_map.joint_velocity_mujoco(self.runtime.data))
        self.vector(row, "torque_last_nm", result.applied_torque)
        row.update(result.metrics)
        for physical in self.physics_rows:
            self.write_row("physics", physical)
        self.write_row("control", row)
        for stream in self.streams.values():
            stream.flush()
        self.last_control = row
        self.control_row = None

    def overlay(self):
        if self.last_control is None:
            return "Six joints: waiting for first control step"
        row = self.last_control
        lines = ["Joint        action    target    actual     torque",
                 "Legs: rad | wheels: rad/s | torque: Nm"]
        for index, name in enumerate(self.names):
            target = "position_target_rad" if index < 4 else "velocity_target_rad_s"
            actual = "position_after_rad" if index < 4 else "velocity_after_rad_s"
            lines.append(f"{name:12s} {row[name+'_action_clipped']:+.2f} "
                         f"{row[name+'_'+target]:+.3f} {row[name+'_'+actual]:+.3f} "
                         f"{row[name+'_torque_last_nm']:+.2f}")
        lines.append("M: mark visible jitter for diagnosis")
        return "\n".join(lines)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.event("closed")
        self.runtime.controller.compute_torque = self.original_compute
        for stream in self.streams.values():
            stream.close()
        self.events.close()
