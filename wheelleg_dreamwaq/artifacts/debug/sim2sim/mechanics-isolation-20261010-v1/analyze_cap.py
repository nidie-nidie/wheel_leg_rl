"""Validate default/high angular cap controls and their full provenance."""
import json
from pathlib import Path

import numpy as np
from analyze import ROOT, OLD, JOINTS, SIGNS, TIMES, read, digest, sample, velocity, rms


def main():
    base_analysis = read(ROOT / "analysis.json")
    assert base_analysis["passed"]
    for path, expected in base_analysis["provenance_sha256"].items(): assert digest(path) == expected, path
    protected = read(ROOT / "protected-before.json")
    for path, expected in protected.items(): assert digest(path) == expected, path
    provenance = {str(p): digest(p) for p in ROOT.glob("*.py")}
    provenance[str(ROOT / "analysis.json")] = digest(ROOT / "analysis.json")
    provenance[str(ROOT / "cap-execution.json")] = digest(ROOT / "cap-execution.json")
    for relative in ("source/wheelleg_dreamwaq/wheelleg_dreamwaq/assets/wheelleg.py",
                     "dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/sim/schemas/schemas_cfg.py",
                     "dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/sim/schemas/schemas.py"):
        p = ROOT.parents[3] / relative
        provenance[str(p)] = digest(p)
    records = read(ROOT / "cap-execution.json")
    data, selected, reproduction, initial, max_body_speed = {}, [], {}, {}, {}
    for case in ("baseline", "closure_off"):
        original = read(ROOT / f"isaac-{case}/evidence.json")
        for cap in (100, 10000):
            matches = [r for r in records if r["case"] == case and r["cap_deg_s"] == cap
                       and r["exit_code"] == 0 and r["evidence_complete"]
                       and r["input_hashes"][str(ROOT / "probe_isaac_cap.py")] == digest(ROOT / "probe_isaac_cap.py")]
            assert matches
            record = matches[-1]
            for path, expected in record["input_hashes"].items():
                assert digest(path) == expected
                provenance[path] = expected
            p = Path(record["output"]) / "evidence.json"
            d = read(p)
            data[case, cap] = d
            selected.append(record)
            assert d["angular_cap_deg_s"] == cap and d["mechanics_case"] == case
            assert d["sources_unchanged"] and d["debug_only"] and d["physics_loop_only"]
            assert d["free_base"] and not d["ground_enabled"] and not d["formal_ranking_eligible"]
            assert d["physics_dt_s"] == .005 and d["reset_physics_time_advanced_s"] == 0.
            assert d["gravity_m_s2"] == [0., 0., -9.81] and d["drive_after"] == original["drive_after"]
            assert d["torque_input_sha256"] == digest(OLD / "inputs.json")
            assert d["probe_sha256"] == digest(ROOT / "probe_isaac_cap.py")
            expected = dict(d["rigid_properties_before"])
            assert expected["max_angular_velocity"] == 100.
            expected["max_angular_velocity"] = cap
            assert d["rigid_properties_after"] == expected
            assert len(d["runtime_usd_rigid_caps_deg_s"]) == 216
            assert set(d["runtime_usd_rigid_caps_deg_s"].values()) == {cap}
            for path, expected in d["protected_source_sha256"].items(): assert digest(path) == expected == protected[path]
            fields = ("root_pose_xyzw", "root_velocity_world", "joint_position", "joint_velocity")
            initial[case + "-" + str(cap)] = max(float(np.abs(np.asarray(sample(d, "isaac", 0, 0)[f]) -
                                              np.asarray(sample(original, "isaac", 0, 0)[f])).max()) for f in fields)
            assert initial[case + "-" + str(cap)] == 0.
            if cap == 100:
                reproduction[case] = max(float(np.abs(np.asarray(sample(d, "isaac", 0, ms)[f]) -
                                               np.asarray(sample(original, "isaac", 0, ms)[f])).max())
                                         for f in fields for ms in (0, *TIMES))
                assert reproduction[case] == 0.
            ids = [d["joint_names"].index(n) for n in JOINTS]
            torque = np.asarray(read(OLD / "inputs.json")["shared_first_torque_control_nm"])
            speed_rows = []
            for s in d["samples"]:
                if s["phase"] != "post_physics_step": continue
                submitted = np.asarray(s["external_force_argument_native_nm"])
                assert np.array_equal(submitted[:, ids] * SIGNS, torque)
                assert np.max(np.abs(np.delete(submitted, ids, axis=1))) == 0.
                assert np.max(np.abs(s["normal_force_world_n"])) == 0.
                assert np.max(np.abs(np.asarray(s["joint_velocity"])[:, ids])) < 45.
                angular_norm = np.linalg.norm(np.asarray(s["body_velocity_world"])[:, :, 3:], axis=-1)
                speed_rows.append({"time_ms": s["elapsed_s"] * 1000, "max_body_angular_rad_s": float(angular_norm.max()),
                                   "body_samples_over_100_deg_s": int((angular_norm > np.deg2rad(100)).sum())})
            max_body_speed[case + "-" + str(cap)] = speed_rows
            for p in (p, Path(record["console_log"])): provenance[str(p)] = digest(p)
    first_step_errors = {}
    for case in ("baseline", "closure_off"):
        d100, dhi = data[case, 100], data[case, 10000]
        first_step_errors[case] = max(float(np.abs(np.asarray(sample(d100, "isaac", 0, 5)[f]) -
                                                 np.asarray(sample(dhi, "isaac", 0, 5)[f])).max())
                                     for f in ("root_pose_xyzw", "root_velocity_world", "joint_position", "joint_velocity"))
        assert first_step_errors[case] == 0.
    rows, aggregate = [], []
    for case in ("baseline", "closure_off"):
        m = read(ROOT / f"mujoco-{case}/evidence.json")
        assert m["matching_initial_state_sha256"] == digest(OLD / "isaac-formal/evidence.json")
        provenance[str(ROOT / f"mujoco-{case}/evidence.json")] = digest(ROOT / f"mujoco-{case}/evidence.json")
        for cap in (100, 10000):
            i = data[case, cap]
            for ms in TIMES:
                local = []
                for index in range(8):
                    ia, ij, _ = velocity(i, "isaac", index, ms)
                    ma, mj, _ = velocity(m, "mujoco", index, ms)
                    aa, jj, _ = velocity(data[case, 100], "isaac", index, ms)
                    angular_scale = float((np.linalg.norm(ia) + np.linalg.norm(ma)) / 2)
                    joint_scale = (rms(ij) + rms(mj)) / 2
                    row = {"case": case, "cap_deg_s": cap, "time_ms": ms, "environment_index": index,
                           "angular_error_l2_rad_s": float(np.linalg.norm(ia - ma)), "joint_error_rms_rad_s": rms(ij - mj),
                           "relative_angular_error": float(np.linalg.norm(ia - ma) / angular_scale),
                           "relative_joint_error": rms(ij - mj) / joint_scale,
                           "angular_response_mean_norm_rad_s": angular_scale, "joint_response_mean_rms_rad_s": joint_scale,
                           "cap_change_angular_l2_rad_s": float(np.linalg.norm(ia - aa)), "cap_change_joint_rms_rad_s": rms(ij - jj),
                           "isaac_angular_rad_s": ia.tolist(), "mujoco_angular_rad_s": ma.tolist(),
                           "isaac_joint_rad_s": ij.tolist(), "mujoco_joint_rad_s": mj.tolist()}
                    rows.append(row)
                    local.append(row)
                summary = {"case": case, "cap_deg_s": cap, "time_ms": ms}
                for key in local[0]:
                    if key.endswith("rad_s") or key.startswith("relative_"):
                        if isinstance(local[0][key], list): continue
                        summary[key] = float(np.mean([r[key] for r in local]))
                summary["conditions_improved_over_100_deg_s"] = sum(
                    r["angular_error_l2_rad_s"] < next(b["angular_error_l2_rad_s"] for b in rows
                        if b["case"] == case and b["cap_deg_s"] == 100 and b["time_ms"] == ms and b["environment_index"] == r["environment_index"])
                    for r in local)
                aggregate.append(summary)
    result = {"schema_version": "AngularCapIsolationV1", "passed": True, "selected_processes": selected,
              "default_cap_reproduction_max_errors": reproduction, "within_engine_initial_max_errors": initial,
              "first_physics_step_cap_change_max_errors": first_step_errors,
              "body_speed_observations": max_body_speed, "per_condition": rows, "aggregate": aggregate,
              "protected_file_count": len(protected), "provenance_sha256": provenance,
              "unit_evidence": {"local_cfg": "assets/wheelleg.py:51 max_angular_velocity=100.0",
                                "local_api": "isaaclab/sim/schemas/schemas_cfg.py:92-93 in deg/s"},
              "limitation": "Post-solve articulated link angular velocities can exceed the authored cap; this is not an every-snapshot hard clipping claim."}
    (ROOT / "cap-analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("CAP_VERIFIED", len(selected), "processes", len(protected), "protected files")
    for r in aggregate:
        print(r["case"], r["cap_deg_s"], r["time_ms"], "angular", f'{r["angular_error_l2_rad_s"]:.9f}',
              "joint", f'{r["joint_error_rms_rad_s"]:.9f}', "relative angular", f'{r["relative_angular_error"]:.6%}')


if __name__ == "__main__":
    main()
