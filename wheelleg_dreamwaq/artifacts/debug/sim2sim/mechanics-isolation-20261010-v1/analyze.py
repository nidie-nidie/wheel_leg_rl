"""Verify one-factor interventions, then compare common physical times."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from wheelleg_dreamwaq.schemas.frames import quat_rotate_inverse_wxyz, transform_usd_vector_to_control

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
OLD = ROOT.parent / "same-torque-20261010-v1"
sys.path.insert(0, str(PROJECT))
from debug.sim2sim.root_cause_suite.trace_contract import load_verified_trace
from debug.sim2sim.root_cause_suite.analysis import p30_odd_response, p40_odd_response

JOINTS = ("jIJ", "jIO", "jAB", "jAG", "jwheel_left", "jwheel_right")
SIGNS = np.asarray([1., 1., 1., 1., 1., -1.])
ROTATION = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
CASES = ("baseline", "passive_off", "closure_off", "dt_5ms")
TIMES = (5, 10, 15, 20)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def sample(data, engine, index, ms):
    rows = [s for s in data["samples"] if s["phase"] != "constructor_after_cache_load"
            and abs(s["elapsed_s"] - ms / 1000) < 1e-10
            and (engine == "isaac" or s["environment_index"] == index)]
    assert len(rows) == 1, (engine, index, ms)
    return rows[0]


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value) ** 2)))


def velocity(data, engine, index, ms):
    s = sample(data, engine, index, ms)
    ids = [data["joint_names"].index(name) for name in JOINTS]
    if engine == "isaac":
        q = torch.tensor(np.asarray(s["root_pose_xyzw"])[index, [6, 3, 4, 5]], dtype=torch.float64)
        w = torch.tensor(s["root_velocity_world"][index][3:], dtype=torch.float64)
        angular = transform_usd_vector_to_control(quat_rotate_inverse_wxyz(q, w)).numpy()
        all_joint = np.asarray(s["joint_velocity"])[index]
    else:
        angular = ROTATION @ np.asarray(s["root_qvel"])[3:]
        assert np.max(np.abs(angular - s["angular_control_fresh_jacobian_rad_s"])) < 1e-12
        all_joint = np.asarray(s["joint_velocity"])
    return angular, all_joint[ids] * SIGNS, all_joint


def historical(provenance):
    runs = PROJECT / "debug/sim2sim/root_cause_suite/runs"
    names = {"p30_closed_I": "task8-p30-closed-direct-01/worker", "p30_closed_M": "task7-p30-closed-direct-01",
             "p30_open_I": "task8-p30-open-direct-smoke-01/worker", "p30_open_M": "task7-p30-open-direct-batch-01",
             "p40_on_I": "task8-p40-on-01/worker", "p40_on_M": "task7-p40-on-01",
             "p40_off_I": "task8-p40-off-01/worker", "p40_off_M": "task7-p40-off-01"}
    responses, records = {}, {}
    for name, directory in names.items():
        path = runs / directory
        trace = load_verified_trace(path / "trace")
        record = read(path / "result.json")
        assert digest(path / "trace/trace.npz") == record["trace_sha256"]
        assert digest(path / "trace/trace.json") == record["trace_metadata_sha256"]
        response, repeat = (p30_odd_response(trace) if name.startswith("p30") else p40_odd_response(trace))
        responses[name] = response
        records[name] = {"directory": str(path), "comparison_profile_hash": record["comparison_profile_hash"],
                         "observed_repeat_spread_rad_s": repeat, "physics_dt_s": record["physics_dt_s"]}
        for relative in ("result.json", "trace/trace.json", "trace/trace.npz"):
            f = path / relative
            provenance[str(f)] = digest(f)
    comparisons = []
    for condition in ("p30_closed", "p30_open", "p40_on", "p40_off"):
        i, m = responses[condition + "_I"], responses[condition + "_M"]
        for k, ms in enumerate((0, 5, 10, 15, 20, 40, 100)):
            comparisons.append({"condition": condition, "time_ms": ms, "joint_response_difference_rms_rad_s": rms(i[k] - m[k])})
    property_records = {}
    for engine, directory in (("isaac", "task8-p20-properties-04/worker"), ("mujoco", "task7-p20-properties-02")):
        p = runs / directory / "properties.json"
        d = read(p)
        assert d["passed"] and d["body_count"] == 27 and not d["failed_bodies"]
        property_records[engine] = d
        provenance[str(p)] = digest(p)
    props_i = {b["mapping"]["canonical"]: b["compiled"] for b in property_records["isaac"]["bodies"]}
    props_m = {b["mapping"]["canonical"]: b["compiled"] for b in property_records["mujoco"]["bodies"]}
    assert props_i.keys() == props_m.keys()
    property_errors = {key: max(float(np.abs(np.asarray(props_i[name][field]) - props_m[name][field]).max())
                                for name in props_i)
                       for key, field in (("mass_kg", "mass_kg"), ("com_m", "center_of_mass_m"),
                                          ("inertia_element_kg_m2", "inertia_com_body_kg_m2"))}
    return {"formal_suite_passed": False, "records": records, "comparisons": comparisons,
            "compiled_property_cross_engine_errors": property_errors,
            "limitations": "Development probes only; G01 not passed. P30-open profile identities differ. Different force, damping and base conditions from new experiment."}


def main():
    protected = read(ROOT / "protected-before.json")
    for path, expected in protected.items():
        assert digest(path) == expected, path
    provenance = {str(p): digest(p) for p in ROOT.glob("*.py")}
    provenance[str(ROOT / "probe-source-hashes.json")] = digest(ROOT / "probe-source-hashes.json")
    for path, expected in read(ROOT / "probe-source-hashes.json").items():
        assert digest(path) == expected
        provenance[path] = expected
    records = read(ROOT / "execution.json")
    data, selected = {}, []
    torque = np.asarray(read(OLD / "inputs.json")["shared_first_torque_control_nm"])
    expected_pairs = [(engine, case) for case in CASES for engine in ("isaac", "mujoco")
                      if not (case == "dt_5ms" and engine == "isaac")]
    for engine, case in expected_pairs:
        candidates = [r for r in records if r["engine"] == engine and r["case"] == case
                      and r["exit_code"] == 0 and (Path(r["output"]) / "evidence.json").is_file()
                      and r.get("evidence_complete", True)
                      and r["input_hashes"][str(ROOT / f"probe_{engine}.py")] == digest(ROOT / f"probe_{engine}.py")]
        assert candidates
        record = candidates[-1]
        for path, expected in record["input_hashes"].items():
            assert digest(path) == expected
            provenance[path] = expected
        path = Path(record["output"]) / "evidence.json"
        d = read(path)
        assert d["mechanics_case"] == case and d["drive_mode"] == "direct_shared"
        assert d["sources_unchanged"] and d["debug_only"] and not d["formal_ranking_eligible"]
        assert not d["ground_enabled"] and d["free_base"] and d["gravity_m_s2"] == [0., 0., -9.81]
        assert d["reset_physics_time_advanced_s"] == 0 and d["control_ticks"] == 1
        assert "PROBE_COMPLETE" in Path(record["console_log"]).read_text(encoding="utf-8", errors="replace")
        assert d["probe_sha256"] == digest(ROOT / f"probe_{engine}.py")
        assert d["physics_dt_s"] == (0.005 if engine == "isaac" or case == "dt_5ms" else 0.001)
        for p, expected in d["protected_source_sha256"].items():
            assert digest(p) == expected == protected[p]
        if engine == "mujoco":
            assert max(d["observer_replay_max_errors"]) == 0 and len(d["observer_replay_max_errors"]) == 8
            assert d["only_declared_mechanics_changes"]
            assert d["matching_initial_state_sha256"] == digest(OLD / "isaac-formal/evidence.json")
        else:
            assert d["physics_loop_only"]
            assert len(d["disabled_closure_paths"]) == (32 if case == "closure_off" else 0)
        data[engine, case] = d
        selected.append(record)
        for p in (path, Path(record["console_log"])):
            provenance[str(p)] = digest(p)
        for p in (path.parent / "model").glob("*.xml*"):
            provenance[str(p)] = digest(p)

    baseline_errors, initial_errors, force_errors, max_speed = {}, {}, {}, {}
    for engine, case in expected_pairs:
        d, baseline = data[engine, case], data[engine, "baseline"]
        fields = ("root_pose_xyzw", "root_velocity_world", "joint_position", "joint_velocity") if engine == "isaac" else (
            "root_qpos", "root_qvel", "joint_position", "joint_velocity")
        initial_errors[engine + "-" + case] = max(float(np.abs(np.asarray(sample(d, engine, i, 0)[f]) -
                                                     np.asarray(sample(baseline, engine, i, 0)[f])).max())
                                                 for i in range(8) for f in fields)
        assert initial_errors[engine + "-" + case] == 0
        if case == "baseline":
            previous_path = OLD / ("isaac-direct_shared" if engine == "isaac" else "mujoco-direct_shared-matched") / "evidence.json"
            previous = read(previous_path)
            provenance[str(previous_path)] = digest(previous_path)
            baseline_errors[engine] = max(float(np.abs(np.asarray(sample(d, engine, i, ms)[f]) -
                                               np.asarray(sample(previous, engine, i, ms)[f])).max())
                                         for i in range(8) for ms in (0, *TIMES) for f in fields)
            assert baseline_errors[engine] == 0
        if engine == "isaac":
            expected = json.loads(json.dumps(data[engine, "baseline"]["drive_after"]))
            active_ids = d["active_joint_ids"]
            passive_ids = [j for j in range(26) if j not in active_ids]
            if case == "passive_off":
                for row in expected["damping"]:
                    for j in passive_ids: row[j] = 0.
                expected["actuator_cache"]["passive"]["damping"] = np.zeros_like(expected["actuator_cache"]["passive"]["damping"]).tolist()
            if case == "closure_off":
                expected["closure_enabled"] = {k: False for k in expected["closure_enabled"]}
            assert d["drive_after"] == expected
        else:
            expected = json.loads(json.dumps(data[engine, "baseline"]["mechanics_after"]))
            assert d["mechanics_before"] == data[engine, "baseline"]["mechanics_before"]
            if case == "passive_off":
                for joint in expected["joints"]:
                    if joint["name"] != "base_free" and joint["name"] not in JOINTS: joint["damping"] = [0.]
            if case == "closure_off":
                for eq in expected["equalities"]: eq["active_initial"] = False
            if case == "dt_5ms": expected["options"]["timestep"] = .005
            assert d["mechanics_after"] == expected
        force_errors[engine + "-" + case] = 0.
        max_speed[engine + "-" + case] = 0.
        for s in d["samples"]:
            if s["phase"] != "post_physics_step": continue
            ids = [d["joint_names"].index(n) for n in JOINTS]
            if engine == "isaac":
                forces = np.asarray(s["external_force_argument_native_nm"])
                error = np.abs(forces[:, ids] * SIGNS - torque).max()
                assert np.max(np.abs(np.delete(forces, ids, axis=1))) == 0.
                assert np.max(np.abs(s["normal_force_world_n"])) == 0.
                speed = np.asarray(s["joint_velocity"])[:, ids]
            else:
                index = s["environment_index"]
                error = max(np.abs(np.asarray(s[f]) * SIGNS - torque[index]).max()
                            for f in ("external_ctrl_native_nm", "actuator_force_native_nm", "generalized_actuator_force_native_nm"))
                assert s["current_geometry_contact_count"] == 0 and not s["contacts_last_solve"]
                assert s["equality_active_live"] == [case != "closure_off"] * 8
                speed = np.asarray(s["joint_velocity"])[ids]
            force_errors[engine + "-" + case] = max(force_errors[engine + "-" + case], float(error))
            max_speed[engine + "-" + case] = max(max_speed[engine + "-" + case], float(np.abs(speed).max()))
        assert force_errors[engine + "-" + case] == 0 and max_speed[engine + "-" + case] < 45.

    cross_initial = {}
    rows, aggregate = [], []
    for case in CASES:
        i_case = "baseline" if case == "dt_5ms" else case
        di, dm = data["isaac", i_case], data["mujoco", case]
        mapping = [dm["joint_names"].index(name) for name in di["joint_names"]]
        errors = {"q_rad": 0., "qd_rad_s": 0., "root_z_m": 0., "normalized_quaternion": 0., "root_velocity": 0.}
        for index in range(8):
            i, m = sample(di, "isaac", index, 0), sample(dm, "mujoco", index, 0)
            for f, key in (("joint_position", "q_rad"), ("joint_velocity", "qd_rad_s")):
                errors[key] = max(errors[key], float(np.abs(np.asarray(i[f])[index] - np.asarray(m[f])[mapping]).max()))
            errors["root_z_m"] = max(errors["root_z_m"], abs(i["root_pose_xyzw"][index][2] - m["root_qpos"][2]))
            qi, qm = np.asarray(i["root_pose_xyzw"])[index, [6, 3, 4, 5]], np.asarray(m["root_qpos"])[3:]
            errors["normalized_quaternion"] = max(errors["normalized_quaternion"], float(np.abs(qi / np.linalg.norm(qi) - qm / np.linalg.norm(qm)).max()))
            errors["root_velocity"] = max(errors["root_velocity"], float(np.abs(i["root_velocity_world"][index]).max()), float(np.abs(m["root_qvel"]).max()))
        assert max(errors.values()) == 0.
        cross_initial[case] = errors
        for ms in TIMES:
            local = []
            for index in range(8):
                ia, ij, iall = velocity(di, "isaac", index, ms)
                ma, mj, mall = velocity(dm, "mujoco", index, ms)
                angular_error, joint_error = float(np.linalg.norm(ia - ma)), rms(ij - mj)
                angular_scale, joint_scale = float((np.linalg.norm(ia) + np.linalg.norm(ma)) / 2), (rms(ij) + rms(mj)) / 2
                ba_i, bj_i, _ = velocity(data["isaac", "baseline"], "isaac", index, ms)
                ba_m, bj_m, _ = velocity(data["mujoco", "baseline"], "mujoco", index, ms)
                row = {"case": case, "time_ms": ms, "environment_index": index,
                       "angular_error_l2_rad_s": angular_error, "joint_error_rms_rad_s": joint_error,
                       "all_joint_error_rms_rad_s": rms(iall - mall[mapping]),
                       "relative_angular_error": angular_error / angular_scale if angular_scale > .01 else None,
                       "relative_joint_error": joint_error / joint_scale if joint_scale > .01 else None,
                       "angular_response_mean_norm_rad_s": angular_scale, "joint_response_mean_rms_rad_s": joint_scale,
                       "isaac_angular_change_from_baseline_rad_s": float(np.linalg.norm(ia - ba_i)),
                       "mujoco_angular_change_from_baseline_rad_s": float(np.linalg.norm(ma - ba_m)),
                       "isaac_joint_change_from_baseline_rad_s": rms(ij - bj_i),
                       "mujoco_joint_change_from_baseline_rad_s": rms(mj - bj_m),
                       "isaac_angular_rad_s": ia.tolist(), "mujoco_angular_rad_s": ma.tolist(),
                       "isaac_joint_rad_s": ij.tolist(), "mujoco_joint_rad_s": mj.tolist(),
                       "per_joint_abs_error_rad_s": np.abs(ij - mj).tolist()}
                local.append(row)
                rows.append(row)
            summary = {"case": case, "time_ms": ms}
            for key in local[0]:
                if key.endswith("rad_s") or key.startswith("relative_"):
                    if isinstance(local[0][key], (list, dict)): continue
                    values = [r[key] for r in local if r[key] is not None]
                    summary[key] = float(np.mean(values)) if values else None
            summary["conditions_with_smaller_angular_error_than_baseline"] = sum(
                r["angular_error_l2_rad_s"] < next(b["angular_error_l2_rad_s"] for b in rows
                    if b["case"] == "baseline" and b["time_ms"] == ms and b["environment_index"] == r["environment_index"])
                for r in local)
            aggregate.append(summary)
    result = {"schema_version": "MechanicsIsolationAnalysisV1", "passed": True,
              "protected_file_count": len(protected), "selected_processes": selected,
              "baseline_reproduction_max_errors": baseline_errors, "within_engine_initial_errors": initial_errors,
              "cross_engine_initial_errors": cross_initial, "external_force_max_errors_nm": force_errors,
              "max_active_speed_rad_s": max_speed, "per_condition": rows, "aggregate": aggregate,
              "historical_auxiliary": historical(provenance), "provenance_sha256": provenance}
    (ROOT / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("VERIFIED", len(selected), "processes", len(protected), "protected files")
    for r in aggregate:
        print(r["case"], r["time_ms"], "angular", f'{r["angular_error_l2_rad_s"]:.9f}',
              "joint", f'{r["joint_error_rms_rad_s"]:.9f}', "relative angular", f'{r["relative_angular_error"]:.6%}')


if __name__ == "__main__":
    main()
