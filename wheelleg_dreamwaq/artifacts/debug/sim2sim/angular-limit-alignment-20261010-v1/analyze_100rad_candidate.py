"""Verify the fresh 100 rad/s candidate against immutable earlier evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "mechanics-isolation-20261010-v1"
sys.path.insert(0, str(OLD))
from analyze import JOINTS, SIGNS, TIMES, sample, velocity, rms


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def main():
    protected = read(ROOT / "protected-before.json")
    for path, expected in protected.items():
        assert digest(path) == expected, path
    records = read(ROOT / "candidate-execution.json")
    provenance = {}
    rows = []
    reproduction = {}
    for case in ("baseline", "closure_off"):
        record = next(r for r in reversed(records) if r["case"] == case and r["evidence_complete"] and r["exit_code"] == 0)
        for path, expected in record["input_hashes"].items():
            assert digest(path) == expected
            provenance[path] = expected
        evidence_path = Path(record["output"]) / "evidence.json"
        current = read(evidence_path)
        high_path = OLD / f"isaac-{case}-cap-10000/evidence.json"
        high = read(high_path)
        mujoco_path = OLD / f"mujoco-{case}/evidence.json"
        mujoco = read(mujoco_path)
        for path in (evidence_path, high_path, mujoco_path, Path(record["console_log"])):
            provenance[str(path)] = digest(path)
        assert current["debug_only"] and current["sources_unchanged"] and not current["ground_enabled"]
        assert current["physics_dt_s"] == .005 and current["reset_physics_time_advanced_s"] == 0.
        assert current["drive_after"] == high["drive_after"]
        assert len(current["runtime_usd_rigid_caps_deg_s"]) == 216
        assert set(current["runtime_usd_rigid_caps_deg_s"].values()) == {5729.578125}
        expected = dict(current["rigid_properties_before"])
        expected["max_angular_velocity"] = 5729.578125
        assert current["rigid_properties_after"] == expected
        fields = ("root_pose_xyzw", "root_velocity_world", "joint_position", "joint_velocity", "body_velocity_world")
        reproduction[case] = max(float(np.abs(np.asarray(sample(current, "isaac", 0, ms)[f]) -
                                                  np.asarray(sample(high, "isaac", 0, ms)[f])).max())
                                 for f in fields for ms in (0, *TIMES))
        assert reproduction[case] == 0., reproduction
        ids = [current["joint_names"].index(name) for name in JOINTS]
        torque_input = next(Path(path) for path in record["input_hashes"] if path.endswith("inputs.json"))
        torques = np.asarray(read(torque_input)["shared_first_torque_control_nm"])
        maximum_body_angular = 0.
        for s in current["samples"]:
            if s["phase"] != "post_physics_step":
                continue
            submitted = np.asarray(s["external_force_argument_native_nm"])
            assert np.array_equal(submitted[:, ids] * SIGNS, torques)
            assert np.count_nonzero(np.delete(submitted, ids, axis=1)) == 0
            assert np.count_nonzero(s["normal_force_world_n"]) == 0
            maximum_body_angular = max(maximum_body_angular,
                                      float(np.linalg.norm(np.asarray(s["body_velocity_world"])[:, :, 3:], axis=-1).max()))
        assert maximum_body_angular < 100.
        for ms in TIMES:
            local = []
            for index in range(8):
                ia, ij, _ = velocity(current, "isaac", index, ms)
                ma, mj, _ = velocity(mujoco, "mujoco", index, ms)
                local.append({"angular_l2_rad_s": float(np.linalg.norm(ia - ma)), "joint_rms_rad_s": rms(ij - mj)})
            rows.append({"case": case, "time_ms": ms, "mean_angular_l2_rad_s": float(np.mean([r["angular_l2_rad_s"] for r in local])),
                         "mean_joint_rms_rad_s": float(np.mean([r["joint_rms_rad_s"] for r in local])),
                         "maximum_body_angular_rad_s": maximum_body_angular})
    for path in (Path(__file__), ROOT / "check_100rad_candidate.py", ROOT / "candidate-execution.json"):
        provenance[str(path)] = digest(path)
    result = {"schema_version": "Angular100RadCandidateV1", "passed": True, "formal_sources_unchanged": True,
              "actual_usd_cap_deg_s": 5729.578125, "actual_usd_cap_rad_s": float(np.deg2rad(5729.578125)),
              "all_state_max_error_against_10000_deg_s": reproduction, "aggregates": rows, "provenance_sha256": provenance}
    (ROOT / "candidate-analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "provenance_sha256"}, indent=2))


if __name__ == "__main__":
    main()
