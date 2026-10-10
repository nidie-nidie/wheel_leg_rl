"""Test the published PhysX per-link angular bias torque on debug MuJoCo copies."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
OLD = ROOT.parent / "mechanics-isolation-20261010-v1"
TORQUE = ROOT.parent / "same-torque-20261010-v1"
GROUND = ROOT.parent / "ground-on-off-20261010-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    protected = json.loads((ROOT / "protected-before.json").read_text(encoding="utf-8"))
    for path, expected in protected.items():
        assert digest(path) == expected, path
    original = (OLD / "probe_mujoco.py").read_text(encoding="utf-8")
    candidate = replace_once(original, 'args = parser.parse_args()',
                             'parser.add_argument("--limit-period-s", type=float, choices=(.001, .005), required=True)\nargs = parser.parse_args()')
    candidate = replace_once(candidate, '    initial_record = None if args.initial_state_from is None', '''    # Debug only: per-body bias torque -I_world * omega * (1-cap/|omega|) / period.
    # This follows PhysX 5.6.1 forwardDynamic2.cu; never writes qvel.
    cap_shadow = mujoco.MjData(model)
    limit_steps = round(args.limit_period_s / contract.physics_dt_s)
    assert limit_steps >= 1 and abs(limit_steps * contract.physics_dt_s - args.limit_period_s) < 1.e-12
    cap = np.deg2rad(100.)
    original_apply_torque = runtime.controller.apply_torque
    limit_events = []

    def limited_apply_torque(actual_data, torque):
        original_apply_torque(actual_data, torque)
        substep = round(actual_data.time / contract.physics_dt_s)
        if substep % limit_steps == 0:
            mujoco.mj_copyData(cap_shadow, model, actual_data)
            mujoco.mj_forward(model, cap_shadow)
            actual_data.xfrc_applied.fill(0.)
            for body_id in range(1, model.nbody):
                spatial = np.zeros(6)
                mujoco.mj_objectVelocity(model, cap_shadow, mujoco.mjtObj.mjOBJ_BODY, body_id, spatial, 0)
                omega = spatial[:3]
                speed = np.linalg.norm(omega)
                if speed > cap:
                    rotation = cap_shadow.ximat[body_id].reshape(3, 3)
                    inertia = rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T
                    actual_data.xfrc_applied[body_id, 3:] = -(inertia @ omega) * (1. - cap / speed) / args.limit_period_s
            limit_events.append({"time_s": actual_data.time,
                                 "body_torque_world_nm": actual_data.xfrc_applied[:, 3:].tolist()})

    runtime.controller.apply_torque = limited_apply_torque
    initial_record = None if args.initial_state_from is None''')
    candidate = replace_once(candidate, '"schema_version": "MechanicsIsolationMujocoV1",',
                             '"schema_version": "AngularLimitMujocoCandidateV1", "angular_limit_deg_s": 100.,\n'
                             '              "limit_period_s": args.limit_period_s, "limit_events": limit_events,')
    script = ROOT / "probe_mujoco_limit.py"
    if script.exists():
        assert script.read_text(encoding="utf-8") == candidate
    else:
        script.write_text(candidate, encoding="utf-8")
    records_path = ROOT / "mujoco-candidate-execution.json"
    records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.exists() else []
    for case in ("baseline", "closure_off"):
        for period in (.001, .005):
            inputs = [script, TORQUE / "inputs.json", TORQUE / "isaac-formal/evidence.json",
                      GROUND / "isaac-shared_first_action_hold-ground-off/actions.json"]
            hashes = {str(path): digest(path) for path in inputs}
            if any(r["case"] == case and r["period_s"] == period and r["input_hashes"] == hashes
                   and r["evidence_complete"] and r["exit_code"] == 0 for r in records):
                print("REUSE_LIMIT", case, period, flush=True)
                continue
            output = ROOT / f"mujoco-{case}-limit-{period}"
            attempt = 1
            while output.exists():
                attempt += 1
                output = ROOT / f"mujoco-{case}-limit-{period}-attempt-{attempt}"
            command = [str(PROJECT / "sim2sim/mujoco/.venv/Scripts/python.exe"), "-B", str(script),
                       "--output", str(output), "--actions-from", str(inputs[3]), "--torques-from", str(inputs[1]),
                       "--initial-state-from", str(inputs[2]), "--no-ground", "--control-ticks", "1",
                       "--case", "shared_first_action_hold", "--drive-mode", "direct_shared",
                       "--mechanics-case", case, "--limit-period-s", str(period)]
            log = ROOT / (output.name + "-console.log")
            print("START_LIMIT", case, period, flush=True)
            start = time.monotonic()
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=PROJECT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
            complete = (output / "evidence.json").is_file() and "PROBE_COMPLETE" in log.read_text(encoding="utf-8", errors="replace")
            records.append({"case": case, "period_s": period, "command": command, "output": str(output),
                            "exit_code": result.returncode, "evidence_complete": complete, "input_hashes": hashes,
                            "console_log": str(log), "elapsed_s": time.monotonic() - start})
            records_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            print("EXIT_LIMIT", case, period, result.returncode, complete, flush=True)
            if result.returncode or not complete:
                print(log.read_text(encoding="utf-8", errors="replace")[-4500:], flush=True)
                raise SystemExit(result.returncode or 1)
    for path, expected in protected.items():
        assert digest(path) == expected, path
    print("MUJOCO_LIMIT_CANDIDATE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
