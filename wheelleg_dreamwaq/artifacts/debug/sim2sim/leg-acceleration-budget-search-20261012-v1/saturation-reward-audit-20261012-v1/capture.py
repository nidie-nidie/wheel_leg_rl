"""Diagnostic-only exact MuJoCo replays; no changed control semantics."""
from pathlib import Path
import csv, json, sys, time
import numpy as np
import torch, mujoco
OUT = Path(__file__).resolve().parent
BASE = OUT.parent
PROJECT = BASE.parents[3]
sys.path.insert(0, str(PROJECT / "sim2sim/mujoco"))
from wheelleg_mujoco.contract import load_policy_contract, sha256_file
from wheelleg_mujoco.runner import WheelLegMujocoRuntime
from wheelleg_mujoco.observation import collect_kinematic_state, build_actor_observation
from wheelleg_mujoco.evaluation import FORMAL_SCENARIOS
torch.set_num_threads(1)
protected = json.loads((OUT / "protected-before.json").read_text())
assert all(sha256_file(Path(p)) == h for p, h in protected.items())
started = time.perf_counter()
arrays, checks = {}, []

def load(arm):
    export = BASE / arm / "export"
    contract, manifest, mm = load_policy_contract(export / "policy_manifest.json", export / "actor.ts", PROJECT / "sim2sim/mujoco/model_manifest.json")
    runtime = WheelLegMujocoRuntime(PROJECT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml", contract, mm)
    return runtime, torch.jit.load(str(export / "actor.ts"), map_location="cpu").eval()

def capture(runtime, policy, commands, original_rows, key, interactive=False):
    obs = runtime.reset(commands[0])
    floor = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    wheel_bodies = [mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("jwheel_left", "jwheel_right")]
    assert floor >= 0 and all(b>=0 for b in wheel_bodies)
    names = runtime.model_map.joint_names
    data = {n: [] for n in ("time_before", "time_after", "command", "action_raw", "action_clipped", "target", "q", "v", "acc", "tau", "root_v", "root_w", "gravity", "height", "phi", "length", "contact_normal_last", "roll_pitch_yaw", "effort_saturation", "tilt")}
    max_error, max_action_error, max_history_error = 0.0, 0.0, 0.0
    for i, (command, original) in enumerate(zip(commands, original_rows)):
        if interactive:
            # Match the recorded interactive runner's current-command refresh.
            current = build_actor_observation(collect_kinematic_state(runtime.model,runtime.data,runtime.model_map,runtime.contract),command,runtime.previous_action,runtime.contract)
            current[6] = command[0]/runtime.contract.normalization["vx_max_abs"]
            runtime._history[-1] = current
            obs = runtime._history.reshape(-1).astype(np.float32,copy=True)
            logged_history=np.array([float(original[f"history_{j:03d}"]) for j in range(125)],dtype=np.float32)
            max_history_error=max(max_history_error,float(np.abs(obs-logged_history).max()))
            np.testing.assert_allclose(obs,logged_history,atol=1e-7,rtol=0)
        with torch.inference_mode():
            action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy()[0]
        if interactive:
            recorded = np.array([float(original[n + "_action_raw"]) for n in names])
            max_action_error = max(max_action_error, float(np.abs(recorded-action).max()))
            np.testing.assert_allclose(action, recorded, atol=1e-7, rtol=0)
            action = recorded
        vb = runtime.model_map.joint_velocity_control(runtime.data).copy()
        t_before = float(runtime.data.time)
        targets = runtime.controller.prepare(action)
        target = np.r_[targets.leg_position_target_mujoco.copy(), targets.wheel_velocity_target_mujoco.copy()*runtime.contract.wheel_signs]
        result = runtime.step(action, command)
        state = collect_kinematic_state(runtime.model, runtime.data, runtime.model_map, runtime.contract)
        va = state.joint_velocity_control.copy()
        qa = state.joint_position_control.copy()
        acc = (va-vb)/runtime.contract.control_dt_s
        metrics = result.metrics
        if interactive:
            pairs = []
            qm = runtime.model_map.joint_position_mujoco(runtime.data)
            vm = runtime.model_map.joint_velocity_mujoco(runtime.data)
            for j,n in enumerate(names):
                pairs.extend([(qm[j], float(original[n+"_position_after_rad"])), (vm[j], float(original[n+"_velocity_after_rad_s"]))])
            pairs.extend((metrics[n], float(original[n])) for n in ("vx_mps", "yaw_rate_rad_s", "base_height_m", "phi0_left_rad", "phi0_right_rad"))
            pairs.append((runtime.data.time, float(original["time_after_s"])))
        else:
            pairs = [(runtime.data.time, float(original["time_s"]))]
            pairs.extend((v, float(original[n])) for n,v in metrics.items() if n != "action_saturation_fraction")
            for j,n in enumerate(names[:4]):
                for suffix,value in (("q_rad",qa[j]),("qdot_rad_s",va[j]),("qddot_rad_s2",acc[j]),("target_rad",target[j])):
                    pairs.append((value,float(original[n+"_"+suffix])))
        error = max(abs(x-y) for x,y in pairs)
        max_error = max(max_error, error)
        assert error < 1e-7, (key, i, error)
        # Last solved contact forces precede final integration by one physics dt.
        normal = np.zeros(2)
        for contact_id in range(runtime.data.ncon):
            c = runtime.data.contact[contact_id]
            if floor not in (c.geom1,c.geom2):
                continue
            other = c.geom2 if c.geom1 == floor else c.geom1
            body = int(runtime.model.geom_bodyid[other])
            if body in wheel_bodies:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(runtime.model,runtime.data,contact_id,wrench)
                assert wrench[0] >= -1e-8
                normal[wheel_bodies.index(body)] += max(float(wrench[0]),0)
        values = dict(time_before=t_before,time_after=float(runtime.data.time),command=command,action_raw=action,action_clipped=result.clipped_action,target=target,q=qa,v=va,acc=acc,tau=result.applied_torque,
                      root_v=state.com_linear_velocity_control,root_w=state.angular_velocity_control,gravity=state.projected_gravity_control,height=state.base_height_m,
                      phi=[metrics["phi0_left_rad"],metrics["phi0_right_rad"]],length=[metrics["l0_left_m"],metrics["l0_right_m"]],contact_normal_last=normal,
                      roll_pitch_yaw=[metrics[n] for n in ("roll_rad","pitch_rad","yaw_rad")],effort_saturation=metrics["effort_saturation_fraction"],tilt=metrics["tilt_rad"])
        for n,value in values.items():
            data[n].append(np.array(value,copy=True))
        obs = result.observation
    for n,values in data.items():
        arrays[key+"__"+n] = np.array(values)
    checks.append({"trace":key,"ticks":len(commands),"replay_max_abs_error":max_error,"interactive_inference_max_abs_error":max_action_error,"interactive_history_max_abs_error":max_history_error})
    print(json.dumps(checks[-1]),flush=True)

for arm in ("baseline","85"):
    runtime,policy = load(arm)
    for name,command in FORMAL_SCENARIOS:
        with (BASE/arm/"motion"/(name+".csv")).open(newline="") as f:
            original = list(csv.DictReader(f))
        assert len(original)==500 and all(r["survived"]=="1" for r in original)
        capture(runtime,policy,np.repeat(np.array(command)[None,:],len(original),axis=0),original,arm+"_"+name)
runtime,policy = load("85")
session = BASE/"85/interactive-20261012-100842"
with (session/"control.csv").open(newline="") as f:
    original = list(csv.DictReader(f))
assert len({r["episode_id"] for r in original})==1
commands = np.array([[float(r[n]) for n in ("command_vx_mps","command_yaw_rad_s","command_height_m")] for r in original])
capture(runtime,policy,commands,original,"85_interactive",True)
np.savez_compressed(OUT/"trajectories.npz",**arrays)
assert all(sha256_file(Path(p))==h for p,h in protected.items())
(OUT/"capture-verification.json").write_text(json.dumps({"schema":"SaturationRewardCaptureV1","checks":checks,"control_dt_s":runtime.contract.control_dt_s,"physics_dt_s":runtime.contract.physics_dt_s,"contact_force_timestamp":"last_substep_before_integration", "protected_files_unchanged":True,"elapsed_wall_s":time.perf_counter()-started},indent=2),encoding="utf-8")
