"""Replay recorded commands/actions to reconstruct synchronized estimator evidence."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from wheelleg_mujoco.contract import load_policy_contract, sha256_file
from wheelleg_mujoco.observation import build_actor_observation, collect_kinematic_state
from wheelleg_mujoco.runner import WheelLegMujocoRuntime

parser=argparse.ArgumentParser()
parser.add_argument("--session",type=Path,required=True)
parser.add_argument("--export",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
project=Path(r"E:\wheel_leg_rl-main\wheelleg_dreamwaq")
args.output.mkdir(exist_ok=True)
protected_paths=[]
for directory in (project/"source/wheelleg_dreamwaq/wheelleg_dreamwaq",
                  project/"sim2sim/mujoco/wheelleg_mujoco",project/"scripts"):
    protected_paths.extend(directory.rglob("*.py"))
protected_paths.extend([project/"sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml",
                        args.export/"actor.ts",args.export/"policy_manifest.json",
                        args.session/"control.csv",args.session/"physics.csv",args.session/"events.jsonl"])
before={str(path):sha256_file(path) for path in protected_paths}
(args.output/"protected-before.json").write_text(json.dumps(before,indent=2),encoding="utf-8")
contract,manifest,model_manifest=load_policy_contract(args.export/"policy_manifest.json",
    args.export/"actor.ts",project/"sim2sim/mujoco/model_manifest.json")
runtime=WheelLegMujocoRuntime(project/"sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml",contract,model_manifest)
policy=torch.jit.load(str(args.export/"actor.ts"),map_location="cpu").eval()
torch.set_num_threads(1)
with (args.session/"control.csv").open(newline="",encoding="utf-8") as stream:
    rows=list(csv.DictReader(stream))
logged={key:np.asarray([float(row[key]) for row in rows]) for key in rows[0]}
history=np.stack([logged[f"history_{i:03d}"] for i in range(125)],axis=1).astype(np.float32)
actions=np.stack([logged[name+"_action_raw"] for name in runtime.model_map.joint_names],axis=1).astype(np.float32)
commands=np.stack([logged["command_"+field] for field in ("vx_mps","yaw_rad_s","height_m")],axis=1)
max_errors={}
def compare(label,actual,expected):
    error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    max_errors[label]=max(max_errors.get(label,0.),error)
    if error>1e-7:
        raise AssertionError((label,error))
true_velocity=[]
angular_velocity=[]
encodings=[]
current_episode=None
started=time.perf_counter()
for index,row in enumerate(rows):
    episode=int(logged["episode_id"][index])
    command=commands[index].copy()
    if episode!=current_episode:
        runtime.reset(command)
        runtime._history[:,6]=command[0]/contract.normalization["vx_max_abs"]
        current_episode=episode
    compare("time_before",runtime.data.time,logged["time_before_s"][index])
    state=collect_kinematic_state(runtime.model,runtime.data,runtime.model_map,contract)
    current=build_actor_observation(state,command,runtime.previous_action,contract)
    current[6]=command[0]/contract.normalization["vx_max_abs"]
    runtime._history[-1]=current
    observation=runtime._history.reshape(-1).astype(np.float32,copy=True)
    compare("full_history",observation,history[index])
    compare("joint_position_before",runtime.model_map.joint_position_mujoco(runtime.data),
            [logged[name+"_position_before_rad"][index] for name in runtime.model_map.joint_names])
    compare("joint_velocity_before",runtime.model_map.joint_velocity_mujoco(runtime.data),
            [logged[name+"_velocity_before_rad_s"][index] for name in runtime.model_map.joint_names])
    with torch.inference_mode():
        tensor=torch.from_numpy(observation).unsqueeze(0)
        encoded=policy.encoder(tensor)
        action=policy(tensor).numpy()[0]
    compare("policy_raw_action",action,actions[index])
    compare("recorded_estimated_vx",float(encoded[0,0])/contract.normalization["root_linear_velocity_scale"],
            logged["estimated_vx_before_action_mps"][index])
    true_velocity.append(state.com_linear_velocity_control.copy())
    angular_velocity.append(state.angular_velocity_control.copy())
    encodings.append(encoded.numpy()[0].copy())
    # Reuse recorded actions so the replay itself cannot change due to inference roundoff.
    result=runtime.step(actions[index],command)
    runtime._history[-1,6]=command[0]/contract.normalization["vx_max_abs"]
    compare("joint_position_after",runtime.model_map.joint_position_mujoco(runtime.data),
            [logged[name+"_position_after_rad"][index] for name in runtime.model_map.joint_names])
    compare("joint_velocity_after",runtime.model_map.joint_velocity_mujoco(runtime.data),
            [logged[name+"_velocity_after_rad_s"][index] for name in runtime.model_map.joint_names])
    compare("vx_after",result.metrics["vx_mps"],logged["vx_mps"][index])
    compare("torque_after",result.applied_torque,
            [logged[name+"_torque_last_nm"][index] for name in runtime.model_map.joint_names])
    compare("time_after",runtime.data.time,logged["time_after_s"][index])
encoded=np.asarray(encodings)
true_velocity=np.asarray(true_velocity)
same_episode=np.diff(logged["episode_id"])==0
rolling_error=float(np.max(np.abs(history[1:].reshape(-1,5,25)[same_episode,:4,:]
                                      -history[:-1].reshape(-1,5,25)[same_episode,1:,:])))
previous_action_error=float(np.max(np.abs(history[1:,-6:][same_episode]
                                        -np.clip(actions[:-1],-1,1)[same_episode])))
assert rolling_error==0 and previous_action_error==0
after={path:sha256_file(Path(path)) for path in before}
assert before==after
report={"passed":True,"control_frames":len(rows),"physics_steps":len(rows)*20,
        "comparison_max_abs_errors":max_errors,"history_rolling_max_abs_error":rolling_error,
        "previous_action_max_abs_error":previous_action_error,
        "protected_files_unchanged":True,"protected_file_count":len(before),
        "elapsed_wall_s":time.perf_counter()-started,
        "source_completed_iterations":manifest["source_completed_iterations"]}
np.savez_compressed(args.output/"feedback-replay.npz",
    time_before=logged["time_before_s"],time_after=logged["time_after_s"],
    episode=logged["episode_id"],history=history,command=commands,action_raw=actions,
    action_clipped=np.clip(actions,-1,1),true_velocity=true_velocity,
    estimated_velocity=encoded[:,:3]/contract.normalization["root_linear_velocity_scale"],
    encoded=encoded,angular_velocity=np.asarray(angular_velocity),
    logged_vx_after=logged["vx_mps"],normalization_velocity_scale=contract.normalization["root_linear_velocity_scale"])
(args.output/"replay-verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report),flush=True)
