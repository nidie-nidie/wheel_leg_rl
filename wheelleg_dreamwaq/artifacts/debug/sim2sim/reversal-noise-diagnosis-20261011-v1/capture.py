"""Replay original recorded actions; passive extra reads only. No training."""
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
import torch

ROOT=Path(r"E:\wheel_leg_rl-main")
PROJECT=ROOT/"wheelleg_dreamwaq"
TRIAL=PROJECT/"artifacts/phase2_dreamwaq/leg-reversal-trial-20261011-v1"
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT/"sim2sim/mujoco"))
from wheelleg_mujoco.contract import load_policy_contract, sha256_file
from wheelleg_mujoco.runner import WheelLegMujocoRuntime

torch.set_num_threads(1)
def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def rows(p):
    with Path(p).open(encoding="utf-8",newline="") as f: return list(csv.DictReader(f))
def write(p,v): Path(p).write_bytes((json.dumps(v,indent=2,sort_keys=True)+"\n").encode("utf-8"))

def main():
    manifest=read(TRIAL/"experiment-manifest.json")
    assert manifest["status"]=="evaluated"
    captures=[]
    for arm in manifest["arms"]:
        folder=TRIAL/arm["name"]
        checkpoint=Path(arm["checkpoint"])
        actor_path=folder/"export/actor.ts"
        assert sha256_file(checkpoint)==arm["checkpoint_sha256"]
        assert sha256_file(actor_path)==arm["actor_sha256"]
        contract, policy_manifest, model_manifest=load_policy_contract(folder/"export/policy_manifest.json",actor_path,PROJECT/"sim2sim/mujoco/model_manifest.json")
        model=PROJECT/"sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"
        assert sha256_file(model)==model_manifest["model_xml"]["sha256"]
        actor=torch.jit.load(str(actor_path),map_location="cpu").eval()
        runtime=WheelLegMujocoRuntime(model,contract,model_manifest)
        for scenario in ("nominal_stand","forward","reverse"):
            reference=folder/"mujoco"/(scenario+".csv")
            identity=read(folder/"mujoco/summary.json")["scenario_files"][scenario]
            assert sha256_file(reference)==identity["sha256"]
            original=rows(reference)
            command=np.array([float(original[0][key]) for key in ("target_vx_mps","target_yaw_rate_rad_s","target_base_height_m")])
            obs=runtime.reset(command)
            data={k:[] for k in ("history","mean_action","q_before","q_after","v_before","v_after","leg_target","torque","time_s")}
            differences={k:0.0 for k in ("network_mean","metric","clipped_action","target","torque","qvel","acceleration")}
            for row in original:
                action=np.array([float(row[f"raw_action_{j}"]) for j in range(6)])
                with torch.inference_mode(): mean=actor(torch.from_numpy(obs).unsqueeze(0)).numpy()[0]
                differences["network_mean"]=max(differences["network_mean"],float(np.abs(action-mean).max()))
                q_before=runtime.model_map.joint_position_mujoco(runtime.data).copy()
                v_before=runtime.model_map.joint_velocity_mujoco(runtime.data).copy()
                target=runtime.controller.prepare(action).leg_position_target_mujoco.copy()
                data["history"].append(obs.copy()); data["mean_action"].append(mean.copy())
                data["q_before"].append(q_before); data["v_before"].append(v_before)
                result=runtime.step(action,command)
                q_after=runtime.model_map.joint_position_mujoco(runtime.data).copy()
                v_after=runtime.model_map.joint_velocity_mujoco(runtime.data).copy()
                for key,value in result.metrics.items():
                    differences["metric"]=max(differences["metric"],abs(float(row[key])-value))
                differences["clipped_action"]=max(differences["clipped_action"],max(abs(float(row[f"clipped_action_{j}"])-result.clipped_action[j]) for j in range(6)))
                differences["torque"]=max(differences["torque"],max(abs(float(row[f"applied_torque_{j}_nm"])-result.applied_torque[j]) for j in range(6)))
                differences["target"]=max(differences["target"],max(abs(float(row[f"leg_target_{j}_rad"])-target[j]) for j in range(4)))
                differences["qvel"]=max(differences["qvel"],max(abs(float(row[f"leg_velocity_{j}_rad_s"])-v_after[j]) for j in range(4)))
                differences["acceleration"]=max(differences["acceleration"],max(abs(float(row[f"leg_acceleration_{j}_rad_s2"])-(v_after[j]-v_before[j])/contract.control_dt_s) for j in range(4)))
                data["q_after"].append(q_after); data["v_after"].append(v_after)
                data["leg_target"].append(target); data["torque"].append(result.applied_torque.copy())
                data["time_s"].append(runtime.data.time)
                assert row["survived"]=="1"
                obs=result.observation
            assert len(original)==500
            assert differences["network_mean"]<=1e-7,differences
            assert max(v for k,v in differences.items() if k!="network_mean")<1e-10,differences
            npz=OUT/"captures"/f"{arm['name']}-{scenario}.npz"
            npz.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(npz,**{k:np.array(v) for k,v in data.items()})
            record={"arm":arm["name"],"scenario":scenario,"frames":len(original),"command":command.tolist(),
                    "checkpoint":str(checkpoint),"checkpoint_sha256":arm["checkpoint_sha256"],
                    "reference":str(reference),"reference_sha256":identity["sha256"],
                    "capture":str(npz),"capture_sha256":sha256_file(npz),"max_errors":differences,
                    "q_nominal":contract.q_nominal.tolist(),"leg_action_scale":contract.leg_action_scale,
                    "target_lower":contract.leg_target_lower.tolist(),"target_upper":contract.leg_target_upper.tolist(),
                    "dt":contract.control_dt_s,"joint_order":list(runtime.model_map.joint_names)}
            captures.append(record)
            write(OUT/"capture-verification.json",{"captures":captures,"completed":len(captures)==9})
            print(json.dumps({"arm":arm["name"],"scenario":scenario,"max_errors":differences}),flush=True)
    print("PASS: 4500-frame exact-trajectory replay capture",flush=True)
if __name__=="__main__": main()
