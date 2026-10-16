"""Replay saved networks in the unchanged MuJoCo runtime; no training."""
from pathlib import Path
import sys,json,time
import numpy as np
import torch
import torch.nn.functional as F
OUT=Path(__file__).resolve().parent
EXP=OUT.parent
PROJECT=EXP.parents[3]
sys.path[:0]=[str(PROJECT/"sim2sim/mujoco"),str(PROJECT/"scripts")]
from wheelleg_mujoco.contract import load_policy_contract,sha256_file
from wheelleg_mujoco.runner import WheelLegMujocoRuntime
from evaluate_mujoco import _failure_reason
torch.set_num_threads(1)
frozen=json.loads((PROJECT/"artifacts/phase2_dreamwaq/split-action-smooth-20261013-v1/frozen-source.json").read_text())
def verify_frozen(): assert all(sha256_file(PROJECT/p)==h for p,h in frozen.items())
verify_frozen()
exports={"baseline":PROJECT/"artifacts/phase2_dreamwaq/training-suite-20261011-stop-reverse-v1/exports/run-03",
         "leg02":EXP/"leg02/export"}
common=[]
for label in exports:
    with np.load(EXP/label/"physics1ms/nominal_stand.npz") as z: common.append(z["history"][100::20])
batch=torch.from_numpy(np.vstack(common))
saved={}; results=[]; checkpoints={}
started=time.perf_counter()
def independent(state,x):
    encoded=x
    for i in (0,2): encoded=F.elu(F.linear(encoded,state[f"cenet.encoder.{i}.weight"],state[f"cenet.encoder.{i}.bias"]))
    encoded=F.linear(encoded,state["cenet.encoder.4.weight"],state["cenet.encoder.4.bias"])
    z=torch.cat((x[:,-25:],encoded[:,:19]),dim=-1)
    for i in (0,2,4): z=F.elu(F.linear(z,state[f"actor.{i}.weight"],state[f"actor.{i}.bias"]))
    return F.linear(z,state["actor.6.weight"],state["actor.6.bias"]).clamp(-20,20)
for label,export in exports.items():
    contract,manifest,model_manifest=load_policy_contract(export/"policy_manifest.json",export/"actor.ts",PROJECT/"sim2sim/mujoco/model_manifest.json")
    runtime=WheelLegMujocoRuntime(PROJECT/"sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml",contract,model_manifest)
    run=Path(manifest["source_checkpoint"]).parent
    indices=[999] if label=="baseline" else [0,100,200,300,500,700,900,999]
    for index in indices:
        cp=run/f"model_{index}.pt"; checkpoints[str(cp)]=sha256_file(cp)
        full=torch.load(cp,map_location="cpu",weights_only=False)
        state=full["model_state_dict"]; info=full["infos"]
        assert info["dreamwaq_algorithm_contract_hash"]==manifest["dreamwaq_algorithm_contract_hash"]
        actor=torch.jit.load(str(export/"actor.ts"),map_location="cpu").eval()
        mapped={k:state["cenet."+k] if k.startswith("encoder.") else state[k] for k in actor.state_dict()}
        actor.load_state_dict(mapped,strict=True)
        assert all(torch.equal(actor.state_dict()[k],v) for k,v in mapped.items())
        with torch.inference_mode():
            output=actor(batch); error=float((output-independent(state,batch)).abs().max())
        assert error<1e-5
        std=state["raw_std"].abs().clamp(1e-4,2).numpy()
        item={"label":label,"completed_iterations":info["completed_iterations"],"checkpoint":str(cp),
              "checkpoint_sha256":checkpoints[str(cp)],"manual_graph_max_error":error,
              "same_input_negative_clip_fraction":(output.numpy()<=-1).mean(0).tolist(),
              "effective_std":std.tolist(),"scenes":{}}
        for scene,command in [("nominal_stand",np.array([0.,0.,.2])),("forward",np.array([1.,0.,.2]))]:
            obs=runtime.reset(command); actions=[]; controls=[]; torques=[]; torque_ticks=[]
            original=runtime.controller.compute_torque
            tick=0
            def observed(data,targets):
                torque=original(data,targets)
                torques.append(torque.copy()); torque_ticks.append(tick)
                return torque
            runtime.controller.compute_torque=observed
            failure=None
            try:
                for tick in range(500):
                    with torch.inference_mode(): action=actor(torch.from_numpy(obs).unsqueeze(0)).numpy()[0]
                    before=runtime.model_map.joint_velocity_control(runtime.data).copy()
                    target=runtime.controller.prepare(action).leg_position_target_mujoco.copy()
                    step=runtime.step(action,command)
                    after=runtime.model_map.joint_velocity_control(runtime.data).copy()
                    q=runtime.model_map.joint_position_control(runtime.data).copy()
                    actions.append(action.copy())
                    controls.append([float(runtime.data.time),*q[:4],*target,*((after-before)[:4]/.02),
                                     step.metrics["vx_mps"],step.metrics["yaw_rate_rad_s"],step.metrics["tilt_rad"],
                                     step.metrics["phi0_delta_abs_rad"]])
                    failure=_failure_reason(runtime,step.metrics)
                    if failure: break
                    obs=step.observation
            finally: runtime.controller.compute_torque=original
            raw=np.array(actions); c=np.array(controls); torque=np.array(torques)
            assert len(torque)==20*len(raw)
            key=label+"_"+str(info["completed_iterations"])+"_"+scene
            saved[key+"__raw"]=raw; saved[key+"__control"]=c
            saved[key+"__torque"]=torque; saved[key+"__torque_tick"]=np.array(torque_ticks)
            steady=raw[100:]; ct=c[100:]; tt=torque[np.array(torque_ticks)>=100]
            equal=None
            if index==999:
                with np.load(EXP/label/"physics1ms"/(scene+".npz")) as z:
                    np.testing.assert_array_equal(raw,z["raw_action"])
                    np.testing.assert_array_equal(c[:,:5],z["control"][:,:5])
                equal=True
            stats={"ticks":len(raw),"failure_reason":failure,"final_raw_action_time_q_exact":equal,"reset_raw_action":raw[0].tolist(),
                   "original_torque_call_count":len(torque)}
            if len(steady):
                clipped=np.clip(raw,-1,1)
                second=(clipped[2:]-2*clipped[1:-1]+clipped[:-2])[98:]
                sigma=torch.tensor(std,dtype=torch.float64)
                mu=torch.tensor(steady,dtype=torch.float64)
                p_unclipped=.5*(torch.erf((1-mu)/(sigma*2**.5))-torch.erf((-1-mu)/(sigma*2**.5)))
                stats.update(negative_clip_fraction=(steady<=-1).mean(0).tolist(),
                    raw_mean=steady.mean(0).tolist(),target_range_deg=np.rad2deg(np.ptp(ct[:,5:9],axis=0)).tolist(),
                    leg_second_difference_mean=float((second[:,:4]**2).sum(1).mean()),
                    actual_acceleration_rms_20ms=np.sqrt((ct[:,9:13]**2).mean(0)).tolist(),
                    effort_saturation_per_joint=np.isclose(np.abs(tt),runtime.controller.effort_limits).mean(0).tolist(),
                    torque_rms_per_joint=np.sqrt((tt**2).mean(0)).tolist(),
                    conditional_sample_within_action_bounds_probability_mean=p_unclipped.mean(0).numpy().tolist(),
                    vx_mae_mps=float(np.abs(ct[:,13]-command[0]).mean()),
                    yaw_mae_rad_s=float(np.abs(ct[:,14]-command[1]).mean()),phi0_delta_mean_rad=float(ct[:,16].mean()))
            item["scenes"][scene]=stats
        results.append(item)
        print(json.dumps({"label":label,"iterations":info["completed_iterations"],"stand":item["scenes"]["nominal_stand"],
                          "forward":item["scenes"]["forward"]}),flush=True)
verify_frozen()
np.savez_compressed(OUT/"snapshot-trajectories.npz",**saved)
(OUT/"snapshots.json").write_text(json.dumps({"schema":"SplitSmoothSnapshotTraceV1","unchanged_frozen_files":frozen,
    "training_started":False,"isaac_started":False,"checkpoint_inputs":checkpoints,"results":results,
    "elapsed_wall_s":time.perf_counter()-started},indent=2,allow_nan=False),encoding="utf-8")

