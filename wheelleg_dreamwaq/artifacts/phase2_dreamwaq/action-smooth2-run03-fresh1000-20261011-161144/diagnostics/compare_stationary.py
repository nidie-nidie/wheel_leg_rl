from pathlib import Path
import argparse,csv,hashlib,json,sys,subprocess
import numpy as np
import torch
project=Path(r"E:\wheel_leg_rl-main\wheelleg_dreamwaq")
sys.path.insert(0,str(project/"scripts"))
sys.path.insert(0,str(project/"artifacts/debug/sim2sim/interactive-mujoco-20261010-v1"))
from evaluate_mujoco import DEFAULT_MODEL,DEFAULT_MODEL_MANIFEST,_failure_reason
from joint_telemetry import JointTelemetry
from wheelleg_mujoco.contract import load_policy_contract,sha256_file
from wheelleg_mujoco.runner import WheelLegMujocoRuntime
from wheelleg_mujoco.model_semantics import stable_hash
parser=argparse.ArgumentParser()
parser.add_argument("--experiment",type=Path,required=True)
parser.add_argument("--smoke",action="store_true")
parser.add_argument("--export-override",type=Path)
args=parser.parse_args()
record=json.loads((args.experiment/"experiment-manifest.json").read_text(encoding="utf-8"))
command=np.array([0.,0.,.2])
exports=[("old_1000",Path(record["baseline_export"])),
         ("new_1000",args.export_override or args.experiment/"export"),
         ("old_1401_reference",Path(record["extra_reference_export"]))]
results={}; traces={}
for label,folder in exports:
    actor_path=folder/"actor.ts"; manifest_path=folder/"policy_manifest.json"
    contract,manifest,model_manifest=load_policy_contract(manifest_path,actor_path,DEFAULT_MODEL_MANIFEST)
    assert sha256_file(DEFAULT_MODEL)==model_manifest["model_xml"]["sha256"]
    actor=torch.jit.load(str(actor_path),map_location="cpu").eval()
    runtime=WheelLegMujocoRuntime(DEFAULT_MODEL,contract,model_manifest)
    folder_out=args.experiment/("stationary-smoke" if args.smoke else "stationary-traces")/label
    observation=runtime.reset(command)
    failure=None
    logger=JointTelemetry(runtime,folder_out,manifest)
    try:
        logger.reset(command)
        for tick in range(20 if args.smoke else 500):
            with torch.inference_mode():
                tensor=torch.from_numpy(observation).unsqueeze(0)
                action=actor(tensor).cpu().numpy().reshape(-1)
                estimated=float(actor.encoder(tensor)[0,0]/manifest["normalization"]["root_linear_velocity_scale"])
            assert action.shape==(6,) and np.isfinite(action).all()
            logger.begin_control(action,command,observation,estimated)
            step=runtime.step(action,command)
            logger.end_control(step)
            failure=_failure_reason(runtime,step.metrics)
            observation=step.observation
            if failure:
                logger.event("evaluation_failure",reason=failure)
                break
    finally:
        logger.close()
    with (folder_out/"control.csv").open(encoding="utf-8",newline="") as f:rows=list(csv.DictReader(f))
    def col(name):return np.array([float(row[name]) for row in rows])
    time=col("time_before_s")
    mask=(time>=2-1e-9)&(time<=10+1e-9)
    names=manifest["action"]["canonical_joint_order"]
    action=np.column_stack([col(name+"_action_clipped") for name in names])
    target=np.column_stack([col(name+"_position_target_rad") for name in names[:4]])
    velocity=np.column_stack([col(name+"_velocity_before_rad_s") for name in names[:4]])
    actual=np.column_stack([col(name+"_position_before_rad") for name in names[:4]])
    wheel_target=np.column_stack([col(name+"_velocity_target_rad_s") for name in names[4:]])
    def rms(x):return float(np.sqrt(np.mean(x*x))) if x.size else None
    sample={"checkpoint":manifest["source_checkpoint"],"checkpoint_sha256":manifest["source_checkpoint_sha256"],
        "completed_iterations":manifest["source_completed_iterations"],"source_seed":manifest["source_seed"],
        "action_smooth_weight":manifest["training_reward_weights"]["action_smooth"],
        "control_frames":len(rows),"completed":len(rows)==(20 if args.smoke else 500) and failure is None,
        "failure_reason":failure,"window_s":[2,10],"window_frames":int(mask.sum()),
        "joint_metrics":{},"vx_mae_mps":float(np.abs(col("vx_mps")[mask]).mean()) if mask.any() else None}
    delta_mask=mask[1:]&mask[:-1]
    second_mask=mask[2:]&mask[1:-1]&mask[:-2]
    for i,name in enumerate(names[:4]):
        sample["joint_metrics"][name]={
            "target_step_rms_rad":rms(np.diff(target[:,i])[delta_mask]),
            "actual_velocity_rms_rad_s":rms(velocity[mask,i]),
            "actual_target_error_rms_rad":rms((actual-target)[mask,i]),
            "action_second_difference_rms":rms(np.diff(action[:,i],n=2)[second_mask]),
            "target_std_rad":float(target[mask,i].std()) if mask.any() else None,
            "action_at_limit_fraction":float((np.abs(action[mask,i])>=.999).mean()) if mask.any() else None}
    sample["wheel_target_step_rms_rad_s"]=[rms(np.diff(wheel_target[:,i])[delta_mask]) for i in range(2)]
    sample["phi0_mean_deg"]=[float(np.rad2deg(col(name)[mask]).mean()) if mask.any() else None for name in ["phi0_left_rad","phi0_right_rad"]]
    sample["pitch_rms_deg"]=rms(np.rad2deg(col("pitch_rad")[mask]))
    with (folder_out/"physics.csv").open(encoding="utf-8",newline="") as f:
        physical=[row for row in csv.DictReader(f) if 2-1e-9<=float(row["time_before_s"])<=10+1e-9]
    for name in names[:4]:
        accel=np.array([float(row[name+"_qacc_rad_s2"]) for row in physical])
        sample["joint_metrics"][name]["actual_acceleration_rms_rad_s2"]=rms(accel)
    results[label]=sample
    traces[label]={"time":time,"target":target,"velocity":velocity,"vx":col("vx_mps")}
result={"schema_version":"MujocoStationaryJointComparisonV1","smoke":args.smoke,
        "command":[0,0,.2],"duration_s":.4 if args.smoke else 10,
        "window":"simulated 2–10s; initial transient excluded",
        "main_comparison":"old_1000 vs new_1000; old_1401 is supplementary",
        "semantics":"Same frozen model, reset, command, runtime. Logging changes no control outputs or simulation steps.",
        "model_xml_sha256":sha256_file(DEFAULT_MODEL),"results":results}
result["report_hash"]=stable_hash(result)
path=args.experiment/("stationary-smoke-comparison.json" if args.smoke else "stationary-comparison.json")
path.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
if not args.smoke:
    np.savez_compressed(args.experiment/"stationary-plot-data.npz", **{label+"_"+key:value for label,trace in traces.items() for key,value in trace.items()})
    subprocess.run([str(project/".venv/Scripts/python.exe"),str(args.experiment/"diagnostics/plot_stationary.py"),"--experiment",str(args.experiment)],check=True)

print(json.dumps({"output":str(path),"completed":{k:v["completed"] for k,v in results.items()}}))
