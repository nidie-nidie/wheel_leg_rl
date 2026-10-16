from pathlib import Path
import csv,json,sys
import numpy as np
import torch
root=Path(r"E:\wheel_leg_rl-main")
project=root/"wheelleg_dreamwaq"
sys.path.insert(0,str(project/"scripts"))
sys.path.insert(0,str(project/"artifacts/debug/sim2sim/interactive-mujoco-20261010-v1"))
from evaluate_mujoco import DEFAULT_MODEL,DEFAULT_MODEL_MANIFEST,_failure_reason
from joint_telemetry import JointTelemetry
from wheelleg_mujoco.contract import load_policy_contract,sha256_file
from wheelleg_mujoco.runner import WheelLegMujocoRuntime
from wheelleg_mujoco.command_practice import command_at_tick,validate_command_practice
loc=json.loads((Path(__file__).resolve().parent.parent/"input-identity.json").read_text(encoding="utf-8"))
out=Path(loc["output"]);exp=Path(loc["experiment"])
actorpath=exp/"export/actor.ts"
contract,manifest,model_manifest=load_policy_contract(exp/"export/policy_manifest.json",actorpath,DEFAULT_MODEL_MANIFEST)
assert sha256_file(DEFAULT_MODEL)==model_manifest["model_xml"]["sha256"]
actor=torch.jit.load(str(actorpath),map_location="cpu").eval()
payload=validate_command_practice(manifest)
runtime=WheelLegMujocoRuntime(DEFAULT_MODEL,contract,model_manifest)
reports=[]
for name,initial,is_practice in [("forward",[1.,0.,.2],False),("reverse",[-1.,0.,.2],False),("forward_stop_reverse",[.5,0.,.2],True)]:
    initial=np.array(initial,dtype=np.float64)
    obs=runtime.reset(initial)
    logger=JointTelemetry(runtime,out/"supplemental-traces"/name,manifest)
    logger.reset(initial)
    failure=None
    try:
        for tick in range(500):
            current=command_at_tick(initial,tick,payload) if is_practice else initial
            next_command=command_at_tick(initial,tick+1,payload) if is_practice else initial
            with torch.inference_mode():
                tensor=torch.from_numpy(obs).unsqueeze(0)
                action=actor(tensor).cpu().numpy().reshape(-1)
                estimated=float(actor.encoder(tensor)[0,0]/manifest["normalization"]["root_linear_velocity_scale"])
            logger.begin_control(action,current,obs,estimated)
            result=runtime.step(action,next_command)
            logger.end_control(result)
            failure=_failure_reason(runtime,result.metrics)
            if failure:
                logger.event("evaluation_failure",reason=failure);break
            obs=result.observation
    finally:logger.close()
    tracepath=out/"supplemental-traces"/name/"control.csv"
    with tracepath.open(encoding="utf-8",newline="") as f:actual=list(csv.DictReader(f))
    referencepath=exp/("mujoco_command_practice" if is_practice else "mujoco_evaluation")/(name+".csv")
    with referencepath.open(encoding="utf-8",newline="") as f:reference=list(csv.DictReader(f))
    assert len(actual)==len(reference)==500 and failure is None
    maxerror={"raw_action":0.,"vx":0.,"yaw_rate":0.,"base_height":0.,"command":0.,"clipped_action":0.}
    for a,b in zip(actual,reference):
        for i,joint in enumerate(runtime.model_map.joint_names):
            maxerror["raw_action"]=max(maxerror["raw_action"],abs(float(a[joint+"_action_raw"])-float(b["raw_action_"+str(i)])))
            reference_clipped=np.clip(float(b["raw_action_"+str(i)]),-1.,1.) if is_practice else float(b["clipped_action_"+str(i)])
            maxerror["clipped_action"]=max(maxerror["clipped_action"],abs(float(a[joint+"_action_clipped"])-reference_clipped))
        for key,kind in [("vx_mps","vx"),("yaw_rate_rad_s","yaw_rate"),("base_height_m","base_height")]:
            maxerror[kind]=max(maxerror[kind],abs(float(a[key])-float(b[key])))
        for akey,bkey in [("command_vx_mps","target_vx_mps"),("command_yaw_rad_s","target_yaw_rate_rad_s"),("command_height_m","target_height_m" if is_practice else "target_base_height_m")]:
            maxerror["command"]=max(maxerror["command"],abs(float(a[akey])-float(b[bkey])))
    assert max(maxerror.values())<1e-8,(name,maxerror)
    reports.append({"scenario":name,"completed_frames":500,"failure_reason":failure,"reference":str(referencepath),
        "reference_sha256":sha256_file(referencepath),"max_abs_difference_vs_existing_evaluation":maxerror,"clipped_action_reference":"clip(original raw action, -1, +1)" if is_practice else "original logged clipped_action columns",
        "source_checkpoint_sha256":manifest["source_checkpoint_sha256"],"policy_sha256":sha256_file(actorpath),"model_sha256":sha256_file(DEFAULT_MODEL),
        "new_telemetry":str(tracepath),"purpose":"diagnostic joint telemetry only, not an additional training run or formal suite"})
(out/"supplemental-capture-verification.json").write_text(json.dumps(reports,indent=2)+"\n",encoding="utf-8")
print(json.dumps(reports))
