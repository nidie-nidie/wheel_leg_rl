"""Fixed-input counterfactual sampling and command/physical motion analysis."""
from pathlib import Path
from datetime import datetime
import hashlib,json,sys
import numpy as np
import torch
from tensordict import TensorDict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(r"E:\wheel_leg_rl-main")
PROJECT=ROOT/"wheelleg_dreamwaq"
TRIAL=PROJECT/"artifacts/phase2_dreamwaq/leg-reversal-trial-20261011-v1"
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT/"scripts"))
from export_dreamwaq_actor import _build_policy
from wheelleg_dreamwaq.training.dreamwaq_checkpoint import load_dreamwaq_checkpoint_artifact
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.leg_reversal import LegReversalConfig,LegTargetReversalCounter
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.control import ControlLimits,compute_action_targets
torch.set_num_threads(1)
REPLICAS=64
SEED=8642011
DT=.02
KEEP=np.arange(100,500)
VERIFICATION={"independent_counter_max_error":0.0,"counter_joint_frames":0,"native_export_mean_max_error":0.0,"target_mapping_max_error":0.0}

def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def write(p,v): Path(p).write_bytes((json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8"))

def numpy_counter(targets,nominal):
    targets=np.asarray(targets,dtype=np.float32)
    t,n,j=targets.shape
    extreme=np.repeat(np.array(nominal,dtype=np.float32)[None,:],n,axis=0)
    direction=np.zeros((n,j),np.int8)
    ring=np.zeros((50,n,j),bool)
    counts=np.zeros((n,j),np.int32)
    rates=np.zeros_like(targets)
    events=np.zeros_like(targets,dtype=bool)
    amplitudes=np.zeros_like(targets)
    for k,value in enumerate(targets):
        peak=np.where(direction>0,np.maximum(extreme,value),np.where(direction<0,np.minimum(extreme,value),extreme))
        reverse=((direction>0)&(value<=peak-np.float32(.01)))|((direction<0)&(value>=peak+np.float32(.01)))
        initial_up=(direction==0)&(value>=peak+np.float32(.01))
        initial_down=(direction==0)&(value<=peak-np.float32(.01))
        amplitudes[k]=np.where(reverse,np.abs(value-peak),0)
        events[k]=reverse
        direction=np.where(reverse,-direction,np.where(initial_up,1,np.where(initial_down,-1,direction))).astype(np.int8)
        extreme=np.where(reverse|initial_up|initial_down,value,peak)
        cursor=k%50
        counts-=ring[cursor]; ring[cursor]=reverse; counts+=reverse
        rates[k]=counts
    return rates,events,amplitudes

def checked_counter(targets,nominal):
    rates,events,amplitudes=numpy_counter(targets,nominal)
    counter=LegTargetReversalCounter(targets.shape[1],LegReversalConfig(),"cpu")
    counter.reset(torch.arange(targets.shape[1]),torch.tensor(nominal,dtype=torch.float32).expand(targets.shape[1],-1))
    for k in range(len(targets)):
        actual=counter.update(torch.from_numpy(targets[k].copy()))
        err=float(np.abs(counter.rate.numpy()-rates[k]).max())
        VERIFICATION["independent_counter_max_error"]=max(VERIFICATION["independent_counter_max_error"],err)
        assert np.array_equal(actual.numpy(),events[k])
    VERIFICATION["counter_joint_frames"]+=int(targets.shape[0]*targets.shape[1]*4)
    assert VERIFICATION["independent_counter_max_error"]==0.
    return rates,events,amplitudes

def targets_from_actions(actions,record):
    limits=ControlLimits(leg_action_scale=record["leg_action_scale"],
        leg_target_lower=tuple(record["target_lower"]),leg_target_upper=tuple(record["target_upper"]))
    _,targets,_=compute_action_targets(actions,torch.tensor(record["q_nominal"],dtype=torch.float32),limits)
    return targets.numpy().copy()

def peak_frequency(x):
    x=np.asarray(x,dtype=np.float64)
    centered=x-x.mean()
    if np.std(centered)<1e-7: return None
    power=np.abs(np.fft.rfft(centered*np.hanning(len(x))))**2
    return float(np.fft.rfftfreq(len(x),DT)[1+np.argmax(power[1:])])

def vector_rms(x): return np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2,axis=0)).tolist()
def corr(x,y):
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-10 and np.std(y)>1e-10 else None

def summarize_targets(targets,nominal):
    rates,events,amplitudes=checked_counter(targets,nominal)
    valid=rates[KEEP]
    per_replica=valid.mean(axis=0,dtype=np.float64)
    deltas=targets[KEEP]-targets[KEEP-1]
    centered=targets[KEEP]-targets[KEEP].mean(axis=0,keepdims=True)
    detail={"replicas":targets.shape[1],"target_rate_mean_per_s":per_replica.mean(axis=0).tolist(),
        "target_rate_replica_p05":np.percentile(per_replica,5,axis=0).tolist(),
        "target_rate_replica_p95":np.percentile(per_replica,95,axis=0).tolist(),
        "per_replica_target_rate":per_replica.tolist(),
        "target_step_rms_rad":np.sqrt(np.mean(deltas.astype(np.float64)**2,axis=(0,1))).tolist(),
        "target_centered_rms_rad":np.sqrt(np.mean(centered.astype(np.float64)**2,axis=(0,1))).tolist(),
        "mean_event_excursion_rad":[],"event_excursion_p95_rad":[],"budgets":{}}
    for j in range(4):
        excursion=amplitudes[KEEP,:,j][events[KEEP,:,j]]
        detail["mean_event_excursion_rad"].append(float(excursion.mean()) if excursion.size else None)
        detail["event_excursion_p95_rad"].append(float(np.percentile(excursion,95)) if excursion.size else None)
    for budget in (12,8):
        excess=np.maximum(valid/budget-1.,0.)
        means=excess.mean(axis=(0,1),dtype=np.float64)
        detail["budgets"][str(budget)]={"mean_excess":means.tolist(),
            "max_joint_mean_excess":float(means.max()),
            "hypothetical_gain_from_this_slice":float(min(means.max()**2,1)),
            "full_gain_penalty_rate_per_s":float(.02*means.sum()),
            "fraction_joint_samples_over_budget":(valid>budget).mean(axis=(0,1)).tolist()}
    return detail

def training_events(arm):
    ea=EventAccumulator(arm["run_dir"],size_guidance={"scalars":0}); ea.Reload()
    output={}
    for tag in ea.Tags()["scalars"]:
        if tag.startswith("Reversal/") or tag in ("Reward/leg_reversal","Reward/action_smooth","Policy/mean_noise_std"):
            values=[p.value for p in ea.Scalars(tag)]
            output[tag]={"count":len(values),"last":values[-1],"last40_mean":float(np.mean(values[-40:]))}
            if "gain" in tag: output[tag]["fraction_at_one"]=float(np.mean(np.array(values)>=.999999))
    return output

def main():
    captures=read(OUT/"capture-verification.json")
    assert captures["completed"] and len(captures["captures"])==9
    manifest=read(TRIAL/"experiment-manifest.json")
    standardized_noise=torch.randn(500,REPLICAS,6,generator=torch.Generator().manual_seed(SEED))
    np.savez_compressed(OUT/"standardized-noise.npz",noise=standardized_noise.numpy())
    rows=[]; policies={}; gains=[]
    for arm in manifest["arms"]:
        payload,_,_=load_dreamwaq_checkpoint_artifact(Path(arm["checkpoint"]))
        policies[arm["name"]]=_build_policy(payload["model_state_dict"])
        gains.append({"name":arm["name"],"events":training_events(arm),"checkpoint_gain":arm["gain_state"],
            "effective_action_std":policies[arm["name"]].action_std.detach().numpy().tolist()})
    for record in captures["captures"]:
        assert sha(record["capture"])==record["capture_sha256"]
        assert sha(record["reference"])==record["reference_sha256"]
        data=np.load(record["capture"])
        policy=policies[record["arm"]]
        mean=torch.tensor(data["mean_action"],dtype=torch.float32)
        std=policy._effective_std(mean[:1])[0].detach()
        for index in (0,100,250,499):
            history=torch.tensor(data["history"][index:index+1])
            obs=TensorDict({"policy":history[:,-25:],"policy_history":history,"critic":torch.zeros(1,41)},batch_size=[1])
            with torch.inference_mode(): native=policy.act_inference(obs)
            err=float((native[0]-mean[index]).abs().max())
            VERIFICATION["native_export_mean_max_error"]=max(VERIFICATION["native_export_mean_max_error"],err)
            assert err<=1e-7
            policy._set_distribution(native)
            assert torch.equal(policy.action_std[0],std)
        mean_targets=targets_from_actions(mean[:,None,:],record)
        err=float(np.abs(mean_targets[:,0]-data["leg_target"]).max())
        VERIFICATION["target_mapping_max_error"]=max(VERIFICATION["target_mapping_max_error"],err)
        assert err<1e-7
        no_noise=summarize_targets(mean_targets,record["q_nominal"])
        sampled=mean[:,None,:]+standardized_noise*std
        with_noise=summarize_targets(targets_from_actions(sampled,record),record["q_nominal"])
        held_index=250
        held_mean=mean[held_index].view(1,1,6).expand(500,1,6)
        fixed_no_noise=summarize_targets(targets_from_actions(held_mean,record),record["q_nominal"])
        fixed_with_noise=summarize_targets(targets_from_actions(held_mean+standardized_noise*std,record),record["q_nominal"])
        actual={"velocity_rms_rad_s":vector_rms(data["v_after"][KEEP,:4]),
            "acceleration_rms_rad_s2":vector_rms((data["v_after"][KEEP,:4]-data["v_before"][KEEP,:4])/DT),
            "position_centered_rms_rad":vector_rms(data["q_after"][KEEP,:4]-data["q_after"][KEEP,:4].mean(axis=0)),
            "target_peak_hz":[peak_frequency(mean_targets[KEEP,0,j]) for j in range(4)],
            "velocity_peak_hz":[peak_frequency(data["v_after"][KEEP,j]) for j in range(4)],
            "position_peak_hz":[peak_frequency(data["q_after"][KEEP,j]) for j in range(4)],"one_second_windows":[]}
        for start in range(100,500,50):
            indices=np.arange(start,start+50)
            rates,_,_=numpy_counter(mean_targets,record["q_nominal"])
            actual["one_second_windows"].append({"post_time_start":float(data["time_s"][start]),"post_time_end":float(data["time_s"][start+49]),
                "target_rate":rates[indices,0].mean(axis=0).tolist(),
                "target_step_rms":vector_rms(mean_targets[indices,0]-mean_targets[indices-1,0]),
                "velocity_rms":vector_rms(data["v_after"][indices,:4]),
                "acceleration_rms":vector_rms((data["v_after"][indices,:4]-data["v_before"][indices,:4])/DT)})
        sweeps={}
        if record["scenario"]=="nominal_stand":
            for factor in (.25,.5):
                sweeps[str(factor)]=summarize_targets(targets_from_actions(mean[:,None,:]+standardized_noise*std*factor,record),record["q_nominal"])
        row={"arm":record["arm"],"scenario":record["scenario"],"input_capture":record["capture"],"window":"post-action time 2s<t<=10s, 400 frames",
            "effective_action_std":std.tolist(),"mean_action_p05":np.percentile(mean.numpy()[KEEP],5,axis=0).tolist(),
            "mean_action_p50":np.percentile(mean.numpy()[KEEP],50,axis=0).tolist(),"mean_action_p95":np.percentile(mean.numpy()[KEEP],95,axis=0).tolist(),
            "mean_action_saturation_fraction":(np.abs(mean.numpy()[KEEP])>=1).mean(axis=0).tolist(),
            "sampled_action_saturation_fraction":(np.abs(sampled.numpy()[KEEP])>=1).mean(axis=(0,1)).tolist(),
            "mean_only":no_noise,"same_histories_with_noise":with_noise,
            "held_input":{"index":held_index,"pre_action_time_s":held_index*DT,"raw_mean_action":mean[held_index].tolist(),
                "mean_only":fixed_no_noise,"with_noise":fixed_with_noise},
            "sigma_sweep":sweeps,"actual":actual}
        rows.append(row)
        print(json.dumps({"arm":record["arm"],"scenario":record["scenario"],"mean_rate":no_noise["target_rate_mean_per_s"][:2],
            "sampled_rate":with_noise["target_rate_mean_per_s"][:2],"held_noise_rate":fixed_with_noise["target_rate_mean_per_s"][:2],
            "mean_target_step_rms":no_noise["target_step_rms_rad"][:2],"mean_event_excursion":no_noise["mean_event_excursion_rad"][:2]}),flush=True)
    template=captures["captures"][0]
    t=np.arange(500)*DT
    sine=[]
    for amplitude in (.02,.20):
        targets=np.array(template["q_nominal"],dtype=np.float32)[None,None,:]+(amplitude*np.sin(2*np.pi*5*t)).astype(np.float32)[:,None,None]
        metrics=summarize_targets(targets,template["q_nominal"])
        sine.append({"amplitude_rad":amplitude,"frequency_hz":5.,"metrics":metrics})
    assert np.allclose(sine[0]["metrics"]["target_rate_mean_per_s"],sine[1]["metrics"]["target_rate_mean_per_s"],atol=1e-7)
    assert np.allclose(np.array(sine[1]["metrics"]["target_step_rms_rad"])/np.array(sine[0]["metrics"]["target_step_rms_rad"]),10.,rtol=1e-5)
    grid=[]
    for raw_mean in (-3.,-2.,-1.,-.5,0.,.5,1.,2.,3.):
        mean=torch.full((500,1,6),raw_mean)
        std=policies["budget12"].raw_std.detach().abs().clamp(1e-4,2.)
        metrics=summarize_targets(targets_from_actions(mean+standardized_noise*std,template),template["q_nominal"])
        grid.append({"raw_action_mean":raw_mean,"metrics":metrics})
    results={"diagnostic_only":True,"completed_at":datetime.now().astimezone().isoformat(),"frames_replayed":4500,
        "sampling_replicas":REPLICAS,"sampling_seed":SEED,"shared_standardized_noise_sha256":sha(OUT/"standardized-noise.npz"),
        "verification":VERIFICATION,"training_logs":gains,"comparisons":rows,"same_frequency_different_amplitude":sine,
        "fixed_mean_sampling_grid":grid,
        "limitations":["Same-history sampled actions were not applied to physics; fixed-input conditional experiment, not real training rollout.",
            "MuJoCo histories contain no injected training observation noise; no causal decomposition of full training.",
            "Per-replica quantiles reflect Monte Carlo sampling uncertainty on one observed trajectory, not independent physical trials.",
            "No stochastic feedback propagation, reset, command/randomization changes or training in this diagnosis.",
            "Target rate and joint velocity are associated in closed loop; association is not a complete physical root-cause proof."]}
    write(OUT/"results.json",results)
    print(json.dumps({"verification":VERIFICATION,"synthetic_rates":[x["metrics"]["target_rate_mean_per_s"] for x in sine]}),flush=True)
if __name__=="__main__": main()
