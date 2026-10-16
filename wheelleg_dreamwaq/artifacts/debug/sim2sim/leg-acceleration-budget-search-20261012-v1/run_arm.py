from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
PY = ROOT / ".venv/Scripts/python.exe"
MJ = ROOT / "sim2sim/mujoco/.venv/Scripts/python.exe"
SOURCE = ROOT / "logs/rsl_rl/wheelleg_flat_dreamwaq/2026-10-11_21-56-09_rtx5070_fudan-v1_leg-reversal-trial-baseline-200/model_199.pt"
BASELINE = ROOT / "logs/rsl_rl/wheelleg_flat_dreamwaq/2026-10-12_01-27-01_rtx5070_fudan-v1_leg-accel-budget-search-20261012-baseline-200"
SEED=41882943

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()

def run(label, args):
    log = OUT / (label + ".log")
    print("START " + label, flush=True)
    with log.open("wb") as stream:
        result=subprocess.run([str(x) for x in args], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        print(log.read_text(encoding="utf-8", errors="replace")[-5000:], flush=True)
        raise RuntimeError(f"{label} failed: {result.returncode}")
    print("DONE " + label, flush=True)

def train(label, profile, iterations=200, resume=None):
    name="leg-accel-budget-search-20261012-" + label
    args=[PY,"scripts/train_dreamwaq.py","--profile","rtx5070","--max-iterations",iterations,
          "--seed",SEED,"--task-profile",profile,"--run-name",name,"--headless"]
    args += ["--resume", resume] if resume else ["--initialize-from",SOURCE]
    run(label+"-train",args)
    matches=list((ROOT/"logs/rsl_rl/wheelleg_flat_dreamwaq").glob("*_"+name))
    if len(matches)!=1:
        raise RuntimeError(f"ambiguous run: {matches}")
    summary=json.loads((matches[0]/"training_summary.json").read_text())
    if summary["completed_iterations"]!=iterations:
        raise RuntimeError("wrong iteration count")
    return matches[0],summary

def export(label, checkpoint):
    out=OUT/label/"export"
    run(label+"-export",[PY,"scripts/export_dreamwaq_actor.py","--checkpoint",checkpoint,"--output",out])
    vectors=torch.load(out/"golden_vectors.pt", map_location="cpu", weights_only=False)
    actor=torch.jit.load(str(out/"actor.ts"), map_location="cpu").eval()
    errors={}
    with torch.inference_mode():
        for batch in (1,3,32):
            expected=vectors["expected_action_mean"][:batch]
            actual=actor(vectors["history"][:batch])
            error=float((expected-actual).abs().max())
            assert actual.shape == (batch,6) and error<=1.e-5
            errors[str(batch)]=error
    (out.parent/"golden-verification.json").write_text(json.dumps(errors,indent=2))
    return out

def evaluate(label, export_dir):
    run(label+"-motion",[MJ,"scripts/evaluate_leg_acceleration_mujoco.py",
        "--policy",export_dir/"actor.ts","--manifest",export_dir/"policy_manifest.json",
        "--output",OUT/label/"motion"])
    run(label+"-command-practice",[MJ,"scripts/evaluate_command_practice_mujoco.py",
        "--policy",export_dir/"actor.ts","--manifest",export_dir/"policy_manifest.json",
        "--output",OUT/label/"command_practice"])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("arm", choices=("baseline","smoke","80","75","70","85","90"))
    arm=parser.parse_args().arm
    assert sha(SOURCE)=="BCDD726CE5D6F89600547F34CA285282053D4D07B07575604172E6A11A5912B7"
    if arm=="smoke":
        fresh, fs=train("smoke-fresh2","stop_reverse_accel80_v1",2)
        restored, rs=train("smoke-resume3","stop_reverse_accel80_v1",3,fs["final_checkpoint"])
        assert rs["starting_completed_iterations"]==2 and rs["iterations_executed"]==1
        assert fs["leg_acceleration_gain_state"]["completed_rollouts"]==2
        assert rs["leg_acceleration_gain_state"]["completed_rollouts"]==3
        package=export("smoke",rs["final_checkpoint"])
        info={"fresh_run":str(fresh),"resume_run":str(restored),"export":str(package),
              "completed_rollouts":[2,3],"golden":json.loads((package.parent/"golden-verification.json").read_text())}
    else:
        if arm=="baseline":
            run_dir=BASELINE
            ts=json.loads((run_dir/"training_summary.json").read_text())
        else:
            run_dir, ts=train("budget"+arm+"-200","stop_reverse_accel"+arm+"_v1")
        package=export(arm,ts["final_checkpoint"])
        evaluate(arm,package)
        info={"run_dir":str(run_dir),"training_summary":ts,"export":str(package),
              "source_checkpoint":str(SOURCE),"source_checkpoint_sha256":sha(SOURCE)}
    (OUT/(arm+"-result.json")).write_text(json.dumps(info,indent=2,sort_keys=True))
    print("FINISHED " + arm,flush=True)

if __name__=="__main__":
    main()

