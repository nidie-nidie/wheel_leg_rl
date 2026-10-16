from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,json,sys
project=Path(r"E:\wheel_leg_rl-main\wheelleg_dreamwaq")
sys.path.insert(0,str(project/"scripts"))
from run_training_suite import _stable_hash,_write_json,_assert_suite_source_fingerprint
parser=argparse.ArgumentParser()
parser.add_argument("--experiment",type=Path,required=True)
parser.add_argument("--preflight",action="store_true")
args=parser.parse_args()
record=json.loads((args.experiment/"experiment-manifest.json").read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def read(p):
    value=json.loads(Path(p).read_text(encoding="utf-8"))
    if "report_hash" in value:
        assert value["report_hash"]==_stable_hash({k:v for k,v in value.items() if k!="report_hash"})
    return value
def diff(a,b,p=""):
    if isinstance(a,dict) and isinstance(b,dict):return [v for k in sorted(set(a)|set(b)) for v in diff(a.get(k),b.get(k),p+"."+k)]
    return [] if a==b else [{"path":p,"before":a,"after":b}]
old_manifest=read(Path(record["baseline_record"]["checkpoint"]).parent/"run_manifest.json")
old_isaac=read(record["baseline_record"]["isaac_evaluation"])
new_run=read(args.experiment/"live-configuration-verification.json")["run_dir"]
new_manifest=read(Path(new_run)/"run_manifest.json")
changed=diff(old_manifest["base_task_contract"],new_manifest["base_task_contract"])
assert {v["path"] for v in changed}=={".contract_hash",".task.reward_weights.action_smooth"}
assert old_manifest["seed"]==new_manifest["seed"]==41882943
old_hash=old_manifest["base_task_contract"]["contract_hash"]
new_hash=new_manifest["base_task_contract"]["contract_hash"]
if args.preflight:
    acceptance=read(project/"artifacts/phase2_dreamwaq/action-smooth-acceptance-20261011/acceptance.json")
    new_isaac=read(acceptance["integration_evidence"]["isaac_evaluation"])
else:
    assert record["completed_iterations"]==1000 and not record["evaluation_errors"]
    new_isaac=read(args.experiment/"isaac_evaluation/summary.json")
a,b=old_isaac["evaluation_contract"],new_isaac["evaluation_contract"]
for contract,base_hash in [(a,old_hash),(b,new_hash)]:
    assert contract["contract_hash"]==_stable_hash({k:v for k,v in contract.items() if k!="contract_hash"})
    assert contract["base_task_contract_hash"]==base_hash
left={k:v for k,v in a.items() if k not in {"base_task_contract_hash","contract_hash"}}
right={k:v for k,v in b.items() if k not in {"base_task_contract_hash","contract_hash"}}
if args.preflight:
    # Smoke fixture uses a separate file; same physical tensors remain mandatory.
    for value in [left,right]:
        value["reset_cache"]={k:v for k,v in value["reset_cache"].items() if k not in {"path","file_sha256","identity_hash"}}
assert left==right,diff(left,right)
if args.preflight:
    _write_json(args.experiment/"comparison-preflight.json",{
        "passed":True,"only_training_configuration_change":changed,
        "isaac_evaluation_conditions_same":True,"initial_cache_tensor_sha256":a["reset_cache"]["tensor_sha256"],
        "reason_for_distinct_full_contract_hash":"Training base-task identity includes the intentionally changed reward.",
        "final_comparison_requires_identical_cache_file":True})
    print("PASS: reward identity differs, actual evaluation conditions and reset tensors match.")
    raise SystemExit(0)
_assert_suite_source_fingerprint(record["source_fingerprint_hash"])
assert record["architecture_sha256"]==sha(project.parent/"docs/2026-10-03-wheelleg-dreamwaq-architecture.md")
assert sha(record["baseline_record"]["checkpoint"])==record["baseline_checkpoint_sha256"]
comparisons={"identity_verification":{"base_contract_changes":changed,"isaac_protocol_and_cache_identical":True}}
for label,key in [("isaac","isaac_evaluation"),("mujoco","mujoco_evaluation")]:
    new=read(args.experiment/(key+"/summary.json"));old=read(record["baseline_record"][key])
    assert new["source_checkpoint_sha256"]==sha(record["final_checkpoint"])
    if label=="mujoco":
        a,b=old["evaluation_contract"],new["evaluation_contract"]
        for value,expected in [(a,old_hash),(b,new_hash)]:
            assert value["task_contract_hash"]==value["base_task_contract_hash"]==expected
        ignored={"training_suite","task_contract_hash","base_task_contract_hash"}
        assert {k:v for k,v in a.items() if k not in ignored}=={k:v for k,v in b.items() if k not in ignored}
        assert b["training_suite"]["suite_mode"]=="single_seed_reward_diagnostic"
        assert b["training_suite"]["run_count"]==1 and b["training_suite"]["iterations_per_run"]==1000
        assert b["training_suite"]["context_hash"]==_stable_hash({k:v for k,v in b["training_suite"].items() if k!="context_hash"})
        comparisons["identity_verification"]["mujoco_protocol_model_failure_scoring_identical"]=True
    previous={row["name"]:row for row in old["aggregate"]["scenarios"]}
    rows=[]
    for row in new["aggregate"]["scenarios"]:
        before=previous[row["name"]]
        entry={"name":row["name"],"completed_before":before["completed"],"completed_after":row["completed"]}
        for metric in ("vx_mae","yaw_rate_mae","height_mae","phi0_delta_rms","action_saturation_fraction","effort_saturation_fraction"):
            if row.get(metric) is not None and before.get(metric) is not None:
                entry[metric]={"before":before[metric],"after":row[metric],"delta":row[metric]-before[metric]}
        rows.append(entry)
    comparisons[label]=rows
    for entry in new.get("scenario_files",{}).values():
        assert sha(entry["path"])==entry["sha256"]
comparisons["new_command_practice"]=read(args.experiment/"mujoco_command_practice/summary.json")
comparisons["old_command_practice"]=read(record["baseline_record"]["command_practice_evaluation"])
assert comparisons["new_command_practice"]["command_practice_contract"]==comparisons["old_command_practice"]["command_practice_contract"]
stationary=read(args.experiment/"stationary-comparison.json")
assert not stationary["smoke"] and stationary["results"]["new_1000"]["completed_iterations"]==1000
assert stationary["results"]["old_1000"]["completed_iterations"]==1000
_write_json(args.experiment/"movement-comparison.json",comparisons)
stderr=(args.experiment/"wrapper.stderr.log").read_text(encoding="utf-8",errors="replace")
if stderr:
    assert "AssertionError" in stderr and 'new["evaluation_contract"]==old["evaluation_contract"]' in stderr,stderr
record.update(status="completed_diagnostic",finished_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
              movement_comparison=str(args.experiment/"movement-comparison.json"),
              comparison_identity_verification=comparisons["identity_verification"],
              comparison_check_correction={
                  "original":"Whole evaluation contracts required equality although training reward identity changed.",
                  "corrected":"Validate each own full identity and compare all actual evaluation conditions, with identical Isaac cache.",
                  "training_or_evaluation_parameters_changed":False,
                  "orchestrator_stderr_retained":bool(stderr)})
_write_json(args.experiment/"experiment-manifest.json",record)
_write_json(project.parent/".git/action-smooth-process.json",record)
print(json.dumps({"status":record["status"],"output":str(args.experiment)},indent=2))
