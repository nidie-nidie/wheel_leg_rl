from pathlib import Path
from datetime import datetime,timezone,timedelta
import contextlib,hashlib,json,os,sys,traceback
import torch
project=Path(r"E:\wheel_leg_rl-main\wheelleg_dreamwaq")
sys.path.insert(0,str(project/"scripts"))
from run_training_suite import _stream_process,_monitor_report,_run_checked,_assert_suite_source_fingerprint,_stable_hash,_write_json
from run_dreamwaq_training_suite import _training_fingerprint
locator=project.parent/".git/action-smooth-process.json"
record=json.loads(locator.read_text(encoding="utf-8"))
output=Path(record["output"])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save():
    _write_json(output/"experiment-manifest.json",record)
    _write_json(locator,record)
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def read(path):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    if "report_hash" in data:
        assert data["report_hash"]==_stable_hash({k:v for k,v in data.items() if k!="report_hash"}),path
    return data
def differences(a,b,p=""):
    if isinstance(a,dict) and isinstance(b,dict):
        result=[]
        for key in sorted(set(a)|set(b)):
            result+=differences(a.get(key),b.get(key),p+"."+key)
        return result
    return [] if a==b else [{"path":p,"before":a,"after":b}]
_assert_suite_source_fingerprint(record["source_fingerprint_hash"])
assert record["architecture_sha256"]==sha(project.parent/"docs/2026-10-03-wheelleg-dreamwaq-architecture.md")
assert sha(record["baseline_record"]["checkpoint"])==record["baseline_checkpoint_sha256"]
assert sha(record["diagnostic_entrypoint"]["diagnostic"])==record["diagnostic_entrypoint"]["diagnostic_sha256"]
if "--validate-only" in sys.argv:
    print("READY: independent fresh seed41882943/1000 and frozen sources validated.")
    raise SystemExit(0)
record.update(status="training",pid=os.getpid(),started_at=now())
save()
try:
    with (output/"orchestrator.log").open("w",encoding="utf-8") as stream,contextlib.redirect_stdout(stream):
        code,run_dir=_stream_process(record["training_command"],output/"training.log",
                                    monitor_dir=output/"monitor",monitor_interval_s=1800)
    record.update(training_return_code=code,run_dir=None if run_dir is None else str(run_dir))
    save()
    assert code==0 and run_dir is not None,(code,run_dir)
    _assert_suite_source_fingerprint(record["source_fingerprint_hash"])
    summary=read(run_dir/"training_summary.json")
    manifest=read(run_dir/"run_manifest.json")
    assert summary["starting_completed_iterations"]==0 and summary["iterations_executed"]==1000
    assert summary["completed_iterations"]==1000 and summary["resumed_from"] is None
    assert manifest["seed"]==41882943 and manifest["num_envs"]==256 and manifest["num_mini_batches"]==8
    assert manifest["resume_provenance"]=={"schema_version":"ResumeMetadataV2","mode":"fresh"}
    assert manifest["task_profile"]=="stop_reverse_smooth_v1"
    old_manifest=read(Path(record["baseline_record"]["checkpoint"]).parent/"run_manifest.json")
    changed=differences(old_manifest["base_task_contract"],manifest["base_task_contract"])
    assert {row["path"] for row in changed}=={".contract_hash",".task.reward_weights.action_smooth"},changed
    assert manifest["dreamwaq_algorithm_contract"]==old_manifest["dreamwaq_algorithm_contract"]
    assert manifest["dreamwaq_export_contract"]==old_manifest["dreamwaq_export_contract"]
    fingerprint,fingerprint_hash=_training_fingerprint(manifest)
    _write_json(output/"training-fingerprint.json",fingerprint)
    monitor=_monitor_report(output/"monitor")
    assert not monitor["hard_anomalies"],monitor
    record.update(status="trained",final_checkpoint=summary["final_checkpoint"],completed_iterations=1000,
                  training_duration_seconds=summary["duration_seconds"],context_acceptance=summary["context_acceptance"],
                  training_fingerprint_hash=fingerprint_hash,monitor=monitor,
                  only_base_contract_changes=changed,finished_training_at=now())
    save()
except Exception as error:
    record.update(status="training_failed",error=f"{type(error).__name__}: {error}")
    (output/"training-failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
    save()
    raise

context={"schema_version":"DreamWaQTrainingEvaluationContextV1","suite_mode":"single_seed_reward_diagnostic",
         "suite_id":output.name,"run_count":1,"iterations_per_run":1000,
         "training_fingerprint_hash":record["training_fingerprint_hash"],
         "suite_source_fingerprint_hash":record["source_fingerprint_hash"],
         "diagnostic_entrypoint":record["diagnostic_entrypoint"]}
context["context_hash"]=_stable_hash(context)
_write_json(output/"evaluation-context.json",context)
export=output/"export"; isaac=output/"isaac_evaluation"; mj=output/"mujoco_evaluation"; dynamic=output/"mujoco_command_practice"
mjpython=str(project/"sim2sim/mujoco/.venv/Scripts/python.exe")
reset_cache=project/"artifacts/phase2_dreamwaq/training-suite-20261011-stop-reverse-v1/isaac_evaluation/evaluation-reset-cache.pt"
stages=[
("export",[sys.executable,"scripts/export_dreamwaq_actor.py","--checkpoint",record["final_checkpoint"],"--output",str(export)],export/"policy_manifest.json"),
("isaac",[sys.executable,"scripts/evaluate_isaac.py","--checkpoint",record["final_checkpoint"],"--candidate-only","--reset-cache",str(reset_cache),"--output",str(isaac),"--headless"],isaac/"summary.json"),
("mujoco",[mjpython,record["diagnostic_entrypoint"]["diagnostic"],"--policy",str(export/"actor.ts"),"--manifest",str(export/"policy_manifest.json"),"--output",str(mj),"--suite-context",str(output/"evaluation-context.json"),"--run-index","1","--completed-iterations","1000"],mj/"summary.json"),
("command_practice",[mjpython,"scripts/evaluate_command_practice_mujoco.py","--policy",str(export/"actor.ts"),"--manifest",str(export/"policy_manifest.json"),"--output",str(dynamic)],dynamic/"summary.json"),
("stationary_comparison",[mjpython,str(output/"diagnostics/compare_stationary.py"),"--experiment",str(output)],output/"stationary-comparison.json")]
record.update(status="evaluating",evaluation_errors={})
save()
for label,command,expected in stages:
    record["evaluation_stage"]=label
    save()
    try:
        _assert_suite_source_fingerprint(record["source_fingerprint_hash"])
        with (output/"evaluation-orchestrator.log").open("a",encoding="utf-8") as stream,contextlib.redirect_stdout(stream):
            _run_checked(command,output/(label+".log"))
        assert expected.is_file(),expected
        if label=="export":
            policy=read(expected)
            assert policy["source_completed_iterations"]==1000 and policy["source_seed"]==41882943
            assert policy["training_reward_weights"]["action_smooth"]==-.02
            assert policy["source_checkpoint_sha256"]==sha(record["final_checkpoint"])
            assert policy["actor_sha256"]==sha(export/"actor.ts")
            assert policy["golden_vectors_sha256"]==sha(export/"golden_vectors.pt")
            golden=torch.load(export/"golden_vectors.pt",map_location="cpu",weights_only=False)
            actor=torch.jit.load(str(export/"actor.ts"),map_location="cpu").eval()
            with torch.inference_mode():
                error=float((actor(golden["history"])-golden["expected_action_mean"]).abs().max())
                shapes={str(n):list(actor(golden["history"][:n]).shape) for n in (1,3,32)}
            assert error<=1e-7 and shapes=={"1":[1,6],"3":[3,6],"32":[32,6]}
            record["golden_verification"]={"max_error":error,"batch_shapes":shapes}
        record[label+"_artifact"]=str(expected)
    except Exception as error:
        record["evaluation_errors"][label]=f"{type(error).__name__}: {error}"
        (output/(label+"-failure.txt")).write_text(traceback.format_exc(),encoding="utf-8")
        if label=="export":
            save()
            break
    save()
if not record["evaluation_errors"]:
    comparisons={}
    for label,folder,key in [("isaac",isaac,"isaac_evaluation"),("mujoco",mj,"mujoco_evaluation")]:
        new=read(folder/"summary.json"); old=read(record["baseline_record"][key])
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
        if label=="isaac":
            assert new["evaluation_contract"]==old["evaluation_contract"]
        else:
            assert {k:v for k,v in new["evaluation_contract"].items() if k!="training_suite"}=={k:v for k,v in old["evaluation_contract"].items() if k!="training_suite"}
    comparisons["new_command_practice"]=read(dynamic/"summary.json")
    comparisons["old_command_practice"]=read(record["baseline_record"]["command_practice_evaluation"])
    assert comparisons["new_command_practice"]["command_practice_contract"]==comparisons["old_command_practice"]["command_practice_contract"]
    _write_json(output/"movement-comparison.json",comparisons)
    record.update(status="completed_diagnostic",finished_at=now(),movement_comparison=str(output/"movement-comparison.json"))
else:
    record.update(status="evaluation_incomplete",finished_at=now())
_assert_suite_source_fingerprint(record["source_fingerprint_hash"])
save()
print(json.dumps({"status":record["status"],"output":str(output),"errors":record["evaluation_errors"]}),flush=True)
