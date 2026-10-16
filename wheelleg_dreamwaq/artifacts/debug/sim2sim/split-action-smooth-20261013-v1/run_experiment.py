"""Bounded serial training/evaluation worker. All motion tests run in MuJoCo."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RECORD = PROJECT / "artifacts/phase2_dreamwaq/split-action-smooth-20261013-v1"
TRAIN_PY = PROJECT / ".venv/Scripts/python.exe"
MJ_PY = PROJECT / "sim2sim/mujoco/.venv/Scripts/python.exe"
BASE_EXPORT = PROJECT / "artifacts/phase2_dreamwaq/training-suite-20261011-stop-reverse-v1/exports/run-03"
SEED = 41882943
MAIN = ("nominal_stand", "forward", "reverse")
STATE = {"schema_version": "SplitSmoothExperimentV1", "pid": os.getpid(), "status": "running",
         "seed": SEED, "iterations_per_run": 1000, "max_long_training_runs": 2,
         "events": [], "runs": []}


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def save():
    path = OUT / "state.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(STATE, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def freeze_check():
    expected = json.loads((RECORD / "frozen-source.json").read_text(encoding="utf-8"))
    for path, digest in expected.items():
        if sha(PROJECT / path) != digest:
            raise RuntimeError("Code/model changed while experiment running: " + path)


def command(name, argv):
    freeze_check()
    log = OUT / "logs" / (name + ".log")
    log.parent.mkdir(exist_ok=True)
    STATE["stage"] = name
    event = {"stage": name, "argv": [str(x) for x in argv], "start_unix": time.time(), "log": str(log)}
    STATE["events"].append(event)
    save()
    environment = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1",
                       PYTHONIOENCODING="utf-8",
                       PYTHONPATH=str(PROJECT / "sim2sim/mujoco") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen([str(x) for x in argv], cwd=PROJECT, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT)
        STATE["child_pid"] = process.pid
        save()
        last_monitor = time.monotonic()
        while process.poll() is None:
            time.sleep(10)
            if time.monotonic() - last_monitor >= 1800:
                tail = log.read_text(encoding="utf-8", errors="replace")[-30000:]
                matches = re.findall(r"Learning iteration\s+(\d+)\s*/\s*(\d+)", tail)
                snapshot = {"stage": name, "unix": time.time(), "pid": process.pid,
                            "last_iteration": list(matches[-1]) if matches else None,
                            "fatal_marker": bool(re.search(r"Traceback|Reward contains NaN|Non-finite", tail))}
                STATE.setdefault("monitor", []).append(snapshot)
                save()
                last_monitor = time.monotonic()
        event["exit_code"] = process.returncode
    event["end_unix"] = time.time()
    STATE["child_pid"] = None
    save()
    freeze_check()
    if event["exit_code"] != 0:
        raise RuntimeError(name + " failed; see " + str(log))
    return log


def verify_golden(export):
    actor = torch.jit.load(str(export / "actor.ts"), map_location="cpu").eval()
    golden = torch.load(export / "golden_vectors.pt", map_location="cpu", weights_only=False)
    errors = {}
    with torch.inference_mode():
        actual = actor(golden["history"])
        errors[32] = float((actual - golden["expected_action_mean"]).abs().max())
        assert actual.shape == (32, 6) and errors[32] <= 1e-7
        errors[1] = 0.0 if actor(golden["history"][:1]).shape == (1, 6) else float("inf")
        errors[3] = 0.0 if actor(golden["history"][:3]).shape == (3, 6) else float("inf")
        assert errors[1] == errors[3] == 0.0
    return errors


def evaluate(label, export):
    destination = OUT / label
    destination.mkdir(exist_ok=True)
    common = ["--policy", export / "actor.ts", "--manifest", export / "policy_manifest.json"]
    command(label + "-motion20ms", [MJ_PY, "scripts/evaluate_leg_acceleration_mujoco.py", *common,
                                    "--output", destination / "motion20ms"])
    command(label + "-command-practice", [MJ_PY, "scripts/evaluate_command_practice_mujoco.py", *common,
                                         "--output", destination / "command_practice"])
    command(label + "-physics1ms", [MJ_PY, OUT / "sample_physics.py",
                                   "--export", export, "--output", destination / "physics1ms"])
    return {"motion20ms": json.loads((destination / "motion20ms/summary.json").read_text(encoding="utf-8")),
            "command_practice": json.loads((destination / "command_practice/summary.json").read_text(encoding="utf-8")),
            "physics1ms": json.loads((destination / "physics1ms/summary.json").read_text(encoding="utf-8"))}


def compare(baseline, candidate):
    old = {s["scenario"]: s for s in baseline["motion20ms"]["scenarios"]}
    new = {s["scenario"]: s for s in candidate["motion20ms"]["scenarios"]}
    old_ms = {s["scenario"]: s for s in baseline["physics1ms"]["scenarios"]}
    new_ms = {s["scenario"]: s for s in candidate["physics1ms"]["scenarios"]}
    speed_failures, saturation_failures, prolonged = [], [], []
    all_survived = all(s["survived"] for s in new.values())
    valid20 = all(new[n]["metric_valid"] and old[n]["metric_valid"] for n in MAIN)
    valid1 = all(new_ms[n]["survived"] and old_ms[n]["survived"]
                 and new_ms[n]["window_count"] == old_ms[n]["window_count"] == 8 for n in MAIN)
    reductions = {}
    for name in MAIN:
        limit = .10 if name == "nominal_stand" else .15
        current, previous = new[name]["steady_vx_mae_mps"], old[name]["steady_vx_mae_mps"]
        if current is None or previous is None or current > limit or current > previous + max(.1*previous, .03):
            speed_failures.append(name)
        if any(v is not None and v >= .90 for v in new_ms[name]["leg_action_saturation_fraction"]):
            prolonged.append(name)
    for name in new:
        for key in ("action_saturation_fraction", "effort_saturation_fraction"):
            current, previous = new[name][key], old[name][key]
            if current is not None and previous is not None and current > previous + .05:
                saturation_failures.append(name + ":" + key)
    for label, valid, field, reference, observed in (
        ("20ms", valid20, "actual_acceleration_rms_mean_rad_s2", old, new),
        ("1ms", valid1, "rms_mean", old_ms, new_ms)):
        previous = float(np.mean([reference[n][field] for n in MAIN])) if valid else None
        current = float(np.mean([observed[n][field] for n in MAIN])) if valid else None
        reductions[label] = {"baseline": previous, "candidate": current,
                             "reduction_fraction": 1-current/previous if valid and previous > 0 else None}
    twitch_pass = all(item["reduction_fraction"] is not None and item["reduction_fraction"] >= .5
                      for item in reductions.values()) and not prolonged and not saturation_failures
    primary_control_pass = all_survived and not speed_failures
    complete_control_pass = primary_control_pass and candidate["command_practice"]["performance_accepted"]
    return {"all_eight_survived": all_survived, "primary_control_pass": primary_control_pass,
            "complete_control_pass": complete_control_pass, "twitch_pass": twitch_pass,
            "numeric_acceptance": complete_control_pass and twitch_pass,
            "requires_visual_confirmation": True, "speed_failures": speed_failures,
            "saturation_failures": saturation_failures, "prolonged_leg_saturation_scenarios": prolonged,
            "acceleration_comparison": reductions,
            "command_practice_pass_count": sum(s["performance_accepted"] for s in candidate["command_practice"]["scenarios"]),
            "candidate_motion": candidate["motion20ms"]["scenarios"],
            "candidate_physics": candidate["physics1ms"]["scenarios"]}


def train_and_evaluate(label, profile, baseline):
    log = command(label + "-fresh1000", [TRAIN_PY, "scripts/train_dreamwaq.py", "--headless",
                  "--profile", "rtx5070", "--task-profile", profile, "--max-iterations", "1000",
                  "--seed", str(SEED), "--run-name", "split-smooth-" + label + "-fresh1000-20261013"])
    match = re.search(r"\[INFO\] DreamWaQ run directory: (.+)", log.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError("Training log lacks run directory")
    run = Path(match.group(1).strip())
    summary = json.loads((run / "training_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert summary["completed_iterations"] == 1000 and summary["starting_completed_iterations"] == 0
    assert summary["resumed_from"] is None and "weights_initialization" not in summary
    assert manifest["task_profile"] == profile
    source = json.loads((BASE_EXPORT / "policy_manifest.json").read_text(encoding="utf-8"))
    old_run = json.loads((Path(source["source_checkpoint"]).parent / "run_manifest.json").read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(old_run["base_task_contract"]))
    expected.pop("contract_hash")
    actual = json.loads(json.dumps(manifest["base_task_contract"]))
    actual.pop("contract_hash")
    expected["schemas"]["reward"] = actual["schemas"]["reward"]
    expected["task"]["reward_weights"] = actual["task"]["reward_weights"]
    assert expected == actual, "Unexpected non-reward task difference against original baseline"
    export = OUT / label / "export"
    export.parent.mkdir(exist_ok=True)
    command(label + "-export", [TRAIN_PY, "scripts/export_dreamwaq_actor.py",
                               "--checkpoint", summary["final_checkpoint"], "--output", export])
    golden = verify_golden(export)
    evaluation = evaluate(label, export)
    result = compare(baseline, evaluation)
    result.update(label=label, profile=profile, run_dir=str(run), completed_iterations=1000,
                  final_checkpoint=summary["final_checkpoint"],
                  checkpoint_sha256=sha(summary["final_checkpoint"]),
                  source_manifest_sha256=sha(run / "run_manifest.json"), golden_errors=golden,
                  context_acceptance=summary["context_acceptance"])
    STATE["runs"].append(result)
    save()
    (OUT / label / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def write_results():
    lines = ["# Split action smooth experiment results", "",
             "Isaac was used only for training. All motion evaluation used the unchanged MuJoCo controller.",
             "Each long run is fresh seed41882943 / 256 environments / 1000 iterations.",
             "Numeric gates are exploratory; visual confirmation is still required.", ""]
    for run in STATE["runs"]:
        lines += ["## " + run["label"], "",
                  "- Primary speed/control pass: " + str(run["primary_control_pass"]),
                  "- Complete control pass: " + str(run["complete_control_pass"]),
                  "- Twitch gate pass: " + str(run["twitch_pass"]),
                  "- Numeric complete acceptance: " + str(run["numeric_acceptance"]),
                  "- Command practice: " + str(run["command_practice_pass_count"]) + "/4",
                  "- Checkpoint: " + run["final_checkpoint"],
                  "- Speed failures: " + repr(run["speed_failures"]),
                  "- Prolonged leg saturation: " + repr(run["prolonged_leg_saturation_scenarios"]),
                  "- Saturation failures: " + repr(run["saturation_failures"]), ""]
        for scale, values in run["acceleration_comparison"].items():
            lines += ["- " + scale + " acceleration RMS: " + json.dumps(values)]
        lines += [""]
    if not any(run["numeric_acceptance"] for run in STATE["runs"]):
        lines += ["No tested configuration met every numerical gate. Do not claim the jitter was fixed.", ""]
    (OUT / "results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    save()
    source = json.loads((BASE_EXPORT / "policy_manifest.json").read_text(encoding="utf-8"))
    assert source["source_completed_iterations"] == 1000
    assert sha(source["source_checkpoint"]) == "3D621280CF7F11088C972B0DD84CF2E698C29A4C8CDA8BC370BB80A2669DF885"
    STATE["baseline_checkpoint"] = source["source_checkpoint"]
    STATE["baseline_golden_errors"] = verify_golden(BASE_EXPORT)
    baseline = evaluate("baseline", BASE_EXPORT)
    first = train_and_evaluate("leg02", "stop_reverse_split_smooth_v1", baseline)
    if first["primary_control_pass"] and not first["twitch_pass"]:
        STATE["tuning_reason"] = "All eight survived and primary speed passed, but twitch gate failed."
        save()
        train_and_evaluate("leg04", "stop_reverse_split_smooth04_v1", baseline)
    else:
        STATE["tuning_reason"] = "No automatic increase: first twitch passed or primary control failed."
    write_results()
    STATE.update(status="completed", stage="finished", completed_unix=time.time())
    save()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        STATE.update(status="failed", error=traceback.format_exc())
        save()
        raise
