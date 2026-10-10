"""Keep production files immutable and instrument a private evaluator copy."""
from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
SUITE = PROJECT / "artifacts/phase2_dreamwaq/training-suite-20261010-001203-reset-v2"
EXPORT = SUITE / "exports/run-04"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def main():
    prior = PROJECT / "artifacts/debug/sim2sim/angular-limit-alignment-20261010-v1/protected-before.json"
    protected_paths = set(json.loads(prior.read_text(encoding="utf-8")))
    for directory in (PROJECT / "source/wheelleg_dreamwaq/wheelleg_dreamwaq",
                      PROJECT / "sim2sim/mujoco/wheelleg_mujoco", PROJECT / "scripts"):
        protected_paths.update(str(path.resolve()) for path in directory.rglob("*.py"))
    manifest = json.loads((EXPORT / "policy_manifest.json").read_text(encoding="utf-8"))
    for path in (*EXPORT.iterdir(), Path(manifest["source_checkpoint"]), Path(manifest["source_run_manifest"]),
                 SUITE / "isaac_evaluation/evaluation-reset-cache.pt",
                 SUITE / "isaac_evaluation/run-04/summary.json",
                 PROJECT / "artifacts/debug/sim2sim/angular-limit-alignment-20261010-v1/mujoco-evaluation/run-04/nominal_stand.csv"):
        protected_paths.add(str(path.resolve()))
    protected = {path: digest(path) for path in sorted(protected_paths)}
    before = ROOT / "protected-before.json"
    assert not before.exists(), "Use a fresh diagnostic directory"
    before.write_text(json.dumps(protected, indent=2), encoding="utf-8")
    original = PROJECT / "scripts/evaluate_isaac.py"
    source = original.read_text(encoding="utf-8")
    modified = source

    def replace(old, new):
        nonlocal modified
        assert modified.count(old) == 1, old
        modified = modified.replace(old, new)

    replace('PROJECT_ROOT = Path(__file__).resolve().parents[1]', f'PROJECT_ROOT = Path({str(PROJECT)!r})')
    replace('from wheelleg_dreamwaq.training.dreamwaq_checkpoint import validate_dreamwaq_checkpoint_metadata',
            'from wheelleg_dreamwaq.training.dreamwaq_checkpoint import validate_dreamwaq_checkpoint_metadata\nfrom trace_recorder import StandingTrace')
    replace('    try:\n        for action_step in range(1, EVALUATION_ACTION_STEPS + 1):',
            '    trace = StandingTrace(output, "isaac", vars(env_cfg.normalization), env_cfg.control.wheel_action_scale)\n'
            '    try:\n        for action_step in range(1, EVALUATION_ACTION_STEPS + 1):')
    replace('            actions = actions.clone()',
            '            if bool(active_before[0].item()):\n'
            '                state = direct_env._current_state()\n'
            '                trace.before(action_step - 1, (action_step - 1) * direct_env.step_dt, commands[0],\n'
            '                    base_observations["policy"][0], history.flat()[0], state.root_com_linear_velocity[0],\n'
            '                    estimated_velocity[0], policy.estimator_inference(observations)[1][0], actions[0],\n'
            '                    state.joint_position[0], state.joint_velocity[0], state.projected_gravity[0],\n'
            '                    state.base_height[0, 0].item(), base_observations["critic"][0, 25:28])\n'
            '            actions = actions.clone()')
    replace('            next_observations, rewards, dones, extras = env.step(actions)',
            '            next_observations, rewards, dones, extras = env.step(actions)\n'
            '            if bool(active_before[0].item()):\n'
            '                state = direct_env._current_state()\n'
            '                trace.after(state.root_com_linear_velocity[0], direct_env._canonical_action[0],\n'
            '                            state.applied_torque[0], bool(dones[0].item()))')
    replace('    finally:\n        env.close()', '    finally:\n        trace.save()\n        env.close()')
    start = modified.index('    if policy_kind == "dreamwaq":\n        if args_cli.baseline_report is None:')
    end = modified.index('\n    report = {', start)
    modified = modified[:start] + modified[end:]
    replace('"evaluation_schema_version": ISAAC_EVALUATION_SCHEMA_VERSION,',
            '"evaluation_schema_version": "StandingIsaacDiagnosticV1",')
    # Use the context already computed by the actual inference, avoiding another encoder call.
    replace('actions, estimated_velocity, _ = policy.act_inference_with_estimator(observations)',
            'actions, estimated_velocity, context_mu = policy.act_inference_with_estimator(observations)')
    replace('policy.estimator_inference(observations)[1][0]', 'context_mu[0]')
    script = ROOT / "probe_isaac.py"
    script.write_text(modified, encoding="utf-8")
    (ROOT / "isaac-instrumentation.diff").write_text(''.join(difflib.unified_diff(
        source.splitlines(True), modified.splitlines(True), fromfile=str(original), tofile=str(script))), encoding="utf-8")
    records = []

    def run(engine, cmd):
        log = ROOT / f"{engine}-console.log"
        print("START_STANDING", engine, flush=True)
        started = time.monotonic()
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run([str(v) for v in cmd], cwd=PROJECT,
                                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"},
                                    stdout=stream, stderr=subprocess.STDOUT)
        complete = "STANDING_TRACE_COMPLETE" in log.read_text(encoding="utf-8", errors="replace")
        records.append({"engine": engine, "command": [str(v) for v in cmd], "exit_code": result.returncode,
                        "elapsed_s": time.monotonic() - started, "complete": complete, "log": str(log)})
        (ROOT / "execution.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print("EXIT_STANDING", engine, result.returncode, complete, flush=True)
        assert result.returncode == 0 and complete, log.read_text(encoding="utf-8", errors="replace")[-6000:]

    run("mujoco", [PROJECT / "sim2sim/mujoco/.venv/Scripts/python.exe", "-B", ROOT / "probe_mujoco.py",
                   "--export", EXPORT, "--output", ROOT / "mujoco"])
    run("isaac", [PROJECT / ".venv/Scripts/python.exe", "-B", script, "--checkpoint", manifest["source_checkpoint"],
                  "--reset-cache", SUITE / "isaac_evaluation/evaluation-reset-cache.pt", "--output", ROOT / "isaac",
                  "--headless", "--device", "cuda:0"])
    for path, expected in protected.items():
        assert digest(path) == expected, f"Protected file changed: {path}"
    (ROOT / "immutable-verification.json").write_text(json.dumps({
        "passed": True, "protected_file_count": len(protected), "formal_files_changed": [],
        "diagnostic_sources": {str(path): digest(path) for path in ROOT.glob("*.py")},
    }, indent=2), encoding="utf-8")
    print("STANDING_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
