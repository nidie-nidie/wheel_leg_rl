import csv
import json
from pathlib import Path

import numpy as np
import torch

from debug.sim2sim.action_mode_engine_probe_20261020 import PROJECT, OUTPUT as MATRIX, SIM, load_actor, locations
from debug.sim2sim.audit_reward_economics_20261019 import frozen_identity, digest, read_json, write_json
from debug.sim2sim.evaluate_baseline_mujoco import run_protocol
from wheelleg_mujoco.evaluation import FORMAL_SCENARIOS
from wheelleg_mujoco.runner import WheelLegMujocoRuntime


OUTPUT = PROJECT / 'debug/sim2sim/runs/wheel-feedback-paths-20261020'
CASES = (('wheel_common_0', 'wheel_common', 0.), ('wheel_common_05', 'wheel_common', .5),
         ('wheel_common_075', 'wheel_common', .75), ('legs_0', 'legs', 0.),
         ('wheel_diff_0', 'wheel_diff', 0.), ('all_0', 'all', 0.))


def neutral_history(history):
    changed = history.reshape(-1, 5, 25).clone()
    wheels = changed[:, :, 23:25]
    wheels.sub_(wheels.mean(-1, keepdim=True))
    return changed.reshape(-1, 125)


def path_action(raw, neutral, route, alpha):
    action = raw.clone()
    if route == 'wheel_common':
        action[:, 4:] += (1 - alpha) * (neutral[:, 4:].mean(-1, keepdim=True) - raw[:, 4:].mean(-1, keepdim=True))
    elif route == 'legs':
        action[:, :4] += (1 - alpha) * (neutral[:, :4] - raw[:, :4])
    elif route == 'wheel_diff':
        difference = (neutral[:, 4:5] - neutral[:, 5:6] - raw[:, 4:5] + raw[:, 5:6]) / 2
        action[:, 4:5] += (1 - alpha) * difference
        action[:, 5:6] -= (1 - alpha) * difference
    elif route == 'all':
        action += (1 - alpha) * (neutral - raw)
    else:
        raise ValueError(route)
    return action


class PathActor:
    def __init__(self, actor, route, alpha):
        self.actor, self.route, self.alpha = actor, route, alpha
        self.histories, self.raw_actions, self.neutral_actions, self.actions = [], [], [], []

    def __call__(self, history):
        raw = self.actor(history)
        neutral = self.actor(neutral_history(history))
        action = path_action(raw, neutral, self.route, self.alpha)
        self.histories.append(history.numpy().copy())
        self.raw_actions.append(raw.numpy().copy())
        self.neutral_actions.append(neutral.numpy().copy())
        self.actions.append(action.numpy().copy())
        return action


def csv_metrics(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    valid = [row for row in rows if row['failure_reason'] == '']
    tail = valid[50:]
    assert tail
    acceleration = np.array([[float(row[f'leg_acceleration_{index}_rad_s2']) for index in range(4)] for row in tail])
    result = {'valid_ticks': len(valid), 'completed': len(valid) == 500,
              'failure': rows[-1]['failure_reason'] or None,
              'leg_rms': float(np.sqrt(np.mean(acceleration ** 2))),
              'leg_rms_per_joint': np.sqrt(np.mean(acceleration ** 2, axis=0)).tolist(),
              'leg_mean_abs': float(np.abs(acceleration).mean()),
              'speed_mae': float(np.mean([abs(float(row['vx_mps']) - float(row['target_vx_mps'])) for row in tail])),
              'yaw_rate_mae': float(np.mean([abs(float(row['yaw_rate_rad_s']) - float(row['target_yaw_rate_rad_s'])) for row in tail])),
              'yaw_drift': float(abs(float(tail[-1]['yaw_rad']) - float(tail[0]['yaw_rad']))),
              'effort_saturation': float(np.mean([float(row['effort_saturation_fraction']) for row in tail])),
              'csv': str(path), 'csv_sha256': digest(path)}
    return result


def aggregate(records):
    return {'leg_rms': float(np.sqrt(np.mean([row['leg_rms'] ** 2 for row in records]))),
            'speed_mae': float(np.mean([row['speed_mae'] for row in records])),
            'yaw_rate_mae': float(np.mean([row['yaw_rate_mae'] for row in records])),
            'effort_saturation': float(np.mean([row['effort_saturation'] for row in records])),
            'completed_scenarios': sum(row['completed'] for row in records),
            'valid_ticks': sum(row['valid_ticks'] for row in records)}


def sensitivity(actor, histories):
    inputs = torch.from_numpy(histories[50::10].copy())
    epsilon = .001
    records = []
    for frame in range(5):
        delta = torch.zeros_like(inputs).view(-1, 5, 25)
        delta[:, frame, 23:25] = epsilon
        delta = delta.reshape(-1, 125)
        with torch.inference_mode():
            derivative = (actor(inputs + delta) - actor(inputs - delta)) / (2 * epsilon)
        values = derivative.numpy()
        records.append({'frame': frame, 'wheel_common_mean': float(values[:, 4:].mean()),
                        'wheel_common_rms': float(np.sqrt(np.mean(values[:, 4:].mean(-1) ** 2))),
                        'leg_response_rms': float(np.sqrt(np.mean(values[:, :4] ** 2))),
                        'wheel_diff_rms': float(np.sqrt(np.mean(((values[:, 4] - values[:, 5]) / 2) ** 2)))})
    return records


def main():
    assert torch.version.cuda is None
    torch.set_num_threads(1)
    before = frozen_identity()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    actor, _, contract, model, identity = load_actor(401, 'cpu')
    parameter_before = {key: value.clone() for key, value in actor.state_dict().items()}
    baseline = read_json(MATRIX / 'mujoco/result.json')
    result = {'schema': 'WheelCommonHistoryOutputPathDiagnosticV1', 'policy_updates': 0, 'identity': identity,
              'script_sha256': digest(__file__), 'cases': {}, 'reused': {}}
    for update in (301, 401):
        records = []
        for name, _ in FORMAL_SCENARIOS:
            path = locations(update)[2] / f'{name}.csv'
            expected = baseline['models'][str(update)]['mean'][name]['csv_sha256']
            assert digest(path) == expected
            records.append(csv_metrics(path))
        result['reused'][str(update)] = {'scenes': records, 'aggregate': aggregate(records)}
    for label, route, alpha in CASES:
        folder = OUTPUT / label
        folder.mkdir()
        runtime = WheelLegMujocoRuntime(SIM / 'models' / model['model_xml']['path'], contract, model)
        runtime.heading.enabled = False
        scenes = []
        for name, command in FORMAL_SCENARIOS:
            probe = PathActor(actor, route, alpha)
            summary = run_protocol(runtime, probe, name, [command] * 500, folder)
            arrays = {'history': np.concatenate(probe.histories), 'raw': np.concatenate(probe.raw_actions),
                      'neutral': np.concatenate(probe.neutral_actions), 'action': np.concatenate(probe.actions)}
            path = folder / f'{name}_policy.npz'
            np.savez_compressed(path, **arrays)
            row = csv_metrics(folder / f'{name}.csv')
            row.update(name=name, trace=str(path), trace_sha256=digest(path),
                       action_change_rms=np.sqrt(np.mean((arrays['action'] - arrays['raw']) ** 2, axis=0)).tolist())
            if name == 'nominal_stand':
                row['feedback_sensitivity_on_actual_history'] = sensitivity(actor, arrays['history'])
            scenes.append(row)
        result['cases'][label] = {'route': route, 'alpha': alpha, 'scenes': scenes, 'aggregate': aggregate(scenes)}
        write_json(OUTPUT / 'progress.json', result)
        print(json.dumps({'case': label, **aggregate(scenes)}, ensure_ascii=False), flush=True)
    assert all(torch.equal(parameter_before[key], value) for key, value in actor.state_dict().items())
    assert frozen_identity() == before
    result['frozen_files_checked'] = len(before)
    result['parameters_unchanged'] = True
    write_json(OUTPUT / 'result.json', result)


if __name__ == '__main__':
    main()
