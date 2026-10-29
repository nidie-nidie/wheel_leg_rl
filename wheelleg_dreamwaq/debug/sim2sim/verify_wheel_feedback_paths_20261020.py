import csv
from pathlib import Path

import numpy as np
import torch

from debug.sim2sim.action_mode_engine_probe_20261020 import load_actor
from debug.sim2sim.audit_reward_economics_20261019 import digest, frozen_identity, read_json, write_json
from debug.sim2sim.wheel_feedback_paths_20261020 import OUTPUT, neutral_history, path_action


def rows_of(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def independent_metrics(rows):
    assert len(rows) == 500 and all(row['failure_reason'] == '' for row in rows)
    tail = rows[50:]
    acceleration = np.array([[float(row[f'leg_acceleration_{index}_rad_s2']) for index in range(4)] for row in tail])
    return {'leg_rms': float(np.sqrt((acceleration * acceleration).sum() / (450 * 4))),
            'speed_mae': sum(abs(float(row['vx_mps']) - float(row['target_vx_mps'])) for row in tail) / 450,
            'yaw_rate_mae': sum(abs(float(row['yaw_rate_rad_s']) - float(row['target_yaw_rate_rad_s'])) for row in tail) / 450}


def verify_case(actor, row, route, alpha, slow):
    assert digest(row['csv']) == row['csv_sha256']
    assert digest(row['trace']) == row['trace_sha256']
    rows = rows_of(row['csv'])
    recalculated = independent_metrics(rows)
    for key, value in recalculated.items():
        assert abs(value - row[key]) < 1e-12
    archive = np.load(row['trace'], allow_pickle=False)
    history, raw, neutral, action = [archive[key] for key in ('history', 'raw', 'neutral', 'action')]
    assert history.shape == (500, 125) and all(value.shape == (500, 6) for value in (raw, neutral, action))
    assert all(np.isfinite(value).all() for value in (history, raw, neutral, action))
    csv_action = np.array([[float(item[f'raw_action_{index}']) for index in range(6)] for item in rows])
    assert np.array_equal(csv_action, action)
    frames = history.reshape(500, 5, 25)
    # 真实历史始终回填真正执行的动作，未回填辅助策略的输出。
    assert np.array_equal(frames[1:, -1, 19:25], action[:-1])
    assert np.array_equal(frames[1:, :-1], frames[:-1, 1:])
    assert np.array_equal(frames[0], np.repeat(frames[0, :1], 5, axis=0))
    assert np.max(np.abs(frames[0, :, 19:25])) == 0
    max_raw_error, max_aux_error, max_action_error = 0., 0., 0.
    for tick in range(500):
        value = torch.from_numpy(history[tick:tick + 1])
        changed = neutral_history(value)
        if slow:
            changed = value.view(1, 5, 25).clone()
            common = changed[:, :, 23:25].mean(-1, keepdim=True)
            changed[:, :, 23:25] += common.mean(1, keepdim=True) - common
            assert torch.equal(changed[:, :, :23], value.view(1, 5, 25)[:, :, :23])
            original_wheel = value.view(1, 5, 25)[:, :, 23:25]
            assert float((changed[:, :, 23] - changed[:, :, 24] - original_wheel[:, :, 0] + original_wheel[:, :, 1]).abs().max()) < 2e-6
            assert float((changed[:, :, 23:25].mean((1, 2)) - original_wheel.mean((1, 2))).abs().max()) < 2e-6
            changed = changed.view(1, 125)
        with torch.inference_mode():
            actual_raw = actor(value)
            actual_aux = actor(changed)
            actual_action = path_action(actual_raw, actual_aux, route, alpha)
        max_raw_error = max(max_raw_error, float((actual_raw - torch.from_numpy(raw[tick:tick + 1])).abs().max()))
        max_aux_error = max(max_aux_error, float((actual_aux - torch.from_numpy(neutral[tick:tick + 1])).abs().max()))
        max_action_error = max(max_action_error, float((actual_action - torch.from_numpy(action[tick:tick + 1])).abs().max()))
    assert max_raw_error == max_aux_error == max_action_error == 0
    velocities = np.array([[float(item[f'leg_velocity_{index}_rad_s']) for index in range(4)] for item in rows])
    recorded_acc = np.array([[float(item[f'leg_acceleration_{index}_rad_s2']) for index in range(4)] for item in rows])
    acceleration_error = float(np.abs(np.diff(velocities, axis=0) / .02 - recorded_acc[1:]).max())
    assert acceleration_error < 1e-10
    return {'csv_sha256': row['csv_sha256'], 'trace_sha256': row['trace_sha256'],
            'raw_aux_action_max_error': [max_raw_error, max_aux_error, max_action_error],
            'acceleration_from_real_velocity_max_error': acceleration_error, **recalculated}


def main():
    torch.set_num_threads(1)
    assert torch.version.cuda is None
    before = frozen_identity()
    path_result = read_json(OUTPUT / 'result.json')
    slow_result = read_json(OUTPUT / 'slow_history/result.json')
    adam_result = read_json(OUTPUT / 'actual_update318/result.json')
    actor, _, _, _, identity = load_actor(401, 'cpu')
    assert identity == path_result['identity'] == slow_result['identity']
    result = {'cases': {}, 'old701_recomputed': {}, 'transition_events': {}, 'frozen_files_checked': 118}
    for group, source, slow in (('paths', path_result, False), ('slow_history', slow_result, True)):
        for label, case in source['cases'].items():
            verified = [verify_case(actor, row, case.get('route', 'wheel_common'), case['alpha'], slow) for row in case['scenes']]
            result['cases'][f'{group}/{label}'] = verified
            rms = float(np.sqrt(np.mean([row['leg_rms'] ** 2 for row in verified])))
            assert abs(rms - case['aggregate']['leg_rms']) < 1e-12
            assert abs(np.mean([row['speed_mae'] for row in verified]) - case['aggregate']['speed_mae']) < 1e-12
            if slow:
                for transition in case['transitions']:
                    metric = transition['metrics']
                    assert digest(metric['csv']) == metric['csv_sha256']
                    rows = rows_of(metric['csv'])
                    independent_metrics(rows)
                    events = [(int(row['tick']), row['event']) for row in rows if row['event']]
                    if transition['name'] == 'reset_toggle':
                        assert events == [(101, 'heading_off'), (201, 'heading_on'), (301, 'reset')]
                        assert float(rows[100]['heading_enabled']) == 0 and float(rows[200]['heading_enabled']) == 1
                        assert abs(float(rows[300]['time_s']) - .02) < 1e-12
                    result['transition_events'][f'{label}/{transition["name"]}'] = {'events': events, 'ticks': len(rows), 'csv_sha256': metric['csv_sha256']}
    old = []
    for row in slow_result['old701']['scenes']:
        assert digest(row['csv']) == row['csv_sha256']
        values = independent_metrics(rows_of(row['csv']))
        assert all(abs(values[key] - row[key]) < 1e-12 for key in values)
        old.append(values)
    result['old701_recomputed'] = {'leg_rms': float(np.sqrt(np.mean([row['leg_rms'] ** 2 for row in old]))),
                                   'speed_mae': float(np.mean([row['speed_mae'] for row in old]))}
    assert len(adam_result['minibatches']) == 40
    for key in ('actual_metric_change_actor_only', 'first_order_actual_change', 'carried_first_moment_change', 'current_gradient_numerator_change'):
        assert sum(row[key] for row in adam_result['minibatches']) == adam_result['summary'][key]
    assert max(row['adam_split_max_abs_error'] for row in adam_result['minibatches']) < 1e-7
    result['adam_summary'] = adam_result['summary']
    assert frozen_identity() == before
    result['source_hashes'] = {path.name: digest(path) for path in (
        Path(__file__), Path(__file__).with_name('wheel_feedback_paths_20261020.py'),
        Path(__file__).with_name('wheel_feedback_slow_history_20261020.py'),
        Path(__file__).with_name('wheel_feedback_update318_20261020.py'))}
    result['verified_fixed_scenes'] = sum(len(value) for value in result['cases'].values())
    result['verified_transitions'] = len(result['transition_events'])
    write_json(OUTPUT / 'verification.json', result)
    print({'fixed_scenes': result['verified_fixed_scenes'], 'transitions': result['verified_transitions'], 'old701': result['old701_recomputed']}, flush=True)


if __name__ == '__main__':
    main()
