import math

import numpy as np
import torch

from debug.sim2sim.audit_reward_economics_20261019 import DEFAULT_RUN as RUN, digest, frozen_identity, read_json, write_json
from debug.sim2sim.verify_wheel_feedback_paths_20261020 import rows_of, independent_metrics
from debug.sim2sim.wheel_feedback_paths_20261020 import aggregate
from debug.sim2sim.wheel_feedback_reward_origin_20261020 import OUTPUT, UPDATES, VALUE_NAMES, load_histories
from debug.sim2sim.wheel_feedback_update318_20261020 import OUTPUT as PREVIOUS, KEYS, make_policy, probe_inputs, score


def check_finite(value):
    if isinstance(value, dict):
        for item in value.values():
            check_finite(item)
    elif isinstance(value, list):
        for item in value:
            check_finite(item)
    elif isinstance(value, float):
        assert math.isfinite(value)


def main():
    torch.set_num_threads(1)
    assert torch.version.cuda is None
    frozen = frozen_identity()
    result = read_json(OUTPUT / 'result.json')
    assert digest(__file__.replace('verify_', '', 1)) == result['source_sha256']
    reference = read_json(PREVIOUS / 'result.json')
    previous_fresh = read_json(PREVIOUS / 'reward_directions_recorded_forward_cpu/result.json')
    histories = load_histories(reference)
    verified = {'updates': [], 'interventions': {}, 'frozen_files_checked': 118,
                'strict_gpu_reward_attribution_passed': False}
    names = list(result['updates'][0]['contributions']['fresh'])[:-1]
    for expected_update, header in zip(UPDATES, result['updates'], strict=True):
        item = read_json(OUTPUT / f'update_{expected_update:04d}.json')
        assert item['update'] == expected_update and len(item['minibatches']) == 40
        assert all(item[key] == value for key, value in header.items())
        directory = RUN / f'training_evidence/update_{expected_update:04d}'
        assert digest(directory / 'index.json') == item['identity']['index_sha256']
        assert item['identity']['verified_artifacts'] == 67
        for label in ('fresh', 'with_within_update_momentum'):
            for name in names:
                assert item['contributions'][label][name] == sum(row[label + '_reward_estimates'][name] for row in item['minibatches'])
            assert item['contributions'][label]['unassigned_cpu_residual'] == sum(row[label + '_unassigned_cpu_residual'] for row in item['minibatches'])
        assert max(row['adam_reconstruction_error'] for row in item['minibatches']) < 1e-7
        assert item['summary']['expanded_reward_plus_incoming_accounting_error'] < 2e-6
        previous_after = None
        for number in range(40):
            batch = torch.load(directory / f'minibatch_{number:02d}.pt', map_location='cpu', weights_only=False, mmap=True)
            if previous_after is not None:
                assert all(torch.equal(previous_after[key], batch['policy_before'][key]) for key in KEYS)
            previous_after = {key: batch['actor_parameters_after'][key].clone() for key in KEYS}
        checkpoint = RUN / f'model_{expected_update - 1}.pt'
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        assert all(torch.equal(previous_after[key], payload['model_state_dict'][key]) for key in KEYS)
        policy = make_policy(payload['model_state_dict'])
        with torch.inference_mode():
            full_after, _ = score(policy.actor, probe_inputs(policy, histories), result['frequency_hz'])
        full_change = float(full_after) - item['minibatches'][0]['score_before']
        verified['updates'].append({'update': expected_update, 'full_policy_score_change': full_change,
                                    'sum_actor_only_change': item['summary']['actor_only_score_change'],
                                    'sum_encoder_and_other_change': full_change - item['summary']['actor_only_score_change'],
                                    'checkpoint_sha256': digest(checkpoint),
                                    'result_sha256': digest(OUTPUT / f'update_{expected_update:04d}.json')})
        if expected_update == 318:
            for name in previous_fresh['estimated_summed_contributions']:
                value = (sum(item['contributions']['fresh'][key] for key in VALUE_NAMES) if name == 'value_bootstrap'
                         else item['contributions']['fresh'][name])
                assert abs(value - previous_fresh['estimated_summed_contributions'][name]) < 2e-6
            assert abs(item['summary']['actual_projection'] - reference['summary']['first_order_actual_change']) < 1e-12
    rows = result['updates']
    table = {}
    for name in names:
        values = [row['contributions']['with_within_update_momentum'][name] for row in rows]
        table[name] = {'positive_updates': sum(value > 0 for value in values), 'negative_updates': sum(value < 0 for value in values),
                       'min': min(values), 'max': max(values), 'mean_selected_updates': float(np.mean(values))}
    verified['sign_consistency'] = table
    centering = read_json(OUTPUT / 'centering_sensitivity/result.json')
    weights = read_json(OUTPUT / 'weight_sensitivity/result.json')
    assert digest(__file__.replace('verify_wheel_feedback_reward_origin', 'wheel_feedback_centering_sensitivity')) == centering['source_sha256']
    assert digest(__file__.replace('verify_wheel_feedback_reward_origin', 'wheel_feedback_weight_sensitivity')) == weights['source_sha256']
    for centered, weighted, source in zip(centering['updates'], weights['updates'], result['updates'], strict=True):
        assert centered['update'] == weighted['update'] == source['update']
        original = source['contributions']['with_within_update_momentum']
        for name, value in centered['uncentered_rewards'].items():
            expected = original[name] + centered['reward_normalized_mean_constants'][name] * centered['constant_loss_adam_projection']
            assert abs(value - expected) < 1e-12
        accounting = sum(centered['uncentered_rewards'].values()) - sum(centered['centered'].values()) + centered['compensating_value_change']
        assert abs(accounting) < 1e-7
        cpu_total = sum(value for name, value in original.items() if name != 'unassigned_cpu_residual')
        for name, value in weighted['reward_scale_derivative'].items():
            assert abs(value - (original[name] - weighted['normalization_variance_derivative'][name] * cpu_total)) < 1e-12
        assert max(weighted['analytic_advantage_derivative_errors'].values()) < 1e-7
    verified['centering_sensitivity_signs'] = centering['signs']
    verified['reward_weight_derivative_positive_counts'] = weights['positive_derivative_counts']
    verified['additional_source_result_sha256'] = {str(path.relative_to(OUTPUT)): digest(path) for path in (
        OUTPUT / 'centering_sensitivity/result.json', OUTPUT / 'weight_sensitivity/result.json')}
    physical = read_json(OUTPUT / 'interventions_318/result.json')
    assert digest(__file__.replace('verify_wheel_feedback_reward_origin', 'wheel_feedback_reward_intervention')) == physical['source_sha256']
    for label, case in {'baseline': physical['baseline'], **physical['cases']}.items():
        if label != 'baseline':
            assert digest(case['state_path']) == case['state_sha256']
            state = torch.load(case['state_path'], map_location='cpu', weights_only=False)['model_state_dict']
            original_state = torch.load(physical['source_checkpoint'], map_location='cpu', weights_only=False, mmap=True)['model_state_dict']
            assert all(torch.equal(state[key], original_state[key]) for key in state if key not in KEYS)
            policy = make_policy(state)
            frequency_checks = {}
            for epsilon in (.0005, .001, .002):
                inputs = probe_inputs(policy, histories, epsilon)
                with torch.inference_mode():
                    frequency_checks[str(epsilon)] = {str(frequency): float(score(policy.actor, inputs, frequency, epsilon)[0])
                                                     for frequency in (4., 6., result['frequency_hz'], 10., 12., 16.)}
        else:
            frequency_checks = {}
        metrics, acceleration_error = [], 0.
        for row in case['scenes']:
            assert digest(row['csv']) == row['csv_sha256']
            csv_rows = rows_of(row['csv'])
            values = independent_metrics(csv_rows)
            for name, value in values.items():
                assert abs(value - row[name]) < 1e-12
            velocity = np.array([[float(tick[f'leg_velocity_{index}_rad_s']) for index in range(4)] for tick in csv_rows])
            acceleration = np.array([[float(tick[f'leg_acceleration_{index}_rad_s2']) for index in range(4)] for tick in csv_rows])
            error = float(np.max(np.abs(np.diff(velocity, axis=0) / .02 - acceleration[1:])))
            acceleration_error = max(acceleration_error, error)
            assert error < 1e-10
            metrics.append(row)
        assert aggregate(metrics) == case['aggregate']
        verified['interventions'][label] = {'aggregate': aggregate(metrics), 'acceleration_reconstruction_error': acceleration_error,
                                           'frequency_checks': frequency_checks}
    verified['second_difference_immediate_target_transfer_gain_at_8_444Hz'] = math.sqrt(5 - 4 * math.cos(2 * math.pi * result['frequency_hz'] * .02))
    assert frozen_identity() == frozen
    check_finite(result)
    check_finite(physical)
    check_finite(verified)
    verified['verified_new_fixed_scenes'] = 40
    verified['source_sha256'] = digest(__file__)
    write_json(OUTPUT / 'verification.json', verified)
    print({'updates_verified': len(verified['updates']), 'new_scenes_verified': 40,
           'wheel_smooth_signs': table['action_smooth_wheel'], 'yaw_enhance_signs': table['tracking_yaw_rate_enhance'],
           'max_cpu_residual_relative_norm': max(row['max_cpu_residual_relative_norm'] for row in result['updates']),
           'physical': {label: value['aggregate'] for label, value in verified['interventions'].items()}})


if __name__ == '__main__':
    main()
