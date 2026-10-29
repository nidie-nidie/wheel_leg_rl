import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import psutil
import torch

import wheelleg_dreamwaq.schemas
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import RawSplitStraightRewardWeights, compute_reward
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.state import WheelLegState


PROJECT = Path(__file__).resolve().parents[2]
REWARDS = ('tracking_vx', 'tracking_vx_enhance', 'tracking_yaw_rate', 'tracking_yaw_rate_enhance',
           'tracking_base_height', 'lin_vel_z', 'ang_vel_xy', 'orientation', 'dof_vel', 'dof_acc',
           'torques', 'action_rate', 'dof_pos_limits', 'phi0_symmetry', 'termination',
           'action_smooth_leg', 'action_smooth_wheel', 'wheel_differential')
EXTRA = ('total_reward', 'leg_acceleration_squared_mean', 'wheel_acceleration_squared_mean',
         'leg_acceleration_reward', 'wheel_acceleration_reward', 'vx_mae', 'yaw_rate_mae',
         'height_mae', 'vertical_velocity_squared', 'orientation_error', 'leg_velocity_squared_mean',
         'wheel_common_action_squared', 'wheel_common_action_difference_squared')
FEATURES = REWARDS + EXTRA
DEFAULT_RUN = PROJECT / 'logs/rsl_rl/wheelleg_flat_dreamwaq/2026-10-18_00-26-31_rtx5070_fudan-v1_baseline-yaw1-headingoff-env6144-causal-capture401-retry-20261018'
FROZEN = PROJECT / 'debug/sim2sim/runs/baseline-yaw1-headingoff-env4096-fresh701-seed41882943-20261017-launch/source_before.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest().upper()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def frozen_identity():
    entries = read_json(FROZEN)
    assert len(entries) == 118
    for entry in entries:
        assert digest(entry['path']) == entry['sha256'], entry['path']
    return {entry['path']: entry['sha256'] for entry in entries}


def load_artifact(directory, name, index):
    path = directory / name
    record = index[directory.name + '/' + name]
    assert path.stat().st_size == record['bytes'] and digest(path) == record['sha256'], str(path)
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def classification(command):
    vx, yaw = command[:, 0], command[:, 1]
    return np.where((vx == 0) & (yaw == 0), 0,
                    np.where(yaw == 0, 1, np.where(vx == 0, 2, 3)))


def describe(matrix, mask=None):
    selected = matrix if mask is None else matrix[mask]
    if not len(selected):
        return {'count': 0}
    result = {'count': len(selected), 'means': dict(zip(FEATURES, selected.mean(axis=0, dtype=np.float64).tolist()))}
    result['leg_rms'] = float(np.sqrt(result['means']['leg_acceleration_squared_mean']))
    result['wheel_rms'] = float(np.sqrt(result['means']['wheel_acceleration_squared_mean']))
    return result


def matched_comparison(values, commands, ages, updates, env_ids, exact_environment=False):
    energy = values[:, FEATURES.index('leg_acceleration_squared_mean')]
    low_threshold, high_threshold = np.quantile(energy, [0.25, 0.75])
    low, high = energy <= low_threshold, energy >= high_threshold
    columns = [(updates - 251) // 10, np.rint(commands[:, 0] / 0.1),
               np.rint(commands[:, 1] / 0.1), np.rint(commands[:, 2] / 0.01), ages // 100]
    if exact_environment:
        columns.append(env_ids)
    keys = np.stack(columns, axis=-1).astype(np.int32)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    group_count = int(inverse.max()) + 1
    low_count = np.bincount(inverse[low], minlength=group_count)
    high_count = np.bincount(inverse[high], minlength=group_count)
    valid = (low_count >= 2) & (high_count >= 2)
    weight = np.minimum(low_count, high_count)[valid].astype(np.float64)
    result = {'low_rms_threshold': float(np.sqrt(low_threshold)),
              'high_rms_threshold': float(np.sqrt(high_threshold)), 'matched_strata': int(valid.sum()),
              'low_windows_in_overlap': int(low_count[valid].sum()),
              'high_windows_in_overlap': int(high_count[valid].sum()),
              'total_low_windows': int(low.sum()), 'total_high_windows': int(high.sum()),
              'exact_environment': exact_environment,
              'strata': '10-update block, vx 0.1m/s, yaw 0.1rad/s, target height 0.01m, episode age 100 ticks',
              'interpretation': 'Observational conditional association; no random intervention and no matching of unrecorded disturbances.'}
    if not len(weight):
        return result
    means = []
    for mask, counts in ((low, low_count), (high, high_count)):
        mean = np.array([np.sum(np.bincount(inverse[mask], weights=values[mask, column],
                                           minlength=group_count)[valid] / counts[valid] * weight) / weight.sum()
                         for column in range(len(FEATURES))])
        means.append(mean)
    result['low_means'] = dict(zip(FEATURES, means[0].tolist()))
    result['high_means'] = dict(zip(FEATURES, means[1].tolist()))
    result['high_minus_low'] = dict(zip(FEATURES, (means[1] - means[0]).tolist()))
    return result


def analyse(run, output):
    torch.set_num_threads(1)
    assert torch.version.cuda is None, torch.__version__
    assert psutil.virtual_memory().available > 1024**3
    frozen = frozen_identity()
    manifest = read_json(run / 'run_manifest.json')
    weights = RawSplitStraightRewardWeights(tracking_yaw_rate=1.0, tracking_yaw_rate_enhance=1.0)
    assert manifest['num_envs'] == 6144 and manifest['seed'] == 41882943
    assert manifest['base_task_contract']['task']['reward_weights'] == vars(weights)
    n, width = 6144, len(FEATURES)
    window_sum = np.zeros((n, width), dtype=np.float64)
    window_count = np.zeros(n, dtype=np.int32)
    previous_age = np.full(n, -1, dtype=np.int64)
    previous_command = np.zeros((n, 3), dtype=np.float32)
    previous_velocity = None
    previous_done = None
    window_start = np.zeros(n, dtype=np.int32)
    windows, window_metadata, rows, artifacts = [], [], [], []
    failures = {'reward_term_max_abs_error': 0., 'total_reward_max_abs_error': 0.,
                'rollout_reward_max_abs_error': 0., 'gae_internal_max_abs_error': 0.,
                'gae_final_done_max_abs_error': 0., 'normalized_advantage_max_abs_error': 0.,
                'physical_acceleration_max_abs_error': 0.}
    boundary = dict(terminated=0, timeout=0, both=0, transitions=0, gae_internal_checked=0,
                    gae_final_done_checked=0, gae_final_continuing_unobserved=0,
                    both_timeout_bootstrap_sum=0., timeout_bootstrap_sum=0.)
    for update in range(251, 402):
        directory = run / 'training_evidence' / f'update_{update:04d}'
        index_document = read_json(directory / 'index.json')
        assert index_document['complete'] and index_document['update'] == update
        index = {entry['path']: entry for entry in index_document['artifacts']}
        rollout = load_artifact(directory, 'rollout.pt', index)
        assert float(rollout['algorithm']['gamma']) == 0.99 and float(rollout['algorithm']['lam']) == 0.95
        physical_reward, terminated, timeout = [], [], []
        sum_features = np.zeros(width, dtype=np.float64)
        command_counts = np.zeros(4, dtype=np.int64)
        age_counts = np.zeros(5, dtype=np.int64)
        term_age_counts = np.zeros(5, dtype=np.int64)
        group_sums = np.zeros((4, width), dtype=np.float64)
        group_counts = np.zeros(4, dtype=np.int64)
        for step in range(24):
            sample = load_artifact(directory, f'physical_step_{step:02d}.pt', index)
            state = WheelLegState(**sample['state'])
            total, components = compute_reward(state, weights, control_dt=0.02, terminated=sample['terminated'])
            for name in REWARDS:
                error = float((components[name] - sample['weighted_rewards'][name]).abs().max())
                failures['reward_term_max_abs_error'] = max(failures['reward_term_max_abs_error'], error)
            error = float((total - sample['total_reward']).abs().max())
            failures['total_reward_max_abs_error'] = max(failures['total_reward_max_abs_error'], error)
            assert failures['reward_term_max_abs_error'] < 3e-6 and failures['total_reward_max_abs_error'] < 3e-6
            weighted = torch.stack([sample['weighted_rewards'][name] for name in REWARDS], dim=-1)
            leg_energy = state.joint_acceleration[:, :4].square().mean(-1)
            wheel_energy = state.joint_acceleration[:, 4:].square().mean(-1)
            common = state.last_applied_action[:, 4:].mean(-1)
            previous_common = state.previous_applied_action[:, 4:].mean(-1)
            extra = torch.stack((sample['total_reward'], leg_energy, wheel_energy,
                                 leg_energy * (4 * -2.5e-7 * 0.02), wheel_energy * (2 * -2.5e-7 * 0.02),
                                 (state.command[:, 0] - state.root_com_linear_velocity[:, 0]).abs(),
                                 (state.command[:, 1] - state.root_angular_velocity[:, 2]).abs(),
                                 (state.command[:, 2] - state.base_height[:, 0]).abs(),
                                 state.root_com_linear_velocity[:, 2].square(),
                                 -sample['weighted_rewards']['orientation'] / 2.,
                                 state.joint_velocity[:, :4].square().mean(-1), common.square(),
                                 (common - previous_common).square()), dim=-1)
            features = torch.cat((weighted, extra), dim=-1).numpy()
            assert np.isfinite(features).all()
            command = state.command.numpy()
            age = sample['episode_length'].numpy().astype(np.int64)
            term, tout = sample['terminated'].numpy().astype(bool), sample['timeouts'].numpy().astype(bool)
            done = term | tout
            if previous_velocity is not None:
                expected_acc = (state.joint_velocity.numpy() - np.where(previous_done[:, None], 0., previous_velocity)) / 0.02
                error = float(np.max(np.abs(expected_acc - state.joint_acceleration.numpy())))
                failures['physical_acceleration_max_abs_error'] = max(failures['physical_acceleration_max_abs_error'], error)
                assert error < 0.01, (update, step, error)
                assert np.all(age[~previous_done] == previous_age[~previous_done] + 1)
                assert np.all(age[previous_done] == 1)
            previous_velocity = state.joint_velocity.numpy().copy()
            previous_done = done.copy()
            physical_reward.append(sample['total_reward'].clone())
            terminated.append(sample['terminated'].clone())
            timeout.append(sample['timeouts'].clone())
            sum_features += features.sum(axis=0, dtype=np.float64)
            family = classification(command)
            command_counts += np.bincount(family, minlength=4)
            for category in range(4):
                mask = family == category
                group_sums[category] += features[mask].sum(axis=0, dtype=np.float64)
                group_counts[category] += int(mask.sum())
            age_bucket = np.minimum(age // 50, 4)
            age_counts += np.bincount(age_bucket, minlength=5)
            term_age_counts += np.bincount(age_bucket[term], minlength=5)
            continuing = (age == previous_age + 1) & np.all(command == previous_command, axis=1)
            eligible = (age > 50) & ~done
            reset = ~continuing | ~eligible
            window_count[reset] = 0
            window_sum[reset] = 0.
            window_start[(window_count == 0) & eligible] = update
            window_sum[eligible] += features[eligible]
            window_count[eligible] += 1
            complete = window_count == 50
            if complete.any():
                ids = np.flatnonzero(complete)
                windows.append((window_sum[complete] / 50.).astype(np.float32))
                window_metadata.append(np.column_stack((ids, window_start[complete], np.full(len(ids), update),
                                                        age[complete], command[complete])).astype(np.float32))
                window_sum[complete] = 0.
                window_count[complete] = 0
            previous_age, previous_command = age.copy(), command.copy()
            del sample, state, weighted, extra, features
        physical_reward = torch.stack(physical_reward).unsqueeze(-1)
        term, tout = torch.stack(terminated).unsqueeze(-1).bool(), torch.stack(timeout).unsqueeze(-1).bool()
        done = rollout['dones'].bool()
        assert torch.equal(done, term | tout)
        assert torch.equal(rollout['reconstruction_mask'].bool(), ~done)
        expected_reward = physical_reward + 0.99 * rollout['values'] * tout.float()
        error = float((expected_reward - rollout['rewards']).abs().max())
        failures['rollout_reward_max_abs_error'] = max(failures['rollout_reward_max_abs_error'], error)
        assert error < 2e-6
        raw = rollout['returns'] - rollout['values']
        assert torch.equal(raw, rollout['raw_advantages'])
        delta = rollout['rewards'][:-1] + 0.99 * (~done[:-1]).float() * rollout['values'][1:] - rollout['values'][:-1]
        expected = delta + (0.99 * 0.95) * (~done[:-1]).float() * raw[1:]
        error = float((expected - raw[:-1]).abs().max())
        failures['gae_internal_max_abs_error'] = max(failures['gae_internal_max_abs_error'], error)
        assert error < 2e-5
        last_done = done[-1]
        if last_done.any():
            error = float((raw[-1] - (rollout['rewards'][-1] - rollout['values'][-1]))[last_done].abs().max())
            failures['gae_final_done_max_abs_error'] = max(failures['gae_final_done_max_abs_error'], error)
            assert error < 2e-5
        expected = (raw - raw.mean()) / (raw.std() + 1e-8)
        error = float((expected - rollout['advantages']).abs().max())
        failures['normalized_advantage_max_abs_error'] = max(failures['normalized_advantage_max_abs_error'], error)
        assert error < 2e-5
        both = term & tout
        timeout_bonus = 0.99 * rollout['values'] * tout.float()
        boundary['terminated'] += int(term.sum())
        boundary['timeout'] += int(tout.sum())
        boundary['both'] += int(both.sum())
        boundary['transitions'] += done.numel()
        boundary['gae_internal_checked'] += raw[:-1].numel()
        boundary['gae_final_done_checked'] += int(last_done.sum())
        boundary['gae_final_continuing_unobserved'] += int((~last_done).sum())
        boundary['timeout_bootstrap_sum'] += float(timeout_bonus.double().sum())
        boundary['both_timeout_bootstrap_sum'] += float(timeout_bonus[both].double().sum())
        sigma = rollout['sigma'][0, 0]
        assert float((rollout['sigma'] - sigma).abs().max()) == 0.
        sampling_rms = (rollout['actions'] - rollout['mu']).square().mean((0, 1)).sqrt()
        row = {'update': update, 'transition_count': 24*n,
               'means': dict(zip(FEATURES, (sum_features / (24*n)).tolist())),
               'command_counts_stand_straight_turn_mixed': command_counts.tolist(),
               'episode_age_counts_0_49_50_99_100_149_150_199_200plus': age_counts.tolist(),
               'termination_counts_same_age_bins': term_age_counts.tolist(),
               'command_groups': {str(k): {'count': int(group_counts[k]),
                    'means': dict(zip(FEATURES, (group_sums[k] / max(1, group_counts[k])).tolist()))} for k in range(4)},
               'terminated': int(term.sum()), 'timeouts': int(tout.sum()), 'both': int(both.sum()),
               'sigma': sigma.tolist(), 'sampled_action_minus_mean_rms': sampling_rms.tolist()}
        rows.append(row)
        artifacts.append({'update': update, 'index_sha256': digest(directory / 'index.json'),
                          'verified_artifacts': 25, 'rollout_sha256': index[directory.name+'/rollout.pt']['sha256']})
        del rollout, raw, expected, delta, expected_reward, physical_reward, terminated, timeout, timeout_bonus
        gc.collect()
        if update % 10 == 1 or update in (317, 318, 319):
            print(json.dumps({'update': update, 'window_count': sum(map(len, windows)),
                              'private_bytes': psutil.Process().memory_info().private}), flush=True)
    matrix, metadata = np.concatenate(windows), np.concatenate(window_metadata)
    np.savez_compressed(output / 'continuous_windows.npz', values=matrix, metadata=metadata,
                        features=np.array(FEATURES),
                        metadata_names=np.array(('env_id','start_update','end_update','episode_age_end','command_vx','command_yaw','command_height')))
    del windows, window_metadata
    cmd, age, end_update = metadata[:, 4:7], metadata[:, 3].astype(int), metadata[:, 2].astype(int)
    groups = classification(cmd)
    windows_summary = {'all': describe(matrix),
                       'command_groups': {str(k): describe(matrix, groups == k) for k in range(4)},
                       'matched': matched_comparison(matrix, cmd, age, end_update, metadata[:, 0]),
                       'matched_same_environment': matched_comparison(matrix, cmd, age, end_update, metadata[:, 0], True)}
    for label, first, last in (('early',251,275),('precritical',301,317),('postcritical',319,335),('late',377,401)):
        mask = (metadata[:, 1] >= first) & (end_update <= last)
        windows_summary[label] = describe(matrix, mask)
        windows_summary[label]['command_groups'] = {str(k): describe(matrix, mask & (groups == k)) for k in range(4)}
    series = read_json(run / 'series_stand_progress.json')
    reward_progress = read_json(run / 'causal_gradient_analysis.json')['reward_progress']
    for row, old in zip(rows, reward_progress, strict=True):
        assert row['update'] == old['update']
        assert np.max(np.abs(np.array([row['means'][k] for k in REWARDS])-old['weighted_mean'])) < 1e-7
        assert abs(sum(row['means'][k] for k in REWARDS) - row['means']['total_reward']) < 1e-7
        assert abs(row['means']['leg_acceleration_reward']+row['means']['wheel_acceleration_reward']-row['means']['dof_acc']) < 1e-9
    result = {'schema_version':'RewardEconomicsAuditV1', 'run':str(run), 'features':list(FEATURES),
              'cpu_torch_version':torch.__version__, 'control_dt':0.02, 'updates':rows,
              'source_sha256':digest(__file__), 'artifact_validation':artifacts,
              'checks':failures, 'boundaries':boundary, 'continuous_windows':windows_summary,
              'mujoco_stand_existing_series':series,
              'window_method':'Nonoverlapping continuous 50-step windows; episode age>50; no termination, reset, or command changes inside a window. A window can cross a PPO update; analysis stage filters use both start and end updates.',
              'frozen_before':frozen, 'frozen_after':frozen_identity()}
    write_json(output / 'results.json', result)
    print(json.dumps({'output':str(output),'windows':len(matrix),'checks':failures,'boundaries':boundary},indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.run / 'reward_economics_20261019'
    output.mkdir(exist_ok=False)
    analyse(args.run, output)


if __name__ == '__main__':
    main()
