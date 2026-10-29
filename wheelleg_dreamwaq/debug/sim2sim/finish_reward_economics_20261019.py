import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from debug.sim2sim.audit_reward_economics_20261019 import DEFAULT_RUN, FEATURES, REWARDS, classification, digest, read_json, write_json
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import RawSplitStraightRewardWeights, compute_reward


RUN = DEFAULT_RUN
OUT = RUN / 'reward_economics_20261019'
FOLDER = RUN.parent / '2026-10-17_17-56-29_rtx5070_fudan-v1_baseline-yaw1-headingoff-env4096-fresh701-20261017/degeneration_audit/within_run_training'


def standardize(values, low, high, keys, names):
    _, groups = np.unique(keys.astype(np.int32), axis=0, return_inverse=True)
    size = int(groups.max()) + 1
    counts = [np.bincount(groups[mask], minlength=size) for mask in (low, high)]
    valid = (counts[0] >= 2) & (counts[1] >= 2)
    weights = np.minimum(*counts)[valid].astype(np.float64)
    result = {'strata': int(valid.sum()), 'low_overlap': int(counts[0][valid].sum()),
              'high_overlap': int(counts[1][valid].sum())}
    if not len(weights):
        return result
    means = []
    for mask, count in zip((low, high), counts, strict=True):
        means.append(np.array([(np.bincount(groups[mask], weights=values[mask, column], minlength=size)[valid]
                                / count[valid] * weights).sum() / weights.sum() for column in range(values.shape[1])]))
    result.update(low=dict(zip(names, means[0].tolist())), high=dict(zip(names, means[1].tolist())),
                  high_minus_low=dict(zip(names, (means[1]-means[0]).tolist())))
    return result


def main():
    torch.set_num_threads(1)
    assert torch.version.cuda is None
    result = read_json(OUT / 'results.json')
    archive = np.load(OUT / 'continuous_windows.npz', allow_pickle=False)
    matrix, metadata = archive['values'], archive['metadata']
    assert tuple(archive['features']) == FEATURES
    assert matrix.shape == (254610, len(FEATURES)) and np.isfinite(matrix).all()
    energy = matrix[:, FEATURES.index('leg_acceleration_squared_mean')]
    low_q, high_q = np.quantile(energy, [0.25,0.75])
    low, high = energy <= low_q, energy >= high_q
    keys = np.column_stack(((metadata[:,2]-251)//10, np.rint(metadata[:,4]/0.1),
                            np.rint(metadata[:,5]/0.1),np.rint(metadata[:,6]/0.01),metadata[:,3]//100))
    checked = standardize(matrix, low, high, keys, FEATURES)
    assert checked['strata'] == result['continuous_windows']['matched']['matched_strata']
    for key, value in checked['high_minus_low'].items():
        assert abs(value-result['continuous_windows']['matched']['high_minus_low'][key]) < 1e-10
    value_names = ('recorded_value','recorded_lambda_return','raw_advantage','normalized_advantage')
    value_windows = np.zeros((len(metadata),4),dtype=np.float64)
    history = []
    for update in range(251,402):
        directory = RUN/'training_evidence'/f'update_{update:04d}'
        rollout = torch.load(directory/'rollout.pt',map_location='cpu',weights_only=False,mmap=True)
        history.append(np.stack([rollout[key][:,:,0].numpy() for key in ('values','returns','raw_advantages','advantages')],axis=-1))
        history = history[-4:]
        indices = np.flatnonzero(metadata[:,2] == update)
        if len(indices):
            physical = torch.load(directory/'physical_step_00.pt',map_location='cpu',weights_only=False,mmap=True)
            env_ids = metadata[indices,0].astype(int)
            steps = metadata[indices,3].astype(int) - physical['episode_length'].numpy()[env_ids]
            assert np.all((steps>=0)&(steps<24))
            stacked = np.concatenate(history)
            prefix = np.concatenate((np.zeros((1,6144,4)),stacked.cumsum(axis=0,dtype=np.float64)),axis=0)
            end = steps + stacked.shape[0]-24
            assert np.all(end>=49), (update, steps.min(), steps.max(), metadata[indices[end<49]].tolist()[:3])
            value_windows[indices] = (prefix[end+1,env_ids]-prefix[end-49,env_ids])/50.
        del rollout
    mode_keys = np.column_stack((keys,classification(metadata[:,4:7])))
    mode_matching = standardize(matrix,low,high,mode_keys,FEATURES)
    value_matching = standardize(value_windows,low,high,mode_keys,value_names)
    phase_keys = mode_keys[:,1:]
    pre = (metadata[:,1]>=301)&(metadata[:,2]<=317)
    post = (metadata[:,1]>=319)&(metadata[:,2]<=335)
    phase_matching = standardize(matrix,pre,post,phase_keys,FEATURES)
    phases = {}
    for name, first, last in (('early',251,275),('precritical',301,317),('postcritical',319,335),
                              ('late',377,401),('first50',251,300),('last50',352,401)):
        rows = [row for row in result['updates'] if first<=row['update']<=last]
        means = {key:float(np.mean([row['means'][key] for row in rows])) for key in FEATURES}
        phases[name] = {'updates':[first,last], 'means':means,
                        'leg_rms':float(np.sqrt(means['leg_acceleration_squared_mean'])),
                        'wheel_rms':float(np.sqrt(means['wheel_acceleration_squared_mean'])),
                        'termination_fraction':sum(row['terminated'] for row in rows)/(len(rows)*24*6144),
                        'sigma_mean':np.array([row['sigma'] for row in rows]).mean(0).tolist()}
    temporal_delta = {key:phases['last50']['means'][key]-phases['first50']['means'][key] for key in FEATURES}
    termination_age = np.array([row['termination_counts_same_age_bins'] for row in result['updates']]).sum(0)
    mujoco = read_json(FOLDER/'real_trajectory_rewards.json')
    mujoco_identity = []
    reward_recompute = []
    reward_weights = RawSplitStraightRewardWeights(**mujoco['weights'])
    for update in ('301','401','1000'):
        for scenario, data in mujoco['models'][update]['scenarios'].items():
            assert digest(data['formal_csv']) == data['formal_csv_sha256']
            assert data['diagnostic_csv_sha256'] == data['formal_csv_sha256']
            assert data['actual_steps'] == 500
            mujoco_identity.append({'update':update,'scenario':scenario,'csv_sha256':data['formal_csv_sha256']})
            state_path = FOLDER/'reward_trajectory_diagnostics'/update/(scenario+'_states.npz')
            stored = np.load(state_path,allow_pickle=False)
            names = tuple(stored['reward_names'])
            assert names == REWARDS+('total',)
            state = SimpleNamespace(**{key:torch.from_numpy(stored[key]).float() for key in stored.files
                                       if key not in ('weighted_rewards','reward_names')})
            recorded_terminated = torch.from_numpy(stored['weighted_rewards'][:,names.index('termination')] != 0)
            assert not recorded_terminated.any()
            total, components = compute_reward(state,reward_weights,control_dt=0.02,
                                               terminated=recorded_terminated)
            rebuilt = np.column_stack([components[key].numpy() for key in REWARDS]+[total.numpy()])
            error = float(np.max(np.abs(rebuilt-stored['weighted_rewards'])))
            assert error < 3e-6
            averages = rebuilt[50:].mean(0,dtype=np.float64)
            for key,value in zip(names,averages,strict=True):
                assert abs(value-data['after_first_50_weighted_means'][key]) < 3e-7
            reward_recompute.append({'update':update,'scenario':scenario,'state_sha256':digest(state_path),
                                     'reward_max_abs_error':error})
    mujoco_delta = {key:mujoco['models']['401']['eight_scenario_equal_weight_after_first_50'][key]
                       -mujoco['models']['301']['eight_scenario_equal_weight_after_first_50'][key]
                    for key in (*REWARDS,'total')}
    supplement = {'source_sha256':digest(__file__), 'result_sha256':digest(OUT/'results.json'),
                  'continuous_windows_sha256':digest(OUT/'continuous_windows.npz'),
                  'independent_matched_statistic_recheck':checked, 'value_window_matching':value_matching,
                  'mode_sensitive_matched_statistics':mode_matching,
                  'matched_post_minus_pre':phase_matching, 'all_samples_phases':phases,
                  'all_samples_last50_minus_first50':temporal_delta,
                  'termination_age_counts_0_49_50_99_100_149_150_199_200plus':termination_age.tolist(),
                  'mujoco_reward_source_sha256':digest(FOLDER/'real_trajectory_rewards.json'),
                  'mujoco_existing_csv_identity':mujoco_identity, 'mujoco_401_minus_301':mujoco_delta,
                  'mujoco_reward_recompute':reward_recompute,
                  'value_window_note':'Averages of recorded predictions and GAE targets; targets are not independent Monte Carlo ground truth. Continuous windows may span policy updates.',
                  'exact_environment_matching_has_overlap':False}
    np.savez_compressed(OUT/'window_value_metrics.npz',values=value_windows,names=np.array(value_names))
    write_json(OUT/'supplement.json',supplement)
    print(json.dumps({'value_window_matching':value_matching,'all_samples_last50_minus_first50':temporal_delta,
                      'mujoco_401_minus_301':mujoco_delta,'matched_post_minus_pre':phase_matching},indent=2),flush=True)


if __name__ == '__main__':
    main()
