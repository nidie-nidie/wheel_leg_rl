import json
from pathlib import Path

import numpy as np
import torch

from debug.sim2sim.audit_reward_economics_20261019 import DEFAULT_RUN, FEATURES, FROZEN, REWARDS, digest, read_json, write_json


def equal_tree(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left,right)
    if isinstance(left,np.ndarray):
        return isinstance(right,np.ndarray) and left.dtype == right.dtype and np.array_equal(left,right)
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys() == right.keys() and all(equal_tree(left[k],right[k]) for k in left)
    if isinstance(left,(tuple,list)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_tree(a,b) for a,b in zip(left,right,strict=True))
    return left == right


def main():
    torch.set_num_threads(1)
    assert torch.version.cuda is None
    output = DEFAULT_RUN/'reward_economics_20261019'
    result, supplement = read_json(output/'results.json'), read_json(output/'supplement.json')
    scripts = Path(__file__).resolve().parent
    assert result['source_sha256'] == digest(scripts/'audit_reward_economics_20261019.py')
    assert supplement['source_sha256'] == digest(scripts/'finish_reward_economics_20261019.py')
    assert supplement['result_sha256'] == digest(output/'results.json')
    assert supplement['continuous_windows_sha256'] == digest(output/'continuous_windows.npz')
    frozen = read_json(FROZEN)
    assert len(frozen) == 118
    assert result['frozen_before'] == result['frozen_after']
    for entry in frozen:
        assert digest(entry['path']) == entry['sha256']
    archive = np.load(output/'continuous_windows.npz',allow_pickle=False)
    values = archive['values']
    assert values.shape == (254610,len(FEATURES))
    assert np.isfinite(values).all()
    reward_sum_error = float(np.max(np.abs(values[:,:18].astype(np.float64).sum(-1)-values[:,18])))
    assert reward_sum_error < 2e-8
    assert sum(row['verified_artifacts'] for row in result['artifact_validation']) == 3775
    assert len(result['updates']) == 151 and result['boundaries']['transitions'] == 22265856
    assert len(supplement['mujoco_reward_recompute']) == len(supplement['mujoco_existing_csv_identity']) == 24
    match = supplement['mode_sensitive_matched_statistics']
    assert abs(sum(match['high_minus_low'][key] for key in REWARDS)-match['high_minus_low']['total_reward']) < 1e-8
    reference = DEFAULT_RUN.parent/'2026-10-17_14-57-16_rtx5070_fudan-v1_baseline-yaw1-headingoff-env6144-fresh1000-20261017'
    identities = []
    for number in (300,400):
        actual_path, reference_path = DEFAULT_RUN/f'model_{number}.pt', reference/f'model_{number}.pt'
        actual = torch.load(actual_path,map_location='cpu',weights_only=False,mmap=True)
        expected = torch.load(reference_path,map_location='cpu',weights_only=False,mmap=True)
        checks = {key:equal_tree(actual[key],expected[key]) for key in ('model_state_dict','optimizer_state_dict')}
        for key in ('rng_state','environment_rng_state','estimator_monitor_state',
                    'algorithm_learning_rate','optimizer_learning_rates'):
            checks[key] = equal_tree(actual['infos'][key],expected['infos'][key])
        assert all(checks.values()), (number,checks)
        identities.append({'completed_updates':number+1,'checks':checks,'actual_sha256':digest(actual_path),
                           'reference_sha256':digest(reference_path)})
    verified = {'frozen_files_checked':len(frozen),'verified_artifacts':3775,'transitions':22265856,
                'continuous_windows':len(values),'window_reward_sum_max_abs_error':reward_sum_error,
                'mujoco_trajectories_recomputed':24,'source_policy_identity':identities,
                'results_sha256':digest(output/'results.json'),'supplement_sha256':digest(output/'supplement.json'),
                'verification_source_sha256':digest(__file__),'torch_version':torch.__version__,
                'rng_scope':'Stored CPU/CUDA RNG states compared for equality without restoring CUDA generators; previous full strict artifact validation is retained.'}
    write_json(output/'verification.json',verified)
    print(json.dumps(verified,indent=2),flush=True)


if __name__ == '__main__':
    main()
