import json
import sys

import torch

from debug.sim2sim.audit_reward_economics_20261019 import DEFAULT_RUN as RUN, digest, frozen_identity, read_json, write_json
from debug.sim2sim.wheel_feedback_reward_origin_20261020 import OUTPUT, UPDATES, discounted_sum


def main():
    torch.set_num_threads(1)
    assert torch.version.cuda is None
    frozen = frozen_identity()
    destination = OUTPUT / 'weight_sensitivity'
    destination.mkdir(exist_ok=False)
    sys.path.insert(0, str(RUN))
    import causal_gradient_analysis as base

    result = {'source_sha256': digest(__file__), 'updates': [],
              'method': 'Local derivative with respect to multiplying one recorded weighted reward by alpha at alpha=1. The saved critic values and transitions remain fixed; recompute total-advantage centering and standard deviation analytically. Original PPO branch, gradient clipping coefficient and Adam second-moment denominators remain fixed. Includes propagation of fresh gradients through later first moments within the same update. CPU Jacobian estimate, not a full new optimizer/rollout trajectory.'}
    for update in UPDATES:
        _, rollout, _, weighted = base.load_update(update)
        source = read_json(OUTPUT / f'update_{update:04d}.json')
        reward_gae = discounted_sum(weighted, rollout['dones'].bool(), .99 * .95).double().reshape(-1, 18)
        raw = rollout['raw_advantages'].double().flatten()
        centered = raw - raw.mean()
        scale = raw.std() + 1e-8
        constants = ((centered[:, None] * (reward_gae - reward_gae.mean(0))).sum(0) / centered.square().sum()) * (raw.std() / scale)
        contributions = source['contributions']['with_within_update_momentum']
        cpu_total = sum(value for name, value in contributions.items() if name != 'unassigned_cpu_residual')
        sensitivities = {name: contributions[name] - float(constants[index]) * cpu_total
                         for index, name in enumerate(base.REWARDS[:18])}
        errors = {}
        for name in ('action_smooth_wheel', 'tracking_yaw_rate_enhance'):
            column = base.REWARDS.index(name)
            reward = reward_gae[:, column]
            analytic = (reward - reward.mean()) / scale - centered / scale * constants[column]
            epsilon = 1e-4
            upper, lower = raw + epsilon * reward, raw - epsilon * reward
            finite = ((upper - upper.mean()) / (upper.std() + 1e-8)
                      - (lower - lower.mean()) / (lower.std() + 1e-8)) / (2 * epsilon)
            error = float((analytic - finite).abs().max())
            assert error < 1e-7
            errors[name] = error
        row = {'update': update, 'normalization_variance_derivative': dict(zip(base.REWARDS[:18], constants.tolist(), strict=True)),
               'reward_scale_derivative': sensitivities, 'analytic_advantage_derivative_errors': errors}
        result['updates'].append(row)
    result['positive_derivative_counts'] = {name: sum(row['reward_scale_derivative'][name] > 0 for row in result['updates'])
                                          for name in base.REWARDS[:18]}
    assert frozen_identity() == frozen
    result['frozen_files_checked'] = 118
    write_json(destination / 'result.json', result)
    print(json.dumps({'positive_derivative_counts': result['positive_derivative_counts'],
                      'update318': next(row for row in result['updates'] if row['update'] == 318)}, indent=2))


if __name__ == '__main__':
    main()
