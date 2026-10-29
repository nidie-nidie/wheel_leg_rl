import gc
import json
import sys

import torch
from torch.distributions import Normal
from torch.func import functional_call, jvp

from debug.sim2sim.audit_reward_economics_20261019 import DEFAULT_RUN as RUN, digest, frozen_identity, read_json, write_json
from debug.sim2sim.wheel_feedback_reward_origin_20261020 import OUTPUT, UPDATES, discounted_sum, load_histories, prepare
from debug.sim2sim.wheel_feedback_update318_20261020 import OUTPUT as PREVIOUS, make_policy


def main():
    torch.set_num_threads(1)
    assert torch.version.cuda is None
    frozen = frozen_identity()
    destination = OUTPUT / 'centering_sensitivity'
    destination.mkdir(exist_ok=False)
    reference = read_json(PREVIOUS / 'result.json')
    histories = load_histories(reference)
    sys.path.insert(0, str(RUN))
    import causal_gradient_analysis as base

    result = {'source_sha256': digest(__file__), 'updates': [],
              'method': 'Keep the total normalized advantage, original PPO branches, actual Adam denominator and incoming moment identical. Alternative decomposition leaves the eighteen reward GAE components uncentered and moves their centering constants into the combined value component. CPU Jacobian estimate only. This checks whether reward labels depend on this baseline convention.'}
    for update in UPDATES:
        folder, _, _, propagated, _ = prepare(update, histories, reference['frequency_hz'])
        _, rollout, _, weighted = base.load_update(update)
        original = read_json(OUTPUT / f'update_{update:04d}.json')
        constants = discounted_sum(weighted, rollout['dones'].bool(), .99 * .95).mean((0, 1)) / (rollout['raw_advantages'].std() + 1e-8)
        observations = rollout['observations']['policy'].flatten(0, 1)
        actions = rollout['actions'].flatten(0, 1)
        old = rollout['actions_log_prob'].flatten()
        advantage = rollout['advantages'].flatten()
        scalar_sum = 0.
        for number in range(40):
            batch = torch.load(folder / f'minibatch_{number:02d}.pt', map_location='cpu', weights_only=False, mmap=True)
            policy = make_policy(batch['policy_before'])
            parameters = dict(policy.actor.named_parameters())
            indices, forward = batch['indices'].long(), batch['forward']
            data = torch.cat((observations[indices], forward['estimated_velocity'], forward['context_mu']), -1)

            def constant_loss(values):
                cpu_mean = functional_call(policy.actor, values, (data,)).clamp(-20, 20)
                mean = forward['action_mean'] + (cpu_mean - cpu_mean.detach())
                cpu_log = Normal(mean, forward['action_sigma'], validate_args=False).log_prob(actions[indices]).sum(-1)
                probability = forward['action_log_prob'] + (cpu_log - cpu_log.detach())
                ratio = torch.exp(probability - old[indices])
                clipped = ratio.clamp(.8, 1.2)
                selected = torch.where(-advantage[indices] * ratio >= -advantage[indices] * clipped, ratio, clipped)
                return -selected.mean()

            clip = min(1., float(rollout['algorithm']['max_grad_norm']) / (float(batch['global_gradient_norm_before_clipping']) + 1e-6))
            direction = {name: value * clip for name, value in zip(parameters, propagated[number], strict=True)}
            _, derivative = jvp(constant_loss, (parameters,), (direction,))
            scalar_sum += float(derivative)
            del policy, parameters, direction, batch
            gc.collect()
        values = original['contributions']['with_within_update_momentum']
        alternative = {name: values[name] + float(constants[index]) * scalar_sum for index, name in enumerate(base.REWARDS[:18])}
        row = {'update': update, 'constant_loss_adam_projection': scalar_sum,
               'reward_normalized_mean_constants': dict(zip(base.REWARDS[:18], constants.double().tolist(), strict=True)),
               'centered': {name: values[name] for name in base.REWARDS[:18]},
               'uncentered_rewards': alternative,
               'compensating_value_change': -float(constants.double().sum()) * scalar_sum}
        result['updates'].append(row)
        write_json(destination / 'progress.json', result)
        print(json.dumps({'update': update, 'wheel_smooth_centered': values['action_smooth_wheel'],
                          'wheel_smooth_uncentered': alternative['action_smooth_wheel'],
                          'yaw_enhance_centered': values['tracking_yaw_rate_enhance'],
                          'yaw_enhance_uncentered': alternative['tracking_yaw_rate_enhance']}), flush=True)
    result['signs'] = {name: {'centered_positive': sum(row['centered'][name] > 0 for row in result['updates']),
                            'uncentered_positive': sum(row['uncentered_rewards'][name] > 0 for row in result['updates'])}
                       for name in base.REWARDS[:18]}
    assert frozen_identity() == frozen
    result['frozen_files_checked'] = 118
    write_json(destination / 'result.json', result)


if __name__ == '__main__':
    main()
