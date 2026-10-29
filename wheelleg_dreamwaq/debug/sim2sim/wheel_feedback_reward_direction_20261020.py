import gc
import json
import sys

import numpy as np
import torch
from torch.distributions import Normal
from torch.func import functional_call, jvp

from debug.sim2sim.action_mode_engine_probe_20261020 import AUDIT
from debug.sim2sim.audit_reward_economics_20261019 import DEFAULT_RUN as RUN, digest, frozen_identity, read_json, write_json
from debug.sim2sim.wheel_feedback_update318_20261020 import OUTPUT as INPUT, KEYS, make_policy, probe_inputs, score


OUTPUT = INPUT / 'reward_directions'


def main():
    torch.set_num_threads(1)
    assert torch.version.cuda is None
    before = frozen_identity()
    OUTPUT.mkdir(exist_ok=False)
    sys.path.insert(0, str(RUN))
    import causal_gradient_analysis as base

    evidence = RUN / 'training_evidence/update_0318'
    index = read_json(evidence / 'index.json')
    for record in index['artifacts']:
        path = evidence / record['path'].split('/')[-1]
        assert path.stat().st_size == record['bytes'] and digest(path) == record['sha256']
    folder, rollout, terms, _ = base.load_update(318)
    advantage_error = float((terms.sum(-1, keepdim=True) - rollout['advantages']).abs().max())
    assert advantage_error <= 2e-5
    source = read_json(INPUT / 'result.json')
    histories = []
    for record in source['probe_identity']:
        assert digest(record['path']) == record['sha256']
        histories.append(np.load(record['path'], allow_pickle=False)['history'][record['indices']])
    histories = torch.from_numpy(np.concatenate(histories))
    observations = rollout['observations']['policy'].reshape(-1, 25)
    actions = rollout['actions'].reshape(-1, 6)
    old_log_prob = rollout['actions_log_prob'].reshape(-1)
    advantages = rollout['advantages'].reshape(-1)
    terms = terms.reshape(-1, 19)
    rows = []
    result = {'source_sha256': digest(__file__), 'gradient_tool_sha256': digest(RUN / 'causal_gradient_analysis.py'),
              'update': 318, 'frequency_hz': source['frequency_hz'], 'advantage_reconstruction_error': advantage_error,
              'method': 'Exact forward directional derivative of each real PPO contribution. Total-advantage normalization, clipping branch, recorded global gradient scale, actual Adam second-moment denominator and actual pre-update policy are held fixed. Positive values locally increase squared wheel-common history response at the observed frequency. Carried first moment remains a separate contribution, with its reward origin unassigned.',
              'reward_names': list(base.REWARDS), 'minibatches': rows}
    for number in range(40):
        batch = torch.load(folder / f'minibatch_{number:02d}.pt', map_location='cpu', weights_only=False, mmap=True)
        policy = make_policy(batch['policy_before'])
        inputs = probe_inputs(policy, histories)
        metric, _ = score(policy.actor, inputs, source['frequency_hz'])
        actor_parameters = dict(policy.actor.named_parameters())
        metric_gradient = torch.autograd.grad(metric, tuple(actor_parameters.values()))
        indices = batch['indices'].long()
        data = torch.cat((observations[indices], batch['forward']['estimated_velocity'], batch['forward']['context_mu']), dim=-1)
        action, old, advantage, component = actions[indices], old_log_prob[indices], advantages[indices], terms[indices]
        sigma = batch['policy_before']['raw_std'].abs().clamp(1e-4, 2.)

        def losses(parameters):
            mean = functional_call(policy.actor, parameters, (data,)).clamp(-20, 20)
            log_prob = Normal(mean, sigma, validate_args=False).log_prob(action).sum(-1)
            ratio = torch.exp(log_prob - old)
            clipped_ratio = ratio.clamp(.8, 1.2)
            branch = -advantage * ratio >= -advantage * clipped_ratio
            selected = torch.where(branch, ratio, clipped_ratio)
            return -(component * selected[:, None]).mean(0)

        total_components = losses(actor_parameters)
        gradients = torch.autograd.grad(total_components.sum(), tuple(actor_parameters.values()))
        gradient_error = max(float((gradient - batch['actor_gradient_before_clipping'][key]).abs().max())
                             for gradient, key in zip(gradients, KEYS, strict=True))
        assert gradient_error <= 2e-5, gradient_error
        global_norm = float(batch['global_gradient_norm_before_clipping'])
        clip_factor = min(1., float(rollout['algorithm']['max_grad_norm']) / (global_norm + 1e-6))
        clip_error = max(float((batch['actor_gradient_before_clipping'][key] * clip_factor - batch['actor_gradient_after_clipping'][key]).abs().max()) for key in KEYS)
        assert clip_error <= 1e-7, clip_error
        group = batch['parameter_groups'][0]
        beta1, beta2 = group['betas']
        direction = {}
        for name, key, derivative in zip(actor_parameters, KEYS, metric_gradient, strict=True):
            state = batch['actor_adam_before'][key]
            gradient = batch['actor_gradient_after_clipping'][key]
            step = int(state['step']) + 1
            second = beta2 * state['exp_avg_sq'] + (1 - beta2) * gradient.square()
            denominator = (second / (1 - beta2 ** step)).sqrt() + group['eps']
            factor = -group['lr'] * (1 - beta1) / (1 - beta1 ** step) / denominator
            direction[name] = (derivative * factor * clip_factor).detach()
        _, directional = jvp(losses, (actor_parameters,), (direction,))
        values = directional.detach().double().tolist()
        reference = source['minibatches'][number]['current_gradient_numerator_change']
        projection_error = abs(sum(values) - reference)
        assert projection_error <= 2e-5, projection_error
        row = {'minibatch': number, 'gradient_sum_max_error': gradient_error,
               'recorded_gradient_clipping_max_error': clip_error, 'fresh_projection_sum_error': projection_error,
               'current_gradient_projection': reference, 'contributions': dict(zip(base.REWARDS, values, strict=True))}
        rows.append(row)
        write_json(OUTPUT / 'progress.json', result)
        print(json.dumps({'minibatch': number, 'gradient_error': gradient_error, 'projection_error': projection_error}), flush=True)
        del policy, inputs, actor_parameters, metric_gradient, total_components, gradients, direction, batch
        gc.collect()
    totals = {name: sum(row['contributions'][name] for row in rows) for name in base.REWARDS}
    result['summed_contributions'] = dict(sorted(totals.items(), key=lambda pair: pair[1], reverse=True))
    result['max_gradient_reconstruction_error'] = max(row['gradient_sum_max_error'] for row in rows)
    result['summed_fresh_projection_error'] = abs(sum(totals.values()) - source['summary']['current_gradient_numerator_change'])
    result['carried_first_moment_projection'] = source['summary']['carried_first_moment_change']
    assert frozen_identity() == before
    result['frozen_files_checked'] = 118
    write_json(OUTPUT / 'result.json', result)
    print(json.dumps({'summed_contributions': result['summed_contributions'], 'sum_error': result['summed_fresh_projection_error']}), flush=True)


if __name__ == '__main__':
    main()
