import gc
import json
import sys

import numpy as np
import torch
from torch.distributions import Normal
from torch.func import functional_call, jvp

from debug.sim2sim.audit_reward_economics_20261019 import DEFAULT_RUN as RUN, digest, frozen_identity, read_json, write_json
from debug.sim2sim.wheel_feedback_update318_20261020 import OUTPUT as INPUT, KEYS, make_policy, probe_inputs, score


OUTPUT = INPUT / 'reward_directions_recorded_forward_cpu'


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
              'update': 318, 'frequency_hz': source['frequency_hz'], 'strict_reward_attribution_passed': False,
              'method': 'CPU estimate with recorded GPU forward means, sigmas, log probabilities and original clipping branches fixed. Actor Jacobians use CPU arithmetic. The difference from the recorded total actor gradient is retained as a separate unassigned residual. Nineteen estimated reward/value contributions alone do not meet the strict original-gradient reconstruction gate. This estimate is not a completed GPU reward attribution.',
              'reward_names': list(base.REWARDS), 'minibatches': rows}
    for number in range(40):
        batch = torch.load(folder / f'minibatch_{number:02d}.pt', map_location='cpu', weights_only=False, mmap=True)
        policy = make_policy(batch['policy_before'])
        metric, _ = score(policy.actor, probe_inputs(policy, histories), source['frequency_hz'])
        parameters = dict(policy.actor.named_parameters())
        metric_gradient = torch.autograd.grad(metric, tuple(parameters.values()))
        indices = batch['indices'].long()
        data = torch.cat((observations[indices], batch['forward']['estimated_velocity'], batch['forward']['context_mu']), dim=-1)
        action, old, advantage, component = actions[indices], old_log_prob[indices], advantages[indices], terms[indices]
        forward = batch['forward']

        def losses(values):
            cpu_mean = functional_call(policy.actor, values, (data,)).clamp(-20, 20)
            # 使用真实前向数值，CPU仅提供局部反向导数；其误差单列，不能冒充GPU原计算。
            mean = forward['action_mean'] + (cpu_mean - cpu_mean.detach())
            cpu_log = Normal(mean, forward['action_sigma'], validate_args=False).log_prob(action).sum(-1)
            log_prob = forward['action_log_prob'] + (cpu_log - cpu_log.detach())
            ratio = torch.exp(log_prob - old)
            clipped = ratio.clamp(.8, 1.2)
            selected = torch.where(-advantage * ratio >= -advantage * clipped, ratio, clipped)
            return -(component * selected[:, None]).mean(0)

        components = losses(parameters)
        gradients = torch.autograd.grad(components.sum(), tuple(parameters.values()))
        residuals = [batch['actor_gradient_before_clipping'][key] - gradient for key, gradient in zip(KEYS, gradients, strict=True)]
        actual_norm = torch.cat([batch['actor_gradient_before_clipping'][key].flatten() for key in KEYS]).norm()
        residual_norm = torch.cat([value.flatten() for value in residuals]).norm()
        global_norm = float(batch['global_gradient_norm_before_clipping'])
        clip_factor = min(1., float(rollout['algorithm']['max_grad_norm']) / (global_norm + 1e-6))
        clip_error = max(float((batch['actor_gradient_before_clipping'][key] * clip_factor - batch['actor_gradient_after_clipping'][key]).abs().max()) for key in KEYS)
        assert clip_error <= 1e-7
        group = batch['parameter_groups'][0]
        beta1, beta2 = group['betas']
        direction = {}
        for name, key, derivative in zip(parameters, KEYS, metric_gradient, strict=True):
            state = batch['actor_adam_before'][key]
            gradient = batch['actor_gradient_after_clipping'][key]
            step = int(state['step']) + 1
            second = beta2 * state['exp_avg_sq'] + (1 - beta2) * gradient.square()
            denominator = (second / (1 - beta2 ** step)).sqrt() + group['eps']
            factor = -group['lr'] * (1 - beta1) / (1 - beta1 ** step) / denominator
            direction[name] = (derivative * factor * clip_factor).detach()
        _, directional = jvp(losses, (parameters,), (direction,))
        values = directional.detach().double().tolist()
        residual_projection = sum(float((value.double() * vector.double()).sum()) for value, vector in zip(residuals, direction.values(), strict=True))
        reference = source['minibatches'][number]['current_gradient_numerator_change']
        accounting_error = abs(sum(values) + residual_projection - reference)
        assert accounting_error < 2e-6
        row = {'minibatch': number, 'estimated_contributions': dict(zip(base.REWARDS, values, strict=True)),
               'unassigned_cpu_backward_residual_projection': residual_projection,
               'gradient_residual_max_abs': max(float(value.abs().max()) for value in residuals),
               'gradient_residual_relative_norm': float(residual_norm / actual_norm),
               'estimated_plus_residual_projection_error': accounting_error}
        rows.append(row)
        write_json(OUTPUT / 'progress.json', result)
        print(json.dumps({'minibatch': number, 'residual_relative_norm': row['gradient_residual_relative_norm'], 'accounting_error': accounting_error}), flush=True)
        del policy, parameters, metric_gradient, gradients, residuals, direction, components, batch
        gc.collect()
    totals = {name: sum(row['estimated_contributions'][name] for row in rows) for name in base.REWARDS}
    result['estimated_summed_contributions'] = dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))
    result['unassigned_cpu_backward_residual_projection'] = sum(row['unassigned_cpu_backward_residual_projection'] for row in rows)
    result['max_gradient_residual_relative_norm'] = max(row['gradient_residual_relative_norm'] for row in rows)
    result['carried_first_moment_projection'] = source['summary']['carried_first_moment_change']
    assert frozen_identity() == before
    result['frozen_files_checked'] = 118
    write_json(OUTPUT / 'result.json', result)
    print(json.dumps({'contributions_estimated': result['estimated_summed_contributions'], 'residual': result['unassigned_cpu_backward_residual_projection']}), flush=True)


if __name__ == '__main__':
    main()
