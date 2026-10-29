import gc
import json
import sys
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.distributions import Normal
from torch.func import functional_call, jvp

from debug.sim2sim.audit_reward_economics_20261019 import DEFAULT_RUN as RUN, digest, frozen_identity, read_json, write_json
from debug.sim2sim.wheel_feedback_update318_20261020 import OUTPUT as PREVIOUS, KEYS, make_policy, probe_inputs, score, dot


OUTPUT = Path(__file__).resolve().parents[0] / 'runs/wheel-feedback-reward-origin-20261020'
UPDATES = (301, 313, 317, 318, 319, 320, 321, 350, 400, 401)
VALUE_NAMES = ('value_current_baseline', 'value_internal_future', 'value_horizon_inferred',
               'value_timeout_compensation', 'gae_arithmetic_residual')


def discounted_sum(values, dones, gamma_lambda):
    result = torch.zeros_like(values)
    carry = torch.zeros_like(values[0])
    for step in reversed(range(len(values))):
        carry = values[step] + gamma_lambda * (~dones[step]).float() * carry
        result[step] = carry
    return result


def expanded_terms(rollout, weighted, original):
    raw, values = rollout['raw_advantages'], rollout['values']
    dones = rollout['dones'].bool()
    gamma, lam = float(rollout['algorithm']['gamma']), float(rollout['algorithm']['lam'])
    reward_parts = discounted_sum(weighted, dones, gamma * lam)
    baseline = -values
    internal = torch.zeros_like(values)
    for step in reversed(range(len(values) - 1)):
        internal[step] = (~dones[step]).float() * (gamma * (1 - lam) * values[step + 1]
                                                            + gamma * lam * internal[step + 1])
    timeout_bonus = rollout['rewards'] - weighted.sum(-1, keepdim=True)
    timeout = discounted_sum(timeout_bonus, dones, gamma * lam)
    # 原记录没有独立保存末端V；从最后TD残差推得的量必须标明来源。
    horizon_seed = torch.zeros_like(values)
    horizon_seed[-1] = (raw[-1] - rollout['rewards'][-1] + values[-1]) * (~dones[-1]).float()
    horizon = discounted_sum(horizon_seed, dones, gamma * lam)
    residual = raw - reward_parts.sum(-1, keepdim=True) - baseline - internal - horizon - timeout
    parts = torch.cat((reward_parts, baseline, internal, horizon, timeout, residual), -1)
    scale = raw.std() + 1e-8
    normalized = (parts - parts.mean((0, 1), keepdim=True)) / scale
    sum_error = float((normalized.sum(-1, keepdim=True) - rollout['advantages']).abs().max())
    value_error = float((normalized[..., 18:].sum(-1) - original[..., 18]).abs().max())
    assert sum_error < 2e-5 and value_error < 2e-5, (sum_error, value_error)
    return normalized, {'normalized_sum_error': sum_error, 'value_subdivision_error': value_error,
                        'raw_gae_roundoff_max_abs': float(residual.abs().max()),
                        'raw_component_std': parts.std((0, 1)).tolist(),
                        'horizon_value_independently_recorded': False}


def load_histories(reference):
    histories = []
    for record in reference['probe_identity']:
        assert digest(record['path']) == record['sha256']
        histories.append(np.load(record['path'], allow_pickle=False)['history'][record['indices']])
    return torch.from_numpy(np.concatenate(histories))


def prepare(update, histories, frequency):
    directory = RUN / f'training_evidence/update_{update:04d}'
    index = read_json(directory / 'index.json')
    assert index['complete'] and index['update'] == update
    for record in index['artifacts']:
        path = directory / record['path'].split('/')[-1]
        assert path.stat().st_size == record['bytes'] and digest(path) == record['sha256'], str(path)
    rows, directions = [], []
    initial_moment, previous_moment = None, None
    for number in range(40):
        batch = torch.load(directory / f'minibatch_{number:02d}.pt', map_location='cpu', weights_only=False, mmap=True)
        policy = make_policy(batch['policy_before'])
        inputs = probe_inputs(policy, histories)
        metric, _ = score(policy.actor, inputs, frequency)
        gradients = torch.autograd.grad(metric, tuple(policy.actor.parameters()))
        group = batch['parameter_groups'][0]
        beta1, beta2 = group['betas']
        assert beta1 == .9 and group['weight_decay'] == 0
        if initial_moment is None:
            initial_moment = [batch['actor_adam_before'][key]['exp_avg'].clone() for key in KEYS]
        actual, fresh, carry, incoming, coefficients, next_moment = [], [], [], [], [], []
        moment_error = 0.
        for part, (key, derivative) in enumerate(zip(KEYS, gradients, strict=True)):
            state = batch['actor_adam_before'][key]
            if previous_moment is not None:
                moment_error = max(moment_error, float((state['exp_avg'] - previous_moment[part]).abs().max()))
            gradient = batch['actor_gradient_after_clipping'][key]
            step = int(state['step']) + 1
            second = beta2 * state['exp_avg_sq'] + (1 - beta2) * gradient.square()
            denominator = (second / (1 - beta2 ** step)).sqrt() + group['eps']
            factor = -group['lr'] / (1 - beta1 ** step) / denominator
            coefficients.append((factor * derivative).detach())
            fresh.append(factor * (1 - beta1) * gradient)
            carry.append(factor * beta1 * state['exp_avg'])
            incoming.append(factor * beta1 ** (number + 1) * initial_moment[part])
            actual.append(batch['actor_parameters_after'][key] - batch['policy_before'][key])
            next_moment.append(beta1 * state['exp_avg'] + (1 - beta1) * gradient)
        previous_moment = next_moment
        adam_error = max(float((a + b - c).abs().max()) for a, b, c in zip(fresh, carry, actual, strict=True))
        assert adam_error < 1e-7 and moment_error < 1e-6, (update, number, adam_error, moment_error)
        with torch.no_grad():
            for parameter, key in zip(policy.actor.parameters(), KEYS, strict=True):
                parameter.copy_(batch['actor_parameters_after'][key])
            after, _ = score(policy.actor, inputs, frequency)
        row = {'minibatch': number, 'score_before': float(metric), 'actor_only_score_change': float(after - metric),
               'actual_projection': dot(gradients, actual), 'fresh_projection': dot(gradients, fresh),
               'carried_projection': dot(gradients, carry), 'incoming_before_update_projection': dot(gradients, incoming),
               'adam_reconstruction_error': adam_error, 'first_moment_recurrence_error': moment_error,
               'learning_rate': group['lr']}
        rows.append(row)
        directions.append(coefficients)
        del policy, inputs, gradients, batch
    # 在真实轨迹和二阶动量分母固定时，反向累积某次梯度对其后所有一阶动量步的影响。
    future = [torch.zeros_like(value) for value in directions[0]]
    propagated = [None] * 40
    for number in reversed(range(40)):
        future = [value + .9 * following for value, following in zip(directions[number], future, strict=True)]
        propagated[number] = [.1 * value for value in future]
    incoming_identity = dot(initial_moment, [.9 * value for value in future])
    incoming_sum = sum(row['incoming_before_update_projection'] for row in rows)
    assert abs(incoming_identity - incoming_sum) < 2e-7
    return directory, rows, directions, propagated, {'index_sha256': digest(directory / 'index.json'),
            'verified_artifacts': len(index['artifacts']), 'incoming_adjoint_identity_error': abs(incoming_identity - incoming_sum)}


def analyse(update, histories, frequency, base):
    directory, rows, direct, propagated, identity = prepare(update, histories, frequency)
    folder, rollout, original, weighted = base.load_update(update)
    terms, gae_checks = expanded_terms(rollout, weighted, original)
    names = tuple(base.REWARDS[:18]) + VALUE_NAMES
    observations = rollout['observations']['policy'].reshape(-1, 25)
    actions = rollout['actions'].reshape(-1, 6)
    old_probs = rollout['actions_log_prob'].reshape(-1)
    advantages = rollout['advantages'].reshape(-1)
    terms = terms.reshape(-1, len(names))
    for number in range(40):
        batch = torch.load(folder / f'minibatch_{number:02d}.pt', map_location='cpu', weights_only=False, mmap=True)
        policy = make_policy(batch['policy_before'])
        parameters = dict(policy.actor.named_parameters())
        indices, forward = batch['indices'].long(), batch['forward']
        data = torch.cat((observations[indices], forward['estimated_velocity'], forward['context_mu']), -1)
        action, old, advantage, component = actions[indices], old_probs[indices], advantages[indices], terms[indices]

        def losses(values):
            cpu_mean = functional_call(policy.actor, values, (data,)).clamp(-20, 20)
            mean = forward['action_mean'] + (cpu_mean - cpu_mean.detach())
            cpu_log = Normal(mean, forward['action_sigma'], validate_args=False).log_prob(action).sum(-1)
            log_probability = forward['action_log_prob'] + (cpu_log - cpu_log.detach())
            ratio = torch.exp(log_probability - old)
            clipped = ratio.clamp(.8, 1.2)
            selected = torch.where(-advantage * ratio >= -advantage * clipped, ratio, clipped)
            return -(component * selected[:, None]).mean(0)

        gradient = torch.autograd.grad(losses(parameters).sum(), tuple(parameters.values()))
        residuals = [batch['actor_gradient_before_clipping'][key] - value for key, value in zip(KEYS, gradient, strict=True)]
        actual_norm = torch.cat([batch['actor_gradient_before_clipping'][key].flatten() for key in KEYS]).norm()
        residual_norm = torch.cat([value.flatten() for value in residuals]).norm()
        clip = min(1., float(rollout['algorithm']['max_grad_norm']) / (float(batch['global_gradient_norm_before_clipping']) + 1e-6))
        clip_error = max(float((batch['actor_gradient_before_clipping'][key] * clip - batch['actor_gradient_after_clipping'][key]).abs().max()) for key in KEYS)
        assert clip_error < 1e-7
        row = rows[number]
        row['cpu_residual_relative_norm'] = float(residual_norm / actual_norm)
        for label, vectors in (('fresh', [.1 * value for value in direct[number]]), ('with_within_update_momentum', propagated[number])):
            direction = {name: value * clip for name, value in zip(parameters, vectors, strict=True)}
            _, change = jvp(losses, (parameters,), (direction,))
            row[label + '_reward_estimates'] = dict(zip(names, change.detach().double().tolist(), strict=True))
            row[label + '_unassigned_cpu_residual'] = dot(residuals, tuple(direction.values()))
            reconstructed = float(change.double().sum()) + row[label + '_unassigned_cpu_residual']
            actual = dot([batch['actor_gradient_before_clipping'][key] for key in KEYS], tuple(direction.values()))
            row[label + '_projection_accounting_error'] = abs(reconstructed - actual)
            assert row[label + '_projection_accounting_error'] < 2e-6
            if label == 'fresh':
                assert abs(actual - row['fresh_projection']) < 2e-7
        del batch, policy, parameters, gradient, residuals
        gc.collect()
    totals = {}
    for label in ('fresh', 'with_within_update_momentum'):
        totals[label] = {name: sum(row[label + '_reward_estimates'][name] for row in rows) for name in names}
        totals[label]['unassigned_cpu_residual'] = sum(row[label + '_unassigned_cpu_residual'] for row in rows)
    summary = {name: sum(row[name] for row in rows) for name in ('actor_only_score_change', 'actual_projection',
                'fresh_projection', 'carried_projection', 'incoming_before_update_projection')}
    summary['within_update_previous_gradients_projection'] = summary['carried_projection'] - summary['incoming_before_update_projection']
    summary['expanded_reward_plus_incoming_accounting_error'] = abs(sum(totals['with_within_update_momentum'].values())
                                                   + summary['incoming_before_update_projection'] - summary['actual_projection'])
    assert summary['expanded_reward_plus_incoming_accounting_error'] < 2e-6
    result = {'update': update, 'identity': identity, 'gae_checks': gae_checks, 'summary': summary,
              'contributions': totals, 'minibatches': rows,
              'max_cpu_residual_relative_norm': max(row['cpu_residual_relative_norm'] for row in rows)}
    write_json(OUTPUT / f'update_{update:04d}.json', result)
    print(json.dumps({'update': update, 'summary': summary,
                      'contributions': totals['with_within_update_momentum'],
                      'max_cpu_residual_relative_norm': result['max_cpu_residual_relative_norm'],
                      'private_bytes': psutil.Process().memory_info().private}), flush=True)
    return result


def main():
    torch.set_num_threads(1)
    assert torch.version.cuda is None
    assert psutil.virtual_memory().available > 3 * 1024**3
    frozen = frozen_identity()
    OUTPUT.mkdir(exist_ok=False)
    reference = read_json(PREVIOUS / 'result.json')
    histories = load_histories(reference)
    sys.path.insert(0, str(RUN))
    import causal_gradient_analysis as base

    result = {'source_sha256': digest(__file__), 'gradient_tool_sha256': digest(RUN / 'causal_gradient_analysis.py'),
              'frequency_hz': reference['frequency_hz'], 'probe_identity': reference['probe_identity'],
              'updates': [], 'strict_gpu_reward_attribution_passed': False,
              'method': 'Recorded real forward values and PPO branches; CPU actor Jacobian estimate. Each original Adam denominator and clipping factor fixed. The adjoint propagates each fresh gradient through subsequent first-moment steps within the same 40-minibatch update. Incoming pre-update moment and GPU/CPU gradient residual remain unassigned. This is local sensitivity attribution, not a re-trained policy or closed-loop stability proof.',
              'value_limit': 'One critic predicts total return. The value decomposition locates timing in GAE; it cannot uniquely assign the critic value to a reward source. Horizon bootstrap is inferred from saved raw advantages and not an independent final-value observation.'}
    for update in UPDATES:
        row = analyse(update, histories, result['frequency_hz'], base)
        result['updates'].append({key: value for key, value in row.items() if key != 'minibatches'})
        write_json(OUTPUT / 'progress.json', result)
        gc.collect()
    assert frozen_identity() == frozen
    result['frozen_files_checked'] = 118
    write_json(OUTPUT / 'result.json', result)


if __name__ == '__main__':
    main()
