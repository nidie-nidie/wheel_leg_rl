import csv
import json

import numpy as np
import torch
from tensordict import TensorDict

from debug.sim2sim.action_mode_engine_probe_20261020 import PROJECT, AUDIT
from debug.sim2sim.audit_reward_economics_20261019 import DEFAULT_RUN as RUN, frozen_identity, digest, read_json, write_json
from debug.sim2sim.wheel_feedback_paths_20261020 import OUTPUT as PARENT
from wheelleg_dreamwaq.algorithms.dreamwaq.actor_critic import DreamWaQActorCritic


OUTPUT = PARENT / 'actual_update318'
KEYS = ('actor.0.weight', 'actor.0.bias', 'actor.2.weight', 'actor.2.bias',
        'actor.4.weight', 'actor.4.bias', 'actor.6.weight', 'actor.6.bias')


def make_policy(state):
    observations = TensorDict({'policy': torch.zeros(1, 25), 'policy_history': torch.zeros(1, 125),
                               'critic': torch.zeros(1, 41)}, batch_size=[1])
    policy = DreamWaQActorCritic(observations, {'policy': ['policy'], 'critic': ['critic']}, 6,
                                decoder_action_clip=None)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return policy


def probe_inputs(policy, histories, epsilon=.001):
    inputs = []
    with torch.no_grad():
        for frame in range(5):
            perturbation = torch.zeros_like(histories).view(-1, 5, 25)
            perturbation[:, frame, 23:25] = epsilon
            perturbation = perturbation.reshape(-1, 125)
            for sign in (-1, 1):
                history = histories + sign * perturbation
                encoded = policy.cenet.encode(history)
                inputs.append(torch.cat((history[:, -25:], encoded.estimated_velocity, encoded.context_mu), -1))
    return torch.stack(inputs)


def score(actor, inputs, frequency, epsilon=.001):
    output = actor(inputs.flatten(0, 1)).clamp(-20, 20)
    common = output[:, 4:].mean(-1).reshape(5, 2, inputs.shape[1])
    kernels = (common[:, 1] - common[:, 0]) / (2 * epsilon)
    phase = 2 * torch.pi * frequency * .02 * torch.arange(4, -1, -1)
    real = (kernels * torch.cos(phase)[:, None]).sum(0)
    imaginary = (kernels * torch.sin(phase)[:, None]).sum(0)
    gain_squared = (real.square() + imaginary.square()).mean()
    return gain_squared, kernels.mean(-1)


def dot(left, right):
    return float(sum((a.double() * b.double()).sum() for a, b in zip(left, right, strict=True)))


def main():
    torch.set_num_threads(1)
    assert torch.version.cuda is None
    before = frozen_identity()
    OUTPUT.mkdir(exist_ok=False)
    evidence = RUN / 'training_evidence/update_0318'
    index = read_json(evidence / 'index.json')
    assert index['complete'] and index['update'] == 318
    artifact_index = {row['path'].split('/')[-1]: row for row in index['artifacts']}
    inputs_identity = read_json(AUDIT / 'within_run_training/objective_and_response.json')['input_files']
    histories, identities = [], []
    for update in (301, 401):
        path = AUDIT / f'feedback_probes/6144_{update}/baseline_internal.npz'
        assert digest(path) == inputs_identity[f'6144_{update}']
        archive = np.load(path, allow_pickle=False)
        selection = np.linspace(0, len(archive['history']) - 1, 16).astype(int)
        histories.append(archive['history'][selection])
        identities.append({'path': str(path), 'sha256': digest(path), 'indices': selection.tolist()})
        if update == 401:
            common = archive['action'][:, 4:].mean(-1)
            frequencies = np.fft.rfftfreq(len(common), .02)
            power = np.abs(np.fft.rfft(common - common.mean())) ** 2
            power[frequencies < 2] = 0
            frequency = float(frequencies[np.argmax(power)])
    histories = torch.from_numpy(np.concatenate(histories))
    result = {'update': 318, 'frequency_hz': frequency, 'probe_identity': identities,
              'method': 'Real 40-step Adam numerator split with each actual second-moment denominator fixed. Projections use the local first derivative of squared wheel-common history frequency response on 32 saved real states. This is a local sensitivity attribution, not a closed-loop stability theorem or a fresh training run.',
              'source_sha256': digest(__file__), 'minibatches': [], 'artifacts': []}
    sum_momentum = None
    sum_fresh = None
    for number in range(40):
        path = evidence / f'minibatch_{number:02d}.pt'
        record = artifact_index[path.name]
        assert path.stat().st_size == record['bytes'] and digest(path) == record['sha256']
        batch = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        policy = make_policy(batch['policy_before'])
        inputs = probe_inputs(policy, histories)
        actual_before, kernel = score(policy.actor, inputs, frequency)
        parameters = list(policy.actor.parameters())
        metric_gradients = torch.autograd.grad(actual_before, parameters)
        group = batch['parameter_groups'][0]
        assert group['weight_decay'] == 0
        beta1, beta2 = group['betas']
        momentum, fresh, actual = [], [], []
        for key in KEYS:
            state = batch['actor_adam_before'][key]
            gradient = batch['actor_gradient_after_clipping'][key]
            step = int(state['step'].item()) + 1
            second = beta2 * state['exp_avg_sq'] + (1 - beta2) * gradient.square()
            denominator = (second / (1 - beta2 ** step)).sqrt() + group['eps']
            factor = -group['lr'] / (1 - beta1 ** step) / denominator
            momentum.append(factor * beta1 * state['exp_avg'])
            fresh.append(factor * (1 - beta1) * gradient)
            actual.append(batch['actor_parameters_after'][key] - batch['policy_before'][key])
        error = max(float((a + b - c).abs().max()) for a, b, c in zip(momentum, fresh, actual, strict=True))
        assert error < 1e-7, error
        if sum_momentum is None:
            sum_momentum = [torch.zeros_like(value) for value in momentum]
            sum_fresh = [torch.zeros_like(value) for value in fresh]
        for index_parameter in range(len(momentum)):
            sum_momentum[index_parameter] += momentum[index_parameter]
            sum_fresh[index_parameter] += fresh[index_parameter]
        with torch.no_grad():
            for parameter, key in zip(parameters, KEYS, strict=True):
                parameter.copy_(batch['actor_parameters_after'][key])
            actual_after, kernel_after = score(policy.actor, inputs, frequency)
        row = {'minibatch': number, 'frequency_response_squared_before': float(actual_before),
               'frequency_response_squared_after_actor_only': float(actual_after),
               'actual_metric_change_actor_only': float(actual_after - actual_before),
               'first_order_actual_change': dot(metric_gradients, actual),
               'carried_first_moment_change': dot(metric_gradients, momentum),
               'current_gradient_numerator_change': dot(metric_gradients, fresh),
               'adam_split_max_abs_error': error, 'kernel_before': kernel.detach().tolist(),
               'kernel_after_actor_only': kernel_after.detach().tolist(),
               'learning_rate': group['lr'], 'beta1': beta1, 'beta2': beta2,
               'step_before': int(batch['actor_adam_before'][KEYS[0]]['step'].item())}
        result['minibatches'].append(row)
        result['artifacts'].append({'path': str(path), 'sha256': record['sha256']})
        write_json(OUTPUT / 'progress.json', result)
        print(json.dumps(row), flush=True)
    actual_payload = torch.load(RUN / 'model_317.pt', map_location='cpu', weights_only=False, mmap=True)
    for label, removed in (('without_carried_first_moment', sum_momentum), ('without_current_gradient_numerator', sum_fresh)):
        state = {key: value.clone() for key, value in actual_payload['model_state_dict'].items()}
        for key, delta in zip(KEYS, removed, strict=True):
            state[key] -= delta
        path = OUTPUT / f'{label}_diagnostic_policy_state.pt'
        torch.save({'model_state_dict': state, 'diagnostic_only': True, 'source_checkpoint': str(RUN / 'model_317.pt'),
                    'source_checkpoint_sha256': digest(RUN / 'model_317.pt'), 'removed_component': label}, path)
        result[label] = {'path': str(path), 'sha256': digest(path), 'removed_delta_norm': float(torch.cat([value.flatten() for value in removed]).norm())}
    result['summary'] = {key: sum(row[key] for row in result['minibatches']) for key in (
        'actual_metric_change_actor_only', 'first_order_actual_change', 'carried_first_moment_change', 'current_gradient_numerator_change')}
    result['summary']['max_adam_split_error'] = max(row['adam_split_max_abs_error'] for row in result['minibatches'])
    assert frozen_identity() == before
    result['frozen_files_checked'] = 118
    write_json(OUTPUT / 'result.json', result)
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
