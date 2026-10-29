import gc
import json
import sys

import numpy as np
import torch
from torch.distributions import Normal

from debug.sim2sim.action_mode_engine_probe_20261020 import SIM
from debug.sim2sim.audit_reward_economics_20261019 import DEFAULT_RUN as RUN, digest, frozen_identity, read_json, write_json
from debug.sim2sim.evaluate_baseline_mujoco import run_protocol
from debug.sim2sim.wheel_feedback_adam_physics_20261020 import MeanActor
from debug.sim2sim.wheel_feedback_paths_20261020 import aggregate, csv_metrics
from debug.sim2sim.wheel_feedback_update318_20261020 import OUTPUT as INPUT, KEYS, make_policy, probe_inputs, score, dot
from wheelleg_mujoco.contract import load_policy_contract
from wheelleg_mujoco.evaluation import FORMAL_SCENARIOS
from wheelleg_mujoco.runner import WheelLegMujocoRuntime


OUTPUT = INPUT / 'wheel_smooth_recorded_forward_cpu_intervention'


def main():
    torch.set_num_threads(1)
    assert torch.version.cuda is None
    before = frozen_identity()
    OUTPUT.mkdir(exist_ok=False)
    sys.path.insert(0, str(RUN))
    import causal_gradient_analysis as base

    estimate = read_json(INPUT / 'reward_directions_recorded_forward_cpu/result.json')
    source = read_json(INPUT / 'result.json')
    folder, rollout, terms, _ = base.load_update(318)
    observations = rollout['observations']['policy'].reshape(-1, 25)
    actions = rollout['actions'].reshape(-1, 6)
    old_log_probs = rollout['actions_log_prob'].reshape(-1)
    advantages = rollout['advantages'].reshape(-1)
    components = terms.reshape(-1, 19)
    target = base.REWARDS.index('action_smooth_wheel')
    histories = torch.from_numpy(np.concatenate([np.load(row['path'], allow_pickle=False)['history'][row['indices']] for row in source['probe_identity']]))
    total_removed = None
    rows = []
    for number in range(40):
        path = folder / f'minibatch_{number:02d}.pt'
        assert digest(path) == source['artifacts'][number]['sha256']
        batch = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        policy = make_policy(batch['policy_before'])
        parameters = tuple(policy.actor.parameters())
        metric, _ = score(policy.actor, probe_inputs(policy, histories), source['frequency_hz'])
        metric_gradient = torch.autograd.grad(metric, parameters)
        indices = batch['indices'].long()
        forward = batch['forward']
        data = torch.cat((observations[indices], forward['estimated_velocity'], forward['context_mu']), -1)
        cpu_mean = policy.actor(data).clamp(-20, 20)
        mean = forward['action_mean'] + (cpu_mean - cpu_mean.detach())
        cpu_log = Normal(mean, forward['action_sigma']).log_prob(actions[indices]).sum(-1)
        log_prob = forward['action_log_prob'] + (cpu_log - cpu_log.detach())
        ratio = torch.exp(log_prob - old_log_probs[indices])
        clipped = ratio.clamp(.8, 1.2)
        selected = torch.where(-advantages[indices] * ratio >= -advantages[indices] * clipped, ratio, clipped)
        loss = -(components[indices, target] * selected).mean()
        gradients = torch.autograd.grad(loss, parameters)
        group = batch['parameter_groups'][0]
        beta1, beta2 = group['betas']
        clip_factor = min(1., float(rollout['algorithm']['max_grad_norm']) / (float(batch['global_gradient_norm_before_clipping']) + 1e-6))
        removed = []
        for key, gradient in zip(KEYS, gradients, strict=True):
            state = batch['actor_adam_before'][key]
            actual = batch['actor_gradient_after_clipping'][key]
            step = int(state['step']) + 1
            second = beta2 * state['exp_avg_sq'] + (1 - beta2) * actual.square()
            denominator = (second / (1 - beta2 ** step)).sqrt() + group['eps']
            removed.append(-group['lr'] * (1 - beta1) / (1 - beta1 ** step) / denominator * clip_factor * gradient)
        if total_removed is None:
            total_removed = [torch.zeros_like(value) for value in removed]
        for accumulator, value in zip(total_removed, removed, strict=True):
            accumulator.add_(value.detach())
        projection = dot(metric_gradient, removed)
        expected = estimate['minibatches'][number]['estimated_contributions']['action_smooth_wheel']
        assert abs(projection - expected) < 2e-6
        rows.append({'minibatch': number, 'feedback_metric_projection': projection, 'reverse_vs_forward_derivative_error': abs(projection - expected)})
        del policy, parameters, metric_gradient, gradients, removed, batch, mean, cpu_mean, log_prob, ratio, loss
        gc.collect()
    checkpoint = RUN / 'model_317.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    state = {key: value.clone() for key, value in payload['model_state_dict'].items()}
    for key, value in zip(KEYS, total_removed, strict=True):
        state[key] -= value
    state_path = OUTPUT / 'diagnostic_policy_state.pt'
    torch.save({'model_state_dict': state, 'diagnostic_only': True, 'source_checkpoint_sha256': digest(checkpoint),
                'removed': 'estimated fresh action_smooth_wheel numerator only; recorded Adam history and denominator fixed'}, state_path)
    export = RUN / 'series_stand/update_0318/export'
    contract, metadata, model = load_policy_contract(export / 'policy_manifest.json', export / 'actor.ts', SIM / 'model_manifest.json')
    assert metadata['source_checkpoint_sha256'] == digest(checkpoint)
    policy = make_policy(state)
    baseline_policy = make_policy(payload['model_state_dict'])
    with torch.inference_mode():
        metric_before, _ = score(baseline_policy.actor, probe_inputs(baseline_policy, histories), source['frequency_hz'])
        metric_after, _ = score(policy.actor, probe_inputs(policy, histories), source['frequency_hz'])
    actor = MeanActor(policy)
    runtime = WheelLegMujocoRuntime(SIM / 'models' / model['model_xml']['path'], contract, model)
    runtime.heading.enabled = False
    scenes = []
    for name, command in FORMAL_SCENARIOS:
        run_protocol(runtime, actor, name, [command] * 500, OUTPUT)
        row = csv_metrics(OUTPUT / f'{name}.csv')
        row['name'] = name
        scenes.append(row)
    result = {'source_sha256': digest(__file__), 'strict_reward_attribution_passed': False,
              'method': 'Local diagnostic around actual318 using the CPU Jacobian estimate with recorded GPU forward values. Only the estimated fresh wheel-smooth contribution is removed. Actual Adam history, denominator and all later recorded minibatches stay fixed. This is not a new training trajectory or a completed GPU attribution.',
              'state_path': str(state_path), 'state_sha256': digest(state_path), 'minibatches': rows,
              'feedback_metric_before': float(metric_before), 'feedback_metric_after': float(metric_after),
              'scenes': scenes, 'aggregate': aggregate(scenes)}
    assert all(torch.equal(state[key], value) for key, value in policy.state_dict().items())
    assert frozen_identity() == before
    result['frozen_files_checked'] = 118
    write_json(OUTPUT / 'result.json', result)
    print(json.dumps({'feedback_metric_before': float(metric_before), 'feedback_metric_after': float(metric_after), **aggregate(scenes)}), flush=True)


if __name__ == '__main__':
    main()
