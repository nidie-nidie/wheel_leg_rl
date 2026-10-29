import gc
import json
import sys

import torch
from torch.distributions import Normal

from debug.sim2sim.action_mode_engine_probe_20261020 import SIM
from debug.sim2sim.audit_reward_economics_20261019 import DEFAULT_RUN as RUN, digest, frozen_identity, read_json, write_json
from debug.sim2sim.evaluate_baseline_mujoco import run_protocol
from debug.sim2sim.wheel_feedback_adam_physics_20261020 import MeanActor
from debug.sim2sim.wheel_feedback_paths_20261020 import aggregate, csv_metrics
from debug.sim2sim.wheel_feedback_reward_origin_20261020 import OUTPUT as SOURCE, expanded_terms, load_histories
from debug.sim2sim.wheel_feedback_update318_20261020 import OUTPUT as PREVIOUS, KEYS, make_policy, probe_inputs, score, dot
from wheelleg_mujoco.contract import load_policy_contract
from wheelleg_mujoco.evaluation import FORMAL_SCENARIOS
from wheelleg_mujoco.runner import WheelLegMujocoRuntime


OUTPUT = SOURCE / 'interventions_318'
CASES = {'wheel_smooth': (16,), 'yaw_enhance': (3,), 'yaw_pair': (2, 3),
         'wheel_smooth_and_yaw_pair': (16, 2, 3), 'value_combined': (18, 19, 20, 21, 22)}


def main():
    torch.set_num_threads(1)
    assert torch.version.cuda is None
    frozen = frozen_identity()
    OUTPUT.mkdir(exist_ok=False)
    attribution = read_json(SOURCE / 'update_0318.json')
    reference = read_json(PREVIOUS / 'result.json')
    histories = load_histories(reference)
    sys.path.insert(0, str(RUN))
    import causal_gradient_analysis as base

    folder, rollout, original, weighted = base.load_update(318)
    components, checks = expanded_terms(rollout, weighted, original)
    components = components.flatten(0, 1)
    observations = rollout['observations']['policy'].flatten(0, 1)
    actions = rollout['actions'].flatten(0, 1)
    old_probability = rollout['actions_log_prob'].flatten()
    advantages = rollout['advantages'].flatten()
    moments = {label: [torch.zeros_like(value) for value in make_policy(torch.load(folder / 'minibatch_00.pt', map_location='cpu', weights_only=False, mmap=True)['policy_before']).actor.parameters()]
               for label in CASES}
    removed = {label: [torch.zeros_like(value) for value in moments[label]] for label in CASES}
    accumulated_projection = dict.fromkeys(CASES, 0.)
    for number in range(40):
        batch = torch.load(folder / f'minibatch_{number:02d}.pt', map_location='cpu', weights_only=False, mmap=True)
        policy = make_policy(batch['policy_before'])
        parameters = tuple(policy.actor.parameters())
        metric, _ = score(policy.actor, probe_inputs(policy, histories), reference['frequency_hz'])
        metric_gradient = torch.autograd.grad(metric, parameters)
        indices, forward = batch['indices'].long(), batch['forward']
        data = torch.cat((observations[indices], forward['estimated_velocity'], forward['context_mu']), -1)
        cpu_mean = policy.actor(data).clamp(-20, 20)
        mean = forward['action_mean'] + (cpu_mean - cpu_mean.detach())
        cpu_log = Normal(mean, forward['action_sigma'], validate_args=False).log_prob(actions[indices]).sum(-1)
        log_probability = forward['action_log_prob'] + (cpu_log - cpu_log.detach())
        ratio = torch.exp(log_probability - old_probability[indices])
        clipped = ratio.clamp(.8, 1.2)
        selected = torch.where(-advantages[indices] * ratio >= -advantages[indices] * clipped, ratio, clipped)
        losses = -(components[indices] * selected[:, None]).mean(0)
        clip = min(1., float(rollout['algorithm']['max_grad_norm']) / (float(batch['global_gradient_norm_before_clipping']) + 1e-6))
        group = batch['parameter_groups'][0]
        beta1, beta2 = group['betas']
        for label, selection in CASES.items():
            gradients = torch.autograd.grad(losses[list(selection)].sum(), parameters, retain_graph=True)
            for part, (key, gradient) in enumerate(zip(KEYS, gradients, strict=True)):
                state = batch['actor_adam_before'][key]
                step = int(state['step']) + 1
                actual_gradient = batch['actor_gradient_after_clipping'][key]
                second = beta2 * state['exp_avg_sq'] + (1 - beta2) * actual_gradient.square()
                denominator = (second / (1 - beta2 ** step)).sqrt() + group['eps']
                moments[label][part] = beta1 * moments[label][part] + (1 - beta1) * clip * gradient.detach()
                delta = -group['lr'] / (1 - beta1 ** step) / denominator * moments[label][part]
                removed[label][part] += delta
                accumulated_projection[label] += float((metric_gradient[part].double() * delta.double()).sum())
        del policy, parameters, metric_gradient, gradients, batch, losses, selected, ratio, log_probability, mean, cpu_mean
        gc.collect()
    names = tuple(base.REWARDS[:18]) + ('value_current_baseline', 'value_internal_future', 'value_horizon_inferred',
                                     'value_timeout_compensation', 'gae_arithmetic_residual')
    projection_checks = {}
    for label, selection in CASES.items():
        expected = sum(attribution['contributions']['with_within_update_momentum'][names[index]] for index in selection)
        error = abs(accumulated_projection[label] - expected)
        assert error < 2e-6
        projection_checks[label] = {'reverse_vs_adjoint_error': error, 'projection': accumulated_projection[label]}
    checkpoint = RUN / 'model_317.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    export = RUN / 'series_stand/update_0318/export'
    contract, metadata, model = load_policy_contract(export / 'policy_manifest.json', export / 'actor.ts', SIM / 'model_manifest.json')
    assert metadata['source_checkpoint_sha256'] == digest(checkpoint)
    original_policy = make_policy(payload['model_state_dict'])
    with torch.inference_mode():
        original_metric, _ = score(original_policy.actor, probe_inputs(original_policy, histories), reference['frequency_hz'])
        same_batch_error = float((MeanActor(original_policy)(histories) - torch.jit.load(str(export / 'actor.ts')).eval()(histories)).abs().max())
    assert same_batch_error <= 1e-7
    baseline = read_json(PREVIOUS / 'physical_counterfactuals/result.json')['baseline']
    for row in baseline['scenes']:
        assert digest(row['csv']) == row['csv_sha256']
    result = {'source_sha256': digest(__file__), 'strict_gpu_reward_attribution_passed': False,
              'method': 'CPU Jacobian diagnostic: remove each selected reward gradient and its propagated first-moment contribution generated within real update318. Incoming pre318 first moment, original second moments, global clipping factors, original subsequent batches and encoder trajectory remain fixed. These are endpoint parameter interventions, not a fresh training trajectory or an acceptance candidate.',
              'source_checkpoint': str(checkpoint), 'source_checkpoint_sha256': digest(checkpoint),
              'same_batch_python_torchscript_error': same_batch_error, 'baseline': baseline,
              'projection_checks': projection_checks, 'gae_checks': checks, 'cases': {}}
    for label in CASES:
        folder = OUTPUT / label
        folder.mkdir()
        state = {key: value.clone() for key, value in payload['model_state_dict'].items()}
        for key, delta in zip(KEYS, removed[label], strict=True):
            state[key] -= delta
        path = folder / 'diagnostic_policy_state.pt'
        torch.save({'model_state_dict': state, 'diagnostic_only': True, 'removed': label,
                    'source_checkpoint_sha256': digest(checkpoint)}, path)
        policy = make_policy(state)
        with torch.inference_mode():
            metric_after, _ = score(policy.actor, probe_inputs(policy, histories), reference['frequency_hz'])
        runtime = WheelLegMujocoRuntime(SIM / 'models' / model['model_xml']['path'], contract, model)
        runtime.heading.enabled = False
        actor = MeanActor(policy)
        scenes = []
        for name, command in FORMAL_SCENARIOS:
            run_protocol(runtime, actor, name, [command] * 500, folder)
            row = csv_metrics(folder / f'{name}.csv')
            row['name'] = name
            scenes.append(row)
        assert all(torch.equal(state[key], value) for key, value in policy.state_dict().items())
        result['cases'][label] = {'state_path': str(path), 'state_sha256': digest(path),
                                 'feedback_metric_before': float(original_metric), 'feedback_metric_after': float(metric_after),
                                 'scenes': scenes, 'aggregate': aggregate(scenes)}
        write_json(OUTPUT / 'progress.json', result)
        print(json.dumps({'case': label, 'feedback_metric_after': float(metric_after), **aggregate(scenes)}), flush=True)
    assert frozen_identity() == frozen
    result['frozen_files_checked'] = 118
    write_json(OUTPUT / 'result.json', result)


if __name__ == '__main__':
    main()
