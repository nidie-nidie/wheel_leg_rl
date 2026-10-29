import argparse
import json
import pathlib
import sys

import torch
from torch.distributions import Normal


def raw_gae(weighted, dones, gamma, lam):
    terms = torch.zeros_like(weighted)
    follow = torch.zeros_like(weighted[0])
    for step in reversed(range(weighted.shape[0])):
        follow = weighted[step] + gamma * lam * (1.0 - dones[step].float()).unsqueeze(-1) * follow
        terms[step] = follow
    return terms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--scales', nargs='+', type=float, default=[1.0, 10.0, 30.0, 100.0])
    args = parser.parse_args()
    run = pathlib.Path(args.run).resolve()
    sys.path.insert(0, str(run))
    import causal_gradient_analysis as base

    torch.set_num_threads(1)
    names = list(base.REWARDS[:18])
    target_index = names.index('dof_acc')
    folder, rollout, _, weighted = base.load_update(318)
    raw_terms = raw_gae(weighted, rollout['dones'].bool().squeeze(-1),
                        float(rollout['algorithm']['gamma']), float(rollout['algorithm']['lam']))
    raw_advantage = rollout['raw_advantages'].float()
    bootstrap = raw_advantage - raw_terms.sum(dim=-1, keepdim=True)
    original_normalized = (raw_advantage - raw_advantage.mean()) / raw_advantage.std()
    normalization_error = float((original_normalized - rollout['advantages']).abs().max())
    obs = rollout['observations']['policy'].reshape(-1, 25)
    actions = rollout['actions'].reshape(-1, 6)
    old_log_probs = rollout['actions_log_prob'].reshape(-1)
    out_dir = run / 'dof_acc_weight_scan_update318'
    out_dir.mkdir(exist_ok=True)
    results = []

    for scale in args.scales:
        scaled_raw = raw_terms.sum(dim=-1, keepdim=True) + bootstrap
        scaled_raw = scaled_raw + (scale - 1.0) * raw_terms[..., target_index:target_index + 1]
        advantage = (scaled_raw - scaled_raw.mean()) / scaled_raw.std()
        difference_by_parameter = None
        rows = []
        for number in range(40):
            batch = base.load(folder / f'minibatch_{number:02d}.pt')
            actor = base.make_actor(batch['policy_before'])
            parameters = list(actor.parameters())
            if difference_by_parameter is None:
                difference_by_parameter = [torch.zeros_like(parameter) for parameter in parameters]
            indices = batch['indices'].long()
            forward = batch['forward']
            actor_input = torch.cat((obs[indices].to(base.DEVICE),
                                     forward['estimated_velocity'].to(base.DEVICE),
                                     forward['context_mu'].to(base.DEVICE)), dim=-1)
            mean = actor(actor_input).clamp(-20.0, 20.0)
            sigma = (batch['policy_before']['raw_std'].to(base.DEVICE).abs()
                     .clamp(1.0e-4, 2.0).expand_as(mean))
            new_log_prob = Normal(mean, sigma).log_prob(actions[indices].to(base.DEVICE)).sum(dim=-1)
            ratio = torch.exp(new_log_prob - old_log_probs[indices].to(base.DEVICE))
            local_advantage = advantage.reshape(-1).to(base.DEVICE)[indices]
            unclipped = -local_advantage * ratio
            clipped = -local_advantage * ratio.clamp(0.8, 1.2)
            selected_ratio = torch.where(unclipped >= clipped, ratio, ratio.clamp(0.8, 1.2))
            loss = -(local_advantage * selected_ratio).mean()
            gradients = list(torch.autograd.grad(loss, parameters))
            actor_gradient_norm = float(base.flat(gradients).norm())
            recorded_actor = [batch['actor_gradient_before_clipping'][key].to(base.DEVICE) for key in base.ACTOR_KEYS]
            recorded_actor_norm_sq = float(base.flat(recorded_actor).square().sum())
            recorded_global_norm = float(batch['global_gradient_norm_before_clipping'].item())
            unchanged_gradient_norm_sq = max(recorded_global_norm ** 2 - recorded_actor_norm_sq, 0.0)
            total_gradient_norm = (unchanged_gradient_norm_sq + actor_gradient_norm ** 2) ** 0.5
            clip_factor = min(1.0, float(rollout['algorithm']['max_grad_norm']) / max(total_gradient_norm, 1.0e-12))
            scaled_delta = base.adam_delta(gradients, batch['actor_adam_before'], batch['parameter_groups'], clip_factor)
            actual_delta = [(batch['policy_before'][key] - batch['actor_parameters_after'][key]).to(base.DEVICE)
                            for key in base.ACTOR_KEYS]
            if scale == 1.0:
                difference = [torch.zeros_like(value) for value in actual_delta]
            else:
                difference = [scaled_delta[index] - actual_delta[index] for index in range(len(actual_delta))]
            for index, value in enumerate(difference):
                difference_by_parameter[index].add_(value.detach())
            rows.append({'minibatch': number, 'actor_gradient_norm': actor_gradient_norm,
                         'total_gradient_norm': total_gradient_norm, 'clip_factor': clip_factor,
                         'max_delta_difference': max(float(value.abs().max()) for value in difference)})
            del actor, batch, actor_input, mean, new_log_prob, ratio, loss, gradients

        checkpoint = base.load(run / 'model_317.pt')
        for key, difference in zip(base.ACTOR_KEYS, difference_by_parameter, strict=True):
            checkpoint['model_state_dict'][key] = checkpoint['model_state_dict'][key] + difference.detach().cpu()
        label = f'{scale:g}'.replace('.', 'p')
        checkpoint_path = out_dir / f'model_317_dof_acc_x{label}.pt'
        torch.save(checkpoint, checkpoint_path)
        results.append({'scale': scale, 'raw_advantage_mean': float(scaled_raw.mean()),
                        'raw_advantage_std': float(scaled_raw.std()),
                        'min_clip_factor': min(row['clip_factor'] for row in rows),
                        'max_total_gradient_norm': max(row['total_gradient_norm'] for row in rows),
                        'mean_total_gradient_norm': sum(row['total_gradient_norm'] for row in rows) / len(rows),
                        'sum_delta_difference_norm': sum(float(value.square().sum().sqrt()) for value in difference_by_parameter),
                        'checkpoint': str(checkpoint_path), 'minibatches': rows})

    report = {'schema_version': 'DofAccWeightScanUpdate318V2', 'update': 318,
              'target': 'dof_acc', 'scales': args.scales,
              'method': 'At each real minibatch policy_before and recorded Adam state, only the dof_acc GAE contribution is rescaled. The recorded actual update is used as the baseline and the counterfactual delta difference is accumulated around model_317. Non-actor gradient norm remains fixed. This is a local actor-update counterfactual, not a second training run.',
              'original_advantage_reconstruction_max_abs_error': normalization_error,
              'results': results}
    (out_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'schema_version': report['schema_version'], 'update': 318,
                      'normalization_error': normalization_error,
                      'results': [{k: row[k] for k in ('scale', 'raw_advantage_std', 'min_clip_factor', 'max_total_gradient_norm', 'mean_total_gradient_norm', 'sum_delta_difference_norm', 'checkpoint')} for row in results]}, indent=2))


if __name__ == '__main__':
    main()

