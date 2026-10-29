import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from debug.sim2sim.audit_reward_economics_20261019 import digest, frozen_identity, read_json, write_json


PROJECT = Path(__file__).resolve().parents[2]
LOGS = PROJECT / 'logs/rsl_rl/wheelleg_flat_dreamwaq'
SOURCE = LOGS / '2026-10-17_14-57-16_rtx5070_fudan-v1_baseline-yaw1-headingoff-env6144-fresh1000-20261017'
AUDIT = LOGS / '2026-10-17_17-56-29_rtx5070_fudan-v1_baseline-yaw1-headingoff-env4096-fresh701-20261017/degeneration_audit'
OUTPUT = PROJECT / 'debug/sim2sim/runs/action-mode-engine-20261020'
SIM = PROJECT / 'sim2sim/mujoco'
REPLICAS = 16
STEPS = 500
DT = 0.02
FIELDS = ('leg_acc_0', 'leg_acc_1', 'leg_acc_2', 'leg_acc_3', 'vx', 'yaw_rate', 'height',
          'effort_saturation', 'wheel_common_action', 'wheel_common_mean', 'leg_velocity_rms',
          'tilt', 'yaw', 'wheel_common_velocity')


def locations(update):
    checkpoint = SOURCE / f'model_{update - 1}.pt'
    export = AUDIT / f'checkpoints_6144/model_{update - 1}/export'
    return checkpoint, export, export.parent / 'evaluation/disabled'


def noise_for(scenario_index):
    return np.random.default_rng(41882943 + scenario_index).standard_normal(
        (STEPS, REPLICAS, 6), dtype=np.float32)


def load_actor(update, device):
    checkpoint, export, _ = locations(update)
    from wheelleg_mujoco.contract import load_policy_contract
    contract, metadata, model = load_policy_contract(
        export / 'policy_manifest.json', export / 'actor.ts', SIM / 'model_manifest.json')
    assert Path(metadata['source_checkpoint']).resolve() == checkpoint.resolve()
    assert metadata['source_checkpoint_sha256'] == digest(checkpoint)
    assert metadata['source_completed_iterations'] == update
    assert contract.control_dt_s == DT and np.isinf(contract.action_clip)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    sigma = payload['model_state_dict']['raw_std'].abs().clamp(1e-4, 2.).numpy().copy()
    assert sigma.shape == (6,) and np.isfinite(sigma).all()
    actor = torch.jit.load(str(export / 'actor.ts'), map_location=device).eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    identity = {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                'actor_sha256': digest(export / 'actor.ts'), 'sigma': sigma.tolist()}
    return actor, sigma, contract, model, identity


def describe(values, valid, command):
    assert values.shape == (STEPS, len(valid[0]), len(FIELDS))
    records = []
    for index in range(values.shape[1]):
        mask = valid[:, index].copy()
        mask[:50] = False
        tail = values[mask, index].astype(np.float64)
        row = {'replica': index, 'valid_ticks': int(valid[:, index].sum()), 'tail_ticks': len(tail)}
        if len(tail):
            qdd = tail[:, :4]
            row.update({'leg_rms': float(np.sqrt(np.mean(qdd ** 2))),
                        'leg_rms_per_joint': np.sqrt(np.mean(qdd ** 2, axis=0)).tolist(),
                        'leg_mean_abs': float(np.abs(qdd).mean()),
                        'speed_mae': float(np.abs(tail[:, 4] - command[0]).mean()),
                        'yaw_rate_mae': float(np.abs(tail[:, 5] - command[1]).mean()),
                        'yaw_drift_abs': float(abs(tail[-1, 12] - tail[0, 12])),
                        'height_mae': float(np.abs(tail[:, 6] - command[2]).mean()),
                        'effort_saturation': float(tail[:, 7].mean()),
                        'wheel_common_velocity_rms': float(np.sqrt(np.mean(tail[:, 13] ** 2)))})
        records.append(row)
    mask = valid.copy()
    mask[:50] = False
    tail = values[mask].astype(np.float64)
    result = {'replicas': records, 'valid_ticks': int(valid.sum()), 'tail_ticks': len(tail),
              'valid_fraction': float(valid.mean())}
    if len(tail):
        result.update({'leg_rms': float(np.sqrt(np.mean(tail[:, :4] ** 2))),
                       'speed_mae': float(np.abs(tail[:, 4] - command[0]).mean()),
                       'yaw_rate_mae': float(np.abs(tail[:, 5] - command[1]).mean()),
                       'effort_saturation': float(tail[:, 7].mean())})
    return result


def save_trace(directory, name, values, valid, **extra):
    path = directory / f'{name}.npz'
    assert not path.exists()
    np.savez_compressed(path, values=values, valid=valid, fields=np.asarray(FIELDS), **extra)
    return {'path': str(path), 'sha256': digest(path)}


def run_mujoco(directory):
    from scripts.evaluate_mujoco import _failure_reason
    from wheelleg_mujoco.evaluation import FORMAL_SCENARIOS
    from wheelleg_mujoco.runner import WheelLegMujocoRuntime

    result = {'engine': 'mujoco', 'models': {}, 'policy_updates': 0, 'optimizer_updates': 0}
    for update in (301, 401):
        actor, sigma, contract, model, identity = load_actor(update, 'cpu')
        runtime = WheelLegMujocoRuntime(SIM / 'models' / model['model_xml']['path'], contract, model)
        runtime.heading.enabled = False
        model_result = {'identity': identity, 'mean': {}, 'sampled': {}}
        for scenario_index, (name, command) in enumerate(FORMAL_SCENARIOS):
            source_command = np.asarray(command, dtype=np.float64)
            existing = locations(update)[2] / f'{name}.csv'
            with existing.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == STEPS and all(row['failure_reason'] == '' for row in rows)
            previous_report = read_json(locations(update)[2].parent / 'summary.json')
            saved = next(row for row in previous_report['groups']['disabled']['scenarios'] if row['name'] == name)
            assert saved['csv_sha256'] == digest(existing)
            mean_values = np.zeros((STEPS, 1, len(FIELDS)), dtype=np.float64)
            for tick, row in enumerate(rows):
                mean_values[tick, 0, :4] = [float(row[f'leg_acceleration_{j}_rad_s2']) for j in range(4)]
                mean_values[tick, 0, 4:8] = [float(row[k]) for k in ('vx_mps', 'yaw_rate_rad_s', 'base_height_m', 'effort_saturation_fraction')]
                mean_values[tick, 0, 8:10] = np.mean([float(row[f'raw_action_{j}']) for j in (4, 5)])
                mean_values[tick, 0, 10] = np.sqrt(np.mean([float(row[f'leg_velocity_{j}_rad_s']) ** 2 for j in range(4)]))
                mean_values[tick, 0, 11:13] = [float(row['tilt_rad']), float(row['yaw_rad'])]
            mean_valid = np.ones((STEPS, 1), dtype=bool)
            model_result['mean'][name] = {'metrics': describe(mean_values, mean_valid, command),
                                         'reused_csv': str(existing), 'csv_sha256': digest(existing)}
            noise = noise_for(scenario_index)
            values = np.zeros((STEPS, REPLICAS, len(FIELDS)), dtype=np.float64)
            valid = np.zeros((STEPS, REPLICAS), dtype=bool)
            failures, resets = [], []
            for replica in range(REPLICAS):
                observation = runtime.reset(source_command)
                resets.append({'metrics': dict(runtime.last_reset_metrics), 'observation': observation.tolist(),
                               'joint_velocity': runtime.model_map.joint_velocity_control(runtime.data).tolist()})
                failure = None
                for tick in range(STEPS):
                    observation = runtime.prepare_policy_observation(source_command)
                    velocity_before = runtime.model_map.joint_velocity_control(runtime.data).copy()
                    with torch.inference_mode():
                        mean = actor(torch.from_numpy(observation).view(1, -1)).numpy()[0]
                    action = mean + sigma * noise[tick, replica]
                    step = runtime.step(action, source_command)
                    velocity = runtime.model_map.joint_velocity_control(runtime.data).copy()
                    metrics = step.metrics
                    values[tick, replica] = np.r_[(velocity[:4] - velocity_before[:4]) / DT,
                        metrics['vx_mps'], metrics['yaw_rate_rad_s'], metrics['base_height_m'],
                        metrics['effort_saturation_fraction'], action[4:].mean(), mean[4:].mean(),
                        np.sqrt(np.mean(velocity[:4] ** 2)), metrics['tilt_rad'], metrics['yaw_rad'], velocity[4:].mean()]
                    failure = _failure_reason(runtime, metrics)
                    if failure is not None:
                        break
                    valid[tick, replica] = True
                failures.append({'replica': replica, 'reason': failure, 'last_tick': tick + 1})
            trace = save_trace(directory, f'{update}_{name}_sampled', values, valid, noise=noise)
            model_result['sampled'][name] = {'metrics': describe(values, valid, command), 'trace': trace,
                                           'failures': failures, 'resets': resets}
            print(json.dumps({'engine': 'mujoco', 'update': update, 'scenario': name,
                              'mean': model_result['mean'][name]['metrics']['leg_rms'],
                              'sampled': model_result['sampled'][name]['metrics'].get('leg_rms'),
                              'valid_fraction': float(valid.mean())}), flush=True)
            write_json(directory / 'progress.json', {'update': update, 'scenario': name})
        result['models'][str(update)] = model_result
    return result


def run_isaac(directory, args):
    import tensordict
    from isaaclab.app import AppLauncher
    from debug.sim2sim.root_cause_suite.isaac_bootstrap import prepare_kit_paths, kit_argument_string, root_cause_launcher_class

    paths = prepare_kit_paths(directory / 'kit', run_root=directory)
    launcher = root_cause_launcher_class(AppLauncher)(headless=True, device='cuda:0', kit_args=kit_argument_string(paths))
    app = launcher.app
    from wheelleg_dreamwaq.training.dreamwaq_checkpoint import load_dreamwaq_checkpoint_artifact
    from wheelleg_dreamwaq.schemas.randomization import NOMINAL_EVALUATION_PROFILE_V1
    from wheelleg_dreamwaq.schemas.observation import initialize_frame_major_history, append_frame_major_history, flatten_frame_major_history
    from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv
    from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import WheelLegFlatEnvCfg
    from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.observations import build_observations
    from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.training_profiles import apply_task_profile, disable_command_practice
    from wheelleg_dreamwaq.schemas.heading_control import yaw_from_quaternion
    from wheelleg_mujoco.evaluation import FORMAL_SCENARIOS

    class CapturedEnv(WheelLegFlatEnv):
        def _get_dones(self):
            terminated, truncated = super()._get_dones()
            self.probe_state = {key: value.detach().clone() for key, value in vars(self._state).items()}
            self.probe_terminated = terminated.clone()
            self.probe_truncated = truncated.clone()
            return terminated, truncated

    identities = {}
    for update in (301, 401):
        payload, manifest, metadata = load_dreamwaq_checkpoint_artifact(locations(update)[0])
        assert metadata['completed_iterations'] == update
        identities[str(update)] = {'strict_checkpoint_validation': True, 'metadata_completed_iterations': metadata['completed_iterations']}
        del payload
    cfg = WheelLegFlatEnvCfg()
    apply_task_profile(cfg, 'stop_reverse_raw_split_straight_yaw1_headingoff_v1')
    cfg.seed = 41882943
    cfg.scene.num_envs = REPLICAS * 4
    cfg.randomization = NOMINAL_EVALUATION_PROFILE_V1
    cfg.sim.device = 'cuda:0'
    cfg.log_dir = str(directory / 'isaaclab_logs')
    disable_command_practice(cfg)
    direct = CapturedEnv(cfg)
    assert direct.step_dt == DT and cfg.action_clip is None and cfg.heading_hold.enabled is False
    actors, sigmas = {}, {}
    result = {'engine': 'isaac', 'models': {}, 'policy_updates': 0, 'optimizer_updates': 0,
              'physics_dt': direct.physics_dt, 'control_dt': direct.step_dt,
              'environment_count': direct.num_envs, 'randomization': direct.randomization_audit,
              'evaluation_overrides': ['NominalEvaluationProfileV1', 'fixed commands with disabled command practice'],
              'termination': vars(cfg.termination), 'episode_length': direct.max_episode_length}
    for update in (301, 401):
        actors[update], sigma, _, _, identity = load_actor(update, 'cuda:0')
        sigmas[update] = torch.from_numpy(sigma).cuda()
        result['models'][str(update)] = {'identity': {**identity, **identities[str(update)]}, 'mean': {}, 'sampled': {}}
    groups = [(301, 'mean'), (301, 'sampled'), (401, 'mean'), (401, 'sampled')]
    parameter_before = {update: {key: value.cpu().clone() for key, value in actor.state_dict().items()} for update, actor in actors.items()}
    for scenario_index, (name, command) in enumerate(FORMAL_SCENARIOS):
        direct.reset(seed=41882943)
        command_tensor = torch.tensor(command, device=direct.device).expand(direct.num_envs, -1).clone()
        direct.set_source_command(command_tensor)
        current, critic = build_observations(direct._current_state(), q_reference=direct._q_reference,
                                             normalization=cfg.normalization, actor_noise=None)
        history = initialize_frame_major_history(current)
        initial_state = {key: value.cpu().numpy() for key, value in vars(direct._current_state()).items()}
        initial_state['root_link_pos_w'] -= direct.scene.env_origins.cpu().numpy()
        initial_state['root_com_pos_w'] -= direct.scene.env_origins.cpu().numpy()
        noise_array = noise_for(scenario_index)
        noise = torch.from_numpy(noise_array).cuda()
        values = np.zeros((STEPS, direct.num_envs, len(FIELDS)), dtype=np.float64)
        valid = np.zeros((STEPS, direct.num_envs), dtype=bool)
        active = torch.ones(direct.num_envs, dtype=torch.bool, device=direct.device)
        first_failure = [None] * direct.num_envs
        action_records = np.zeros((STEPS, direct.num_envs, 6), dtype=np.float32)
        mean_records = np.zeros_like(action_records)
        initial_observation = flatten_frame_major_history(history).cpu().numpy()
        for tick in range(STEPS):
            observation = flatten_frame_major_history(history)
            means, actions = [], []
            with torch.inference_mode():
                for group_index, (update, mode) in enumerate(groups):
                    selection = slice(group_index * REPLICAS, (group_index + 1) * REPLICAS)
                    mean = actors[update](observation[selection])
                    means.append(mean)
                    actions.append(mean if mode == 'mean' else mean + sigmas[update] * noise[tick])
                mean = torch.cat(means)
                action = torch.cat(actions)
            previous_velocity = direct._current_state().joint_velocity.clone()
            observations, reward, terminated, truncated, extras = direct.step(action)
            state = direct.probe_state
            velocity = state['joint_velocity']
            acceleration = (velocity[:, :4] - previous_velocity[:, :4]) / DT
            tilt = torch.acos((-state['projected_gravity'][:, 2]).clamp(-1, 1))
            yaw = yaw_from_quaternion(state['root_link_quat_w'])
            effort = torch.isclose(state['applied_torque'].abs(), direct._controlled_effort_limits).float().mean(-1)
            row = torch.cat((acceleration, state['root_com_linear_velocity'][:, :1],
                             state['root_angular_velocity'][:, 2:3], state['base_height'], effort[:, None],
                             action[:, 4:].mean(-1, keepdim=True), mean[:, 4:].mean(-1, keepdim=True),
                             velocity[:, :4].square().mean(-1, keepdim=True).sqrt(), tilt[:, None], yaw[:, None],
                             velocity[:, 4:].mean(-1, keepdim=True)), dim=-1)
            finite = torch.isfinite(row).all(-1)
            assert finite.all()
            failed = direct.probe_terminated
            for index in torch.where(active & failed)[0].tolist():
                first_failure[index] = tick + 1
            active &= ~failed
            valid[tick] = active.cpu().numpy()
            values[tick] = row.cpu().numpy()
            action_records[tick] = action.cpu().numpy()
            mean_records[tick] = mean.cpu().numpy()
            current = observations['policy']
            history = append_frame_major_history(history, current, terminated | truncated)
            # 正式timeout会自动reset；该行保留终止前状态，后续行停止计入同一次初态评估。
            active &= ~direct.probe_truncated
            direct.set_source_command(command_tensor)
        for group_index, (update, mode) in enumerate(groups):
            selection = slice(group_index * REPLICAS, (group_index + 1) * REPLICAS)
            trace = save_trace(directory, f'{update}_{name}_{mode}', values[:, selection], valid[:, selection],
                               actions=action_records[:, selection], means=mean_records[:, selection], noise=noise_array,
                               initial_observation=initial_observation[selection],
                               **{f'initial_{key}': value[selection] for key, value in initial_state.items()})
            result['models'][str(update)][mode][name] = {'metrics': describe(values[:, selection], valid[:, selection], command),
                                                       'first_terminated_tick': first_failure[selection], 'trace': trace}
        print(json.dumps({'engine': 'isaac', 'scenario': name, 'groups': [
            {'update': update, 'mode': mode, **{k: v for k, v in result['models'][str(update)][mode][name]['metrics'].items() if k != 'replicas'}}
            for update, mode in groups]}), flush=True)
        write_json(directory / 'progress.json', {'scenario': name})
    for update, actor in actors.items():
        assert all(torch.equal(value.cpu(), parameter_before[update][key]) for key, value in actor.state_dict().items())
    result['parameters_unchanged'] = True
    write_json(directory / 'result.json', result)
    app.close(skip_cleanup=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=['mujoco', 'isaac'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    before = frozen_identity()
    directory = OUTPUT / args.engine
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'start.json', {'engine': args.engine, 'source_sha256': digest(__file__), 'frozen_files': before,
               'noise_seed': 41882943, 'replicas': REPLICAS, 'steps': STEPS, 'fields': FIELDS,
               'sampling': 'mean + checkpoint_sigma * fixed_numpy_standard_normal; shared draws across engines and models'})
    if args.engine == 'mujoco':
        assert torch.version.cuda is None
        result = run_mujoco(directory)
    else:
        assert torch.cuda.is_available()
        os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
        result = run_isaac(directory, args)
    assert before == frozen_identity()
    result['frozen_files_checked'] = len(before)
    result['source_sha256'] = digest(__file__)
    write_json(directory / 'result.json', result)
    print(f'COMPLETED {directory}', flush=True)


if __name__ == '__main__':
    main()
