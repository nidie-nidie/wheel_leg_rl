import json

import numpy as np
import torch

from debug.sim2sim import wheel_feedback_paths_20261020 as paths
from debug.sim2sim.action_mode_engine_probe_20261020 import PROJECT, SIM, load_actor
from debug.sim2sim.audit_reward_economics_20261019 import frozen_identity, digest, read_json, write_json
from debug.sim2sim.evaluate_baseline_mujoco import run_protocol
from wheelleg_mujoco.evaluation import FORMAL_SCENARIOS
from wheelleg_mujoco.runner import WheelLegMujocoRuntime


OUTPUT = paths.OUTPUT / 'slow_history'


class SlowHistoryActor(paths.PathActor):
    def __call__(self, history):
        changed = history.reshape(-1, 5, 25).clone()
        common = changed[:, :, 23:25].mean(-1, keepdim=True)
        # 保留五帧共同分量的均值，仅移除帧间快速变化；轮差速和其他输入保持原值。
        changed[:, :, 23:25] += common.mean(1, keepdim=True) - common
        raw = self.actor(history)
        neutral = self.actor(changed.reshape(-1, 125))
        action = paths.path_action(raw, neutral, 'wheel_common', self.alpha)
        self.histories.append(history.numpy().copy())
        self.raw_actions.append(raw.numpy().copy())
        self.neutral_actions.append(neutral.numpy().copy())
        self.actions.append(action.numpy().copy())
        return action


def main():
    torch.set_num_threads(1)
    assert torch.version.cuda is None
    before = frozen_identity()
    OUTPUT.mkdir(exist_ok=False)
    actor, _, contract, model, identity = load_actor(401, 'cpu')
    parameter_before = {key: value.clone() for key, value in actor.state_dict().items()}
    result = {'identity': identity, 'source_sha256': digest(__file__), 'policy_updates': 0, 'cases': {}, 'old701': {}}
    old = PROJECT / 'artifacts/phase2_dreamwaq/baseline-heading-implementation-20261016/old701-mujoco-final'
    old_summary = read_json(old / 'summary.json')
    assert old_summary['source_checkpoint_sha256'] == 'A351C42E999F873FDA0921940BA9F2C57FB2143485971DF50A8757B067507FB9'
    old_rows = []
    for name, _ in FORMAL_SCENARIOS:
        path = old / 'disabled' / f'{name}.csv'
        expected = next(row for row in old_summary['groups']['disabled']['scenarios'] if row['name'] == name)
        assert digest(path) == expected['csv_sha256']
        old_rows.append(paths.csv_metrics(path))
    result['old701'] = {'scenes': old_rows, 'aggregate': paths.aggregate(old_rows)}
    for alpha in (0., .5):
        label = f'alpha_{alpha:g}'.replace('.', 'p')
        folder = OUTPUT / label
        folder.mkdir()
        runtime = WheelLegMujocoRuntime(SIM / 'models' / model['model_xml']['path'], contract, model)
        runtime.heading.enabled = False
        scenes = []
        for name, command in FORMAL_SCENARIOS:
            probe = SlowHistoryActor(actor, 'wheel_common', alpha)
            run_protocol(runtime, probe, name, [command] * 500, folder)
            arrays = {'history': np.concatenate(probe.histories), 'raw': np.concatenate(probe.raw_actions),
                      'neutral': np.concatenate(probe.neutral_actions), 'action': np.concatenate(probe.actions)}
            path = folder / f'{name}_policy.npz'
            np.savez_compressed(path, **arrays)
            row = paths.csv_metrics(folder / f'{name}.csv')
            row.update(name=name, trace=str(path), trace_sha256=digest(path))
            scenes.append(row)
        transitions = []
        protocols = {
            'forward_stop_reverse': [(1., 0., .2)] * 100 + [(0., 0., .2)] * 100 + [(-1., 0., .2)] * 100 + [(0., 0., .2)] * 100 + [(1., 0., .2)] * 100,
            'turn_release': [(0., .6, .2)] * 100 + [(1., 0., .2)] * 150 + [(0., -.6, .2)] * 100 + [(1., 0., .2)] * 150,
            'reset_toggle': [(0., 0., .2)] * 500}
        for name, commands in protocols.items():
            events = {100: 'heading_off', 200: 'heading_on', 300: 'reset'} if name == 'reset_toggle' else {}
            probe = SlowHistoryActor(actor, 'wheel_common', alpha)
            summary = run_protocol(runtime, probe, name, commands, folder, events)
            transitions.append({'name': name, 'summary': summary, 'metrics': paths.csv_metrics(folder / f'{name}.csv')})
        result['cases'][label] = {'alpha': alpha, 'scenes': scenes, 'transitions': transitions, 'aggregate': paths.aggregate(scenes)}
        write_json(OUTPUT / 'progress.json', result)
        print(json.dumps({'case': label, **paths.aggregate(scenes), 'transitions_complete': [row['summary']['completed'] for row in transitions]}), flush=True)
    assert frozen_identity() == before
    assert all(torch.equal(parameter_before[key], value) for key, value in actor.state_dict().items())
    result['frozen_files_checked'] = 118
    result['parameters_unchanged'] = True
    write_json(OUTPUT / 'result.json', result)


if __name__ == '__main__':
    main()
