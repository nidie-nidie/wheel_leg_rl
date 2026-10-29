import json

import numpy as np
import torch

from debug.sim2sim.action_mode_engine_probe_20261020 import SIM
from debug.sim2sim.audit_reward_economics_20261019 import DEFAULT_RUN as RUN, digest, frozen_identity, read_json, write_json
from debug.sim2sim.evaluate_baseline_mujoco import run_protocol
from debug.sim2sim.wheel_feedback_paths_20261020 import aggregate, csv_metrics
from debug.sim2sim.wheel_feedback_update318_20261020 import OUTPUT as INPUT, make_policy
from wheelleg_mujoco.contract import load_policy_contract
from wheelleg_mujoco.evaluation import FORMAL_SCENARIOS
from wheelleg_mujoco.runner import WheelLegMujocoRuntime


OUTPUT = INPUT / 'physical_counterfactuals'


class MeanActor:
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, history):
        encoded = self.policy.cenet.encode(history)
        return self.policy._effective_mean(history[:, -25:], encoded)


def main():
    torch.set_num_threads(1)
    assert torch.version.cuda is None
    before = frozen_identity()
    OUTPUT.mkdir(exist_ok=False)
    source = read_json(INPUT / 'result.json')
    export = RUN / 'series_stand/update_0318/export'
    evaluation = export.parent / 'evaluation'
    contract, metadata, model = load_policy_contract(export / 'policy_manifest.json', export / 'actor.ts', SIM / 'model_manifest.json')
    checkpoint = RUN / 'model_317.pt'
    assert metadata['source_checkpoint_sha256'] == digest(checkpoint)
    assert metadata['source_completed_iterations'] == 318
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    original = MeanActor(make_policy(payload['model_state_dict']))
    scripted = torch.jit.load(str(export / 'actor.ts')).eval()
    histories = np.concatenate([np.load(row['path'], allow_pickle=False)['history'][row['indices']] for row in source['probe_identity']])
    with torch.inference_mode():
        tensor = torch.from_numpy(histories)
        golden_error = float((original(tensor) - scripted(tensor)).abs().max())
    assert golden_error <= 1e-7, golden_error
    old = read_json(evaluation / 'summary.json')
    assert old['source_checkpoint_sha256'] == digest(checkpoint)
    result = {'diagnostic_only': True, 'source_sha256': digest(__file__), 'same_batch_python_torchscript_error': golden_error,
              'method': 'Subtract the sum of one recorded Adam numerator component from the actual318 endpoint. Later batches, second moments and the encoder follow the recorded trajectory. This local physical intervention does not reproduce a new optimizer trajectory.',
              'baseline': {}, 'cases': {}}
    scenes = []
    for name, _ in FORMAL_SCENARIOS:
        path = evaluation / 'disabled' / f'{name}.csv'
        expected = next(row['csv_sha256'] for row in old['groups']['disabled']['scenarios'] if row['name'] == name)
        assert digest(path) == expected
        row = csv_metrics(path)
        row['name'] = name
        scenes.append(row)
    result['baseline'] = {'scenes': scenes, 'aggregate': aggregate(scenes)}
    for label in ('without_carried_first_moment', 'without_current_gradient_numerator'):
        identity = source[label]
        assert digest(identity['path']) == identity['sha256']
        candidate = torch.load(identity['path'], map_location='cpu', weights_only=False)
        assert candidate['diagnostic_only'] and candidate['source_checkpoint_sha256'] == digest(checkpoint)
        policy = make_policy(candidate['model_state_dict'])
        parameters = {key: value.clone() for key, value in policy.state_dict().items()}
        actor = MeanActor(policy)
        folder = OUTPUT / label
        folder.mkdir()
        runtime = WheelLegMujocoRuntime(SIM / 'models' / model['model_xml']['path'], contract, model)
        runtime.heading.enabled = False
        scenes = []
        for name, command in FORMAL_SCENARIOS:
            run_protocol(runtime, actor, name, [command] * 500, folder)
            row = csv_metrics(folder / f'{name}.csv')
            row['name'] = name
            scenes.append(row)
        assert all(torch.equal(parameters[key], value) for key, value in policy.state_dict().items())
        result['cases'][label] = {'identity': identity, 'scenes': scenes, 'aggregate': aggregate(scenes)}
        write_json(OUTPUT / 'progress.json', result)
        print(json.dumps({'case': label, **aggregate(scenes)}), flush=True)
    assert frozen_identity() == before
    result['frozen_files_checked'] = 118
    write_json(OUTPUT / 'result.json', result)


if __name__ == '__main__':
    main()
