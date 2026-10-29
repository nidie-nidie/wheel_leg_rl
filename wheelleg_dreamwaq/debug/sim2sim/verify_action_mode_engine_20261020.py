import csv
import json
from pathlib import Path

import numpy as np
import torch

from debug.sim2sim.action_mode_engine_probe_20261020 import OUTPUT, PROJECT, FIELDS, REPLICAS, STEPS, DT, locations, noise_for, describe
from debug.sim2sim.audit_reward_economics_20261019 import frozen_identity, digest, read_json, write_json
from wheelleg_mujoco.evaluation import FORMAL_SCENARIOS


def verify():
    frozen = frozen_identity()
    source = Path(__file__).with_name('action_mode_engine_probe_20261020.py')
    results = {engine: read_json(OUTPUT / engine / 'result.json') for engine in ('isaac', 'mujoco')}
    verified, initial_difference, rows = [], [], []
    for engine, result in results.items():
        start = read_json(OUTPUT / engine / 'start.json')
        assert start['frozen_files'] == frozen
        assert start['source_sha256'] == digest(source)
        if engine == 'mujoco':
            assert result['source_sha256'] == digest(source)
            assert result['frozen_files_checked'] == 118
        assert result['policy_updates'] == result['optimizer_updates'] == 0
        if engine == 'isaac':
            assert result['parameters_unchanged'] and result['control_dt'] == DT and result['physics_dt'] == 0.005
        for update in (301, 401):
            model = result['models'][str(update)]
            checkpoint, export, _ = locations(update)
            assert model['identity']['checkpoint_sha256'] == digest(checkpoint)
            assert model['identity']['actor_sha256'] == digest(export / 'actor.ts')
            sigma = np.asarray(model['identity']['sigma'])
            for mode in ('mean', 'sampled'):
                scene_rows = []
                for scene_index, (name, command) in enumerate(FORMAL_SCENARIOS):
                    scene = model[mode][name]
                    if 'trace' in scene:
                        assert digest(scene['trace']['path']) == scene['trace']['sha256']
                        archive = np.load(scene['trace']['path'], allow_pickle=False)
                        values, valid = archive['values'], archive['valid']
                        assert values.shape == (STEPS, REPLICAS, len(FIELDS))
                        assert np.isfinite(values).all() and valid.dtype == bool
                        assert np.array_equal(archive['noise'], noise_for(scene_index))
                        recomputed = describe(values, valid, command)
                        for key in ('leg_rms', 'speed_mae', 'yaw_rate_mae', 'effort_saturation', 'valid_fraction'):
                            assert abs(recomputed[key] - scene['metrics'][key]) < 1e-10
                        injected = (archive['noise'][:, :, 4:] * sigma[4:]).mean(-1) if mode == 'sampled' else np.zeros(valid.shape)
                        error = float(np.max(np.abs((values[:, :, 8] - values[:, :, 9])[valid] - injected[valid])))
                        assert error < 3e-6, error
                        if engine == 'isaac':
                            assert np.max(np.abs(archive['initial_observation'][:, -6:])) == 0
                            assert np.max(np.abs(archive['initial_joint_velocity'])) == 0
                            for key in ('actions', 'means'):
                                assert archive[key].shape == (STEPS, REPLICAS, 6)
                            expected = archive['means'] + archive['noise'] * sigma.astype(np.float32) if mode == 'sampled' else archive['means']
                            assert np.array_equal(archive['actions'], expected)
                        common_mask = valid.copy()
                        common_mask[:50] = False
                        common_mask[499:] = False
                        common_values = values[common_mask]
                        common_rms = float(np.sqrt(np.mean(common_values[:, :4] ** 2)))
                        verified.append({'engine': engine, 'update': update, 'mode': mode, 'scene': name,
                                         'trace_sha256': scene['trace']['sha256'], 'wheel_sampling_max_abs_error': error})
                    else:
                        assert engine == 'mujoco' and mode == 'mean'
                        assert digest(scene['reused_csv']) == scene['csv_sha256']
                        with Path(scene['reused_csv']).open(encoding='utf-8-sig', newline='') as stream:
                            csv_rows = list(csv.DictReader(stream))
                        acceleration = np.array([[float(row[f'leg_acceleration_{j}_rad_s2']) for j in range(4)] for row in csv_rows[50:]])
                        assert abs(np.sqrt(np.mean(acceleration ** 2)) - scene['metrics']['leg_rms']) < 1e-10
                        common_rms = float(np.sqrt(np.mean(acceleration[:-1] ** 2)))
                        verified.append({'engine': engine, 'update': update, 'mode': mode, 'scene': name, 'csv_sha256': scene['csv_sha256']})
                    scene_rows.append({**scene['metrics'], 'common_horizon_leg_rms': common_rms})
                aggregate = {'engine': engine, 'update': update, 'mode': mode,
                             'leg_rms': float(np.sqrt(np.mean([row['leg_rms'] ** 2 for row in scene_rows]))),
                             'speed_mae': float(np.mean([row['speed_mae'] for row in scene_rows])),
                             'yaw_rate_mae': float(np.mean([row['yaw_rate_mae'] for row in scene_rows])),
                             'effort_saturation': float(np.mean([row['effort_saturation'] for row in scene_rows])),
                             'valid_fraction': float(np.mean([row['valid_fraction'] for row in scene_rows]))}
                aggregate['common_horizon_leg_rms'] = float(np.sqrt(np.mean([row['common_horizon_leg_rms'] ** 2 for row in scene_rows])))
                aggregate['early_termination_count'] = sum(
                    sum(value is not None for value in model[mode][name].get('first_terminated_tick', []))
                    if engine == 'isaac' else sum(row['reason'] is not None for row in model[mode][name].get('failures', []))
                    for name, _ in FORMAL_SCENARIOS)
                if mode == 'sampled':
                    paired = []
                    other = results[engine]['models']['301'][mode]
                    for replica in range(REPLICAS):
                        rms_a = np.sqrt(np.mean([other[name]['metrics']['replicas'][replica]['leg_rms'] ** 2 for name, _ in FORMAL_SCENARIOS]))
                        rms_b = np.sqrt(np.mean([model[mode][name]['metrics']['replicas'][replica]['leg_rms'] ** 2 for name, _ in FORMAL_SCENARIOS]))
                        paired.append(float(rms_b - rms_a))
                    aggregate['paired_401_minus_301_replica_rms'] = paired
                rows.append(aggregate)
    for name, _ in FORMAL_SCENARIOS:
        baseline = np.load(OUTPUT / 'isaac' / f'301_{name}_mean.npz', allow_pickle=False)
        delta = {}
        for update, mode in ((301, 'sampled'), (401, 'mean'), (401, 'sampled')):
            other = np.load(OUTPUT / 'isaac' / f'{update}_{name}_{mode}.npz', allow_pickle=False)
            delta[f'{update}_{mode}'] = {key: float(np.max(np.abs(other[key] - baseline[key])))
                                       for key in ('initial_observation', 'initial_joint_position', 'initial_joint_velocity', 'initial_base_height')}
        mujoco_reset = results['mujoco']['models']['301']['sampled'][name]['resets'][0]
        cross = {'policy_observation_max_abs_difference': float(np.max(np.abs(baseline['initial_observation'][0] - np.asarray(mujoco_reset['observation'])))),
                 'base_height_abs_difference_m': float(abs(baseline['initial_base_height'][0, 0] - mujoco_reset['metrics']['base_height_m']))}
        initial_difference.append({'scene': name, 'isaac_groups': delta, 'cross_engine': cross})
    verification = {'frozen_files_checked': 118, 'checked_conditions': len(verified), 'traces_and_csv': verified,
                    'rows': rows, 'reset_differences': initial_difference, 'verification_source_sha256': digest(__file__),
                    'engine_result_sha256': {engine: digest(OUTPUT / engine / 'result.json') for engine in results},
                    'common_noise_seed': 41882943, 'policy_updates': 0, 'optimizer_updates': 0}
    assert frozen_identity() == frozen
    write_json(OUTPUT / 'verification.json', verification)
    make_report(results, verification)
    print(json.dumps({'conditions': len(verified), 'rows': rows, 'reset_differences': initial_difference[0]}, indent=2), flush=True)


def make_report(results, verification):
    lines = ['# 固定策略：执行方式与仿真器诊断', '',
             '两个已有模型实际完成301、401次更新。所有新增测试均冻结策略，无优化器、无参数更新。MuJoCo平均动作复用已有八场景CSV；新增采样测试使用16组固定独立标准正态序列，各条件共享同一序列。', '',
             '## 八场景等权结果', '',
             '持续腿部加速度取第51步之后、四个实际驱动关节速度差除以0.02计算。表中RMS对八场景的加速度能量等权合并。速度误差也排除最初1秒。', '',
             '| 仿真器 | 动作方式 | 完成更新 | 持续腿部RMS rad/s² | 速度MAE m/s | yaw速度MAE rad/s | 力矩饱和比例 | 有效控制步比例 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in verification['rows']:
        lines.append(f"| {row['engine']} | {row['mode']} | {row['update']} | {row['leg_rms']:.6f} | {row['speed_mae']:.6f} | {row['yaw_rate_mae']:.6f} | {row['effort_saturation']:.4%} | {row['valid_fraction']:.4%} |")
    lines += ['', '## 各场景301到401的变化', '',
              '| 仿真器 | 场景 | 平均动作301 RMS | 平均动作401 RMS | 采样动作301 RMS | 采样动作401 RMS |',
              '|---|---|---:|---:|---:|---:|']
    for engine, result in results.items():
        for name, _ in FORMAL_SCENARIOS:
            numbers = [result['models'][str(update)][mode][name]['metrics']['leg_rms']
                       for mode in ('mean', 'sampled') for update in (301, 401)]
            lines.append(f"| {engine} | {name} | " + ' | '.join(f'{value:.6f}' for value in numbers) + ' |')
    lines += ['', '## 定位结论', '',
              '平均动作下，301到401的持续腿部运动在Isaac和MuJoCo内部都明显增加。该退化可以在Isaac本身重现，MuJoCo特有问题无法解释全部现象。', '',
              '同一冻结策略仅切换执行方式，就会出现不同的运动分布：Normal采样动作的腿部RMS一直很高，两个模型之间的变化远小于平均动作下的变化。这里存在训练所观察的随机策略与实际平均策略闭环稳定性的差距；采样RMS接近不能作为平均策略稳定的保证。', '',
              '正式PPO在ppo.py的act中通过act_with_context收集采样动作；update使用surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy_mean，并叠加CENet辅助项。当前训练没有用另一次平均动作物理闭环rollout直接约束其持续振荡。', '',
              '这项测试支持后续直接验证平均策略的闭环稳定性约束及其与速度改善能否同时成立。它尚未证明某一个奖励权重、entropy_coef或某一条梯度是唯一根因，也没有证明降低探索噪声便能修复问题。原有实际轮共同动作历史干预仍是反馈回路参与振荡的证据；本次确定该现象同时存在于两个物理引擎。', '',
              '## 证据范围', '',
              '- Isaac使用64个名义评估环境，正式动力学和动作限制保持；名义评估关闭观测噪声、动力学随机化和随机初速度，固定命令关闭训练命令练习。每个条件16环境，其中平均动作重复只用于检查并行环境的差异。',
              '- 两个引擎均保持20毫秒控制周期、5毫秒物理步长、500步和航向保持关闭。原始动作不进行±1执行裁剪；原有腿目标角限幅保留。',
              '- Isaac保留正式训练终止和10步宽限，正式timeout发生在第499步，随后reset的数据不计入原始轨迹。MuJoCo保留正式即时失败、闭链误差、接触等额外条件。有效控制步比例会体现timeout，不能把正常timeout称作倒地。',
              '- verification.json另列全部条件共同第51～499步的RMS，检查时间范围差异对结论的影响。',
              '- Isaac力矩饱和来自控制步末实际力矩，MuJoCo来自四个物理子步的饱和比例，因此主要比较各引擎内部301到401的变化。',
              '- Isaac四组条件的初始关节位置、速度、高度及观测逐位相同。跨引擎初始策略观测最大绝对差为4.6e-7，高度差为零；完整指标位于verification.json。被动关节状态与后续动力学响应仍不能据此宣称逐位相同。',
              '- 探索噪声采用各模型保存的六维标准差；使用相同标准正态序列保证配对，未强行固定两个模型的标准差。采用mean+sigma*epsilon实现同一Normal分布，未声称重演原训练的随机数序列。',
              '- 本次名义测试用于分离执行方式和引擎影响；不能替代原6144环境的全部随机化训练分布，也不能根据采样RMS较高就断言训练失败。',
              '', '## 复核', '',
              f"独立复核{verification['checked_conditions']}个场景条件，118个冻结文件指纹一致；核对模型、正式导出、复用CSV与新增NPZ的SHA256。采样动作与保存的mean/sigma/epsilon逐位或在浮点舍入误差内相符，Isaac全部策略参数执行前后逐位相同。", '',
              '主要记录：isaac/result.json、mujoco/result.json、verification.json；逐步物理记录位于对应引擎的NPZ。']
    (OUTPUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    torch.set_num_threads(1)
    assert torch.version.cuda is None
    verify()
