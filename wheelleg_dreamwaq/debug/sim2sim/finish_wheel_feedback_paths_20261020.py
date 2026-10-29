import json
from pathlib import Path

import numpy as np
import torch

from debug.sim2sim.action_mode_engine_probe_20261020 import SIM
from debug.sim2sim.audit_reward_economics_20261019 import DEFAULT_RUN as RUN, digest, frozen_identity, read_json, write_json
from debug.sim2sim.verify_wheel_feedback_paths_20261020 import rows_of, independent_metrics
from debug.sim2sim.wheel_feedback_paths_20261020 import OUTPUT
from debug.sim2sim.wheel_feedback_update318_20261020 import KEYS, make_policy, probe_inputs, score


def check_scenes(scenes):
    full_speed = []
    for record in scenes:
        assert digest(record['csv']) == record['csv_sha256']
        rows = rows_of(record['csv'])
        values = independent_metrics(rows)
        assert all(abs(values[key] - record[key]) < 1e-12 for key in values)
        full_speed.append(sum(abs(float(row['vx_mps']) - float(row['target_vx_mps'])) for row in rows) / 500)
    assert len(scenes) == 8
    return {'leg_rms': float(np.sqrt(np.mean([row['leg_rms'] ** 2 for row in scenes]))),
            'speed_mae_tail450': float(np.mean([row['speed_mae'] for row in scenes])),
            'speed_mae_all500': float(np.mean(full_speed))}


def main():
    torch.set_num_threads(1)
    assert torch.version.cuda is None
    before = frozen_identity()
    verified = read_json(OUTPUT / 'verification.json')
    script_dir = Path(__file__).resolve().parent
    for name, expected in verified['source_hashes'].items():
        assert digest(script_dir / name) == expected
    paths = read_json(OUTPUT / 'result.json')
    slow = read_json(OUTPUT / 'slow_history/result.json')
    adam = read_json(OUTPUT / 'actual_update318/result.json')
    physical = read_json(OUTPUT / 'actual_update318/physical_counterfactuals/result.json')
    reward = read_json(OUTPUT / 'actual_update318/reward_directions_recorded_forward_cpu/result.json')
    wheel = read_json(OUTPUT / 'actual_update318/wheel_smooth_recorded_forward_cpu_intervention/result.json')
    assert physical['same_batch_python_torchscript_error'] == 0
    assert not reward['strict_reward_attribution_passed'] and not wheel['strict_reward_attribution_passed']
    reward_residual = 0.
    for row in reward['minibatches']:
        value = sum(row['estimated_contributions'].values()) + row['unassigned_cpu_backward_residual_projection']
        expected = adam['minibatches'][row['minibatch']]['current_gradient_numerator_change']
        error = abs(value - expected)
        assert error < 2e-6
        reward_residual = max(reward_residual, error)
    assert len(reward['minibatches']) == 40
    for case in physical['cases'].values():
        assert digest(case['identity']['path']) == case['identity']['sha256']
        values = check_scenes(case['scenes'])
        assert abs(values['leg_rms'] - case['aggregate']['leg_rms']) < 1e-12
    assert digest(wheel['state_path']) == wheel['state_sha256']
    values = check_scenes(wheel['scenes'])
    assert abs(values['leg_rms'] - wheel['aggregate']['leg_rms']) < 1e-12
    for row in wheel['minibatches']:
        expected = reward['minibatches'][row['minibatch']]['estimated_contributions']['action_smooth_wheel']
        assert abs(row['feedback_metric_projection'] - expected) < 2e-6
    payload = torch.load(RUN / 'model_317.pt', map_location='cpu', weights_only=False, mmap=True)
    initial = torch.load(RUN / 'model_316.pt', map_location='cpu', weights_only=False, mmap=True)
    first = torch.load(RUN / 'training_evidence/update_0318/minibatch_00.pt', map_location='cpu', weights_only=False, mmap=True)
    last = torch.load(RUN / 'training_evidence/update_0318/minibatch_39.pt', map_location='cpu', weights_only=False, mmap=True)
    assert all(torch.equal(first['policy_before'][key], initial['model_state_dict'][key]) for key in KEYS)
    assert all(torch.equal(last['actor_parameters_after'][key], payload['model_state_dict'][key]) for key in KEYS)
    history = torch.from_numpy(np.concatenate([np.load(row['path'], allow_pickle=False)['history'][row['indices']] for row in adam['probe_identity']]))
    gains = {}
    for label, state in (('before318', initial['model_state_dict']), ('after318', payload['model_state_dict'])):
        policy = make_policy(state)
        gains[label] = {}
        for epsilon in (.0005, .001, .002):
            with torch.inference_mode():
                value, kernel = score(policy.actor, probe_inputs(policy, history, epsilon), adam['frequency_hz'], epsilon)
            gains[label][str(epsilon)] = {'gain_squared': float(value), 'kernel': kernel.tolist()}
    assert all(gains['after318'][str(epsilon)]['gain_squared'] > gains['before318'][str(epsilon)]['gain_squared'] for epsilon in (.0005, .001, .002))
    old_summary = read_json(Path(slow['old701']['scenes'][0]['csv']).parents[1] / 'summary.json')
    source_changes = {name: {'old701': expected, 'current': digest(SIM / 'wheelleg_mujoco' / name)}
                      for name, expected in old_summary['evaluation_contract']['runtime_source_sha256'].items()
                      if expected != digest(SIM / 'wheelleg_mujoco' / name)}
    assert set(source_changes) <= {'contract.py', 'evaluation.py'}
    source_audit = read_json(OUTPUT / 'old701_evaluation_source_audit.json')
    assert source_audit['old_sha256'] == source_changes['evaluation.py']['old701']
    assert source_audit['current_sha256'] == source_changes['evaluation.py']['current']
    assert source_audit['physical_trajectory_equal_all_eight']
    for key, path in (('model_xml_sha256', SIM / 'models/wheelleg.xml'),
                      ('model_manifest_sha256', SIM / 'model_manifest.json'),
                      ('diagnostic_source_sha256', script_dir / 'evaluate_baseline_mujoco.py'),
                      ('failure_source_sha256', script_dir.parents[1] / 'training/scripts/evaluate_mujoco.py')):
        if key == 'model_xml_sha256':
            manifest = read_json(SIM / 'model_manifest.json')
            path = SIM / 'models' / manifest['model_xml']['path']
        assert digest(path) == old_summary['evaluation_contract'][key]
    comparison = {'old701': check_scenes(slow['old701']['scenes']),
                  'original401': check_scenes(paths['reused']['401']['scenes']),
                  'filter_alpha0': check_scenes(slow['cases']['alpha_0']['scenes']),
                  'filter_alpha05': check_scenes(slow['cases']['alpha_0p5']['scenes'])}
    report = [
        '# 2026-10-20 轮共同动作历史反馈定位',
        '',
        '冻结401次更新策略的物理干预，将主要持续振荡定位到“上一轮左右轮共同动作历史→下一轮共同轮动作输出→机器人状态”的闭环。仅移除辅助历史中的快速帧间变化，并只使用辅助策略的轮共同输出，MuJoCo八场景持续腿部RMS降至0.168880 rad/s²。每一步腿输出和轮差速输出仍由原策略对真实历史计算；真实历史始终回填真正执行的动作。',
        '',
        '当前仍未达到用户要求的完整目标：相同450步口径下，候选速度误差比旧701高约10.1%；右转腿部RMS比旧701高；Isaac候选验证和完整GPU奖励归因待执行。全部新增工作位于debug，训练新增更新次数为0，118冻结文件逐项SHA256保持一致。',
        '',
        '持续指标均排除最初50个控制步，使用后450步；腿部RMS合并四个驱动关节和八个场景，单位rad/s²。速度MAE同时列出后450步和完整500步，单位m/s。固定场景heading关闭。',
        '',
        '| 策略 | 持续腿部RMS | 速度MAE后450步 | 速度MAE全500步 |',
        '|---|---:|---:|---:|']
    labels = {'old701': '旧4096/701', 'original401': '6144/401原策略', 'filter_alpha0': '401仅移除快速共同历史作用', 'filter_alpha05': '401保留一半快速共同历史作用'}
    for label, row in comparison.items():
        report.append(f'| {labels[label]} | {row["leg_rms"]:.9f} | {row["speed_mae_tail450"]:.9f} | {row["speed_mae_all500"]:.9f} |')
    report += ['', '直接把所有轮共同历史作用取消时，RMS为0.150333，但速度MAE恶化至0.088503。只干预腿输出时RMS为22.018548，只干预差速输出时为28.814286。保留历史均值、移除帧间快速变化的干预保留了大部分401速度表现。', '',
               '| 场景 | 旧701腿部RMS | 候选腿部RMS | 旧701速度MAE | 候选速度MAE |', '|---|---:|---:|---:|---:|']
    for old, candidate in zip(slow['old701']['scenes'], slow['cases']['alpha_0']['scenes'], strict=True):
        assert Path(old['csv']).stem == candidate['name']
        report.append(f'| {candidate["name"]} | {old["leg_rms"]:.6f} | {candidate["leg_rms"]:.6f} | {old["speed_mae"]:.6f} | {candidate["speed_mae"]:.6f} |')
    report += ['',
        '两个快速历史干预条件各完成八场景500步和三组切换500步，未发生正式失败。H关闭、H开启、reset事件分别实际到达第101、201、301步；reset后仿真时间恢复到0.02秒。三组切换完成只证明协议执行与存活；切换中的腿部加速度仍有明显瞬态，未据此宣称救车或所有转向行为验收合格。',
        '',
        '真实第318次更新的40个Adam小批次，与原始起止checkpoint的actor参数逐位对应。按每次实际二阶动量分母，将更新分为历史一阶动量和当前梯度两部分，逐参数重建最大误差5.958827e-8。针对32个已保存的真实MuJoCo状态，观测频率8.444444Hz的轮共同历史响应平方增益被两部分共同增强：历史动量一阶贡献+0.00676775，当前梯度+0.00585756。40次actor局部实际变化和为+0.01310968；每次之间encoder也改变，该数值不等于完整策略起止差。完整策略起止响应在epsilon=0.0005、0.001、0.002三种有限差分下均增强。此响应指标描述局部策略敏感度，未包含完整机器人传递函数，不能独立充当闭环稳定性定理。',
        '', '| 318局部物理干预 | 持续腿部RMS | 速度MAE后450步 |', '|---|---:|---:|']
    for label, row in [('实际318', physical['baseline']['aggregate']), ('移除历史一阶动量分量', physical['cases']['without_carried_first_moment']['aggregate']), ('移除当前梯度分量', physical['cases']['without_current_gradient_numerator']['aggregate']), ('移除当前轮平滑估计分量', wheel['aggregate'])]:
        report.append(f'| {label} | {row["leg_rms"]:.9f} | {row["speed_mae"]:.9f} |')
    report += ['',
        '这些局部参数干预保持后续记录的批次、原Adam历史和二阶动量分母，围绕实际318终点减去相应更新分量。它们未重新生成后续训练轨迹。轮平滑干预还使用CPU反向估计，因此其结论范围更有限。历史动量、当前梯度、单项轮平滑贡献分别移除后均只有部分改善，当前证据无法把全部退化归给其中一个。',
        '',
        '奖励分项的严格直接CPU重建未通过原梯度误差门槛，完整GPU分项归因仍未通过。为提供后续检查方向，另做了固定真实GPU action_mean、sigma、log_prob与原裁剪分支的CPU估计，并把总梯度差额单列为未归属残差。最大残差范数约为原actor梯度的0.0552%；估计分项加残差投影能重建记录的当前梯度投影。该残差不能证明每一个奖励分项都具有同样的误差界。',
        '', '| 当前梯度的CPU估计分项 | 对8.444444Hz响应平方增益的一阶贡献 |', '|---|---:|']
    for name in ('value_bootstrap', 'action_smooth_wheel', 'tracking_yaw_rate', 'tracking_yaw_rate_enhance', 'tracking_vx', 'tracking_vx_enhance', 'dof_acc'):
        report.append(f'| {name} | {reward["estimated_summed_contributions"][name]:+.9f} |')
    report += ['',
        '正值表示局部增强该反馈，负值表示局部削弱。基础前进速度项在这一批次中为负；当前证据不支持“前进速度权重独自制造全部振荡”。轮平滑分量的局部CPU干预使反馈指标0.30845985降至0.30588982，同时腿部RMS14.721029降至13.994485，支持该贡献参与当前退化的线索，未证明直接删除该奖励能在完整训练中解决问题。value/bootstrap合并项仍包含value差分与尾部估值，不能把它的全部贡献解释为某项即时奖励的收益。',
        '',
        '工程中action_smooth_wheel的精确公式为-0.01*0.02*sum((a_t-2*a_(t-1)+a_(t-2))²)，属于原始轮动作二阶差分。固定历史和当前Gaussian sigma时，该单项即时惩罚的条件期望为-0.0002*sum((mu_t-2*a_(t-1)+a_(t-2))²+sigma²)。仅优化这个即时单项时，均值的最优目标是2*a_(t-1)-a_(t-2)。因此，历史动作输入、带噪声采样动作和二阶平滑惩罚之间存在明确的数学耦合；该目标本身不提供物理阻尼保证。这个公式只解释单项即时激励，完整PPO还包含未来回报、其他奖励和Adam历史，不能用它单独证明8.44Hz振荡的全部来源。',
        '',
        '下一项已经明确的验证对象是：相同冻结候选在Isaac中的真实闭环表现，以及使用原GPU算术重新分解同一真实318批次的轮平滑、偏航和value贡献。当前另一个项目正在执行6144环境训练，快照记录GPU仅剩约1.9GiB，因此没有并行启动Isaac。候选的跨仿真器合格状态保持待验收。完整修复还需要改善旧701前进/倒车与右转表现，并验证用户关心的救车能力；本次平地数据没有救车标签或专门扰动协议。',
        '',
        '旧701评估记录与当前源码的差异包含contract.py与evaluation.py。旧evaluation.py的完整原文本尚未恢复，因此未宣称源码完全等价。另复用已经完成的正式4096/701评估：它的运行源码指纹与当前全部一致；八场景各500步的27个物理、动作、命令字段与旧701 CSV数值逐位相同。XML、控制器、runner、观测、失败判据的旧记录指纹与当前一致。指标直接从这些CSV重新计算，当前118指纹保持冻结。',
        '',
        '证据：result.json、slow_history/result.json、verification.json、old701_evaluation_source_audit.json、actual_update318/result.json、actual_update318/physical_counterfactuals/result.json、actual_update318/reward_directions_recorded_forward_cpu/result.json、actual_update318/wheel_smooth_recorded_forward_cpu_intervention/result.json、resource_snapshot.json。原始失败记录与原训练checkpoint全部保留。']
    assert frozen_identity() == before
    write_json(OUTPUT / 'final_verification.json', {
        'new_fixed_scenes_verified': 88, 'switch_protocols_verified': 6, 'frozen_files_checked': 118,
        'training_updates_started': 0, 'candidate_isaac_accepted': False, 'strict_gpu_reward_attribution_passed': False,
        'comparison': comparison, 'full_policy_feedback_gain': gains, 'runtime_source_differences_from_old701': source_changes,
        'reward_estimates_plus_residual_max_projection_error': reward_residual,
        'real318_endpoint_actor_parameters_exact': True,
        'sources': {path.name: digest(path) for path in script_dir.glob('*wheel_feedback*20261020.py')},
        'result_hashes': {str(path.relative_to(OUTPUT)): digest(path) for path in OUTPUT.rglob('result.json')},
        'resource_snapshot_sha256': digest(OUTPUT / 'resource_snapshot.json')})
    (OUTPUT / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(json.dumps({'comparison': comparison, 'strict_gpu_attribution': False, 'isaac_candidate_accepted': False,
                      'new_fixed_scenes_verified': 88, 'report': str(OUTPUT / 'report.md')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
