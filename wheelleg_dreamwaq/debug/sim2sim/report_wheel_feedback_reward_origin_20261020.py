from pathlib import Path

from debug.sim2sim.audit_reward_economics_20261019 import digest, frozen_identity, read_json, write_json
from debug.sim2sim.wheel_feedback_reward_origin_20261020 import OUTPUT, VALUE_NAMES


def main():
    frozen_identity()
    result = read_json(OUTPUT / 'result.json')
    verified = read_json(OUTPUT / 'verification.json')
    physical = read_json(OUTPUT / 'interventions_318/result.json')
    critical = next(row for row in result['updates'] if row['update'] == 318)
    values = critical['contributions']['with_within_update_momentum']
    labels = {'action_smooth_wheel': '轮原始动作二阶平滑，权重-0.01',
              'tracking_yaw_rate_enhance': '增强偏航追踪，权重1', 'tracking_yaw_rate': '基础偏航追踪，权重1',
              'tracking_vx': '基础前进速度追踪，权重2', 'tracking_vx_enhance': '增强前进速度追踪，权重2',
              'action_smooth_leg': '腿原始动作二阶平滑，权重-0.02',
              'dof_acc': '实际关节角加速度，权重-2.5e-7', 'action_rate': '一阶动作变化，权重-0.01',
              'torques': '驱动力矩，权重-1e-4'}
    lines = ['# 2026-10-20 哪些奖励增强轮共同动作历史反馈', '',
             '真实记录的CPU分解与新的MuJoCo局部干预共同支持：轮原始动作二阶平滑和偏航追踪参与增强这条快速反馈。轮平滑与增强偏航追踪在预先选定的10次更新中均为增强方向。当前证据定位了具体参与项，完整GPU逐项梯度验收和跨仿真器修复尚未完成。', '',
             '本轮新增训练更新为0，未启动额外CUDA进程；所有新文件位于debug。118冻结文件逐项SHA256一致。其他项目正在GPU上训练，当前GPU严格归因与Isaac候选验证保留待验状态。', '',
             '## 真实更新中的分项', '',
             '更新号为301、313、317、318、319、320、321、350、400、401，每次40个真实PPO小批次。每次67份原工件的长度和SHA256均核对。使用同一组32个已保存真实状态和8.444444Hz响应平方增益；正号表示局部增强该频率的轮共同历史响应。这个量不是奖励收益、腿部RMS或完整闭环稳定判据。', '',
             '以下贡献包括当次产生的梯度，以及该梯度在后续小批次中累积的一阶Adam动量；进入本次更新之前的动量单列。实际PPO裁剪分支、全局裁剪系数、学习率和每一步Adam二阶动量分母均保持原记录。', '',
             '| 奖励项 | 10次中增强/削弱 | 第318次局部贡献 |',
             '|---|---:|---:|']
    for name, label in labels.items():
        signs = verified['sign_consistency'][name]
        lines.append(f'| {label} | {signs["positive_updates"]}/{signs["negative_updates"]} | {values[name]:+.9f} |')
    value_sum = sum(values[name] for name in VALUE_NAMES)
    lines += ['', f'第318次value合并项贡献为{value_sum:+.9f}；进入该次更新之前的Adam一阶动量为{critical["summary"]["incoming_before_update_projection"]:+.9f}。加上其他奖励及单列残差后，重建实际actor更新的一阶投影{critical["summary"]["actual_projection"]:+.9f}。分项含正负抵消，不能把某项除以净值解释为独立因果百分比。', '',
              '轮平滑在10次中的贡献范围为+0.011341514～+0.034153142；增强偏航为+0.002065008～+0.015607101。基础偏航和前进速度项随批次改变符号，因此不能笼统归结为所有速度权重过大。腿动作二阶平滑9次增强、1次削弱，量级较小；它与真实关节加速度惩罚的含义不同。', '',
              '还检查了分解方法的敏感性：将18项奖励的GAE分量改为不分别减均值，将对应常数统一放入value分量，总优势保持完全相同。轮平滑仍10/10次增强；增强偏航变为9/10，301次出现反向。这说明奖励贡献标签受基线分配约定影响，增强偏航的10/10结论必须限定在原分解约定下。', '',
              '另外直接计算了单项权重倍率在alpha=1处的局部导数，并纳入总优势均值及标准差随该权重的变化。轮平滑、增强偏航的导数均10/10为正；第318次分别为+0.017168812、+0.007416962。这个导数固定原批次、critic、PPO分支、全局裁剪和Adam二阶分母，只能支持当前位置的局部方向，不能预测大幅改权重后的完整训练。优势导数与有限差分独立核对，误差低于1.2e-9。相关证据为centering_sensitivity/result.json和weight_sensitivity/result.json。', '',
              '## 新的局部物理干预', '',
              '围绕实际318次终点，只减去相应奖励在该次40个小批次内形成的actor参数更新，包括它在当次一阶动量中的后续影响。更早动量、二阶动量分母、后续原批次和encoder轨迹固定。这种终点参数干预没有重新生成完整训练轨迹，也不是删奖励后重新训练的结果。', '',
              '原318基线复用已完成且CSV指纹一致的评估；五个新条件各运行八场景500步、heading OFF。持续指标剔除首50步，合并四腿关节及八场景。', '',
              '| 移除的当次贡献 | 持续腿部RMS rad/s² | 速度MAE m/s | 偏航速度MAE rad/s |',
              '|---|---:|---:|---:|']
    cases = {'baseline': '实际318基线', 'wheel_smooth': '轮二阶平滑', 'yaw_enhance': '增强偏航追踪',
             'yaw_pair': '两项偏航追踪', 'wheel_smooth_and_yaw_pair': '轮二阶平滑及两项偏航', 'value_combined': '整个value合并项'}
    baseline = physical['baseline']['aggregate']
    for name, label in cases.items():
        case = physical['baseline'] if name == 'baseline' else physical['cases'][name]
        row = case['aggregate']
        lines.append(f'| {label} | {row["leg_rms"]:.9f} | {row["speed_mae"]:.9f} | {row["yaw_rate_mae"]:.9f} |')
    reduction = 1 - physical['cases']['wheel_smooth_and_yaw_pair']['aggregate']['leg_rms'] / baseline['leg_rms']
    lines += ['', f'轮平滑与两项偏航贡献同时移除时，RMS下降约{reduction * 100:.1f}%，反馈平方增益由0.308459848降至0.275447667；40个新场景全部完成。偏航误差变大，速度等指标存在取舍。移除value虽然减小抖动，速度与偏航严重恶化，不能作为修复。当前数据支持这些贡献参与物理振荡，不能据此断言全部退化由这几项独立造成或直接删掉就能完成双仿真器验收。', '',
              '## 轮二阶平滑为什么会加强历史反馈', '',
              '冻结工程的精确公式是：', '',
              '`r_wheel = -0.01 * 0.02 * sum((a_t - 2*a_(t-1) + a_(t-2))²)`', '',
              '其中a是实际执行的原始采样轮动作；相应历史动作也进入策略观测。给定历史和当前Gaussian sigma，只看该即时单项的条件期望，最优均值为`mu_t = 2*a_(t-1) - a_(t-2)`。它要求外推前两次采样动作的趋势；其中也包含探索噪声。它没有直接要求腿关节的实际加速度小，也没有提供轮闭环的物理阻尼保证。', '',
              f'这个单项条件最优目标对历史的传递式为`2*z^-1 - z^-2`，在8.444444Hz、dt=0.02秒时幅值为{verified["second_difference_immediate_target_transfer_gain_at_8_444Hz"]:.6f}。这是公式推导，描述单项即时目标；完整策略同时受未来回报、其他奖励、估值和物理反馈影响，不能把1.746解释为机器人实际增益。真实分解中轮平滑10/10次增强反馈、局部移除后抖动下降，给这条数学耦合补上了训练及物理证据。', '',
              '增强偏航项的公式为`0.02*(exp(-yaw_error²/2.5)-1)`。它是宽误差范围的偏航追踪项。记录中它10/10次增强同一轮共同反馈，单独移除相应当次贡献也减小腿部RMS；现阶段尚不能把其作用进一步唯一拆成身体动力学耦合、采样相关性或某一种网络特征。', '',
              '## value与算法范围', '',
              'value合并项已按GAE恒等式拆为当前状态负估值、区间内后续估值、24步末端bootstrap、超时补偿及舍入残差。第318次含当次动量的贡献如下：', '',
              '| value部分 | 局部贡献 |', '|---|---:|']
    for name in VALUE_NAMES:
        lines.append(f'| {name} | {values[name]:+.9f} |')
    lines += ['', '这些value分项明显抵消。末端V未独立保存，末端贡献从原始优势与最后TD残差推得；不能把推得结果冒充独立观测。当前critic只有一个总回报输出，因此这份时序分解不能唯一证明“value主要来自某项奖励”。原始合并value贡献已经保存，未向即时奖励强行分摊。', '',
              '实际Adam更新逐参数重建误差小于1e-7。CPU与记录GPU总actor梯度的最大相对范数残差为0.0709523%，已单列；分项与残差共同重建局部投影。这个总残差不构成每个奖励分项的误差上界。严格GPU分项归因状态仍为false。', '',
              '本次奖励归因覆盖actor参数路径。encoder更新也改变完整策略的历史响应，且不同更新中方向不同：第318次actor局部变化和为+0.013109684，encoder及其他变化为-0.004717439，完整策略净变化+0.008392245；第319次encoder及其他变化则为正。verification.json逐次列出，未把encoder变化全归给actor奖励分量。', '',
              '本轮十个更新是既有预选取证点；它们不能代表301～401每一次更新，更不能把十次分项简单相加当成全部训练的累积因果。全部为同一个种子和真实原批次。', '',
              '## 下一步依据', '',
              '优先验证轮原始动作二阶平滑与动作历史的耦合，以及增强偏航梯度怎样形成轮共同输出的快速响应。仅调大腿部动作平滑可能继续增加反馈；真实关节加速度惩罚在本次分解中方向相反。删除整个bootstrap会破坏控制表现，不能作为修复。后续debug改法须同时保住偏航、前进速度、Isaac与MuJoCo表现，不能只以这份局部RMS下降验收。', '',
              '证据文件：result.json、update_0301.json等10份更新、interventions_318/result.json、verification.json。脚本均位于debug/sim2sim。所有原训练checkpoint和已有评估原样保留。', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(OUTPUT / 'report_identity.json', {'report_sha256': digest(OUTPUT / 'report.md'),
               'result_sha256': digest(OUTPUT / 'result.json'), 'verification_sha256': digest(OUTPUT / 'verification.json'),
               'physical_result_sha256': digest(OUTPUT / 'interventions_318/result.json'),
               'source_sha256': digest(__file__), 'frozen_files_checked': 118})
    print(str(OUTPUT / 'report.md'))


if __name__ == '__main__':
    main()
