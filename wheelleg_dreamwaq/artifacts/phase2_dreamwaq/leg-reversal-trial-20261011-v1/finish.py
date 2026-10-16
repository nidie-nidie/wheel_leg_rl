"""Independent scalar verification and matched-condition outcome report."""
from pathlib import Path
from collections import deque
import csv, hashlib, json, math, struct

OUT = Path(__file__).resolve().parent
PROJECT = Path(r"E:\wheel_leg_rl-main\wheelleg_dreamwaq")
NAMES = ["baseline", "budget12", "budget8"]

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()

def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]

def scalar_counter(data, nominal):
    directions = [0] * 4
    extremes = [f32(v) for v in nominal]
    windows = [deque(maxlen=50) for _ in range(4)]
    max_error = 0.
    for row in data:
        for joint in range(4):
            value = f32(float(row[f"leg_target_{joint}_rad"]))
            direction = directions[joint]
            extreme = extremes[joint]
            if direction > 0:
                extreme = max(extreme, value)
            elif direction < 0:
                extreme = min(extreme, value)
            event = False
            if direction > 0 and value <= f32(extreme - f32(.01)):
                direction = -1; extreme = value; event = True
            elif direction < 0 and value >= f32(extreme + f32(.01)):
                direction = 1; extreme = value; event = True
            elif direction == 0 and value >= f32(extreme + f32(.01)):
                direction = 1; extreme = value
            elif direction == 0 and value <= f32(extreme - f32(.01)):
                direction = -1; extreme = value
            directions[joint] = direction
            extremes[joint] = extreme
            windows[joint].append(int(event))
            calculated = sum(windows[joint])
            expected = float(row[f"leg_reversal_rate_{joint}_per_s"])
            max_error = max(max_error, abs(calculated - expected))
    return max_error

def rms(values):
    return math.sqrt(sum(v*v for v in values) / len(values)) if values else None

def metrics(data):
    valid = [r for r in data if r["survived"] == "1" and 2.0 + 1e-9 < float(r["time_s"]) <= 10.0 + 1e-8]
    if not valid:
        return {"samples": 0, "leg_target_reversal_rate_per_s": None,
                "leg_velocity_rms_rad_s": None, "leg_acceleration_rms_rad_s2": None}
    vector_mean = lambda stem: [sum(float(r[stem.format(j)]) for r in valid) / len(valid) for j in range(4)]
    vector_rms = lambda stem: [rms([float(r[stem.format(j)]) for r in valid]) for j in range(4)]
    return {"samples": len(valid), "window": "valid post-action samples 2s<t<=10s",
            "leg_target_reversal_rate_per_s": vector_mean("leg_reversal_rate_{}_per_s"),
            "leg_velocity_rms_rad_s": vector_rms("leg_velocity_{}_rad_s"),
            "leg_acceleration_rms_rad_s2": vector_rms("leg_acceleration_{}_rad_s2"),
            "vx_mae_mps": sum(abs(float(r["vx_mps"]) - float(r["target_vx_mps"])) for r in valid) / len(valid),
            "pitch_rms_rad": rms([float(r["pitch_rad"]) for r in valid]),
            "action_saturation_fraction": sum(float(r["action_saturation_fraction"]) for r in valid) / len(valid),
            "effort_saturation_fraction": sum(float(r["effort_saturation_fraction"]) for r in valid) / len(valid)}

def main():
    manifest = read(OUT / "experiment-manifest.json")
    assert manifest["status"] == "evaluated", manifest["status"]
    comparisons = []
    maximum_counter_error = 0.
    verified_frames = 0
    acceleration_error = 0.
    trace_hashes = {}
    for arm in manifest["arms"]:
        name = arm["name"]
        folder = OUT / name
        policy = read(folder / "export/policy_manifest.json")
        summary = {}
        for backend in ["isaac", "mujoco"]:
            report = read(folder / backend / "summary.json")
            assert report["report_hash"] == hashlib.sha256(json.dumps(
                {k: v for k, v in report.items() if k != "report_hash"},
                sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest().upper()
            scenarios = report["aggregate"]["scenarios"]
            gate_scenarios = {"nominal_stand": .1, "low_stand": .1, "high_stand": .1,
                              "forward": .2, "reverse": .2}
            summary[backend] = {
                "full_survival": report["aggregate"]["full_survival"],
                "scenarios": {r["name"]: {"completed": r["completed"], "vx_mae": r.get("vx_mae"),
                                         "yaw_rate_mae": r.get("yaw_rate_mae"),
                                         "failure_reason": r["failure_reason"]} for r in scenarios},
                "vx_gate_passed": all(r["completed"] and r.get("vx_mae") is not None and
                                     r["vx_mae"] <= gate_scenarios[r["name"]]
                                     for r in scenarios if r["name"] in gate_scenarios)}
        diagnostic = {}
        for scenario, identity in read(folder / "mujoco/summary.json")["scenario_files"].items():
            path = Path(identity["path"])
            assert sha(path) == identity["sha256"]
            trace_hashes[str(path)] = sha(path)
            data = rows(path)
            maximum_counter_error = max(maximum_counter_error, scalar_counter(data, policy["q_nominal"]))
            verified_frames += len(data)
            for prev, current in zip(data, data[1:]):
                for joint in range(4):
                    estimated = (float(current[f"leg_velocity_{joint}_rad_s"]) -
                                 float(prev[f"leg_velocity_{joint}_rad_s"])) / .02
                    acceleration_error = max(acceleration_error, abs(
                        estimated - float(current[f"leg_acceleration_{joint}_rad_s2"])))
            diagnostic[scenario] = metrics(data)
        practice = read(folder / "command_practice/summary.json")
        summary["command_practice"] = {
            "performance_accepted": practice["performance_accepted"],
            "scenarios": {s["name"]: s for s in practice["scenarios"]}}
        summary["diagnostic"] = diagnostic
        summary["name"] = name
        summary["profile"] = arm["profile"]
        summary["gain_state"] = arm["gain_state"]
        summary["training_minutes"] = arm["duration_seconds"] / 60.
        comparisons.append(summary)
    assert maximum_counter_error == 0., maximum_counter_error
    assert acceleration_error < 1e-10, acceleration_error
    baseline = read(OUT / "baseline.json")
    protected = {**baseline["protected"], **baseline["preexisting_edit_hashes"]}
    unchanged = all(sha(PROJECT.parent / p) == h for p, h in protected.items())
    assert unchanged
    results = {"experimental_only": True, "seed": 41882943,
               "new_iterations_per_arm": 200, "weights_only_source_iterations": 1000,
               "comparison": comparisons, "counter_verification": {
                   "independent_scalar_max_error": maximum_counter_error, "verified_frames": verified_frames, "independent_acceleration_max_error": acceleration_error},
               "protected_files": len(protected), "all_protected_hashes_unchanged": unchanged,
               "trace_hashes": trace_hashes,
               "limitations": ["One seed and 200 new iterations per arm; exploratory, not formal acceptance.",
                   "Target reversals and actual joint motion are separate acceptance metrics.",
                   "Source optimizer reset identically in all three arms.",
                   "1s post-reset warmup excludes short-lived episodes from reversal costs."]}
    write(OUT / "results.json", results)
    table = []
    for arm in comparisons:
        parking = arm["diagnostic"]["nominal_stand"]
        values = [arm["name"], str(parking["samples"])]
        for key in ("leg_target_reversal_rate_per_s", "leg_velocity_rms_rad_s", "leg_acceleration_rms_rad_s2"):
            pair = parking.get(key)
            values.append("/".join(f"{v:.3f}" for v in pair[:2]) if pair else "无有效窗口")
        for scenario in ("nominal_stand", "forward", "reverse"):
            value = arm["mujoco"]["scenarios"][scenario]["vx_mae"]
            values.append(f"{value:.4f}" if value is not None else "null")
        table.append("| " + " | ".join(values) + " |")
    report = """# 四腿目标换向约束：一种子三组200轮试验

结论：这版换向约束没有解决左腿抽搐，暂不扩大训练。相对于同样新增200轮的baseline，N12的左腿目标反向次数约减少6.8%，但jIJ/jIO实际角速度RMS增加10.7%/36.3%；N8的jIJ增加17.6%，jIO减少13.7%。目标指令少反向，不能直接等同于实际关节更平稳。

MuJoCo后退跟踪有所改善，但两组的停车和前进误差均大于baseline。三组在两端八场景都存活；只有N8通过MuJoCo的vx子门槛，三组都未通过Isaac的vx子门槛及全部停车换向专项。不能把“不倒”当作整体性能验收。

两个自适应组最终gain都达到1.0，停车换向率仍超过各自预算。这次没有固定gain对照，不能据此证明动态调权优于固定惩罚。这里只采用A1启发的有界ROGER-style单项调权，不是论文完整ROGER算法。一个种子、每组200轮只支持当前配置不宜扩大，不能证明平稳与速度跟踪不能同时实现。

本报告是探索结果，不是正式四种子验收。三组网络均从当前PhysicsV5 stop_reverse_v1 run03 model_999复制，optimizer/RNG/iteration重置；种子41882943，256环境/8minibatches/24步。原六维action_smooth=-0.01和速度/姿态等奖励不变。仅N12/N8增加0.01rad迟滞、1s窗口与有界ROGER-style单项调权。

## MuJoCo停车实际运动

左腿顺序jIJ/jIO。抖动采用有效动作后2s<t<=10s，目标反向为最近1秒滑窗的均值；加速度为20ms前后速度差，不是1ms瞬时qacc。速度误差表列为原八场景全程指标；短时间跌倒的速度误差不可单独当作通过。

| 组 | 停车有效帧 | 左腿目标反向 次/s | 左腿实际速度RMS rad/s | 左腿20ms加速度RMS rad/s² | 停车vx MAE | 前进vx MAE | 后退vx MAE |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
""" + "\n".join(table) + "\n\n## 存活与原验收门槛\n\n"
    for arm in comparisons:
        gain = arm["gain_state"]
        report += (f"- {arm['name']}: 训练{arm['training_minutes']:.2f}分钟；"
                   f"Isaac八场景存活={arm['isaac']['full_survival']}，vx门槛={arm['isaac']['vx_gate_passed']}；"
                   f"MuJoCo八场景存活={arm['mujoco']['full_survival']}，vx门槛={arm['mujoco']['vx_gate_passed']}；"
                   f"四换向专项全部通过={arm['command_practice']['performance_accepted']}；"
                   f"最终gain={gain['gain'] if gain else 'disabled'}。\n")
    report += f"""
## 核验与限制

201项CPU单测及2项真实Isaac集成通过。恢复包含非零gain单测；真实初始化/resume/play/export、标准golden32与同批量Python/TorchScript 1/3/32均保留1e-7精度。原目标/时序/物理/动作/命令/随机化/history未改。{len(protected)}个受保护文件hash不变，{verified_frames}帧的换向率由独立标量算法重算，最大误差{maximum_counter_error}。

三组相同网络起点与同一reset cache已核验，源码训练期间冻结。普通resume保持跨合同拒绝，weights-only采用独立身份。MuJoCo独立入口只适配三组200轮上下文及增加无副作用的目标/速度读取，原正式4×1000入口与排名拒绝规则保留。

ROGER-style调权只调整新增惩罚，不学习预算，不衰减主奖励，也不是论文全算法。实际抖动与目标反向需同时判断；一种子200轮不足以确定跨种子效果。reset后1秒预热会排除很短episode的约束成本，不能把这些episode的低成本当作健康控制。完整逐场景/逐腿/换向阶段数据见results.json与各组summary。
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for name, color in zip(NAMES, ["#444444", "#0072B2", "#D55E00"]):
        data = rows(OUT / name / "mujoco/nominal_stand.csv")
        data = [r for r in data if r["survived"] == "1" and float(r["time_s"]) > 2.]
        if not data:
            continue
        ts = [float(r["time_s"]) for r in data]
        for j in range(2):
            axes[j].plot(ts, [float(r[f"leg_velocity_{j}_rad_s"]) for r in data],
                         label=name, color=color, alpha=.8, linewidth=1.)
        axes[2].plot(ts, [float(r["vx_mps"]) for r in data], label=name, color=color)
    for i, ax in enumerate(axes):
        ax.grid(alpha=.25)
        ax.legend(ncol=3)
        ax.set_ylabel(["Left jIJ velocity (rad/s)", "Left jIO velocity (rad/s)", "COM vx (m/s)"][i])
    axes[2].axhline(0., color="black", linestyle="--", linewidth=.7)
    axes[2].set_xlabel("Time after reset (s)")
    fig.suptitle("Matched parking: actual joint motion and speed")
    fig.tight_layout()
    fig.savefig(OUT / "parking-comparison.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"counter_max_error": maximum_counter_error, "frames": verified_frames,
                      "protected_unchanged": unchanged, "report": str(OUT / "report.md")}, indent=2))

if __name__ == "__main__":
    main()
