import torch


def smoothstep(value):
    value = value.clamp(0., 1.)
    return value.square() * (3. - 2. * value)


def deadband_score(value, deadband, scale):
    return 1. / (1. + ((value.abs() - deadband).clamp_min(0.) / scale).square())


def compute_independent_reward(*, q, dq, omega, gravity, linear, height, torque,
                               actions, previous_actions, previous_previous_actions,
                               wheel_forces_mean, nonwheel_force_mean, reference, weight_n,
                               success, physical_failure, timeout, dt=.02):
    if dt != .02:
        raise ValueError("独立站立奖励固定使用 20ms 控制周期")
    n = q.shape[0]
    fields = {"q": (q, (n, 4)), "dq": (dq, (n, 6)), "omega": (omega, (n, 3)),
              "gravity": (gravity, (n, 3)), "linear": (linear, (n, 3)), "height": (height, (n,)),
              "torque": (torque, (n, 6)), "actions": (actions, (n, 6)),
              "previous_actions": (previous_actions, (n, 6)),
              "previous_previous_actions": (previous_previous_actions, (n, 6)),
              "wheel_forces_mean": (wheel_forces_mean, (n, 2)),
              "nonwheel_force_mean": (nonwheel_force_mean, (n,)),
              "success": (success, (n,)), "physical_failure": (physical_failure, (n,)),
              "timeout": (timeout, (n,))}
    for name, (value, shape) in fields.items():
        if value.shape != shape or not torch.isfinite(value).all():
            raise ValueError(f"独立站立奖励字段 {name} 形状或有限性无效")
    for value in (actions, previous_actions, previous_previous_actions):
        if torch.any(value.abs() > 1.000001):
            raise ValueError("独立站立奖励动作必须为实际执行的有界动作")
    if torch.any(success & (physical_failure | timeout)) or torch.any(physical_failure & timeout):
        raise ValueError("独立站立终态标记互斥")
    weight = torch.as_tensor(weight_n, device=q.device, dtype=q.dtype)
    if weight.ndim > 1 or (weight.ndim == 1 and weight.shape != (n,)) or not torch.isfinite(weight).all() or torch.any(weight <= 0):
        raise ValueError("独立站立重量无效")
    match = reference.nearest(q, height)
    theta = torch.acos((-gravity[:, 2]).clamp(-1., 1.))
    upright_gate = smoothstep((.50 - theta) / .35)
    standing_gate = upright_gate * smoothstep((height - .12) / .06)
    support = .5 * ((wheel_forces_mean / (.25 * weight[..., None])).clamp(0., 1.).min(dim=1).values
                      + deadband_score(wheel_forces_mean.sum(dim=1) / weight - 1., .20, .30))
    effort = q.new_tensor([18., 18., 18., 18., 9., 9.])
    components = {
        "upright": dt * 2. * (1. - gravity[:, 2]) / 2.,
        "height": dt * upright_gate * deadband_score(height - match.height, .003, .020),
        "pose": dt * .8 * standing_gate * deadband_score(q - match.q, .04, .10).mean(dim=1),
        "angular_velocity": dt * .5 * standing_gate * deadband_score(omega.norm(dim=1), .20, .50),
        "leg_velocity": dt * .25 * standing_gate * deadband_score(dq[:, :4], .20, 1.).mean(dim=1),
        "wheel_velocity": dt * .10 * standing_gate * deadband_score(dq[:, 4:], 1., 4.).mean(dim=1),
        "horizontal_velocity": dt * .30 * standing_gate * deadband_score(linear[:, :2].norm(dim=1), .01, .05),
        "support": dt * .50 * standing_gate * support,
        "nonwheel_support": -dt * .50 * standing_gate * (nonwheel_force_mean / (.10 * weight)).clamp(0., 1.),
        "torque": -dt * .002 * (torque / effort).square().mean(dim=1),
        "action_first_difference": -dt * .010 * (actions - previous_actions).square().mean(dim=1),
        "action_second_difference": -dt * .005 * (actions - 2. * previous_actions + previous_previous_actions).square().mean(dim=1),
        "time": q.new_full((n,), -dt * .10),
        "success": 20. * success.to(q.dtype),
        "physical_failure": -10. * physical_failure.to(q.dtype),
        "timeout": -5. * timeout.to(q.dtype),
    }
    total = torch.stack(list(components.values())).sum(dim=0)
    if not torch.isfinite(total).all():
        raise FloatingPointError("独立站立奖励出现非有限数值")
    return total, components
