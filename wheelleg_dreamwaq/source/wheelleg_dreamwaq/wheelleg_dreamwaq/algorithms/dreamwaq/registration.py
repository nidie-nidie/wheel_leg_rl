from __future__ import annotations


def register_dreamwaq_rsl_rl_classes() -> None:
    """Register project classes in the module where RSL-RL 3.1.2 resolves ``class_name`` with eval."""

    import rsl_rl.runners.on_policy_runner as on_policy_runner

    from .actor_critic import DreamWaQActorCritic
    from .ppo import DreamWaQPPO

    on_policy_runner.DreamWaQActorCritic = DreamWaQActorCritic
    on_policy_runner.DreamWaQPPO = DreamWaQPPO
    if on_policy_runner.DreamWaQActorCritic is not DreamWaQActorCritic:
        raise RuntimeError("DreamWaQActorCritic registration failed")
    if on_policy_runner.DreamWaQPPO is not DreamWaQPPO:
        raise RuntimeError("DreamWaQPPO registration failed")

