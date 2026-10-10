from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DreamWaQActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "DreamWaQActorCritic"
    init_noise_std = 1.0
    noise_std_type = "scalar"
    state_dependent_std = False
    actor_obs_normalization = False
    critic_obs_normalization = False
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    activation = "elu"
    current_obs_group = "policy"
    history_obs_group = "policy_history"
    history_length = 5
    current_obs_dim = 25
    history_obs_dim = 125
    critic_obs_dim = 41
    velocity_dim = 3
    context_dim = 16
    reconstruction_target_dim = 16
    cenet_encoder_hidden_dims = [128, 64]
    cenet_decoder_hidden_dims = [64, 128]
    actor_context_mode = "deterministic_context_mu"
    action_mean_clip = 20.0
    action_std_min = 1.0e-4
    action_std_max = 2.0


@configclass
class DreamWaQPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "DreamWaQPPO"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0
    normalize_advantage_per_mini_batch = False
    rnd_cfg = None
    symmetry_cfg = None
    velocity_coef = 1.0
    reconstruction_coef = 1.0
    kl_beta = 1.0
    strict_finite_checks = True
    context_mu_min_feature_std = 1.0e-3
    context_mu_min_initial_std_ratio = 0.10
    context_monitor_final_window_iterations = 10
    velocity_mse_max_zero_baseline_ratio = 0.80
    velocity_mse_denominator_floor = 1.0e-8


@configclass
class WheelLegFlatDreamWaQRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "DreamWaQContractOnPolicyRunner"
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 100
    experiment_name = "wheelleg_flat_dreamwaq"
    run_name = ""
    logger = "tensorboard"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0
    policy = DreamWaQActorCriticCfg()
    algorithm = DreamWaQPPOAlgorithmCfg()

