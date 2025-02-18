from pathlib import Path
from torch.nn.modules.activation import ReLU

# Project paths
PROJECT_ROOT = Path(__file__).parent
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
LOGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Environment settings
ENV_ID = "simglucose-adolescent2-v0"
SIMULATION_TIME = 24  # hours
SAMPLE_TIME = 5  # minutes

# Common settings
SEED = 42
TOTAL_TIMESTEPS = 1_000_000
GAMMA = 0.99

# PPO specific settings
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 32,
    "n_epochs": 10,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,
    "policy_kwargs": dict(
        net_arch=dict(pi=[64, 64], qf=[64, 64])
    )
}

# DQN specific settings
DQN_CONFIG = {
    "learning_rate": 1e-4,
    "buffer_size": 100000,
    "learning_starts": 5000,
    "batch_size": 32,
    "target_update_interval": 100,
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "policy_kwargs": dict(
        net_arch=[64, 64]
    )
}

# SAC specific settings
SAC_CONFIG = {
    "learning_rate": 1e-4,
    "buffer_size": 100000,
    "learning_starts": 5000,
    "batch_size": 64,
    "tau": 0.005,
    "gamma": 0.95,
    "ent_coef": "auto",
    "target_entropy": "auto",
    "train_freq": (1, "episode"),
    "gradient_steps": -1,
    "policy_kwargs": dict(
        activation_fn=ReLU,
        net_arch=dict(pi=[64, 64], qf=[64, 64])
    )
}

# TD3 specific settings
TD3_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 100000,
    "learning_starts": 1000,
    "batch_size": 32,
    "tau": 0.005,
    "gamma": 0.95,
    "train_freq": (1, "episode"),
    "gradient_steps":-1,
    "policy_delay": 2,  # TD3 specific
    "target_policy_noise": 0.2,  # TD3 specific
    "target_noise_clip": 0.5,    # TD3 specific
    "policy_kwargs": dict(
        net_arch=dict(pi=[128, 64], qf=[128, 64])
    )
}

# Evaluation settings
N_EVAL_EPISODES = 10
DETERMINISTIC_EVAL = True 