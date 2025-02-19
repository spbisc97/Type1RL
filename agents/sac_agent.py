from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np
import torch
from pathlib import Path
#from the environment.py file

from config import SAC_CONFIG, MODELS_DIR, LOGS_DIR
from utils.callbacks import LearningRateScheduler

def create_sac_agent(env, tensorboard_log=True):
    """
    Creates a SAC agent with the configuration from config.py
    
    Args:
        env: The gymnasium environment
        tensorboard_log: Whether to enable tensorboard logging
    
    Returns:
        SAC agent
    """
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create action noise for exploration
    n_actions = env.action_space.shape[0]
    # action_noise = NormalActionNoise(
    #     mean=np.zeros(n_actions),
    #     sigma=0.1 * np.ones(n_actions)
    # )
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )
    
    # Create the SAC agent
    model = SAC(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        tensorboard_log=str(LOGS_DIR) if tensorboard_log else None,
        device=device,
        **SAC_CONFIG
    )
    
    return model

def train_sac(
    env,
    total_timesteps,
    log_interval=100,
    eval_freq=1000,
    n_eval_episodes=3,
    save_path=None
):
    """Train the SAC agent with learning rate scheduling"""
    
    # Create the agent
    model = create_sac_agent(env)
    
    # Create callbacks
    lr_scheduler = LearningRateScheduler(
        initial_lr=SAC_CONFIG["learning_rate"],
        min_lr=1e-5,
        decay_type='exponential',
        total_timesteps=total_timesteps,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path=str(MODELS_DIR / "sac_best_model"),
        log_path=str(LOGS_DIR / "sac_results"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Combine callbacks
    callbacks = [eval_callback, lr_scheduler]
    
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=log_interval
    )
    
    if save_path:
        model.save(save_path)
    
    return model

def load_sac(path):
    """
    Load a trained SAC agent
    
    Args:
        path: Path to the saved model
        
    Returns:
        Loaded SAC model
    """
    return SAC.load(path)

if __name__ == "__main__":
    from environment import make_env_mod
    
    # Create the environment
    env = make_env_mod()
    
    # Train the SAC agent
    model = train_sac(env, total_timesteps=100000)
    
    # Save the model
    model.save(MODELS_DIR / "sac_model")
    
    