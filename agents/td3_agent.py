from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import torch
from pathlib import Path

from config import TD3_CONFIG, MODELS_DIR, LOGS_DIR
from utils.callbacks import LearningRateScheduler

def create_td3_agent(env, tensorboard_log=True):
    """
    Creates a TD3 agent with the configuration from config.py
    
    Args:
        env: The gymnasium environment
        tensorboard_log: Whether to enable tensorboard logging
    
    Returns:
        TD3 agent
    """
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create action noise for exploration
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.3 * np.ones(n_actions)
    )
    
    # Create the TD3 agent
    model = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        tensorboard_log=str(LOGS_DIR) if tensorboard_log else None,
        device=device,
        **TD3_CONFIG
    )
    
    return model

def train_td3(
    env,
    total_timesteps,
    log_interval=100,
    eval_freq=1000,
    n_eval_episodes=3,
    save_path=None
):
    """Train the TD3 agent with learning rate scheduling"""
    
    # Create the agent
    model = create_td3_agent(env)
    
    # Create callbacks
    lr_scheduler = LearningRateScheduler(
        initial_lr=TD3_CONFIG["learning_rate"],
        min_lr=1e-5,
        decay_type='exponential',
        total_timesteps=total_timesteps,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path=str(MODELS_DIR / "td3_best_model"),
        log_path=str(LOGS_DIR / "td3_results"),
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

def load_td3(path):
    """
    Load a trained TD3 agent
    
    Args:
        path: Path to the saved model
        
    Returns:
        Loaded TD3 model
    """
    return TD3.load(path)

if __name__ == "__main__":
    from environment import make_env_mod
    
    # Create the environment
    env = make_env_mod()
    
    # Train the TD3 agent
    model = train_td3(env, total_timesteps=100000)
    
    # Save the model
    model.save(MODELS_DIR / "td3_model")
    
    # Close environment
    env.close() 