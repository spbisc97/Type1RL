from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import torch
from pathlib import Path

from config import DQN_CONFIG, MODELS_DIR, LOGS_DIR
from utils.callbacks import LearningRateScheduler

def create_dqn_agent(env, tensorboard_log=True):
    """
    Creates a DQN agent with the configuration from config.py
    
    Args:
        env: The gymnasium environment
        tensorboard_log: Whether to enable tensorboard logging
    
    Returns:
        DQN agent
    """
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the DQN agent
    model = DQN(
        "MlpPolicy",
        env,
        tensorboard_log=str(LOGS_DIR) if tensorboard_log else None,
        device=device,
        **DQN_CONFIG
    )
    
    return model

def train_dqn(
    env,
    total_timesteps,
    log_interval=100,
    eval_freq=1000,
    n_eval_episodes=3,
    save_path=None
):
    """Train the DQN agent with learning rate scheduling"""
    
    # Create the agent
    model = create_dqn_agent(env)
    
    # Create callbacks
    lr_scheduler = LearningRateScheduler(
        initial_lr=DQN_CONFIG["learning_rate"],
        min_lr=1e-5,
        decay_type='exponential',
        total_timesteps=total_timesteps,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path=str(MODELS_DIR / "dqn_best_model"),
        log_path=str(LOGS_DIR / "dqn_results"),
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

def load_dqn(path):
    """
    Load a trained DQN agent
    
    Args:
        path: Path to the saved model
        
    Returns:
        Loaded DQN model
    """
    return DQN.load(path)

if __name__ == "__main__":
    from environment import make_env_mod
    
    # Create the environment
    env = make_env_mod()
    
    # Train the DQN agent
    model = train_dqn(env, total_timesteps=100000)
    
    # Save the model
    model.save(MODELS_DIR / "dqn_model")
    
    # Close environment
    env.close() 