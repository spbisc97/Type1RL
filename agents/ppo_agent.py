from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import torch
from pathlib import Path

from config import PPO_CONFIG, MODELS_DIR, LOGS_DIR

def create_ppo_agent(env, tensorboard_log=True):
    """
    Creates a PPO agent with the configuration from config.py
    
    Args:
        env: The gymnasium environment
        tensorboard_log: Whether to enable tensorboard logging
    
    Returns:
        PPO agent
    """
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        tensorboard_log=str(LOGS_DIR) if tensorboard_log else None,
        device=device,
        **PPO_CONFIG
    )
    
    return model

def train_ppo(
    env,
    total_timesteps,
    log_interval=100,
    eval_freq=1000,
    n_eval_episodes=3,
    save_path=None
):
    """
    Train the PPO agent with more frequent evaluations
    
    Args:
        env: Training environment
        total_timesteps: Number of training timesteps
        log_interval: Log interval for training info
        eval_freq: Evaluation frequency in timesteps
        n_eval_episodes: Number of episodes for evaluation
        save_path: Path to save the model (optional)
    
    Returns:
        Trained PPO model
    """
    
    # Create the agent
    model = create_ppo_agent(env)
    
    # Create eval callback with more frequent evaluations
    eval_callback = EvalCallback(
        env,
        best_model_save_path=str(MODELS_DIR / "ppo_best_model"),
        log_path=str(LOGS_DIR / "ppo_results"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        log_interval=log_interval
    )
    
    # Save the final model if path is provided
    if save_path:
        model.save(save_path)
    
    return model

def load_ppo(path):
    """
    Load a trained PPO agent
    
    Args:
        path: Path to the saved model
        
    Returns:
        Loaded PPO model
    """
    return PPO.load(path)

if __name__ == "__main__":
    from environment import make_env_mod
    
    # Create the gymnasium environment
    env = make_env_mod()
    
    # Train the PPO agent
    model = train_ppo(env, total_timesteps=100000)
    
    # Save the model
    model.save(MODELS_DIR / "ppo_model")
    
    # Close environment
    env.close()
