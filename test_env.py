import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np

def make_test_env(env_id="LunarLander-v3", continuous=True):
    """
    Create a test environment to verify algorithm implementations
    
    Args:
        env_id: Gymnasium environment ID
        continuous: Whether to use continuous or discrete action space
    """
    if continuous:
        env_id = "LunarLanderContinuous-v3"
    
    # Create vectorized environment
    def make_env():
        env = gym.make(env_id)
        env = Monitor(env)
        return env
    
    # env = DummyVecEnv([make_env])
    # env = VecNormalize(
    #     env,
    #     norm_obs=True,
    #     norm_reward=True,
    #     clip_obs=10.0,
    #     gamma=0.99
    # )
    env = make_env()
    
    return env

def test_env(env):
    """Test if environment works correctly"""
    obs = env.reset()
    done = False
    total_reward = 0
    
    for _ in range(10):
        action = env.action_space.sample()
        print(f"Action: {action}")
        obs, reward, done,trunc, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done} Truncated: {trunc}, Info: {info}")
        total_reward += reward
        if done:
            break
    
    print(f"Environment test completed. Total reward: {total_reward}")
    return True

if __name__ == "__main__":
    from agents.sac_agent import train_sac
    from agents.ppo_agent import train_ppo
    from agents.td3_agent import train_td3
    from agents.dqn_agent import train_dqn
    from config import MODELS_DIR
    
    # Test each algorithm separately to isolate issues
    try:
        # Test SAC
        print("Training SAC...")
        env = make_test_env(continuous=True)
        test_env(env)  # Test environment first
        sac_model = train_sac(
            env, 
            total_timesteps=100_000,
            save_path=MODELS_DIR / "sac_lunar_test.zip"
        )
        env.close()
    except Exception as e:
        print(f"SAC training failed: {e}")
    
    try:
        # Test PPO
        print("\nTraining PPO...")
        env = make_test_env(continuous=True)
        test_env(env)  # Test environment first
        ppo_model = train_ppo(
            env, 
            total_timesteps=100_000,
            save_path=MODELS_DIR / "ppo_lunar_test.zip"
        )
        env.close()
    except Exception as e:
        print(f"PPO training failed: {e}")
    
    try:
        # Test TD3
        print("\nTraining TD3...")
        env = make_test_env(continuous=True)
        test_env(env)  # Test environment first
        td3_model = train_td3(
            env, 
            total_timesteps=100_000,
            save_path=MODELS_DIR / "td3_lunar_test.zip"
        )
        env.close()
    except Exception as e:
        print(f"TD3 training failed: {e}")
    
    try:
        # Test DQN
        print("\nTraining DQN...")
        env = make_test_env(continuous=False)
        test_env(env)  # Test environment first
        dqn_model = train_dqn(
            env, 
            total_timesteps=100_000,
            save_path=MODELS_DIR / "dqn_lunar_test.zip"
        )
        env.close()
    except Exception as e:
        print(f"DQN training failed: {e}") 