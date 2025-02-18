import argparse
from pathlib import Path
import gymnasium as gym

from environment import create_env, make_env_mod
from bergman_env import make_bergman_env
from agents.sac_agent import train_sac, load_sac
from agents.ppo_agent import train_ppo, load_ppo
from agents.td3_agent import train_td3, load_td3
from agents.dqn_agent import train_dqn, load_dqn
from config import MODELS_DIR, TOTAL_TIMESTEPS, RESULTS_DIR, LOGS_DIR
from utils.plotting import plot_evaluation_episode, plot_training_results

GCM_hist_len = 4

def parse_args():
    parser = argparse.ArgumentParser(description='Train or evaluate RL agents for T1D control')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                      help='Mode: train or evaluate')
    parser.add_argument('--agent', type=str, default='sac', 
                      choices=['sac', 'ppo', 'td3', 'dqn'],
                      help='Agent type: sac, ppo, td3, or dqn')
    parser.add_argument('--env', type=str, default='bergman',
                      choices=['simglucose', 'bergman'],
                      help='Environment: simglucose or bergman')
    parser.add_argument('--model-path', type=str, 
                      default=str(MODELS_DIR / 'model.zip'),
                      help='Path to save/load the model')
    parser.add_argument('--timesteps', type=int, 
                      default=TOTAL_TIMESTEPS,
                      help='Total timesteps for training')
    return parser.parse_args()

def create_env_by_type(env_type: str, discrete_actions: bool = False):
    """Create environment based on type"""
    if env_type == 'bergman':
        return make_bergman_env(CGM_hist_len=GCM_hist_len)
    else:  # simglucose
        return make_env_mod(CGM_hist_len=GCM_hist_len, discrete_actions=discrete_actions)

def train(args):
    """Train the selected agent"""
    env = create_env_by_type(args.env, discrete_actions=args.agent=='dqn')
    
    print(f"Starting training for {args.timesteps} timesteps...")
    print(f"Environment: {args.env}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    if args.agent == 'ppo':
        model = train_ppo(env, total_timesteps=args.timesteps, save_path=args.model_path)
    elif args.agent == 'td3':
        model = train_td3(env, total_timesteps=args.timesteps, save_path=args.model_path)
    elif args.agent == 'dqn':
        model = train_dqn(env, total_timesteps=args.timesteps, save_path=args.model_path)
    else:  # sac
        model = train_sac(env, total_timesteps=args.timesteps, save_path=args.model_path)
    
    env.close()
    print(f"Training completed. Model saved to {args.model_path}")
    
def evaluate(args):
    """Evaluate the trained agent"""
    env = create_env_by_type(args.env, discrete_actions=args.agent=='dqn')
    
    print(f"Loading model from {args.model_path}")
    if args.agent == 'ppo':
        model = load_ppo(args.model_path)
    elif args.agent == 'td3':
        model = load_td3(args.model_path)
    elif args.agent == 'dqn':
        model = load_dqn(args.model_path)
    else:  # sac
        model = load_sac(args.model_path)
    
    # Run evaluation episodes
    n_eval_episodes = 5
    episode_rewards = []
    
    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        # Lists to store episode data for plotting
        rewards = []
        cgm_values = []
        actions = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            cgm_values.append(obs[0])  # Store CGM value
            actions.append(action[0])     # Store action
            
            obs, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)        # Store reward
            episode_reward += reward
            done = done or truncated
            
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        # Plot the episode results
        plot_evaluation_episode(
            rewards=rewards,
            cgm_values=cgm_values,
            actions=actions,
            save_path=RESULTS_DIR / f"{args.agent}_{args.env}_episode_{episode+1}.png",
            title=f'{args.env} - Episode {episode+1}'
        )
    
    # Print and plot evaluation results
    mean_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"\nEvaluation over {n_eval_episodes} episodes:")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Min reward: {min(episode_rewards):.2f}")
    print(f"Max reward: {max(episode_rewards):.2f}")
    
    # Plot training results
    plot_training_results(
        log_path=LOGS_DIR / f"{args.agent}_results",
        save_path=RESULTS_DIR / f"{args.agent}_{args.env}_training_results.png",
        title=f'{args.agent.upper()} Training Results on {args.env}'
    )
    
    env.close()

def main():
    args = parse_args()
    
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)

if __name__ == "__main__":
    main()
