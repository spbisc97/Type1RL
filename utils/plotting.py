import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from stable_baselines3.common.results_plotter import load_results

def plot_training_results(log_path, save_path=None, title='Training Results'):
    """
    Plot training results including rewards and episode lengths
    
    Args:
        log_path: Path to the training logs
        save_path: Path to save the plot (optional)
        title: Title for the plot
    """
    # Load results
    data = load_results(log_path)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot rewards
    ax1.plot(data.timesteps, data.rolling(window=50).mean()['r'], label='Reward')
    ax1.fill_between(
        data.timesteps,
        data.rolling(window=50).min()['r'],
        data.rolling(window=50).max()['r'],
        alpha=0.2
    )
    ax1.set_ylabel('Episode Reward')
    ax1.set_title(f'{title} - Moving Average (window=50)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot episode lengths
    ax2.plot(data.timesteps, data.rolling(window=50).mean()['l'], label='Length')
    ax2.fill_between(
        data.timesteps,
        data.rolling(window=50).min()['l'],
        data.rolling(window=50).max()['l'],
        alpha=0.2
    )
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Episode Length')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_evaluation_episode(rewards, cgm_values, actions, save_path=None, title='Evaluation Episode'):
    """
    Plot a single evaluation episode showing CGM values, rewards, and actions
    
    Args:
        rewards: List of rewards
        cgm_values: List of CGM values
        actions: List of actions (insulin doses)
        save_path: Path to save the plot (optional)
        title: Title for the plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Create time arrays for each plot
    time_cgm = np.arange(len(cgm_values))
    time_actions = np.arange(len(actions))
    time_rewards = np.arange(len(rewards))
    
    # Plot CGM values
    ax1.plot(time_cgm, cgm_values, label='CGM')
    ax1.axhline(y=70, color='r', linestyle='--', label='Hypoglycemia threshold')
    ax1.axhline(y=180, color='r', linestyle='--', label='Hyperglycemia threshold')
    ax1.set_ylabel('Blood Glucose (mg/dL)')
    ax1.set_title(f'{title} - CGM Values')
    ax1.grid(True)
    ax1.legend()
    
    # Plot rewards
    ax2.plot(time_rewards, rewards, label='Reward')
    ax2.set_ylabel('Reward')
    ax2.grid(True)
    ax2.legend()
    
    # Plot actions
    ax3.plot(time_actions, actions, label='Insulin dose')
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Insulin (U/min)')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_evaluation_metrics(eval_path, save_path=None):
    """
    Plot evaluation metrics over time
    
    Args:
        eval_path: Path to evaluation results
        save_path: Path to save the plot (optional)
    """
    # Load evaluation results
    results = pd.read_csv(eval_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot mean reward with std deviation
    ax.plot(results['timesteps'], results['mean_reward'], label='Mean Reward')
    ax.fill_between(
        results['timesteps'],
        results['mean_reward'] - results['std_reward'],
        results['mean_reward'] + results['std_reward'],
        alpha=0.2
    )
    
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Reward')
    ax.set_title('Evaluation Metrics Over Time')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show() 