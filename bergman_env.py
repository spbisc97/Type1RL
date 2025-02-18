import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from collections import deque

class BergmanTrueDynamics(nn.Module):
    def __init__(self, p1=0.0337, p2=0.0209, p3=0.0000876, G_b=4.5, n=0.2659):
        # Using parameters from literature that are more numerically stable
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.G_b = G_b
        self.n = n

    def forward(self, t, state, D=0, U=0):
        # Ensure all inputs are tensors and clip to prevent instability
        G = torch.clamp(state[0], min=0.0, max=25.0)  # Glucose in mmol/L
        X = torch.clamp(state[1], min=0.0, max=1.0)   # Insulin effect
        I = torch.clamp(state[2], min=0.0, max=10.0)  # Plasma insulin in mU/L
        
        # Convert meal disturbance and insulin input to tensors
        D = torch.as_tensor(D, dtype=torch.float32)
        U = torch.as_tensor(U, dtype=torch.float32)
        
        # Calculate derivatives with clipping to prevent extreme values
        dGdt = torch.clamp(-self.p1 * G - (G + self.G_b) * X + D, min=-10.0, max=10.0)
        dXdt = torch.clamp(-self.p2 * X + self.p3 * I, min=-1.0, max=1.0)
        dIdt = torch.clamp(U - self.n * I, min=-5.0, max=5.0)
        
        return torch.stack([dGdt, dXdt, dIdt])

class BergmanEnv(gym.Env):
    """
    Gymnasium environment for the Bergman minimal model of glucose-insulin dynamics.
    
    State: Last n glucose measurements
    Action: [U] - Insulin infusion rate
    """
    
    def __init__(
        self,
        dt: float = 5.0,  # 5 minutes
        simulation_time: float = 24.0 * 60,  # 24 hours in minutes
        CGM_hist_len: int = 4,  # Number of glucose measurements to keep
        meal_schedule: Optional[list] = None
    ):
        super().__init__()
        
        # Initialize Bergman model
        self.model = BergmanTrueDynamics()
        
        # Time settings
        self.dt = dt  # Time step (minutes)
        self.simulation_time = simulation_time
        self.current_time = 0
        
        # Glucose history settings
        self.CGM_hist_len = CGM_hist_len
        self.glucose_history = deque(maxlen=CGM_hist_len)
        
        # Define observation space (glucose history)
        self.observation_space = spaces.Box(
            low=0,
            high=500,  # Maximum reasonable glucose value
            shape=(CGM_hist_len,),
            dtype=np.float32
        )
        
        # Define action space (insulin infusion rate)
        self.action_space = spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),  # Reduced from 100 to 1
            dtype=np.float32
        )
        
        # Meal schedule (time in minutes, carbs in grams)
        self.default_meals = [
            (7 * 60, 45),    # Breakfast at 7 AM
            (12 * 60, 75),   # Lunch at 12 PM
            (18 * 60, 85),   # Dinner at 6 PM
        ]
        self.meal_schedule = meal_schedule if meal_schedule is not None else self.default_meals
        
        # Initialize state
        self.state = None
        
    def _get_meal_disturbance(self, time: float) -> float:
        """Calculate meal disturbance at current time"""
        D = 0.0
        for meal_time, meal_size in self.meal_schedule:
            # Simple meal absorption model
            if abs(time - meal_time) < self.dt:
                D += meal_size / self.dt  # Convert carbs to glucose rate
        return D
    
    def _compute_reward(self, glucose: float) -> float:
        """
        Compute reward based on blood glucose level
        Converting from mmol/L to mg/dL (multiply by 18)
        Ideal range: 70-180 mg/dL (3.9-10 mmol/L)
        """
        glucose = glucose * 18  # Convert to mg/dL
        
        if 70 <= glucose <= 180:
            return 1.0
        elif glucose < 70:
            return -((70 - glucose) / 70) ** 2  # Quadratic penalty for hypoglycemia
        else:
            return -((glucose - 180) / 180) ** 2  # Quadratic penalty for hyperglycemia
            
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Scale meal disturbance and action to appropriate ranges
        D = self._get_meal_disturbance(self.current_time) / 18  # Convert to mmol/L
        action = action * 0.1  # Scale action to reasonable insulin range
        
        # Convert state and action to tensors
        state_tensor = torch.as_tensor(self.state, dtype=torch.float32)
        action_tensor = torch.as_tensor(action[0], dtype=torch.float32)
        
        # Get meal disturbance
        D = self._get_meal_disturbance(self.current_time)
        
        # Simulate one step using RK4 method
        k1 = self.model(self.current_time, state_tensor, D, action_tensor)
        k2 = self.model(self.current_time + self.dt/2, state_tensor + k1*self.dt/2, D, action_tensor)
        k3 = self.model(self.current_time + self.dt/2, state_tensor + k2*self.dt/2, D, action_tensor)
        k4 = self.model(self.current_time + self.dt, state_tensor + k3*self.dt, D, action_tensor)
        
        # Update internal state
        self.state = (state_tensor + (k1 + 2*k2 + 2*k3 + k4) * self.dt/6).numpy()
        
        # Update glucose history
        self.glucose_history.append(self.state[0])
        
        # Increment time
        self.current_time += self.dt
        
        # Calculate reward
        reward = self._compute_reward(self.state[0])
        
        # Check if episode is done
        done = self.current_time >= self.simulation_time
        
        # Additional info
        info = {
            "glucose": self.state[0],
            "insulin_effect": self.state[1],
            "plasma_insulin": self.state[2],
            "time": self.current_time
        }
        
        # Return glucose history as observation
        return np.array(self.glucose_history), reward, done, False, info
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        # Initialize with more reasonable values
        initial_glucose = np.random.uniform(4.0, 10.0)  # In mmol/L (roughly 70-180 mg/dL)
        self.state = np.array([
            initial_glucose,
            0.0,  # Initial insulin effect
            0.0   # Initial plasma insulin
        ], dtype=np.float32)
        
        # Initialize glucose history
        self.glucose_history.clear()
        for _ in range(self.CGM_hist_len):
            self.glucose_history.append(initial_glucose)
        
        self.current_time = 0
        
        return np.array(self.glucose_history), {}
    
    def render(self):
        pass

def make_bergman_env(CGM_hist_len=4):
    """Factory function to create the Bergman environment"""
    env = BergmanEnv(CGM_hist_len=CGM_hist_len)
    
    # Add Monitor wrapper
    env = Monitor(env)
    
    # Add time limit
    env = TimeLimit(env, max_episode_steps=1440)  # 24 hours with 5-min steps
    
    # Wrap in DummyVecEnv and VecNormalize
    # env = DummyVecEnv([lambda: env])
    # env = VecNormalize(
    #     env,
    #     norm_obs=True,
    #     norm_reward=True,
    #     clip_obs=10.0,
    #     gamma=0.99
    # )
    
    return env

if __name__ == "__main__":
    # Test the environment
    env = BergmanEnv()
    obs, _ = env.reset()
    print("Initial observation (glucose history):", obs)
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(f"Step {i}:")
        print(f"  Action: {action}")
        print(f"  Observation: {obs}")
        print(f"  Reward: {reward}")
        print(f"  Current glucose: {info['glucose']:.1f} mg/dL")
        
        if done:
            break 