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
    """
    Bergman minimal model of glucose-insulin dynamics.
    The model is described by the following differential equations:
    
    dG/dt = -p1 * G - (G + G_b) * X + D
    dX/dt = -p2 * X + p3 * I
    dI/dt = U - n * I
    
    where:
    - G is the glucose concentration in mmol/L
    - X is the insulin effect (arbitrary units)??
    - I is the plasma insulin concentration in mU/L
    - D is the meal disturbance (carbs in grams)?
    - U is the insulin infusion rate (mU/min)??
    - p1, p2, p3, G_b, and n are model parameters 
    
    -p1 is in 1/min? and is the glucose effectiveness? 
    -p2 is in 1/min? and is the insulin sensitivity?
    -p3 is in 1/min? and is the insulin effectiveness?
    -G_b is the basal glucose concentration in mmol/L
    -n is the insulin clearance rate in 1/min
    
    # TODO: Check units and ranges of parameters 
    # TODO: Use better equations
    
    
    # maybe this could be converted to numpy?
    """
    
    def __init__(self, p1=2.3e-6, p2=0.088, p3=0.63e-3, G_b=50, n=0.09):
        # Using parameters from literature that are more numerically stable
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.G_b = G_b
        self.n = n
        self.range = 100.0  # Range for clipping

    def forward(self, t, state, D=0, U=0):
        r=self.range
        # Ensure all inputs are tensors and clip to prevent instability
        G = torch.clamp(state[0], min=-r, max=r)  # Glucose in mmol/L
        X = torch.clamp(state[1], min=-r, max=r)   # Insulin effect
        I = torch.clamp(state[2], min=-r, max=r)  # Plasma insulin in mU/L
        # Ensure all inputs are tensors 

        
        
        
        # Convert meal disturbance and insulin input to tensors
        D = torch.as_tensor(D, dtype=torch.float32)
        U = torch.as_tensor(U, dtype=torch.float32)
        
        # Calculate derivatives with clipping to prevent extreme values
        dGdt = torch.clamp(-self.p1 * G - (G + self.G_b) * X + D, min=-r, max=r)
        dXdt = torch.clamp(-self.p2 * X + self.p3 * I, min=-r, max=r)
        dIdt = torch.clamp(U - self.n * I, min=-r, max=r)
        
        
        
        # dimensionality analysis
        # dGdt = -p1 * G - (G + G_b) * X + D
        # dXdt = -p2 * X + p3 * I
        # dIdt = U - n * I
        
        
        
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
        initial_glucose_range: Tuple[float, float] = (-10, 10),  # Initial glucose range
        CGM_hist_len: int = 4,  # Number of glucose measurements to keep
        meal_schedule: Optional[list] = None
    ):
        super().__init__()
        
        # Initialize Bergman model
        self.model = BergmanTrueDynamics()
        self.range = self.model.range
        
        # Time settings
        self.dt = dt  # Time step (minutes)
        # total simulation time is 24 hours (1440 minutes)
        # might this be moved outside of the class?
        
        self.current_time = 0
        
        # Glucose history settings
        self.CGM_hist_len = CGM_hist_len
        self.glucose_history = deque(maxlen=CGM_hist_len+2) # Keep extra for reward calculation
        
        # Define observation space (glucose history)
        self.observation_space = spaces.Box(
            low=-self.range,  # Minimum range glucose value
            high=self.range,  # Maximum range glucose value
            shape=(CGM_hist_len,),
            dtype=np.float32
        )
        
        # Define action space (insulin infusion rate)
        self.action_space = spaces.Box(
            low=np.array([0.0]),
            high=np.array([10.0]),  # Reduced from 100 to 1
            dtype=np.float32
        )
        
        # Initial glucose range
        self.initial_glucose_range = initial_glucose_range
        
        # Meal schedule (time in minutes, carbs in grams)
        self.default_meals = [
            (7 * 60, 5),    # Breakfast at 7 AM
            (12 * 60, 5),   # Lunch at 12 PM
            (18 * 60, 5),   # Dinner at 6 PM
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
                D += meal_size / self.dt  # Convert carbs to glucose rate (mmol/L per minute)
        return D
    
    def _compute_reward(self, glucose_history: float) -> float:
        """
        Compute reward based on blood glucose level
        Converting from mmol/L to mg/dL (multiply by 18)
        Ideal range: 70-180 mg/dL (3.9-10 mmol/L)
        
        Remember that the system is centered around 0, so the ideal range is -70 to 110
        """
        # glucose = glucose * 18  # Convert to mg/dL maybe?
        

        

        
        # reward is the negative of the glucose value
        reward =0 
        #reward for being in the ideal range
        reward += -np.log(np.abs(glucose_history[-1])+ 1) +np.log(4)
        #reward for derivative of glucose being zero
        reward += -np.log(np.abs(glucose_history[-1]-glucose_history[-2])+1) 
        return reward
    
            
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Scale meal disturbance and action to appropriate ranges?
        D = self._get_meal_disturbance(self.current_time)  # Convert to mmol/L ??
        action = action  # Scale action to reasonable insulin range
        
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
        # Clamp glucose to prevent instability
        # self.state[0] = np.clip(self.state[0], -self.range, self.range)
        # self.state[1] = np.clip(self.state[1], -self.range, self.range)
        # self.state[2] = np.clip(self.state[2], -self.range, self.range)
        
        # Check for termination conditions
        terminated = False
        truncated = False
        
        # Terminate if glucose is out of safe range
        if abs(self.state[0]) >= 80:
            terminated = True
            
        # if self.current_time >= 1440:  # 24 hours
        #     terminated = True
        #     truncated = True
        
        # Update glucose history
        self.glucose_history.append(self.state[0])
        
        # Increment time
        self.current_time += self.dt
        
        # Calculate reward
        reward = self._compute_reward(self.glucose_history)
        
        # Additional info
        info = {
            "glucose": self.state[0],
            "insulin_effect": self.state[1],
            "plasma_insulin": self.state[2],
            "time": self.current_time
        }
        
        # Return glucose history as observation
        glucose_history_obs = np.array(list(self.glucose_history)[-self.CGM_hist_len:], dtype=np.float32)
        return glucose_history_obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        # Initialize with more reasonable values
        initial_glucose = np.random.uniform(self.initial_glucose_range[0], self.initial_glucose_range[1])
        self.state = np.array([
            initial_glucose,
            0.0,
            0.0
        ], dtype=np.float32)
        
        # Initialize glucose history
        self.glucose_history.clear()
        for _ in range(self.CGM_hist_len + 2):  # +2 for reward calculation
            self.glucose_history.append(initial_glucose)
        
        self.current_time = 0
        
        # Convert deque to numpy array for observation
        glucose_history_obs = np.array(list(self.glucose_history)[-self.CGM_hist_len:], dtype=np.float32)
        return glucose_history_obs, {}
    
    def render(self):
        pass

def make_bergman_env(CGM_hist_len=4):
    """Factory function to create the Bergman environment"""
    env = BergmanEnv(CGM_hist_len=CGM_hist_len)
    
        # Add time limit - this will set truncated=True when max steps reached
    # TODO there is a problem with this wrapper!!!! needs to be before the monitor
    env = TimeLimit(env, max_episode_steps=288)  # 24 hours with 5-min steps 
    
    # Add Monitor wrapper
    env = Monitor(env)
    

    
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

def main():
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
        
# test the equilibrium point of the system
# the equilibrium point is when the system is at rest
# the system is at rest when the derivatives are zero
def test_equilibrium():
    model = BergmanTrueDynamics()
    # Initial state (G, X, I)
    state = torch.tensor([50, 0, 0], dtype=torch.float32)
    D = 0
    U = 0
    print("Initial state:", state)
    print("Initial derivatives:", model(0, state, D, U))
    
    # Find equilibrium point
    lr = 1
    max_iters = 10_000_000
    
    for i in range(max_iters):
        # Calculate derivatives
        derivatives = model(0, state, D, U)
        
        # Update state
        state = state + lr * derivatives # Euler update
        if i % 10000 == 0:
            print("Iteration:", i)
            print("State:", state)
            print("Derivatives:", derivatives)
        
        # Check for convergence
        if torch.all(torch.abs(derivatives) < 1e-8):
            print("Converged at iteration", i)
            break
if __name__ == "__main__":
    main()