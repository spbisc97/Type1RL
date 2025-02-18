# this file is a wrapper for the simglucose environment to make it compatible with stable-baselines3.

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.envs.simglucose_gym_env import T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario

# from simglucose.controller.base import Action
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.envs import register
from gymnasium.wrappers import TimeLimit
import datetime

# change here if you are really curious about the environment
# class T1DEnvironment(gym.Env):
#     """
#     Custom Environment that follows gym interface for T1D simulation.
#     Wraps the simglucose environment to make it compatible with stable-baselines3.
#     """

#     def __init__(self, patient_name="adolescent#002", seed=None, CGM_hist_len=2,start_time=None):
#         super().__init__()
#         if start_time is None:
#             start_time = datetime.datetime(2025, 2, 24, 0, 0)

#         # Create the T1D environment
#         patient = T1DPatient.withName('adolescent#001')
#         sensor = CGMSensor.withName('Dexcom', seed=1)
#         pump = InsulinPump.withName('Insulet')
#         scenario = RandomScenario(start_time=start_time,seed=1)

#         self.env = T1DSimEnv(patient, sensor, pump, scenario)

#         # Observation space includes CGM and CGM@t-1
#         self.observation_space = spaces.Box(
#             low=0,
#             high=np.inf,
#             shape=(2,),  # [CGM, CGM@t-1]
#             dtype=np.float32,
#         )
#         ub= self.env.pump._params["max_basal"]
#         self.action_space = spaces.Box(
#             low=0, high=ub, shape=(1,), dtype=np.float32
#         )
#         self.cgm_history = []

#     def _step(self, action):
#         # Convert normalized action to insulin dose
#         insulin_action = Action(basal=0, bolus=action)

#         # Take step in environment
#         next_state, reward, done, info = self.env.step(basal=action, bolus=0)

#         self.cgm_history.append(next_state.CGM)
#         self.cgm_history = self.cgm_history[-2:]


#         observation = np.array(
#             [self.cgm_history,  ], dtype=np.float32
#         )

#         return observation, reward, done, False, info

#     def reset(self, seed=None, options=None):
#         # Reset the environment
#         initial_step = self.env.reset()

#         # Process initial observation
#         cgm_value = initial_step.observation.CGM
#         self.cgm_history = [cgm_value, cgm_value]
#         observation = np.array(
#             [cgm_value, ], dtype=np.float32
#         )

#         return observation, {}

#     def close(self):
#         self.env.close()


# def make_env(patient_name="adolescent#002", seed=None):
#     #set start time at the beginning of 24 of feb 2025
#     start_time = datetime.datetime(2025, 2, 24, 0, 0)
#     """
#     Factory function to create the T1D environment
#     """
#     return T1DEnvironment(patient_name=patient_name, seed=seed)


class T1DSimGymnaisumMod(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    MAX_BG = 1000

    def __init__(
        self,
        patient_name=None,
        custom_scenario=None,
        reward_fun=None,
        seed=None,
        render_mode=None,
        CGM_hist_len=2,
        discrete_actions=False,
        n_discrete_actions=10,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.env = T1DSimEnv(
            patient_name=patient_name,
            custom_scenario=custom_scenario,
            reward_fun=reward_fun,
            seed=seed,
        )
        self.CGM_hist_len = CGM_hist_len
        self.discrete_actions = discrete_actions
        self.n_discrete_actions = n_discrete_actions

        # Observation space
        self.observation_space = spaces.Box(
            low=0, high=self.MAX_BG, shape=(CGM_hist_len,), dtype=np.float32
        )

        # Action space
        if discrete_actions:
            self.action_space = spaces.Discrete(n_discrete_actions)
            self.action_values = np.linspace(0, self.env.max_basal, n_discrete_actions)
        else:
            self.action_space = spaces.Box(
                low=0, high=self.env.max_basal, shape=(1,), dtype=np.float32
            )
        self.cgm_history = []

    def step(self, action):
        if self.discrete_actions:
            # Convert discrete action to continuous
            action = np.array([self.action_values[action]])

        obs, reward, done, info = self.env.step(action)
        truncated = False

        self.cgm_history.append(obs.CGM)
        self.cgm_history = self.cgm_history[-self.CGM_hist_len :]
        return (
            np.array(self.cgm_history, dtype=np.float32),
            reward,
            done,
            truncated,
            info,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, _, _, info = self.env._raw_reset()
        self.cgm_history = [obs.CGM] * self.CGM_hist_len
        return np.array(self.cgm_history, dtype=np.float32), info

    def render(self):
        if self.render_mode == "human":
            self.env.render()

    def close(self):
        print("Closing environment")
        self.env.close()


def custom_reward(BG_last_hour):
    """Modified reward function with smoother transitions"""
    bg = BG_last_hour[-1]

    # Base reward for being in range
    if 90 <= bg <= 150:
        reward = 2.0
    else:
        # Smooth penalty based on distance from target range
        distance_from_range = min(abs(bg - 70), abs(bg - 180))
        reward = -np.tanh(distance_from_range / 50.0)  # Smooth penalty

    # Additional severe penalty for dangerous levels
    if bg < 54 or bg > 250:
        reward -= 100.0

    return reward


def easy_reward(BG_last_hour):
    """Simple reward function with a penalty for being out of range"""
    bg = BG_last_hour[-1]

    # Base reward for being in range
    if 90 <= bg <= 150:
        reward = 2.0
    else:
        reward = -1.0  # Penalty for being out of range

    return reward


def make_env_mod(CGM_hist_len=2, discrete_actions=False):
    """Factory function to create the T1D environment"""
    register(
        id="simglucose_mod/adolescent2-v0",
        entry_point="environment:T1DSimGymnaisumMod",
        kwargs={
            "patient_name": "adolescent#002",
            "CGM_hist_len": CGM_hist_len,
            "reward_fun": easy_reward,
            "discrete_actions": discrete_actions,
        },
    )
    env = gym.make("simglucose_mod/adolescent2-v0")
    env = Monitor(env)
    # Limit the number of steps per episode
    env = TimeLimit(env, max_episode_steps=288)

    # Wrap environment in VecEnv
    env = DummyVecEnv([lambda: env])
    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    return env


def create_env():
    register(
        id="simglucose/adolescent2-v0",
        entry_point="simglucose.envs:T1DSimGymnaisumEnv",  #        max_episode_steps=10,
        kwargs={"patient_name": "adolescent#002"},
    )
    env = gym.make("simglucose/adolescent2-v0")
    env = Monitor(env)  # Logs episode statistics
    # env = DummyVecEnv([lambda: env])  # Wraps the environment for compatibility
    return env


if __name__ == "__main__":
    env = make_env_mod()
    env.close()
    # write a test to check if the environment is working as expected
    env = make_env_mod()
    obs = env.reset()
    print("Initial observation:", obs)
    action = env.action_space.sample()
    print("Sampled action:", action)
    obs, reward, done, _, _ = env.step(action)
    print("Next observation:", obs)
    print("Reward:", reward)
    print("Done:", done)
    env.close()
