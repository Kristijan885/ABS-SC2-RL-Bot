import pickle

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pysc2.env import sc2_env
from pysc2.lib import features

from actions import *


class PySC2GymWrapper(gym.Env):
    def __init__(self, num_actions, action_manager=ActionManager(), map_name="BuildMarines", step_mul=8, visualize=False):
        super(PySC2GymWrapper, self).__init__()

        self.sc2_env = sc2_env.SC2Env(
            map_name=map_name,
            players=[
                sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.easy)  # Second player (bot with race and difficulty)
            ],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True,
                crop_to_playable_area=True
                # raw_resolution=(84, 84),
                # use_raw_actions=True,
                # use_raw_units=True
            ),
            step_mul=step_mul,
            visualize=visualize,
            realtime=False
        )

        self.action_space = spaces.MultiDiscrete(num_actions)
        minimap_shape = self.sc2_env.observation_spec()[0]["feature_minimap"]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=minimap_shape, dtype=np.float32
        )

        self.current_obs = None
        self.action_manager = action_manager

    def reset(self, *args, **kwargs):
        timestep = self.sc2_env.reset()  # or self.sc2_env.reset(**kwargs) if needed
        observation = self._process_observation_to_minimap(timestep[0])
        info = {}  # Used to satisfy the format needs
        return observation, info

    def step(self, action):
        action_step = self.action_manager.get_actions(self.current_obs.observation, action)
        timestep = self.sc2_env.step(action_step)

        observation = self._process_observation_to_minimap(timestep[0])
        reward = timestep[0].reward
        done = timestep[0].last()

        return observation, reward, done, False, {}

    def render(self, mode="human"):
        pass  # Use SC2's built-in render when initializing env

    def close(self):
        self.sc2_env.close()

    def _process_observation_to_minimap(self, timestep):
        self.current_obs = timestep

        # Extract and preprocess the minimap data
        minimap = timestep.observation["feature_minimap"]
        processed_minimap = minimap / 255.0  # Normalizing
        return processed_minimap
