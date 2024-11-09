import pickle

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pysc2.env import sc2_env
from pysc2.lib import actions

from agent import SC2Agent

from actions import get_actions


class PySC2GymWrapper(gym.Env):
    def __init__(self, num_actions, map_name="BuildMarines", step_mul=8, visualize=False):
        super(PySC2GymWrapper, self).__init__()

        self.sc2_env = sc2_env.SC2Env(
            map_name=map_name,
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            # agent_interface_format=sc2_env.parse_agent_interface_format(
            #     feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64)
            # ),
            step_mul=step_mul,
            # game_steps_per_episode=0,
            visualize=visualize
        )

        self.action_space = spaces.Discrete(num_actions)
        minimap_shape = self.sc2_env.observation_spec()[0]["minimap"].shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=minimap_shape, dtype=np.float32
        )

        self.current_obs = None

    def reset(self):
        timestep = self.sc2_env.reset()
        observation = self._process_observation(timestep[0])
        self.current_obs = observation
        return observation

    async def step(self, action):
        action_step = get_actions(self.current_obs, action)

        # Execute the action in the SC2 environment
        timestep = self.sc2_env.step(action_step)

        # Process the new observation, reward, and done information
        observation = self._process_observation(timestep[0])
        self.current_obs = observation
        reward = timestep[0].reward
        done = timestep[0].last()

        return observation, reward, done, {}

    def render(self, mode="human"):
        pass  # Use SC2's built-in render when initializing env

    def close(self):
        self.sc2_env.close()

    def _process_observation(self, timestep):
        # Extract and preprocess the minimap data
        minimap = timestep.observation["minimap"]
        processed_minimap = minimap / 255.0  # Normalizing
        return processed_minimap
