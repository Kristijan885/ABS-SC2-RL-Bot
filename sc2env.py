import pickle
from types import NoneType

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pysc2.env import sc2_env
from pysc2.lib import features

from agent import SC2Agent

from actions import get_actions


class PySC2GymWrapper(gym.Env):
    def __init__(self, num_actions, map_name="BuildMarines", step_mul=8, visualize=False):
        super(PySC2GymWrapper, self).__init__()

        self.sc2_env = sc2_env.SC2Env(
            map_name=map_name,
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True
            ),
            step_mul=step_mul,
            visualize=visualize
        )
        print(self.sc2_env.observation_spec())
        self.action_space = spaces.Discrete(num_actions)
        minimap_shape = self.sc2_env.observation_spec()[0]["feature_minimap"]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=minimap_shape, dtype=np.float32
        )

        self.current_obs = None

    def reset(self, *args, **kwargs):
        #Used to allow any extra parameters to be passed but not carried through
        timestep = self.sc2_env.reset()  # or self.sc2_env.reset(**kwargs) if needed
        observation = self._process_observation(timestep[0])
        self.current_obs = observation
        info = {}  # Used to satisfy the format needs
        return observation, info

    def step(self, action):
        # Get the action step from the current observation and the action passed
        action_step = get_actions(self.current_obs, action)

        # Execute the action in the SC2 environment (no need to await it)
        timestep = self.sc2_env.step(action_step)

        # Process the new observation, reward, and done information
        observation = self._process_observation(timestep[0])
        self.current_obs = observation
        reward = timestep[0].reward
        done = timestep[0].last()
        if self.current_obs is None:
            print("Warning: Observations are None!")
            return observation, reward, done

        return observation, reward, done, {}

    def render(self, mode="human"):
        pass  # Use SC2's built-in render when initializing env

    def close(self):
        self.sc2_env.close()

    def _process_observation(self, timestep):
        # Extract and preprocess the minimap data
        minimap = timestep.observation["feature_minimap"]
        processed_minimap = minimap / 255.0  # Normalizing
        return processed_minimap
