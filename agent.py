import random

import numpy
from gymnasium.spaces import Discrete

from stable_baselines3 import PPO

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from actions import get_actions


class SC2Agent(base_agent.BaseAgent):
    def __init__(self, model_path):
        super().__init__()
        self.model = PPO.load(model_path)

    def step(self, obs):
        action, _states = self.model.predict(obs)
        return get_actions(obs, action)
