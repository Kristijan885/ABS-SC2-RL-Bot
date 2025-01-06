from abc import ABC

import torch
from pysc2.lib import units
from stable_baselines3.common import policies

from build_marines_actions import *

from imitation.policies.base import NonTrainablePolicy

import numpy as np


class ExpertPolicy(policies.BasePolicy, ABC):
    def __init__(self, env):
        super().__init__(env.observation_space, env.action_space)
        self.env = env
        self.supply_threshold = 8  # Build supply depot when supply is near this
        self.barracks_threshold = 1  # Number of Barracks to aim for

    # Called by _predict in imitation and then predict in sb3
    def _predict(self, obs, deterministic=False):
        obs = self.env.current_obs.observation

        minerals = obs.player[1]
        supply_used = obs.player[3]
        supply_max = obs.player[4]
        barracks_count = get_unit_count(obs, units.Terran.Barracks)
        supply_depot_count = get_unit_count(obs, units.Terran.SupplyDepot)

        # 1. Build Supply Depot
        if supply_max - supply_used <= self.supply_threshold and minerals >= 100:
            coords = get_free_build_location(obs)
            if coords:
                action = np.array([[2, coords[0], coords[1]]])

        # 2. Build Barracks
        elif barracks_count < self.barracks_threshold and minerals >= 150:
            coords = get_free_build_location(obs)
            if coords:
                action = np.array([[1, coords[0], coords[1]]])

        # 3. Train Marines
        elif minerals >= 50 and barracks_count > 0:
            action = np.array([[0, 0, 0]])

        # 4. Select SCV for building tasks
        elif not is_scv_selected(obs):
            action = np.array([[0, 0, 0]]) # coords are irrelevant

        # 5. No-op if no other actions are valid
        else:
            action = np.array([[5, 0, 0]])

        return torch.from_numpy(action)


def get_unit_count(obs, unit_type):
    """Count the number of units of a specific type."""
    return len([unit for unit in obs.feature_units if unit.unit_type == unit_type])


def get_free_build_location(_):
    """Find a free location to build a structure (simplified)."""
    # Example: Just return a random location for simplicity
    return np.random.randint(0, 85), np.random.randint(0, 85)


def is_scv_selected(obs):
    """Check if an SCV is selected."""
    for unit in obs.feature_units:
        if unit.unit_type == units.Terran.SCV and unit.is_selected:
            return True
    return False
