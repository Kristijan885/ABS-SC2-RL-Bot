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
        self.last_barracks = (10, 10)
        self.last_depot = (83, 83)

    # Called by _predict in imitation and then predict in sb3
    def _predict(self, obs, deterministic=False):
        obs = self.env.current_obs.observation

        minerals = obs.player[1]
        supply_used = obs.player[3]
        supply_max = obs.player[4]
        barracks_count = get_unit_count(obs, units.Terran.Barracks)
        supply_depot_count = get_unit_count(obs, units.Terran.SupplyDepot)
        barracks = get_feature_units_by_tye(obs, units.Terran.Barracks)
        barracks = [barrack for barrack in barracks if
                    barrack.build_progress == 100 and barrack.order_length < 5]

        # No op
        action = np.array([[5, 0, 0]])

        # Build Supply Depot
        if supply_max - supply_used <= self.supply_threshold and minerals >= 100:
            coords = (self.last_depot[0], self.last_depot[1] - 5)
            if coords[0] < 0:
                coords = (coords[0] - 5, 84)
            action = np.array([[2, coords[0], coords[1]]])
            self.last_depot = coords

        # Train marine
        elif is_unit_type_selected(obs, units.Terran.Barracks) and minerals >= 50:
            action = np.array([[4, 0, 0]])

        # Select barrack
        elif minerals >= 50 and len(barracks) > 0:
            coords = (barracks[0].x, barracks[0].y)
            action = np.array([[3, coords[0], coords[1]]])

        # Select SCV for building tasks
        elif not is_unit_type_selected(obs, units.Terran.SCV):
            action = np.array([[0, 0, 0]])  # coords are irrelevant

        # Build Barracks
        elif minerals >= 150:
            coords = (self.last_barracks[0] + 10, self.last_barracks[1])
            if coords[0] > 83:
                coords = (10, coords[1] + 10)
            action = np.array([[1, coords[0], coords[1]]])
            self.last_barracks = coords

        return torch.from_numpy(action)


def get_feature_units_by_tye(obs, unit_type):
    feature_units = [unit for unit in obs.feature_units if unit.unit_type == unit_type]
    return feature_units


def get_unit_count(obs, unit_type):
    """Count the number of units of a specific type."""
    return len([unit for unit in obs.feature_units if unit.unit_type == unit_type])


def get_free_build_location(_):
    """Find a free location to build a structure (simplified)."""
    # Example: Just return a random location for simplicity
    return np.random.randint(0, 85), np.random.randint(0, 85)


def is_unit_type_selected(obs, unit_type):
    """Check if an SCV is selected."""
    for unit in obs.feature_units:
        if unit.unit_type == unit_type and unit.is_selected:
            return True
    return False
