from abc import ABC

import torch
from stable_baselines3.common import policies

from build_marines_actions import *

DEPOT_CHASE_BUILD_ID = 222
BARRACK_CHASE_BUILD_ID = 185
DEPOT_BUILD_ORDER_ID = 362

BUILD_ORDERS = [DEPOT_CHASE_BUILD_ID, BARRACK_CHASE_BUILD_ID, DEPOT_BUILD_ORDER_ID]

DEPOT_DIAMETER = 8
BARRACK_DIAMETER = 12
DEPOT_RADIUS = 4
BARRACK_RADIUS = 6


class ExpertPolicy(policies.BasePolicy, ABC):
    def __init__(self, env):
        super().__init__(env.observation_space, env.action_space)
        self.env = env
        self.supply_threshold = 5  # Threshold to build a supply depot
        self.last_barracks = (10, 10)
        self.last_depot = (64, 64)  # Map height is 64
        self.expected_barrack_count = 0
        self.expected_depot_count = 0

    def _predict(self, obs, deterministic=False):
        observation = self.env.current_obs.observation

        minerals = observation.player[1]
        supply_used = observation.player[3]
        supply_max = observation.player[4]
        barracks = get_units_by_type(observation, units.Terran.Barracks)
        available_barracks = [b for b in barracks if b.build_progress == 100 and b.order_length < 5]

        # Default action no op
        action = no_op_action()

        if self.should_build_supply_depot(observation, minerals, supply_used, supply_max):
            action = self.build_supply_depot(observation)

        elif self.should_train_marine(observation, minerals, supply_max, supply_used):
            action = train_marine_action()

        elif self.should_select_barracks(minerals, supply_max, supply_used, available_barracks):
            action = select_barracks_action(available_barracks[0])

        elif not is_unit_selected(observation, units.Terran.SCV):
            action = select_scv_action()

        elif self.should_build_barracks(observation, minerals):
            action = self.build_barracks(observation)

        return torch.from_numpy(action)

    def should_build_supply_depot(self, observation, minerals, supply_used, supply_max):
        return (
                is_unit_selected(observation, units.Terran.SCV)
                and supply_max - supply_used <= self.supply_threshold
                and minerals >= 100
                and not is_selected_unit_building(observation, units.Terran.SCV)
        )

    def should_train_marine(self, observation, minerals, supply_max, supply_used):
        return (
                is_unit_selected(observation, units.Terran.Barracks)
                and minerals >= 50
                and len(observation.production_queue) < 5
                and supply_max - supply_used > self.supply_threshold
        )

    def should_select_barracks(self, minerals, supply_max, supply_used, available_barracks):
        return (
                minerals >= 50
                and available_barracks
                and supply_max - supply_used > self.supply_threshold
        )

    def should_build_barracks(self, observation, minerals):
        return (
                minerals >= 150
                and not is_selected_unit_building(observation, units.Terran.SCV)
        )

    def build_supply_depot(self, observation):
        coords = self.get_next_depot_coords()

        while not are_unoccupied_coords_depot(observation, coords):
            coords = self.get_next_depot_coords()

        return build_action(2, coords)

    def build_barracks(self, observation):
        coords = self.get_next_barracks_coords()

        while not are_unoccupied_coords_barrack(observation, coords):
            coords = self.get_next_barracks_coords()

        return build_action(1, coords)

    def get_next_depot_coords(self):
        coords = (self.last_depot[0] - DEPOT_RADIUS, self.last_depot[1])
        if coords[0] < 1:
            coords = (64 - DEPOT_RADIUS, max(coords[1] - DEPOT_RADIUS, 1))

        self.last_depot = coords
        return coords

    def get_next_barracks_coords(self):
        coords = (self.last_barracks[0] + BARRACK_RADIUS, self.last_barracks[1])
        if coords[0] > 84:
            coords = (10, min(coords[1] + BARRACK_RADIUS + 1, 64))

        self.last_barracks = coords
        return coords


# Actions
def no_op_action():
    return np.array([[5, 0, 0]])


def train_marine_action():
    return np.array([[4, 0, 0]])


def select_barracks_action(barracks):
    return np.array([[3, barracks.x, barracks.y]])


def select_scv_action():
    return np.array([[0, 0, 0]])


def build_action(action_id, coords):
    return np.array([[action_id, coords[0], coords[1]]])


# Util
def get_units_by_type(obs, unit_type):
    return [unit for unit in obs.feature_units if unit.unit_type == unit_type]


def is_unit_selected(obs, unit_type):
    return any(unit.unit_type == unit_type and unit.is_selected for unit in obs.feature_units)


def is_selected_unit_building(obs, unit_type):
    selected_units = [unit for unit in obs.feature_units if unit.is_selected and unit.unit_type == unit_type]

    if not selected_units:
        return False

    unit = selected_units[0]
    return unit.order_id_0 in BUILD_ORDERS or unit.order_id_1 in BUILD_ORDERS


def are_unoccupied_coords_barrack(obs, coords):
    # Invert because feature screen is in matrix
    # Out of bounds safety checks are redundant but don't hurt
    for y in range(max(coords[0] - BARRACK_RADIUS, 1), min(coords[0] + BARRACK_RADIUS, 84)):
        for x in range(max(coords[0] - BARRACK_RADIUS, 1), min(coords[1] + BARRACK_RADIUS, 64)):
            screen_unit = obs.feature_screen.unit_type[x][y]
            if screen_unit != 0 and (screen_unit != units.Terran.SCV and screen_unit != units.Terran.Marine):
                return False

    return True


def are_unoccupied_coords_depot(obs, coords):
    for y in range(max(coords[0] - DEPOT_RADIUS, 1), min(coords[0] + DEPOT_RADIUS, 84)):
        for x in range(max(coords[1] - DEPOT_RADIUS, 1), min(coords[1] + DEPOT_RADIUS, 64)):
            screen_unit = obs.feature_screen.unit_type[x][y]
            if screen_unit != 0 and (screen_unit != units.Terran.SCV and screen_unit != units.Terran.Marine):
                return False

    return True

