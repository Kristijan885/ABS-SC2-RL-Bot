import random
import numpy as np

from pysc2.lib import actions, units, features
from actions_util import *
from pysc2.lib.actions import TYPES


class ActionManager:
    def __init__(self):
        self.workers_selected = False
        self.last_sent = None
        self.iteration = 0
        self.actions = [

        ]

    def get_actions(self, state, action):

        actions_list = []
        if idle_workers_exist(state):
            actions_list.append(redistribute_workers(state))

        # 0: expand (move to next spot or build units)
        if action[0]:
            try:
                player = state['player']
                food_used = player[3]  # index 3 corresponds to 'food_used'
                food_cap = player[4]  # index 4 corresponds to 'food_cap'
                food_left = food_cap - food_used
                selected_unit = state.single_select

                if not selected_unit.any():
                    # Select an idle worker if available
                    if idle_workers_exist(state):
                        actions_list.append(actions.FunctionCall(actions.FUNCTIONS.select_idle_worker.id, ["select"]))

                    else:
                        actions_list.append(select_unit_by_type(state, units.Protoss.Probe))

                    # return actions_list

                queued = False
                if food_left < 4:
                    # Build a Pylon near the first townhall
                    townhall = next((unit for unit in state.feature_units if unit.unit_type == units.Protoss.Nexus),
                                    None)
                    if townhall.any() and actions.FUNCTIONS.Build_Pylon_screen.id in state['available_actions']:
                        pylon_position = (action[1], action[2])  # Example position near townhall
                        print(pylon_position)
                        actions_list.append(actions.FunctionCall(actions.FUNCTIONS.Build_Pylon_screen.id,
                                                                 [[queued], pylon_position]))
                else:
                    # Build a new Nexus
                    townhall = next((unit for unit in state.feature_units if unit.unit_type == units.Protoss.Nexus),
                                    None)
                    if townhall.any() and actions.FUNCTIONS.Build_Nexus_screen.id in state['available_actions']:
                        expansion_position = (townhall.x + 10, townhall.y + 10)  # Example position further away
                        actions_list.append(actions.FunctionCall(actions.FUNCTIONS.Build_Nexus_screen.id,
                                                                 [[queued], expansion_position]))

                # Return the list of actions
                return actions_list

            except Exception as e:
                print("Error during action 0 expansion:", e)
                return [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]  # No-op in case of an error

        # 1: build stargate (or other buildings)
        if action[1] == 1:
            try:
                # Build stargate and related structures
                for nexus in state.townhalls:
                    if not state.structures("STARGATE").closer_than(10, nexus):
                        if state.can_afford("STARGATE") and state.already_pending("STARGATE") == 0:
                            actions_list.append(actions.FUNCTIONS.Build_Stargate_screen("now", nexus))

            except Exception as e:
                print("Error during action 1 building:", e)

        # 2: Build Voidray (from a random stargate)
        elif action == 2:
            try:
                if state.can_afford("VOIDRAY"):
                    for sg in state.structures("STARGATE").ready.idle:
                        if state.can_afford("VOIDRAY"):
                            actions_list.append(actions.FUNCTIONS.Train_VoidRay_quick("now", sg))
            except Exception as e:
                print("Error during action 2 building Voidray:", e)

        # 3: Send scout (idle probe sent to enemy base)
        elif action == 3:
            try:
                if (self.iteration - self.last_sent) > 200:
                    if state.units("PROBE").idle.exists:
                        probe = random.choice(state.units("PROBE").idle)
                    else:
                        probe = random.choice(state.units("PROBE"))
                    self.last_sent = self.iteration
                # TODO CHECK IF VALID SYNTAX!!!
                actions_list.append(actions.FUNCTIONS.Attack_minimap("now", (probe, state.enemy_start_locations[0])))
            except Exception as e:
                print("Error during action 3 scout:", e)

        # 4: Attack with Voidrays
        elif action == 4:
            try:
                for voidray in state.feature_units("VOIDRAY").idle:
                    # If there are enemy units in range, attack them
                    enemy_units = state.enemy_units.closer_than(10, voidray)
                    if enemy_units.exists:
                        actions_list.append(actions.FUNCTIONS.Attack_unit("now", random.choice(enemy_units)))

                    # If there are enemy structures in range, attack them
                    enemy_structures = state.enemy_structures.closer_than(10, voidray)
                    if enemy_structures.exists:
                        actions_list.append(actions.FUNCTIONS.Attack_unit("now", random.choice(enemy_structures)))

                    # Otherwise attack the enemy base
                    if state.enemy_start_locations:
                        actions_list.append(actions.FUNCTIONS.Attack_minimap("now", state.enemy_start_locations[0]))

            except Exception as e:
                print("Error during action 4 attack:", e)

        # 5: Voidray flee (move back to base)
        elif action == 5:
            try:
                if state.units("VOIDRAY").amount > 0:
                    for vr in state.units("VOIDRAY"):
                        actions_list.append(actions.FUNCTIONS.Attack_minimap("now", state.start_location))

            except Exception as e:
                print("Error during action 5 flee:", e)

        return actions_list if len(actions_list) > 0 else [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]


def idle_workers_exist(obs):
    return obs.player.idle_worker_count > 0 and actions.FUNCTIONS.select_idle_worker.id in obs.available_actions


def move_screen(obs, action):
    if actions.FUNCTIONS.move_camera.id in obs.available_actions:
        return actions.FunctionCall(actions.FUNCTIONS.move_camera.id, [(action[1], action[2])])

    return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])


# Needs to have selected worker first
# TODO Make sure selected geyser isn't in use
def build_assimilator(obs):
    actions_list = []
    queued = False
    can_build = False

    if not is_worker_selected(obs):
        actions_list.append(select_worker(obs))
        queued = True
        mineral_count = obs.player[1]
        can_build = mineral_count > 74

    if (actions.FUNCTIONS.Build_Assimilator_screen.id in obs.available_actions) or (queued and can_build):
        geysers = next((unit for unit in obs.feature_units if unit.unit_type == units.Neutral.VespeneGeyser
                        and unit.assigned_harvesters == 0), None)

        if geysers is not None and geysers.any():
            actions_list.append(actions.FunctionCall(actions.FUNCTIONS.Build_Assimilator_screen.id,
                                                     [[queued], (geysers.x, geysers.y)]))
            return actions_list

    return [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]


# TODO Needs to have selected worker first
def build_stargate(obs, xy_coords):
    actions_list = []
    queued = False
    can_build = False

    if not is_worker_selected(obs):
        actions_list.append(select_worker(obs))
        queued = True
        mineral_count = obs.player[1]
        gas_count = obs.player[2]
        can_build = mineral_count >= 150 and gas_count >= 150

    if (actions.FUNCTIONS.Build_Stargate_screen.id in obs.available_actions) or (queued and can_build):
        return [
            actions.FunctionCall(actions.FUNCTIONS.Build_Stargate_screen.id, [[queued], (xy_coords[1], xy_coords[2])])]

    return [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]


# TODO Cybernetics Core
# TODO Pylons

def select_worker(obs):
    if idle_workers_exist(obs):
        return actions.FunctionCall(actions.FUNCTIONS.select_idle_worker.id, ["select"])

    if actions.FUNCTIONS.select_point.id in obs.available_actions:
        probes = next((unit for unit in obs.feature_units if unit.unit_type == units.Protoss.Probe), None)

        if probes is not None and probes.any():
            queued = False
            return actions.FunctionCall(actions.FUNCTIONS.select_point.id, [[queued], (probes.x, probes.y)])

    return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])


def redistribute_workers(obs):
    """
    Redistribute idle workers to mineral field in view
    """
    selected_unit = obs.single_select

    if (selected_unit is None or selected_unit[0].unit_type != units.Protoss.Probe) and idle_workers_exist(obs):
        return actions.FunctionCall(actions.FUNCTIONS.select_idle_worker.id, ["select"])

    minerals = next((unit for unit in obs.feature_units if unit.unit_type == units.Neutral.MineralField),
                    None)

    queued = False
    if selected_unit.any() and minerals.any():
        return actions.FunctionCall(actions.FUNCTIONS.Smart_screen.id, [[queued], [minerals.x, minerals.y]])

    return [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]


def _xy_locs(mask):
    """Returns the (y, x) coordinates of non-zero values in a mask."""
    y, x = np.nonzero(mask)
    return list(zip(x, y))
