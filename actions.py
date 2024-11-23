import random
import traceback
from types import NoneType

import actions_util
from actions_util import *
from pysc2.lib.actions import TYPES, Queued


class ActionManager:
    def __init__(self):
        self.workers_selected = False
        self.last_sent = None
        self.iteration = 0
        self.actions = [
            move_screen,
            build_pylon,
            # build_nexus,
            build_stargate,
            build_assimilator,
            redistribute_workers,
            select_worker
        ]

    def get_actions(self, state, action):
        """
        Executes the appropriate action based on the given integer input.
        """
        print("------------------------------------------------------------")
        try:

            return self.actions[1](state, (action[1], action[2]))

        except Exception as e:
            print("Error during zero action (select workers):", e)
            print("Traceback:", traceback.format_exc())
            return [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]


    # def zero_action(self, state):
    #     """
    #     Action 0: Select workers (idle or specific type).
    #     """
    #     actions_list = []
    #
    #     selected_unit = state.single_select
    #
    #     if not selected_unit.any():
    #         # Select an idle worker if available
    #         if idle_workers_exist(state):
    #             actions_list.append(actions.FunctionCall(actions.FUNCTIONS.select_idle_worker.id, ["select"]))
    #         else:
    #             # Select a worker unit by type if idle workers are not available
    #             actions_list.append(select_unit_by_type(state, units.Protoss.Probe))
    #
    #     return actions_list



def build_pylon(state, action):
    """
    Action 1: Build a Pylon near the first townhall.
    """
    actions_list = []
    queued = False
    can_build = False

    if not is_worker_selected(state):
        actions_list.append(select_worker(state))
        queued = True
        mineral_count = state.player[1]
        can_build = mineral_count > 100

    townhall = next(
        (unit for unit in state.feature_units if unit.unit_type == units.Protoss.Nexus),
        None
    )
    if (townhall is not None and townhall.any() and actions.FUNCTIONS.Build_Pylon_screen.id in state['available_actions']) or (queued and can_build):
        pylon_position = actions_util.random_position_near_townhall((townhall.x, townhall.y), random.randint(1, 10))
        if pylon_position[0] >= 0 and pylon_position[1] >= 0:
            position = validate_screen_coords(pylon_position[0], pylon_position[1])
            print(f"Building Pylon at {pylon_position}")
            if actions.FUNCTIONS.move_camera.id in state.available_actions:
                actions_list.append(actions.FunctionCall(actions.FUNCTIONS.move_camera.id, [(position[0], position[1])]))
                actions_list.append(actions.FunctionCall(actions.FUNCTIONS.Build_Pylon_screen.id,
                                                     [[queued], position]))
    return actions_list


def build_nexus(state, action):
    """
    Action 2: Build a Nexus near the first townhall.
    """
    actions_list = []

    townhall = next(
        (unit for unit in state.feature_units if unit.unit_type == units.Protoss.Nexus),
        None
    )
    if townhall.any() and actions.FUNCTIONS.Build_Nexus_screen.id in state['available_actions']:
        nexus_position = actions_util.random_position_near_townhall((townhall.x, townhall.y), random.randint(1, 12))
        if nexus_position[0] >= 0 and nexus_position[1] >= 0:
            position = validate_screen_coords(nexus_position[0], nexus_position[1])
            print(f"Building Nexus at {nexus_position}")
            actions_list.append(actions.FunctionCall(actions.FUNCTIONS.Build_Nexus_screen.id,
                                                 [[False], position]))
    return actions_list

        #
        # # 3: Send scout (idle probe sent to enemy base)
        # elif action == 3:
        #     try:
        #         if (self.iteration - self.last_sent) > 200:
        #             if state.units("PROBE").idle.exists:
        #                 probe = random.choice(state.units("PROBE").idle)
        #             else:
        #                 probe = random.choice(state.units("PROBE"))
        #             self.last_sent = self.iteration
        #         # TODO CHECK IF VALID SYNTAX!!!
        #         actions_list.append(actions.FUNCTIONS.Attack_minimap("now", (probe, state.enemy_start_locations[0])))
        #     except Exception as e:
        #         print("Error during action 3 scout:", e)
        #
        #
        #     except Exception as e:
        #         print("Error during action 4 attack:", e)
        #
        # # 5: Voidray flee (move back to base)
        # elif action == 5:
        #     try:
        #         if state.units("VOIDRAY").amount > 0:
        #             for vr in state.units("VOIDRAY"):
        #                 actions_list.append(actions.FUNCTIONS.Attack_minimap("now", state.start_location))
        #
        #     except Exception as e:
        #         print("Error during action 5 flee:", e)
        #
        # return actions_list if len(actions_list) > 0 else [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]


def move_screen(obs, xy_coords):
    if actions.FUNCTIONS.move_camera.id in obs.available_actions:
        return [actions.FunctionCall(actions.FUNCTIONS.move_camera.id, [(xy_coords[0], xy_coords[1])])]

    return [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]


# Needs to have selected worker first
# TODO Make sure selected geyser isn't in use
# Non-positive check, selected unit build queue check
from pysc2.lib import actions, units
import numpy as np


def is_worker_selected(obs):
    """Check if a worker is selected."""
    if len(obs.single_select) > 0 and obs.single_select[0].unit_type == units.Protoss.Probe:
        return True
    if len(obs.multi_select) > 0 and obs.multi_select[0].unit_type == units.Protoss.Probe:
        return True
    return False


# def select_worker(obs):
#     """Select a worker unit."""
#     workers = next(unit for unit in obs.feature_units
#                if unit.unit_type == units.Protoss.Probe
#                and unit.is_selected == 0)
#
#     if workers is not None and workers.any():
#         return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
#
#
#
#     worker = workers[0]
#     return actions.FunctionCall(actions.FUNCTIONS.select_point.id,
#                                 [[0], [int(worker.x), int(worker.y)]])


def build_assimilator(obs, _):
    """Build an assimilator on an available geyser."""
    actions_list = []
    queued = 0  # Initialize queued as integer 0
    can_build = False

    # First check if we need to select a worker
    if not is_worker_selected(obs):
        select_action = select_worker(obs)
        if select_action.function != actions.FUNCTIONS.no_op.id:
            actions_list.append(select_action)
            queued = 0  # Reset queued after selection
            mineral_count = obs.player[1]
            can_build = mineral_count >= 75

    # Check if we can build an assimilator
    if (actions.FUNCTIONS.Build_Assimilator_screen.id in obs.available_actions or
            (actions_list and can_build)):

        # Find available geysers
        geysers = [unit for unit in obs.feature_units
                   if unit.unit_type == units.Neutral.VespeneGeyser
                   and unit.assigned_harvesters == 0]

        if geysers:
            geyser = geysers[0]

            # Ensure coordinates are valid positive integers within screen bounds
            x_coord = max(0, min(83, int(round(geyser.x))))
            y_coord = max(0, min(83, int(round(geyser.y))))

            # Verify coordinates are valid before creating action
            if x_coord >= 0 and y_coord >= 0:
                build_action = actions.FunctionCall(
                    actions.FUNCTIONS.Build_Assimilator_screen.id,
                    [[queued], [x_coord, y_coord]])
                actions_list.append(build_action)
                return actions_list

    # If we can't build, return no_op
    if not actions_list:
        return [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]

    return actions_list


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
        return [actions.FunctionCall(actions.FUNCTIONS.select_idle_worker.id, ["select"])]

    if actions.FUNCTIONS.select_point.id in obs.available_actions:
        probes = next((unit for unit in obs.feature_units if unit.unit_type == units.Protoss.Probe), None)

        if probes is not None and probes.any():
            queued = False
            return [actions.FunctionCall(actions.FUNCTIONS.select_point.id, [[queued], (probes.x, probes.y)])]

    return [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]


def redistribute_workers(obs, _):
    """
    Redistribute idle workers to mineral field in view
    """
    selected_unit = obs.single_select

    if (selected_unit is None or selected_unit[0].unit_type != units.Protoss.Probe) and idle_workers_exist(obs):
        return [actions.FunctionCall(actions.FUNCTIONS.select_idle_worker.id, ["select"])]

    minerals = next((unit for unit in obs.feature_units if unit.unit_type == units.Neutral.MineralField),
                    None)

    queued = False
    if selected_unit.any() and minerals.any():
        return [actions.FunctionCall(actions.FUNCTIONS.Smart_screen.id, [[queued], [minerals.x, minerals.y]])]

    return [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]


def _xy_locs(mask):
    """Returns the (y, x) coordinates of non-zero values in a mask."""
    y, x = np.nonzero(mask)
    return list(zip(x, y))
