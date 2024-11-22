import random
import numpy as np
from numpy.random import random_integers

from pysc2.lib import actions, units, features
from skvideo.measure import scenedet
from sympy.stats.sampling.sample_numpy import numpy

import actions_util
from actions_util import *
from pysc2.lib.actions import TYPES


class ActionManager:
    def __init__(self):
        self.workers_selected = False
        self.last_sent = None
        self.iteration = 0
        self.actions = [
            self.zero_action,   # Action 0: Select workers
            self.build_pylon,   # Action 1: Build a Pylon
            self.build_nexus    # Action 2: Build a Nexus
        ]

    def get_actions(self, state, action):
        """
        Executes the appropriate action based on the given integer input.
        """
        a = random.randint(0, 2)
        print("------------------------------------------------------------")
        print(a)
        print
        try:

            return self.actions[a](state)

        except Exception as e:
            print("Error during zero action (select workers):", e)
            print("Traceback:", e.__traceback__)
            return [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]  # No-op in case of an error


    def zero_action(self, state):
        """
        Action 0: Select workers (idle or specific type).
        """
        actions_list = []

        selected_unit = state.single_select

        if not selected_unit.any():
            # Select an idle worker if available
            if idle_workers_exist(state):
                actions_list.append(actions.FunctionCall(actions.FUNCTIONS.select_idle_worker.id, ["select"]))
            else:
                # Select a worker unit by type if idle workers are not available
                actions_list.append(select_unit_by_type(state, units.Protoss.Probe))

        return actions_list



    def build_pylon(self, state):
        """
        Action 1: Build a Pylon near the first townhall.
        """
        actions_list = []

        townhall = next(
            (unit for unit in state.feature_units if unit.unit_type == units.Protoss.Nexus),
            None
        )
        if townhall.any() and actions.FUNCTIONS.Build_Pylon_screen.id in state['available_actions']:
            pylon_position = actions_util.random_position_near_townhall((townhall.x, townhall.y), random.randint(1, 15))
            print(f"Building Pylon at {pylon_position}")
            actions_list.append(actions.FunctionCall(actions.FUNCTIONS.Build_Pylon_screen.id,
                                                     [[False], pylon_position]))
        return actions_list


    def build_nexus(self, state):
        """
        Action 2: Build a Nexus near the first townhall.
        """
        actions_list = []

        townhall = next(
            (unit for unit in state.feature_units if unit.unit_type == units.Protoss.Nexus),
            None
        )
        if townhall.any() and actions.FUNCTIONS.Build_Nexus_screen.id in state['available_actions']:
            nexus_position = actions_util.random_position_near_townhall((townhall.x, townhall.y), random.randint(1, 15))
            print(f"Building Nexus at {nexus_position}")
            actions_list.append(actions.FunctionCall(actions.FUNCTIONS.Build_Nexus_screen.id,
                                                     [[False], nexus_position]))
        return actions_list

        # 1: build stargate (or other buildings)
        # elif action == 1:
        #     try:
        #         # Build stargate and related structures

        #
        #     except Exception as e:
        #         print("Error during action 1 building:", e)
        #

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
