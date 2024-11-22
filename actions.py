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


# def redistribute_workers(obs):
#     """
#     Redistribute idle or oversaturated workers to nearby resources.
#     """
#     player_relative = obs.feature_screen.player_relative
#     selected_unit = obs.single_select
#
#     actions_list = []
#
#     # Step 1: Select all idle workers
#     if idle_workers_exist(obs):
#         actions_list.append(actions.FUNCTIONS.select_idle_worker("select"))
#
#     # Step 2: Command workers to nearest mineral patch
#     minerals = np.array(_xy_locs(player_relative == features.PlayerRelative.NEUTRAL))
#     if selected_unit.any() and minerals.any():
#         worker_x, worker_y = selected_unit[0].x, selected_unit[0].y
#         distances = np.linalg.norm(minerals - np.array([worker_x, worker_y]), axis=1)
#         closest_mineral_patch = minerals[np.argmin(distances)]
#
#         actions_list.append(actions.FUNCTIONS.Smart_screen("now", closest_mineral_patch))
#
#     # If no actions are available, return a no-op
#     if not actions_list:
#         return [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]
#
#     return actions_list


def _xy_locs(mask):
    """Returns the (y, x) coordinates of non-zero values in a mask."""
    y, x = np.nonzero(mask)
    return list(zip(x, y))
