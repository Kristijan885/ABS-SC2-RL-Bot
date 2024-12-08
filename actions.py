import random
import traceback

import actions_util
from pysc2.lib import actions, units
from actions_util import *
import numpy as np


class ActionManager:
    def __init__(self):
        self.workers_selected = False
        self.last_sent = None
        self.iteration = 0
        self.actions = [
            # move_screen,
            # build_pylon,
            # build_assimilator,
            # build_nexus,
            # build_stargate,
            # redistribute_workers,
            select_worker,
            build_barracks,
            build_supply_depot,
            train_marines,
            # build_cybernetics_core,
            # build_gateway,

        ]

    def get_actions(self, state, action):
        """
        Executes the appropriate action based on the given integer input.
        """
        try:
            return self.actions[action[0]](state, (action[1], action[2]))

        except Exception as e:
            print(f"Error during action: {self.actions[action[0]].__name__}: {e}")
            print("Traceback:", traceback.format_exc())
            return [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]





def build_barracks(obs, coords):
    mineral_count = obs.player[1]

    if not mineral_count >= 150:
        return []

    return build_object(obs, coords, actions.FUNCTIONS.Build_Barracks_screen)

def build_supply_depot(obs, coords):
    mineral_count = obs.player[1]

    if not mineral_count >= 100:
        return []

    return build_object(obs, coords, actions.FUNCTIONS.Build_SupplyDepot_screen)


def train_marines(obs, coords):
    mineral_count = obs.player[1]
    boolean, coordinates = has_barracks(obs)

    if mineral_count < 50:
        return []

    if not boolean:
        return []

    # Check if selecting a unit is available
    if actions.FUNCTIONS.select_point.id not in obs.available_actions:
        return []

    # Add a select_point action to select the Barracks
    actions_list = [actions.FunctionCall(actions.FUNCTIONS.select_point.id, [[0], coordinates])]

    # Ensure Train_Marine_quick is available after selecting
    # if actions.FUNCTIONS.Train_Marine_quick.id not in obs.available_actions:
    #     return []

    # Add the train marine action
    actions_list.append(actions.FunctionCall(actions.FUNCTIONS.Train_Marine_quick.id, []))

    return actions_list



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
    if selected_unit.any() and minerals.any() and (minerals.x >= 0 and minerals.y >= 0):
        return [actions.FunctionCall(actions.FUNCTIONS.Smart_screen.id, [[queued], [minerals.x, minerals.y]])]

    return [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]
def is_worker_selected(obs):
    """Check if a worker is selected."""
    if len(obs.single_select) > 0 and obs.single_select[0].unit_type == units.Terran.SCV:
        return True
    if len(obs.multi_select) > 0 and obs.multi_select[0].unit_type == units.Terran.SCV:
        return True
    return False

def build_assimilator(obs, _):
    mineral_count = obs.player[1]
    if not mineral_count >= 75:
        return []

    geysers = next((unit for unit in obs.feature_units if unit.unit_type == units.Neutral.VespeneGeyser
                    and unit.assigned_harvesters == 0), None)

    if geysers is not None and geysers.any():
        return build_object(obs, (geysers.x, geysers.y), actions.FUNCTIONS.Build_Assimilator_screen)

    return []
def build_pylon(obs, coords):
    mineral_count = obs.player[1]

    if not mineral_count >= 100:
        return []

    return build_object(obs, coords, actions.FUNCTIONS.Build_Pylon_screen)
def build_nexus(obs, _):
    mineral_count = obs.player[1]
    if mineral_count < 400:
        return []

    geysers = [unit for unit in obs.feature_units if unit.unit_type == units.Neutral.VespeneGeyser and
               unit.alliance == 3]

    if len(geysers) < 2:
        return []

    x, y = get_camera_position_quadrant(obs)

    nexus_coords = (
        (max(geysers[0].x, geysers[1].x) - 5) if x == 0 else (min(geysers[0].x, geysers[1].x) + 5),
        (max(geysers[0].y, geysers[1].y) - 5) if y == 0 else (min(geysers[0].y, geysers[1].y) + 5),
    )

    return build_object(obs, nexus_coords, actions.FUNCTIONS.Build_Nexus_screen)
def train_probe(obs, _):
    action_list = []
    mineral_count = obs.player[1]
    food_used = obs.player[3]
    food_cap = obs.player[4]

    # Check if we have enough resources and supply
    if mineral_count < 50 or food_used >= food_cap:
        return []

    nexus = next((unit for unit in obs.feature_units if unit.unit_type == units.Protoss.Nexus and
                  unit.owner == obs.player[0]), None)

    if nexus is None:
        return []

    if actions.FUNCTIONS.Train_Probe_quick.id in obs.available_actions:
        if actions.FUNCTIONS.select_point.id in obs.available_actions:
            action_list.append(actions.FunctionCall(actions.FUNCTIONS.select_point.id,
                                         [[0], [nexus.x, nexus.y]]))


        action_list.append(actions.FunctionCall(actions.FUNCTIONS.Train_Probe_quick.id, []))
        print('it should be building a probe')
    return action_list

def move_screen(obs, xy_coords):
    if actions.FUNCTIONS.move_camera.id in obs.available_actions:
        return [actions.FunctionCall(actions.FUNCTIONS.move_camera.id, [(xy_coords[0], xy_coords[1])])]

    return [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]
def build_stargate(obs, xy_coords):
    mineral_count = obs.player[1]
    vespene_count = obs.player[2]

    if mineral_count < 150 or vespene_count < 150:
        return []

    pylons = get_pylons(obs)

    if not pylons:
        return []

    pylon_power = is_pylon_in_range(obs, pylons, xy_coords)

    if pylon_power:
        return build_object(obs, xy_coords, actions.FUNCTIONS.Build_Stargate_screen)


    return []
def build_cybernetics_core(obs, xy_coords):
    mineral_count = obs.player[1]

    if mineral_count < 150:
        return []

    return build_object(obs, xy_coords, actions.FUNCTIONS.Build_CyberneticsCore_screen)
def build_gateway(obs, xy_coords):
    mineral_count = obs.player[1]

    if mineral_count < 150:
        return []


    return build_object(obs, xy_coords, actions.FUNCTIONS.Build_Gateway_screen)
















def _xy_locs(mask):
    """Returns the (y, x) coordinates of non-zero values in a mask."""
    y, x = np.nonzero(mask)
    return list(zip(x, y))


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

# def build_assimilator(obs, _):
#     """Build an assimilator on an available geyser."""
#     actions_list = []
#     queued = 0  # Initialize queued as integer 0
#     can_build = False
#
#     # First check if we need to select a worker
#     if not is_worker_selected(obs):
#         select_action = select_worker(obs)
#         if select_action.function != actions.FUNCTIONS.no_op.id:
#             actions_list.append(select_action)
#             queued = 0  # Reset queued after selection
#             mineral_count = obs.player[1]
#             can_build = mineral_count >= 75
#
#     # Check if we can build an assimilator
#     if (actions.FUNCTIONS.Build_Assimilator_screen.id in obs.available_actions or
#             (actions_list and can_build)):
#
#         # Find available geysers
#         geysers = [unit for unit in obs.feature_units
#                    if unit.unit_type == units.Neutral.VespeneGeyser
#                    and unit.assigned_harvesters == 0]
#
#         if geysers:
#             geyser = geysers[0]
#
#             # Ensure coordinates are valid positive integers within screen bounds
#             x_coord = max(0, min(83, int(round(geyser.x))))
#             y_coord = max(0, min(83, int(round(geyser.y))))
#
#             # Verify coordinates are valid before creating action
#             if x_coord >= 0 and y_coord >= 0:
#                 build_action = actions.FunctionCall(
#                     actions.FUNCTIONS.Build_Assimilator_screen.id,
#                     [[queued], [x_coord, y_coord]])
#                 actions_list.append(build_action)
#                 return actions_list
#
#     # If we can't build, return no_op
#     if not actions_list:
#         return [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]
#
#     return actions_list

