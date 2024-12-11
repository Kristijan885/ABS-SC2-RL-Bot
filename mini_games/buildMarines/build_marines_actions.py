import traceback

from actions_util import *


class BuildMarinesActionManager:
    def __init__(self):
        self.workers_selected = False
        self.last_sent = None
        self.iteration = 0
        self.actions_list = [
            select_scv_worker,
            build_barracks,
            build_supply_depot,
            select_barracks,
            train_marines,
            no_op,
        ]

    def get_actions(self, state, action):
        """
        Executes the appropriate action based on the given integer input.
        """
        try:
            return self.actions_list[action[0]](state, (action[1], action[2]))

        except Exception as e:
            print(f"Error during action: {self.actions_list[action[0]].__name__}: {e}")
            print("Traceback:", traceback.format_exc())
            return [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]


def select_scv_worker(obs, _):
    return select_worker(obs, units.Terran.SCV)


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


def select_barracks(obs, _):
    coordinates = get_obs_unit_coords(obs, units.Terran.Barracks)

    if not coordinates:
        return []

    if actions.FUNCTIONS.select_point.id not in obs.available_actions:
        return []

    return [actions.FunctionCall(actions.FUNCTIONS.select_point.id, [[0], coordinates])]


def train_marines(obs, _):
    mineral_count = obs.player[1]

    if mineral_count < 50:
        return []

    if actions.FUNCTIONS.Train_Marine_quick.id not in obs.available_actions:
        return []

    return [actions.FunctionCall(actions.FUNCTIONS.Train_Marine_quick.id, [[True]])]


def no_op(*_):
    return [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]
