import numpy as np

from pysc2.lib import features, actions


def select_unit_by_type(state, unit_type_selection):
    worker_units = [
        unit for unit in state.feature_units
        if unit.unit_type == unit_type_selection
    ]

    if worker_units:
        worker = worker_units[0]  # Select the first available worker
        worker_position = [worker.x, worker.y]
        select_func_id = 0
        return actions.FunctionCall(actions.FUNCTIONS.select_point.id, [[select_func_id], worker_position])

    return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])


def get_my_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.feature_units
            if unit.unit_type == unit_type
            and unit.alliance == features.PlayerRelative.SELF]


def get_enemy_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.feature_units
            if unit.unit_type == unit_type
            and unit.alliance == features.PlayerRelative.ENEMY]


def get_my_completed_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.feature_units
            if unit.unit_type == unit_type
            and unit.build_progress == 100
            and unit.alliance == features.PlayerRelative.SELF]


def get_enemy_completed_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.feature_units
            if unit.unit_type == unit_type
            and unit.build_progress == 100
            and unit.alliance == features.PlayerRelative.ENEMY]


def get_distances(self, obs, units, xy):
    units_xy = [(unit.x, unit.y) for unit in units]
    return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)