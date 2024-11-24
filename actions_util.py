import numpy as np
import random
import math

from pysc2.lib import features, actions, units


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


def is_worker_selected(obs):
    selected_unit = obs.single_select
    return selected_unit.any() and selected_unit[0].unit_type == units.Protoss.Probe


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


def random_position_near_townhall(townhall, radius):
    # Generate a random angle in radians
    theta = random.uniform(0, 2 * math.pi)

    x = int(townhall[0] + radius * math.cos(theta))
    y = int(townhall[1] + radius * math.sin(theta))

    return x, y


def idle_workers_exist(obs):
    return obs.player.idle_worker_count > 0 and actions.FUNCTIONS.select_idle_worker.id in obs.available_actions


def validate_screen_coords(x, y, max_size=83):
    """Validate and clamp screen coordinates to valid ranges."""
    x = max(0, min(max_size, int(round(float(x)))))
    y = max(0, min(max_size, int(round(float(y)))))
    return x, y


def select_worker(obs):
    if idle_workers_exist(obs):
        return [actions.FunctionCall(actions.FUNCTIONS.select_idle_worker.id, ["select"])]

    if actions.FUNCTIONS.select_point.id in obs.available_actions:
        probes = next((unit for unit in obs.feature_units if unit.unit_type == units.Protoss.Probe), None)

        if probes is not None and probes.any():
            queued = False
            return [actions.FunctionCall(actions.FUNCTIONS.select_point.id, [[queued], (probes.x, probes.y)])]

    return [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]


# TODO Check if selected worker has build queue
def build_object(obs, coords, function):
    actions_list = []
    queued = False

    if not is_worker_selected(obs):
        actions_list.append(select_worker(obs))
        queued = True

    if (coords[0] >= 0 and coords[1] >= 0) and (function.id in obs.available_actions or queued):
        actions_list.append(actions.FunctionCall(function.id, [[queued], coords]))
        return actions_list

    return []


def get_camera_position_quadrant(obs):
    column = np.argmax(np.sum(obs.feature_minimap.camera, axis=0))
    row = max(enumerate(obs.feature_minimap.camera), key=lambda z: sum(z[1]))[0]

    x = 0 if column < 32 else 1
    y = 0 if row < 32 else 1

    return x, y
