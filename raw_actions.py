from pysc2.lib import *
from actions_util import *


def build_pylon_raw(state, _):
    """
    Action 1: Build a Pylon near the first townhall using RAW_FUNCTIONS.
    """
    mineral_count = state.player.minerals
    pylon_cost = 100
    if mineral_count < pylon_cost:  # Pylon costs 100 minerals
        return []

    # Find the nexus
    nexus = next(
        (unit for unit in state.raw_units
         if unit.unit_type == units.Protoss.Nexus
         and unit.alliance == features.PlayerRelative.SELF),
        None
    )

    if nexus is None:
        return []

    # Find an available probe
    probe = next(
        (unit for unit in state.raw_units
         if unit.unit_type == units.Protoss.Probe
         and unit.alliance == features.PlayerRelative.SELF
         and unit.build_progress == 100  # Ensure probe is complete
         ),
        None
    )

    if probe is None:
        return []

    # Calculate pylon position (1 unit right and down from nexus)
    pylon_x = nexus.x + 3
    pylon_y = nexus.y + 3

    # Build the pylon using RAW_FUNCTIONS
    action = actions.RAW_FUNCTIONS.Build_Pylon_pt
    # Parameters: [queued], unit_tags, x, y
    return [actions.FunctionCall(action.id, [[0], [probe.tag], [pylon_x, pylon_y]])]


def build_pylons(state, _):
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
        (unit for unit in state.feature_units if unit.unit_type == units.Protoss.Nexus and unit.alliance == 1),
        None
    )

    if (townhall is not None and townhall.any() and actions.FUNCTIONS.Build_Pylon_screen.id in state.available_actions) \
            or (queued and can_build):
        # pylon_position = actions_util.random_position_near_townhall((townhall.x, townhall.y), random.randint(1, 10))
        pylon_position = (townhall.x + 5, townhall.y + 5)
        if pylon_position[0] >= 0 and pylon_position[1] >= 0:
            print(f"Building Pylon at {(townhall.x + 2, townhall.y + 2)}")

            if actions.FUNCTIONS.move_camera.id in state.available_actions:
                # actions_list.append(actions.FunctionCall(actions.FUNCTIONS.move_camera.id, [pylon_position]))
                actions_list.append(actions.FunctionCall(actions.FUNCTIONS.Build_Pylon_screen.id,
                                                         [[queued], (int(townhall.x) - 5, int(townhall.y) - 5)]))
    return actions_list