import random
import numpy as np

from pysc2.lib import actions, units, features


def stop_all_workers(obs):
    # Find all worker units controlled by the player
    workers = [unit for unit in obs.feature_units if unit.unit_type in [
        units.Terran.SCV, units.Protoss.Probe, units.Zerg.Drone] and unit.alliance == features.PlayerRelative.SELF]

    action_calls = []

    if workers:
        for worker in workers:
            # Select each worker with the correct argument structure
            select_action = actions.FunctionCall(
                actions.FUNCTIONS.select_point.id,
                [[actions.SelectPointAct.select], [worker.x, worker.y]]
            )
            action_calls.append(select_action)

            # Issue the Stop command for each selected worker
            stop_action = actions.FunctionCall(actions.FUNCTIONS.Stop_quick.id, [])
            action_calls.append(stop_action)

    return action_calls if action_calls else [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]


def get_actions(state, action: int):
    action_id = state.available_actions[action]

    print(state.available_actions)

    # Define default arguments for actions
    args = [[0]]  # Default no-op argument; customize as needed

    # Check for specific actions that may need different arguments

    if action_id == actions.FUNCTIONS.Move_screen.id:
        # Random target on the screen for the Move action
        target = [random.randint(0, state.feature_screen.shape[1] - 1),
                  random.randint(0, state.feature_screen.shape[2] - 1)]
        args = [[0], target]
    elif action_id == actions.FUNCTIONS.Build_Pylon_screen.id:
        # Random location for building a Pylon
        build_location = [random.randint(0, state.feature_screen.shape[1] - 1),
                          random.randint(0, state.feature_screen.shape[2] - 1)]
        args = [[0], build_location]
    elif action_id == actions.FUNCTIONS.Attack_minimap.id:
        # Random target on the minimap for Attack action
        attack_target = [random.randint(0, state.feature_minimap.shape[1] - 1),
                         random.randint(0, state.feature_minimap.shape[2] - 1)]
        args = [[0], attack_target]

    elif action_id == actions.FUNCTIONS.select_rect.id:
        # Random target on the minimap for Attack action
        return [actions.FUNCTIONS.select_rect("select", [10, 10], [30, 30])]  # Example rectangle coordinates

    elif action_id == actions.FUNCTIONS.move_camera.id:
        # Random target on the minimap for Attack action
        attack_target = [random.randint(0, state.feature_minimap.shape[1] - 1),
                         random.randint(0, state.feature_minimap.shape[2] - 1)]
        args = [attack_target]

    if action_id == actions.FUNCTIONS.select_point.id:
        target = [random.randint(0, state.feature_screen.shape[1] - 1),
                  random.randint(0, state.feature_screen.shape[2] - 1)]
        args = [[0], target]

    if action_id == actions.FUNCTIONS.select_control_group.id:
        target = [random.randint(0, 4),
                  random.randint(0, 9)]
        args = [[target[0]], [target[1]]]

    elif action_id == actions.FUNCTIONS.no_op.id:
        return [actions.FunctionCall(action_id, [])]

    # Return the randomly selected action with its arguments
    return [actions.FunctionCall(action_id, args)]
