import random

from pysc2.lib import actions


def get_actions(state, action):
    # 0: expand (move to next spot or build units)
    if action == 0:
        try:
            # expand logic here (build pylons, probes, etc.)
            if state.observation.player_relative != 1:
                # build pylons
                if state.observation.food_left < 4:
                    return [actions.FUNCTIONS.Build_Pylon_screen("now", state.townhalls[0])]

            # other build commands (nexus, assimilator)
            return [actions.FUNCTIONS.Build_Nexus_screen("now", state.townhalls[0])]

        except Exception as e:
            print("Error during action 0 expansion:", e)

    # 1: build stargate (or other buildings)
    elif action == 1:
        try:
            # Build stargate and related structures
            for nexus in state.townhalls:
                if not state.structures("STARGATE").closer_than(10, nexus):
                    if state.can_afford("STARGATE") and state.already_pending("STARGATE") == 0:
                        return [actions.FUNCTIONS.Build_Stargate_screen("now", nexus)]

            return [actions.FUNCTIONS.no_op()]

        except Exception as e:
            print("Error during action 1 building:", e)

    # 2: Build Voidray (from a random stargate)
    elif action == 2:
        try:
            if state.can_afford("VOIDRAY"):
                for sg in state.structures("STARGATE").ready.idle:
                    if state.can_afford("VOIDRAY"):
                        return actions.FUNCTIONS.Train_VoidRay_quick("now", sg)
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
                return actions.FUNCTIONS.Attack_minimap("now", state.enemy_start_locations[0])
        except Exception as e:
            print("Error during action 3 scout:", e)

    # 4: Attack with Voidrays
    elif action == 4:
        try:
            for voidray in state.units("VOIDRAY").idle:
                # If there are enemy units in range, attack them
                enemy_units = state.enemy_units.closer_than(10, voidray)
                if enemy_units.exists:
                    return actions.FUNCTIONS.Attack_unit("now", random.choice(enemy_units))
                # If there are enemy structures in range, attack them
                enemy_structures = state.enemy_structures.closer_than(10, voidray)
                if enemy_structures.exists:
                    return actions.FUNCTIONS.Attack_unit("now", random.choice(enemy_structures))
                # Otherwise attack the enemy base
                if state.enemy_start_locations:
                    return actions.FUNCTIONS.Attack_minimap("now", state.enemy_start_locations[0])

        except Exception as e:
            print("Error during action 4 attack:", e)

    # 5: Voidray flee (move back to base)
    elif action == 5:
        try:
            if state.units("VOIDRAY").amount > 0:
                for vr in state.units("VOIDRAY"):
                    return actions.FUNCTIONS.Attack_minimap("now", state.start_location)
        except Exception as e:
            print("Error during action 5 flee:", e)

    return None
