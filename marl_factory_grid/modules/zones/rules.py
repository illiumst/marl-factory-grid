from random import choices, choice

from . import constants as z, Zone
from ..destinations import constants as d
from ..destinations.entitites import BoundDestination
from ...environment.rules import Rule
from ...environment import constants as c


class ZoneInit(Rule):

    def __init__(self):
        super().__init__()

    def on_init(self, state, lvl_map):
        zones = []
        z_idx = 1

        while z_idx:
            zone_positions = lvl_map.get_coordinates_for_symbol(z_idx)
            if len(zone_positions):
                zones.append(Zone([state[c.FLOOR].by_pos(pos) for pos in zone_positions]))
                z_idx += 1
            else:
                z_idx = 0
        state[z.ZONES].add_items(zones)
        return []


class AgentSingleZonePlacement(Rule):

    def __init__(self):
        super().__init__()

    def on_init(self, state, lvl_map):
        n_agents = len(state[c.AGENT])
        assert len(state[z.ZONES]) >= n_agents

        z_idxs = choices(list(range(len(state[z.ZONES]))), k=n_agents)
        for agent in state[c.AGENT]:
            agent.move(state[z.ZONES][z_idxs.pop()].random_tile)
        return []

    def tick_step(self, state):
        return []


class IndividualDestinationZonePlacement(Rule):

    def __init__(self):
        super().__init__()

    def on_init(self, state, lvl_map):
        for agent in state[c.AGENT]:
            self.trigger_destination_spawn(agent, state)
            pass
        return []

    def tick_step(self, state):
        return []

    @staticmethod
    def trigger_destination_spawn(agent, state):
        agent_zones = state[z.ZONES].by_pos(agent.pos)
        other_zones = [x for x in state[z.ZONES] if x not in agent_zones]
        already_has_destination = True
        while already_has_destination:
            tile = choice(other_zones).random_tile
            if state[d.BOUNDDESTINATION].by_pos(tile.pos) is None:
                already_has_destination = False
                destination = BoundDestination(agent, tile)
                state[d.BOUNDDESTINATION].add_item(destination)
            continue
        return c.VALID
