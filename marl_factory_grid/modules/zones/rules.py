from random import choices, choice

from . import constants as z, Zone
from .. import Destination
from ..destinations import constants as d
from ...environment.rules import Rule
from ...environment import constants as c


class ZoneInit(Rule):

    def __init__(self):
        super().__init__()
        self._zones = list()

    def on_init(self, state, lvl_map):
        z_idx = 1

        while z_idx:
            zone_positions = lvl_map.get_coordinates_for_symbol(z_idx)
            if len(zone_positions):
                self._zones.append(Zone(zone_positions))
                z_idx += 1
            else:
                z_idx = 0

    def on_reset(self, state):
        state[z.ZONES].add_items(self._zones)
        return []


class AgentSingleZonePlacement(Rule):

    def __init__(self):
        super().__init__()

    def on_reset(self, state):
        n_agents = len(state[c.AGENT])
        assert len(state[z.ZONES]) >= n_agents

        z_idxs = choices(list(range(len(state[z.ZONES]))), k=n_agents)
        for agent in state[c.AGENT]:
            agent.move(state[z.ZONES][z_idxs.pop()].random_pos, state)
        return []


class IndividualDestinationZonePlacement(Rule):

    def __init__(self):
        raise NotImplementedError("This is pretty new, and needs to be debugged, after the zones")
        super().__init__()

    def on_reset(self, state):
        for agent in state[c.AGENT]:
            self.trigger_spawn(agent, state)
        return []

    @staticmethod
    def trigger_spawn(agent, state):
        agent_zones = state[z.ZONES].by_pos(agent.pos)
        other_zones = [x for x in state[z.ZONES] if x not in agent_zones]
        already_has_destination = True
        while already_has_destination:
            pos = choice(other_zones).random_pos
            if state[d.DESTINATION].by_pos(pos) is None:
                already_has_destination = False
                destination = Destination(pos, bind_to=agent)

                state[d.DESTINATION].add_item(destination)
            continue
        return c.VALID
