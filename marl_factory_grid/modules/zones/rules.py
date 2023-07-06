from random import choices

from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.environment import constants as c
from marl_factory_grid.modules.zones import Zone
from . import constants as z


class AgentSingleZonePlacement(Rule):

    def __init__(self, n_zones=3):
        super().__init__()
        self.n_zones = n_zones

    def on_init(self, state, lvl_map):
        zones = []

        for z_idx in range(1, self.n_zones):
            zone_positions = lvl_map.get_coordinates_for_symbol(z_idx)
            assert len(zone_positions)
            zones.append(Zone([state[c.FLOOR].by_pos(pos) for pos in zone_positions]))
        state[z.ZONES].add_items(zones)

        n_agents = len(state[c.AGENT])
        assert len(state[z.ZONES]) >= n_agents

        z_idxs = choices(list(range(len(state[z.ZONES]))), k=n_agents)
        for agent in state[c.AGENT]:
            agent.move(state[z.ZONES][z_idxs.pop()].random_tile)
        return []

    def tick_step(self, state):
        return []
