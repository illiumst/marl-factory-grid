from random import shuffle
from typing import List, Tuple

from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.environment import constants as c
from marl_factory_grid.modules.destinations import constants as d
from marl_factory_grid.modules.destinations.entitites import BoundDestination


class NarrowCorridorSpawn(Rule):
    def __init__(self, positions: List[Tuple[int, int]], fixed: bool = False):
        super().__init__()
        self.fixed = fixed
        self.positions = positions

    def on_init(self, state, lvl_map):
        if not self.fixed:
            shuffle(self.positions)
        for agent in state[c.AGENT]:
            pass

    def trigger_destination_spawn(self, state):
        for (agent_name, position_list) in self.per_agent_positions.items():
            agent = state[c.AGENT][agent_name]
            destinations = [BoundDestination(agent, pos) for pos in position_list]
            state[d.DESTINATION].add_items(destinations)
        return c.VALID

